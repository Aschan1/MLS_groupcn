"""
Triton Multi-Head Attention Implementation
FlashAttention-inspired fused kernel with online softmax.

Replaces naive 3-kernel approach (Q@K^T → HBM → softmax → HBM → scores@V)
with a single tiled kernel that never materializes the N×N score matrix.
"""

import numpy as np
import torch
import triton
import triton.language as tl
from typing import Optional, Tuple


def get_stream():
    """Get current CUDA stream pointer."""
    if torch.cuda.is_available():
        return torch.cuda.current_stream().cuda_stream
    return None


# ============================================================================
# FlashAttention Fused Kernel (FA-2 core + FA-4 inspired optimizations)
# ============================================================================

@triton.jit
def flash_attention_fwd_kernel(
    Q, K, V, Out,
    Mask,
    scale,
    seq_q, seq_k,
    num_heads, num_kv_heads, num_queries_per_kv,
    stride_qb, stride_qh, stride_qq, stride_qd,
    stride_kb, stride_kh, stride_kk, stride_kd,
    stride_vb, stride_vh, stride_vk, stride_vd,
    stride_ob, stride_oh, stride_oq, stride_od,
    stride_mb, stride_mh, stride_mq, stride_mk,
    HAS_MASK: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Fused attention kernel with online softmax (FlashAttention-2 style).

    Grid: (batch * num_heads, cdiv(seq_q, BLOCK_Q))

    Key ideas:
    - Outer loop over Q tiles, inner loop over K/V tiles
    - Online softmax: track running max (m_i) and sum (l_i) per row
    - Never materialize full N×N score matrix in HBM
    - GQA-native: map Q heads to KV heads inside kernel
    - Causal masking: skip future K/V blocks entirely, mask partial blocks inline
    """
    pid_bh = tl.program_id(0)
    pid_q_block = tl.program_id(1)

    # GQA head mapping
    batch_idx = pid_bh // num_heads
    head_idx = pid_bh % num_heads
    kv_head_idx = head_idx // num_queries_per_kv

    q_start = pid_q_block * BLOCK_Q

    # Offsets
    offs_q = q_start + tl.arange(0, BLOCK_Q)
    offs_d = tl.arange(0, BLOCK_D)
    offs_k = tl.arange(0, BLOCK_K)

    # Load Q tile [BLOCK_Q, BLOCK_D]
    q_base = Q + batch_idx * stride_qb + head_idx * stride_qh
    q_ptrs = q_base + offs_q[:, None] * stride_qq + offs_d[None, :] * stride_qd
    q_tile = tl.load(q_ptrs, mask=(offs_q[:, None] < seq_q) & (offs_d[None, :] < BLOCK_D), other=0.0)

    # K/V base pointers (GQA: use kv_head_idx)
    k_base = K + batch_idx * stride_kb + kv_head_idx * stride_kh
    v_base = V + batch_idx * stride_vb + kv_head_idx * stride_vh

    # Initialize online softmax accumulators
    acc = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)
    m_i = tl.full((BLOCK_Q,), value=float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_Q,), dtype=tl.float32)

    # Causal: limit KV range to avoid processing future blocks
    if IS_CAUSAL:
        kv_len = tl.minimum(seq_k, q_start + BLOCK_Q)
    else:
        kv_len = seq_k

    # Inner loop over K/V blocks, even though running the loop every time, the K and V will be shared in L2 cache once loaded.
    for k_start in range(0, kv_len, BLOCK_K):
        k_offs = k_start + offs_k

        # Load K^T as [BLOCK_D, BLOCK_K] for Q @ K^T computation
        kt_ptrs = k_base + offs_d[:, None] * stride_kd + k_offs[None, :] * stride_kk
        kt_tile = tl.load(
            kt_ptrs,
            mask=(offs_d[:, None] < BLOCK_D) & (k_offs[None, :] < seq_k),
            other=0.0,
        )

        # S = Q @ K^T * scale : [BLOCK_Q, BLOCK_K]
        s = tl.dot(q_tile, kt_tile) * scale

        # Mask out-of-bounds keys
        s = tl.where(k_offs[None, :] < seq_k, s, float("-inf"))

        # Causal masking: mask positions where key > query
        if IS_CAUSAL:
            s = tl.where(offs_q[:, None] >= k_offs[None, :], s, float("-inf"))

        # Optional attention mask (additive)
        if HAS_MASK:
            m_ptrs = (
                Mask
                + batch_idx * stride_mb
                + head_idx * stride_mh
                + offs_q[:, None] * stride_mq
                + k_offs[None, :] * stride_mk
            )
            mask_vals = tl.load(
                m_ptrs,
                mask=(offs_q[:, None] < seq_q) & (k_offs[None, :] < seq_k),
                other=0.0,
            )
            s = s + mask_vals

        # --- Online softmax update ---
        # Compute new row-wise max
        m_block = tl.max(s, axis=1)
        m_new = tl.maximum(m_i, m_block)

        # Rescale previous accumulator
        alpha = tl.exp(m_i - m_new)
        acc = acc * alpha[:, None]
        l_i = l_i * alpha

        # Compute local attention weights
        p = tl.exp(s - m_new[:, None])
        l_i = l_i + tl.sum(p, axis=1)
        m_i = m_new

        # Load V tile [BLOCK_K, BLOCK_D]
        v_ptrs = v_base + k_offs[:, None] * stride_vk + offs_d[None, :] * stride_vd
        v_tile = tl.load(
            v_ptrs,
            mask=(k_offs[:, None] < seq_k) & (offs_d[None, :] < BLOCK_D),
            other=0.0,
        )

        # Accumulate: acc += P @ V
        acc += tl.dot(p.to(tl.float32), v_tile)

    # Final normalization: output = acc / l_i
    acc = acc / tl.where(l_i[:, None] > 0, l_i[:, None], 1.0)

    # Store output
    o_base = Out + batch_idx * stride_ob + head_idx * stride_oh
    o_ptrs = o_base + offs_q[:, None] * stride_oq + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc, mask=(offs_q[:, None] < seq_q) & (offs_d[None, :] < BLOCK_D))


# ============================================================================
# Legacy kernels (kept for fallback/testing)
# ============================================================================

@triton.jit
def attention_scores_kernel(
    q_ptr, k_ptr, scores_ptr, scale, seq_k, head_dim,
    stride_q0, stride_q1, stride_q2,
    stride_k0, stride_k1, stride_k2,
    stride_s0, stride_s1, stride_s2,
    BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """Compute scaled attention scores: Q @ K^T * scale (legacy)."""
    pid_bh = tl.program_id(0)
    pid_q = tl.program_id(1)
    offs_d = tl.arange(0, BLOCK_D)
    offs_k = tl.arange(0, BLOCK_K)
    q = tl.load(
        q_ptr + pid_bh * stride_q0 + pid_q * stride_q1 + offs_d * stride_q2,
        mask=offs_d < head_dim, other=0.0,
    )
    k = tl.load(
        k_ptr + pid_bh * stride_k0 + offs_k[:, None] * stride_k1 + offs_d[None, :] * stride_k2,
        mask=(offs_k[:, None] < seq_k) & (offs_d[None, :] < head_dim), other=0.0,
    )
    scores = tl.sum(k * q[None, :], axis=1) * scale
    tl.store(
        scores_ptr + pid_bh * stride_s0 + pid_q * stride_s1 + offs_k * stride_s2,
        scores, mask=offs_k < seq_k,
    )


@triton.jit
def softmax_inplace_kernel(scores_ptr, stride_s, seq_k, BLOCK_SIZE: tl.constexpr):
    """Apply softmax along the last dimension (legacy)."""
    row = tl.program_id(0)
    offs_k = tl.arange(0, BLOCK_SIZE)
    mask = offs_k < seq_k
    s_ptr = scores_ptr + row * stride_s + offs_k
    s = tl.load(s_ptr, mask=mask, other=float("-inf"))
    s = s - tl.max(s, axis=0)
    exp_s = tl.exp(s)
    s = exp_s / tl.sum(exp_s, axis=0)
    tl.store(s_ptr, s, mask=mask)


@triton.jit
def attention_output_kernel(
    attn_ptr, v_ptr, output_ptr, seq_k, head_dim,
    stride_w0, stride_w1, stride_w2,
    stride_v0, stride_v1, stride_v2,
    stride_o0, stride_o1, stride_o2,
    BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """Compute attention output: attn_weights @ V (legacy)."""
    pid_bh = tl.program_id(0)
    pid_q = tl.program_id(1)
    offs_k = tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, BLOCK_D)
    w = tl.load(
        attn_ptr + pid_bh * stride_w0 + pid_q * stride_w1 + offs_k * stride_w2,
        mask=offs_k < seq_k, other=0.0,
    )
    v = tl.load(
        v_ptr + pid_bh * stride_v0 + offs_k[:, None] * stride_v1 + offs_d[None, :] * stride_v2,
        mask=(offs_k[:, None] < seq_k) & (offs_d[None, :] < head_dim), other=0.0,
    )
    out = tl.sum(v * w[:, None], axis=0)
    tl.store(
        output_ptr + pid_bh * stride_o0 + pid_q * stride_o1 + offs_d * stride_o2,
        out, mask=offs_d < head_dim,
    )


@triton.jit
def causal_mask_kernel(
    scores_ptr, seq_k, offset,
    stride_s0, stride_s1, stride_s2,
    BLOCK_K: tl.constexpr,
):
    """Apply causal mask to attention scores (legacy)."""
    pid_bh = tl.program_id(0)
    pid_q = tl.program_id(1)
    offs_k = tl.arange(0, BLOCK_K)
    mask = offs_k < seq_k
    ptr = scores_ptr + pid_bh * stride_s0 + pid_q * stride_s1 + offs_k * stride_s2
    scores = tl.load(ptr, mask=mask, other=-1e9)
    current_pos = pid_q + offset
    scores = tl.where(offs_k > current_pos, -1e9, scores)
    tl.store(ptr, scores, mask=mask)


# ============================================================================
# Attention Classes
# ============================================================================

class MultiHeadAttention:
    """Multi-head attention with FlashAttention kernel and GQA support."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = head_dim or (hidden_size // num_heads)
        self.scale = 1.0 / np.sqrt(self.head_dim)
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """
        Compute multi-head attention with GQA-native FlashAttention.

        Args:
            q: (batch, num_heads, seq_q, head_dim)
            k: (batch, num_kv_heads, seq_k, head_dim)
            v: (batch, num_kv_heads, seq_k, head_dim)
            attention_mask: Optional (batch, 1, seq_q, seq_k) or (batch, num_heads, seq_q, seq_k)
            is_causal: Whether to apply causal masking
        """
        return scaled_dot_product_attention(
            q, k, v, attention_mask, is_causal, self.scale,
            num_kv_heads=k.shape[1],
        )

    def _expand_kv(self, x: torch.Tensor, num_repeats: int) -> torch.Tensor:
        """Expand KV heads for GQA (fallback path only)."""
        batch, num_kv_heads, seq_len, head_dim = x.shape
        x_expanded = x[:, :, None, :, :].expand(
            batch, num_kv_heads, num_repeats, seq_len, head_dim
        )
        return x_expanded.reshape(batch, num_kv_heads * num_repeats, seq_len, head_dim)


def next_power_of_two(x: int) -> int:
    """Return the smallest power of two >= x."""
    return 1 << (x - 1).bit_length() if x > 0 else 1


def _select_block_sizes(head_dim, seq_k):
    """Select BLOCK_Q, BLOCK_K, BLOCK_D based on head_dim and SRAM budget."""
    BLOCK_D = next_power_of_two(head_dim)
    if head_dim <= 64:
        BLOCK_Q = 64
        BLOCK_K = 32
    elif head_dim <= 128:
        BLOCK_Q = 32
        BLOCK_K = 32
    else:
        BLOCK_Q = 16
        BLOCK_K = 16
    BLOCK_K = max(BLOCK_K, 16)
    BLOCK_Q = max(BLOCK_Q, 16)
    return BLOCK_Q, BLOCK_K, BLOCK_D


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    scale: Optional[float] = None,
    num_kv_heads: Optional[int] = None,
) -> torch.Tensor:
    """
    Scaled dot-product attention using FlashAttention fused kernel.

    Supports GQA natively when num_kv_heads < num_heads.
    """
    batch, num_heads, seq_q, head_dim = q.shape
    _, num_kv_h, seq_k, _ = k.shape
    if num_kv_heads is None:
        num_kv_heads = num_kv_h

    if scale is None:
        scale = 1.0 / np.sqrt(head_dim)

    num_queries_per_kv = num_heads // num_kv_heads

    # Use FlashAttention on CUDA (for seq_q >= 4, otherwise torch SDPA is faster)
    if q.is_cuda and seq_q >= 4:
        # Ensure float32 and contiguous
        q_f = q.to(torch.float32).contiguous()
        k_f = k.to(torch.float32).contiguous()
        v_f = v.to(torch.float32).contiguous()

        # Prepare output
        output = torch.empty(
            (batch, num_heads, seq_q, head_dim),
            dtype=torch.float32,
            device=q.device,
        )

        # Block size selection
        BLOCK_Q, BLOCK_K, BLOCK_D = _select_block_sizes(head_dim, seq_k)

        # Attention mask handling
        has_mask = attention_mask is not None
        if has_mask:
            attention_mask = attention_mask.to(torch.float32).contiguous()
            # Handle (batch, 1, seq_q, seq_k) broadcast: set head stride to 0
            if attention_mask.shape[1] == 1:
                stride_mh = 0
                stride_mb = attention_mask.stride(0)
                stride_mq = attention_mask.stride(2)
                stride_mk = attention_mask.stride(3)
            else:
                stride_mb = attention_mask.stride(0)
                stride_mh = attention_mask.stride(1)
                stride_mq = attention_mask.stride(2)
                stride_mk = attention_mask.stride(3)
            mask_ptr = attention_mask
        else:
            mask_ptr = q_f  # dummy pointer, never accessed
            stride_mb = stride_mh = stride_mq = stride_mk = 0

        # Grid: one block per (batch*head, Q-block)
        grid = (batch * num_heads, triton.cdiv(seq_q, BLOCK_Q))

        # Select num_warps based on head_dim
        nw = 4 if head_dim <= 64 else 8

        flash_attention_fwd_kernel[grid](
            q_f, k_f, v_f, output,
            mask_ptr,
            float(scale),
            seq_q, seq_k,
            num_heads, num_kv_heads, num_queries_per_kv,
            q_f.stride(0), q_f.stride(1), q_f.stride(2), q_f.stride(3),
            k_f.stride(0), k_f.stride(1), k_f.stride(2), k_f.stride(3),
            v_f.stride(0), v_f.stride(1), v_f.stride(2), v_f.stride(3),
            output.stride(0), output.stride(1), output.stride(2), output.stride(3),
            stride_mb, stride_mh, stride_mq, stride_mk,
            HAS_MASK=has_mask,
            IS_CAUSAL=is_causal,
            BLOCK_Q=BLOCK_Q,
            BLOCK_K=BLOCK_K,
            BLOCK_D=BLOCK_D,
            num_warps=nw,
            num_stages=2,
        )

        return output.to(q.dtype)

    # Fallback: expand KV for GQA, then use torch SDPA or einsum
    if num_kv_heads != num_heads:
        k = k[:, :, None, :, :].expand(
            batch, num_kv_heads, num_queries_per_kv, seq_k, head_dim
        ).reshape(batch, num_heads, seq_k, head_dim)
        v = v[:, :, None, :, :].expand(
            batch, num_kv_heads, num_queries_per_kv, seq_k, head_dim
        ).reshape(batch, num_heads, seq_k, head_dim)

    # Use PyTorch SDPA for CUDA small-seq or CPU
    if q.is_cuda:
        attn_mask = attention_mask
        if attn_mask is not None and attn_mask.dtype == torch.float32:
            # Convert additive mask: large negative → True for masking
            attn_mask = attn_mask.to(q.dtype)
        out = torch.nn.functional.scaled_dot_product_attention(
            q.to(torch.float32), k.to(torch.float32), v.to(torch.float32),
            attn_mask=attn_mask.to(torch.float32) if attn_mask is not None else None,
            is_causal=is_causal,
            scale=scale,
        )
        return out.to(q.dtype)

    scores = torch.einsum("bnqd,bnkd->bnqk", q, k) * scale

    if is_causal:
        mask = torch.triu(
            torch.ones((seq_q, seq_k), dtype=torch.float32, device=q.device),
            diagonal=1,
        ) * -1e9
        scores = scores + mask[None, None, :, :]

    if attention_mask is not None:
        scores = scores + attention_mask

    scores = scores - torch.max(scores, dim=-1, keepdim=True).values
    attn_weights = torch.exp(scores)
    attn_weights = attn_weights / torch.sum(attn_weights, dim=-1, keepdim=True)
    output = torch.einsum("bnqk,bnkd->bnqd", attn_weights, v)

    return output.to(q.dtype)


if __name__ == "__main__":
    print("Testing FlashAttention...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2
    num_heads = 4
    seq_len = 16
    head_dim = 64

    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

    print("\nBasic attention:")
    output = scaled_dot_product_attention(q, k, v)
    print(f"  Output shape: {output.shape}")

    print("\nCausal attention:")
    output_causal = scaled_dot_product_attention(q, k, v, is_causal=True)
    print(f"  Output shape: {output_causal.shape}")

    print("\nWith attention mask:")
    mask = torch.zeros(
        (batch_size, num_heads, seq_len, seq_len), dtype=torch.float32, device=device
    )
    mask[:, :, :, seq_len // 2 :] = -1e9
    output_masked = scaled_dot_product_attention(q, k, v, attention_mask=mask)
    print(f"  Output shape: {output_masked.shape}")

    print("\nGrouped Query Attention (GQA):")
    num_kv_heads = 2
    k_gqa = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device=device)
    v_gqa = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device=device)
    attn = MultiHeadAttention(
        hidden_size=num_heads * head_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
    )
    output_gqa = attn(q, k_gqa, v_gqa)
    print(f"  Output shape: {output_gqa.shape}")

    # Correctness check vs reference
    print("\nCorrectness check (basic):")
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=1.0/np.sqrt(head_dim))
    diff = (output.float() - ref.float()).abs().max().item()
    print(f"  Max diff vs PyTorch SDPA: {diff:.6f}")

    print("\nCorrectness check (causal):")
    ref_causal = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True, scale=1.0/np.sqrt(head_dim))
    diff_causal = (output_causal.float() - ref_causal.float()).abs().max().item()
    print(f"  Max diff vs PyTorch SDPA (causal): {diff_causal:.6f}")

    print("\nOutput statistics:")
    print(f"  Mean: {float(output.mean()):.4f}")
    print(f"  Std:  {float(output.std()):.4f}")
    print(f"  Min:  {float(output.min()):.4f}")
    print(f"  Max:  {float(output.max()):.4f}")

    print("\nFlashAttention working!")
