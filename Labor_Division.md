# Triton ASR项目TODO分工计划（4人并行，均分工作量）

基于项目文件（`rope.py`、`layers.py`、`attention.py`）的分析，总共有**10个TODO内核**需要实现（每个内核涉及Triton代码编写、调试和测试）。为了均分工作量，我调整了之前的分配，确保每个人的任务数量和复杂度大致均衡（每个组2-4个内核，复杂度考虑：简单激活<规范化<位置嵌入<矩阵乘法<注意力）。复杂度评估基于代码量和数学运算（例如，激活函数简单，注意力复杂但相关内核可批量实现）。

## 总体计划概述
- **目标**：完成所有Triton内核，实现端到端ASR模型的基础组件。
- **总时间**：10-14天（假设每天2-4小时工作，考虑调试时间）。
- **里程碑**：
  - **Day 1-3**：理解Triton语法，实现内核代码（每个人独立）。
  - **Day 4-7**：本地测试（运行各文件的`if __name__ == "__main__"`部分），修复bug。
  - **Day 8-10**：集成测试（合并到`model.py`或其他文件中，验证端到端）。
  - **Day 11-14**：优化和文档（如果有性能TODO）。
- **协作工具**：使用Git分支（每个人一个分支，如`feature-personA-rope`），每天推送代码并合并到主分支。使用Discord/微信群同步进度，遇到问题时分享代码片段。
- **依赖**：任务相对独立，但注意力（人D）可能依赖RoPE（人A）——人D可先用Torch fallback测试。
- **测试方法**：每个内核完成后，运行对应文件的测试脚本（e.g., `python rope.py`）。如果失败，优先修复（最多3次迭代）。
- **风险**：如果某人进度慢，可临时调配（e.g., 人A帮人D）。

## 每个人的详细TODO列表
分配原则：均分数量（2-3个），复杂度均衡（简单任务配中等任务）。每个TODO包括：文件、内核名、描述、步骤提示、预期时间。

### 人A（RoPE + 激活函数，2个内核，中等复杂度）
- **TODO 1: compute_freqs_kernel (rope.py)**  
  描述：实现RoPE的频率计算内核，用于预计算cos/sin缓存。  
  步骤：加载位置和逆频率，计算freqs = position * inv_freq，计算cos/sin，存储到缓存。  
  预期时间：2-3小时（中等，涉及数学运算）。  
  测试：运行`rope.py`的主测试，检查cos/sin形状和值。
- **TODO 2: gelu_kernel (layers.py)**  
  描述：实现GELU激活内核（使用tanh近似）。  
  步骤：加载输入tile，计算tanh近似（sqrt(2/pi) * (x + 0.044715 * x^3)），应用激活，存储输出。  
  预期时间：1-2小时（简单，纯数学）。  
  测试：运行`layers.py`的GELU测试，验证输出与Torch一致。

### 人B（规范化层 + 线性，3个内核，中等-复杂）
- **TODO 1: rmsnorm_kernel (layers.py)**  
  描述：实现RMSNorm内核（x / RMS(x) * weight）。  
  步骤：加载输入行和权重，计算方差=均值(x^2)，归一化x / sqrt(variance + eps)，应用权重，存储。  
  预期时间：2-3小时（中等，涉及统计计算）。  
  测试：运行`layers.py`的RMSNorm测试，检查输出形状和归一化效果。
- **TODO 2: layernorm_kernel (layers.py)**  
  描述：实现LayerNorm内核（(x - mean) / sqrt(var + eps) * weight + bias）。  
  步骤：加载输入、权重、偏置，计算均值和方差，中心化，归一化，应用仿射变换。  
  预期时间：2-3小时（中等，与RMSNorm类似但多偏置）。  
  测试：运行`layers.py`的LayerNorm测试，验证输出。
- **TODO 3: linear_kernel_tf32 (layers.py)**  
  描述：实现TF32风格矩阵乘法内核（A @ B）。  
  步骤：初始化累加器，循环K tiles，累加tl.dot，存储结果。  
  预期时间：3-4小时（复杂，涉及tiled矩阵乘法）。  
  测试：运行`layers.py`的Linear测试。

### 人C（激活 + Softmax，2个内核，中等复杂度）
- **TODO 1: silu_kernel (layers.py)**  
  描述：实现SiLU激活内核（x * sigmoid(x)）。  
  步骤：加载输入tile，计算sigmoid=1/(1+exp(-x))，乘以x，存储。  
  预期时间：1-2小时（简单，类似GELU）。  
  测试：运行`layers.py`的SiLU测试。
- **TODO 2: softmax_kernel (layers.py)**  
  描述：实现Softmax内核（数值稳定版本）。  
  步骤：加载行（带masking），减去max，计算exp，归一化，存储。  
  预期时间：2-3小时（中等，涉及循环和masking）。  
  测试：运行`layers.py`的Softmax测试，检查sum=1。

### 人D（注意力，3个内核，高复杂度，但相关可批量处理）
- **TODO 1: attention_scores_kernel (attention.py)**  
  描述：实现注意力分数计算内核（Q @ K^T * scale）。  
  步骤：加载query和keys，计算点积，缩放，存储分数。  
  预期时间：2-3小时（中等，作为注意力的一部分）。  
  测试：运行`attention.py`的主测试。
- **TODO 2: softmax_inplace_kernel (attention.py)**  
  描述：实现就地Softmax内核（用于注意力）。  
  步骤：类似layers的softmax，但就地修改。  
  预期时间：1-2小时（简单，复用逻辑）。  
  测试：集成到注意力测试中。
- **TODO 3: attention_output_kernel (attention.py)**  
  描述：实现注意力输出内核（weights @ V）。  
  步骤：加载权重和values，计算加权和，存储输出。  
  预期时间：2-3小时（中等）。  
  测试：运行`attention.py`的主测试，检查输出形状。

## 额外建议
- **优先级**：先做简单任务（如激活函数），再做复杂任务（如注意力）。
- **帮助**：如果卡住，可以问我生成代码片段或运行测试（e.g., 用`run_in_terminal`运行脚本）。
- **验证**：完成后，确保所有内核在CUDA上运行（如果有GPU）。如果没有，fallback到CPU。
- **调整**：如果工作量不均（e.g., 人D太忙），可以交换任务（如人C的Softmax换给人D）。

这个计划确保并行高效。如果需要调整或更多细节，告诉我！