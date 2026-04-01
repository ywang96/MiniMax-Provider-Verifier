# MiniMax-Provider-Verifier

[English](README.md) | [中文](README_CN.md)

**MiniMax-Provider-Verifier** 提供了一个严格、中立的验证方法，旨在评估第三方部署的 MiniMax M2 模型的正确性与可靠性。自 M2 开源以来，大量用户将其集成至生产环境。为了确保广大用户能够持续获得高效、高质量的体验，并践行我们 "Intelligence with Everyone" 的愿景，本工具包提供了一套客观且可复用的标准，用于验证模型部署是否符合预期行为。

## 评估指标

我们从工具调用（Tool Calling）、模式正确性及配置稳定性（如 top-k 设置）等多个维度对供应商的部署进行评估。

主要指标包括：

* **请求成功率（Query-Success-Rate）：** 衡量在允许最多 `max_retry=10` 次尝试的情况下，供应商最终成功返回有效响应的概率。
  - `query_success_rate = successful_query_count / total_query_count`

* **工具调用匹配率（ToolCalls-Match-Rate）：** 衡量模型在"是否触发工具调用"这一行为上与预期标签的匹配程度。每个测试用例都标注了 `expected_tool_call`（是否预期触发工具调用），该指标计算预期与实际结果相符的比例。
  - `tool_calls_match_rate = (tool_calls_finish_tool_calls + stop_finish_stop) / success_count`
  - 四项限统计（混淆矩阵）：
    - `tool_calls_finish_tool_calls`: 预期 tool_call，实际 tool_call (TP)
    - `tool_calls_finish_stop`: 预期 tool_call，实际 stop (FN)
    - `stop_finish_tool_calls`: 预期 stop，实际 tool_call (FP)
    - `stop_finish_stop`: 预期 stop，实际 stop (TN)

* **工具调用参数准确率（ToolCalls-Schema-Accuracy）：** 在触发工具调用的前提下，生成的函数名和参数符合预期模式的比例。
  - `schema_accuracy = tool_calls_successful_count / tool_calls_finish_tool_calls`


* **响应有效率（Response-Success-Rate Not Only Reasoning）：** 检测模型是否陷入“仅输出推理过程（Chain-of-Thought）但无有效内容或工具调用”的错误模式。此类模式通常意味着部署存在问题。
  - `Response-success-rate = response_not_only_reasoning_count / only_reasoning_checked_count`

* **语言遵循成功率（Language-Following-Success-Rate）：** 在小语种场景下，检查模型是否能正确遵循语言指令。该指标对 top-k 等解码参数非常敏感。
  - `language_following_success-rate = language_following_valid_count / language_following_checked_count`

## 评估结果

基于我们发布的初始测试集，以下是各供应商的评估结果（每个供应商执行 10 次取平均值）。其中 `minimax` 代表[官方 MiniMax 开放平台](https://platform.minimaxi.com/)的部署表现，作为基准参考。

### MiniMax-M2.5/M2.7 模型，26年4月数据（指标改版后）

| 指标 | 请求成功率 | 工具调用匹配率 | 工具调用准确率 | 响应有效率 | 语言遵循成功率 |
|--------|--------------------|-----------------------------|--------------------|--------------------------------------------|----------------------------------|
| MiniMax-M2.5 | 100% | 99.19% | 96.31% | 100% | 80% |
| MiniMax-M2.7 | 100% | 99.29% | 96.66% | 100% | 80% |

### MiniMax-M2.5 模型，26年2月数据（指标改版前）

| 指标 | 请求成功率 | 工具调用率 | 工具调用触发相似度 | 工具调用准确率 | 响应有效率 | 语言遵循成功率 |
|--------|--------------------|-----------------------|------------------------------|--------------------|--------------------------------------------|----------------------------------|
| minimax-m2.5 | 100% | 84.75% | - | 97.26% | 100% | 90% |
| openRouter-minimax-fp8 | 100% | 84.55% | 98.98% | 97.25% | 100% | 80% |
| openRouter-minimax-highspeed | 100% | 84.14% | 99.22% | 97.24% | 100% | 80% |
| openRouter-novita-bf16 | 100% | 84.65% | 99.05% | 97.5% | 100% | 70% |
| openRouter-siliconflow/fp8 | 100% | 84.24% | 99.28% | 98.68% | 100% | 80% |
| openRouter-atlas-cloud/fp8 | 100% | 84.75% | 99.10% | 96.18% | 100% | 70% |
| openRouter-fireworks | 96.32% | 81.63% | 98.87% | 96.19% | 100% | 80% |

### MiniMax-M2.1模型，26年1月数据

| 指标 | 请求成功率 | 工具调用率 | 工具调用触发相似度 | 工具调用准确率 | 响应有效率 | 语言遵循成功率 |
|--------|--------------------|-----------------------|------------------------------|--------------------|--------------------------------------------|----------------------------------|
| minimax-m2.1 | 100% | 83.33% | - | 96.61% | 100% | 90.00% |
| minimax-m2.1-vllm(without topk) | 99.90% | 81.84% | 98.78% | 96.42% | 100% | 60.00% |
| minimax-m2.1-vllm | 100% | 82.83% | 98.90% | 93.91% | 100% | 90% |
| minimax-m2.1-sglang | 100% | 83.03% | 99.15% | 95.01% | 100% | 90% |
| infini-ai | 100% | 80.61% | 97.46% | 100% | 100% | 100% |
| openRouter-minimax/fp8 | 100% | 83.23% | 99.03% | 96.11% | 100% | 90% |
| openRouter-minimax/lightning | 99.90% | 83.15% | 98.97% | 96.48% | 100% | 80% |
| openRouter-gmicloud/fp8 | 83.72% | 55.5% | 81.37% | 84.58% | 100% | 70% |
| OpenRouter-novita/fp8 | 99.32% | 83.07% | 99.21% | 96.03% | 100% | 90% |
| fireworks | 100% | 81.1% | 97.77% | 94.29% | 100% | 60% |
| siliconflow | 100% | 82.42% | 98.47% | 96.19% | 100% | 60% |


### MiniMax-M2模型，25年12月数据

| 指标 | 请求成功率 | 工具调用率 | 工具调用触发相似度 | 工具调用准确率 | 响应有效率 | 语言遵循成功率 |
|--------|--------------------|-----------------------|------------------------------|--------------------|--------------------------------------------|----------------------------------|
| minimax | 100% | 83.74% | - | 99.16% | 100% | 90.00% |
| minimax-vllm | 100% | 82.1% | 99.39% | 97.93% | 100% | 50% |
| minimax-sglang | 100% | 82.2% | 99.39% | 98.42% | 100% | - |
| openrouter-atlas-cloud | 90.7% | 76.6% | 98.78% | 98.83% | 99.7% | 50% |
| openrouter-deepinfra | 99.50% | 82.82% | 99.52% | 98.67% | 99.70% | 20.00% |
| openrouter-google-vertex | 100.00% | 82.93% | 99.33% | 98.06% | 99.49% | 40.00% |
| siliconflow | 100.00% | 83.23% | 99.39% | 97.82% | 100.00% | 40.00% |
| fireworks | 100% | 80.2% | 97.72% | 97.86% | 100% | 50% |


## 参考阈值

基于当前的评估集，我们提供以下参考阈值来解释供应商性能。符合这些阈值通常意味着部署配置正确且运行稳定：

* **请求成功率** (使用 **max_retry=10**):
应为 **100%**，表明模型可以在合理的重试预算内稳定返回结果。

* **完成工具调用率**:
应 **≈80%**，基于当前评估集，在内部重复测试中，波动幅度通常在 ±2.5% 以内。

* **工具调用触发相似度**:
应 **≥98%**，对官方渠道进行 10 次重复测试后观察到最小相似度为 98.2%。因此，98% 是一个合理的下限。

* **工具调用准确率**:
应 **≥98%**，反映格式标准且符合模式的工具调用。

* **响应有效率**:
应为 **100%**，任何"仅推理"输出（无最终答案或工具调用）的出现都显著表明部署存在异常。

* **语言遵循成功率**:
应 **≥40%**，基于内部多次评估结果，低于此阈值表明潜在的解码问题，在小语种场景尤为明显。

## 快速开始

使用 `verify.py` 脚本即可针对 JSONL 测试集运行验证管道。脚本支持并发请求，并会自动输出详细日志及摘要。

### 示例

```bash
python verify.py sample.jsonl \
  --model minimax/minimax-m2 \
  --base-url https://YOUR_BASE_URL/v1 \
  --api-key YOUR_API_KEY \
  --concurrency 5
```

### 参数说明

- `sample.jsonl`: JSONL 测试集的路径。
- `--model` (必需): 评估模型名称。示例：`minimax/minimax-m2`。
- `--base-url` (必需): 供应商的 API base url。示例：`https://YOUR_BASE_URL/v1`。
- `--api-key` (可选): 认证令牌；或者在环境中设置 `OPENAI_API_KEY`。
- `--concurrency` (可选，整数，默认值=5): 最大并发请求数。
- `--output` (可选，默认值=results.jsonl): 写入详细的每个输入结果的文件。
- `--summary` (可选，默认值=summary.json): 写入从结果计算的统计指标的文件。
- `--timeout` (可选，整数，默认值=600): 每个请求的超时时间（秒）。
- `--retries` (可选，整数，默认值=3): 每个请求的最大自动重试次数。
- `--extra-body` (可选，JSON 字符串): 要合并到每个请求中的额外字段（例如解码参数）。
- `--incremental` (标志): 仅重新运行之前失败或新的请求，与现有输出合并，重新统计结果。

### 批量测试（推荐：pass@10）

为了获得具有统计显著性的全面评估结果，我们推荐使用批量验证脚本，该脚本会执行多次迭代并聚合指标：

```bash
bash run_batch_sequential.sh \
  --module 'provider-name' \
  --url 'https://api.example.com/v1/' \
  --model 'model-name' \
  --api-key 'your-api-key' \
  --max-workers 10 \
  --mm-model 'MiniMax-M2.5'
```

**参数说明：**

- `--module`: Provider/模块名称（必需）
- `--url`: API 端点 URL（必需）
- `--model`: 模型名称（必需）
- `--api-key`: API 密钥（必需）
- `--max-workers`: 并发请求数（默认：10）
- `--stream`: 启用流式模式
- `--debug`: 调试模式，仅运行前 10 个用例
- `--extra-body`: 额外的请求体参数（JSON 格式）
- `--mm-model`: 用于对比的 MiniMax 基准模型名称（默认：MiniMax-M2.5）

脚本将执行以下操作：
1. 执行 10 次验证循环（pass@10）
2. 计算所有循环的聚合指标
3. 与基准模型进行对比
4. 在 `output-dir/batch_<timestamp>/` 中生成详细的指标报告

**输出文件：**
- `metrics_report.json`: 所有循环的聚合指标
- `comparison_report.json`: 与基准模型的对比结果
- 各循环的单独结果位于 `loop_01/`、`loop_02/` 等目录

## 后续计划

- [ ] 扩展和完善评估集。
- [ ] 跟踪和测试更多供应商。

## 致谢

ToolCalls-Trigger Similarity 指标的定义方式借鉴自 K2-Vendor-Verifier 框架，并与其保持一致。参见：[MoonshotAI/K2-Vendor-Verifier](https://github.com/MoonshotAI/K2-Vendor-Verifier)。

