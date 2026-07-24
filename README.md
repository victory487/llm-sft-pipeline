# 高质量 SFT 数据过滤

设计高质量数据过滤方案，以期通过 SFT 来提升模型的 Coding 和 Code Agent 能力。

## 实验结果

实验配置：

- 基座模型：`Qwen3.5-35B-A3B`
- 数据集：[Nemotron-SFT-SWE-v2](https://huggingface.co/datasets/nvidia/Nemotron-SFT-SWE-v2)

| Method | SWE-bench Lite |
| :--- | :---: |
| Qwen3.5-35B-A3B | 69.3 |
| 基于 Nemotron-SFT-SWE-v2 Agentless（前 10000 条） | - |
| 基于 Nemotron-SFT-SWE-v2 SWE（前 10000 条） | - |

后续思路：

- 切换优化器，例如从默认的 AdamW 切换为 Muon。
