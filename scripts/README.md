# 脚本使用说明

以下命令均在仓库根目录执行。转换前请将 Nemotron-SFT-SWE-v2 源数据放在 `data/Nemotron-SFT-SWE-v2/data/`；转换结果写入 `LlamaFactory/data/`。数据格式和 `dataset_info.json` 配置请参考 [数据格式说明](../docs/_data-format.md)。

## `parse_nemotron_v2_agentless.py`

将 Agentless JSONL 转换为 LlamaFactory ShareGPT JSON，并将 `reasoning_content` 合并为 assistant 内容中的 `<think>` 块。

```bash
uv run python scripts/parse_nemotron_v2_agentless.py
```

- 输入：`data/Nemotron-SFT-SWE-v2/data/agentless.jsonl`
- 输出：`LlamaFactory/data/nemotron_sft_swe_v2_agentless.json`
- 默认转换前 `COUNT = 10_000` 条；`COUNT` 是脚本常量，需要修改代码才能调整。

## `parse_nemotron_v2_swe.py`

将 SWE agent 轨迹按每个 assistant 决策点展开为逐回合 ShareGPT 训练样本，并输出转换统计报告。

```bash
uv run python scripts/parse_nemotron_v2_swe.py
```

- 输入：`data/Nemotron-SFT-SWE-v2/data/swe.jsonl`
- 输出：`LlamaFactory/data/nemotron_sft_swe_v2_swe_turns.json`
- 直接执行时处理前 `COUNT = 200` 条输入轨迹；一条轨迹会展开为多个样本，因此输出样本数可能更多。
- 训练该逐回合数据集时应设置 `mask_history: true`。

需要自定义输入、输出或条数时，可调用已有 API：

```python
from pathlib import Path

from scripts.parse_nemotron_v2_swe import parse

report = parse(
    Path("输入.jsonl"),
    Path("输出.json"),
    count=1_000,
)
```

## `statis_nemotron_v2.py`

按 LlamaFactory 的 Qwen3.5 模板、`mask_history: true` 和 `enable_thinking: true` 编码规则，统计转换后数据集的 token 长度分布及不同 `cutoff_len` 下的截断比例。该脚本依赖 LlamaFactory 与 Transformers，需在 LlamaFactory 环境中运行。

```bash
uv run --directory LlamaFactory python ../scripts/statis_nemotron_v2.py
```

默认使用 `/cpfs01/llm_team/models/Qwen3.5-35B-A3B` tokenizer，统计两个转换后的数据集。可以指定 tokenizer 和一个或多个数据集：

```bash
uv run --directory LlamaFactory python ../scripts/statis_nemotron_v2.py \
  --tokenizer-path /path/to/Qwen3.5-35B-A3B \
  --datasets data/nemotron_sft_swe_v2_agentless.json data/nemotron_sft_swe_v2_swe_turns.json
```

使用 `--help` 查看全部参数；`--datasets` 下的路径相对于 `LlamaFactory/` 目录。
