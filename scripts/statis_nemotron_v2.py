"""统计 Nemotron-SFT-SWE-v2 数据集在 LlamaFactory Qwen3.5 模板下的 token 长度分布。

运行：
    uv run --directory LlamaFactory python ../scripts/statis_nemotron_v2.py

统计长度与 LlamaFactory SFT 预处理一致：包含聊天模板、system/tools 字段，
并使用 ``mask_history: true``、``enable_thinking: true`` 的编码规则。
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parents[1]
LLAMAFACTORY_DIR = PROJECT_DIR / "LlamaFactory"
DEFAULT_TOKENIZER_PATH = Path("/cpfs01/llm_team/models/Qwen3.5-35B-A3B")
DEFAULT_DATASETS = (
    LLAMAFACTORY_DIR / "data/nemotron_sft_swe_v2_agentless.json",
    LLAMAFACTORY_DIR / "data/nemotron_sft_swe_v2_swe_turns.json",
)
PERCENTILES = (50, 90, 95, 99, 99.5, 99.9, 100)
CUTOFF_LENS = (4096, 8192, 16384, 32768, 65536)
ROLE_MAPPING = {
    "human": "user",
    "gpt": "assistant",
    "observation": "observation",
    "function_call": "function",
}


@dataclass(frozen=True)
class DatasetStats:
    """单个数据集的长度统计结果。"""

    name: str
    lengths: list[int]


def _load_template(tokenizer_path: Path):
    """加载与训练配置相同的 Qwen3.5 tokenizer 和 LlamaFactory 模板。"""
    sys.path.insert(0, str(LLAMAFACTORY_DIR / "src"))

    from transformers import AutoTokenizer

    from llamafactory.data.template import get_template_and_fix_tokenizer
    from llamafactory.hparams.data_args import DataArguments

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, local_files_only=True)
    data_args = DataArguments(template="qwen3_5", enable_thinking=True, mask_history=True)
    return tokenizer, get_template_and_fix_tokenizer(tokenizer, data_args)


def _to_messages(sample: dict[str, Any]) -> list[dict[str, str]]:
    """将 ShareGPT conversations 转为 LlamaFactory 模板需要的内部角色。"""
    messages = []
    for message in sample["conversations"]:
        role = message["from"]
        if role not in ROLE_MAPPING:
            raise ValueError(f"不支持的对话角色：{role}")
        messages.append({"role": ROLE_MAPPING[role], "content": message["value"]})
    return messages


def _get_length(sample: dict[str, Any], tokenizer: Any, template: Any) -> int:
    """计算样本在 SFT 截断前的完整 token 数。"""
    messages = _to_messages(sample)
    encoded_pairs = template.encode_multiturn(
        tokenizer,
        messages,
        sample.get("system"),
        sample.get("tools"),
        discarding_history_cot=True,
    )
    return sum(len(source_ids) + len(target_ids) for source_ids, target_ids in encoded_pairs) + int(template.efficient_eos)


def _percentile(sorted_lengths: list[int], percentile: float) -> int:
    """返回 nearest-rank 百分位数。"""
    index = max(0, (len(sorted_lengths) * percentile + 99) // 100 - 1)
    return sorted_lengths[int(index)]


def _stat_dataset(filepath: Path, tokenizer: Any, template: Any) -> DatasetStats:
    """读取一个 JSON 数据集并统计每个样本的 token 长度。"""
    with open(filepath, encoding="utf-8") as f:
        samples: list[dict[str, Any]] = json.load(f)

    lengths = []
    for sample in tqdm(samples, desc=filepath.stem, unit="sample"):
        lengths.append(_get_length(sample, tokenizer, template))

    return DatasetStats(name=filepath.stem, lengths=lengths)


def _print_stats(stats: DatasetStats) -> None:
    """输出百分位数和不同 cutoff_len 下的截断比例。"""
    lengths = sorted(stats.lengths)
    print(f"\n数据集：{stats.name}")
    print(f"样本数：{len(lengths):,}")
    print(f"平均长度：{sum(lengths) / len(lengths):,.1f}")
    print(f"最小/最大长度：{lengths[0]:,} / {lengths[-1]:,}")
    print("长度百分位数：")
    for percentile in PERCENTILES:
        print(f"  p{percentile:g}: {_percentile(lengths, percentile):,}")

    print("cutoff_len 截断情况：")
    for cutoff_len in CUTOFF_LENS:
        truncated = sum(length > cutoff_len for length in lengths)
        print(f"  {cutoff_len:>5,}: {truncated:>6,} / {len(lengths):,} ({truncated / len(lengths):.2%})")


def main() -> None:
    parser = argparse.ArgumentParser(description="统计 Nemotron-SFT-SWE-v2 的 token 长度分布。")
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=DEFAULT_TOKENIZER_PATH,
        help=f"Tokenizer 路径（默认：{DEFAULT_TOKENIZER_PATH}）。",
    )
    parser.add_argument(
        "--datasets",
        type=Path,
        nargs="+",
        default=DEFAULT_DATASETS,
        help="待统计的 JSON 数据集路径。",
    )
    args = parser.parse_args()

    tokenizer, template = _load_template(args.tokenizer_path)
    for filepath in args.datasets:
        _print_stats(_stat_dataset(filepath, tokenizer, template))


if __name__ == "__main__":
    main()
