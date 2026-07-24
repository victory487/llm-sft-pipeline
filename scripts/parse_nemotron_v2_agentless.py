"""Nemotron-SFT-SWE-v2 Agentless 数据解析器

- 链接：https://huggingface.co/datasets/nvidia/Nemotron-SFT-SWE-v2
- 运行：python scripts/parse_nemotron_v2_agentless.py
- 配置：可以控制 COUNT 参数决定解析数据的数量
"""

import json
from pathlib import Path

from tqdm import tqdm

COUNT = 10_000


def _parse_agentless_sample(sample: dict) -> dict:
    """将样本从单轮对话格式转换为 LlamaFactory 的 ShareGPT 格式
    格式参考：https://llamafactory.readthedocs.io/zh-cn/latest/getting_started/data_preparation.html#id22
    """

    def _escape_mm_placeholders(text: str) -> str:
        # 文本中的字面量 <audio>/<image>/<video> 会被 LlamaFactory 计为多模态占位符
        for token in ("<audio>", "<image>", "<video>"):
            text = text.replace(token, token[0] + " " + token[1:])
        return text

    parsed_sample = {
        "conversations": [
            {
                "from": "human",
                "value": _escape_mm_placeholders(sample["messages"][0]["content"]),
            },
            {
                "from": "gpt",
                "value": _escape_mm_placeholders(
                    "<think>\n"
                    + sample["messages"][1]["reasoning_content"]
                    + "\n</think>\n\n"
                    + sample["messages"][1]["content"]
                ),
            },
        ]
    }

    return parsed_sample


def parse(input_jsonl_filepath: Path, output_json_filepath: Path) -> None:
    # parse
    json_data = []
    pbar = tqdm(total=209976)
    with open(input_jsonl_filepath, encoding="utf-8") as f:
        for line in f:
            line.strip()
            if not line:
                continue
            raw_sample: dict = json.loads(line)
            parsed_sample: dict = _parse_agentless_sample(raw_sample)
            json_data.append(parsed_sample)
            pbar.update(n=1)
            if pbar.n >= COUNT:
                break

    # save
    with open(output_json_filepath, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parse(
        input_jsonl_filepath=Path("data/Nemotron-SFT-SWE-v2/data/agentless.jsonl"),
        output_json_filepath="LlamaFactory/data/nemotron_sft_swe_v2_agentless.json",
    )
