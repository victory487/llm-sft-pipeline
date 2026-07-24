"""Nemotron-SFT-SWE-v2 SWE 数据集解析器（逐回合）

将每个 assistant 决策点转换为一条 LlamaFactory ShareGPT 样本。
训练时需要设置 mask_history: true，仅监督每条样本的最后一个 assistant 回合。
"""

import json
import os
import tempfile
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

from tqdm import tqdm

COUNT = 200
DATASET_NAME = "nemotron_sft_swe_v2_swe_turns"


class InvalidRecordError(ValueError):
    """源样本无法转换为合法 ShareGPT 对话。"""


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise InvalidRecordError(f"{field}_not_string")
    return value


def _escape_tool_response_placeholders(text: str) -> str:
    """只转义工具输出中会被识别为多模态占位符的文本。"""
    return text.replace("<audio>", "&lt;audio&gt;").replace("<video>", "&lt;video&gt;")


def _parse_function_calls(tool_calls: Any) -> str:
    if not isinstance(tool_calls, list) or not tool_calls:
        raise InvalidRecordError("tool_calls_missing")

    functions = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict) or not isinstance(tool_call.get("function"), dict):
            raise InvalidRecordError("tool_call_invalid")

        function = tool_call["function"]
        name = function.get("name")
        if not isinstance(name, str) or not name.strip():
            raise InvalidRecordError("tool_name_missing")

        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise InvalidRecordError("tool_arguments_not_json") from exc
        elif arguments is None:
            arguments = {}

        try:
            json.dumps(arguments, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            raise InvalidRecordError("tool_arguments_not_json_serializable") from exc

        functions.append({"name": name, "arguments": arguments})

    return json.dumps(functions, ensure_ascii=False)


def _parse_assistant_message(message: dict[str, Any]) -> dict[str, str]:
    if message.get("tool_calls"):
        return {"from": "function_call", "value": _parse_function_calls(message["tool_calls"])}

    content = _require_text(message.get("content"), "assistant_content")
    if message.get("reasoning_content") is not None:
        reasoning_content = _require_text(message["reasoning_content"], "reasoning_content")
        content = f"<think>\n{reasoning_content}\n</think>\n\n{content}"
    if not content.strip():
        raise InvalidRecordError("assistant_empty")

    return {"from": "gpt", "value": content}


def _validate_conversations(conversations: list[dict[str, str]]) -> None:
    if len(conversations) < 2 or len(conversations) % 2:
        raise InvalidRecordError("conversation_turn_count_invalid")

    for index, message in enumerate(conversations):
        expected_roles = {"human", "observation"} if index % 2 == 0 else {"gpt", "function_call"}
        if message.get("from") not in expected_roles or not isinstance(message.get("value"), str):
            raise InvalidRecordError("conversation_role_order_invalid")


def _parse_swe_sample(sample: dict) -> list[dict]:
    """将一条 SWE 轨迹展开为多个逐 assistant 回合的 ShareGPT 样本。"""
    messages = sample.get("messages")
    if not isinstance(messages, list) or len(messages) < 3:
        raise InvalidRecordError("messages_invalid")
    if not all(isinstance(message, dict) for message in messages):
        raise InvalidRecordError("message_invalid")
    if messages[0].get("role") != "system" or messages[1].get("role") != "user":
        raise InvalidRecordError("initial_roles_invalid")

    tools = sample.get("tools")
    if not isinstance(tools, (list, dict)):
        raise InvalidRecordError("tools_invalid")

    try:
        tools = json.dumps(tools, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise InvalidRecordError("tools_not_json_serializable") from exc

    system = _require_text(messages[0].get("content"), "system_content")
    conversations = [
        {"from": "human", "value": _require_text(messages[1].get("content"), "user_content")}
    ]
    parsed_samples = []
    message_index = 2

    while message_index < len(messages):
        message = messages[message_index]
        if message.get("role") != "assistant":
            raise InvalidRecordError("assistant_expected")

        assistant_message = _parse_assistant_message(message)
        current_conversations = deepcopy(conversations + [assistant_message])
        _validate_conversations(current_conversations)
        parsed_samples.append(
            {
                "conversations": current_conversations,
                "system": system,
                "tools": tools,
            }
        )
        conversations.append(assistant_message)
        message_index += 1

        if assistant_message["from"] == "gpt":
            if message_index != len(messages):
                raise InvalidRecordError("message_after_text_assistant")
            continue

        # 轨迹末尾的 finish 类函数调用没有工具结果，仍是有效监督目标。
        if message_index == len(messages):
            continue

        tool_responses = []
        while message_index < len(messages) and messages[message_index].get("role") == "tool":
            tool_responses.append(
                _escape_tool_response_placeholders(
                    _require_text(messages[message_index].get("content"), "tool_content")
                )
            )
            message_index += 1
        if not tool_responses:
            raise InvalidRecordError("tool_response_missing")

        conversations.append({"from": "observation", "value": "\n".join(tool_responses)})

    return parsed_samples


def _atomic_dump_json(filepath: Path, data: Any) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_filepath = tempfile.mkstemp(
        prefix=f".{filepath.name}.", suffix=".tmp", dir=filepath.parent, text=True
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary_filepath, filepath)
    except Exception:
        if os.path.exists(temporary_filepath):
            os.unlink(temporary_filepath)
        raise


def parse(input_jsonl_filepath: Path, output_json_filepath: Path, count: int = COUNT) -> dict:
    """读取 SWE JSONL，将前 count 条轨迹转换为 JSON 数组。"""
    json_data = []
    skipped_by_reason = Counter()
    report = Counter()
    pbar = tqdm(total=count)

    with open(input_jsonl_filepath, encoding="utf-8") as f:
        for _, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            if report["input_records"] >= count:
                break

            report["input_records"] += 1
            try:
                raw_sample = json.loads(line)
                if not isinstance(raw_sample, dict):
                    raise InvalidRecordError("record_not_object")
                parsed_samples = _parse_swe_sample(raw_sample)
            except json.JSONDecodeError:
                report["skipped"] += 1
                skipped_by_reason["input_json_invalid"] += 1
            except InvalidRecordError as exc:
                report["skipped"] += 1
                skipped_by_reason[str(exc)] += 1
            else:
                report["valid_trajectories"] += 1
                json_data.extend(parsed_samples)
                report["wrote"] += len(parsed_samples)
                report["tool_call_turns"] += sum(
                    sample["conversations"][-1]["from"] == "function_call"
                    for sample in parsed_samples
                )
                report["text_turns"] += sum(
                    sample["conversations"][-1]["from"] == "gpt" for sample in parsed_samples
                )
            pbar.update(n=1)

    pbar.close()
    _atomic_dump_json(output_json_filepath, json_data)
    report["skipped"] += 0
    report["skipped_by_reason"] = dict(skipped_by_reason)
    return dict(report)


if __name__ == "__main__":
    report = parse(
        input_jsonl_filepath=Path("data/Nemotron-SFT-SWE-v2/data/swe.jsonl"),
        output_json_filepath=Path("LlamaFactory/data/nemotron_sft_swe_v2_swe_turns.json"),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
