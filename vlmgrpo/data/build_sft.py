"""Build SFT cold-start data from gold trajectories.

Each train/val task's minimal gold tool sequence is executed through the *real*
tool sandbox, so the supervised targets contain real observations.  Examples are
rendered in exactly the chat/ReAct format used by the rollout agent, one example
per assistant turn (prefix -> next assistant message).
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from ..config import RunConfig, load_config
from ..prompts import build_messages, format_prompt, observation_text
from ..registry import detect_device, load_processor, pick_model_tier
from ..tasks.families import TaskItem
from ..tools import ToolSandbox
from ..utils import encode_text, read_jsonl, seed_everything, write_json, write_jsonl

TOOL_THINK = {
    "python_exec": "I will use Python to compute the aggregate exactly.",
    "retrieve": "I will retrieve the text values rendered in the image.",
    "click": "I will open the card so its fields can be read.",
    "read_field": "I will read the requested field from the open card.",
}
ANSWER_THINK = "The observation gives the value I need; I will answer with it."


def _target_text(name: str, args: dict[str, Any], raw_tool: str | None = None) -> str:
    payload = raw_tool or json.dumps({"name": name, "args": args}, ensure_ascii=False)
    return f"<think>{TOOL_THINK.get(name, 'I will call the tool.')}</think>\n<tool>{payload}</tool>"


def build_examples_for_item(item: TaskItem, max_turns: int = 4) -> list[dict[str, Any]]:
    """Execute the gold calls and return the chat turns (text only)."""
    sandbox = ToolSandbox(item)
    messages = build_messages(item.question)
    turns: list[dict[str, Any]] = []
    for call in item.gold_tool_calls[:max_turns]:
        name = call["name"]
        args = call.get("args", {})
        target = _target_text(name, args)
        turns.append({"role": "assistant", "content": target})
        result = sandbox.execute(name, args)
        turns.append({"role": "user", "content": observation_text(name, args, result)})
    turns.append(
        {"role": "assistant", "content": f"<think>{ANSWER_THINK}</think>\n<answer>{item.gold_answer}</answer>"}
    )
    return turns


def render_examples(
    items: list[TaskItem],
    processor: Any,
    max_length: int = 1280,
    max_turns: int = 4,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Render and tokenize every assistant turn; return records + length stats."""
    records: list[dict[str, Any]] = []
    skipped = 0
    prompt_lens: list[int] = []
    target_lens: list[int] = []
    all_messages: list[list[dict[str, Any]]] = []

    for item in items:
        sandbox = ToolSandbox(item)
        messages = build_messages(item.question)
        turns = list(item.gold_tool_calls[:max_turns])
        targets = [_target_text(c["name"], c.get("args", {})) for c in turns]
        targets.append(f"<think>{ANSWER_THINK}</think>\n<answer>{item.gold_answer}</answer>")
        for turn_index, target in enumerate(targets):
            prompt = format_prompt(processor, messages)
            from PIL import Image

            image = Image.open(item.image_path).convert("RGB")
            enc = processor(text=[prompt], images=[image], return_tensors="pt", padding=False)
            n_prompt = int(enc["input_ids"].shape[1])
            n_target = len(encode_text(processor.tokenizer, target)) + 1
            total = n_prompt + n_target
            if total > max_length:
                skipped += 1
            else:
                records.append(
                    {
                        "task_id": item.task_id,
                        "family": item.family,
                        "turn": turn_index,
                        "prompt_text": prompt,
                        "target_text": target,
                        "image_path": item.image_path,
                        "n_prompt_tokens": n_prompt,
                        "n_target_tokens": n_target,
                        "total_tokens": total,
                    }
                )
                prompt_lens.append(n_prompt)
                target_lens.append(n_target)
                all_messages.append(list(messages))
            if turn_index < len(turns):
                call = turns[turn_index]
                result = sandbox.execute(call["name"], call.get("args", {}))
                messages.append({"role": "assistant", "content": target})
                messages.append(
                    {
                        "role": "user",
                        "content": observation_text(call["name"], call.get("args", {}), result),
                    }
                )
    stats = {
        "examples": len(records),
        "skipped_over_max_length": skipped,
        "tasks": len(items),
        "max_length": max_length,
        "prompt_tokens_mean": round(statistics.mean(prompt_lens), 1) if prompt_lens else 0,
        "prompt_tokens_median": round(statistics.median(prompt_lens), 1) if prompt_lens else 0,
        "prompt_tokens_p95": round(sorted(prompt_lens)[int(0.95 * (len(prompt_lens) - 1))], 1) if prompt_lens else 0,
        "target_tokens_mean": round(statistics.mean(target_lens), 1) if target_lens else 0,
        "total_tokens_mean": round(statistics.mean([r["total_tokens"] for r in records]), 1) if records else 0,
        "total_tokens_max": max((r["total_tokens"] for r in records), default=0),
    }
    return records, stats


def build_sft_split(
    data_dir: Path,
    split: str,
    processor: Any,
    max_length: int,
    max_turns: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = data_dir / f"{split}.jsonl"
    items = [TaskItem.from_dict(row) for row in read_jsonl(path)]
    rows, stats = render_examples(items, processor, max_length=max_length, max_turns=max_turns)
    out = data_dir / f"sft_{split}.jsonl"
    write_jsonl(out, rows)
    print(f"[sft-data] {split}: {len(rows)} examples from {len(items)} tasks -> {out}")
    print(f"[sft-data] {split} stats: {json.dumps(stats, sort_keys=True)}")
    return rows, stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build gold-trajectory SFT data")
    parser.add_argument("--config", default="configs/tiny.yaml")
    parser.add_argument("--model-tier", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-length", type=int, default=None)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    seed_everything(cfg.seed)
    device = args.device or cfg.device
    if device == "auto":
        device = detect_device()
    tier = pick_model_tier(device, override=args.model_tier or cfg.model_tier)
    processor = load_processor(tier)
    max_length = args.max_length or cfg.sft.max_length
    data_dir = Path(cfg.data_dir)
    all_stats = {}
    for split in ("train", "val"):
        _, stats = build_sft_split(data_dir, split, processor, max_length, cfg.rollout.max_turns)
        all_stats[split] = stats
    write_json(data_dir / "sft_stats.json", all_stats)
    print(f"[sft-data] wrote {data_dir / 'sft_stats.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
