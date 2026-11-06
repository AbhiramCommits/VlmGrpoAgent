"""Optional TRL ``GRPOTrainer`` route (single-turn) for comparison.

The main results in this repo use the custom multi-turn loop in
``vlmgrpo/train_grpo.py`` because TRL's GRPOTrainer samples one completion per
prompt and does not execute tools between turns.  This module demonstrates the
same verifiable reward functions plugged into TRL when it is installed, and
skips cleanly (exit 0) when the environment cannot run it.

Run: ``python -m vlmgrpo.train_grpo_trl --config configs/tiny.yaml --dry-run``
"""

from __future__ import annotations

import argparse
import json

from .config import load_config
from .rewards import RewardConfig
from .utils import read_jsonl
from .verify import format_ok, parse_answer_text, verify_answer


def trl_available() -> tuple[bool, str]:
    try:
        import trl

        return True, getattr(trl, "__version__", "unknown")
    except Exception as exc:  # pragma: no cover - depends on environment
        return False, str(exc)


# --------------------------------------------------------------------------- #
# Reward functions with the same signature TRL expects: (completions, **kwargs)
# --------------------------------------------------------------------------- #


def reward_answer(completions, gold_answers=None, **kwargs):
    gold_answers = gold_answers or [""] * len(completions)
    return [1.0 if verify_answer(parse_answer_text(c) or c, g) else 0.0 for c, g in zip(completions, gold_answers)]


def reward_format(completions, **kwargs):
    return [1.0 if format_ok(c) else 0.0 for c in completions]


def reward_tool_schema(completions, **kwargs):
    import re

    from .verify import verify_tool_call_schema

    scores = []
    for completion in completions:
        tools = re.findall(r"<tool>(.*?)</tool>", completion, flags=re.DOTALL)
        scores.append(sum(1.0 for t in tools if verify_tool_call_schema(t)) / len(tools) if tools else 0.0)
    return scores


def build_dataset(config_path: str, limit: int = 64):
    from datasets import Dataset

    cfg = load_config(config_path)
    rows = read_jsonl(f"{cfg.data_dir}/train.jsonl")[:limit]
    return Dataset.from_dict(
        {
            "prompt": [r["question"] for r in rows],
            "image": [r["image_paths"][0] for r in rows],
            "gold_answer": [r["gold_answer"] for r in rows],
        }
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TRL GRPOTrainer route (optional)")
    parser.add_argument("--config", default="configs/tiny.yaml")
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args(argv)

    ok, version = trl_available()
    print(f"[trl] available={ok} version={version}")
    if not ok:
        print("[trl] Install `trl` to use this route; the custom GRPO loop is the supported path.")
        return 0

    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    from .registry import detect_device, load_vlm, pick_model_tier

    cfg = load_config(args.config)
    device = detect_device()
    tier = pick_model_tier(device, override=cfg.model_tier)
    if device != "cuda":
        print(f"[trl] device={device}: TRL GRPOTrainer here targets CUDA; skip (use the custom loop).")
    dataset = build_dataset(args.config)
    print(f"[trl] dataset rows={len(dataset)} reward_cfg={RewardConfig().__dict__}")

    if args.dry_run:
        plan = {
            "trainer": "trl.GRPOTrainer",
            "model_tier": tier,
            "device": device,
            "rows": len(dataset),
            "reward_funcs": ["reward_answer", "reward_format", "reward_tool_schema"],
            "note": "multi-turn tool execution is handled by vlmgrpo.train_grpo",
        }
        print(json.dumps(plan, indent=2))
        return 0

    model, processor = load_vlm(tier, device, cfg.dtype)
    peft_config = LoraConfig(
        r=cfg.sft.lora_r,
        lora_alpha=cfg.sft.lora_alpha,
        lora_dropout=cfg.sft.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    training_args = GRPOConfig(
        output_dir=f"{cfg.runs_dir}/{cfg.run_id}/grpo_trl",
        max_steps=args.steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=cfg.grpo.group_size,
        learning_rate=cfg.grpo.lr,
        num_generations=cfg.grpo.group_size,
        temperature=cfg.grpo.temperature,
        max_completion_length=cfg.grpo.max_new_tokens,
        logging_steps=1,
        remove_unused_columns=False,
        report_to=[],
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_answer, reward_format, reward_tool_schema],
        args=training_args,
        train_dataset=dataset,
        processing_class=processor,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(f"{cfg.runs_dir}/{cfg.run_id}/grpo_trl")
    print("[trl] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
