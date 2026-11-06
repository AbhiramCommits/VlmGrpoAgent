"""GRPO / RLVR online RL loop starting from the SFT cold-start checkpoint.

Per step:
  1. sample a group of G rollouts for one prompt at temperature T
  2. score every rollout with the verifiable reward stack (no reward model)
  3. compute group-relative advantages (skip zero-variance groups)
  4. token-level policy gradient over assistant tokens with PPO clipping + KL
     to the frozen SFT reference, applying the trajectory advantage to every
     generated turn (or only the last turn with ``last_turn_only``).
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from .agent import make_generator, run_episodes
from .config import RunConfig, load_config
from .grpo import compute_advantages, grpo_loss, token_logprobs
from .prompts import build_messages, format_prompt
from .registry import detect_device, dtype_name, load_vlm, pick_model_tier, resolve_dtype
from .rewards import reward_breakdown
from .tasks.families import TaskItem
from .tracking import Tracker
from .utils import (
    count_params,
    encode_text,
    git_sha,
    human_hours,
    read_jsonl,
    register_checkpoint,
    resolve_checkpoint,
    seed_everything,
    write_json,
)
from .verify import verify_answer, verify_tool_trace

IMAGE_CACHE: dict[str, Image.Image] = {}


def _image(path: str) -> Image.Image:
    if path not in IMAGE_CACHE:
        IMAGE_CACHE[path] = Image.open(path).convert("RGB")
    return IMAGE_CACHE[path]


def build_turn_examples(
    trajectories: list[Any],
    items: list[TaskItem],
    advantages: torch.Tensor,
    processor: Any,
    cfg: RunConfig,
) -> list[dict[str, Any]]:
    """Reconstruct the exact prompt prefix of every generated turn."""
    examples: list[dict[str, Any]] = []
    for traj, item, adv in zip(trajectories, items, advantages.tolist()):
        messages = build_messages(item.question)
        last_index = len(traj.turns) - 1
        for turn_index, turn in enumerate(traj.turns):
            keep = not (cfg.grpo.last_turn_only and turn_index != last_index)
            if keep:
                examples.append(
                    {
                        "prompt_text": format_prompt(processor, messages),
                        "target_text": turn.raw_text,
                        "image_path": item.image_paths[0],
                        "advantage": float(adv),
                        "task_id": item.task_id,
                        "turn_index": turn_index,
                    }
                )
            messages.append({"role": "assistant", "content": turn.raw_text})
            if turn.observation:
                messages.append({"role": "user", "content": turn.observation})
    return examples


def collate_grpo(processor: Any, device: str, max_length: int, records: list[dict[str, Any]]) -> dict[str, Any]:
    tokenizer = processor.tokenizer
    eos_id = getattr(tokenizer, "eos_token_id", None)
    pad_id = getattr(tokenizer, "pad_token_id", 0) or 0
    input_ids, labels, attn, pv, pm, advs = [], [], [], [], [], []
    for rec in records:
        enc = processor(text=[rec["prompt_text"]], images=[_image(rec["image_path"])],
                        return_tensors="pt", padding=False)
        prefix = enc["input_ids"][0].tolist()
        target = encode_text(tokenizer, rec["target_text"])
        if eos_id is not None:
            target = target + [eos_id]
        if len(prefix) + len(target) > max_length:
            keep = max(16, max_length - len(target))
            prefix = prefix[-keep:]
        ids = prefix + target
        input_ids.append(ids)
        labels.append([-100] * len(prefix) + target)
        attn.append([1] * len(ids))
        if enc.get("pixel_values") is not None:
            pv.append(enc["pixel_values"][0])
        if enc.get("pixel_attention_mask") is not None:
            pm.append(enc["pixel_attention_mask"][0])
        advs.append(rec["advantage"])
    max_len = max(len(x) for x in input_ids)
    for i in range(len(input_ids)):
        pad = max_len - len(input_ids[i])
        input_ids[i] += [pad_id] * pad
        labels[i] += [-100] * pad
        attn[i] += [0] * pad
    batch: dict[str, Any] = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attn, dtype=torch.long, device=device),
        "labels": torch.tensor(labels, dtype=torch.long, device=device),
        "advantages": torch.tensor(advs, dtype=torch.float32, device=device),
    }
    if pv:
        batch["pixel_values"] = torch.stack(pv).to(device)
    if pm:
        batch["pixel_attention_mask"] = torch.stack(pm).to(device)
    return batch


def update_policy(
    model: Any,
    ref_model: Any,
    examples: list[dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    cfg: RunConfig,
    processor: Any,
    device: str,
    micro_batch: int = 4,
) -> dict[str, Any]:
    model.eval()  # dropout off: log-probs and gradients stay consistent
    ref_model.eval()
    optimizer.zero_grad(set_to_none=True)
    chunks = [examples[i : i + micro_batch] for i in range(0, len(examples), micro_batch)]
    totals = {"loss": 0.0, "pg_loss": 0.0, "kl": 0.0, "clip_fraction": 0.0, "ratio_mean": 0.0}
    weights = 0
    entropy = 0.0
    n_tokens = 0
    for chunk in chunks:
        if not chunk:
            continue
        batch = collate_grpo(processor, device, cfg.grpo.max_length, chunk)
        with torch.no_grad():
            old = token_logprobs(model, batch)
            ref = token_logprobs(ref_model, batch)
        new = token_logprobs(model, batch, need_entropy=True)
        if new["n_tokens"] == 0:
            continue
        out = grpo_loss(
            new["logp"],
            old["logp"],
            ref["logp"],
            new["adv"],
            clip_eps=cfg.grpo.clip_eps,
            kl_beta=cfg.grpo.kl_beta,
            kl_kind=cfg.grpo.kl_kind,
        )
        (out["loss"] / len(chunks)).backward()
        w = new["n_tokens"]
        for key in totals:
            totals[key] += float(out[key].detach().cpu()) * w
        entropy += float(new["entropy"]) * w
        n_tokens += w
        weights += w
    if weights:
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg.grpo.grad_clip)
        optimizer.step()
        scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    denom = max(1, weights)
    return {
        **{k: round(v / denom, 6) for k, v in totals.items()},
        "entropy": round(entropy / denom, 6),
        "tokens": n_tokens,
        "examples": len(examples),
    }


@torch.no_grad()
def quick_validation(
    model: Any,
    processor: Any,
    cfg: RunConfig,
    tier: str,
    device: str,
    val_items: list[TaskItem],
    generator: Any,
) -> dict[str, float]:
    if not val_items:
        return {}
    rollout_cfg = replace(
        cfg.rollout,
        max_turns=cfg.grpo.max_turns,
        max_new_tokens=cfg.grpo.max_new_tokens,
        temperature=0.7,
        do_sample=True,
        batch_size=len(val_items),
    )
    vcfg = replace(cfg, rollout=rollout_cfg)
    model.eval()
    trajs = run_episodes(model, processor, val_items, vcfg, generator=generator, tier=tier, device=device)
    correct = [1.0 if verify_answer(t.pred_answer, it.gold_answer) else 0.0 for t, it in zip(trajs, val_items)]
    complete = [
        1.0 if (c > 0 and verify_tool_trace(t.tool_calls, it)) else 0.0
        for c, t, it in zip(correct, trajs, val_items)
    ]
    return {
        "val_pass@1": round(sum(correct) / len(correct), 4),
        "val_completion": round(sum(complete) / len(complete), 4),
        "val_mean_turns": round(sum(t.n_turns for t in trajs) / len(trajs), 3),
    }


def peak_memory_mb(device: str) -> float:
    try:
        if device == "cuda":
            return round(torch.cuda.max_memory_allocated() / 1e6, 1)
        if device == "mps":
            return round(torch.mps.driver_allocated_memory() / 1e6, 1)
    except Exception:
        pass
    return 0.0


def train(cfg: RunConfig, sft_checkpoint: str | Path | None = None, max_steps: int | None = None) -> dict[str, Any]:
    seed_everything(cfg.grpo.seed or cfg.seed)
    device = cfg.device if cfg.device != "auto" else detect_device()
    tier = pick_model_tier(device, override=cfg.model_tier)
    dtype = resolve_dtype(device, cfg.dtype)
    adapter = Path(sft_checkpoint) if sft_checkpoint else resolve_checkpoint(cfg.runs_dir, "sft")
    if adapter is None or not Path(adapter).exists():
        raise FileNotFoundError("SFT checkpoint not found; run `make sft` first or pass --sft-checkpoint")

    run_id = cfg.run_id
    tracker = Tracker(
        run_id=run_id,
        runs_dir=cfg.runs_dir,
        config={"phase": "grpo", "tier": tier, "device": device, "dtype": dtype_name(dtype),
                "sft_checkpoint": str(adapter), **cfg.grpo.__dict__},
        project=cfg.tracking.project,
        use_wandb=cfg.tracking.use_wandb,
    )
    t0 = time.perf_counter()
    model, processor = load_vlm(tier, device, cfg.dtype, adapter_path=adapter, trainable=True)
    ref_model, _ = load_vlm(tier, device, cfg.dtype, adapter_path=adapter)
    for param in ref_model.parameters():
        param.requires_grad_(False)
    ref_model.eval()
    trainable = count_params(model, trainable_only=True)
    print(f"[grpo] tier={tier} device={device} dtype={dtype_name(dtype)} "
          f"trainable={trainable:,} group={cfg.grpo.group_size}")

    train_items = [TaskItem.from_dict(r) for r in read_jsonl(Path(cfg.data_dir) / "train.jsonl")]
    val_items = [TaskItem.from_dict(r) for r in read_jsonl(Path(cfg.data_dir) / "val.jsonl")]
    if not train_items:
        raise FileNotFoundError("data/train.jsonl missing; run `make data`")
    val_items = val_items[: cfg.grpo.val_items] if cfg.grpo.val_items else []

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.grpo.lr, weight_decay=0.0)
    steps = max_steps if max_steps and max_steps > 0 else cfg.grpo.steps
    from transformers import get_cosine_schedule_with_warmup

    scheduler = get_cosine_schedule_with_warmup(optimizer, max(1, steps // 20), max(1, steps))
    generator = make_generator(model, processor, replace(cfg, rollout=replace(cfg.rollout, backend=cfg.grpo.rollout_backend)), tier, device)

    rollout_cfg = replace(
        cfg.rollout,
        max_turns=cfg.grpo.max_turns,
        max_new_tokens=cfg.grpo.max_new_tokens,
        temperature=cfg.grpo.temperature,
        top_p=cfg.grpo.top_p,
        do_sample=True,
        batch_size=cfg.grpo.group_size,
        backend=cfg.grpo.rollout_backend,
    )
    gcfg = replace(cfg, rollout=rollout_cfg, reward=cfg.reward)

    rng = random.Random(cfg.grpo.seed)
    order = list(range(len(train_items)))
    rng.shuffle(order)
    cursor = 0
    skipped_groups = 0
    reward_history: list[float] = []
    completion_history: list[float] = []
    rollouts_done = 0
    step_times: list[float] = []
    peak = 0.0

    for step in range(1, steps + 1):
        step_start = time.perf_counter()
        if cursor >= len(order):
            rng.shuffle(order)
            cursor = 0
        item = train_items[order[cursor]]
        cursor += 1

        model.eval()
        group_items = [TaskItem.from_dict(item.to_dict()) for _ in range(cfg.grpo.group_size)]
        trajectories = run_episodes(
            model, processor, group_items, gcfg, generator=generator, tier=tier, device=device
        )
        rollouts_done += len(trajectories)
        rewards = [reward_breakdown(t, item, cfg) for t in trajectories]
        reward_values = [r["total"] for r in rewards]
        reward_history.extend(reward_values)
        completion_history.extend(
            1.0 if (r["answer_correct"] > 0 and r["required_tool_used"] > 0) else 0.0 for r in rewards
        )
        advantages, adv_stats = compute_advantages(reward_values, skip_zero_variance=cfg.grpo.skip_zero_variance)

        update_stats: dict[str, Any] = {}
        if adv_stats["skipped"]:
            skipped_groups += 1
        else:
            examples = build_turn_examples(trajectories, group_items, advantages, processor, cfg)
            if examples:
                update_stats = update_policy(
                    model, ref_model, examples, optimizer, scheduler, cfg, processor, device,
                    micro_batch=max(1, min(2, cfg.grpo.group_size)),
                )

        step_seconds = time.perf_counter() - step_start
        if device == "mps":
            torch.mps.empty_cache()
        step_times.append(step_seconds)
        peak = max(peak, peak_memory_mb(device))
        row = {
            "phase": "grpo",
            "step": step,
            "split": "train",
            "task_id": item.task_id,
            "family": item.family,
            "reward_mean": round(sum(reward_values) / len(reward_values), 4),
            "reward_std": adv_stats["std"],
            "reward_min": adv_stats["min"],
            "reward_max": adv_stats["max"],
            "answer_correct": round(sum(r["answer_correct"] for r in rewards) / len(rewards), 4),
            "tool_schema_valid_rate": round(
                sum(r["tool_schema_valid_rate"] for r in rewards) / len(rewards), 4
            ),
            "required_tool_used": round(
                sum(r["required_tool_used"] for r in rewards) / len(rewards), 4
            ),
            "format_ok": round(sum(r["format_ok"] for r in rewards) / len(rewards), 4),
            "length_penalty": round(sum(r["length_penalty"] for r in rewards) / len(rewards), 4),
            "perception_recall": round(
                sum(r["perception_recall"] for r in rewards) / len(rewards), 4
            ),
            "train_completion": round(
                sum(1.0 for r in rewards if r["answer_correct"] > 0 and r["required_tool_used"] > 0)
                / len(rewards),
                4,
            ),
            "mean_turns": round(sum(t.n_turns for t in trajectories) / len(trajectories), 3),
            "advantage_mean": round(float(advantages.mean()), 6),
            "advantage_max_abs": round(float(advantages.abs().max()), 6),
            "group_skipped": adv_stats["skipped"],
            "skipped_group_rate": round(skipped_groups / step, 4),
            "step_seconds": round(step_seconds, 2),
            "wall_clock_sec": round(time.perf_counter() - t0, 1),
            **{k: update_stats.get(k) for k in ("loss", "pg_loss", "kl", "clip_fraction", "ratio_mean", "entropy", "tokens")},
            "lr": round(scheduler.get_last_lr()[0], 8),
            "peak_memory_mb": peak,
        }
        if cfg.grpo.val_every and step % cfg.grpo.val_every == 0 and val_items:
            row.update(quick_validation(model, processor, cfg, tier, device, val_items, generator))
        tracker.log(row, step=step)
        print(
            f"[grpo] step {step}/{steps} reward={row['reward_mean']:.3f} "
            f"ans={row['answer_correct']:.2f} trace={row['required_tool_used']:.2f} "
            f"schema={row['tool_schema_valid_rate']:.2f} kl={row.get('kl') or 0:.4f} "
            f"turns={row['mean_turns']:.2f} skip={row['group_skipped']} ({step_seconds:.0f}s)"
        )

    wall = time.perf_counter() - t0
    ckpt_dir = Path(cfg.runs_dir) / run_id / "grpo"
    model.save_pretrained(ckpt_dir)
    try:
        processor.save_pretrained(ckpt_dir)
    except Exception:
        pass
    rollouts_per_min = round(rollouts_done / max(1e-6, wall / 60.0), 2)
    summary = {
        "phase": "grpo",
        "tier": tier,
        "device": device,
        "dtype": dtype_name(dtype),
        "trainable_params": trainable,
        "steps": steps,
        "group_size": cfg.grpo.group_size,
        "skipped_groups": skipped_groups,
        "rollouts": rollouts_done,
        "rollouts_per_min": rollouts_per_min,
        "mean_step_seconds": round(sum(step_times) / max(1, len(step_times)), 2),
        "reward_first_window": round(sum(reward_history[: max(1, len(reward_history) // 3)]) /
                                     max(1, len(reward_history[: max(1, len(reward_history) // 3)])), 4),
        "reward_last_window": round(sum(reward_history[-max(1, len(reward_history) // 3):]) /
                                    max(1, len(reward_history[-max(1, len(reward_history) // 3):])), 4),
        "completion_first_window": round(sum(completion_history[: max(1, len(completion_history) // 3)]) /
                                         max(1, len(completion_history[: max(1, len(completion_history) // 3)])), 4),
        "completion_last_window": round(sum(completion_history[-max(1, len(completion_history) // 3):]) /
                                          max(1, len(completion_history[-max(1, len(completion_history) // 3):])), 4),
        "kl_beta": cfg.grpo.kl_beta,
        "clip_eps": cfg.grpo.clip_eps,
        "last_turn_only": cfg.grpo.last_turn_only,
        "peak_memory_mb": peak,
        "wall_clock_sec": round(wall, 2),
        "wall_clock_hours": human_hours(wall),
        "commit_sha": git_sha(),
        "checkpoint": str(ckpt_dir),
    }
    write_json(Path(cfg.runs_dir) / run_id / "grpo_summary.json", summary)
    tracker.log_summary(summary)
    register_checkpoint(
        cfg.runs_dir, "grpo", ckpt_dir,
        {"tier": tier, "device": device, "steps": steps, "run_id": run_id, "phase": "grpo"},
    )
    print(f"[grpo] done: steps={steps} wall={wall/60:.1f}min rollouts/min={rollouts_per_min} -> {ckpt_dir}")
    tracker.finish()
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GRPO / RLVR online RL from SFT")
    parser.add_argument("--config", default="configs/tiny.yaml")
    parser.add_argument("--sft-checkpoint", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--devices", default=None, help="comma-separated device indices; the custom loop uses the first")
    parser.add_argument("--model-tier", default=None)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.device:
        cfg.device = args.device
    if args.devices:
        first = args.devices.split(",")[0].strip()
        cfg.device = f"cuda:{first}" if first.isdigit() else first
        if "," in args.devices:
            print("[config] custom training loop runs on one device; --devices selects the first")
    if args.model_tier:
        cfg.model_tier = args.model_tier
    if args.run_id:
        cfg.run_id = args.run_id
    summary = train(cfg, sft_checkpoint=args.sft_checkpoint, max_steps=args.max_steps)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
