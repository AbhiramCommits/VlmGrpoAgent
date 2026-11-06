"""Held-out evaluation for BASE / SFT / GRPO checkpoints."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from .agent import make_generator, run_episodes
from .config import RunConfig, load_config
from .registry import load_vlm, pick_model_tier
from .rewards import reward_breakdown
from .tasks.families import FAMILIES, TaskItem
from .utils import (
    count_params,
    git_sha,
    read_jsonl,
    resolve_checkpoint,
    seed_everything,
    write_json,
    write_jsonl,
)
from .verify import perception_recall, verify_answer, verify_tool_trace

METRIC_KEYS = (
    "pass@1",
    "pass@k",
    "task_completion_rate",
    "tool_call_schema_validity",
    "mean_turns",
    "perception_recall",
    "tool_trace_rate",
    "format_rate",
)


def load_split(cfg: RunConfig, split: str, max_items: int | None = None) -> list[TaskItem]:
    path = Path(cfg.data_dir) / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing split {path}; run `make data` first")
    items = [TaskItem.from_dict(row) for row in read_jsonl(path)]
    if max_items and max_items < len(items):
        # stratified: take the same number from every family, preserving seed order
        per_family = max(1, max_items // len(FAMILIES))
        picked: list[TaskItem] = []
        for family in FAMILIES:
            picked.extend([i for i in items if i.family == family][:per_family])
        items = picked[:max_items]
    return items


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _per_family_metrics(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for family in FAMILIES:
        fam = [r for r in rows if r["family"] == family]
        if not fam:
            continue
        out[family] = {
            "n_items": len({r["task_id"] for r in fam}),
            "n_rollouts": len(fam),
            "pass@1": _mean([r["correct"] for r in fam]),
            "task_completion_rate": _mean([r["completion"] for r in fam]),
            "tool_call_schema_validity": _mean([r["schema_rate"] for r in fam]),
            "mean_turns": _mean([r["turns"] for r in fam]),
            "perception_recall": _mean([r["recall"] for r in fam]),
        }
    return out


def evaluate(
    checkpoint_name: str,
    split: str,
    k: int = 4,
    cfg: RunConfig | None = None,
    max_items: int | None = None,
    checkpoint_path: str | Path | None = None,
    device: str | None = None,
    seed: int | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Evaluate one checkpoint on one split; write JSON + trajectories."""
    cfg = cfg or load_config("configs/tiny.yaml")
    device = device or cfg.device
    if device == "auto":
        from .registry import detect_device

        device = detect_device()
    adapter = Path(checkpoint_path) if checkpoint_path else resolve_checkpoint(cfg.runs_dir, checkpoint_name)
    if checkpoint_name != "base" and adapter is None:
        raise FileNotFoundError(
            f"No checkpoint registered for {checkpoint_name!r}; run the corresponding training step or pass --checkpoint-path"
        )
    from .registry import TIERS
    from .registry import dtype_name as _dtype_name
    from .registry import resolve_dtype as _resolve_dtype

    tier = pick_model_tier(device, override=cfg.model_tier)
    model_id = TIERS.get(tier, {}).get("hf_id") or "vlmgrpo-stub"
    dtype = _resolve_dtype(device, cfg.dtype)
    t_load = time.perf_counter()
    model, processor = load_vlm(tier, device, cfg.dtype, adapter_path=adapter)
    load_seconds = time.perf_counter() - t_load

    items = load_split(cfg, split, max_items if max_items is not None else cfg.eval.max_items)
    if not items:
        raise RuntimeError(f"No items for split {split}")
    sfx = seed if seed is not None else cfg.seed
    seed_everything(sfx)
    # Standard pass@k protocol: one greedy rollout per item for pass@1, plus
    # k-1 temperature samples; pass@k counts any success among all k.
    base_rollout = replace(
        cfg.rollout,
        max_turns=cfg.eval.max_turns,
        max_new_tokens=cfg.eval.max_new_tokens,
        batch_size=cfg.eval.batch_size,
    )
    greedy_cfg = replace(cfg, rollout=replace(base_rollout, do_sample=False, temperature=0.0), seed=sfx)
    sample_cfg = replace(
        cfg,
        rollout=replace(base_rollout, do_sample=True, temperature=cfg.eval.temperature, top_p=0.95),
        seed=sfx,
    )
    generator = make_generator(model, processor, sample_cfg, tier, device)

    t0 = time.perf_counter()
    greedy_trajs = run_episodes(
        model, processor, items, greedy_cfg, generator=generator, tier=tier, device=device
    )
    n_samples = max(0, int(k) - 1)
    sampled: list[Any] = []
    sampled_items: list[TaskItem] = []
    if n_samples:
        for rep in range(n_samples):
            for item in items:
                clone = TaskItem.from_dict(item.to_dict())
                clone.seed = item.seed + (rep + 1) * 1_000_003
                sampled_items.append(clone)
        sampled = run_episodes(
            model, processor, sampled_items, sample_cfg, generator=generator, tier=tier, device=device
        )
    trajectories = []
    repeated = []
    for idx, item in enumerate(items):
        trajectories.append(greedy_trajs[idx])
        repeated.append(item)
        for rep in range(n_samples):
            trajectories.append(sampled[rep * len(items) + idx])
            repeated.append(sampled_items[rep * len(items) + idx])
    wall = time.perf_counter() - t0

    rows: list[dict[str, Any]] = []
    by_task: dict[str, list[dict[str, Any]]] = {}
    all_calls = 0
    all_valid = 0
    for traj, item in zip(trajectories, repeated):
        correct = 1.0 if verify_answer(traj.pred_answer, item.gold_answer) else 0.0
        trace_ok = 1.0 if verify_tool_trace(traj.tool_calls, item) else 0.0
        completion = 1.0 if (correct and trace_ok) else 0.0
        calls = traj.tool_calls
        schema_rate = (
            sum(1 for c in calls if c.get("schema_valid") or c.get("ok")) / len(calls) if calls else 0.0
        )
        all_calls += len(calls)
        all_valid += sum(1 for c in calls if c.get("schema_valid") or c.get("ok"))
        row = {
            "task_id": traj.task_id,
            "family": traj.family,
            "sample": len(by_task.get(traj.task_id, [])),
            "correct": correct,
            "completion": completion,
            "trace_ok": trace_ok,
            "schema_rate": round(schema_rate, 4),
            "turns": traj.n_turns,
            "recall": perception_recall(traj.perception_text(), item.gold_perception_facts),
            "format_ok": 1.0 if traj.format_valid else 0.0,
            "pred_answer": traj.pred_answer,
            "gold_answer": item.gold_answer,
            "stopped_reason": traj.stopped_reason,
            "reward": reward_breakdown(traj, item, cfg)["total"],
        }
        rows.append(row)
        by_task.setdefault(traj.task_id, []).append(row)

    pass_at_k = _mean(
        [1.0 if any(r["correct"] > 0 for r in group) else 0.0 for group in by_task.values()]
    )
    metrics = {
        "checkpoint": checkpoint_name,
        "checkpoint_path": str(adapter) if adapter else None,
        "split": split,
        "k": k,
        "n_items": len({r["task_id"] for r in rows}),
        "n_rollouts": len(rows),
        "pass@1": _mean([r["correct"] for r in rows]),
        "pass@k": pass_at_k if k > 1 else _mean([r["correct"] for r in rows]),
        "task_completion_rate": _mean([r["completion"] for r in rows]),
        "tool_call_schema_validity": round(all_valid / all_calls, 4) if all_calls else 0.0,
        "tool_call_schema_validity_rollout_mean": _mean([r["schema_rate"] for r in rows]),
        "mean_turns": _mean([float(r["turns"]) for r in rows]),
        "perception_recall": _mean([r["recall"] for r in rows]),
        "tool_trace_rate": _mean([r["trace_ok"] for r in rows]),
        "format_rate": _mean([r["format_ok"] for r in rows]),
        "mean_reward": _mean([r["reward"] for r in rows]),
        "per_family": _per_family_metrics(rows),
        "environment": {
            "device": device,
            "model_tier": tier,
            "model_id": model_id,
            "dtype": _dtype_name(dtype),
            "backend": getattr(generator, "name", "hf"),
            "rollout_backend_requested": cfg.rollout.backend,
            "commit_sha": git_sha(),
            "model_params": count_params(model),
            "load_seconds": round(load_seconds, 2),
            "eval_wall_clock_sec": round(wall, 2),
            "eval_wall_clock_hours": round(wall / 3600.0, 4),
            "timestamp": int(time.time()),
        },
    }

    out_dir = Path(cfg.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"eval_{checkpoint_name}_{split}.json"
    write_json(json_path, metrics)
    traj_path = out_dir / "trajectories" / f"{checkpoint_name}_{split}.jsonl"
    write_jsonl(traj_path, [t.to_dict() for t in trajectories])

    if verbose:
        print(
            f"[eval] {checkpoint_name} {split}: pass@1={metrics['pass@1']:.3f} "
            f"pass@{k}={metrics['pass@k']:.3f} completion={metrics['task_completion_rate']:.3f} "
            f"schema={metrics['tool_call_schema_validity']:.3f} turns={metrics['mean_turns']:.2f} "
            f"recall={metrics['perception_recall']:.3f} format={metrics['format_rate']:.3f} "
            f"({wall:.0f}s) -> {json_path}"
        )
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on the VLM-GRPO suite")
    parser.add_argument("--config", default="configs/tiny.yaml")
    parser.add_argument("--checkpoint", default="base", help="base | sft | grpo | registered name")
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--split", default=None, help="one split (default: all three eval splits)")
    parser.add_argument("--splits", default="val,test,test_hard")
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="alias for --max-items")
    parser.add_argument("--device", default=None)
    parser.add_argument("--model-tier", default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.model_tier:
        cfg.model_tier = args.model_tier
    if args.device:
        cfg.device = args.device
    splits = [args.split] if args.split else [s.strip() for s in args.splits.split(",") if s.strip()]
    k = args.k if args.k is not None else cfg.eval.k
    max_items = args.max_items if args.max_items is not None else args.limit
    results = []
    for split in splits:
        metrics = evaluate(
            checkpoint_name=args.checkpoint,
            split=split,
            k=k,
            cfg=cfg,
            max_items=max_items,
            checkpoint_path=args.checkpoint_path,
            device=args.device,
            seed=args.seed,
        )
        results.append(metrics)
    print(json.dumps([{k: r[k] for k in ("checkpoint", "split", "pass@1", "pass@k")} for r in results], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
