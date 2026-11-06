"""Run the full {BASE, SFT, GRPO} x {test, test_hard} evaluation matrix.

Writes the single canonical ``results/results.json`` with the environment block,
per-checkpoint/per-split metrics, per-family breakdowns, absolute and relative
deltas vs BASE and vs SFT, the failure mix, and the cost line.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from ..config import RunConfig, load_config
from ..evaluate import evaluate
from ..utils import git_sha, read_json, write_json
from .failures import save_failure_analysis
from .plots import make_plots

CHECKPOINTS = (("base", "BASE"), ("sft", "SFT"), ("grpo", "GRPO"))
DELTA_METRICS = (
    "pass@1",
    "pass@k",
    "task_completion_rate",
    "tool_call_schema_validity",
    "mean_turns",
    "perception_recall",
)


def _eval_path(cfg: RunConfig, checkpoint: str, split: str) -> Path:
    return Path(cfg.results_dir) / f"eval_{checkpoint}_{split}.json"


def run_matrix(
    cfg: RunConfig,
    splits: tuple[str, ...] = ("test", "test_hard"),
    k: int | None = None,
    max_items: int | None = None,
    force: bool = False,
) -> dict[str, dict[str, Any]]:
    """Evaluate every checkpoint on every split (reuse existing JSONs unless ``force``)."""
    matrix: dict[str, dict[str, Any]] = {}
    for checkpoint, label in CHECKPOINTS:
        matrix[label] = {}
        for split in splits:
            path = _eval_path(cfg, checkpoint, split)
            if path.exists() and not force:
                metrics = read_json(path)
                print(f"[compare] reuse {path}")
            else:
                metrics = evaluate(
                    checkpoint_name=checkpoint,
                    split=split,
                    k=k if k is not None else cfg.eval.k,
                    cfg=cfg,
                    max_items=max_items,
                    verbose=True,
                )
            matrix[label][split] = metrics
    return matrix


def deltas(matrix: dict[str, dict[str, Any]], split: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for _, label in CHECKPOINTS:
        if split not in matrix.get(label, {}):
            continue
        current = matrix[label][split]
        out[label] = {}
        for ref_label in ("BASE", "SFT"):
            if label == ref_label:
                continue
            if ref_label not in matrix or split not in matrix[ref_label]:
                continue
            ref = matrix[ref_label][split]
            abs_d: dict[str, float] = {}
            rel_d: dict[str, float | None] = {}
            for metric in DELTA_METRICS:
                a = float(current.get(metric, 0.0) or 0.0)
                b = float(ref.get(metric, 0.0) or 0.0)
                abs_d[metric] = round(a - b, 4)
                rel_d[metric] = round((a - b) / b, 4) if b else None
            out[label][f"vs_{ref_label}"] = {"absolute": abs_d, "relative": rel_d}
    return out


def _training_summary(cfg: RunConfig, phase: str) -> dict[str, Any]:
    path = Path(cfg.runs_dir) / cfg.run_id / f"{phase}_summary.json"
    return read_json(path) if path.exists() else {}


def build_results(cfg: RunConfig, matrix: dict[str, dict[str, Any]], splits: tuple[str, ...]) -> dict[str, Any]:
    sft = _training_summary(cfg, "sft")
    grpo = _training_summary(cfg, "grpo")
    eval_hours = sum(
        float(m.get("environment", {}).get("eval_wall_clock_hours", 0.0))
        for per_split in matrix.values()
        for m in per_split.values()
    )
    train_hours = float(sft.get("wall_clock_hours", 0.0)) + float(grpo.get("wall_clock_hours", 0.0))
    device = next(iter(matrix.values()))[splits[0]]["environment"]["device"] if matrix else "unknown"
    tier = next(iter(matrix.values()))[splits[0]]["environment"]["model_tier"] if matrix else "unknown"
    dtype = next(iter(matrix.values()))[splits[0]]["environment"]["dtype"] if matrix else "unknown"
    failure_mix = save_failure_analysis(cfg.results_dir, cfg.data_dir)

    results: dict[str, Any] = {
        "environment": {
            "device": device,
            "model_tier": tier,
            "dtype": dtype,
            "commit_sha": git_sha(),
            "model_id": next(iter(matrix.values()))[splits[0]]["environment"].get("model_id"),
            "model_params": next(iter(matrix.values()))[splits[0]]["environment"].get("model_params"),
            "eval_wall_clock_hours": round(eval_hours, 4),
            "train_wall_clock_hours": round(train_hours, 4),
            "total_wall_clock_hours": round(eval_hours + train_hours, 4),
            "total_gpu_hours": round(eval_hours + train_hours, 4) if device == "cuda" else 0.0,
            "total_cpu_hours": 0.0 if device == "cuda" else round(eval_hours + train_hours, 4),
            "generated_at": int(time.time()),
        },
        "config": {
            "seed": cfg.seed,
            "eval_k": cfg.eval.k,
            "eval_max_items": cfg.eval.max_items,
            "max_turns": cfg.rollout.max_turns,
            "reward_weights": cfg.reward.__dict__,
            "sft": cfg.sft.__dict__,
            "grpo": cfg.grpo.__dict__,
        },
        "training": {"sft": sft, "grpo": grpo},
        "metrics": {
            label: {split: matrix[label][split] for split in splits if split in matrix.get(label, {})}
            for _, label in CHECKPOINTS
        },
        "per_family": {
            label: {
                split: matrix[label][split].get("per_family", {})
                for split in splits
                if split in matrix.get(label, {})
            }
            for _, label in CHECKPOINTS
        },
        "deltas": {split: deltas(matrix, split) for split in splits},
        "failure_mix": failure_mix,
    }
    # headline: which bucket GRPO improved
    try:
        base_counts = failure_mix["test"]["base"]["counts"]
        grpo_counts = failure_mix["test"]["grpo"]["counts"]
        results["failure_mix_summary"] = {
            "base_dominant": failure_mix["test"]["base"]["dominant_bucket"],
            "grpo_dominant": failure_mix["test"]["grpo"]["dominant_bucket"],
            "count_changes": {b: grpo_counts[b] - base_counts[b] for b in base_counts},
        }
    except KeyError:
        results["failure_mix_summary"] = {}
    write_json(Path(cfg.results_dir) / "results.json", results)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Full BASE/SFT/GRPO evaluation matrix")
    parser.add_argument("--config", default="configs/tiny.yaml")
    parser.add_argument("--splits", default="test,test_hard")
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="re-run evals even if JSON exists")
    parser.add_argument("--device", default=None)
    parser.add_argument("--model-tier", default=None)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.device:
        cfg.device = args.device
    if args.model_tier:
        cfg.model_tier = args.model_tier
    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    matrix = run_matrix(cfg, splits=splits, k=args.k, max_items=args.max_items, force=args.force)
    results = build_results(cfg, matrix, splits)
    figures = make_plots(cfg)
    print(json.dumps({
        "environment": results["environment"],
        "test": {label: {k: results["metrics"][label]["test"][k] for k in
                         ("pass@1", "pass@k", "task_completion_rate", "tool_call_schema_validity")}
                 for _, label in CHECKPOINTS if "test" in results["metrics"].get(label, {})},
        "figures": figures,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
