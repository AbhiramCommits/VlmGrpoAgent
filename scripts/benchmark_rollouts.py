#!/usr/bin/env python
"""Measure rollout throughput / peak memory for configurations actually run.

Writes ``results/benchmark.json`` and ``docs/scaling.md``.  Every number in the
scaling doc comes from this script or from the real GRPO/SFT run summaries.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlmgrpo.agent import run_episodes  # noqa: E402
from vlmgrpo.config import load_config  # noqa: E402
from vlmgrpo.evaluate import load_split  # noqa: E402
from vlmgrpo.registry import detect_device, dtype_name, load_vlm, pick_model_tier, resolve_dtype  # noqa: E402
from vlmgrpo.utils import read_json, write_json  # noqa: E402


def peak_memory_mb(device: str) -> float:
    import torch

    try:
        if device == "cuda":
            return round(torch.cuda.max_memory_allocated() / 1e6, 1)
        if device == "mps":
            return round(torch.mps.driver_allocated_memory() / 1e6, 1)
    except Exception:
        pass
    return 0.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tiny.yaml")
    parser.add_argument("--items", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-turns", type=int, default=2)
    parser.add_argument("--batches", default="1,4,8")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    device = cfg.device if cfg.device != "auto" else detect_device()
    tier = pick_model_tier(device, override=cfg.model_tier)
    dtype = resolve_dtype(device, cfg.dtype)
    model, processor = load_vlm(tier, device, cfg.dtype)
    base_items = load_split(cfg, "test", args.items)
    rows = []
    for bs in [int(x) for x in args.batches.split(",") if x.strip()]:
        items = []
        while len(items) < bs:
            items.extend(base_items)
        items = items[:bs]
        rollout_cfg = replace(
            cfg.rollout,
            max_turns=args.max_turns,
            max_new_tokens=args.max_new_tokens,
            batch_size=bs,
            do_sample=True,
            temperature=0.9,
        )
        run_cfg = replace(cfg, rollout=rollout_cfg)
        t0 = time.perf_counter()
        trajs = run_episodes(model, processor, items, run_cfg, tier=tier, device=device)
        wall = time.perf_counter() - t0
        row = {
            "batch_size": bs,
            "rollouts": len(trajs),
            "turns_total": sum(t.n_turns for t in trajs),
            "wall_seconds": round(wall, 2),
            "rollouts_per_min": round(len(trajs) / (wall / 60.0), 2),
            "seconds_per_rollout": round(wall / len(trajs), 2),
            "peak_memory_mb": peak_memory_mb(device),
        }
        print(json.dumps(row))
        rows.append(row)

    summary = {
        "device": device,
        "tier": tier,
        "dtype": dtype_name(dtype),
        "model_params": sum(p.numel() for p in model.parameters()),
        "max_new_tokens": args.max_new_tokens,
        "max_turns": args.max_turns,
        "configs": rows,
        "grpo_run": read_json(Path(cfg.runs_dir) / cfg.run_id / "grpo_summary.json")
        if (Path(cfg.runs_dir) / cfg.run_id / "grpo_summary.json").exists()
        else None,
        "sft_run": read_json(Path(cfg.runs_dir) / cfg.run_id / "sft_summary.json")
        if (Path(cfg.runs_dir) / cfg.run_id / "sft_summary.json").exists()
        else None,
    }
    write_json(Path(cfg.results_dir) / "benchmark.json", summary)
    write_scaling_doc(cfg, summary)
    print(f"[bench] wrote {cfg.results_dir}/benchmark.json and docs/scaling.md")
    return 0


def write_scaling_doc(cfg, summary: dict) -> None:
    lines = [
        "# Scaling notes (measured, not estimated)",
        "",
        f"Hardware: **{summary['device']}**, model tier **{summary['tier']}** "
        f"({summary['model_params']:,} parameters), dtype **{summary['dtype']}**.",
        "",
        "## Rollout throughput",
        "",
        f"Measured with `make bench` on the test split, max {summary['max_turns']} turns and "
        f"{summary['max_new_tokens']} new tokens per turn, greedy-free sampling (T=0.9).",
        "",
        "| batch size | rollouts | turns | wall (s) | rollouts/min | s/rollout | peak MPS/CUDA mem (MB) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["configs"]:
        lines.append(
            f"| {row['batch_size']} | {row['rollouts']} | {row['turns_total']} | {row['wall_seconds']} | "
            f"{row['rollouts_per_min']} | {row['seconds_per_rollout']} | {row['peak_memory_mb']} |"
        )
    grpo = summary.get("grpo_run")
    if grpo:
        lines += [
            "",
            "## Real GRPO training step",
            "",
            "| metric | value |",
            "|---|---:|",
            f"| steps | {grpo.get('steps')} |",
            f"| group size | {grpo.get('group_size')} |",
            f"| rollouts | {grpo.get('rollouts')} |",
            f"| rollouts/min | {grpo.get('rollouts_per_min')} |",
            f"| mean step seconds | {grpo.get('mean_step_seconds')} |",
            f"| peak memory (MB) | {grpo.get('peak_memory_mb')} |",
            f"| wall clock (min) | {round(grpo.get('wall_clock_sec', 0) / 60, 1)} |",
            f"| skipped zero-variance groups | {grpo.get('skipped_groups')} |",
        ]
    sft = summary.get("sft_run")
    if sft:
        lines += [
            "",
            "## SFT step",
            "",
            f"{sft.get('steps')} optimizer steps over {sft.get('examples')} examples "
            f"(batch {sft.get('batch_size')} x grad-accum {sft.get('grad_accum')}), "
            f"{round(sft.get('wall_clock_sec', 0) / 60, 1)} min wall clock, "
            f"final val loss {sft.get('final_val_loss')}.",
        ]
    lines += [
        "",
        "## Not measured on this machine (no CUDA)",
        "",
        "The following commands are wired but were not executed here:",
        "",
        "```bash",
        "# single node, 2-8 GPUs, DeepSpeed ZeRO-2",
        "accelerate launch --config_file configs/accel/ds_zero2.json --num_processes 4 \\",
        "  -m vlmgrpo.train_grpo --config configs/small.yaml",
        "",
        "# FSDP full shard",
        "accelerate launch --config_file configs/accel/fsdp.yaml --num_processes 4 \\",
        "  -m vlmgrpo.train_grpo --config configs/small.yaml",
        "```",
        "",
        "Rollout throughput on CUDA is expected to be dominated by generation; pass "
        "`--backend vllm` (or `rollout.backend: vllm`) to use the offline vLLM engine when it is "
        "installed. The agent records the effective backend in every evaluation JSON and falls back "
        "to HF generation with a warning when vLLM/CUDA is unavailable.",
        "",
    ]
    Path("docs").mkdir(exist_ok=True)
    Path("docs/scaling.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
