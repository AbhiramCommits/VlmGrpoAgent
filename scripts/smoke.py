#!/usr/bin/env python
"""Smoke test: render a task, run one stub rollout, print everything.

Runs fully offline on CPU in a few seconds.  Exit code 0 and ``SMOKE OK`` mean
the package, config, model registry, task suite, agent loop, tool sandbox and
processor API all agree end to end.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlmgrpo.agent import run_episode  # noqa: E402
from vlmgrpo.config import load_config  # noqa: E402
from vlmgrpo.registry import build_stub_vlm, detect_device, model_info, pick_model_tier  # noqa: E402
from vlmgrpo.tasks.families import FAMILIES, TaskItem, generate_task  # noqa: E402
from vlmgrpo.utils import count_params, seed_everything  # noqa: E402


def pick_smoke_task(cfg) -> TaskItem:
    """Use the committed test split when available, otherwise render one chart task."""
    test_path = Path(cfg.data_dir) / "test.jsonl"
    if test_path.exists():
        rows = [json.loads(line) for line in test_path.read_text().splitlines() if line.strip()]
        for row in rows:
            if row["family"] == "chart_agg":
                item = TaskItem.from_dict(row)
                if Path(item.image_paths[0]).exists():
                    return item
    return generate_task("chart_agg", 1234, "test", 0, Path(cfg.data_dir) / "smoke")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tiny.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config) if Path(args.config).exists() else load_config("configs/ci.yaml")
    seed_everything(cfg.seed)

    # Always use the stub tier for the smoke test: offline, fast, deterministic.
    model, processor = build_stub_vlm()
    device = detect_device()
    tier = pick_model_tier(device, override=cfg.model_tier)
    print(f"config={args.config} device={device} auto_tier={tier} smoke_tier=stub")
    print(f"stub_params={count_params(model):,} info={model_info('stub')['description']}")
    print(f"families={list(FAMILIES)}")

    task = pick_smoke_task(cfg)
    smoke_cfg = replace(
        cfg,
        rollout=replace(cfg.rollout, max_turns=2, max_new_tokens=24, batch_size=1),
        model_tier="stub",
    )
    traj = run_episode(model, processor, task, smoke_cfg, tier="stub", device="cpu")

    print("\n=== RENDERED TASK PROMPT ===")
    print(traj.prompt)
    print(f"\n=== TASK ===\nfamily={task.family} gold_answer={task.gold_answer} "
          f"required_tools={task.required_tools}")
    print("\n=== STUB MODEL RAW OUTPUT (turn 1) ===")
    print(traj.turns[0].raw_text if traj.turns else "<none>")
    print(f"\n=== TRAJECTORY ===\nturns={traj.n_turns} stopped={traj.stopped_reason} "
          f"pred_answer={traj.pred_answer!r} tool_calls={len(traj.tool_calls)}")
    print("\nSMOKE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
