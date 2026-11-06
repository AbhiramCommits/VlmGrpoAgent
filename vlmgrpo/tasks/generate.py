"""Deterministic task-suite generation.

Usage
-----
# everything (four splits + images + data/suite_stats.md)
python -m vlmgrpo.tasks.generate --config configs/tiny.yaml

# one split with an explicit per-family count
python -m vlmgrpo.tasks.generate --split train --n 400 --out data/train.jsonl
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..config import load_config
from ..utils import write_jsonl
from .families import FAMILIES, TaskItem, generate_task

SPLIT_STYLE = {"train": "clean", "val": "clean", "test": "clean", "test_hard": "hard"}
SPLIT_OFFSET = {"train": 0, "val": 100_000, "test": 200_000, "test_hard": 300_000}
DEFAULT_SPLIT_N = {"train": 160, "val": 40, "test": 40, "test_hard": 40}


def generate_suite(
    n_per_family: int,
    seed: int,
    split: str,
    image_dir: str | Path = "data/images",
    difficulty: int | None = None,
) -> list[TaskItem]:
    """Generate ``n_per_family`` items per family for one split, deterministically."""
    if split not in SPLIT_OFFSET:
        raise ValueError(f"Unknown split {split!r}")
    out_dir = Path(image_dir) / split
    out_dir.mkdir(parents=True, exist_ok=True)
    items: list[TaskItem] = []
    for fi, family in enumerate(FAMILIES):
        for i in range(n_per_family):
            item_seed = seed + SPLIT_OFFSET[split] + fi * 10_000 + i * 7
            item = generate_task(family, item_seed, split, i, out_dir)
            if difficulty is not None:
                item.difficulty = difficulty
            items.append(item)
    return items


def suite_stats(items: list[TaskItem]) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for split in sorted({i.split for i in items}):
        per_family: dict[str, Any] = {}
        for family in FAMILIES:
            fam = [i for i in items if i.split == split and i.family == family]
            if not fam:
                continue
            per_family[family] = {
                "count": len(fam),
                "difficulty_mean": round(sum(i.difficulty for i in fam) / len(fam), 2),
                "avg_tool_calls": round(sum(len(i.gold_tool_calls) for i in fam) / len(fam), 2),
                "images": sum(len(i.image_paths) for i in fam),
            }
        stats[split] = per_family
    return stats


def stats_markdown(stats: dict[str, Any], seed: int) -> str:
    lines = [
        "# Task suite statistics",
        "",
        f"Generated deterministically with base seed `{seed}`. "
        "Every item needs pixel reading plus at least one tool call; "
        "`test_hard` uses a shifted render style, extra distractors and longer tool chains.",
        "",
        "| split | family | count | mean difficulty | avg gold tool calls | images |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for split, fams in stats.items():
        for family, row in fams.items():
            lines.append(
                f"| {split} | {family} | {row['count']} | {row['difficulty_mean']:.2f} | "
                f"{row['avg_tool_calls']:.2f} | {row['images']} |"
            )
    totals = defaultdict(int)
    for fams in stats.values():
        for row in fams.values():
            totals["count"] += row["count"]
            totals["images"] += row["images"]
    lines += ["", f"**Total items:** {totals['count']}  |  **Total images:** {totals['images']}", ""]
    return "\n".join(lines)


def generate_all(
    seed: int = 1234,
    counts: dict[str, int] | None = None,
    n_per_family: int | None = None,
    image_dir: str | Path = "data/images",
    data_dir: str | Path = "data",
) -> dict[str, list[TaskItem]]:
    """Generate the four splits, write JSONL + images, and the stats table."""
    counts = dict(counts or DEFAULT_SPLIT_N)
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    all_splits: dict[str, list[TaskItem]] = {}
    everything: list[TaskItem] = []
    for split in SPLIT_OFFSET:
        n = n_per_family if n_per_family is not None else max(1, counts.get(split, 40) // len(FAMILIES))
        items = generate_suite(n, seed, split, image_dir=image_dir)
        write_jsonl(data_dir / f"{split}.jsonl", [i.to_dict() for i in items])
        all_splits[split] = items
        everything.extend(items)
        print(f"[generate] {split}: {len(items)} items ({n}/family) -> {data_dir / f'{split}.jsonl'}")
    stats = suite_stats(everything)
    (data_dir / "suite_stats.md").write_text(stats_markdown(stats, seed), encoding="utf-8")
    print(f"[generate] stats -> {data_dir / 'suite_stats.md'}")
    return all_splits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the deterministic VLM-GRPO task suite")
    parser.add_argument("--config", default=None, help="YAML config to take counts/seed from")
    parser.add_argument("--split", default="all", choices=["all", *SPLIT_OFFSET])
    parser.add_argument("--n", type=int, default=None, help="items per family (overrides config)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", default=None, help="output JSONL (single-split mode)")
    parser.add_argument("--image-dir", default=None)
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args(argv)

    seed = 1234
    counts = dict(DEFAULT_SPLIT_N)
    image_dir = "data/images"
    data_dir = "data"
    if args.config:
        cfg = load_config(args.config)
        seed = cfg.seed
        counts = {
            "train": cfg.task.train_n,
            "val": cfg.task.val_n,
            "test": cfg.task.test_n,
            "test_hard": cfg.task.test_hard_n,
        }
        image_dir = cfg.task.image_dir
        data_dir = cfg.data_dir
    if args.seed is not None:
        seed = args.seed
    if args.image_dir:
        image_dir = args.image_dir
    if args.data_dir:
        data_dir = args.data_dir

    if args.split == "all":
        generate_all(seed=seed, counts=counts, n_per_family=args.n, image_dir=image_dir, data_dir=data_dir)
        return 0

    n = args.n if args.n is not None else max(1, counts[args.split] // len(FAMILIES))
    items = generate_suite(n, seed, args.split, image_dir=image_dir)
    out = Path(args.out) if args.out else Path(data_dir) / f"{args.split}.jsonl"
    write_jsonl(out, [i.to_dict() for i in items])
    print(f"[generate] {args.split}: {len(items)} items ({n}/family) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
