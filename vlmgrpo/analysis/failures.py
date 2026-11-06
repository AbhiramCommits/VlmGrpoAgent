"""Per-step trajectory failure attribution.

Every failed episode is assigned exactly one bucket:

PERCEPTION   ground-truth visual facts never surfaced (recall < 0.5)
PLANNING     facts surfaced but the required tool sequence is missing/wrong order
TOOL_SCHEMA  a tool call was malformed or rejected by the schema check
EXECUTION    a required tool call errored
ARITHMETIC   right facts + right tool chain, wrong final answer
FORMAT       no parseable <answer> and not even one valid tool call
BUDGET       ran out of turns while making valid tool progress
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ..agent import Trajectory
from ..tasks.families import TaskItem
from ..utils import read_json, read_jsonl, write_json
from ..verify import perception_recall, verify_answer, verify_tool_trace

BUCKETS = ("PERCEPTION", "PLANNING", "TOOL_SCHEMA", "EXECUTION", "ARITHMETIC", "FORMAT", "BUDGET")
PERCEPTION_THRESHOLD = 0.5


def attribute_failure(traj: Trajectory, item: TaskItem) -> str | None:
    """Return one bucket for a failed episode, or ``None`` if it succeeded."""
    correct = verify_answer(traj.pred_answer, item.gold_answer)
    trace_ok = verify_tool_trace(traj.tool_calls, item)
    if correct and trace_ok:
        return None
    calls = traj.tool_calls
    if traj.pred_answer is None or not traj.format_valid:
        if any(c.get("ok") for c in calls):
            return "BUDGET"
        return "FORMAT"
    recall = perception_recall(traj.perception_text(), item.gold_perception_facts)
    if recall < PERCEPTION_THRESHOLD:
        return "PERCEPTION"
    if not trace_ok:
        if any(c.get("schema_valid") is False for c in calls):
            return "TOOL_SCHEMA"
        if any(c.get("ok") is False for c in calls):
            return "EXECUTION"
        return "PLANNING"
    return "ARITHMETIC"


def _items_for_split(data_dir: str | Path, split: str) -> dict[str, TaskItem]:
    rows = read_jsonl(Path(data_dir) / f"{split}.jsonl")
    return {r["task_id"]: TaskItem.from_dict(r) for r in rows}


def analyze_checkpoint(
    checkpoint: str,
    split: str,
    results_dir: str | Path = "results",
    data_dir: str | Path = "data",
) -> list[dict[str, Any]]:
    traj_path = Path(results_dir) / "trajectories" / f"{checkpoint}_{split}.jsonl"
    if not traj_path.exists():
        return []
    items = _items_for_split(data_dir, split)
    rows = []
    for record in read_jsonl(traj_path):
        item = items.get(record["task_id"])
        if item is None:
            continue
        traj = Trajectory.from_dict(record)
        bucket = attribute_failure(traj, item)
        rows.append(
            {
                "checkpoint": checkpoint,
                "split": split,
                "task_id": traj.task_id,
                "family": traj.family,
                "bucket": bucket or "SUCCESS",
                "turns": traj.n_turns,
                "pred_answer": traj.pred_answer,
                "gold_answer": item.gold_answer,
                "perception_recall": perception_recall(traj.perception_text(), item.gold_perception_facts),
                "n_tool_calls": len(traj.tool_calls),
            }
        )
    return rows


def failure_mix(
    checkpoints: tuple[str, ...] = ("base", "sft", "grpo"),
    splits: tuple[str, ...] = ("test", "test_hard"),
    results_dir: str | Path = "results",
    data_dir: str | Path = "data",
) -> dict[str, Any]:
    """Counts, rates and dominant bucket per checkpoint/split."""
    output: dict[str, Any] = {}
    for split in splits:
        output[split] = {}
        for ckpt in checkpoints:
            rows = analyze_checkpoint(ckpt, split, results_dir, data_dir)
            if not rows:
                continue
            counts = Counter(r["bucket"] for r in rows)
            failed = sum(v for k, v in counts.items() if k != "SUCCESS")
            total = len(rows)
            mix = {b: counts.get(b, 0) for b in BUCKETS}
            rates = {b: round(counts.get(b, 0) / total, 4) for b in BUCKETS}
            dominant = max(mix, key=mix.get) if failed else None
            per_family: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
            for r in rows:
                per_family[r["family"]][r["bucket"]] += 1
            output[split][ckpt] = {
                "n_rollouts": total,
                "n_failed": failed,
                "failure_rate": round(failed / total, 4) if total else 0.0,
                "counts": mix,
                "rates": rates,
                "dominant_bucket": dominant,
                "success_count": counts.get("SUCCESS", 0),
                "per_family": {fam: dict(v) for fam, v in per_family.items()},
            }
    return output


def select_examples(
    results_dir: str | Path = "results", data_dir: str | Path = "data"
) -> list[dict[str, Any]]:
    """Pick one success, one PERCEPTION failure and one PLANNING failure (GRPO test)."""
    items = _items_for_split(data_dir, "test")
    wanted = {"SUCCESS": None, "PERCEPTION": None, "PLANNING": None}
    path = Path(results_dir) / "trajectories" / "grpo_test.jsonl"
    if path.exists():
        for record in read_jsonl(path):
            item = items.get(record["task_id"])
            if item is None:
                continue
            traj = Trajectory.from_dict(record)
            bucket = attribute_failure(traj, item)
            bucket = bucket or "SUCCESS"
            if bucket in wanted and wanted[bucket] is None:
                wanted[bucket] = {
                    "bucket": bucket,
                    "task_id": traj.task_id,
                    "family": traj.family,
                    "question": item.question,
                    "gold_answer": item.gold_answer,
                    "pred_answer": traj.pred_answer,
                    "turns": [t.to_dict() for t in traj.turns],
                    "annotation": _annotate(bucket, traj, item),
                }
            if all(v is not None for v in wanted.values()):
                break
    return [v for v in wanted.values() if v is not None]


def _annotate(bucket: str, traj: Trajectory, item: TaskItem) -> str:
    if bucket == "SUCCESS":
        return (
            f"Success: parsed answer {traj.pred_answer!r} matches gold {item.gold_answer!r} and the "
            f"required tool sequence {item.required_tools} was executed."
        )
    if bucket == "PERCEPTION":
        return (
            "Perception failure: the trajectory never surfaced the visual facts "
            f"{list(item.gold_perception_facts)[:4]}; no downstream arithmetic could be right."
        )
    if bucket == "PLANNING":
        return (
            f"Planning failure: facts were surfaced but the required tool sequence "
            f"{item.required_tools} was not executed in a feasible order."
        )
    if bucket == "ARITHMETIC":
        return (
            f"Arithmetic failure: facts and tool chain look right, but {traj.pred_answer!r} != "
            f"{item.gold_answer!r}."
        )
    if bucket == "FORMAT":
        return "Format failure: no parseable <answer> and no valid tool call."
    if bucket == "BUDGET":
        return "Budget failure: the episode hit the turn limit while still calling tools."
    return f"Tool failure: {bucket}."


def save_failure_analysis(results_dir: str | Path = "results", data_dir: str | Path = "data") -> dict[str, Any]:
    results_dir = Path(results_dir)
    mix = failure_mix(results_dir=results_dir, data_dir=data_dir)
    write_json(results_dir / "failure_mix.json", mix)
    examples = select_examples(results_dir, data_dir)
    write_json(results_dir / "examples" / "examples.json", examples)
    for example in examples:
        lines = [
            f"# {example['bucket']} example",
            "",
            example["annotation"],
            "",
            f"- task: `{example['task_id']}` ({example['family']})",
            f"- question: {example['question']}",
            f"- gold answer: `{example['gold_answer']}`",
            f"- predicted answer: `{example['pred_answer']}`",
            "",
            "## Trajectory",
            "",
        ]
        for turn in example["turns"]:
            lines.append(f"### turn {turn['turn']} — {turn['action_type']}")
            lines.append("")
            lines.append("```text")
            lines.append(turn["raw_text"][:1200])
            lines.append("```")
            if turn.get("tool_name"):
                lines.append(
                    f"tool `{turn['tool_name']}` args=`{json.dumps(turn.get('tool_args') or {})}` "
                    f"ok={turn.get('tool_ok')} stdout=`{str(turn.get('tool_stdout', ''))[:300]}`"
                )
            lines.append("")
        path = results_dir / "examples" / f"{example['bucket'].lower()}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines), encoding="utf-8")
    return mix


def improvement_summary(mix: dict[str, Any], split: str = "test") -> str:
    """Human-readable note on which bucket GRPO improved."""
    if split not in mix or "base" not in mix[split] or "grpo" not in mix[split]:
        return "Failure attribution unavailable."
    base = mix[split]["base"]
    grpo = mix[split]["grpo"]
    lines = []
    for bucket in BUCKETS:
        delta = grpo["counts"][bucket] - base["counts"][bucket]
        if delta:
            lines.append(f"{bucket} {base['counts'][bucket]} -> {grpo['counts'][bucket]} ({delta:+d})")
    dominant = grpo["dominant_bucket"]
    return (
        f"Dominant bucket moved from {base['dominant_bucket']} (base) to {dominant} (GRPO). "
        + "; ".join(lines)
    )


__all__ = [
    "BUCKETS",
    "attribute_failure",
    "analyze_checkpoint",
    "failure_mix",
    "save_failure_analysis",
    "select_examples",
    "improvement_summary",
]


def load_metrics(path: str | Path) -> list[dict[str, Any]]:
    return read_jsonl(path)


def load_json(path: str | Path) -> Any:
    return read_json(path)
