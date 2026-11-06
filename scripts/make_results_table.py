#!/usr/bin/env python
"""Regenerate results/RESULTS.md and the README results block from results.json.

No number in the README is ever hand-typed: this script is the only writer.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlmgrpo.config import load_config  # noqa: E402
from vlmgrpo.utils import read_json  # noqa: E402

CHECKPOINTS = ("BASE", "SFT", "GRPO")
MAIN_METRICS = ("pass@1", "pass@k", "task_completion_rate", "tool_call_schema_validity", "mean_turns", "perception_recall")
MAIN_TITLES = {
    "pass@1": "pass@1",
    "pass@k": "pass@k",
    "task_completion_rate": "task completion",
    "tool_call_schema_validity": "tool schema valid",
    "mean_turns": "mean turns",
    "perception_recall": "perception recall",
}
BUCKETS = ("PERCEPTION", "PLANNING", "TOOL_SCHEMA", "EXECUTION", "ARITHMETIC", "FORMAT", "BUDGET")


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def pct(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{100 * float(value):+.1f}%"


def environment_line(results: dict[str, Any]) -> str:
    env = results["environment"]
    cfg = results["config"]
    training = results.get("training", {})
    sft = training.get("sft", {})
    grpo = training.get("grpo", {})
    hours = env.get("total_gpu_hours") or env.get("total_cpu_hours") or env.get("total_wall_clock_hours")
    return (
        f"model tier **{env['model_tier']}** ({env.get('model_params', 0):,} params), "
        f"device **{env['device']}**, dtype **{env['dtype']}**, "
        f"seed **{cfg['seed']}**, SFT steps **{sft.get('steps', 'n/a')}**, "
        f"GRPO steps **{grpo.get('steps', 'n/a')}**, "
        f"train wall **{(env.get('train_wall_clock_hours') or 0):.2f} h**, "
        f"eval wall **{(env.get('eval_wall_clock_hours') or 0):.2f} h**, "
        f"total **{(hours or 0):.2f} device-hours**."
    )


def main_table(results: dict[str, Any]) -> str:
    k = results.get("config", {}).get("eval_k", 4)
    lines = [
        f"| split | checkpoint | pass@1 | pass@{k} | task completion | tool schema valid | mean turns | perception recall |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("test", "test_hard"):
        for label in CHECKPOINTS:
            m = results["metrics"].get(label, {}).get(split)
            if not m:
                continue
            lines.append(
                f"| {split} | {label} | {fmt(m['pass@1'])} | {fmt(m['pass@k'])} | "
                f"{fmt(m['task_completion_rate'])} | {fmt(m['tool_call_schema_validity'])} | "
                f"{fmt(m['mean_turns'], 2)} | {fmt(m['perception_recall'])} |"
            )
    return "\n".join(lines)


def delta_table(results: dict[str, Any]) -> str:
    lines = [
        "Absolute deltas (relative in parentheses). `vs BASE` shows the full effect of post-training; "
        "`vs SFT` isolates the RLVR step.",
        "",
        "| split | checkpoint | reference | pass@1 | completion | schema valid | perception recall |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for split in ("test", "test_hard"):
        for label in CHECKPOINTS:
            d = results.get("deltas", {}).get(split, {}).get(label, {})
            for ref in ("BASE", "SFT"):
                key = f"vs_{ref}"
                if key not in d:
                    continue
                absd, reld = d[key]["absolute"], d[key]["relative"]
                def cell(metric, absd=absd, reld=reld):
                    a = absd.get(metric, 0.0)
                    r = reld.get(metric)
                    return f"{a:+.3f} ({pct(r)})"
                lines.append(
                    f"| {split} | {label} | {ref} | {cell('pass@1')} | {cell('task_completion_rate')} | "
                    f"{cell('tool_call_schema_validity')} | {cell('perception_recall')} |"
                )
    return "\n".join(lines)


def per_family_table(results: dict[str, Any], split: str = "test") -> str:
    families = list(
        dict.fromkeys(
            fam
            for label in CHECKPOINTS
            for fam in results.get("per_family", {}).get(label, {}).get(split, {})
        )
    )
    lines = [
        "| family | checkpoint | pass@1 | task completion | tool schema valid | perception recall |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for fam in families:
        for label in CHECKPOINTS:
            row = results.get("per_family", {}).get(label, {}).get(split, {}).get(fam)
            if not row:
                continue
            lines.append(
                f"| {fam} | {label} | {fmt(row.get('pass@1'))} | {fmt(row.get('task_completion_rate'))} | "
                f"{fmt(row.get('tool_call_schema_validity'))} | {fmt(row.get('perception_recall'))} |"
            )
    return "\n".join(lines)


def failure_table(results: dict[str, Any], split: str = "test") -> str:
    mix = results.get("failure_mix", {}).get(split, {})
    lines = [
        "| checkpoint | " + " | ".join(b.lower() for b in BUCKETS) + " | success | dominant |",
        "|---|" + "---:|" * len(BUCKETS) + "---:|---|",
    ]
    for label in CHECKPOINTS:
        key = label.lower()
        if key not in mix:
            continue
        counts = mix[key]["counts"]
        lines.append(
            f"| {label} | " + " | ".join(str(counts.get(b, 0)) for b in BUCKETS)
            + f" | {mix[key]['success_count']} | {mix[key]['dominant_bucket']} |"
        )
    return "\n".join(lines)


def training_table(results: dict[str, Any]) -> str:
    sft = results.get("training", {}).get("sft", {})
    grpo = results.get("training", {}).get("grpo", {})
    lines = [
        "| stage | steps | wall clock | signal |",
        "|---|---:|---:|---|",
        f"| SFT (LoRA) | {sft.get('steps', 'n/a')} | {fmt(sft.get('wall_clock_sec', 0) / 60, 1)} min | "
        f"final val loss {fmt(sft.get('final_val_loss'), 4)} |",
        f"| GRPO | {grpo.get('steps', 'n/a')} | {fmt(grpo.get('wall_clock_sec', 0) / 60, 1)} min | "
        f"reward {fmt(grpo.get('reward_first_window'), 3)} -> {fmt(grpo.get('reward_last_window'), 3)}, "
        f"completion {fmt(grpo.get('completion_first_window'), 3)} -> {fmt(grpo.get('completion_last_window'), 3)}, "
        f"skipped groups {grpo.get('skipped_groups', 'n/a')} |",
    ]
    if grpo:
        lines.append("")
        lines.append(
            f"GRPO throughput measured during this run: **{fmt(grpo.get('rollouts_per_min'), 1)} rollouts/min**, "
            f"mean step **{fmt(grpo.get('mean_step_seconds'), 1)} s**, peak memory "
            f"**{fmt(grpo.get('peak_memory_mb'), 0)} MB**, KL beta {grpo.get('kl_beta')}, clip eps {grpo.get('clip_eps')}."
        )
    return "\n".join(lines)


def findings(results: dict[str, Any]) -> str:
    test = results["metrics"]
    base = test.get("BASE", {}).get("test", {})
    sft = test.get("SFT", {}).get("test", {})
    grpo = test.get("GRPO", {}).get("test", {})
    hard_grpo = test.get("GRPO", {}).get("test_hard", {})
    lines = []
    def delta(a, b, key):
        if not a or not b:
            return 0.0
        return float(a.get(key, 0) or 0) - float(b.get(key, 0) or 0)
    d_sft = delta(sft, base, "pass@1")
    d_grpo = delta(grpo, sft, "pass@1")
    d_comp = delta(grpo, sft, "task_completion_rate")
    d_schema = delta(grpo, sft, "tool_call_schema_validity")
    d_recall = delta(grpo, sft, "perception_recall")
    if d_grpo > 0.001:
        lines.append(f"- **GRPO improved pass@1** by {d_grpo:+.3f} over SFT on test.")
    elif d_grpo < -0.001:
        lines.append(f"- **GRPO reduced pass@1** by {d_grpo:+.3f} vs SFT on test.")
    else:
        lines.append("- **GRPO did not change pass@1** vs SFT on test.")
    if d_comp > 0.001:
        lines.append(f"- Multi-step task completion improved by {d_comp:+.3f} vs SFT.")
    elif d_comp < -0.001:
        lines.append(f"- Multi-step task completion fell by {d_comp:+.3f} vs SFT.")
    else:
        lines.append("- Multi-step task completion was unchanged vs SFT.")
    if d_schema > 0.001:
        lines.append(f"- Tool-call schema validity improved by {d_schema:+.3f} vs SFT.")
    elif d_schema < -0.001:
        lines.append(f"- Tool-call schema validity fell by {d_schema:+.3f} vs SFT.")
    else:
        lines.append("- Tool-call schema validity was unchanged vs SFT.")
    if d_recall > 0.001:
        lines.append(f"- Perception recall improved by {d_recall:+.3f} vs SFT.")
    elif d_recall < -0.001:
        lines.append(f"- Perception recall fell by {d_recall:+.3f} vs SFT.")
    else:
        lines.append("- Perception recall was unchanged vs SFT.")
    d_fmt = delta(grpo, sft, "format_rate")
    if abs(d_fmt) > 0.001:
        lines.append(f"- ReAct format compliance changed by {d_fmt:+.3f} vs SFT on test.")
    d_turns = delta(sft, grpo, "mean_turns")
    if abs(d_turns) > 0.01:
        lines.append(f"- Mean turns per episode changed by {d_turns:+.2f} (positive = shorter trajectories).")
    if d_sft > 0.001 or d_sft < -0.001:
        lines.append(f"- SFT cold start changed pass@1 by {d_sft:+.3f} vs BASE on test.")
    if grpo:
        lines.append(
            f"- Absolute GRPO test numbers: pass@1 **{fmt(grpo.get('pass@1'))}**, "
            f"completion **{fmt(grpo.get('task_completion_rate'))}**, schema **{fmt(grpo.get('tool_call_schema_validity'))}**; "
            f"on test_hard pass@1 **{fmt(hard_grpo.get('pass@1'))}**."
        )
    mix_summary = results.get("failure_mix_summary", {})
    if mix_summary:
        lines.append(
            f"- Dominant failure bucket: **{mix_summary.get('base_dominant')}** for BASE -> "
            f"**{mix_summary.get('grpo_dominant')}** for GRPO (test)."
        )
        changes = mix_summary.get("count_changes", {})
        improved = [f"{b} ({changes[b]:+d})" for b, v in changes.items() if v < 0]
        if improved:
            lines.append(f"- Buckets reduced by GRPO: {', '.join(improved)}.")
    return "\n".join(lines)


def limitations(results: dict[str, Any]) -> str:
    cfg = results["config"]
    env = results["environment"]
    n_items = cfg.get("eval_max_items")
    k = cfg.get("eval_k")
    return (
        f"- **Small model / single hardware tier.** All numbers are from the `{env['model_tier']}` tier on "
        f"`{env['device']}`; the `small`/`base` CUDA tiers are wired but not exercised on this machine.\n"
        f"- **Synthetic task suite.** Tasks are generated locally from seed {cfg['seed']} and are not a "
        "substitute for real-world chart/document/screenshot benchmarks.\n"
        f"- **Small evaluation sample.** {n_items} items per split (stratified across 4 families) with "
        f"k={k}; pass@4 therefore rests on a handful of items per family and has wide error bars.\n"
        f"- **Single training seed.** SFT and GRPO were run once with seed {cfg['seed']}; no seed-variance "
        "error bars are reported.\n"
        "- **Teacher-forced SFT targets** come from executing the gold tool sequence; the model never sees "
        "alternative valid strategies during cold start.\n"
        "- **Perception recall** counts facts that appear anywhere in the model text, tool arguments or "
        "tool observations; a retrieval tool call can therefore surface a fact without the model reading it."
    )


def figures_section(results: dict[str, Any]) -> str:
    fig_dir = Path("results/figures")
    names = [
        ("reward_curve.png", "GRPO reward curve"),
        ("completion_vs_step.png", "task completion vs GRPO step"),
        ("failure_mix.png", "failure mix across checkpoints"),
        ("per_family_pass1.png", "per-family pass@1"),
        ("reward_components.png", "reward component breakdown"),
        ("summary_metrics.png", "BASE vs SFT vs GRPO summary"),
    ]
    lines = []
    for name, title in names:
        if (fig_dir / name).exists():
            lines.append(f"![{title}](figures/{name})")
    return "\n\n".join(lines)


def build_markdown(results: dict[str, Any], readme_block: bool = False) -> str:
    parts: list[str] = []
    if not readme_block:
        parts += ["# VLM-GRPO results", ""]
        parts += ["All numbers below are generated by `scripts/make_results_table.py` from "
                  "`results/results.json`, which is written only by executed evaluation runs.", ""]
    parts += ["## Environment", "", environment_line(results), ""]
    parts += ["## Headline metrics", "", main_table(results), ""]
    parts += ["## Deltas vs BASE and vs SFT", "", delta_table(results), ""]
    parts += ["## Per-family breakdown (test)", "", per_family_table(results, "test"), ""]
    parts += ["## Failure attribution (test)", "", failure_table(results, "test"), ""]
    parts += ["", failure_table(results, "test_hard"), ""]
    parts += ["## Training and cost", "", training_table(results), ""]
    parts += ["## What improved, what did not", "", findings(results), ""]
    parts += ["## Limitations", "", limitations(results), ""]
    figs = figures_section(results)
    if figs:
        parts += ["## Figures", "", figs, ""]
    if not readme_block:
        parts += [
            "## Reproducibility",
            "",
            "```bash",
            "pip install -r requirements.txt",
            "make data          # deterministic suite + PNGs",
            "make eval-base     # BASE zero-shot",
            "make sft           # LoRA SFT cold start",
            "make grpo          # GRPO/RLVR",
            "make eval-all      # {BASE,SFT,GRPO} x {test,test_hard} -> results/results.json",
            "make report        # regenerate this file and the README tables",
            "```",
            "",
            f"Commit: `{results['environment'].get('commit_sha')}`.",
            "",
        ]
    return "\n".join(parts)


def update_readme(readme_path: Path, block: str) -> bool:
    if not readme_path.exists():
        return False
    text = readme_path.read_text(encoding="utf-8")
    start, end = "<!-- RESULTS_START -->", "<!-- RESULTS_END -->"
    if start not in text or end not in text:
        print(f"[report] README markers {start}/{end} not found; skipping README update")
        return False
    before = text.split(start)[0]
    after = text.split(end)[1]
    readme_path.write_text(before + start + "\n" + block + "\n" + end + after, encoding="utf-8")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Regenerate results tables from results.json")
    parser.add_argument("--config", default="configs/tiny.yaml")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    results_path = Path(cfg.results_dir) / "results.json"
    if not results_path.exists():
        raise SystemExit(f"{results_path} missing; run `make eval-all` first")
    results = read_json(results_path)
    results_md = Path(cfg.results_dir) / "RESULTS.md"
    results_md.write_text(build_markdown(results), encoding="utf-8")
    block = build_markdown(results, readme_block=True)
    updated = update_readme(Path("README.md"), block)
    print(f"[report] wrote {results_md} ({len(build_markdown(results))} chars); README updated={updated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
