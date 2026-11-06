"""Result figures generated from executed runs only.

All plots read ``results/results.json`` and ``runs/<run_id>/metrics.jsonl``;
nothing is hand-typed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..config import RunConfig, load_config
from ..utils import read_json, read_jsonl

COLORS = {"BASE": "#8C8C8C", "SFT": "#4C72B0", "GRPO": "#55A868"}
BUCKET_COLORS = {
    "PERCEPTION": "#C44E52",
    "PLANNING": "#DD8452",
    "TOOL_SCHEMA": "#8172B3",
    "EXECUTION": "#937860",
    "ARITHMETIC": "#4C72B0",
    "FORMAT": "#DA8BC3",
    "BUDGET": "#55A868",
}


def _style(ax, title: str, ylabel: str = "", xlabel: str = "") -> None:
    ax.set_title(title, fontsize=12)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10)
    ax.grid(True, axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)


def _save(fig, path: Path, figures: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figures.append(str(path))


def _grpo_rows(cfg: RunConfig) -> list[dict[str, Any]]:
    path = Path(cfg.runs_dir) / cfg.run_id / "metrics.jsonl"
    rows = read_jsonl(path)
    return [r for r in rows if r.get("phase") == "grpo"]


def make_plots(cfg: RunConfig | None = None) -> list[str]:
    cfg = cfg or load_config("configs/tiny.yaml")
    results_path = Path(cfg.results_dir) / "results.json"
    if not results_path.exists():
        print(f"[plots] missing {results_path}; run the evaluation matrix first")
        return []
    results = read_json(results_path)
    fig_dir = Path(cfg.results_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    figures: list[str] = []
    rows = _grpo_rows(cfg)

    # 1. reward curve
    if rows:
        steps = [r["step"] for r in rows]
        rewards = [r.get("reward_mean", 0.0) for r in rows]
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        ax.plot(steps, rewards, color="#4C72B0", alpha=0.45, label="group mean reward")
        window = max(1, min(5, len(rewards) // 3 or 1))
        if len(rewards) >= window:
            smooth = np.convolve(rewards, np.ones(window) / window, mode="valid")
            ax.plot(steps[window - 1 :], smooth, color="#C44E52", lw=2, label=f"{window}-step mean")
        ax.axhline(0.0, color="#8C8C8C", lw=1, ls="--")
        _style(ax, "GRPO training reward (verifiable, no reward model)", "reward", "GRPO step")
        ax.legend(fontsize=9)
        _save(fig, fig_dir / "reward_curve.png", figures)

    # 2. task completion vs training step
    if rows:
        steps = [r["step"] for r in rows]
        train_completion = [r.get("train_completion", 0.0) for r in rows]
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        ax.plot(steps, train_completion, color="#55A868", alpha=0.5, label="train completion (group)")
        val_steps = [r["step"] for r in rows if "val_completion" in r]
        val_completion = [r["val_completion"] for r in rows if "val_completion" in r]
        if val_steps:
            ax.plot(val_steps, val_completion, "o-", color="#C44E52", lw=2, label="val completion")
        ax.set_ylim(-0.02, 1.02)
        _style(ax, "Multi-step task completion during GRPO", "completion rate", "GRPO step")
        ax.legend(fontsize=9)
        _save(fig, fig_dir / "completion_vs_step.png", figures)

    # 3. failure mix stacked bars
    mix = results.get("failure_mix", {}).get("test", {})
    if mix:
        labels = [lab for lab in ("BASE", "SFT", "GRPO") if lab.lower() in mix]
        buckets = ["PERCEPTION", "PLANNING", "TOOL_SCHEMA", "EXECUTION", "ARITHMETIC", "FORMAT", "BUDGET"]
        fig, ax = plt.subplots(figsize=(6.8, 3.8))
        bottom = np.zeros(len(labels))
        totals = [max(1, mix[lab.lower()]["n_rollouts"]) for lab in labels]
        for bucket in buckets:
            vals = np.array([mix[lab.lower()]["counts"][bucket] / t for lab, t in zip(labels, totals)])
            ax.bar(labels, vals, bottom=bottom, color=BUCKET_COLORS[bucket], label=bucket, width=0.6)
            bottom += vals
        _style(ax, "Failure mix on test (share of all rollouts)", "share", "")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7, ncol=2, loc="upper right")
        _save(fig, fig_dir / "failure_mix.png", figures)

    # 4. per-family pass@1
    per_family = results.get("per_family", {})
    if per_family:
        families = []
        for lab in ("BASE", "SFT", "GRPO"):
            families.extend(per_family.get(lab, {}).get("test", {}).keys())
        families = list(dict.fromkeys(families))
        if families:
            fig, ax = plt.subplots(figsize=(7.2, 3.6))
            width = 0.8 / 3
            for i, lab in enumerate(("BASE", "SFT", "GRPO")):
                vals = [per_family.get(lab, {}).get("test", {}).get(fam, {}).get("pass@1", 0.0) for fam in families]
                ax.bar(np.arange(len(families)) + (i - 1) * width, vals, width=width, label=lab, color=COLORS[lab])
            ax.set_xticks(np.arange(len(families)))
            ax.set_xticklabels(families, rotation=15, fontsize=9)
            ax.set_ylim(0, 1.02)
            _style(ax, "Per-family pass@1 on test", "pass@1")
            ax.legend(fontsize=9)
            _save(fig, fig_dir / "per_family_pass1.png", figures)

    # 5. reward component breakdown during GRPO
    if rows:
        components = {
            "answer_correct": "#C44E52",
            "tool_schema_valid_rate": "#4C72B0",
            "required_tool_used": "#55A868",
            "format_ok": "#8172B3",
        }
        fig, ax = plt.subplots(figsize=(6.6, 3.6))
        steps = [r["step"] for r in rows]
        for key, color in components.items():
            ax.plot(steps, [r.get(key, 0.0) for r in rows], color=color, lw=1.8, label=key)
        _style(ax, "Reward components during GRPO", "component value", "GRPO step")
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=8)
        _save(fig, fig_dir / "reward_components.png", figures)

    # 6. pass@1 / completion / schema summary on both splits
    metrics = results.get("metrics", {})
    if metrics:
        split_names = [s for s in ("test", "test_hard") if any(s in v for v in metrics.values())]
        if split_names:
            fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
            names = ["pass@1", "task_completion_rate", "tool_call_schema_validity"]
            titles = ["pass@1", "task completion", "tool-call schema validity"]
            for ax, metric, title in zip(axes, names, titles):
                x = np.arange(len(split_names))
                width = 0.8 / 3
                for i, lab in enumerate(("BASE", "SFT", "GRPO")):
                    vals = [metrics.get(lab, {}).get(s, {}).get(metric, 0.0) for s in split_names]
                    ax.bar(x + (i - 1) * width, vals, width=width, color=COLORS[lab], label=lab)
                ax.set_xticks(x)
                ax.set_xticklabels(split_names, fontsize=9)
                ax.set_ylim(0, 1.02)
                _style(ax, title)
            axes[0].legend(fontsize=8)
            _save(fig, fig_dir / "summary_metrics.png", figures)
    return figures


if __name__ == "__main__":
    print(make_plots())
