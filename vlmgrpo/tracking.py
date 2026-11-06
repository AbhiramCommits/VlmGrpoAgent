"""Experiment tracking: JSONL always on, W&B only when explicitly available.

The JSONL file at ``runs/<run_id>/metrics.jsonl`` is the source of truth for
every curve plotted in ``results/figures``.  W&B is a mirror, never a
requirement: it is enabled only when ``WANDB_API_KEY`` is set (or
``tracking.use_wandb: always``) and the package is installed.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .utils import append_jsonl, json_safe, write_json


class Tracker:
    """Append-only metrics tracker."""

    def __init__(
        self,
        run_id: str,
        runs_dir: str | Path = "runs",
        config: dict[str, Any] | None = None,
        project: str = "vlm-grpo",
        use_wandb: str = "auto",
        tags: list[str] | None = None,
    ) -> None:
        self.run_id = run_id
        self.run_dir = Path(runs_dir) / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.config = config or {}
        write_json(self.run_dir / "config.json", self.config)
        self._wandb = None
        self._step = 0
        if self._want_wandb(use_wandb):
            try:
                import wandb

                self._wandb = wandb
                wandb.init(
                    project=project,
                    name=run_id,
                    config=self.config,
                    dir=str(self.run_dir),
                    tags=tags or [],
                    reinit=True,
                )
            except Exception as exc:  # pragma: no cover - network dependent
                print(f"[tracker] W&B disabled: {exc}")
                self._wandb = None

    @staticmethod
    def _want_wandb(mode: str) -> bool:
        if mode == "never":
            return False
        if mode == "always":
            return True
        return bool(os.environ.get("WANDB_API_KEY"))

    def log(self, metrics: dict[str, Any], step: int | None = None, **context: Any) -> dict[str, Any]:
        """Append one row to JSONL and mirror to W&B when enabled."""
        row = dict(metrics)
        row.setdefault("run_id", self.run_id)
        row["step"] = int(step if step is not None else self._step)
        row.update(context)
        append_jsonl(self.metrics_path, row)
        if self._wandb is not None:
            try:
                self._wandb.log(json_safe(row), step=row["step"])
            except Exception:
                pass
        if step is None:
            self._step += 1
        else:
            self._step = max(self._step, int(step) + 1)
        return row

    def log_summary(self, metrics: dict[str, Any]) -> None:
        write_json(self.run_dir / "summary.json", metrics)

    @property
    def url(self) -> str | None:
        if self._wandb is not None and getattr(self._wandb, "run", None) is not None:
            try:
                return self._wandb.run.url
            except Exception:
                return None
        return None

    def finish(self) -> None:
        if self._wandb is not None:
            try:
                self._wandb.finish()
            except Exception:
                pass
            self._wandb = None


def read_metrics(run_dir: str | Path) -> list[dict[str, Any]]:
    from .utils import read_jsonl

    return read_jsonl(Path(run_dir) / "metrics.jsonl")
