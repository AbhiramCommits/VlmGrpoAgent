"""Small shared helpers: seeding, timing, git metadata, JSON utilities."""

from __future__ import annotations

import json
import os
import random
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np


def seed_everything(seed: int) -> None:
    """Seed python, numpy and torch (torch is optional at import time)."""
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:  # pragma: no cover - torch always installed in practice
        pass


def git_sha(short: bool = True) -> str:
    """Return the current git commit sha (or ``unknown`` outside a repo)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short" if short else "--verify", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        sha = out.stdout.strip()
        return sha or "unknown"
    except Exception:
        return "unknown"


def git_dirty() -> bool:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, timeout=10, check=False
        )
        return bool(out.stdout.strip())
    except Exception:
        return False


class Timer:
    """Wall-clock timer usable as a context manager."""

    def __init__(self) -> None:
        self.start = time.perf_counter()
        self.elapsed = 0.0

    def __enter__(self) -> Timer:
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.elapsed = time.perf_counter() - self.start

    def stop(self) -> float:
        self.elapsed = time.perf_counter() - self.start
        return self.elapsed


@contextmanager
def timed() -> Iterator[Timer]:
    timer = Timer()
    try:
        yield timer
    finally:
        timer.stop()


def json_safe(obj: Any) -> Any:
    """Recursively convert numpy/torch scalars and containers to JSON types."""
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        if obj != obj or obj in (float("inf"), float("-inf")):
            return None
        return obj
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return json_safe(obj.item())
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    try:
        import torch

        if isinstance(obj, torch.Tensor):
            return json_safe(obj.detach().cpu().tolist())
    except Exception:
        pass
    return str(obj)


def write_json(path: str | Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(json_safe(payload), fh, indent=2, sort_keys=False)
        fh.write("\n")
    return path


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(json_safe(row), sort_keys=False) + "\n")


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(json_safe(row), sort_keys=False) + "\n")
    return path


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def count_params(model: Any, trainable_only: bool = False) -> int:
    try:
        params = model.parameters()
        return int(sum(p.numel() for p in params if (p.requires_grad or not trainable_only)))
    except Exception:
        return 0


def human_hours(seconds: float) -> float:
    return round(seconds / 3600.0, 4)


def encode_text(tokenizer: Any, text: str) -> list[int]:
    """Tokenize one string with an HF tokenizer or the stub char tokenizer."""
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    return list(ids)


# --------------------------------------------------------------------------- #
# Checkpoint registry (runs/checkpoints.json)
# --------------------------------------------------------------------------- #


def register_checkpoint(
    runs_dir: str | Path, name: str, path: str | Path, meta: dict[str, Any] | None = None
) -> Path:
    """Record ``name -> path`` so evaluation can resolve ``sft`` / ``grpo``."""
    manifest_path = Path(runs_dir) / "checkpoints.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    manifest[name] = {"path": str(path), "meta": json_safe(meta or {})}
    write_json(manifest_path, manifest)
    return manifest_path


def read_checkpoints(runs_dir: str | Path) -> dict[str, Any]:
    manifest_path = Path(runs_dir) / "checkpoints.json"
    return read_json(manifest_path) if manifest_path.exists() else {}


def resolve_checkpoint(runs_dir: str | Path, name: str | None) -> Path | None:
    """Resolve a checkpoint name to a directory, or ``None`` for BASE."""
    if not name or name == "base":
        return None
    candidate = Path(name)
    if candidate.exists() and candidate.is_dir():
        return candidate
    manifest = read_checkpoints(runs_dir)
    if name in manifest:
        path = Path(manifest[name]["path"])
        if path.exists():
            return path
        print(f"[checkpoints] registered path for {name!r} does not exist: {path}")
        return None
    return None
