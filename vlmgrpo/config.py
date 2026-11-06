"""Typed configuration objects for VLM-GRPO.

Every experiment is described by a single YAML file that is parsed into a
:class:`RunConfig`.  Defaults are deliberately small so the whole pipeline can
run on a laptop; scale up in the YAML rather than in code.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

# --------------------------------------------------------------------------- #
# Config sections
# --------------------------------------------------------------------------- #


@dataclass
class TaskConfig:
    """Task-suite generation knobs."""

    seed: int = 1234
    train_n: int = 160
    val_n: int = 40
    test_n: int = 40
    test_hard_n: int = 40
    n_per_family: int | None = None  # overrides <split>_n when set
    image_size: int = 448
    image_dir: str = "data/images"


@dataclass
class RolloutConfig:
    """Agent loop / decoding knobs."""

    max_turns: int = 4
    max_new_tokens: int = 192
    temperature: float = 0.7
    top_p: float = 0.95
    do_sample: bool = True
    backend: str = "hf"  # hf | vllm
    batch_size: int = 8
    tool_timeout: float = 2.0


@dataclass
class SFTConfig:
    """LoRA supervised fine-tuning knobs."""

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lr: float = 1e-4
    epochs: int = 3
    batch_size: int = 4
    grad_accum: int = 2
    max_length: int = 1024
    warmup_ratio: float = 0.05
    weight_decay: float = 0.0
    max_steps: int = -1
    full_params: bool = False
    grad_clip: float = 1.0
    val_every: int = 25
    val_items: int = 8


@dataclass
class GRPOConfig:
    """Online RL knobs."""

    group_size: int = 8
    steps: int = 40
    lr: float = 1e-5
    temperature: float = 0.9
    top_p: float = 0.95
    clip_eps: float = 0.2
    kl_beta: float = 0.02
    kl_kind: str = "k3"  # k1 | k3
    last_turn_only: bool = False
    prompts_per_step: int = 1
    max_length: int = 1280
    max_new_tokens: int = 192
    max_turns: int = 4
    val_every: int = 10
    val_items: int = 8
    eval_k: int = 1
    grad_clip: float = 1.0
    skip_zero_variance: bool = True
    seed: int = 0
    rollout_backend: str = "hf"


@dataclass
class EvalConfig:
    """Evaluation knobs."""

    k: int = 4
    max_items: int = 24
    temperature: float = 0.8
    max_turns: int = 4
    max_new_tokens: int = 192
    batch_size: int = 8
    split: str = "test"


@dataclass
class RewardConfig:
    """Weights of the verifiable composite reward (no learned reward model)."""

    w_answer: float = 1.0
    w_tool: float = 0.2
    w_trace: float = 0.3
    w_format: float = 0.1
    w_length: float = 0.05
    length_budget: int = 1600


@dataclass
class TrackingConfig:
    """Metrics plumbing."""

    project: str = "vlm-grpo"
    use_wandb: str = "auto"  # auto | never | always
    log_every: int = 1


@dataclass
class RunConfig:
    """Top-level experiment configuration."""

    run_id: str = "run"
    seed: int = 1234
    model_tier: str = "auto"  # auto | stub | tiny | small | base
    device: str = "auto"  # auto | cuda | mps | cpu
    dtype: str = "auto"  # auto | float32 | float16 | bfloat16
    task: TaskConfig = field(default_factory=TaskConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    sft: SFTConfig = field(default_factory=SFTConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    data_dir: str = "data"
    results_dir: str = "results"
    runs_dir: str = "runs"


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def _coerce(section_cls: type, payload: Any) -> Any:
    """Instantiate ``section_cls`` from a dict, rejecting unknown keys."""
    if payload is None:
        return section_cls()
    if not isinstance(payload, dict):
        raise TypeError(f"{section_cls.__name__} expects a mapping, got {type(payload)}")
    known = {f.name for f in fields(section_cls)}
    unknown = set(payload) - known
    if unknown:
        raise ValueError(f"Unknown keys for {section_cls.__name__}: {sorted(unknown)}")
    return section_cls(**payload)


def run_config_from_dict(payload: dict[str, Any]) -> RunConfig:
    payload = dict(payload or {})
    section_names = {f.name for f in fields(RunConfig) if is_dataclass(getattr(RunConfig, f.name, None))}
    sections: dict[str, Any] = {}
    for name in list(payload):
        if name in section_names:
            sections[name] = payload.pop(name)
    known = {f.name for f in fields(RunConfig)}
    unknown = set(payload) - known
    if unknown:
        raise ValueError(f"Unknown top-level config keys: {sorted(unknown)}")
    return RunConfig(**payload, **{k: _coerce(type(getattr(RunConfig, k, None) or RunConfig.__dataclass_fields__[k].default_factory()), v) for k, v in sections.items()})


def load_config(path: str | Path) -> RunConfig:
    """Read a YAML file into a fully-populated :class:`RunConfig`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        payload = yaml.safe_load(fh) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Config root must be a mapping, got {type(payload)}")

    def section(name: str, cls: type) -> Any:
        return _coerce(cls, payload.pop(name, None))

    cfg = RunConfig(
        run_id=str(payload.pop("run_id", "run")),
        seed=int(payload.pop("seed", 1234)),
        model_tier=str(payload.pop("model_tier", "auto")),
        device=str(payload.pop("device", "auto")),
        dtype=str(payload.pop("dtype", "auto")),
        data_dir=str(payload.pop("data_dir", "data")),
        results_dir=str(payload.pop("results_dir", "results")),
        runs_dir=str(payload.pop("runs_dir", "runs")),
        task=section("task", TaskConfig),
        rollout=section("rollout", RolloutConfig),
        sft=section("sft", SFTConfig),
        grpo=section("grpo", GRPOConfig),
        eval=section("eval", EvalConfig),
        reward=section("reward", RewardConfig),
        tracking=section("tracking", TrackingConfig),
    )
    if payload:
        raise ValueError(f"Unknown top-level config keys: {sorted(payload)}")
    return cfg


def config_to_dict(cfg: RunConfig) -> dict[str, Any]:
    """JSON-friendly dict of a run config."""
    return asdict(cfg)
