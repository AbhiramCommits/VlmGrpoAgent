"""Model-free verifiable reward stack.

``r = w_answer * answer_correct
   + w_tool   * tool_schema_valid_rate
   + w_trace  * required_tool_used
   + w_format * format_ok
   - w_length * length_penalty``

Every component is returned separately so it can be logged and ablated.
There is no learned reward model anywhere in this repo.
"""

from __future__ import annotations

from typing import Any

from .config import RewardConfig
from .verify import perception_recall, verify_answer, verify_tool_trace


def _reward_config(cfg: Any) -> RewardConfig:
    if cfg is None:
        return RewardConfig()
    if isinstance(cfg, RewardConfig):
        return cfg
    return getattr(cfg, "reward", RewardConfig())


def tool_schema_valid_rate(traj: Any) -> float:
    calls = traj.tool_calls
    if not calls:
        return 0.0
    valid = sum(1 for c in calls if c.get("schema_valid") or c.get("ok"))
    return round(valid / len(calls), 4)


def reward_breakdown(traj: Any, item: Any, cfg: Any = None) -> dict[str, float]:
    """Return every reward component plus the weighted total."""
    rc = _reward_config(cfg)
    answer_correct = 1.0 if verify_answer(traj.pred_answer, item.gold_answer, tol=0.01) else 0.0
    schema_rate = tool_schema_valid_rate(traj)
    trace = 1.0 if verify_tool_trace(traj.tool_calls, item) else 0.0
    fmt = 1.0 if getattr(traj, "format_valid", False) else 0.0
    length_penalty = min(1.0, float(getattr(traj, "total_assistant_chars", 0)) / max(1, rc.length_budget))
    total = (
        rc.w_answer * answer_correct
        + rc.w_tool * schema_rate
        + rc.w_trace * trace
        + rc.w_format * fmt
        - rc.w_length * length_penalty
    )
    return {
        "answer_correct": answer_correct,
        "tool_schema_valid_rate": schema_rate,
        "required_tool_used": trace,
        "format_ok": fmt,
        "length_penalty": round(length_penalty, 4),
        "total": round(total, 6),
        "perception_recall": perception_recall(traj.perception_text(), item.gold_perception_facts),
    }


def compute_reward(traj: Any, item: Any, cfg: Any = None) -> float:
    return float(reward_breakdown(traj, item, cfg)["total"])
