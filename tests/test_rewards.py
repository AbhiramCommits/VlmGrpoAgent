"""Reward monotonicity properties and component separability."""

from __future__ import annotations

from vlmgrpo.agent import Trajectory, TurnRecord
from vlmgrpo.config import RunConfig
from vlmgrpo.rewards import compute_reward, reward_breakdown, tool_schema_valid_rate
from vlmgrpo.tasks.families import generate_task


def _item(tmp_path):
    return generate_task("chart_agg", 1234, "test", 0, tmp_path)


def _traj(item, pred, tool_ok=True, schema_valid=True, include_tools=True, text_len=40):
    turns = []
    if include_tools:
        for call in item.gold_tool_calls:
            turns.append(
                TurnRecord(
                    turn=len(turns),
                    raw_text="<think>x</think>",
                    action_type="tool",
                    tool_name=call["name"],
                    tool_args=call.get("args", {}),
                    schema_valid=schema_valid,
                    tool_ok=tool_ok,
                )
            )
    answer_raw = f"<think>done</think><answer>{pred}</answer>"
    turns.append(
        TurnRecord(
            turn=len(turns),
            raw_text=answer_raw + (" x" * text_len),
            action_type="answer",
        )
    )
    return Trajectory(
        task_id=item.task_id,
        family=item.family,
        split=item.split,
        question=item.question,
        gold_answer=item.gold_answer,
        pred_answer=pred,
        turns=turns,
        required_tools=item.required_tools,
        gold_perception_facts=item.gold_perception_facts,
        image_paths=item.image_paths,
    )


def test_correct_answer_scores_higher(tmp_path):
    item = _item(tmp_path)
    cfg = RunConfig()
    right = reward_breakdown(_traj(item, item.gold_answer), item, cfg)
    wrong = reward_breakdown(_traj(item, "999"), item, cfg)
    assert right["total"] > wrong["total"]
    assert right["answer_correct"] == 1.0 and wrong["answer_correct"] == 0.0


def test_required_tools_add_reward(tmp_path):
    item = _item(tmp_path)
    cfg = RunConfig()
    with_tools = reward_breakdown(_traj(item, "999", include_tools=True), item, cfg)
    without = reward_breakdown(_traj(item, "999", include_tools=False), item, cfg)
    assert with_tools["required_tool_used"] == 1.0
    assert without["required_tool_used"] == 0.0
    assert with_tools["total"] > without["total"]


def test_schema_violation_lowers_tool_component(tmp_path):
    item = _item(tmp_path)
    cfg = RunConfig()
    good = reward_breakdown(_traj(item, "999", schema_valid=True, tool_ok=True), item, cfg)
    bad = reward_breakdown(_traj(item, "999", schema_valid=False, tool_ok=False), item, cfg)
    assert good["tool_schema_valid_rate"] == 1.0
    assert bad["tool_schema_valid_rate"] == 0.0
    assert good["total"] > bad["total"]


def test_length_penalty_monotone(tmp_path):
    item = _item(tmp_path)
    cfg = RunConfig()
    short = reward_breakdown(_traj(item, "999", text_len=1), item, cfg)
    long = reward_breakdown(_traj(item, "999", text_len=5000), item, cfg)
    assert long["length_penalty"] == 1.0
    assert long["total"] <= short["total"]


def test_format_component(tmp_path):
    item = _item(tmp_path)
    cfg = RunConfig()
    formatted = reward_breakdown(_traj(item, "999"), item, cfg)
    assert formatted["format_ok"] == 1.0
    traj = _traj(item, None, include_tools=False)
    traj.turns[-1].raw_text = "no tags"
    traj.pred_answer = None
    broken = reward_breakdown(traj, item, cfg)
    assert broken["format_ok"] == 0.0


def test_compute_reward_matches_total(tmp_path):
    item = _item(tmp_path)
    traj = _traj(item, item.gold_answer)
    assert compute_reward(traj, item, RunConfig()) == reward_breakdown(traj, item, RunConfig())["total"]


def test_no_tools_has_zero_schema_rate(tmp_path):
    item = _item(tmp_path)
    traj = _traj(item, "999", include_tools=False)
    assert tool_schema_valid_rate(traj) == 0.0
