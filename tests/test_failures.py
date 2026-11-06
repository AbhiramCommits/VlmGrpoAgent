"""Failure attribution for every bucket, with synthetic trajectories."""

from __future__ import annotations

import json

from vlmgrpo.agent import Trajectory, TurnRecord
from vlmgrpo.analysis.failures import (
    attribute_failure,
    failure_mix,
    improvement_summary,
    select_examples,
)
from vlmgrpo.tasks.families import generate_task


def _item(tmp_path):
    return generate_task("chart_agg", 1234, "test", 0, tmp_path)


def _traj(item, pred=None, calls=(), facts_text="", no_turns=False):
    turns = []
    if facts_text:
        turns.append(TurnRecord(turn=0, raw_text=f"<think>{facts_text}</think>", action_type="none"))
    for call in calls:
        turns.append(
            TurnRecord(
                turn=len(turns),
                raw_text=f'<think>call</think><tool>{json.dumps({"name": call["name"], "args": call["args"]})}</tool>',
                action_type="tool",
                tool_name=call["name"],
                tool_args=call["args"],
                schema_valid=call.get("schema_valid", True),
                tool_ok=call.get("ok", True),
            )
        )
    if pred is not None:
        turns.append(
            TurnRecord(
                turn=len(turns),
                raw_text=f"<think>done</think><answer>{pred}</answer>",
                action_type="answer",
            )
        )
    if no_turns:
        turns = [TurnRecord(turn=0, raw_text="no tags here", action_type="none")]
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


def test_success_is_not_a_failure(tmp_path):
    item = _item(tmp_path)
    traj = _traj(item, pred=item.gold_answer, calls=[{"name": "python_exec", "args": {"code": "print(1)"}}])
    assert attribute_failure(traj, item) is None


def test_perception_bucket(tmp_path):
    item = _item(tmp_path)
    traj = _traj(item, pred="999", calls=[{"name": "python_exec", "args": {"code": "print(1)"}}])
    assert attribute_failure(traj, item) == "PERCEPTION"


def test_planning_bucket(tmp_path):
    item = _item(tmp_path)
    facts = ", ".join(str(v) for v in item.gold_perception_facts.values())
    traj = _traj(item, pred="999", facts_text=facts, calls=[])
    item.required_tools = ["python_exec"]
    assert attribute_failure(traj, item) == "PLANNING"


def test_tool_schema_bucket(tmp_path):
    item = _item(tmp_path)
    facts = ", ".join(str(v) for v in item.gold_perception_facts.values())
    traj = _traj(
        item,
        pred="999",
        facts_text=facts,
        calls=[{"name": "made_up", "args": {}, "schema_valid": False, "ok": False}],
    )
    assert attribute_failure(traj, item) == "TOOL_SCHEMA"


def test_execution_bucket(tmp_path):
    item = _item(tmp_path)
    facts = ", ".join(str(v) for v in item.gold_perception_facts.values())
    traj = _traj(
        item,
        pred="999",
        facts_text=facts,
        calls=[{"name": "python_exec", "args": {"code": "print(1/0)"}, "schema_valid": True, "ok": False}],
    )
    assert attribute_failure(traj, item) == "EXECUTION"


def test_arithmetic_bucket(tmp_path):
    item = _item(tmp_path)
    facts = ", ".join(str(v) for v in item.gold_perception_facts.values())
    traj = _traj(
        item,
        pred="999",
        facts_text=facts,
        calls=[{"name": "python_exec", "args": {"code": "print(1)"}, "schema_valid": True, "ok": True}],
    )
    assert attribute_failure(traj, item) == "ARITHMETIC"


def test_format_bucket(tmp_path):
    item = _item(tmp_path)
    traj = _traj(item, pred=None, no_turns=True)
    assert attribute_failure(traj, item) == "FORMAT"


def test_budget_bucket(tmp_path):
    item = _item(tmp_path)
    traj = _traj(item, pred=None, calls=[{"name": "python_exec", "args": {"code": "print(1)"}}])
    assert attribute_failure(traj, item) == "BUDGET"


def test_failure_mix_and_summary(tmp_path):
    item = _item(tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "test.jsonl").write_text(json.dumps(item.to_dict()) + "\n")
    results = tmp_path / "results"
    traj_dir = results / "trajectories"
    traj_dir.mkdir(parents=True)
    traj = _traj(item, pred="999", no_turns=True)
    (traj_dir / "base_test.jsonl").write_text(json.dumps(traj.to_dict()) + "\n")
    mix = failure_mix(checkpoints=("base",), splits=("test",), results_dir=results, data_dir=data_dir)
    assert mix["test"]["base"]["counts"]["FORMAT"] == 1
    assert mix["test"]["base"]["dominant_bucket"] == "FORMAT"
    assert isinstance(improvement_summary(mix, "test"), str)
    examples = select_examples(results_dir=results, data_dir=data_dir)
    assert isinstance(examples, list)


def test_improvement_summary_and_examples(tmp_path):
    from vlmgrpo.analysis.failures import BUCKETS

    item = _item(tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    (data_dir / "test.jsonl").write_text(json.dumps(item.to_dict()) + "\n")
    results = tmp_path / "results"
    traj_dir = results / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)
    success = _traj(item, pred=item.gold_answer, calls=[{"name": "python_exec", "args": {"code": "print(1)"}}])
    perception = _traj(item, pred="999", calls=[{"name": "python_exec", "args": {"code": "print(1)"}}])
    planning = _traj(item, pred="999", no_turns=True)
    for name, traj in (("base", perception), ("sft", planning), ("grpo", success)):
        (traj_dir / f"{name}_test.jsonl").write_text(json.dumps(traj.to_dict()) + "\n")
    mix = failure_mix(checkpoints=("base", "sft", "grpo"), splits=("test",), results_dir=results, data_dir=data_dir)
    assert mix["test"]["grpo"]["success_count"] == 1
    synthetic = {
        "test": {
            "base": {"counts": {b: 1 for b in BUCKETS}, "dominant_bucket": "PERCEPTION"},
            "grpo": {"counts": {b: 0 for b in BUCKETS}, "dominant_bucket": None},
        }
    }
    summary = improvement_summary(synthetic, "test")
    assert "PERCEPTION" in summary
    examples = select_examples(results_dir=results, data_dir=data_dir)
    assert any(e["bucket"] == "SUCCESS" for e in examples)
