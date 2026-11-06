"""SFT data tests: gold trajectories must round-trip into valid agent actions."""

from __future__ import annotations

import json

import pytest

from vlmgrpo.agent import parse_action
from vlmgrpo.data.build_sft import render_examples
from vlmgrpo.registry import StubProcessor
from vlmgrpo.tasks.families import FAMILIES, generate_task
from vlmgrpo.verify import verify_tool_call_schema


@pytest.fixture(scope="module")
def examples(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("sft_data")
    items = [generate_task(fam, 1234 + i, "test", 0, tmp) for i, fam in enumerate(FAMILIES)]
    records, stats = render_examples(items, StubProcessor(), max_length=4096, max_turns=4)
    return items, records, stats


def test_every_example_parses_into_an_action(examples):
    _, records, _ = examples
    assert records
    for rec in records:
        action = parse_action(rec["target_text"])
        assert action["type"] in ("tool", "answer"), rec["target_text"]


def test_tool_examples_have_valid_schema(examples):
    _, records, _ = examples
    tool_records = [r for r in records if "<tool>" in r["target_text"]]
    assert tool_records
    for rec in tool_records:
        action = parse_action(rec["target_text"])
        assert verify_tool_call_schema({"name": action["name"], "args": action["args"]})


def test_final_turn_is_the_gold_answer(examples):
    items, records, _ = examples
    by_task: dict[str, list[dict]] = {}
    for rec in records:
        by_task.setdefault(rec["task_id"], []).append(rec)
    for item in items:
        last = by_task[item.task_id][-1]
        action = parse_action(last["target_text"])
        assert action["type"] == "answer"
        assert action["answer"] == item.gold_answer


def test_prompts_render_the_question_and_image_token(examples):
    items, records, _ = examples
    for rec in records:
        assert "<image>" in rec["prompt_text"]
        assert rec["prompt_text"].rstrip().endswith("ASSISTANT]") or "ASSISTANT" in rec["prompt_text"]


def test_token_budget_respected(examples):
    _, records, stats = examples
    assert stats["examples"] == len(records)
    for rec in records:
        assert rec["total_tokens"] <= 4096
        assert rec["n_target_tokens"] > 0


def test_build_sft_writes_jsonl(tmp_path):
    from vlmgrpo.data.build_sft import build_sft_split
    from vlmgrpo.tasks.generate import generate_all

    generate_all(
        seed=11,
        counts={"train": 4, "val": 4},
        n_per_family=1,
        image_dir=tmp_path / "images",
        data_dir=tmp_path,
    )
    rows, stats = build_sft_split(tmp_path, "train", StubProcessor(), max_length=2048, max_turns=3)
    path = tmp_path / "sft_train.jsonl"
    assert path.exists()
    on_disk = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert len(on_disk) == len(rows) == stats["examples"]
    assert all("prompt_text" in row and "target_text" in row for row in on_disk)
