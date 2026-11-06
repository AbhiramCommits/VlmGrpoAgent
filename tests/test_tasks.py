"""Task-suite determinism and structural invariants."""

from __future__ import annotations

import hashlib
import json

import pytest

from vlmgrpo.tasks.families import FAMILIES
from vlmgrpo.tasks.generate import generate_all, generate_suite, suite_stats
from vlmgrpo.tools import ToolSandbox
from vlmgrpo.verify import verify_answer


@pytest.fixture(scope="module")
def suite(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("suite")
    items = generate_suite(2, 1234, "test", image_dir=tmp / "images")
    return tmp, items


def test_all_families_present(suite):
    _, items = suite
    assert {i.family for i in items} == set(FAMILIES)
    for item in items:
        assert item.required_tools, item.task_id
        assert item.gold_answer != ""
        assert item.gold_perception_facts, item.task_id
        assert item.question


def test_deterministic_items_and_images(suite):
    tmp, items = suite
    again = generate_suite(2, 1234, "test", image_dir=tmp / "images2")
    assert [i.content_hash() for i in items] == [i.content_hash() for i in again]
    for a, b in zip(items, again):
        ha = hashlib.sha256(open(a.image_path, "rb").read()).hexdigest()
        hb = hashlib.sha256(open(b.image_path, "rb").read()).hexdigest()
        assert ha == hb, a.task_id


def test_different_seed_changes_tasks(suite):
    tmp, items = suite
    other = generate_suite(2, 999, "test", image_dir=tmp / "images3")
    assert [i.content_hash() for i in items] != [i.content_hash() for i in other]


def test_hard_split_is_harder(tmp_path):
    splits = generate_all(
        seed=5,
        counts={"train": 4, "val": 4, "test": 4, "test_hard": 4},
        n_per_family=1,
        image_dir=tmp_path / "images",
        data_dir=tmp_path,
    )
    test = splits["test"]
    hard = splits["test_hard"]
    assert sum(i.difficulty for i in hard) / len(hard) > sum(i.difficulty for i in test) / len(test)
    assert sum(len(i.gold_tool_calls) for i in hard) > sum(len(i.gold_tool_calls) for i in test)
    stats = suite_stats(test + hard)
    assert stats["test_hard"]["chart_agg"]["avg_tool_calls"] >= 2


def test_gold_traces_execute_to_gold_answers(suite):
    _, items = suite
    for item in items:
        if item.family == "screenshot_ui":
            # the final step is arithmetic the model must do; check the chain runs
            sandbox = ToolSandbox(item)
            for call in item.gold_tool_calls:
                assert sandbox.execute(call["name"], call["args"])["ok"], (item.task_id, call)
            continue
        sandbox = ToolSandbox(item)
        result = {}
        for call in item.gold_tool_calls:
            result = sandbox.execute(call["name"], call["args"])
            assert result["ok"], (item.task_id, call, result)
        assert verify_answer(result["stdout"], item.gold_answer), (item.task_id, result, item.gold_answer)


def test_generate_all_writes_splits_and_stats(tmp_path):
    splits = generate_all(
        seed=3,
        counts={"train": 4, "val": 4, "test": 4, "test_hard": 4},
        n_per_family=1,
        image_dir=tmp_path / "images",
        data_dir=tmp_path,
    )
    assert set(splits) == {"train", "val", "test", "test_hard"}
    for split in splits:
        path = tmp_path / f"{split}.jsonl"
        assert path.exists()
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        assert len(rows) == 4
    stats = (tmp_path / "suite_stats.md").read_text()
    assert "| split | family | count |" in stats
    assert "test_hard" in stats
