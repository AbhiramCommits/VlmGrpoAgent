"""Handcrafted positive/negative cases for every programmatic verifier."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vlmgrpo.verify import (
    extract_number,
    format_ok,
    normalize_text,
    perception_recall,
    verify_answer,
    verify_tool_call_schema,
    verify_tool_trace,
)


class TestVerifyAnswer:
    @pytest.mark.parametrize(
        "pred,gold",
        [
            ("16.00", "16.00"),
            ("16", "16.00"),
            ("$16.0", "16.00"),
            ("The mean is 16.0", "16.00"),
            ("24.674", "24.67"),
            ("  24.67  ", "24.67"),
            ("YES", "yes"),
            ("true", "yes"),
        ],
    )
    def test_accepts(self, pred, gold):
        assert verify_answer(pred, gold)

    @pytest.mark.parametrize(
        "pred,gold",
        [
            (None, "16.00"),
            ("", "16.00"),
            ("17.00", "16.00"),
            ("16.5", "16.00"),
            ("no", "yes"),
            ("banana", "16.00"),
        ],
    )
    def test_rejects(self, pred, gold):
        assert not verify_answer(pred, gold)

    def test_tolerance_is_configurable(self):
        assert verify_answer("16.4", "16.0", tol=0.5)
        assert not verify_answer("16.4", "16.0", tol=0.1)

    def test_non_numeric_strings(self):
        assert verify_answer("North", "north")
        assert not verify_answer("South", "North")


class TestToolSchema:
    def test_valid_python_exec(self):
        call = {"name": "python_exec", "args": {"code": "print(1)"}}
        assert verify_tool_call_schema(call)

    def test_valid_from_json_string(self):
        assert verify_tool_call_schema('{"name": "retrieve", "args": {"query": "values"}}')

    def test_json_in_code_fence(self):
        assert verify_tool_call_schema('```json\n{"name": "click", "args": {"element_id": "card_1"}}\n```')

    @pytest.mark.parametrize(
        "call",
        [
            None,
            "not json at all",
            "{}",
            {"name": "made_up_tool", "args": {}},
            {"name": "python_exec", "args": {}},
            {"name": "python_exec", "args": {"code": 42}},
            {"name": "python_exec", "args": {"code": "print(1)", "bogus": 1}},
            {"name": "click", "args": {"code": "x"}},
            {"args": {"code": "print(1)"}},
            ["python_exec", {"code": "print(1)"}],
            {"name": "retrieve", "args": "not-an-object"},
        ],
    )
    def test_invalid(self, call):
        assert not verify_tool_call_schema(call)


class TestToolTrace:
    def _item(self, required):
        return SimpleNamespace(required_tools=required)

    def test_click_read_then_python(self):
        item = self._item(["click", "read_field", "read_field", "python_exec"])
        calls = [
            {"name": "click", "args": {"element_id": "card_2"}, "ok": True},
            {"name": "read_field", "args": {"name": "balance"}, "ok": True},
            {"name": "read_field", "args": {"name": "pending"}, "ok": True},
            {"name": "python_exec", "args": {"code": "print(1)"}, "ok": True},
        ]
        assert verify_tool_trace(calls, item)

    def test_order_matters(self):
        item = self._item(["click", "read_field"])
        calls = [
            {"name": "read_field", "args": {"name": "balance"}, "ok": True},
            {"name": "click", "args": {"element_id": "card_2"}, "ok": True},
        ]
        assert not verify_tool_trace(calls, item)

    def test_failed_call_does_not_count(self):
        item = self._item(["click", "read_field"])
        calls = [
            {"name": "click", "args": {"element_id": "nope"}, "ok": False},
            {"name": "read_field", "args": {"name": "balance"}, "ok": False},
        ]
        assert not verify_tool_trace(calls, item)

    def test_extra_calls_between_are_allowed(self):
        item = self._item(["click", "python_exec"])
        calls = [
            {"name": "click", "args": {"element_id": "card_1"}, "ok": True},
            {"name": "retrieve", "args": {"query": "anything"}, "ok": True},
            {"name": "python_exec", "args": {"code": "print(3)"}, "ok": True},
        ]
        assert verify_tool_trace(calls, item)

    def test_missing_required(self):
        item = self._item(["python_exec", "python_exec"])
        calls = [{"name": "python_exec", "args": {"code": "print(1)"}, "ok": True}]
        assert not verify_tool_trace(calls, item)

    def test_malformed_call_does_not_count(self):
        item = self._item(["python_exec"])
        calls = [{"name": "python_exec", "args": {"nope": 1}, "ok": True}]
        assert not verify_tool_trace(calls, item)


class TestPerceptionRecall:
    def test_full_recall(self):
        facts = {"bar:Mon": 12, "bar:Tue": 7, "bar:Wed": 4}
        text = "values Mon=12, Tue=7, Wed=4 observed"
        assert perception_recall(text, facts) == 1.0

    def test_partial_recall_from_tool_args(self):
        facts = {"a": 12, "b": 7, "c": 4}
        calls = [{"name": "python_exec", "args": {"code": "vals = [12, 7]\nprint(sum(vals))"}}]
        assert perception_recall(calls, facts) == pytest.approx(2 / 3, abs=1e-4)

    def test_zero_recall(self):
        facts = {"a": 12, "b": 7}
        assert perception_recall("I cannot read the chart", facts) == 0.0

    def test_substring_numbers_do_not_count(self):
        facts = {"a": 12}
        assert perception_recall("the year 2012 and 112 items", facts) == 0.0

    def test_floats_accept_two_decimals(self):
        assert perception_recall("balance 544.92 dollars", {"b": 544.92}) == 1.0

    def test_empty_facts(self):
        assert perception_recall("anything", {}) == 1.0

    def test_observations_are_included(self):
        facts = {"card": 544.92}
        traj = {"turns": [{"tool_result": {"stdout": "balance=544.92"}}]}
        assert perception_recall(traj, facts) == 1.0


class TestSmallHelpers:
    def test_extract_number_last(self):
        assert extract_number("first 3 then 16.00") == pytest.approx(16.0)
        assert extract_number("no digits") is None
        assert extract_number("1,234.50") == pytest.approx(1234.5)

    def test_normalize(self):
        assert normalize_text("  Hello,   World!! ") == "hello, world"

    def test_format_ok(self):
        assert format_ok("<think>x</think><answer>1</answer>")
        assert format_ok('<think>x</think><tool>{"name":"python_exec","args":{"code":"print(1)"}}</tool>')
        assert not format_ok("<answer>1</answer>")
        assert not format_ok("<think>x</think>")
        assert not format_ok("")
