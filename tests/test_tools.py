"""Unit tests for the tool registry and the sandboxed executors."""

from __future__ import annotations

import pytest

from vlmgrpo.tasks.families import generate_task
from vlmgrpo.tools import (
    TOOL_SCHEMAS,
    ToolSandbox,
    parse_tool_json,
    python_exec,
    retrieve,
    validate_call,
)


class TestPythonExec:
    def test_arithmetic(self):
        result = python_exec("print(round(sum([1,2,3]) / 3, 2))")
        assert result["ok"]
        assert result["stdout"] == "2.0"

    def test_allowlisted_import(self):
        result = python_exec("import statistics\nprint(statistics.mean([1,2,3]))")
        assert result["ok"]
        assert result["stdout"] == "2"

    @pytest.mark.parametrize(
        "code",
        [
            "import os\nprint(os.getcwd())",
            "from subprocess import run",
            "import socket",
            "from urllib.request import urlopen",
            "import sys",
        ],
    )
    def test_import_violations_blocked(self, code):
        result = python_exec(code)
        assert not result["ok"]
        assert "not allowed" in result["error"]

    @pytest.mark.parametrize(
        "code",
        [
            "print(open('/etc/passwd').read())",
            "print(eval('1+1'))",
            "print(__import__('os').getcwd())",
            "x = ().__class__.__bases__",
            "print(globals())",
        ],
    )
    def test_escape_attempts_blocked(self, code):
        result = python_exec(code)
        assert not result["ok"]
        assert result["error"]

    def test_timeout(self):
        result = python_exec("while True:\n    pass", timeout=0.4)
        assert not result["ok"]
        assert "timeout" in result["error"]

    def test_syntax_error(self):
        result = python_exec("print(")
        assert not result["ok"]
        assert "syntax" in result["error"]

    def test_runtime_error_is_reported(self):
        result = python_exec("print(1/0)")
        assert not result["ok"]
        assert "ZeroDivisionError" in result["error"]

    def test_cwd_is_isolated(self):
        result = python_exec("print(1 + 1)")
        assert result["ok"]
        assert result["stdout"] == "2"


class TestValidateCall:
    def test_all_schemas_validate(self):
        samples = {
            "python_exec": {"code": "print(1)"},
            "retrieve": {"query": "values"},
            "click": {"element_id": "card_1"},
            "read_field": {"name": "balance"},
        }
        assert set(samples) == set(TOOL_SCHEMAS)
        for name, args in samples.items():
            ok, err = validate_call(name, args)
            assert ok, (name, err)

    def test_rejections(self):
        assert not validate_call("nope", {})[0]
        assert not validate_call("python_exec", None)[0]
        assert not validate_call("python_exec", {"code": 1})[0]
        assert not validate_call("python_exec", {})[0]
        assert not validate_call("python_exec", {"code": "x", "junk": 1})[0]


class TestRetrieve:
    def test_returns_best_span(self):
        corpus = {"chart_values": "Mon=12, Tue=7, Wed=4", "chart_title": "Tickets"}
        result = retrieve("what are the chart values", corpus)
        assert result["ok"]
        assert "Mon=12" in result["stdout"]

    def test_empty_corpus(self):
        assert not retrieve("x", {})["ok"]


class TestUISandbox:
    def _item(self, tmp_path):
        return generate_task("screenshot_ui", 1234, "test", 0, tmp_path)

    def test_click_then_read(self, tmp_path):
        item = self._item(tmp_path)
        sandbox = ToolSandbox(item)
        click = sandbox.execute("click", {"element_id": "card_2"})
        assert click["ok"]
        read = sandbox.execute("read_field", {"name": "balance"})
        assert read["ok"]
        assert "balance=" in read["stdout"]

    def test_read_before_click_fails(self, tmp_path):
        item = self._item(tmp_path)
        sandbox = ToolSandbox(item)
        result = sandbox.execute("read_field", {"name": "balance"})
        assert not result["ok"]
        assert "click" in result["error"]

    def test_unknown_element_and_field(self, tmp_path):
        item = self._item(tmp_path)
        sandbox = ToolSandbox(item)
        assert not sandbox.execute("click", {"element_id": "nope"})["ok"]
        sandbox.execute("click", {"element_id": "card_1"})
        assert not sandbox.execute("read_field", {"name": "nope"})["ok"]

    def test_non_ui_task_has_no_widgets(self, tmp_path):
        item = generate_task("chart_agg", 1234, "test", 0, tmp_path)
        sandbox = ToolSandbox(item)
        assert not sandbox.execute("click", {"element_id": "card_1"})["ok"]


class TestParseToolJson:
    def test_plain(self):
        payload, err = parse_tool_json('{"name":"click","args":{"element_id":"card_1"}}')
        assert payload["name"] == "click" and not err

    def test_fence(self):
        payload, _ = parse_tool_json('```json\n{"name":"retrieve","args":{"query":"x"}}\n```')
        assert payload["name"] == "retrieve"

    def test_single_quotes(self):
        payload, _ = parse_tool_json("{'name': 'retrieve', 'args': {'query': 'x'}}")
        assert payload["name"] == "retrieve"

    def test_garbage(self):
        payload, err = parse_tool_json("I will not use a tool")
        assert payload is None and err
