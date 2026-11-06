"""Tool registry and sandboxed executors.

Tools are deterministic and unit-testable.  ``python_exec`` runs in a fresh
subprocess with an import allowlist, a restricted builtin namespace, an audit
hook that refuses network/subprocess events, a throw-away cwd and a hard
timeout.  ``retrieve`` / ``click`` / ``read_field`` read only from the current
task item, so no external state leaks between episodes.
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

# name -> {"description", "required": {arg: type}, "optional": {arg: type}}
TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "python_exec": {
        "description": "Run sandboxed Python (allowlisted imports only, 2s timeout) and return stdout.",
        "required": {"code": "string"},
        "optional": {},
    },
    "retrieve": {
        "description": "Search the text spans rendered in the task image.",
        "required": {"query": "string"},
        "optional": {},
    },
    "click": {
        "description": "Click a UI element by element_id in a screenshot task.",
        "required": {"element_id": "string"},
        "optional": {},
    },
    "read_field": {
        "description": "Read a named field from the currently opened UI card.",
        "required": {"name": "string"},
        "optional": {},
    },
}

TOOL_NAMES = tuple(TOOL_SCHEMAS)

ALLOWED_IMPORTS = {
    "math",
    "statistics",
    "json",
    "re",
    "itertools",
    "functools",
    "collections",
    "decimal",
    "fractions",
    "string",
    "operator",
}

BLOCKED_NAMES = {
    "open",
    "eval",
    "exec",
    "compile",
    "input",
    "globals",
    "locals",
    "vars",
    "dir",
    "getattr",
    "setattr",
    "delattr",
    "breakpoint",
    "memoryview",
    "help",
    "exit",
    "quit",
    "importlib",
    "__import__",
    "__builtins__",
}

BLOCKED_ATTRS = {
    "system",
    "popen",
    "spawn",
    "fork",
    "execv",
    "execve",
    "remove",
    "unlink",
    "rmdir",
    "rmtree",
    "chmod",
    "connect",
    "socket",
}

MAX_CODE_CHARS = 4000
MAX_STDOUT_CHARS = 4000

_RUNNER = r"""
import sys, json, builtins, traceback

code = sys.stdin.read()
if len(code) > 4000:
    print("code too long", file=sys.stderr)
    sys.exit(2)

_BLOCKED_EVENTS = (
    "socket.", "urllib.", "http.client.", "ftplib.", "smtplib.",
    "subprocess.", "os.system", "os.exec", "os.spawn", "os.fork",
    "ctypes.dlopen", "ctypes.call_function", "webbrowser.open",
)

def _hook(event, args):
    for prefix in _BLOCKED_EVENTS:
        if event.startswith(prefix):
            raise RuntimeError("sandbox blocked event: " + event)

sys.addaudithook(_hook)

_SAFE_NAMES = [
    "abs", "all", "any", "ascii", "bin", "bool", "bytearray", "bytes", "callable",
    "chr", "classmethod", "complex", "dict", "divmod", "enumerate", "filter",
    "float", "format", "frozenset", "hash", "hex", "int", "isinstance",
    "issubclass", "iter", "len", "list", "map", "max", "min", "next", "object",
    "oct", "ord", "pow", "print", "property", "range", "repr", "reversed",
    "round", "set", "slice", "sorted", "staticmethod", "str", "sum", "super",
    "tuple", "type", "zip",
    "Exception", "ArithmeticError", "AssertionError", "AttributeError",
    "IndexError", "KeyError", "LookupError", "NameError", "RuntimeError",
    "StopIteration", "TypeError", "ValueError", "ZeroDivisionError",
    "True", "False", "None",
]
_SAFE = {name: getattr(builtins, name) for name in _SAFE_NAMES if hasattr(builtins, name)}
_SAFE["__build_class__"] = builtins.__build_class__
_SAFE["__import__"] = builtins.__import__   # import statements only; AST blocks direct use

_globals = {"__builtins__": _SAFE, "__name__": "__main__", "__doc__": None}
try:
    exec(compile(code, "<agent>", "exec"), _globals)
except SystemExit:
    raise
except Exception:
    traceback.print_exc()
    sys.exit(1)
"""


def tool_schemas_text() -> str:
    return json.dumps({k: {"description": v["description"], "required": v["required"]} for k, v in TOOL_SCHEMAS.items()}, indent=2)


def validate_call(name: Any, args: Any) -> tuple[bool, str]:
    """Validate a tool name + argument object against the registry."""
    if not isinstance(name, str) or name not in TOOL_SCHEMAS:
        return False, f"unknown tool: {name!r}"
    if not isinstance(args, dict):
        return False, "args must be an object"
    schema = TOOL_SCHEMAS[name]
    for arg, expected in schema["required"].items():
        if arg not in args:
            return False, f"missing required arg: {arg}"
        if expected == "string" and not isinstance(args[arg], str):
            return False, f"arg {arg} must be a string"
        if expected == "number" and not isinstance(args[arg], (int, float)):
            return False, f"arg {arg} must be a number"
    for arg in args:
        if arg not in schema["required"] and arg not in schema["optional"]:
            return False, f"unexpected arg: {arg}"
    return True, ""


def _check_code(code: str) -> str | None:
    """Static safety check; returns an error string or ``None``."""
    if not isinstance(code, str):
        return "code must be a string"
    if len(code) > MAX_CODE_CHARS:
        return f"code longer than {MAX_CODE_CHARS} chars"
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        return f"syntax error: {exc.msg} (line {exc.lineno})"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in ALLOWED_IMPORTS:
                    return f"import not allowed: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                return "relative imports are not allowed"
            root = (node.module or "").split(".")[0]
            if root not in ALLOWED_IMPORTS:
                return f"import not allowed: {node.module}"
        elif isinstance(node, ast.Name):
            if node.id in BLOCKED_NAMES or node.id.startswith("_"):
                return f"name not allowed: {node.id}"
        elif isinstance(node, ast.Attribute):
            if node.attr in BLOCKED_ATTRS or node.attr.startswith("_"):
                return f"attribute not allowed: {node.attr}"
    return None


def python_exec(code: str, timeout: float = 2.0) -> dict[str, Any]:
    """Execute ``code`` in an isolated subprocess."""
    err = _check_code(code)
    if err:
        return {"ok": False, "stdout": "", "error": err}
    with tempfile.TemporaryDirectory(prefix="vlmgrpo_sbx_") as tmp:
        env = {
            "PATH": "/usr/bin:/bin",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONHASHSEED": "0",
            "HOME": tmp,
            "TMPDIR": tmp,
            "http_proxy": "",
            "https_proxy": "",
            "NO_PROXY": "*",
            "TOKENIZERS_PARALLELISM": "false",
        }
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", _RUNNER],
                input=code,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tmp,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "stdout": "", "error": f"timeout after {timeout}s"}
        except Exception as exc:  # pragma: no cover - very unusual
            return {"ok": False, "stdout": "", "error": f"executor failure: {exc}"}
    stdout = (proc.stdout or "")[:MAX_STDOUT_CHARS].strip()
    stderr = (proc.stderr or "")[:MAX_STDOUT_CHARS].strip()
    return {"ok": proc.returncode == 0, "stdout": stdout, "error": "" if proc.returncode == 0 else stderr or "nonzero exit"}


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-zA-Z0-9.]+", str(text).lower()) if len(t) > 1}


def retrieve(query: str, corpus: dict[str, str]) -> dict[str, Any]:
    """Rank the task's rendered text spans against ``query`` and return the best."""
    if not corpus:
        return {"ok": False, "stdout": "", "error": "this task has no retrievable spans"}
    q = _tokens(query)
    scored = []
    for key, value in corpus.items():
        overlap = len(q & _tokens(key + " " + value))
        scored.append((overlap, key, value))
    scored.sort(key=lambda x: (-x[0], x[1]))
    selected = [f"[{k}] {v}" for score, k, v in scored[:2] if score >= 0]
    if not q or all(s[0] == 0 for s in scored):
        selected = [f"[{k}] {v}" for _, k, v in scored[:1]]
    text = "\n".join(selected)[:1200]
    return {"ok": True, "stdout": text, "error": ""}


class ToolSandbox:
    """Per-episode tool executor with UI state."""

    def __init__(self, item: Any, timeout: float = 2.0) -> None:
        self.item = item
        self.timeout = timeout
        self.opened: str | None = None
        self.calls: list[dict[str, Any]] = []

    def reset(self) -> None:
        self.opened = None
        self.calls = []

    # -- individual executors ------------------------------------------------ #
    def _ui(self) -> dict[str, Any]:
        return (getattr(self.item, "metadata", {}) or {}).get("ui", {}) or {}

    def click(self, element_id: str) -> dict[str, Any]:
        ui = self._ui()
        if not ui:
            return {"ok": False, "stdout": "", "error": "no UI in this task"}
        cards = ui.get("cards", {})
        if element_id not in cards:
            return {"ok": False, "stdout": "", "error": f"unknown element_id: {element_id}"}
        self.opened = element_id
        card = cards[element_id]
        return {
            "ok": True,
            "stdout": f"opened {element_id} ({card.get('label', 'card')}); fields: balance, pending",
            "error": "",
        }

    def read_field(self, name: str) -> dict[str, Any]:
        ui = self._ui()
        if not ui:
            return {"ok": False, "stdout": "", "error": "no UI in this task"}
        if self.opened is None:
            return {"ok": False, "stdout": "", "error": "no open card; call click(element_id) first"}
        card = ui.get("cards", {}).get(self.opened, {})
        if name not in card:
            return {"ok": False, "stdout": "", "error": f"unknown field: {name}"}
        value = card[name]
        if isinstance(value, (int, float)):
            return {"ok": True, "stdout": f"{name}={value:.2f}", "error": ""}
        return {"ok": True, "stdout": f"{name}={value}", "error": ""}

    # -- dispatch ------------------------------------------------------------ #
    def execute(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        ok, err = validate_call(name, args)
        if not ok:
            result = {"ok": False, "stdout": "", "error": err}
        elif name == "python_exec":
            result = python_exec(args["code"], timeout=self.timeout)
        elif name == "retrieve":
            result = retrieve(args["query"], getattr(self.item, "retrieval_corpus", {}) or {})
        elif name == "click":
            result = self.click(args["element_id"])
        elif name == "read_field":
            result = self.read_field(args["name"])
        else:  # pragma: no cover - validate_call guards this
            result = {"ok": False, "stdout": "", "error": f"unimplemented tool {name}"}
        self.calls.append(
            {
                "name": name,
                "args": args,
                "ok": bool(result.get("ok")),
                "stdout": result.get("stdout", "")[:1200],
                "error": result.get("error", "")[:600],
            }
        )
        return result


def execute_tool(item: Any, name: str, args: dict[str, Any], sandbox: ToolSandbox | None = None) -> dict[str, Any]:
    """Convenience wrapper used by tests and the SFT trajectory builder."""
    sandbox = sandbox or ToolSandbox(item)
    return sandbox.execute(name, args)


def tool_names() -> list[str]:
    return list(TOOL_NAMES)


def parse_tool_json(payload: str) -> tuple[dict[str, Any] | None, str]:
    """Parse a JSON tool payload tolerantly (fences, single quotes, trailing text)."""
    if payload is None:
        return None, "empty payload"
    text = payload.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj, ""
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        candidate = match.group(0)
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj, ""
        except json.JSONDecodeError:
            candidate2 = candidate.replace("'", '"')
            try:
                obj = json.loads(candidate2)
                if isinstance(obj, dict):
                    return obj, ""
            except json.JSONDecodeError:
                pass
    return None, "tool payload is not valid JSON"


def is_safe_path(path: str | Path) -> bool:
    """Guard used by tests: absolute paths outside cwd are never executed on."""
    return not os.path.isabs(str(path))
