"""Programmatic verifiers: answers, tool schemas, traces, perception recall.

Everything here is a pure function of its inputs -- no model, no filesystem, no
network -- so the same checks that score rollouts also score SFT targets and
unit tests.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .tools import parse_tool_json, validate_call

_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    s = str(text).strip().lower()
    s = s.replace("\u2212", "-")
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" \t\n\r.,;:!?\"'`")
    return s


def extract_number(text: Any) -> float | None:
    """Return the last numeric literal in ``text`` (models end with the answer)."""
    if text is None:
        return None
    if isinstance(text, (int, float)) and not isinstance(text, bool):
        return float(text)
    matches = _NUM_RE.findall(str(text).replace("$", "").replace("%", ""))
    if not matches:
        return None
    try:
        return float(matches[-1].replace(",", ""))
    except ValueError:
        return None


def verify_answer(pred: Any, gold: Any, tol: float = 0.01) -> bool:
    """Numeric tolerance match, falling back to normalized string equality."""
    if pred is None:
        return False
    if isinstance(pred, (list, tuple)):
        return any(verify_answer(p, gold, tol) for p in pred)
    p_norm, g_norm = normalize_text(pred), normalize_text(gold)
    if not p_norm:
        return False
    if p_norm == g_norm:
        return True
    # boolean aliases
    aliases = {"true": "yes", "false": "no", "y": "yes", "n": "no"}
    if aliases.get(p_norm, p_norm) == aliases.get(g_norm, g_norm):
        return True
    p_num, g_num = extract_number(pred), extract_number(gold)
    if p_num is not None and g_num is not None:
        return abs(p_num - g_num) <= tol + 1e-9
    return False


def _call_payload(call: Any) -> tuple[dict[str, Any] | None, str]:
    """Normalize a call from a string, dict or trajectory turn."""
    if isinstance(call, str):
        return parse_tool_json(call)
    if not isinstance(call, dict):
        return None, f"call must be a dict, got {type(call).__name__}"
    if "tool" in call and "name" not in call:
        call = call["tool"]
        if isinstance(call, str):
            return parse_tool_json(call)
    name = call.get("name")
    args = call.get("args", {})
    if isinstance(args, str):
        parsed, err = parse_tool_json(args)
        if parsed is None:
            return None, err
        args = parsed
    return {"name": name, "args": args}, ""


def verify_tool_call_schema(call: Any) -> bool:
    """True iff the call parses, names a real tool, and has correct arg types."""
    payload, err = _call_payload(call)
    if payload is None:
        return False
    ok, _ = validate_call(payload.get("name"), payload.get("args"))
    return ok


def tool_trace_detail(calls: list[Any], item: Any) -> dict[str, Any]:
    """Greedy ordered match of ``item.required_tools`` against executed calls."""
    required = list(getattr(item, "required_tools", []) or [])
    normalized: list[dict[str, Any]] = []
    for call in calls or []:
        payload, _ = _call_payload(call)
        if payload is None:
            normalized.append({"name": None, "ok": False, "valid": False})
            continue
        ok_flag = call.get("ok", True) if isinstance(call, dict) else True
        normalized.append(
            {
                "name": payload["name"],
                "args": payload["args"],
                "ok": bool(ok_flag),
                "valid": verify_tool_call_schema(payload),
            }
        )
    if not required:
        ok = any(c["valid"] and c["ok"] for c in normalized)
        return {"ok": ok, "matched": [], "missing": [], "required": [], "ordered": ok}
    i = 0
    matched = []
    for call in normalized:
        if i >= len(required):
            break
        if call["name"] == required[i] and call["ok"] and call["valid"]:
            matched.append(required[i])
            i += 1
    return {
        "ok": i == len(required),
        "matched": matched,
        "missing": required[i:],
        "required": required,
        "ordered": True,
    }


def verify_tool_trace(calls: list[Any], item: Any) -> bool:
    """Did the agent execute the required tool sequence, in a feasible order?"""
    return bool(tool_trace_detail(calls, item)["ok"])


def _flatten_text(value: Any, depth: int = 0) -> str:
    if depth > 6 or value is None:
        return ""
    if isinstance(value, str):
        return value + "\n"
    if isinstance(value, (int, float, bool)):
        return f" {value} \n"
    if isinstance(value, dict):
        return "\n".join(_flatten_text(k, depth + 1) + _flatten_text(v, depth + 1) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return "\n".join(_flatten_text(v, depth + 1) for v in value)
    if hasattr(value, "to_dict"):
        return _flatten_text(value.to_dict(), depth + 1)
    return str(value) + "\n"


def calls_and_text_to_text(calls_and_text: Any) -> str:
    return _flatten_text(calls_and_text)


def _fact_present(value: Any, text: str, text_lower: str) -> bool:
    if isinstance(value, bool):
        return str(value).lower() in text_lower
    if isinstance(value, (int, float)):
        fval = float(value)
        variants = []
        if fval.is_integer():
            variants = [str(int(fval)), f"{fval:.2f}", f"{int(fval):,}"]
        else:
            variants = [str(value), f"{fval:.2f}", f"{fval:.4f}"]
        for v in variants:
            pattern = r"(?<![\d.])" + re.escape(v) + r"(?![\d]|\.\d)"
            if re.search(pattern, text):
                return True
        return False
    s = str(value)
    return bool(s) and s.lower() in text_lower


def perception_recall(calls_and_text: Any, gold_facts: dict[str, Any] | list[Any] | None) -> float:
    """Fraction of ground-truth visual facts surfaced anywhere in the trajectory."""
    if not gold_facts:
        return 1.0
    if isinstance(gold_facts, dict):
        facts = list(gold_facts.values())
    else:
        facts = list(gold_facts)
    if not facts:
        return 1.0
    text = calls_and_text_to_text(calls_and_text)
    text_lower = text.lower()
    hit = sum(1 for fact in facts if _fact_present(fact, text, text_lower))
    return round(hit / len(facts), 4)


def parse_answer_text(text: Any) -> str | None:
    """Extract the content of the last ``<answer>...</answer>`` block."""
    if text is None:
        return None
    matches = re.findall(r"<answer>(.*?)</answer>", str(text), flags=re.DOTALL | re.IGNORECASE)
    if not matches:
        return None
    return matches[-1].strip()


def format_ok(text: Any) -> bool:
    """Check the ReAct contract: a think block plus a parseable tool or answer."""
    if not text:
        return False
    s = str(text)
    has_think = bool(re.search(r"<think>.*?</think>", s, flags=re.DOTALL | re.IGNORECASE))
    has_answer = parse_answer_text(s) is not None
    has_tool = bool(re.search(r"<tool>.*?</tool>", s, flags=re.DOTALL | re.IGNORECASE))
    return has_think and (has_answer or has_tool)


def json_parses(text: Any) -> bool:
    try:
        json.loads(str(text))
        return True
    except Exception:
        return False
