"""Prompt construction for the multi-turn ReAct agent.

The exact same text is used at rollout time, SFT time and GRPO time so the
distribution never shifts between stages.
"""

from __future__ import annotations

import json
from typing import Any

SYSTEM_PROMPT = (
    "You are a careful multimodal research assistant. You are shown an image "
    "and must answer a question about it. You may use tools in a loop.\n"
    "Tools:\n"
    '  python_exec(code): run sandboxed Python (no imports beyond math, '
    'statistics, json, re, itertools, functools, collections, decimal).\n'
    '  retrieve(query): search the text spans rendered in the image.\n'
    "  click(element_id): click a UI widget in a screenshot.\n"
    "  read_field(name): read a named field from a screenshot.\n"
    "Reply with exactly this format each turn:\n"
    "<think>short reasoning</think>\n"
    "<tool>{\"name\": \"tool_name\", \"args\": {...}}</tool>\n"
    "When you know the answer, reply:\n"
    "<think>short reasoning</think>\n"
    "<answer>FINAL ANSWER</answer>\n"
    "Use at least one tool before answering. Keep answers short and numeric "
    "answers rounded to 2 decimals unless stated otherwise."
)

TURN_REMINDER = (
    "Continue. Emit <think>...</think> then either one <tool>...</tool> call "
    "or <answer>...</answer>."
)


def build_messages(question: str, history: list[dict[str, str]] | None = None) -> list[dict[str, Any]]:
    """Build the chat messages for the current turn.

    ``history`` is a list of ``{"role": "assistant"|"user", "content": str}``
    items holding previous assistant outputs and tool observations.
    """
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]},
    ]
    for msg in history or []:
        messages.append({"role": msg["role"], "content": [{"type": "text", "text": msg["content"]}]})
    return messages


def format_prompt(processor: Any, messages: list[dict[str, Any]], add_generation_prompt: bool = True) -> str:
    """Apply the model chat template (stub processor implements the same API)."""
    return processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt
    )


def render_history_for_sft(history: list[dict[str, str]]) -> str:
    """Render tool observations compactly (used in SFT target construction)."""
    out = []
    for msg in history:
        role = msg["role"]
        out.append(f"{role}: {msg['content']}")
    return "\n".join(out)


def observation_text(name: str, args: dict[str, Any], result: dict[str, Any]) -> str:
    """The user-visible observation fed back after a tool call."""
    payload = {
        "tool": name,
        "ok": bool(result.get("ok")),
        "stdout": str(result.get("stdout", ""))[:800],
        "error": str(result.get("error", ""))[:400] if result.get("error") else "",
    }
    return "OBSERVATION " + json.dumps(payload, ensure_ascii=False)
