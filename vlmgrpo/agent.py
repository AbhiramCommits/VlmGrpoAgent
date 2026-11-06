"""Multi-turn ReAct agent over a vision-language model, with batched rollouts.

The agent emits ``<think>...</think>`` followed by either one
``<tool>{"name": ..., "args": {...}}</tool>`` call or an ``<answer>...</answer>``.
Tool results are fed back as observations until the answer is produced or the
turn budget is exhausted.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from .prompts import TURN_REMINDER, build_messages, format_prompt, observation_text
from .tools import ToolSandbox, parse_tool_json
from .verify import format_ok, parse_answer_text, verify_tool_call_schema

# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


def parse_action(text: str | None) -> dict[str, Any]:
    """Tolerant parser for one assistant turn."""
    raw = text or ""
    answer = parse_answer_text(raw)
    if answer is not None:
        return {"type": "answer", "answer": answer, "raw": raw}
    tools = re.findall(r"<tool>(.*?)</tool>", raw, flags=re.DOTALL | re.IGNORECASE)
    if tools:
        payload, err = parse_tool_json(tools[-1])
        if payload is None:
            return {"type": "tool_invalid", "error": err, "raw": raw}
        name = payload.get("name")
        args = payload.get("args", {})
        return {"type": "tool", "name": name, "args": args if isinstance(args, dict) else {},
                "schema_valid": verify_tool_call_schema({"name": name, "args": args}), "raw": raw}
    return {"type": "none", "raw": raw}


# --------------------------------------------------------------------------- #
# Trajectory records
# --------------------------------------------------------------------------- #


@dataclass
class TurnRecord:
    turn: int
    raw_text: str
    action_type: str
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    schema_valid: bool = False
    tool_ok: bool | None = None
    tool_stdout: str = ""
    tool_error: str = ""
    observation: str = ""
    gen_seconds: float = 0.0
    tool_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TurnRecord:
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in payload.items() if k in known})


@dataclass
class Trajectory:
    task_id: str
    family: str
    split: str
    question: str
    gold_answer: str
    pred_answer: str | None = None
    turns: list[TurnRecord] = field(default_factory=list)
    stopped_reason: str = "unknown"
    required_tools: list[str] = field(default_factory=list)
    gold_perception_facts: dict[str, Any] = field(default_factory=dict)
    image_paths: list[str] = field(default_factory=list)
    prompt: str = ""
    duration_seconds: float = 0.0
    backend: str = "hf"

    # -- derived views ------------------------------------------------------ #
    @property
    def tool_calls(self) -> list[dict[str, Any]]:
        calls = []
        for turn in self.turns:
            if turn.action_type in ("tool", "tool_invalid"):
                calls.append(
                    {
                        "name": turn.tool_name,
                        "args": turn.tool_args or {},
                        "ok": bool(turn.tool_ok),
                        "schema_valid": turn.schema_valid,
                        "stdout": turn.tool_stdout,
                        "error": turn.tool_error,
                    }
                )
        return calls

    @property
    def n_turns(self) -> int:
        return len(self.turns)

    @property
    def total_assistant_chars(self) -> int:
        return sum(len(t.raw_text) for t in self.turns)

    @property
    def format_valid(self) -> bool:
        return bool(self.turns) and format_ok(self.turns[-1].raw_text) and self.pred_answer is not None

    def perception_text(self) -> str:
        """Assistant text + tool arguments + tool observations."""
        parts: list[str] = []
        for turn in self.turns:
            parts.append(turn.raw_text)
            if turn.tool_args:
                parts.append(json.dumps(turn.tool_args, ensure_ascii=False))
            if turn.tool_stdout:
                parts.append(turn.tool_stdout)
        return "\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["turns"] = [t.to_dict() for t in self.turns]
        payload["n_turns"] = self.n_turns
        payload["pred_answer"] = self.pred_answer
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Trajectory:
        payload = dict(payload)
        turns = [TurnRecord.from_dict(t) for t in payload.pop("turns", [])]
        payload.pop("n_turns", None)
        known = cls.__dataclass_fields__
        traj = cls(**{k: v for k, v in payload.items() if k in known})
        traj.turns = turns
        return traj


# --------------------------------------------------------------------------- #
# Generation backends
# --------------------------------------------------------------------------- #


class HFGenerator:
    """Batched ``model.generate`` on the loaded HF (or stub) model."""

    name = "hf"

    def __init__(self, model: Any, processor: Any, device: str) -> None:
        self.model = model
        self.processor = processor
        self.device = device
        self.tokenizer = getattr(processor, "tokenizer", processor)

    @torch.no_grad()
    def generate(
        self,
        prompts: list[str],
        images: list[Image.Image],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        do_sample: bool,
    ) -> list[str]:
        inputs = self.processor(text=prompts, images=images, return_tensors="pt", padding=True)
        inputs = {
            k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in inputs.items()
        }
        if "input_ids" in inputs and inputs["input_ids"].shape[0] > 1:
            # Idefics3/SmolVLM yields NaN logits for left-padded rows (their
            # queries attend to nothing).  Treat the pad prefix as ordinary
            # context instead; generation still starts after the real prompt.
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
        prompt_len = inputs["input_ids"].shape[1]
        kwargs: dict[str, Any] = {
            "max_new_tokens": int(max_new_tokens),
            "pad_token_id": getattr(self.tokenizer, "pad_token_id", 0) or 0,
            "eos_token_id": getattr(self.tokenizer, "eos_token_id", None),
        }
        if do_sample:
            kwargs.update({"do_sample": True, "temperature": float(temperature), "top_p": float(top_p)})
        else:
            kwargs.update({"do_sample": False})
        out = self.model.generate(**inputs, **kwargs)
        new_tokens = out[:, prompt_len:]
        return self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)


class VLLMGenerator:
    """Optional offline vLLM backend (CUDA only; requires ``vllm`` installed).

    Multi-turn rollouts call this once per turn with the full rendered prompt
    and the single task image.  When vLLM cannot initialise, :func:`make_generator`
    logs and falls back to the HF backend -- the fallback is recorded in the
    results environment block.
    """

    name = "vllm"

    def __init__(self, tier: str, dtype: str) -> None:  # pragma: no cover - needs CUDA
        from vllm import LLM  # import guard

        from .registry import TIERS

        self.llm = LLM(
            model=TIERS[tier]["hf_id"],
            dtype=None if dtype in ("auto", None) else dtype,
            trust_remote_code=True,
            limit_mm_per_prompt={"image": 1},
        )

    def generate(  # pragma: no cover - needs CUDA
        self,
        prompts: list[str],
        images: list[Image.Image],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        do_sample: bool,
    ) -> list[str]:
        from vllm import SamplingParams

        params = SamplingParams(
            temperature=float(temperature) if do_sample else 0.0,
            top_p=float(top_p),
            max_tokens=int(max_new_tokens),
        )
        requests = [{"prompt": p, "multi_modal_data": {"image": im}} for p, im in zip(prompts, images)]
        outputs = self.llm.generate(requests, params)
        return [o.outputs[0].text for o in outputs]


def vllm_available() -> bool:
    try:
        import vllm  # noqa: F401

        return True
    except Exception:
        return False


def make_generator(model: Any, processor: Any, cfg: Any, tier: str, device: str) -> Any:
    backend = getattr(cfg.rollout, "backend", "hf")
    if backend == "vllm":
        if device != "cuda":
            print("[agent] vllm backend requires CUDA; falling back to hf")
        elif not vllm_available():
            print("[agent] vllm is not installed; falling back to hf")
        else:  # pragma: no cover - CUDA only
            try:
                return VLLMGenerator(tier, cfg.dtype)
            except Exception as exc:
                print(f"[agent] vllm init failed ({exc}); falling back to hf")
    return HFGenerator(model, processor, device)


# --------------------------------------------------------------------------- #
# Rollout loop
# --------------------------------------------------------------------------- #


def _chunks(seq: list[int], size: int) -> list[list[int]]:
    size = max(1, int(size))
    return [seq[i : i + size] for i in range(0, len(seq), size)]


def _open_image(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def run_episodes(
    model: Any,
    processor: Any,
    items: list[Any],
    cfg: Any,
    generator: Any = None,
    tier: str = "stub",
    device: str = "cpu",
) -> list[Trajectory]:
    """Run one episode per item, batching generation across items at each turn.

    Duplicate items are allowed and give independent samples -- this is how
    pass@k and GRPO groups are produced.
    """
    if not items:
        return []
    gen = generator or make_generator(model, processor, cfg, tier, device)
    rc = cfg.rollout
    n = len(items)
    states = [
        {
            "history": [],
            "done": False,
            "answer": None,
            "reason": "turn_limit",
            "turns": [],
            "sandbox": ToolSandbox(item, timeout=rc.tool_timeout),
            "image": _open_image(item.image_paths[0]),
            "prompt": "",
            "gen_seconds": 0.0,
            "tool_seconds": 0.0,
        }
        for item in items
    ]
    wall_start = time.perf_counter()

    for turn_index in range(int(rc.max_turns)):
        active = [i for i in range(n) if not states[i]["done"]]
        if not active:
            break
        for chunk in _chunks(active, rc.batch_size):
            prompts = []
            images = []
            for i in chunk:
                item = items[i]
                messages = build_messages(item.question, states[i]["history"])
                prompt = format_prompt(processor, messages)
                if turn_index == 0:
                    states[i]["prompt"] = prompt
                prompts.append(prompt)
                images.append(states[i]["image"])

            t0 = time.perf_counter()
            try:
                texts = gen.generate(
                    prompts,
                    images,
                    max_new_tokens=rc.max_new_tokens,
                    temperature=rc.temperature,
                    top_p=rc.top_p,
                    do_sample=rc.do_sample,
                )
            except Exception as exc:  # pragma: no cover - defensive
                texts = ["<think>generation error</think>" for _ in chunk]
                print(f"[agent] generation failed: {exc}")
            gen_seconds = time.perf_counter() - t0
            per_item_gen = gen_seconds / max(1, len(chunk))

            for i, text in zip(chunk, texts):
                item = items[i]
                state = states[i]
                state["gen_seconds"] += per_item_gen
                action = parse_action(text)
                record = TurnRecord(
                    turn=turn_index,
                    raw_text=text,
                    action_type=action["type"],
                    schema_valid=bool(action.get("schema_valid", False)),
                    gen_seconds=round(per_item_gen, 4),
                )

                if action["type"] == "answer":
                    state["answer"] = action["answer"]
                    state["done"] = True
                    state["reason"] = "answer"
                elif action["type"] in ("tool", "tool_invalid"):
                    name = action.get("name")
                    args = action.get("args", {}) or {}
                    record.tool_name = name if isinstance(name, str) else None
                    record.tool_args = args
                    record.schema_valid = bool(action.get("schema_valid", False))
                    t_tool = time.perf_counter()
                    result = state["sandbox"].execute(record.tool_name or "", args)
                    record.tool_seconds = round(time.perf_counter() - t_tool, 4)
                    state["tool_seconds"] += record.tool_seconds
                    record.tool_ok = bool(result.get("ok"))
                    record.tool_stdout = str(result.get("stdout", ""))[:1500]
                    record.tool_error = str(result.get("error", ""))[:600]
                    observation = observation_text(record.tool_name or "invalid", args, result)
                    record.observation = observation
                    state["history"].append({"role": "assistant", "content": text})
                    state["history"].append({"role": "user", "content": observation})
                    if turn_index == int(rc.max_turns) - 1:
                        state["done"] = True
                        state["reason"] = "turn_limit"
                else:
                    state["history"].append({"role": "assistant", "content": text})
                    state["history"].append(
                        {
                            "role": "user",
                            "content": "Your previous reply had no parseable <tool> or <answer> block. " + TURN_REMINDER,
                        }
                    )
                    if turn_index == int(rc.max_turns) - 1:
                        state["done"] = True
                        state["reason"] = "turn_limit"
                state["turns"].append(record)

    wall = time.perf_counter() - wall_start
    trajectories: list[Trajectory] = []
    per_episode = wall / max(1, n)
    for i, item in enumerate(items):
        state = states[i]
        trajectories.append(
            Trajectory(
                task_id=item.task_id,
                family=item.family,
                split=getattr(item, "split", "unknown"),
                question=item.question,
                gold_answer=item.gold_answer,
                pred_answer=state["answer"],
                turns=state["turns"],
                stopped_reason=state["reason"] if state["answer"] is not None else (
                    "turn_limit" if state["turns"] else "no_turns"
                ),
                required_tools=list(getattr(item, "required_tools", [])),
                gold_perception_facts=dict(getattr(item, "gold_perception_facts", {})),
                image_paths=list(getattr(item, "image_paths", [])),
                prompt=state["prompt"],
                duration_seconds=round(per_episode, 4),
                backend=getattr(gen, "name", "hf"),
            )
        )
    return trajectories


def run_episode(
    model: Any,
    processor: Any,
    item: Any,
    cfg: Any,
    generator: Any = None,
    tier: str = "stub",
    device: str = "cpu",
) -> Trajectory:
    """Single-episode convenience wrapper."""
    return run_episodes(model, processor, [item], cfg, generator=generator, tier=tier, device=device)[0]
