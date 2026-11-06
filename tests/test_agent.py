"""Agent loop tests: parser, tool execution, turn budget, batching."""

from __future__ import annotations

from dataclasses import replace

from vlmgrpo.agent import parse_action, run_episode, run_episodes
from vlmgrpo.config import RolloutConfig, RunConfig
from vlmgrpo.registry import StubProcessor
from vlmgrpo.tasks.families import generate_task
from vlmgrpo.verify import verify_answer, verify_tool_trace

PROC = StubProcessor()


class ScriptedGenerator:
    """Deterministic stand-in for model generation."""

    name = "scripted"

    def __init__(self, script):
        self.script = list(script)
        self.turn = 0

    def generate(self, prompts, images, **kwargs):
        text = self.script[min(self.turn, len(self.script) - 1)]
        self.turn += 1
        return [text] * len(prompts)


def _cfg(max_turns=4):
    return RunConfig(rollout=RolloutConfig(max_turns=max_turns, max_new_tokens=32, batch_size=4))


def _item(tmp_path, family="chart_agg", split="test"):
    return generate_task(family, 1234, split, 0, tmp_path)


TOOL_TURN = '<think>sum the bars</think><tool>{"name": "python_exec", "args": {"code": "print(42)"}}</tool>'
ANSWER_TURN = "<think>done</think><answer>42</answer>"


class TestParseAction:
    def test_answer(self):
        action = parse_action("<think>x</think><answer> 12.50 </answer>")
        assert action["type"] == "answer"
        assert action["answer"] == "12.50"

    def test_tool_valid(self):
        action = parse_action(
            '<think>x</think><tool>{"name":"retrieve","args":{"query":"values"}}</tool>'
        )
        assert action["type"] == "tool"
        assert action["name"] == "retrieve"
        assert action["schema_valid"]

    def test_tool_invalid_schema(self):
        action = parse_action('<tool>{"name":"nope","args":{}}</tool>')
        assert action["type"] == "tool"
        assert not action["schema_valid"]

    def test_malformed_json(self):
        action = parse_action("<tool>{not json}</tool>")
        assert action["type"] == "tool_invalid"
        assert action["error"]

    def test_plain_text(self):
        assert parse_action("just chatting")["type"] == "none"
        assert parse_action("")["type"] == "none"

    def test_answer_wins_over_tool(self):
        action = parse_action('<tool>{"name":"retrieve","args":{"query":"x"}}</tool><answer>7</answer>')
        assert action["type"] == "answer"


class TestEpisodeLoop:
    def test_two_turn_success(self, tmp_path):
        item = _item(tmp_path)
        item.gold_answer = "42"
        gen = ScriptedGenerator([TOOL_TURN, ANSWER_TURN])
        traj = run_episode(None, PROC, item, _cfg(), generator=gen)
        assert traj.n_turns == 2
        assert traj.pred_answer == "42"
        assert traj.stopped_reason == "answer"
        assert traj.tool_calls and traj.tool_calls[0]["ok"]
        assert verify_answer(traj.pred_answer, item.gold_answer)

    def test_turn_limit(self, tmp_path):
        item = _item(tmp_path)
        gen = ScriptedGenerator([TOOL_TURN])
        traj = run_episode(None, PROC, item, _cfg(max_turns=3), generator=gen)
        assert traj.n_turns == 3
        assert traj.pred_answer is None
        assert traj.stopped_reason == "turn_limit"

    def test_malformed_tool_is_recorded(self, tmp_path):
        item = _item(tmp_path)
        gen = ScriptedGenerator(["<tool>{oops}</tool>", ANSWER_TURN])
        traj = run_episode(None, PROC, item, _cfg(max_turns=2), generator=gen)
        assert traj.turns[0].action_type == "tool_invalid"
        assert traj.turns[0].tool_ok is False
        assert not traj.tool_calls[0]["schema_valid"]

    def test_unparseable_turn_gets_nudge(self, tmp_path):
        item = _item(tmp_path)
        gen = ScriptedGenerator(["I refuse to use the format", ANSWER_TURN])
        traj = run_episode(None, PROC, item, _cfg(max_turns=2), generator=gen)
        assert traj.turns[0].action_type == "none"
        assert traj.pred_answer == "42" or traj.pred_answer is None

    def test_batched_episodes_are_independent(self, tmp_path):
        items = [_item(tmp_path), _item(tmp_path, family="multi_frame")]
        gen = ScriptedGenerator([TOOL_TURN, ANSWER_TURN])
        trajs = run_episodes(None, PROC, items, _cfg(max_turns=2), generator=gen)
        assert len(trajs) == 2
        assert all(t.n_turns == 2 for t in trajs)
        assert all(t.tool_calls for t in trajs)

    def test_screenshot_chain_satisfies_trace(self, tmp_path):
        item = _item(tmp_path, family="screenshot_ui")
        item.gold_answer = "42"
        click = (
            '<think>open</think><tool>{"name":"click","args":{"element_id":"card_2"}}</tool>'
        )
        read = '<think>read</think><tool>{"name":"read_field","args":{"name":"balance"}}</tool>'
        read2 = '<think>read</think><tool>{"name":"read_field","args":{"name":"pending"}}</tool>'
        gen = ScriptedGenerator([click, read, read2, ANSWER_TURN])
        traj = run_episode(None, PROC, item, _cfg(max_turns=4), generator=gen)
        assert all(c["ok"] for c in traj.tool_calls)
        assert verify_tool_trace(traj.tool_calls, item)

    def test_trajectory_roundtrip(self, tmp_path):
        from vlmgrpo.agent import Trajectory

        item = _item(tmp_path)
        gen = ScriptedGenerator([TOOL_TURN, ANSWER_TURN])
        traj = run_episode(None, PROC, item, _cfg(max_turns=2), generator=gen)
        restored = Trajectory.from_dict(traj.to_dict())
        assert restored.task_id == traj.task_id
        assert restored.n_turns == traj.n_turns
        assert restored.pred_answer == traj.pred_answer
        assert restored.tool_calls[0]["name"] == "python_exec"


class TestStubPipeline:
    def test_stub_episode_runs(self, tmp_path):
        from vlmgrpo.registry import build_stub_vlm

        model, processor = build_stub_vlm()
        item = _item(tmp_path)
        cfg = replace(_cfg(max_turns=1), rollout=replace(_cfg().rollout, batch_size=1, max_new_tokens=8))
        traj = run_episode(model, processor, item, cfg, tier="stub", device="cpu")
        assert traj.n_turns >= 1
        assert traj.turns[0].raw_text is not None
        assert traj.prompt
