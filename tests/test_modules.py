"""Cross-module tests: config, utils, tracking, registry, prompts, CLIs."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from vlmgrpo.config import RunConfig, config_to_dict, load_config, run_config_from_dict
from vlmgrpo.prompts import build_messages, format_prompt, observation_text
from vlmgrpo.registry import (
    build_stub_vlm,
    detect_device,
    dtype_name,
    load_processor,
    model_info,
    pick_model_tier,
    resolve_dtype,
)
from vlmgrpo.tasks.families import generate_task
from vlmgrpo.tools import execute_tool, is_safe_path, python_exec, tool_names, tool_schemas_text
from vlmgrpo.tracking import Tracker, read_metrics
from vlmgrpo.utils import (
    Timer,
    encode_text,
    git_dirty,
    git_sha,
    human_hours,
    json_safe,
    read_checkpoints,
    read_json,
    read_jsonl,
    register_checkpoint,
    resolve_checkpoint,
    write_json,
    write_jsonl,
)


class TestConfig:
    def test_defaults(self):
        cfg = run_config_from_dict({"seed": 3})
        assert isinstance(cfg, RunConfig)
        assert cfg.seed == 3
        assert cfg.reward.w_answer == 1.0

    def test_bad_keys(self):
        with pytest.raises(ValueError):
            run_config_from_dict({"nope": 1})
        with pytest.raises(ValueError):
            run_config_from_dict({"task": {"nope": 1}})

    def test_roundtrip(self, tmp_path):
        cfg = run_config_from_dict({"run_id": "x", "task": {"train_n": 8}, "reward": {"w_trace": 0.9}})
        payload = config_to_dict(cfg)
        assert payload["task"]["train_n"] == 8
        path = tmp_path / "c.yaml"
        import yaml

        path.write_text(yaml.safe_dump(payload))
        loaded = load_config(path)
        assert loaded.run_id == "x"
        assert loaded.reward.w_trace == 0.9

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nope.yaml")


class TestUtils:
    def test_json_safe(self):
        payload = {
            "a": np.float32(1.5),
            "b": torch.tensor([1, 2]),
            "c": Path("x"),
            "d": float("nan"),
            "e": {1, 2},
        }
        out = json_safe(payload)
        assert out["a"] == 1.5
        assert out["b"] == [1, 2]
        assert out["c"] == "x"
        assert out["d"] is None
        assert sorted(out["e"]) == [1, 2]

    def test_jsonl_roundtrip(self, tmp_path):
        path = tmp_path / "rows.jsonl"
        write_jsonl(path, [{"a": 1}, {"a": 2}])
        assert read_jsonl(path) == [{"a": 1}, {"a": 2}]
        assert read_jsonl(tmp_path / "missing.jsonl") == []
        write_json(tmp_path / "one.json", {"x": np.int64(3)})
        assert read_json(tmp_path / "one.json")["x"] == 3

    def test_checkpoint_registry(self, tmp_path):
        ckpt = tmp_path / "ckpt"
        ckpt.mkdir()
        register_checkpoint(tmp_path, "sft", ckpt, {"steps": 1})
        assert resolve_checkpoint(tmp_path, "sft") == ckpt
        assert resolve_checkpoint(tmp_path, "base") is None
        assert "sft" in read_checkpoints(tmp_path)
        assert resolve_checkpoint(tmp_path, "missing") is None
        assert resolve_checkpoint(tmp_path, str(ckpt)) == ckpt

    def test_misc(self):
        assert git_sha()
        assert isinstance(git_dirty(), bool)
        assert human_hours(3600) == 1.0
        with Timer() as t:
            pass
        assert t.elapsed >= 0
        assert encode_text(load_processor("stub").tokenizer, "abc") == [
            8 + ord("a"),
            8 + ord("b"),
            8 + ord("c"),
        ]


class TestTracking:
    def test_jsonl_tracker(self, tmp_path):
        tracker = Tracker("run1", runs_dir=tmp_path, config={"x": 1}, use_wandb="never")
        tracker.log({"loss": 1.0}, step=1)
        tracker.log({"loss": 0.5})
        tracker.log_summary({"ok": True})
        tracker.finish()
        rows = read_metrics(tmp_path / "run1")
        assert len(rows) == 2
        assert rows[0]["step"] == 1 and rows[0]["run_id"] == "run1"
        assert (tmp_path / "run1" / "summary.json").exists()
        assert tracker.url is None

    def test_wandb_mode(self, monkeypatch):
        assert Tracker._want_wandb("never") is False
        assert Tracker._want_wandb("always") is True
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        assert Tracker._want_wandb("auto") is False
        monkeypatch.setenv("WANDB_API_KEY", "x")
        assert Tracker._want_wandb("auto") is True


class TestRegistry:
    def test_device_and_tiers(self):
        device = detect_device()
        assert device in ("cuda", "mps", "cpu")
        assert pick_model_tier(device, "stub") == "stub"
        assert pick_model_tier("cpu") == "tiny"
        with pytest.raises(ValueError):
            pick_model_tier(device, "gigantic")

    def test_dtypes(self):
        assert resolve_dtype("cpu") == torch.float32
        assert resolve_dtype("cpu", "float16") == torch.float16
        assert dtype_name(torch.bfloat16) == "bfloat16"
        assert model_info("stub")["tier"] == "stub"

    def test_stub_model(self):
        model, processor = build_stub_vlm()
        assert sum(p.numel() for p in model.parameters()) > 1_000_000
        image = Image.new("RGB", (64, 64), "white")
        text = format_prompt(processor, build_messages("How many?"))
        inputs = processor(text=[text, text], images=[image, image], return_tensors="pt", padding=True)
        assert inputs["input_ids"].shape[0] == 2
        assert inputs["pixel_values"].shape[0] == 2
        out = model.generate(
            **inputs,
            max_new_tokens=4,
            do_sample=True,
            temperature=1.0,
            top_p=0.9,
            pad_token_id=0,
            eos_token_id=2,
        )
        assert out.shape[0] == 2
        decoded = processor.tokenizer.batch_decode(out, skip_special_tokens=True)
        assert len(decoded) == 2

    def test_stub_processor_defaults(self):
        processor = load_processor("stub")
        assert processor.tokenizer.pad_token_id == 0
        assert len(processor.tokenizer) == 512
        assert processor.tokenizer.decode([8 + ord("a")]) == "a"
        assert processor.apply_chat_template([{"role": "user", "content": "hi"}]).endswith("ASSISTANT]\n")

    def test_prompts(self):
        msgs = build_messages("q?", [{"role": "assistant", "content": "a"}])
        assert msgs[1]["content"][0]["type"] == "image"
        obs = observation_text("python_exec", {"code": "print(1)"}, {"ok": True, "stdout": "1", "error": ""})
        assert obs.startswith("OBSERVATION ") and '"ok": true' in obs
        from vlmgrpo.prompts import render_history_for_sft

        rendered = render_history_for_sft([{"role": "user", "content": "hi"}])
        assert rendered == "user: hi"


class TestToolsExtras:
    def test_helpers(self):
        assert set(tool_names()) == {"python_exec", "retrieve", "click", "read_field"}
        assert "python_exec" in tool_schemas_text()
        assert is_safe_path("relative/x")
        assert not is_safe_path("/abs/x")

    def test_execute_tool_wrapper(self, tmp_path):
        item = generate_task("chart_agg", 1234, "test", 0, tmp_path)
        result = execute_tool(item, "python_exec", {"code": "print(7)"})
        assert result["ok"] and result["stdout"] == "7"
        assert not execute_tool(item, "unknown", {})["ok"]

    def test_restricted_builtins(self):
        assert python_exec("print(sum([1,2,3]))")["ok"]
        assert not python_exec("print(dir(1))")["ok"]  # dir blocked
        assert python_exec("x = [1]\nx.append(2)\nprint(x)")["ok"]


class TestGenerateCLI:
    def test_single_split(self, tmp_path):
        from vlmgrpo.tasks.generate import main

        out = tmp_path / "test.jsonl"
        code = main(
            [
                "--split",
                "test",
                "--n",
                "1",
                "--out",
                str(out),
                "--image-dir",
                str(tmp_path / "images"),
                "--seed",
                "42",
            ]
        )
        assert code == 0
        rows = [json.loads(line) for line in out.read_text().splitlines()]
        assert len(rows) == 4

    def test_suite_stats_markdown(self, tmp_path):
        from vlmgrpo.tasks.generate import generate_all, stats_markdown, suite_stats

        splits = generate_all(
            seed=1,
            counts={"train": 4, "test": 4},
            n_per_family=1,
            image_dir=tmp_path / "images",
            data_dir=tmp_path,
        )
        flat = [i for v in splits.values() for i in v]
        stats = suite_stats(flat)
        md = stats_markdown(stats, 1)
        assert "| split | family | count |" in md
        assert "Total items" in md


class TestBuildSftCLI:
    def test_main(self, tmp_path):
        import yaml

        from vlmgrpo.data.build_sft import main
        from vlmgrpo.tasks.generate import generate_all

        data_dir = tmp_path / "data"
        generate_all(
            seed=9,
            counts={"train": 4, "val": 4},
            n_per_family=1,
            image_dir=data_dir / "images",
            data_dir=data_dir,
        )
        config_path = tmp_path / "cfg.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "model_tier": "stub",
                    "device": "cpu",
                    "data_dir": str(data_dir),
                    "task": {"image_dir": str(data_dir / "images")},
                    "sft": {"max_length": 2048},
                    "rollout": {"max_turns": 2},
                }
            )
        )
        assert main(["--config", str(config_path)]) == 0
        assert (data_dir / "sft_train.jsonl").exists()
        assert (data_dir / "sft_stats.json").exists()


class TestEvaluateSmall:
    def test_k2_stub(self, tmp_path):
        from vlmgrpo.evaluate import evaluate, load_split
        from vlmgrpo.tasks.generate import generate_all

        data_dir = tmp_path / "data"
        generate_all(
            seed=5,
            counts={"test": 4, "test_hard": 4},
            n_per_family=1,
            image_dir=data_dir / "images",
            data_dir=data_dir,
        )
        cfg = load_config("configs/ci.yaml")
        cfg.data_dir = str(data_dir)
        cfg.results_dir = str(tmp_path / "results")
        cfg.runs_dir = str(tmp_path / "runs")
        cfg.model_tier = "stub"
        cfg.device = "cpu"
        cfg.eval = replace(cfg.eval, k=2, max_items=4, max_turns=1, max_new_tokens=4, batch_size=4)
        items = load_split(cfg, "test", 4)
        assert len(items) == 4
        metrics = evaluate("base", "test", k=2, cfg=cfg, max_items=4)
        assert metrics["n_rollouts"] == 8
        assert metrics["k"] == 2
        assert "per_family" in metrics
        assert (tmp_path / "results" / "eval_base_test.json").exists()


class TestVerifyExtras:
    def test_helpers(self):
        from vlmgrpo.verify import (
            calls_and_text_to_text,
            parse_answer_text,
            tool_trace_detail,
        )

        assert parse_answer_text("noise <answer> 12 </answer> more") == "12"
        assert parse_answer_text("nope") is None
        assert "3" in calls_and_text_to_text({"a": [1, 2], "b": None, "c": {"d": 3}})
        item = generate_task("chart_agg", 1234, "test", 0, Path("/tmp"))
        detail = tool_trace_detail([], item)
        assert detail["missing"] == item.required_tools
        assert detail["ok"] is False


class TestVerifyMore:
    def test_edge_cases(self):
        from vlmgrpo.verify import (
            extract_number,
            format_ok,
            json_parses,
            normalize_text,
            perception_recall,
            verify_answer,
        )

        assert json_parses('{"a": 1}') and not json_parses("nope")
        assert verify_answer(["wrong", "12.0"], "12")
        assert extract_number("-$1,234.50") == -1234.5
        assert normalize_text(None) == ""
        assert perception_recall("there are 12 apples", [12]) == 1.0
        assert format_ok("<THINK>x</THINK><ANSWER>1</ANSWER>")

    def test_missing_split(self, tmp_path):
        from vlmgrpo.evaluate import load_split

        cfg = load_config("configs/ci.yaml")
        cfg.data_dir = str(tmp_path)
        with pytest.raises(FileNotFoundError):
            load_split(cfg, "nope")


class TestCheckpointLoading:
    def test_stub_full_checkpoint(self, tmp_path):
        from vlmgrpo.registry import _load_adapter

        model, _ = build_stub_vlm()
        model.save_pretrained(tmp_path / "ckpt")
        loaded = _load_adapter(model, tmp_path / "ckpt")
        assert isinstance(loaded, type(model))

    def test_missing_checkpoint(self, tmp_path):
        from vlmgrpo.registry import load_vlm

        with pytest.raises(FileNotFoundError):
            load_vlm("stub", "cpu", adapter_path=tmp_path / "nope")
        (tmp_path / "empty").mkdir()
        with pytest.raises(FileNotFoundError):
            load_vlm("stub", "cpu", adapter_path=tmp_path / "empty")

    def test_evaluate_missing_checkpoint(self, tmp_path):
        from vlmgrpo.evaluate import evaluate

        cfg = load_config("configs/ci.yaml")
        cfg.data_dir = str(tmp_path)
        cfg.runs_dir = str(tmp_path / "runs")
        cfg.results_dir = str(tmp_path / "results")
        cfg.model_tier = "stub"
        cfg.device = "cpu"
        with pytest.raises(FileNotFoundError):
            evaluate("grpo", "test", cfg=cfg)
