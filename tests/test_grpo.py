"""Unit + end-to-end tests for GRPO math and the 2-step stub training run."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from PIL import Image

from vlmgrpo.grpo import (
    clipped_surrogate,
    compute_advantages,
    grpo_loss,
    kl_penalty,
    ppo_ratio,
)


class TestAdvantages:
    def test_known_vector(self):
        adv, stats = compute_advantages([1.0, 2.0, 3.0, 4.0])
        expected = torch.tensor([-1.3416, -0.4472, 0.4472, 1.3416])
        assert torch.allclose(adv, expected, atol=1e-3)
        assert stats["mean"] == pytest.approx(2.5)
        assert not stats["skipped"]

    def test_zero_variance_group_is_skipped(self):
        adv, stats = compute_advantages([0.7, 0.7, 0.7, 0.7])
        assert stats["skipped"]
        assert torch.count_nonzero(adv) == 0

    def test_single_sample(self):
        adv, stats = compute_advantages([1.0])
        assert stats["skipped"]
        assert adv.tolist() == [0.0]

    def test_advantages_zero_mean(self):
        adv, _ = compute_advantages([0.0, 1.0, 5.0])
        assert float(adv.mean()) == pytest.approx(0.0, abs=1e-5)

    def test_skip_flag_off_keeps_zero_advantage(self):
        adv, stats = compute_advantages([2.0, 2.0], skip_zero_variance=False)
        assert stats["skipped"]
        assert adv.tolist() == [0.0, 0.0]


class TestClipping:
    def test_ratio_one_when_unchanged(self):
        logp = torch.tensor([-1.0, -2.0])
        assert torch.allclose(ppo_ratio(logp, logp), torch.ones(2))

    def test_positive_advantage_is_clipped(self):
        old = torch.tensor([-1.0])
        new = old + torch.log(torch.tensor(2.0))  # ratio = 2.0
        loss, ratio = clipped_surrogate(new, old, torch.tensor([1.0]), clip_eps=0.2)
        assert ratio.item() == pytest.approx(2.0, rel=1e-4)
        assert loss.item() == pytest.approx(-1.2, rel=1e-4)  # -min(2, 1.2)

    def test_negative_advantage_clips_on_the_other_side(self):
        old = torch.tensor([-1.0])
        new = old + torch.log(torch.tensor(0.1))  # ratio = 0.1
        loss, _ = clipped_surrogate(new, old, torch.tensor([-1.0]), clip_eps=0.2)
        assert loss.item() == pytest.approx(0.8, rel=1e-4)  # -(-1 * 0.8)

    def test_no_clip_inside_band(self):
        old = torch.tensor([-1.0])
        new = old + torch.log(torch.tensor(1.1))
        loss, ratio = clipped_surrogate(new, old, torch.tensor([1.0]), clip_eps=0.2)
        assert loss.item() == pytest.approx(-1.1, rel=1e-4)


class TestKL:
    def test_zero_when_identical(self):
        x = torch.tensor([-1.0, -2.0, -3.0])
        assert float(kl_penalty(x, x, "k3").mean()) == pytest.approx(0.0, abs=1e-6)
        assert float(kl_penalty(x, x, "k1").mean()) == pytest.approx(0.0, abs=1e-6)

    def test_k3_positive_when_policy_differs(self):
        new = torch.tensor([-1.0])
        ref = torch.tensor([-2.0])
        assert float(kl_penalty(new, ref, "k3")) > 0
        assert float(kl_penalty(ref, new, "k3")) > 0

    def test_k1_sign(self):
        new = torch.tensor([-1.0])
        ref = torch.tensor([-2.0])
        # policy assigns higher probability than reference -> positive KL
        assert float(kl_penalty(new, ref, "k1")) > 0
        assert float(kl_penalty(ref, new, "k1")) < 0

    def test_unknown_kind(self):
        with pytest.raises(ValueError):
            kl_penalty(torch.tensor([0.0]), torch.tensor([0.0]), "nope")


class TestLoss:
    def test_loss_finite_and_clip_fraction(self):
        old = torch.tensor([-1.0, -2.0])
        new = old.clone()
        ref = old.clone()
        out = grpo_loss(new, old, ref, torch.tensor([1.0, -1.0]))
        assert torch.isfinite(out["loss"])
        assert out["clip_fraction"].item() == 0.0
        assert out["kl"].item() == pytest.approx(0.0, abs=1e-6)

    def test_kl_beta_penalises(self):
        old = torch.tensor([-1.0])
        new = torch.tensor([-0.5])
        ref = torch.tensor([-2.0])
        low = grpo_loss(new, old, ref, torch.tensor([1.0]), kl_beta=0.0)
        high = grpo_loss(new, old, ref, torch.tensor([1.0]), kl_beta=1.0)
        assert high["loss"].item() >= low["loss"].item()


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory):
    """Generate data, 1 SFT step, then 2 GRPO steps on the stub tier."""
    tmp = tmp_path_factory.mktemp("vlmgrpo_e2e")
    from vlmgrpo.config import load_config
    from vlmgrpo.data.build_sft import build_sft_split
    from vlmgrpo.registry import load_processor
    from vlmgrpo.tasks.generate import generate_all
    from vlmgrpo.train_grpo import train as train_grpo
    from vlmgrpo.train_sft import train as train_sft

    cfg = load_config("configs/ci.yaml")
    cfg.data_dir = str(tmp / "data")
    cfg.runs_dir = str(tmp / "runs")
    cfg.results_dir = str(tmp / "results")
    cfg.model_tier = "stub"
    cfg.device = "cpu"
    cfg.dtype = "float32"
    cfg.task.image_size = 128
    generate_all(
        seed=7,
        counts={"train": 8, "val": 4, "test": 4, "test_hard": 4},
        n_per_family=1,
        image_dir=tmp / "data" / "images",
        data_dir=cfg.data_dir,
    )
    processor = load_processor("stub")
    build_sft_split(Path(cfg.data_dir), "train", processor, cfg.sft.max_length, 2)
    build_sft_split(Path(cfg.data_dir), "val", processor, cfg.sft.max_length, 2)

    cfg.sft = replace(cfg.sft, epochs=1, batch_size=1, grad_accum=1, max_steps=1, val_items=1)
    sft_summary = train_sft(cfg)

    cfg.grpo = replace(
        cfg.grpo,
        steps=2,
        group_size=2,
        max_turns=2,
        max_new_tokens=8,
        val_every=1,
        val_items=1,
        skip_zero_variance=False,
        prompts_per_step=1,
    )
    grpo_summary = train_grpo(cfg, sft_checkpoint=None, max_steps=2)
    return cfg, sft_summary, grpo_summary


def test_two_step_grpo_runs(tiny_run):
    cfg, sft, grpo = tiny_run
    assert sft["steps"] == 1
    assert grpo["steps"] == 2
    ckpt = Path(cfg.runs_dir) / cfg.run_id / "grpo"
    assert ckpt.exists()
    metrics = Path(cfg.runs_dir) / cfg.run_id / "metrics.jsonl"
    rows = [json.loads(line) for line in metrics.read_text().splitlines() if line.strip()]
    grpo_rows = [r for r in rows if r.get("phase") == "grpo"]
    assert len(grpo_rows) == 2
    for row in grpo_rows:
        assert "reward_mean" in row and "kl" in row and "clip_fraction" in row
        assert "answer_correct" in row and "required_tool_used" in row
    assert grpo["rollouts"] == 4
    registry = json.loads((Path(cfg.runs_dir) / "checkpoints.json").read_text())
    assert "grpo" in registry and "sft" in registry


def test_end_to_end_results_json(tiny_run):
    """Full stub pipeline: eval every checkpoint, assemble results.json."""
    from vlmgrpo.analysis.compare import build_results, run_matrix
    from vlmgrpo.evaluate import evaluate

    cfg, _, _ = tiny_run
    cfg.eval = replace(cfg.eval, k=1, max_items=4, max_turns=2, max_new_tokens=8, batch_size=4)
    for checkpoint in ("base", "sft", "grpo"):
        for split in ("test", "test_hard"):
            evaluate(checkpoint, split, k=1, cfg=cfg, max_items=4)
    matrix = run_matrix(cfg, splits=("test", "test_hard"), k=1, max_items=4)
    results = build_results(cfg, matrix, ("test", "test_hard"))
    for key in ("environment", "config", "metrics", "per_family", "deltas", "failure_mix", "training"):
        assert key in results
    for label in ("BASE", "SFT", "GRPO"):
        for split in ("test", "test_hard"):
            metrics = results["metrics"][label][split]
            assert 0.0 <= metrics["pass@1"] <= 1.0
            assert 0.0 <= metrics["tool_call_schema_validity"] <= 1.0
    assert (Path(cfg.results_dir) / "results.json").exists()


def test_token_logprobs_and_entropy():
    from vlmgrpo.grpo import token_logprobs
    from vlmgrpo.registry import build_stub_vlm

    model, processor = build_stub_vlm()
    image = Image.new("RGB", (64, 64), "white")
    enc = processor(text=["hello world"], images=[image], return_tensors="pt")
    labels = enc["input_ids"].clone()
    labels[:, :-3] = -100
    batch = {**enc, "labels": labels, "advantages": torch.tensor([1.0])}
    out = token_logprobs(model, batch, need_entropy=True)
    assert out["n_tokens"] == 3
    assert out["logp"].shape == (3,)
    assert out["entropy"] is not None
    empty = {**enc, "labels": torch.full_like(enc["input_ids"], -100), "advantages": torch.tensor([1.0])}
    out2 = token_logprobs(model, empty)
    assert out2["n_tokens"] == 0 and out2["logp"].numel() == 0
