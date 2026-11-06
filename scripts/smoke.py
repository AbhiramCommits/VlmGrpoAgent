#!/usr/bin/env python
"""Smoke test: render a task, run one stub rollout, print everything.

Runs fully offline on CPU in a few seconds.  Exit code 0 and ``SMOKE OK`` mean
the package, config, model registry, processor API and generation loop agree.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlmgrpo.config import load_config  # noqa: E402
from vlmgrpo.prompts import build_messages, format_prompt  # noqa: E402
from vlmgrpo.registry import build_stub_vlm, detect_device, model_info, pick_model_tier  # noqa: E402
from vlmgrpo.utils import count_params, seed_everything  # noqa: E402


def render_smoke_chart(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    values = [12, 7, 19, 4, 15]
    fig, ax = plt.subplots(figsize=(4.4, 3.0), dpi=110)
    bars = ax.bar(labels, values, color="#4C72B0")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.4, str(value), ha="center", fontsize=10)
    ax.set_title("Tickets closed per day")
    ax.set_ylabel("tickets")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tiny.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config) if Path(args.config).exists() else load_config("configs/ci.yaml")
    seed_everything(cfg.seed)

    # Always use the stub tier for the smoke test: offline, fast, deterministic.
    model, processor = build_stub_vlm()
    device = detect_device()
    tier = pick_model_tier(device, override=cfg.model_tier)
    print(f"config={args.config} device={device} auto_tier={tier} smoke_tier=stub")
    print(f"stub_params={count_params(model):,} info={model_info('stub')['description']}")

    image_path = Path(cfg.data_dir) / "smoke_chart.png"
    render_smoke_chart(image_path)
    image = Image.open(image_path).convert("RGB")
    question = (
        "The bar chart shows tickets closed per day. "
        "What is the mean of the three lowest bars? Round to 2 decimals."
    )

    messages = build_messages(question)
    prompt = format_prompt(processor, messages)
    inputs = processor(text=[prompt], images=[image], return_tensors="pt", padding=True)

    out = model.generate(
        **inputs,
        max_new_tokens=32,
        do_sample=True,
        temperature=0.9,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
    )
    new_tokens = out[0, inputs["input_ids"].shape[1] :]
    raw = processor.tokenizer.decode(new_tokens, skip_special_tokens=True)

    print("\n=== RENDERED TASK PROMPT ===")
    print(prompt)
    print("\n=== STUB MODEL RAW OUTPUT ===")
    print(raw)
    print("\nSMOKE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
