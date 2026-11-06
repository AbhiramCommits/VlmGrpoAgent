#!/usr/bin/env python
"""Smoke test: render a task, run one stub rollout, print everything.

Runs fully offline on CPU in a few seconds.  Exit code 0 and ``SMOKE OK`` mean
the package, config, model registry, task suite, processor API and generation
loop agree end to end.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlmgrpo.config import load_config  # noqa: E402
from vlmgrpo.prompts import build_messages, format_prompt  # noqa: E402
from vlmgrpo.registry import build_stub_vlm, detect_device, model_info, pick_model_tier  # noqa: E402
from vlmgrpo.tasks.families import FAMILIES, generate_task  # noqa: E402
from vlmgrpo.utils import count_params, seed_everything  # noqa: E402


def pick_smoke_task(cfg) -> dict:
    """Use the committed test split when available, otherwise render one chart task."""
    test_path = Path(cfg.data_dir) / "test.jsonl"
    if test_path.exists():
        rows = [json.loads(line) for line in test_path.read_text().splitlines() if line.strip()]
        for row in rows:
            if row["family"] == "chart_agg":
                return row
    item = generate_task("chart_agg", 1234, "test", 0, Path(cfg.data_dir) / "smoke")
    return item.to_dict()


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
    print(f"families={list(FAMILIES)}")

    task = pick_smoke_task(cfg)
    image_path = task["image_paths"][0]
    if not Path(image_path).exists():
        image_path = str(Path(cfg.data_dir) / "smoke" / Path(image_path).name)
    from PIL import Image

    image = Image.open(image_path).convert("RGB")

    messages = build_messages(task["question"])
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
    print(f"\n=== TASK ===\nfamily={task['family']} gold_answer={task['gold_answer']} "
          f"required_tools={task['required_tools']}")
    print("\n=== STUB MODEL RAW OUTPUT ===")
    print(raw)
    print("\nSMOKE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
