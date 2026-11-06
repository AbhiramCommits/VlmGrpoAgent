"""LoRA supervised fine-tuning (SFT cold start) on gold trajectories.

* LoRA on the LM attention + MLP projections, vision tower frozen.
* Loss masked to assistant spans only.
* Cosine schedule with warmup, gradient accumulation, bf16/fp16 autocast on CUDA.
* ``--full-params`` trains the whole model instead (small models only).

CLI: ``python -m vlmgrpo.train_sft --config configs/tiny.yaml``
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from .config import RunConfig, load_config
from .registry import detect_device, dtype_name, load_vlm, pick_model_tier, resolve_dtype
from .tracking import Tracker
from .utils import (
    count_params,
    encode_text,
    git_sha,
    human_hours,
    read_jsonl,
    register_checkpoint,
    seed_everything,
    write_json,
)

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class SFTDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], max_length: int) -> None:
        self.records = records
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.records[idx]


def _load_image(path: str):
    from PIL import Image

    return Image.open(path).convert("RGB")


def make_collate(processor: Any, device: str, max_length: int):
    tokenizer = processor.tokenizer
    eos_id = getattr(tokenizer, "eos_token_id", None)
    pad_id = getattr(tokenizer, "pad_token_id", 0) or 0

    def collate(records: list[dict[str, Any]]) -> dict[str, Any]:
        input_ids, labels, attn, pixel_values, pixel_masks = [], [], [], [], []
        for rec in records:
            enc = processor(
                text=[rec["prompt_text"]],
                images=[_load_image(rec["image_path"])],
                return_tensors="pt",
                padding=False,
            )
            prefix = enc["input_ids"][0].tolist()
            target = encode_text(tokenizer, rec["target_text"])
            if eos_id is not None:
                target = target + [eos_id]
            if len(prefix) + len(target) > max_length:
                # never truncate the answer; drop the oldest prompt tokens if needed
                keep = max(16, max_length - len(target))
                prefix = prefix[-keep:]
            ids = prefix + target
            input_ids.append(ids)
            labels.append([-100] * len(prefix) + target)
            attn.append([1] * len(ids))
            pv = enc.get("pixel_values")
            if pv is not None:
                pixel_values.append(pv[0])
            pm = enc.get("pixel_attention_mask")
            if pm is not None:
                pixel_masks.append(pm[0])
        max_len = max(len(x) for x in input_ids)
        for i in range(len(input_ids)):
            pad = max_len - len(input_ids[i])
            input_ids[i] = input_ids[i] + [pad_id] * pad
            labels[i] = labels[i] + [-100] * pad
            attn[i] = attn[i] + [0] * pad
        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
            "attention_mask": torch.tensor(attn, dtype=torch.long, device=device),
            "labels": torch.tensor(labels, dtype=torch.long, device=device),
        }
        if pixel_values:
            batch["pixel_values"] = torch.stack(pixel_values).to(device)
        if pixel_masks:
            batch["pixel_attention_mask"] = torch.stack(pixel_masks).to(device)
        return batch

    return collate


def apply_lora(model: Any, cfg: RunConfig) -> Any:
    from peft import LoraConfig, get_peft_model

    lora_cfg = LoraConfig(
        r=cfg.sft.lora_r,
        lora_alpha=cfg.sft.lora_alpha,
        lora_dropout=cfg.sft.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[m for m in TARGET_MODULES],
    )
    model = get_peft_model(model, lora_cfg)
    # freeze the vision tower explicitly even if a name matched
    for name, param in model.named_parameters():
        if "vision_model" in name or "vision_tower" in name:
            param.requires_grad_(False)
    return model


def evaluate_loss(model: Any, loader: DataLoader, device: str, max_batches: int = 4) -> float:
    model.eval()
    losses = []
    if device == "mps":
        torch.mps.empty_cache()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            try:
                out = model(**batch)
                loss = out.loss if hasattr(out, "loss") else out["loss"]
                losses.append(float(loss.detach().cpu()))
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[sft] val batch failed: {exc}")
    model.train()
    return round(sum(losses) / len(losses), 4) if losses else float("nan")


def train(
    cfg: RunConfig,
    full_params: bool = False,
    max_steps: int | None = None,
    resume: bool = False,
    lr: float | None = None,
) -> dict[str, Any]:
    seed_everything(cfg.seed)
    device = cfg.device if cfg.device != "auto" else detect_device()
    tier = pick_model_tier(device, override=cfg.model_tier)
    dtype = resolve_dtype(device, cfg.dtype)
    data_dir = Path(cfg.data_dir)
    train_rows = read_jsonl(data_dir / "sft_train.jsonl")
    val_rows = read_jsonl(data_dir / "sft_val.jsonl")
    if not train_rows:
        raise FileNotFoundError("data/sft_train.jsonl missing; run `make data` and `python -m vlmgrpo.data.build_sft`")
    run_id = cfg.run_id
    tracker = Tracker(
        run_id=run_id,
        runs_dir=cfg.runs_dir,
        config={"phase": "sft", "tier": tier, "device": device, "dtype": dtype_name(dtype),
                "full_params": full_params, **cfg.sft.__dict__},
        project=cfg.tracking.project,
        use_wandb=cfg.tracking.use_wandb,
    )
    t0 = time.perf_counter()
    ckpt_dir = Path(cfg.runs_dir) / run_id / "sft"
    if resume and (ckpt_dir / "adapter_config.json").exists():
        print(f"[sft] resuming adapter from {ckpt_dir}")
        model, processor = load_vlm(tier, device, cfg.dtype, adapter_path=ckpt_dir, trainable=True)
    else:
        model, processor = load_vlm(tier, device, cfg.dtype)
        if full_params:
            for param in model.parameters():
                param.requires_grad_(True)
        else:
            model = apply_lora(model, cfg)
    trainable = count_params(model, trainable_only=True)
    total = count_params(model)
    print(f"[sft] tier={tier} device={device} dtype={dtype_name(dtype)} "
          f"params={total:,} trainable={trainable:,} ({100*trainable/max(1,total):.2f}%)")

    train_ds = SFTDataset(train_rows, cfg.sft.max_length)
    val_ds = SFTDataset(val_rows[: max(1, cfg.sft.val_items * 4)], cfg.sft.max_length)
    collate = make_collate(processor, device, cfg.sft.max_length)
    train_loader = DataLoader(train_ds, batch_size=cfg.sft.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=cfg.sft.batch_size, shuffle=False, collate_fn=collate)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr if lr is not None else cfg.sft.lr, weight_decay=cfg.sft.weight_decay)
    batches_per_epoch = max(1, math.ceil(len(train_ds) / cfg.sft.batch_size))
    steps_per_epoch = max(1, math.ceil(batches_per_epoch / cfg.sft.grad_accum))
    total_steps = cfg.sft.epochs * steps_per_epoch
    if max_steps is not None and max_steps > 0:
        total_steps = min(total_steps, max_steps)
    if cfg.sft.max_steps > 0:
        total_steps = min(total_steps, cfg.sft.max_steps)
    warmup = max(1, int(total_steps * cfg.sft.warmup_ratio))
    from transformers import get_cosine_schedule_with_warmup

    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, max(1, total_steps))
    autocast_dtype = None
    if device == "cuda":
        autocast_dtype = torch.bfloat16 if dtype == torch.bfloat16 else torch.float16

    model.train()
    global_step = 0
    micro = 0
    accum_loss = 0.0
    tokens_seen = 0
    best_val = float("inf")
    log_every = max(1, cfg.tracking.log_every)
    optimizer.zero_grad(set_to_none=True)
    stop = False
    for epoch in range(cfg.sft.epochs):
        for batch in train_loader:
            tokens_seen += int(batch["attention_mask"].sum().item())
            context = torch.autocast("cuda", dtype=autocast_dtype) if autocast_dtype else torch.enable_grad()
            with context:
                out = model(**batch)
                raw_loss = out.loss if hasattr(out, "loss") else out["loss"]
                loss = raw_loss / cfg.sft.grad_accum
            if not torch.isfinite(loss):
                print("[sft] non-finite loss; skipping batch")
                optimizer.zero_grad(set_to_none=True)
                micro += 1
                continue
            loss.backward()
            accum_loss += float(loss.detach().cpu())
            micro += 1
            if micro % cfg.sft.grad_accum == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.sft.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                if device == "mps":
                    torch.mps.empty_cache()
                global_step += 1
                row = {
                    "phase": "sft",
                    "epoch": epoch,
                    "loss": round(accum_loss, 4),
                    "lr": round(scheduler.get_last_lr()[0], 8),
                    "grad_norm": round(float(grad_norm), 4),
                    "tokens_seen": tokens_seen,
                    "wall_clock_sec": round(time.perf_counter() - t0, 1),
                }
                if global_step % cfg.sft.val_every == 0 and val_rows:
                    val_loss = evaluate_loss(model, val_loader, device)
                    row["val_loss"] = val_loss
                    best_val = min(best_val, val_loss)
                    print(f"[sft] step {global_step}/{total_steps} loss={row['loss']:.4f} val={val_loss:.4f}")
                elif global_step % log_every == 0:
                    print(f"[sft] step {global_step}/{total_steps} loss={row['loss']:.4f} "
                          f"lr={row['lr']:.2e}")
                tracker.log(row, step=global_step)
                accum_loss = 0.0
                if global_step >= total_steps:
                    stop = True
                    break
        if stop:
            break
    wall = time.perf_counter() - t0
    if val_rows:
        final_val = evaluate_loss(model, val_loader, device, max_batches=8)
        tracker.log({"phase": "sft", "val_loss": final_val, "wall_clock_sec": round(wall, 1)}, step=global_step)
        best_val = min(best_val, final_val)
    model.save_pretrained(ckpt_dir)
    try:
        processor.save_pretrained(ckpt_dir)
    except Exception:
        pass
    summary = {
        "phase": "sft",
        "tier": tier,
        "device": device,
        "dtype": dtype_name(dtype),
        "params": total,
        "trainable_params": trainable,
        "steps": global_step,
        "epochs": cfg.sft.epochs,
        "examples": len(train_rows),
        "batch_size": cfg.sft.batch_size,
        "grad_accum": cfg.sft.grad_accum,
        "wall_clock_sec": round(wall, 2),
        "wall_clock_hours": human_hours(wall),
        "best_val_loss": best_val if best_val != float("inf") else None,
        "final_val_loss": best_val if best_val != float("inf") else None,
        "commit_sha": git_sha(),
        "checkpoint": str(ckpt_dir),
    }
    write_json(Path(cfg.runs_dir) / run_id / "sft_summary.json", summary)
    tracker.log_summary(summary)
    messenger = register_checkpoint(
        cfg.runs_dir,
        "sft",
        ckpt_dir,
        {"tier": tier, "device": device, "steps": global_step, "run_id": run_id, "phase": "sft"},
    )
    print(f"[sft] done: steps={global_step} wall={wall/60:.1f}min checkpoint={ckpt_dir} registry={messenger}")
    tracker.finish()
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LoRA SFT cold start")
    parser.add_argument("--config", default="configs/tiny.yaml")
    parser.add_argument("--full-params", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--devices", default=None, help="comma-separated device indices; the custom loop uses the first")
    parser.add_argument("--model-tier", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume", action="store_true", help="continue from runs/<run_id>/sft")
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.device:
        cfg.device = args.device
    if args.devices:
        first = args.devices.split(",")[0].strip()
        cfg.device = f"cuda:{first}" if first.isdigit() else first
        if "," in args.devices:
            print("[config] custom training loop runs on one device; --devices selects the first")
    if args.model_tier:
        cfg.model_tier = args.model_tier
    if args.run_id:
        cfg.run_id = args.run_id
    summary = train(cfg, full_params=args.full_params, max_steps=args.max_steps, resume=args.resume, lr=args.lr)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
