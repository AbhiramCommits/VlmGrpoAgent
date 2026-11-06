"""GRPO core: group-relative advantages, PPO clipping, KL control.

Implemented from scratch (no dependency on a particular TRL version) so the
math is auditable and unit-testable.  ``train_grpo.py`` uses these functions.
"""

from __future__ import annotations

from typing import Any

import torch

EPS = 1e-4


def compute_advantages(
    rewards: list[float] | torch.Tensor,
    eps: float = EPS,
    skip_zero_variance: bool = True,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Group-relative advantage ``A_i = (r_i - mean) / (std + eps)``.

    Groups with (near-)zero reward variance carry no learning signal; they are
    returned as zeros and flagged ``skipped=True``.
    """
    r = torch.as_tensor(rewards, dtype=torch.float32)
    if r.numel() == 0:
        return torch.zeros(0), {"mean": 0.0, "std": 0.0, "skipped": True}
    mean = r.mean()
    std = r.std(unbiased=False) if r.numel() > 1 else torch.tensor(0.0)
    skipped = bool(float(std) < eps)
    if skipped and skip_zero_variance:
        adv = torch.zeros_like(r)
    else:
        adv = (r - mean) / (std + eps)
    return adv, {
        "mean": round(float(mean), 6),
        "std": round(float(std), 6),
        "min": round(float(r.min()), 6),
        "max": round(float(r.max()), 6),
        "skipped": skipped,
    }


def ppo_ratio(new_logp: torch.Tensor, old_logp: torch.Tensor) -> torch.Tensor:
    return torch.exp(torch.clamp(new_logp - old_logp, min=-20.0, max=20.0))


def clipped_surrogate(
    new_logp: torch.Tensor,
    old_logp: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token ``-min(r*A, clip(r)*A)`` and the PPO ratio."""
    ratio = ppo_ratio(new_logp, old_logp)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    return -torch.minimum(surr1, surr2), ratio


def kl_penalty(new_logp: torch.Tensor, ref_logp: torch.Tensor, kind: str = "k3") -> torch.Tensor:
    """Per-token KL estimate between policy and reference (always >= 0 for k3)."""
    log_ratio = ref_logp - new_logp
    if kind == "k1":
        return -log_ratio  # log(new/ref) >= 0 when policy concentrates where ref does not
    if kind == "k3":
        return torch.exp(log_ratio) - log_ratio - 1.0
    raise ValueError(f"unknown KL estimator: {kind}")


def grpo_loss(
    new_logp: torch.Tensor,
    old_logp: torch.Tensor,
    ref_logp: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = 0.2,
    kl_beta: float = 0.02,
    kl_kind: str = "k3",
) -> dict[str, Any]:
    """Policy-gradient loss with clipping and a KL penalty to the frozen SFT policy."""
    pg, ratio = clipped_surrogate(new_logp, old_logp, advantages, clip_eps)
    kl = kl_penalty(new_logp, ref_logp, kl_kind)
    loss = pg.mean() + kl_beta * kl.mean()
    clip_fraction = ((ratio - 1.0).abs() > clip_eps).float().mean()
    return {
        "loss": loss,
        "pg_loss": pg.mean().detach(),
        "kl": kl.mean().detach(),
        "clip_fraction": clip_fraction.detach(),
        "ratio_mean": ratio.mean().detach(),
        "ratio_max": ratio.max().detach(),
    }


def token_logprobs(
    model: Any,
    batch: dict[str, Any],
    need_entropy: bool = False,
) -> dict[str, Any]:
    """Forward one batch and return completion-token log-probs + per-token advantages.

    ``batch`` must contain ``input_ids``, ``attention_mask``, ``labels`` and
    ``advantages`` plus optional pixel tensors.  Only tokens whose label is not
    ``-100`` (the assistant spans) contribute.
    """
    forward_keys = ("input_ids", "attention_mask", "pixel_values", "pixel_attention_mask")
    kwargs = {k: batch[k] for k in forward_keys if k in batch}
    out = model(**kwargs)
    logits = out.logits if hasattr(out, "logits") else out["logits"]
    logits = logits[:, :-1, :]
    targets = batch["input_ids"][:, 1:]
    mask = batch["labels"][:, 1:] != -100
    if not bool(mask.any()):
        return {"logp": torch.zeros(0, device=logits.device), "adv": torch.zeros(0, device=logits.device),
                "entropy": torch.tensor(0.0), "n_tokens": 0}
    sel_logits = logits[mask]
    sel_targets = targets[mask]
    logp = torch.log_softmax(sel_logits.float(), dim=-1)
    gathered = logp.gather(1, sel_targets.unsqueeze(1)).squeeze(1)
    adv = batch["advantages"].to(sel_logits.device)[:, None].expand_as(mask)[mask]
    entropy = None
    if need_entropy:
        entropy = -(logp.exp() * logp).sum(-1).mean().detach()
    return {
        "logp": gathered,
        "adv": adv,
        "entropy": entropy,
        "n_tokens": int(mask.sum().item()),
    }
