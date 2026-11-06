"""Device detection, model-tier selection and VLM loading.

Tiers
-----
``stub``  ~1.4M parameter randomly-initialised VLM (no download, CI/test tier)
``tiny``  HuggingFaceTB/SmolVLM-256M-Instruct  (CPU / MPS viable)
``small`` Qwen/Qwen2-VL-2B-Instruct           (CUDA >= 8 GB)
``base``  Qwen/Qwen2.5-VL-3B-Instruct         (CUDA >= 14 GB)

The auto-selection picks the largest tier that fits the detected device and
available memory.  ``--model-tier`` (or ``model_tier`` in YAML) always wins.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers import GenerationMixin, PretrainedConfig, PreTrainedModel

TIERS: dict[str, dict[str, Any]] = {
    "stub": {
        "hf_id": None,
        "params": "1.4M",
        "device": "any",
        "min_vram_gb": 0.0,
        "description": "randomly-initialised local VLM for tests/CI (no download)",
    },
    "tiny": {
        "hf_id": "HuggingFaceTB/SmolVLM-256M-Instruct",
        "params": "256M",
        "device": "cpu/mps/cuda",
        "min_vram_gb": 0.0,
        "description": "SmolVLM-256M-Instruct, CPU/MPS-viable default on no-GPU",
    },
    "small": {
        "hf_id": "Qwen/Qwen2-VL-2B-Instruct",
        "params": "2B",
        "device": "cuda",
        "min_vram_gb": 8.0,
        "description": "Qwen2-VL-2B-Instruct",
    },
    "base": {
        "hf_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "params": "3B",
        "device": "cuda",
        "min_vram_gb": 14.0,
        "description": "Qwen2.5-VL-3B-Instruct",
    },
}

_ORDER = ["stub", "tiny", "small", "base"]


def detect_device() -> str:
    """Return ``cuda``, ``mps`` or ``cpu`` (CUDA wins)."""
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _vram_gb(device: str) -> float:
    if device == "cuda" and torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if device == "mps":
        try:
            return torch.mps.driver_allocated_memory() / (1024**3)
        except Exception:
            # unified memory: assume enough for the tiny tier only
            return 0.0
    return 0.0


def pick_model_tier(device: str | None = None, override: str | None = None) -> str:
    """Pick the largest model tier that fits ``device``."""
    if override and override != "auto":
        if override not in TIERS:
            raise ValueError(f"Unknown model tier {override!r}; choose from {_ORDER}")
        return override
    device = device or detect_device()
    if device == "cuda":
        vram = _vram_gb(device)
        if vram >= TIERS["base"]["min_vram_gb"]:
            return "base"
        if vram >= TIERS["small"]["min_vram_gb"]:
            return "small"
        return "tiny"
    # CPU and MPS: the 256M model is the only sane option.
    return "tiny"


def resolve_dtype(device: str, dtype: str = "auto") -> torch.dtype:
    if dtype and dtype != "auto":
        return {"float32": torch.float32, "fp32": torch.float32,
                "float16": torch.float16, "fp16": torch.float16,
                "bfloat16": torch.bfloat16, "bf16": torch.bfloat16}[dtype]
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    # MPS half precision is still fragile for these models; fp32 is stable.
    return torch.float32


def dtype_name(dtype: torch.dtype) -> str:
    return {torch.float32: "float32", torch.float16: "float16", torch.bfloat16: "bfloat16"}.get(
        dtype, str(dtype).replace("torch.", "")
    )


# --------------------------------------------------------------------------- #
# Stub VLM
# --------------------------------------------------------------------------- #


class StubConfig(PretrainedConfig):
    model_type = "vlmgrpo_stub"

    def __init__(
        self,
        vocab_size: int = 512,
        hidden_size: int = 192,
        num_layers: int = 4,
        num_heads: int = 4,
        ffn_size: int = 384,
        image_size: int = 64,
        image_token_id: int = 3,
        max_position_embeddings: int = 2048,
        **kwargs: Any,
    ) -> None:
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ffn_size = ffn_size
        self.image_size = image_size
        self.image_token_id = image_token_id
        self.max_position_embeddings = max_position_embeddings
        super().__init__(**kwargs)


class StubBlock(nn.Module):
    """Pre-LN transformer block with PEFT-friendly projection names."""

    def __init__(self, d: int, heads: int, ffn: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)
        self.ln2 = nn.LayerNorm(d)
        self.gate_proj = nn.Linear(d, ffn, bias=False)
        self.up_proj = nn.Linear(d, ffn, bias=False)
        self.down_proj = nn.Linear(ffn, d, bias=False)
        self.heads = heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        h = self.ln1(x)
        q = self.q_proj(h).view(b, t, self.heads, d // self.heads).transpose(1, 2)
        k = self.k_proj(h).view(b, t, self.heads, d // self.heads).transpose(1, 2)
        v = self.v_proj(h).view(b, t, self.heads, d // self.heads).transpose(1, 2)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(d // self.heads)
        mask = torch.triu(torch.ones(t, t, dtype=torch.bool, device=x.device), diagonal=1)
        att = att.masked_fill(mask, torch.finfo(att.dtype).min)
        att = torch.softmax(att, dim=-1)
        out = (att @ v).transpose(1, 2).reshape(b, t, d)
        x = x + self.o_proj(out)
        h = self.ln2(x)
        return x + self.down_proj(torch.nn.functional.silu(self.gate_proj(h)) * self.up_proj(h))


class StubVLM(PreTrainedModel, GenerationMixin):
    """A tiny randomly-initialised image+text model with a HF-style ``generate``.

    It is intentionally *not* trained: it exists so the full pipeline (rollouts,
    SFT, GRPO, evaluation, analysis) runs in CI with no network and no weights.
    """

    config_class = StubConfig
    main_input_name = "input_ids"

    def __init__(self, config: StubConfig) -> None:
        super().__init__(config)
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [StubBlock(config.hidden_size, config.num_heads, config.ffn_size) for _ in range(config.num_layers)]
        )
        self.ln_f = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.vision = nn.Sequential(
            nn.Conv2d(3, 16, 4, stride=4),
            nn.GELU(),
            nn.Conv2d(16, 32, 4, stride=4),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(32 * (config.image_size // 16) ** 2, config.hidden_size),
        )
        self.image_token_id = config.image_token_id
        self.post_init()

    def _inject_vision(self, hidden: torch.Tensor, input_ids: torch.Tensor, pixel_values: torch.Tensor) -> torch.Tensor:
        vis = self.vision(pixel_values.to(hidden.dtype))
        hidden = hidden.clone()
        for b in range(input_ids.shape[0]):
            positions = (input_ids[b] == self.image_token_id).nonzero(as_tuple=False).flatten()
            if positions.numel():
                hidden[b, positions[0]] = vis[b]
        return hidden

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Any:
        if input_ids is None:
            raise ValueError("input_ids required")
        hidden = self.embed(input_ids)
        if pixel_values is not None:
            hidden = self._inject_vision(hidden, input_ids, pixel_values)
        for layer in self.layers:
            hidden = layer(hidden)
        logits = self.lm_head(self.ln_f(hidden))
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return {"logits": logits, "loss": loss}

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        max_new_tokens: int = 32,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        pad_token_id: int = 0,
        eos_token_id: int = 2,
        **kwargs: Any,
    ) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        out = input_ids
        mask = attention_mask
        for _ in range(int(max_new_tokens)):
            logits = self.forward(out, attention_mask=mask, pixel_values=pixel_values)["logits"][:, -1]
            if do_sample and temperature and temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                if top_p < 1.0:
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
                    cumulative = torch.cumsum(sorted_probs, dim=-1)
                    keep = cumulative - sorted_probs < top_p
                    keep[..., 0] = True
                    filtered = torch.full_like(probs, 0.0)
                    filtered.scatter_(1, sorted_idx, torch.where(keep, sorted_probs, torch.zeros_like(sorted_probs)))
                    probs = filtered / filtered.sum(dim=-1, keepdim=True).clamp_min(1e-9)
                nxt = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                nxt = logits.argmax(dim=-1)
            out = torch.cat([out, nxt[:, None]], dim=1)
            mask = torch.cat([mask, torch.ones_like(nxt[:, None])], dim=1)
            if eos_token_id is not None and torch.all(nxt == eos_token_id):
                break
        return out

    def prepare_inputs_for_generation(self, input_ids: torch.Tensor, **kwargs: Any) -> dict[str, Any]:
        return {"input_ids": input_ids, **{k: v for k, v in kwargs.items() if k in ("attention_mask", "pixel_values")}}


def build_stub_vlm() -> tuple[StubVLM, StubProcessor]:
    """Build the randomly-initialised stub model and its processor."""
    cfg = StubConfig()
    torch.manual_seed(0)
    model = StubVLM(cfg)
    model.eval()
    return model, StubProcessor()


# --------------------------------------------------------------------------- #
# Stub processor / tokenizer
# --------------------------------------------------------------------------- #

_SPECIALS = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<image>": 3, "<unk>": 4}
_CHAR_OFFSET = 8


class StubTokenizer:
    """Whitespace-free character tokenizer, deterministic and offline."""

    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    image_token_id = 3
    pad_token = "<pad>"
    eos_token = "<eos>"
    bos_token = "<bos>"
    padding_side = "left"

    def __len__(self) -> int:
        return 512

    def _char_id(self, ch: str) -> int:
        return min(_CHAR_OFFSET + ord(ch), 511)

    def _char(self, idx: int) -> str:
        if idx < _CHAR_OFFSET:
            inv = {v: k for k, v in _SPECIALS.items()}
            return inv.get(idx, "")
        return chr(idx - _CHAR_OFFSET)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids: list[int] = []
        for ch in text:
            ids.append(self._char_id(ch))
        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]
        return ids

    def __call__(self, text: str | list[str], add_special_tokens: bool = False, **kwargs: Any) -> dict[str, list[list[int]]]:
        texts = [text] if isinstance(text, str) else list(text)
        return {"input_ids": [self.encode(t, add_special_tokens) for t in texts]}

    def decode(self, ids: Any, skip_special_tokens: bool = True) -> str:
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        out = []
        for i in ids:
            i = int(i)
            if skip_special_tokens and i < _CHAR_OFFSET:
                continue
            out.append(self._char(i))
        return "".join(out)

    def batch_decode(self, batch: Any, skip_special_tokens: bool = True) -> list[str]:
        return [self.decode(ids, skip_special_tokens) for ids in batch]


class StubProcessor:
    """Minimal processor mirroring the HF multimodal processor API."""

    image_size = 64

    def __init__(self) -> None:
        self.tokenizer = StubTokenizer()
        self.image_token = "<image>"
        self.image_token_id = 3

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        **kwargs: Any,
    ) -> str:
        parts = []
        for msg in messages:
            role = msg["role"].upper()
            content = msg.get("content", "")
            if isinstance(content, list):
                chunks = []
                for item in content:
                    if item.get("type") == "image":
                        chunks.append("<image>")
                    elif item.get("type") == "text":
                        chunks.append(item.get("text", ""))
                content = "".join(chunks)
            parts.append(f"[{role}]\n{content}\n")
        if add_generation_prompt:
            parts.append("[ASSISTANT]\n")
        text = "".join(parts)
        if tokenize:
            return self.tokenizer(text)  # type: ignore[return-value]
        return text

    def __call__(
        self,
        text: str | list[str] | None = None,
        images: Any = None,
        return_tensors: str | None = "pt",
        padding: bool | str = False,
        truncation: bool = False,
        max_length: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if text is None:
            raise ValueError("text is required")
        texts = [text] if isinstance(text, str) else list(text)
        seqs: list[list[int]] = []
        for t in texts:
            ids: list[int] = []
            for i, chunk in enumerate(t.split("<image>")):
                if i > 0:
                    ids.append(self.image_token_id)
                ids.extend(self.tokenizer.encode(chunk))
            if truncation and max_length:
                ids = ids[:max_length]
            seqs.append(ids)
        maxlen = max(len(s) for s in seqs) if seqs else 0
        do_pad = padding is True or padding == "max_length" or len(seqs) > 1
        input_ids, attn = [], []
        for s in seqs:
            pad = maxlen - len(s)
            if do_pad:
                input_ids.append([0] * pad + s)  # left padding for generation
                attn.append([0] * pad + [1] * len(s))
            else:
                input_ids.append(s)
                attn.append([1] * len(s))
        out: dict[str, Any] = {"input_ids": input_ids, "attention_mask": attn}
        if images is not None:
            import numpy as np
            from PIL import Image

            imgs = images if isinstance(images, (list, tuple)) else [images]
            arrs = []
            for im in imgs:
                if isinstance(im, (list, tuple)):
                    im = im[0]
                if not isinstance(im, Image.Image):
                    im = Image.open(im)
                im = im.convert("RGB").resize((self.image_size, self.image_size))
                arr = np.asarray(im, dtype="float32") / 255.0
                arr = (arr - 0.5) / 0.5
                arrs.append(arr.transpose(2, 0, 1))
            out["pixel_values"] = arrs
        if return_tensors == "pt":
            out["input_ids"] = torch.tensor(out["input_ids"], dtype=torch.long)
            out["attention_mask"] = torch.tensor(out["attention_mask"], dtype=torch.long)
            if "pixel_values" in out:
                out["pixel_values"] = torch.tensor(np.stack(out["pixel_values"]), dtype=torch.float32)
        return out


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def load_vlm(
    tier: str,
    device: str | None = None,
    dtype: str = "auto",
    adapter_path: str | Path | None = None,
) -> tuple[nn.Module, Any]:
    """Load ``(model, processor)`` for a tier, optionally with a PEFT adapter.

    ``tier='stub'`` never touches the network.
    """
    device = device or detect_device()
    if tier == "stub":
        model, processor = build_stub_vlm()
        model.to(device)
        if adapter_path:
            model = _load_adapter(model, adapter_path)
        return model, processor

    if tier not in TIERS or TIERS[tier]["hf_id"] is None:
        raise ValueError(f"Unknown tier {tier!r}")

    from transformers import AutoModelForVision2Seq, AutoProcessor

    hf_id = TIERS[tier]["hf_id"]
    torch_dtype = resolve_dtype(device, dtype)
    processor = AutoProcessor.from_pretrained(hf_id)
    if getattr(processor, "tokenizer", None) is not None:
        processor.tokenizer.padding_side = "left"

    model_cls: Any = AutoModelForVision2Seq
    if tier in ("small", "base"):
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration

            model_cls = Qwen2_5_VLForConditionalGeneration if tier == "base" else Qwen2VLForConditionalGeneration
        except Exception:  # pragma: no cover - transformers version dependent
            model_cls = AutoModelForVision2Seq

    model = model_cls.from_pretrained(hf_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True)
    model.to(device)
    model.eval()
    if adapter_path:
        model = _load_adapter(model, adapter_path)
    return model, processor


def _load_adapter(model: nn.Module, adapter_path: str | Path) -> nn.Module:
    from peft import PeftModel

    adapter_path = Path(adapter_path)
    if (adapter_path / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(model, str(adapter_path))
        model.eval()
        return model
    # full-parameter checkpoint saved via save_pretrained
    if (adapter_path / "config.json").exists():
        from transformers import AutoModelForVision2Seq

        return AutoModelForVision2Seq.from_pretrained(str(adapter_path), torch_dtype=next(model.parameters()).dtype)
    raise FileNotFoundError(f"No adapter or model checkpoint at {adapter_path}")


def model_info(tier: str) -> dict[str, Any]:
    info = dict(TIERS.get(tier, {}))
    info["tier"] = tier
    return info
