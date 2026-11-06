"""Deterministic multimodal task families rendered to PNG.

Every family requires *reading pixels* **and** at least one tool call: the
question never contains the numbers needed to answer it.  Rendering is fully
seeded (``random.Random(seed)``); the same seed yields byte-identical task
JSON and images.

Families
--------
chart_agg      bar chart, aggregate over >= 3 bars
doc_table      rendered invoice table, filter rows + arithmetic
screenshot_ui  mock app screenshot, requires click()/read_field()
multi_frame    3-frame video strip, cross-frame comparison + arithmetic
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

FAMILIES = ("chart_agg", "doc_table", "screenshot_ui", "multi_frame")

STYLES: dict[str, dict[str, Any]] = {
    "clean": {
        "palette": ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#DA8BC3"],
        "bg": "#FFFFFF",
        "fg": "#1A1A1A",
        "muted": "#8C8C8C",
        "font": "DejaVu Sans",
        "noise": 0.0,
        "distractors": 0,
    },
    "hard": {
        "palette": ["#2F4B7C", "#F95D6A", "#00A6A6", "#FFB000", "#8E44AD", "#1B998B", "#D45087"],
        "bg": "#F4F1EA",
        "fg": "#22201C",
        "muted": "#7A7468",
        "font": "DejaVu Serif",
        "noise": 6.0,
        "distractors": 3,
    },
}

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
PRODUCTS = ["Widget", "Gasket", "Bolt", "Sensor", "Panel", "Cable", "Valve", "Relay", "Motor", "Clamp"]
REGIONS = ["North", "South", "East", "West"]
SURNAMES = ["Ito", "Novak", "Okafor", "Silva", "Haddad", "Kaur", "Meyer", "Rossi"]


@dataclass
class TaskItem:
    """One multimodal agentic task with everything needed to verify it."""

    task_id: str
    family: str
    image_paths: list[str]
    question: str
    gold_answer: str
    gold_tool_calls: list[dict[str, Any]]
    gold_perception_facts: dict[str, Any]
    required_tools: list[str]
    difficulty: int
    seed: int
    split: str = "train"
    retrieval_corpus: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TaskItem:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in payload.items() if k in known})

    @property
    def image_path(self) -> str:
        return self.image_paths[0]

    def content_hash(self) -> str:
        payload = self.to_dict()
        payload.pop("image_paths", None)
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #


def _font(size: int, style: dict[str, Any], bold: bool = False) -> ImageFont.FreeTypeFont:
    try:
        path = font_manager.findfont(font_manager.FontProperties(family=style["font"], weight="bold" if bold else "normal"))
    except Exception:
        path = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans"))
    return ImageFont.truetype(path, size=size)


def _canvas(w: int, h: int, style: dict[str, Any]) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (w, h), style["bg"])
    return img, ImageDraw.Draw(img)


def _finish(img: Image.Image, path: Path, style: dict[str, Any], rng: random.Random) -> None:
    if style["noise"] > 0:
        arr = np.asarray(img, dtype=np.float32)
        noise = rng.gauss(0.0, 1.0)
        # deterministic per-pixel speckle: use a fresh seeded RNG per image
        speckle_rng = np.random.default_rng(rng.randint(0, 2**31 - 1))
        arr = arr + speckle_rng.normal(0.0, style["noise"], arr.shape)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        del noise
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, optimize=False)


def _code(call: dict[str, Any]) -> dict[str, Any]:
    return {"name": "python_exec", "args": {"code": call["code"]}}


# --------------------------------------------------------------------------- #
# Family 1: chart_agg
# --------------------------------------------------------------------------- #


def _render_chart(values: list[int], labels: list[str], title: str, path: Path, style: dict[str, Any], rng: random.Random) -> None:
    palette = style["palette"]
    with plt.rc_context({"font.family": style["font"]}):
        fig, ax = plt.subplots(figsize=(5.0, 3.6), dpi=110)
        fig.patch.set_facecolor(style["bg"])
        ax.set_facecolor(style["bg"])
        bars = ax.bar(labels, values, color=[palette[i % len(palette)] for i in range(len(values))], width=0.68)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + max(values) * 0.02,
                str(value),
                ha="center",
                va="bottom",
                fontsize=9,
                color=style["fg"],
            )
        ax.set_title(title, fontsize=11, color=style["fg"])
        ax.set_ylabel("value", fontsize=9, color=style["fg"])
        ax.tick_params(labelsize=8, colors=style["fg"])
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.set_ylim(0, max(values) * 1.18)
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, facecolor=style["bg"])
        plt.close(fig)


def make_chart_agg(rng: random.Random, item_seed: int, split: str, out_dir: Path) -> dict[str, Any]:
    style = STYLES["hard" if split == "test_hard" else "clean"]
    hard = split == "test_hard"
    n = rng.randint(5, 7) + (2 if hard else 0)
    labels = rng.sample(WEEKDAYS, min(n, len(WEEKDAYS)))
    if n > len(labels):
        labels += rng.sample(MONTHS, n - len(labels))
    values = [rng.randint(6, 48) for _ in range(n)]
    # distractor series only in the hard style: grey bars that must be ignored
    distractors = [rng.randint(4, 55) for _ in range(style["distractors"])] if hard else []
    title = rng.choice(["Tickets closed per day", "Orders shipped per day", "Signups per day"])
    path = out_dir / f"{split}_chart_agg_{item_seed}.png"
    _render_chart(values, labels, title, path, style, rng)
    if distractors:
        # overlay distractor bars in a separate muted panel strip (kept out of the chart math)
        img = Image.open(path).convert("RGB")
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, 74, img.height], fill=style["bg"])
        draw.text((6, 8), "other\nseries:", fill=style["muted"], font=_font(10, style))
        for i, d in enumerate(distractors):
            draw.rectangle([10, 40 + i * 22, 60, 54 + i * 22], fill=style["muted"])
            draw.text((14, 42 + i * 22), str(d), fill="#FFFFFF", font=_font(9, style))
        img.save(path)

    low3 = sorted(values)[:3]
    mean3 = sum(low3) / 3.0
    mean_all = sum(values) / len(values)
    if hard:
        sub = round(mean3, 6)
        gold = round(sub / round(mean_all, 6), 4)
        code_a = f"low = sorted({values})[:3]\nsub = sum(low)/len(low)\nprint(round(sub, 6))"
        code_b = f"print(round({sub!r} / {round(mean_all, 6)!r}, 4))"
        gold_calls = [_code({"code": code_a}), _code({"code": code_b})]
        question = (
            f"The bar chart shows {title.lower()}. Compute the mean of the three lowest colored bars, "
            "then divide that mean by the mean of all colored bars. Report the ratio to 4 decimals."
        )
        gold_answer = f"{gold:.4f}"
        difficulty = 3
    else:
        gold = round(mean3, 2)
        code = f"vals = {values}\nlow = sorted(vals)[:3]\nprint(round(sum(low)/len(low), 2))"
        gold_calls = [_code({"code": code})]
        question = (
            f"The bar chart shows {title.lower()}. What is the mean of the three lowest bars? "
            "Round to 2 decimals."
        )
        gold_answer = f"{gold:.2f}"
        difficulty = 1
    return {
        "image_paths": [str(path)],
        "question": question,
        "gold_answer": gold_answer,
        "gold_tool_calls": gold_calls,
        "required_tools": ["python_exec"] * len(gold_calls),
        "gold_perception_facts": {f"bar:{lab}": v for lab, v in zip(labels, values)},
        "retrieval_corpus": {
            "chart_values": ", ".join(f"{lab}={v}" for lab, v in zip(labels, values)),
            "chart_title": title,
            "chart_all": f"{title}; values: " + ", ".join(f"{lab}={v}" for lab, v in zip(labels, values)),
        },
        "difficulty": difficulty,
        "metadata": {"values": values, "labels": labels, "title": title, "distractor_values": distractors},
    }


# --------------------------------------------------------------------------- #
# Family 2: doc_table
# --------------------------------------------------------------------------- #


def _render_table(rows: list[dict[str, Any]], title: str, path: Path, style: dict[str, Any], rng: random.Random) -> None:
    w, h = 560, 90 + 34 * len(rows)
    img, draw = _canvas(w, h, style)
    draw.text((18, 14), title, fill=style["fg"], font=_font(17, style, bold=True))
    draw.text((18, 40), "Generated ledger extract - amounts in USD", fill=style["muted"], font=_font(10, style))
    headers = ["ID", "Item", "Qty", "Unit", "Region"]
    xs = [18, 92, 300, 352, 448]
    draw.rectangle([14, 62, w - 14, 88], fill=style["muted"])
    for x, head in zip(xs, headers):
        draw.text((x, 68), head, fill="#FFFFFF", font=_font(11, style, bold=True))
    for i, row in enumerate(rows):
        y = 92 + i * 34
        if i % 2 == 0:
            draw.rectangle([14, y, w - 14, y + 32], fill="#E8E8E8" if style["bg"] == "#FFFFFF" else "#E6E0D4")
        vals = [row["id"], row["item"], str(row["qty"]), f"{row['unit']:.2f}", row["region"]]
        for x, val in zip(xs, vals):
            draw.text((x, y + 9), str(val), fill=style["fg"], font=_font(11, style))
    draw.line([14, h - 8, w - 14, h - 8], fill=style["muted"], width=1)
    _finish(img, path, style, rng)


def make_doc_table(rng: random.Random, item_seed: int, split: str, out_dir: Path) -> dict[str, Any]:
    style = STYLES["hard" if split == "test_hard" else "clean"]
    hard = split == "test_hard"
    n = rng.randint(6, 8) + (3 if hard else 0)
    rows = []
    for i in range(n):
        rows.append(
            {
                "id": f"INV-{1000 + item_seed % 700 + i}",
                "item": rng.choice(PRODUCTS),
                "qty": rng.randint(1, 6),
                "unit": round(rng.uniform(3.0, 45.0), 2),
                "region": rng.choice(REGIONS),
            }
        )
    # guarantee the filter actually selects something
    rows[0]["region"] = "North"
    rows[0]["qty"] = max(3, rows[0]["qty"])
    title = rng.choice(["Invoice batch Q3", "Purchase ledger", "Quarterly vendor invoice"])
    path = out_dir / f"{split}_doc_table_{item_seed}.png"
    _render_table(rows, title, path, style, rng)

    selected = [r for r in rows if r["region"] == "North" and r["qty"] >= 3]
    total = round(sum(r["qty"] * r["unit"] for r in selected), 2)
    gold_calls = [
        _code(
            {
                "code": (
                    f"rows = {[(r['qty'], r['unit'], r['region']) for r in rows]}\n"
                    "sel = [q*u for q,u,reg in rows if reg == 'North' and q >= 3]\n"
                    "print(round(sum(sel), 2))"
                )
            }
        )
    ]
    if hard:
        total_all = round(sum(r["qty"] * r["unit"] for r in rows), 2)
        gold = round(total / total_all, 4) if total_all else 0.0
        code_b = f"print(round({total!r} / {total_all!r}, 4))"
        gold_calls.append(_code({"code": code_b}))
        question = (
            "The image is a vendor invoice table. Total the rows where region is North and qty is at "
            "least 3, then divide that total by the total value of all rows. Report the ratio to 4 decimals."
        )
        gold_answer = f"{gold:.4f}"
        difficulty = 3
    else:
        question = (
            "The image is a vendor invoice table. What is the total amount (qty * unit price) for rows "
            "where region is North and qty is at least 3? Round to 2 decimals."
        )
        gold_answer = f"{total:.2f}"
        difficulty = 1
    facts = {}
    for r in selected:
        facts[f"qty:{r['item']}:{r['id']}"] = r["qty"]
        facts[f"unit:{r['item']}:{r['id']}"] = r["unit"]
    for r in rows[-2:]:
        facts[f"distractor_qty:{r['item']}:{r['id']}"] = r["qty"]
    corpus = {
        "table_rows": "\n".join(
            f"{r['id']} | {r['item']} | qty={r['qty']} | unit={r['unit']:.2f} | region={r['region']}" for r in rows
        ),
        "table_title": title,
    }
    return {
        "image_paths": [str(path)],
        "question": question,
        "gold_answer": gold_answer,
        "gold_tool_calls": gold_calls,
        "required_tools": ["python_exec"] * len(gold_calls),
        "gold_perception_facts": facts,
        "retrieval_corpus": corpus,
        "difficulty": difficulty,
        "metadata": {"rows": rows, "title": title},
    }


# --------------------------------------------------------------------------- #
# Family 3: screenshot_ui
# --------------------------------------------------------------------------- #


def _render_dashboard(cards: dict[str, dict[str, Any]], path: Path, style: dict[str, Any], rng: random.Random) -> None:
    w, h = 560, 360
    img, draw = _canvas(w, h, style)
    draw.rectangle([0, 0, w, 54], fill=style["palette"][0])
    draw.text((20, 16), "Acme Bank - Accounts", fill="#FFFFFF", font=_font(18, style, bold=True))
    draw.text((430, 20), "online", fill="#D8F3DC", font=_font(11, style))
    for i, (eid, card) in enumerate(cards.items()):
        x = 20 + i * 268
        draw.rounded_rectangle([x, 78, x + 248, 300], radius=10, fill="#FFFFFF", outline=style["muted"], width=2)
        draw.rectangle([x, 78, x + 248, 118], fill=style["palette"][(i + 1) % len(style["palette"])])
        draw.text((x + 14, 90), f"{card['label']}", fill="#FFFFFF", font=_font(15, style, bold=True))
        draw.text((x + 180, 92), f"[{eid}]", fill="#FFFFFF", font=_font(10, style))
        draw.text((x + 14, 138), "balance", fill=style["muted"], font=_font(10, style))
        draw.text((x + 14, 154), f"${card['balance']:.2f}", fill=style["fg"], font=_font(20, style, bold=True))
        draw.text((x + 14, 194), "pending", fill=style["muted"], font=_font(10, style))
        draw.text((x + 14, 210), f"${card['pending']:.2f}", fill=style["fg"], font=_font(16, style))
        draw.rounded_rectangle([x + 14, 250, x + 120, 282], radius=6, fill=style["palette"][2])
        draw.text((x + 34, 258), f"[{eid}_open]", fill="#FFFFFF", font=_font(10, style))
    draw.text((20, 320), "Tip: open a card before reading its fields.", fill=style["muted"], font=_font(10, style))
    _finish(img, path, style, rng)


def make_screenshot_ui(rng: random.Random, item_seed: int, split: str, out_dir: Path) -> dict[str, Any]:
    style = STYLES["hard" if split == "test_hard" else "clean"]
    hard = split == "test_hard"
    n_cards = 2
    cards = {}
    for i in range(n_cards):
        cards[f"card_{i + 1}"] = {
            "label": rng.choice(["Checking", "Savings", "Business", "Joint"]),
            "balance": round(rng.uniform(120.0, 980.0), 2),
            "pending": round(rng.uniform(5.0, 90.0), 2),
        }
    path = out_dir / f"{split}_screenshot_ui_{item_seed}.png"
    _render_dashboard(cards, path, style, rng)

    if hard:
        gold = round(cards["card_2"]["balance"] - cards["card_2"]["pending"] - 25.0, 2)
        question = (
            "This is a banking app screenshot. Open the second card (element card_2), read its balance "
            "and pending fields, then report (balance - pending - 25.00) rounded to 2 decimals."
        )
        call_plan = [
            ("click", {"element_id": "card_2"}),
            ("read_field", {"name": "balance"}),
            ("read_field", {"name": "pending"}),
        ]
        difficulty = 3
    else:
        gold = round(cards["card_2"]["balance"] - cards["card_2"]["pending"], 2)
        question = (
            "This is a banking app screenshot. Open the second card (element card_2), read its "
            "balance and pending fields, and report balance minus pending rounded to 2 decimals."
        )
        call_plan = [
            ("click", {"element_id": "card_2"}),
            ("read_field", {"name": "balance"}),
            ("read_field", {"name": "pending"}),
        ]
        difficulty = 1
    gold_calls = [{"name": n, "args": a} for n, a in call_plan]
    facts = {}
    for eid, card in cards.items():
        facts[f"balance:{eid}"] = card["balance"]
        facts[f"pending:{eid}"] = card["pending"]
    return {
        "image_paths": [str(path)],
        "question": question,
        "gold_answer": f"{gold:.2f}",
        "gold_tool_calls": gold_calls,
        "required_tools": ["click", "read_field", "read_field"],
        "gold_perception_facts": facts,
        "retrieval_corpus": {
            "ui_fields": "; ".join(
                f"{eid}: balance={c['balance']:.2f}, pending={c['pending']:.2f}" for eid, c in cards.items()
            ),
            "ui_help": "Open a card with click(element_id) before reading its fields.",
        },
        "difficulty": difficulty,
        "metadata": {
            "ui": {
                "cards": cards,
                "fields": {f"{eid}:{name}": card[name] for eid, card in cards.items() for name in ("balance", "pending")},
            }
        },
    }


# --------------------------------------------------------------------------- #
# Family 4: multi_frame
# --------------------------------------------------------------------------- #


def _render_frames(counts: list[int], color_counts: list[int], path: Path, style: dict[str, Any], rng: random.Random) -> None:
    panel_w, panel_h, pad = 180, 260, 12
    w = pad + (panel_w + pad) * len(counts)
    h = panel_h + 68
    img, draw = _canvas(w, h, style)
    for f, (red_n, blue_n) in enumerate(zip(counts, color_counts)):
        x0 = pad + f * (panel_w + pad)
        y0 = 26
        draw.rectangle([x0, y0, x0 + panel_w, y0 + panel_h], fill="#FFFFFF", outline=style["muted"])
        draw.text((x0 + 6, 6), f"frame {f + 1}", fill=style["fg"], font=_font(12, style, bold=True))
        # draw a deterministic grid of crates, red first then blue
        cells = [(i % 4, i // 4) for i in range(12)]
        rng.shuffle(cells)
        for i in range(red_n + blue_n):
            cx, cy = cells[i]
            px = x0 + 16 + cx * 40
            py = y0 + 16 + cy * 52
            color = style["palette"][3] if i < red_n else style["palette"][0]
            draw.rounded_rectangle([px, py, px + 30, py + 30], radius=4, fill=color)
            draw.line([px, py + 10, px + 30, py + 10], fill="#FFFFFF", width=2)
    draw.text((pad, h - 20), "crates per frame (red = target, blue = distractor)", fill=style["muted"], font=_font(10, style))
    _finish(img, path, style, rng)


def make_multi_frame(rng: random.Random, item_seed: int, split: str, out_dir: Path) -> dict[str, Any]:
    style = STYLES["hard" if split == "test_hard" else "clean"]
    hard = split == "test_hard"
    red = [rng.randint(2, 5) for _ in range(3)]
    blue = [rng.randint(1, 4) for _ in range(3)]
    red[2] = red[0] + rng.randint(1, 3)
    if hard:
        blue[1] = min(8, blue[1] + 2)
    path = out_dir / f"{split}_multi_frame_{item_seed}.png"
    _render_frames(red, blue, path, style, rng)
    if hard:
        gold = (red[2] - red[0]) * blue[1] + red[1]
        base = (red[2] - red[0]) * blue[1]
        code_a = (
            f"red = {red}\nblue = {blue}\n"
            "print((red[2] - red[0]) * blue[1])"
        )
        code_b = f"print(round({base!r} + {red[1]!r}, 0))"
        question = (
            "The image is a 3-frame video strip of a warehouse. Count the red crates in each frame and "
            "the blue crates in frame 2. Report (red_frame3 - red_frame1) * blue_frame2 + red_frame2."
        )
        gold_answer = str(gold)
        difficulty = 3
        gold_calls = [_code({"code": code_a}), _code({"code": code_b})]
    else:
        gold = (red[2] - red[0]) * red[1]
        code = f"red = {red}\nprint((red[2] - red[0]) * red[1])"
        question = (
            "The image is a 3-frame video strip of a warehouse. Count the red crates in each frame, then "
            "report (red_frame3 - red_frame1) * red_frame2."
        )
        gold_answer = str(gold)
        difficulty = 1
        gold_calls = [_code({"code": code})]
    facts = {f"red_frame{i + 1}": v for i, v in enumerate(red)}
    if hard:
        facts["blue_frame2"] = blue[1]
    return {
        "image_paths": [str(path)],
        "question": question,
        "gold_answer": gold_answer,
        "gold_tool_calls": gold_calls,
        "required_tools": ["python_exec"] * len(gold_calls),
        "gold_perception_facts": facts,
        "retrieval_corpus": {
            "frame_counts": f"red={red}; blue={blue}",
            "frame_layout": "frame 1 left, frame 2 middle, frame 3 right; red crates are the target",
        },
        "difficulty": difficulty,
        "metadata": {"red_counts": red, "blue_counts": blue, "frames": 3},
    }


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

_BUILDERS = {
    "chart_agg": make_chart_agg,
    "doc_table": make_doc_table,
    "screenshot_ui": make_screenshot_ui,
    "multi_frame": make_multi_frame,
}


def generate_task(family: str, seed: int, split: str, index: int, out_dir: Path) -> TaskItem:
    """Generate one deterministic task."""
    if family not in _BUILDERS:
        raise ValueError(f"Unknown family {family!r}")
    rng = random.Random(seed)
    payload = _BUILDERS[family](rng, seed, split, out_dir)
    task_id = f"{split}-{family}-{index:04d}-{seed % 100000}"
    return TaskItem(
        task_id=task_id,
        family=family,
        seed=seed,
        split=split,
        difficulty=int(payload["difficulty"]),
        image_paths=payload["image_paths"],
        question=payload["question"],
        gold_answer=payload["gold_answer"],
        gold_tool_calls=payload["gold_tool_calls"],
        gold_perception_facts=payload["gold_perception_facts"],
        required_tools=payload["required_tools"],
        retrieval_corpus=payload["retrieval_corpus"],
        metadata=payload["metadata"],
    )
