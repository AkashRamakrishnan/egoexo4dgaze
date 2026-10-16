#!/usr/bin/env python3
"""Visualize random Ego-Exo4D Tier-A samples."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from egoexo4d.paths import annotations_dir, images_dir, load_config, output_root

HEAD_COLOR = (0, 190, 255)
OBJ_COLOR = (255, 60, 60)
GAZE_COLOR = (0, 255, 80)
TEXT_BG = (20, 20, 20)
TEXT_FG = (235, 235, 235)


def load_font(size: int):
    for name in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "Arial.ttf",
    ):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def parse_head_box(cell: str) -> tuple[float, float, float, float] | None:
    import ast

    try:
        boxes = ast.literal_eval(cell)
        if boxes and len(boxes[0]) == 4:
            return tuple(float(v) for v in boxes[0])
    except (SyntaxError, ValueError, TypeError):
        return None
    return None


def render(row: dict, image_path: Path, caption_h: int = 44) -> Image.Image:
    with Image.open(image_path) as img:
        img = img.convert("RGB")
    w, h = img.size
    line_w = max(2, int(min(w, h) * 0.006))
    draw = ImageDraw.Draw(img)

    obj = (
        float(row["box_x1"]),
        float(row["box_y1"]),
        float(row["box_x2"]),
        float(row["box_y2"]),
    )
    draw.rectangle(obj, outline=OBJ_COLOR, width=line_w)

    head = parse_head_box(row["head_boxes"])
    if head:
        draw.rectangle(head, outline=HEAD_COLOR, width=line_w)

    gx, gy = float(row["gaze_x"]), float(row["gaze_y"])
    r = max(4, line_w * 2)
    draw.ellipse((gx - r, gy - r, gx + r, gy + r), fill=GAZE_COLOR)

    caption = f"{row.get('label_text', '')} @ {row['image_path']}"
    font = load_font(max(14, int(min(w, h) * 0.028)))
    canvas = Image.new("RGB", (w, h + caption_h), TEXT_BG)
    canvas.paste(img, (0, 0))
    cap_draw = ImageDraw.Draw(canvas)
    cap_draw.text((8, h + 8), caption, fill=TEXT_FG, font=font)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--images", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--total", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    csv_path = args.csv or (annotations_dir(cfg) / "train_verified.csv")
    img_root = args.images or images_dir(cfg)
    out_dir = args.out or (output_root(cfg) / "viz_samples")

    with csv_path.open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if (img_root / r["image_path"]).exists()]

    if len(rows) < args.total:
        raise RuntimeError(f"Only {len(rows)} renderable rows (need {args.total})")

    rng = random.Random(args.seed)
    picks = rng.sample(rows, args.total)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, row in enumerate(picks, start=1):
        src = img_root / row["image_path"]
        out_name = f"sample_{i:02d}_{Path(row['image_path']).name}"
        render(row, src).save(out_dir / out_name, quality=95)

    print(f"Saved {args.total} samples -> {out_dir}")


if __name__ == "__main__":
    main()
