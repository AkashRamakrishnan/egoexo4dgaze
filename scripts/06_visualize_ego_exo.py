#!/usr/bin/env python3
"""Side-by-side ego + exo QA plots for random Tier-A labeled images."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from egoexo4d.gaze_project import get_projector
from egoexo4d.media import (
    calib_label_for_video,
    find_aria_rgb_video,
    read_video_frame,
    scale_gaze,
)
from egoexo4d.paths import load_config, manifests_dir, output_root, take_dir

OBJ_COLOR = (255, 60, 60)
GAZE_COLOR = (0, 220, 80)
TEXT_BG = (20, 20, 20)
TEXT_FG = (245, 245, 245)
PANEL_BG = (12, 12, 12)


def load_font(size: int):
    for name in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "Arial.ttf",
    ):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def bgr_to_pil(frame) -> Image.Image:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def draw_gaze(draw: ImageDraw.ImageDraw, x: float, y: float, line_w: int) -> None:
    r = max(6, line_w * 3)
    arm = max(10, line_w * 6)
    draw.ellipse((x - r, y - r, x + r, y + r), outline=GAZE_COLOR, width=line_w)
    draw.line((x - arm, y, x + arm, y), fill=GAZE_COLOR, width=line_w)
    draw.line((x, y - arm, x, y + arm), fill=GAZE_COLOR, width=line_w)


def draw_label_box(draw: ImageDraw.ImageDraw, xy, text: str, font, line_w: int) -> None:
    x1, y1, x2, y2 = xy
    draw.rectangle((x1, y1, x2, y2), outline=OBJ_COLOR, width=line_w)
    pad = 4
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    tx, ty = x1, max(0, y1 - th - 2 * pad)
    draw.rectangle((tx, ty, tx + tw + 2 * pad, ty + th + 2 * pad), fill=OBJ_COLOR)
    draw.text((tx + pad, ty + pad), text, fill=(255, 255, 255), font=font)


def caption_panel(img: Image.Image, title: str, subtitle: str, font) -> Image.Image:
    cap_h = 56
    canvas = Image.new("RGB", (img.width, img.height + cap_h), TEXT_BG)
    canvas.paste(img, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, img.height + 6), title, fill=TEXT_FG, font=font)
    draw.text((8, img.height + 28), subtitle, fill=(180, 180, 180), font=font)
    return canvas


def unique_image_rows(rows: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for row in rows:
        key = row["image_path"]
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def render_pair(row: dict, cfg: dict) -> Image.Image:
    take_path = take_dir(cfg, row["take_name"])
    exo_video = take_path / "frame_aligned_videos" / f"{row['cam']}.mp4"
    ego_video = find_aria_rgb_video(take_path)
    frame_idx = int(row["frame_idx"])

    exo_bgr = read_video_frame(exo_video, frame_idx)
    if exo_bgr is None:
        raise FileNotFoundError(f"exo frame missing: {exo_video} @ {frame_idx}")
    exo = bgr_to_pil(exo_bgr)

    ego_note = "ego video missing"
    ego = Image.new("RGB", (exo.width, exo.height), (40, 40, 40))
    if ego_video is not None:
        ego_bgr = read_video_frame(ego_video, frame_idx)
        if ego_bgr is not None:
            ego = bgr_to_pil(ego_bgr)
            ego_note = ego_video.name
        else:
            ego_note = f"ego read failed ({ego_video.name})"

    gaze_cfg = cfg.get("gaze", {})
    projector = get_projector(
        str(take_path),
        bool(gaze_cfg.get("prefer_personalized", True)),
        float(gaze_cfg.get("calib_min_quality", 1.0)),
    )

    line_w = max(2, int(min(exo.size) * 0.005))
    font = load_font(max(16, int(min(exo.size) * 0.028)))
    small = load_font(max(14, int(min(exo.size) * 0.022)))

    exo_draw = ImageDraw.Draw(exo)
    draw_gaze(exo_draw, float(row["gaze_x"]), float(row["gaze_y"]), line_w)
    draw_label_box(
        exo_draw,
        (
            float(row["box_x1"]),
            float(row["box_y1"]),
            float(row["box_x2"]),
            float(row["box_y2"]),
        ),
        str(row.get("label_text", "")),
        font,
        line_w,
    )
    exo_sub = (
        f"gaze=({float(row['gaze_x']):.0f},{float(row['gaze_y']):.0f})  "
        f"{row['cam']}  frame {frame_idx:06d}"
    )
    exo_panel = caption_panel(exo, f"Exocentric  |  {row.get('label_text', '')}", exo_sub, small)

    ego_draw = ImageDraw.Draw(ego)
    ego_sub = ego_note
    if ego_video is not None:
        try:
            packed = projector.project_ego_gaze(
                frame_idx,
                timestamp_cam=row["cam"],
                calib_label=calib_label_for_video(ego_video),
            )
        except Exception as exc:
            packed = None
            ego_sub = f"{ego_note}  |  ego gaze failed: {exc}"
        if packed is not None:
            gx, gy, cal_wh = packed
            gx, gy = scale_gaze((gx, gy), cal_wh, ego.size)
            draw_gaze(ego_draw, gx, gy, max(2, int(min(ego.size) * 0.005)))
            ego_sub = f"{ego_note}  |  gaze=({gx:.0f},{gy:.0f})"
        elif "failed" not in ego_sub:
            ego_sub = f"{ego_note}  |  ego gaze unavailable"
    ego_panel = caption_panel(ego, "Egocentric  |  Aria RGB + GT gaze", ego_sub, small)

    gap = 16
    header_h = 48
    width = ego_panel.width + gap + exo_panel.width
    height = header_h + max(ego_panel.height, exo_panel.height)
    canvas = Image.new("RGB", (width, height), PANEL_BG)
    draw = ImageDraw.Draw(canvas)
    header = (
        f"{row['take_name']}  |  {row['take_uid'][:8]}…  |  "
        f"{row.get('label_text', '')}  |  {row['image_path']}"
    )
    draw.text((12, 12), header, fill=TEXT_FG, font=small)
    canvas.paste(ego_panel, (0, header_h))
    canvas.paste(exo_panel, (ego_panel.width + gap, header_h))
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--total", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    rows_path = manifests_dir(cfg) / "tier_a_rows.json"
    if not rows_path.exists():
        raise FileNotFoundError(
            f"Missing {rows_path}. Run scripts/03_build_dataset.py first."
        )
    with rows_path.open() as f:
        rows = json.load(f)

    unique = unique_image_rows(rows)
    if not unique:
        raise RuntimeError("No Tier-A labeled images to visualize.")
    if len(unique) < args.total:
        raise RuntimeError(
            f"Only {len(unique)} labeled images (need {args.total}). "
            "Increase build.max_takes or check MPS/Relations coverage."
        )

    rng = random.Random(args.seed)
    picks = rng.sample(unique, args.total)
    out_dir = args.out or (output_root(cfg) / "viz_ego_exo")
    out_dir.mkdir(parents=True, exist_ok=True)

    failed = []
    for i, row in enumerate(picks, start=1):
        out_name = f"sample_{i:02d}_{row['cam']}_{int(row['frame_idx']):06d}.jpg"
        try:
            render_pair(row, cfg).save(out_dir / out_name, quality=92)
            print(f"[viz] {out_name}  {row.get('label_text', '')}")
        except Exception as exc:
            failed.append((out_name, str(exc)))
            print(f"[viz] FAIL {out_name}: {exc}")

    if failed:
        raise RuntimeError(f"{len(failed)}/{args.total} plots failed: {failed[:3]}")
    print(f"Saved {args.total} ego/exo plots -> {out_dir}")


if __name__ == "__main__":
    main()
