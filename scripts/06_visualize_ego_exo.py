#!/usr/bin/env python3
"""Side-by-side ego + exo QA plots for random Tier-A labeled images.

Paper figures (no captions, matched height, object mask + bbox + label,
10 gaze-in-mask and 10 gaze-out-of-mask samples per take):

    python scripts/06_visualize_ego_exo.py --config configs/cooking_mvp.yaml \\
        --paper --per-bucket 10 --out ./viz_ego_exo_paper
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from egoexo4d.gaze_project import get_projector
from egoexo4d.masks import decode_mask_dict, mask_to_bbox, point_in_mask
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

_PROJECTORS: dict[str, object] = {}


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


def sample_distinct_takes(rows: list[dict], total: int, rng: random.Random) -> list[dict]:
    """At most one image per take_uid, shuffled."""
    shuffled = list(rows)
    rng.shuffle(shuffled)
    picked = []
    seen = set()
    for row in shuffled:
        uid = row.get("take_uid")
        if uid in seen:
            continue
        seen.add(uid)
        picked.append(row)
        if len(picked) >= total:
            break
    return picked


def match_panel_height(ego: Image.Image, exo: Image.Image) -> tuple[Image.Image, Image.Image]:
    """Same height, keep each aspect ratio (letterbox-free side-by-side)."""
    h = max(ego.height, exo.height)

    def _fit(im: Image.Image) -> Image.Image:
        if im.height == h:
            return im
        w = max(1, int(round(im.width * h / im.height)))
        return im.resize((w, h), Image.Resampling.LANCZOS)

    return _fit(ego), _fit(exo)


def overlay_mask(
    img: Image.Image,
    mask: np.ndarray,
    color: tuple[int, int, int] = OBJ_COLOR,
    alpha: float = 0.38,
) -> Image.Image:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    h, w = arr.shape[:2]
    if mask.shape[:2] != (h, w):
        mask_im = Image.fromarray((mask.astype(np.uint8) * 255))
        mask_im = mask_im.resize((w, h), Image.Resampling.NEAREST)
        hit = np.asarray(mask_im) > 127
    else:
        hit = mask > 0
    tint = np.array(color, dtype=np.float32)
    arr[hit] = (1.0 - alpha) * arr[hit] + alpha * tint
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def encoded_to_mask(rec: dict) -> np.ndarray | None:
    encoded = rec.get("encoded_mask") or rec.get("encodedMask")
    if encoded is None:
        return None
    if isinstance(encoded, bytes):
        encoded = encoded.decode()
    width = rec.get("mask_width", rec.get("width"))
    height = rec.get("mask_height", rec.get("height"))
    if width is None or height is None:
        return None
    try:
        return decode_mask_dict(
            {"width": int(width), "height": int(height), "encodedMask": encoded}
        )
    except Exception:
        return None


def load_mask_index(cfg: dict) -> dict[tuple, dict]:
    path = manifests_dir(cfg) / "frame_index.parquet"
    if not path.exists():
        return {}
    import pandas as pd

    df = pd.read_parquet(path)
    index = {}
    for rec in df.to_dict(orient="records"):
        key = (
            str(rec["take_uid"]),
            str(rec["cam"]),
            int(rec["frame_idx"]),
            str(rec.get("track_name", "")),
        )
        index[key] = rec
    return index


def mask_for_row(row: dict, mask_index: dict[tuple, dict]) -> np.ndarray | None:
    mask = encoded_to_mask(row)
    if mask is not None:
        return mask
    key = (
        str(row["take_uid"]),
        str(row["cam"]),
        int(row["frame_idx"]),
        str(row.get("track_name", "")),
    )
    rec = mask_index.get(key)
    if rec is None:
        return None
    return encoded_to_mask(rec)


def projector_for(cfg: dict, take_name: str):
    take_path = str(take_dir(cfg, take_name))
    cached = _PROJECTORS.get(take_path)
    if cached is not None:
        if isinstance(cached, Exception):
            raise cached
        return cached
    gaze_cfg = cfg.get("gaze", {})
    try:
        projector = get_projector(
            take_path,
            bool(gaze_cfg.get("prefer_personalized", True)),
            float(gaze_cfg.get("calib_min_quality", 1.0)),
        )
    except Exception as exc:
        _PROJECTORS[take_path] = exc
        raise
    _PROJECTORS[take_path] = projector
    return projector


def viz_row_from_record(rec: dict, gaze: tuple[float, float], box: tuple[float, float, float, float]) -> dict:
    take_uid = str(rec["take_uid"])
    cam = str(rec["cam"])
    frame_idx = int(rec["frame_idx"])
    return {
        "take_uid": take_uid,
        "take_name": rec["take_name"],
        "cam": cam,
        "frame_idx": frame_idx,
        "image_path": f"{take_uid}/{cam}/{frame_idx:06d}.jpg",
        "gaze_x": float(gaze[0]),
        "gaze_y": float(gaze[1]),
        "box_x1": float(box[0]),
        "box_y1": float(box[1]),
        "box_x2": float(box[2]),
        "box_y2": float(box[3]),
        "label_text": rec.get("label_text", ""),
        "track_name": rec.get("track_name", ""),
        "encoded_mask": rec.get("encoded_mask"),
        "mask_width": rec.get("mask_width"),
        "mask_height": rec.get("mask_height"),
    }


def sample_in_out_for_take(
    records: list[dict],
    cfg: dict,
    n: int,
    rng: random.Random,
) -> tuple[list[dict], list[dict]] | None:
    """Return n gaze-in-mask and n gaze-out-of-mask rows, or None if a take is short."""
    if not records:
        return None
    take_name = str(records[0]["take_name"])
    try:
        projector = projector_for(cfg, take_name)
    except Exception as exc:
        print(f"[viz] skip {take_name}: projector failed ({exc})")
        return None

    shuffled = list(records)
    rng.shuffle(shuffled)
    inside: list[dict] = []
    outside: list[dict] = []
    used: set[tuple] = set()
    gaze_cache: dict[tuple, tuple[float, float] | None] = {}

    for rec in shuffled:
        if len(inside) >= n and len(outside) >= n:
            break
        cam = str(rec["cam"])
        frame_idx = int(rec["frame_idx"])
        frame_key = (cam, frame_idx)
        if frame_key in used:
            continue
        if frame_key not in gaze_cache:
            try:
                gaze_cache[frame_key] = projector.project_gaze(cam, frame_idx)
            except Exception:
                gaze_cache[frame_key] = None
        gaze = gaze_cache[frame_key]
        if gaze is None:
            continue
        mask = encoded_to_mask(rec)
        if mask is None:
            continue
        try:
            box = mask_to_bbox(mask)
        except ValueError:
            continue
        row = viz_row_from_record(rec, gaze, box)
        hit = point_in_mask(float(gaze[0]), float(gaze[1]), mask)
        if hit and len(inside) < n:
            inside.append(row)
            used.add(frame_key)
        elif (not hit) and len(outside) < n:
            outside.append(row)
            used.add(frame_key)

    if len(inside) < n or len(outside) < n:
        print(
            f"[viz] skip {take_name}: in_mask={len(inside)}/{n} "
            f"out_mask={len(outside)}/{n}"
        )
        return None
    return inside, outside


def render_pair(
    row: dict,
    cfg: dict,
    *,
    no_captions: bool = False,
    draw_label: bool = True,
    match_height: bool = False,
    overlay_object_mask: bool = False,
    mask_index: dict[tuple, dict] | None = None,
) -> Image.Image:
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

    projector = projector_for(cfg, row["take_name"])

    if overlay_object_mask:
        mask = mask_for_row(row, mask_index or {})
        if mask is not None:
            exo = overlay_mask(exo, mask)

    line_w = max(2, int(min(exo.size) * 0.005))
    font = load_font(max(16, int(min(exo.size) * 0.028)))
    small = load_font(max(14, int(min(exo.size) * 0.022)))
    box = (
        float(row["box_x1"]),
        float(row["box_y1"]),
        float(row["box_x2"]),
        float(row["box_y2"]),
    )

    exo_draw = ImageDraw.Draw(exo)
    draw_gaze(exo_draw, float(row["gaze_x"]), float(row["gaze_y"]), line_w)
    if draw_label:
        draw_label_box(exo_draw, box, str(row.get("label_text", "")), font, line_w)
    else:
        exo_draw.rectangle(box, outline=OBJ_COLOR, width=line_w)

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

    if match_height:
        ego, exo = match_panel_height(ego, exo)

    if no_captions:
        canvas = Image.new("RGB", (ego.width + exo.width, max(ego.height, exo.height)), PANEL_BG)
        canvas.paste(ego, (0, 0))
        canvas.paste(exo, (ego.width, 0))
        return canvas

    exo_sub = (
        f"gaze=({float(row['gaze_x']):.0f},{float(row['gaze_y']):.0f})  "
        f"{row['cam']}  frame {frame_idx:06d}"
    )
    exo_panel = caption_panel(exo, f"Exocentric  |  {row.get('label_text', '')}", exo_sub, small)
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


def save_row(
    row: dict,
    cfg: dict,
    out_path: Path,
    *,
    no_captions: bool,
    draw_label: bool,
    match_height: bool,
    overlay_object_mask: bool,
    mask_index: dict[tuple, dict],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    render_pair(
        row,
        cfg,
        no_captions=no_captions,
        draw_label=draw_label,
        match_height=match_height,
        overlay_object_mask=overlay_object_mask,
        mask_index=mask_index,
    ).save(out_path, quality=92)


def run_paper(args, cfg: dict) -> None:
    parquet = manifests_dir(cfg) / "frame_index.parquet"
    if not parquet.exists():
        raise FileNotFoundError(
            f"Missing {parquet}. Run scripts/02_build_index.py first."
        )
    import pandas as pd

    df = pd.read_parquet(parquet)
    if df.empty:
        raise RuntimeError("Empty frame_index.parquet")

    by_take: dict[str, list[dict]] = defaultdict(list)
    take_names: dict[str, str] = {}
    for rec in df.to_dict(orient="records"):
        uid = str(rec["take_uid"])
        by_take[uid].append(rec)
        take_names[uid] = str(rec["take_name"])

    rng = random.Random(args.seed)
    take_uids = list(by_take.keys())
    rng.shuffle(take_uids)
    if args.takes is not None:
        take_uids = take_uids[: args.takes]

    out_dir = args.out or (output_root(cfg) / "viz_ego_exo_paper")
    out_dir.mkdir(parents=True, exist_ok=True)

    n = args.per_bucket
    used_takes = 0
    n_ok = 0
    failed: list[tuple[str, str]] = []

    for uid in take_uids:
        sampled = sample_in_out_for_take(by_take[uid], cfg, n, rng)
        if sampled is None:
            continue
        inside, outside = sampled
        used_takes += 1
        take_name = take_names[uid]
        take_dir_out = out_dir / take_name
        print(f"[viz] {take_name}: {n} in_mask + {n} out_mask")
        for bucket, rows in (("in_mask", inside), ("out_mask", outside)):
            for i, row in enumerate(rows, start=1):
                out_name = f"{i:02d}_{row['cam']}_{int(row['frame_idx']):06d}.jpg"
                out_path = take_dir_out / bucket / out_name
                try:
                    save_row(
                        row,
                        cfg,
                        out_path,
                        no_captions=True,
                        draw_label=not args.no_label,
                        match_height=True,
                        overlay_object_mask=True,
                        mask_index={},
                    )
                    n_ok += 1
                    print(f"[viz] {take_name}/{bucket}/{out_name}  {row.get('label_text', '')}")
                except Exception as exc:
                    failed.append((str(out_path), str(exc)))
                    print(f"[viz] FAIL {out_path}: {exc}")

    if used_takes == 0:
        raise RuntimeError(
            f"No take had {n} gaze-in-mask and {n} gaze-out-of-mask frames."
        )
    if failed:
        raise RuntimeError(f"{len(failed)} plots failed: {failed[:3]}")
    print(f"Saved {n_ok} plots from {used_takes} takes -> {out_dir}")


def run_qa(args, cfg: dict) -> None:
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
    if not args.distinct_takes and len(unique) < args.total:
        raise RuntimeError(
            f"Only {len(unique)} labeled images (need {args.total}). "
            "Increase build.max_takes or check MPS/Relations coverage."
        )

    rng = random.Random(args.seed)
    if args.distinct_takes:
        picks = sample_distinct_takes(unique, args.total, rng)
        if len(picks) < args.total:
            print(
                f"[viz] only {len(picks)} distinct takes available "
                f"(requested {args.total})"
            )
    else:
        picks = rng.sample(unique, args.total)
    out_dir = args.out or (output_root(cfg) / "viz_ego_exo")
    out_dir.mkdir(parents=True, exist_ok=True)

    mask_index = load_mask_index(cfg) if args.overlay_mask else {}
    if args.overlay_mask and not mask_index:
        print("[viz] WARNING: frame_index.parquet missing; masks will be skipped")

    failed = []
    for i, row in enumerate(picks, start=1):
        uid8 = str(row.get("take_uid", "take"))[:8]
        out_name = f"sample_{i:02d}_{uid8}_{row['cam']}_{int(row['frame_idx']):06d}.jpg"
        try:
            save_row(
                row,
                cfg,
                out_dir / out_name,
                no_captions=args.no_text,
                draw_label=not args.no_label,
                match_height=args.match_height,
                overlay_object_mask=args.overlay_mask,
                mask_index=mask_index,
            )
            print(f"[viz] {out_name}  {row.get('label_text', '')}  {row.get('take_name', '')}")
        except Exception as exc:
            failed.append((out_name, str(exc)))
            print(f"[viz] FAIL {out_name}: {exc}")

    n_ok = len(picks) - len(failed)
    if failed:
        raise RuntimeError(f"{len(failed)}/{len(picks)} plots failed: {failed[:3]}")
    print(f"Saved {n_ok} ego/exo plots -> {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--total", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--paper",
        action="store_true",
        help="No captions, match height, mask + bbox + label, 10 in/10 out per take.",
    )
    parser.add_argument("--no-text", action="store_true", help="Drop captions/header.")
    parser.add_argument("--no-label", action="store_true", help="Drop object-name chip.")
    parser.add_argument("--match-height", action="store_true")
    parser.add_argument("--overlay-mask", action="store_true")
    parser.add_argument("--distinct-takes", action="store_true")
    parser.add_argument("--per-bucket", type=int, default=10, help="In-mask and out-of-mask count per take.")
    parser.add_argument("--takes", type=int, default=None, help="Max takes for --paper (default: all that qualify).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.paper:
        run_paper(args, cfg)
    else:
        run_qa(args, cfg)


if __name__ == "__main__":
    main()
