#!/usr/bin/env python3
"""Side-by-side ego + exo QA plots for random Tier-A labeled images.

Paper figures (no captions, matched height, object mask + bbox + label,
10 gaze-in-mask and 10 gaze-out-of-mask samples per scenario):

    python scripts/06_visualize_ego_exo.py --paper --all-scenarios \\
        --per-bucket 10 --out ./viz_ego_exo_paper
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
MASK_PALETTE = (
    (255, 60, 60),
    (50, 120, 255),
    (255, 160, 0),
    (190, 70, 220),
    (0, 190, 200),
    (255, 220, 0),
    (40, 180, 90),
    (255, 90, 160),
)

_PROJECTORS: dict[str, object] = {}

PAPER_SCENARIO_CONFIGS = (
    "configs/cooking_mvp.yaml",
    "configs/health.yaml",
    "configs/bike_repair.yaml",
    "configs/music.yaml",
    "configs/basketball.yaml",
    "configs/soccer.yaml",
)


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


def draw_label_box(
    draw: ImageDraw.ImageDraw,
    xy,
    text: str,
    font,
    line_w: int,
    color: tuple[int, int, int] = OBJ_COLOR,
    *,
    emphasize: bool = False,
) -> None:
    x1, y1, x2, y2 = xy
    if emphasize:
        draw.rectangle((x1, y1, x2, y2), outline=(255, 255, 255), width=line_w + 3)
    draw.rectangle((x1, y1, x2, y2), outline=color, width=line_w)
    pad = 4
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    tx, ty = x1, max(0, y1 - th - 2 * pad)
    draw.rectangle((tx, ty, tx + tw + 2 * pad, ty + th + 2 * pad), fill=color)
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


def object_from_record(rec: dict, gaze: tuple[float, float]) -> dict | None:
    mask = encoded_to_mask(rec)
    if mask is None:
        return None
    try:
        box = mask_to_bbox(mask)
    except ValueError:
        return None
    return {
        "label_text": rec.get("label_text", ""),
        "track_name": rec.get("track_name", ""),
        "box_x1": float(box[0]),
        "box_y1": float(box[1]),
        "box_x2": float(box[2]),
        "box_y2": float(box[3]),
        "encoded_mask": rec.get("encoded_mask"),
        "mask_width": rec.get("mask_width"),
        "mask_height": rec.get("mask_height"),
        "gaze_hit": point_in_mask(float(gaze[0]), float(gaze[1]), mask),
    }


def viz_row_from_frame(
    recs: list[dict],
    gaze: tuple[float, float],
    objects: list[dict],
) -> dict:
    rec = recs[0]
    take_uid = str(rec["take_uid"])
    cam = str(rec["cam"])
    frame_idx = int(rec["frame_idx"])
    primary = next((obj for obj in objects if obj.get("gaze_hit")), objects[0])
    return {
        "take_uid": take_uid,
        "take_name": rec["take_name"],
        "cam": cam,
        "frame_idx": frame_idx,
        "image_path": f"{take_uid}/{cam}/{frame_idx:06d}.jpg",
        "gaze_x": float(gaze[0]),
        "gaze_y": float(gaze[1]),
        "box_x1": primary["box_x1"],
        "box_y1": primary["box_y1"],
        "box_x2": primary["box_x2"],
        "box_y2": primary["box_y2"],
        "label_text": ", ".join(
            str(obj.get("label_text", "")) for obj in objects if obj.get("label_text")
        ),
        "track_name": primary.get("track_name", ""),
        "encoded_mask": primary.get("encoded_mask"),
        "mask_width": primary.get("mask_width"),
        "mask_height": primary.get("mask_height"),
        "objects": objects,
        "gaze_in_any_mask": any(obj.get("gaze_hit") for obj in objects),
    }


def classify_frame(
    recs: list[dict],
    projector,
    used_frames: set[tuple],
    gaze_cache: dict[tuple, tuple[float, float] | None],
) -> tuple[dict, bool, tuple] | None:
    rec = recs[0]
    cam = str(rec["cam"])
    frame_idx = int(rec["frame_idx"])
    frame_key = (str(rec["take_uid"]), cam, frame_idx)
    if frame_key in used_frames:
        return None
    if frame_key not in gaze_cache:
        try:
            gaze_cache[frame_key] = projector.project_gaze(cam, frame_idx)
        except Exception:
            gaze_cache[frame_key] = None
    gaze = gaze_cache[frame_key]
    if gaze is None:
        return None
    objects = []
    for rec in recs:
        obj = object_from_record(rec, gaze)
        if obj is not None:
            objects.append(obj)
    if not objects:
        return None
    row = viz_row_from_frame(recs, gaze, objects)
    return row, bool(row["gaze_in_any_mask"]), frame_key


def sample_in_out_for_scenario(
    records: list[dict],
    cfg: dict,
    n: int,
    rng: random.Random,
) -> tuple[list[dict], list[dict]] | None:
    """n frames with gaze in any mask and n with gaze in none, preferring distinct takes."""
    if not records:
        return None
    frames_by_take: dict[str, dict[tuple, list[dict]]] = defaultdict(dict)
    take_name_by_uid: dict[str, str] = {}
    for rec in records:
        uid = str(rec["take_uid"])
        frame_key = (uid, str(rec["cam"]), int(rec["frame_idx"]))
        frames_by_take[uid].setdefault(frame_key, []).append(rec)
        take_name_by_uid[uid] = str(rec["take_name"])
    take_uids = list(frames_by_take)
    rng.shuffle(take_uids)

    inside: list[dict] = []
    outside: list[dict] = []
    used_frames: set[tuple] = set()
    used_takes: set[str] = set()
    gaze_cache: dict[tuple, tuple[float, float] | None] = {}
    projectors: dict[str, object | None] = {}

    def projector_for_uid(uid: str):
        take_name = take_name_by_uid[uid]
        if take_name not in projectors:
            try:
                projectors[take_name] = projector_for(cfg, take_name)
            except Exception as exc:
                print(f"[viz] skip {take_name}: projector failed ({exc})")
                projectors[take_name] = None
        return projectors[take_name]

    def scan_take(uid: str) -> tuple[tuple[dict, tuple] | None, tuple[dict, tuple] | None]:
        projector = projector_for_uid(uid)
        if projector is None:
            return None, None
        frame_items = list(frames_by_take[uid].items())
        rng.shuffle(frame_items)
        found_in = found_out = None
        for _key, recs in frame_items:
            packed = classify_frame(recs, projector, used_frames, gaze_cache)
            if packed is None:
                continue
            row, hit, key = packed
            if hit and found_in is None:
                found_in = (row, key)
            elif (not hit) and found_out is None:
                found_out = (row, key)
            if found_in is not None and found_out is not None:
                break
        return found_in, found_out

    for uid in take_uids:
        if len(inside) >= n and len(outside) >= n:
            break
        found_in, found_out = scan_take(uid)
        options = []
        if len(inside) < n and found_in is not None:
            options.append(("in", found_in))
        if len(outside) < n and found_out is not None:
            options.append(("out", found_out))
        if not options:
            continue
        bucket, (row, key) = rng.choice(options)
        if bucket == "in":
            inside.append(row)
        else:
            outside.append(row)
        used_frames.add(key)
        used_takes.add(uid)

    if len(inside) < n or len(outside) < n:
        leftover = [u for u in take_uids if u not in used_takes] + [
            u for u in take_uids if u in used_takes
        ]
        for uid in leftover:
            if len(inside) >= n and len(outside) >= n:
                break
            found_in, found_out = scan_take(uid)
            if len(inside) < n and found_in is not None and found_in[1] not in used_frames:
                inside.append(found_in[0])
                used_frames.add(found_in[1])
            if len(outside) < n and found_out is not None and found_out[1] not in used_frames:
                outside.append(found_out[0])
                used_frames.add(found_out[1])

    if len(inside) < n or len(outside) < n:
        print(f"[viz] not enough samples: in_mask={len(inside)}/{n} out_mask={len(outside)}/{n}")
        return None
    return inside[:n], outside[:n]


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

    objects = list(row.get("objects") or [])
    if not objects:
        objects = [
            {
                "label_text": row.get("label_text", ""),
                "box_x1": row["box_x1"],
                "box_y1": row["box_y1"],
                "box_x2": row["box_x2"],
                "box_y2": row["box_y2"],
                "encoded_mask": row.get("encoded_mask"),
                "mask_width": row.get("mask_width"),
                "mask_height": row.get("mask_height"),
                "track_name": row.get("track_name", ""),
                "gaze_hit": True,
            }
        ]

    if overlay_object_mask:
        for i, obj in enumerate(objects):
            color = MASK_PALETTE[i % len(MASK_PALETTE)]
            mask = encoded_to_mask(obj)
            if mask is None:
                mask = mask_for_row(
                    {
                        **row,
                        "track_name": obj.get("track_name", row.get("track_name", "")),
                        "encoded_mask": obj.get("encoded_mask"),
                        "mask_width": obj.get("mask_width"),
                        "mask_height": obj.get("mask_height"),
                    },
                    mask_index or {},
                )
            if mask is not None:
                alpha = 0.46 if obj.get("gaze_hit") else 0.30
                exo = overlay_mask(exo, mask, color, alpha)

    line_w = max(2, int(min(exo.size) * 0.005))
    font = load_font(max(16, int(min(exo.size) * 0.028)))
    small = load_font(max(14, int(min(exo.size) * 0.022)))

    exo_draw = ImageDraw.Draw(exo)
    for i, obj in enumerate(objects):
        color = MASK_PALETTE[i % len(MASK_PALETTE)]
        box = (
            float(obj["box_x1"]),
            float(obj["box_y1"]),
            float(obj["box_x2"]),
            float(obj["box_y2"]),
        )
        if draw_label:
            draw_label_box(
                exo_draw,
                box,
                str(obj.get("label_text", "")),
                font,
                line_w,
                color,
                emphasize=bool(obj.get("gaze_hit")),
            )
        else:
            exo_draw.rectangle(box, outline=color, width=line_w)
    draw_gaze(exo_draw, float(row["gaze_x"]), float(row["gaze_y"]), line_w)

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


def scenario_slug(cfg: dict) -> str:
    return str(cfg.get("scenario", {}).get("slug", "scenario"))


def run_paper_scenario(args, cfg: dict, out_dir: Path) -> int:
    parquet = manifests_dir(cfg) / "frame_index.parquet"
    slug = scenario_slug(cfg)
    if not parquet.exists():
        raise FileNotFoundError(
            f"Missing {parquet} for {slug}. Run scripts/02_build_index.py first."
        )
    import pandas as pd

    df = pd.read_parquet(parquet)
    if df.empty:
        raise RuntimeError(f"Empty frame_index.parquet for {slug}")

    rng = random.Random(args.seed)
    n = args.per_bucket
    sampled = sample_in_out_for_scenario(df.to_dict(orient="records"), cfg, n, rng)
    if sampled is None:
        raise RuntimeError(
            f"{slug}: could not find {n} gaze-in-mask and {n} gaze-out-of-mask frames."
        )
    inside, outside = sampled
    n_ok = 0
    failed: list[tuple[str, str]] = []
    print(f"[viz] {slug}: {len(inside)} in_mask + {len(outside)} out_mask -> {out_dir}")
    for bucket, rows in (("in_mask", inside), ("out_mask", outside)):
        for i, row in enumerate(rows, start=1):
            uid8 = str(row.get("take_uid", "take"))[:8]
            out_name = f"{i:02d}_{uid8}_{row['cam']}_{int(row['frame_idx']):06d}.jpg"
            out_path = out_dir / bucket / out_name
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
                n_obj = len(row.get("objects") or [])
                print(
                    f"[viz] {slug}/{bucket}/{out_name}  "
                    f"{row.get('label_text', '')}  "
                    f"n_masks={n_obj}  {row.get('take_name', '')}"
                )
            except Exception as exc:
                failed.append((str(out_path), str(exc)))
                print(f"[viz] FAIL {out_path}: {exc}")
    if failed:
        raise RuntimeError(f"{slug}: {len(failed)} plots failed: {failed[:3]}")
    return n_ok


def run_paper(args, cfg: dict) -> None:
    base_out = args.out or (Path("./viz_ego_exo_paper").resolve())
    if args.all_scenarios:
        configs = [ROOT / rel for rel in PAPER_SCENARIO_CONFIGS]
    else:
        configs = [Path(cfg["_config_path"])]

    n_ok = 0
    errors: list[str] = []
    for path in configs:
        one = load_config(path)
        slug = scenario_slug(one)
        out_dir = base_out / slug
        try:
            n_ok += run_paper_scenario(args, one, out_dir)
        except Exception as exc:
            msg = f"{slug}: {exc}"
            errors.append(msg)
            print(f"[viz] FAIL {msg}")
            if not args.all_scenarios:
                raise
    if args.all_scenarios and n_ok == 0:
        raise RuntimeError("No scenario produced plots: " + "; ".join(errors[:3]))
    print(f"Saved {n_ok} plots -> {base_out}")


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
        help="No captions, match height, mask + bbox + label, 10 in/10 out per scenario.",
    )
    parser.add_argument(
        "--all-scenarios",
        action="store_true",
        help="Run --paper on cooking, health, bike_repair, music, basketball, soccer.",
    )
    parser.add_argument("--no-text", action="store_true", help="Drop captions/header.")
    parser.add_argument("--no-label", action="store_true", help="Drop object-name chip.")
    parser.add_argument("--match-height", action="store_true")
    parser.add_argument("--overlay-mask", action="store_true")
    parser.add_argument("--distinct-takes", action="store_true")
    parser.add_argument(
        "--per-bucket",
        type=int,
        default=10,
        help="In-mask and out-of-mask count per scenario (default 10).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.paper:
        run_paper(args, cfg)
    else:
        run_qa(args, cfg)


if __name__ == "__main__":
    main()
