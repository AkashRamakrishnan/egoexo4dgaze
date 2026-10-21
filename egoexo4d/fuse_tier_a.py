"""Tier-A fusion: gaze inside Relations exo mask."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import pandas as pd
from tqdm import tqdm

from egoexo4d.gaze_project import get_projector
from egoexo4d.head_detect import get_head_detector
from egoexo4d.index_relations import load_frame_index
from egoexo4d.masks import decode_mask_dict, mask_to_bbox, point_in_mask, valid_xyxy
from egoexo4d.paths import egoexo_root, manifests_dir


def image_rel_path(take_uid: str, cam: str, frame_idx: int) -> str:
    return f"{take_uid}/{cam}/{frame_idx:06d}.jpg"


def summarize_unique_frame_status(frame_status: Dict[tuple, dict]) -> dict:
    """Collapse per-frame flags into unique-frame funnel counts."""
    n = len(frame_status)
    projector_ok = sum(1 for s in frame_status.values() if s.get("projector_ok"))
    gaze_ok = sum(1 for s in frame_status.values() if s.get("gaze_ok"))
    in_mask = sum(1 for s in frame_status.values() if s.get("in_mask"))
    kept = sum(1 for s in frame_status.values() if s.get("kept"))
    return {
        "unique_relations_frames_processed": n,
        "unique_frames_projector_ok": projector_ok,
        "unique_frames_with_gaze": gaze_ok,
        "unique_frames_gaze_in_object_mask": in_mask,
        "unique_frames_kept": kept,
    }


def fuse_tier_a_rows(cfg: Dict[str, Any]) -> tuple[List[dict], dict]:
    """Return Tier-A candidate rows and build statistics."""
    df = load_frame_index(cfg)
    if df.empty:
        return [], {"error": "empty_frame_index"}

    gaze_cfg = cfg.get("gaze", {})
    prefer_personalized = bool(gaze_cfg.get("prefer_personalized", True))
    calib_min_quality = float(gaze_cfg.get("calib_min_quality", 1.0))

    head_cfg = cfg.get("head_detection", {})
    skip_head = bool(cfg.get("build", {}).get("skip_head_detection", False))
    head_detector = None
    if not skip_head:
        head_detector = get_head_detector(
            str(head_cfg.get("model", "yolov8n.pt")),
            float(head_cfg.get("conf", 0.5)),
            float(head_cfg.get("head_fraction", 0.35)),
        )

    root = egoexo_root(cfg)
    rows: List[dict] = []
    drop_reasons: Counter = Counter()
    projector_cache: Dict[str, Any] = {}
    projector_errors: Dict[str, str] = {}
    frame_cache: Dict[tuple, Any] = {}
    unique_images: set = set()
    frame_status: Dict[tuple, dict] = {}

    build_cfg = cfg.get("build", {})
    max_images = build_cfg.get("max_tier_a_images")
    max_images_n = int(max_images) if max_images is not None else None
    if max_images_n is not None:
        seed = int(build_cfg.get("shuffle_seed", 42))
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    for record in tqdm(df.to_dict(orient="records"), desc="fuse_tier_a"):
        take_uid = record["take_uid"]
        take_name = record["take_name"]
        cam = record["cam"]
        frame_idx = int(record["frame_idx"])

        take_path = root / "takes" / take_name
        frame_key = (take_uid, cam, frame_idx)
        st = frame_status.setdefault(
            frame_key,
            {
                "projector_ok": False,
                "gaze_ok": False,
                "in_mask": False,
                "kept": False,
            },
        )
        cache_key = str(take_path)
        if cache_key not in projector_cache:
            try:
                projector_cache[cache_key] = get_projector(
                    cache_key, prefer_personalized, calib_min_quality
                )
            except Exception as exc:
                projector_cache[cache_key] = exc
                projector_errors[take_name] = f"{type(exc).__name__}: {exc}"
                print(f"[fuse] projector_init_failed {take_name}: {type(exc).__name__}: {exc}")

        projector = projector_cache[cache_key]
        if isinstance(projector, Exception):
            drop_reasons["projector_init_failed"] += 1
            continue
        st["projector_ok"] = True

        try:
            gaze = projector.project_gaze(cam, frame_idx)
        except Exception:
            drop_reasons["project_gaze_exception"] += 1
            continue
        if gaze is None:
            drop_reasons["gaze_projection_failed"] += 1
            continue
        st["gaze_ok"] = True
        gaze_x, gaze_y = gaze

        try:
            mask = decode_mask_dict(
                {
                    "width": record["mask_width"],
                    "height": record["mask_height"],
                    "encodedMask": record["encoded_mask"],
                }
            )
        except Exception:
            drop_reasons["mask_decode_failed"] += 1
            continue

        if not point_in_mask(gaze_x, gaze_y, mask):
            drop_reasons["gaze_outside_mask"] += 1
            continue
        st["in_mask"] = True

        try:
            box = mask_to_bbox(mask)
        except ValueError:
            drop_reasons["empty_mask"] += 1
            continue
        if not valid_xyxy(box):
            drop_reasons["invalid_bbox"] += 1
            continue

        head_box: Optional[List[float]] = None
        if head_detector is not None:
            if frame_key not in frame_cache:
                video_path = take_path / "frame_aligned_videos" / f"{cam}.mp4"
                cap = cv2.VideoCapture(str(video_path))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame = cap.read()
                cap.release()
                frame_cache[frame_key] = frame if ok else None
            frame_bgr = frame_cache[frame_key]
            if frame_bgr is None:
                drop_reasons["frame_read_failed"] += 1
                continue
            head_box = head_detector.detect_head(frame_bgr, gaze_xy=(gaze_x, gaze_y))
            if head_box is None or not valid_xyxy(head_box):
                drop_reasons["head_detection_failed"] += 1
                continue
        else:
            # Placeholder head box for pipeline testing without YOLO.
            x1, y1, x2, y2 = box
            head_box = [x1, max(0, y1 - (y2 - y1) * 0.5), x2, y1]

        rel_image = image_rel_path(take_uid, cam, frame_idx)
        rows.append(
            {
                "take_uid": take_uid,
                "take_name": take_name,
                "cam": cam,
                "frame_idx": frame_idx,
                "image_path": rel_image,
                "gaze_x": gaze_x,
                "gaze_y": gaze_y,
                "box_x1": box[0],
                "box_y1": box[1],
                "box_x2": box[2],
                "box_y2": box[3],
                "head_x1": head_box[0],
                "head_y1": head_box[1],
                "head_x2": head_box[2],
                "head_y2": head_box[3],
                "label_text": record["label_text"],
                "track_name": record["track_name"],
                "relations_split": record["relations_split"],
                "score": 1.0,
                "flag": "in",
                "box_idx": 0,
            }
        )
        st["kept"] = True
        unique_images.add(rel_image)
        if max_images_n is not None and len(unique_images) >= max_images_n:
            break

    stats = {
        "input_mask_rows": len(df),
        "tier_a_rows": len(rows),
        "unique_images": len({r["image_path"] for r in rows}),
        "unique_labels": len({r["label_text"] for r in rows}),
        "drop_reasons": dict(drop_reasons),
        "frame_funnel": summarize_unique_frame_status(frame_status),
        "projector_errors": projector_errors,
        "takes": len({r["take_uid"] for r in rows}),
        "max_tier_a_images": max_images_n,
        "early_stop": bool(
            max_images_n is not None and len({r["image_path"] for r in rows}) >= max_images_n
        ),
    }
    return rows, stats


def save_build_stats(cfg: Dict[str, Any], stats: dict) -> Path:
    out = manifests_dir(cfg) / "build_stats.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(stats, f, indent=2)
    return out
