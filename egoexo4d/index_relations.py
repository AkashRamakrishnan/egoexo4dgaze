"""Parse Relations JSON into a flat exo frame index."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from egoexo4d.index_takes import load_cooking_takes
from egoexo4d.labels import normalize_label, should_keep_track
from egoexo4d.paths import manifests_dir, relations_path


def _is_exo_camera(cam_name: str, prefix: str) -> bool:
    return cam_name.startswith(prefix) and not cam_name.startswith("aria")


def index_relations_for_take(
    take_uid: str,
    take_name: str,
    relations_split: str,
    relations_blob: dict,
    *,
    drop_hand_tracks: bool,
    exo_cam_prefix: str,
) -> List[dict]:
    records: List[dict] = []
    object_masks = relations_blob.get("object_masks") or {}

    for track_name, cam_dict in object_masks.items():
        if not should_keep_track(track_name, drop_hand_tracks=drop_hand_tracks):
            continue
        label_text = normalize_label(track_name)

        for cam_name, cam_ann in cam_dict.items():
            if not _is_exo_camera(cam_name, exo_cam_prefix):
                continue
            annotation = cam_ann.get("annotation") or {}
            annotation_fps = float(cam_ann.get("annotation_fps", 1.0))

            for frame_key, frame_ann in annotation.items():
                frame_idx = int(frame_key)
                records.append(
                    {
                        "take_uid": take_uid,
                        "take_name": take_name,
                        "relations_split": relations_split,
                        "track_name": track_name,
                        "label_text": label_text,
                        "cam": cam_name,
                        "frame_idx": frame_idx,
                        "annotation_fps": annotation_fps,
                        "mask_width": int(frame_ann["width"]),
                        "mask_height": int(frame_ann["height"]),
                        "encoded_mask": frame_ann["encodedMask"],
                    }
                )
    return records


def build_frame_index(cfg: Dict[str, Any]) -> Path:
    """Write ``manifests/frame_index.parquet``."""
    takes = load_cooking_takes(cfg)
    take_uids = {t["take_uid"] for t in takes}
    take_meta = {t["take_uid"]: t for t in takes}

    rel_cache: Dict[str, dict] = {}
    for split in ("train", "val"):
        path = relations_path(cfg, split)
        if path.exists():
            with path.open() as f:
                rel_cache[split] = json.load(f)

    filters = cfg.get("filters", {})
    drop_hand = bool(filters.get("drop_hand_tracks", True))
    cam_prefix = str(filters.get("exo_cam_prefix", "cam"))

    all_records: List[dict] = []
    for split, blob in rel_cache.items():
        for take_uid, take_ann in blob.items():
            if take_uid not in take_uids:
                continue
            meta = take_meta[take_uid]
            all_records.extend(
                index_relations_for_take(
                    take_uid,
                    meta["take_name"],
                    split,
                    take_ann,
                    drop_hand_tracks=drop_hand,
                    exo_cam_prefix=cam_prefix,
                )
            )

    out_dir = manifests_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "frame_index.parquet"

    if not all_records:
        pd.DataFrame(
            columns=[
                "take_uid",
                "take_name",
                "relations_split",
                "track_name",
                "label_text",
                "cam",
                "frame_idx",
                "annotation_fps",
                "mask_width",
                "mask_height",
                "encoded_mask",
            ]
        ).to_parquet(out_path, index=False)
        print("[index_relations] WARNING: empty frame index")
        return out_path

    df = pd.DataFrame(all_records)
    df.to_parquet(out_path, index=False)
    print(
        f"[index_relations] {len(df)} exo mask rows, "
        f"{df['take_uid'].nunique()} takes, "
        f"{df['label_text'].nunique()} labels"
    )
    return out_path


def load_frame_index(cfg: Dict[str, Any]) -> pd.DataFrame:
    path = manifests_dir(cfg) / "frame_index.parquet"
    if not path.exists():
        build_frame_index(cfg)
    return pd.read_parquet(path)
