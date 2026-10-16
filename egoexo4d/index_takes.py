"""Index Cooking takes with Relations, MPS gaze, and exo calibration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Set

from egoexo4d.paths import egoexo_root, manifests_dir, relations_path, takes_json_path


def _is_cooking_take(take: dict, cfg: Dict[str, Any]) -> bool:
    scenario = cfg.get("scenario", {})
    task_id = scenario.get("parent_task_id")
    task_name = scenario.get("parent_task_name")
    if task_id is not None and take.get("parent_task_id") == task_id:
        return True
    if task_name and take.get("parent_task_name") == task_name:
        return True
    return False


def _relations_take_uids(cfg: Dict[str, Any]) -> Dict[str, str]:
    """Map take_uid → relations split name (train/val)."""
    out: Dict[str, str] = {}
    for split in ("train", "val"):
        path = relations_path(cfg, split)
        if not path.exists():
            continue
        with path.open() as f:
            data = json.load(f)
        for take_uid in data:
            out[take_uid] = split
    return out


def _take_has_mps(take_path: Path) -> bool:
    eye_gaze = take_path / "eye_gaze"
    if not eye_gaze.is_dir():
        return False
    for pattern in ("personalized_eye_gaze.csv", "general_eye_gaze.csv"):
        if list(eye_gaze.glob(f"**/{pattern}")):
            return True
    return any(eye_gaze.glob("**/*.csv"))


def _take_has_gopro_calib(take_path: Path) -> bool:
    for rel in (
        "trajectory/gopro_calibs.csv",
        "take_trajectory/gopro_calibs.csv",
        "mps/trajectory/gopro_calibs.csv",
    ):
        if (take_path / rel).exists():
            return True
    return False


def _exo_cams_for_take(take_path: Path, prefix: str = "cam") -> List[str]:
    video_dir = take_path / "frame_aligned_videos"
    if not video_dir.is_dir():
        return []
    cams: List[str] = []
    for mp4 in sorted(video_dir.glob(f"{prefix}*.mp4")):
        cams.append(mp4.stem)
    return cams


def build_cooking_takes_manifest(cfg: Dict[str, Any]) -> Path:
    """Write ``manifests/cooking_takes.json`` and return its path."""
    root = egoexo_root(cfg)
    takes_path = takes_json_path(cfg)
    if not takes_path.exists():
        raise FileNotFoundError(f"Missing takes.json: {takes_path}")

    with takes_path.open() as f:
        all_takes = json.load(f)
    takes_by_uid = {t["take_uid"]: t for t in all_takes}
    rel_uids = _relations_take_uids(cfg)

    records: List[dict] = []
    stats = {
        "total_takes_json": len(all_takes),
        "cooking_in_takes_json": 0,
        "in_relations": 0,
        "with_mps": 0,
        "with_calib": 0,
        "with_exo_video": 0,
        "selected": 0,
    }

    for take_uid, rel_split in rel_uids.items():
        take = takes_by_uid.get(take_uid)
        if take is None:
            continue
        if not _is_cooking_take(take, cfg):
            continue
        stats["cooking_in_takes_json"] += 1
        stats["in_relations"] += 1

        take_name = take["take_name"]
        take_path = root / "takes" / take_name
        has_mps = _take_has_mps(take_path)
        has_calib = _take_has_gopro_calib(take_path)
        exo_cams = _exo_cams_for_take(take_path, prefix=cfg["filters"].get("exo_cam_prefix", "cam"))

        if has_mps:
            stats["with_mps"] += 1
        if has_calib:
            stats["with_calib"] += 1
        if exo_cams:
            stats["with_exo_video"] += 1

        if not (has_mps and has_calib and exo_cams):
            continue

        stats["selected"] += 1
        records.append(
            {
                "take_uid": take_uid,
                "take_name": take_name,
                "relations_split": rel_split,
                "parent_task_name": take.get("parent_task_name"),
                "parent_task_id": take.get("parent_task_id"),
                "exo_cams": exo_cams,
                "take_path": str(take_path),
            }
        )

    max_takes = cfg.get("build", {}).get("max_takes")
    if max_takes is not None:
        records = records[: int(max_takes)]

    out_dir = manifests_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "cooking_takes.json"
    payload = {"stats": stats, "takes": records}
    with manifest_path.open("w") as f:
        json.dump(payload, f, indent=2)

    print(
        f"[index_takes] selected {len(records)} cooking takes "
        f"(relations={stats['in_relations']}, mps={stats['with_mps']}, "
        f"calib={stats['with_calib']}, exo={stats['with_exo_video']})"
    )
    return manifest_path


def load_cooking_takes(cfg: Dict[str, Any]) -> List[dict]:
    path = manifests_dir(cfg) / "cooking_takes.json"
    if not path.exists():
        build_cooking_takes_manifest(cfg)
    with path.open() as f:
        payload = json.load(f)
    return payload["takes"]
