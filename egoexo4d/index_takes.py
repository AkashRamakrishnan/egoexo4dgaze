"""Index scenario takes with Relations, MPS gaze, and exo calibration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from egoexo4d.gaze_project import find_aria_vrs
from egoexo4d.paths import (
    egoexo_root,
    manifests_dir,
    relations_path,
    resolve_takes_manifest,
    takes_json_path,
    takes_manifest_path,
)
from egoexo4d.scenarios import (
    is_scenario_take,
    scenario_name,
    scenario_slug,
    text_matches_scenario,
    scenario_spec,
)


def unwrap_relations_dict(data: Any) -> Dict[str, dict]:
    """Official files are ``{"annotations": {take_uid: {...}}}`` (see EgoExo_Relations.ipynb)."""
    if not isinstance(data, dict) or not data:
        return {}
    inner = data.get("annotations")
    if isinstance(inner, dict) and inner:
        sample = next(iter(inner.values()))
        if isinstance(sample, dict):
            return {str(k): v for k, v in inner.items() if isinstance(v, dict)}
    if any(
        isinstance(v, dict) and ("object_masks" in v or "object_names" in v or "take_name" in v)
        for v in data.values()
    ):
        return {str(k): v for k, v in data.items() if isinstance(v, dict)}
    return {}


def _text_looks_cooking(*parts: Any) -> bool:
    return text_matches_scenario(scenario_spec("cooking"), *parts)


def _is_cooking_take(take: dict, cfg: Dict[str, Any]) -> bool:
    """Backward-compatible alias for ``is_scenario_take``."""
    return is_scenario_take(take, cfg)


def load_takes_json(cfg: Dict[str, Any]) -> List[dict]:
    path = takes_json_path(cfg)
    if not path.exists():
        raise FileNotFoundError(f"Missing takes.json: {path}")
    with path.open() as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if isinstance(data.get("takes"), list):
            return data["takes"]
        values = list(data.values())
        if values and isinstance(values[0], dict) and "take_uid" in values[0]:
            return values
    raise ValueError(f"Unrecognized takes.json format: {path}")


def cooking_uids_for_download(cfg: Dict[str, Any]) -> List[str]:
    """Take UIDs for the config scenario, intersected with Relations train/val."""
    return scenario_uids_for_download(cfg)


def scenario_uids_for_download(cfg: Dict[str, Any]) -> List[str]:
    """Scenario take UIDs, intersected with Relations train/val when those JSONs exist."""
    takes = load_takes_json(cfg)
    rel = _relations_take_uids(cfg)
    uids: List[str] = []
    for take in takes:
        if not is_scenario_take(take, cfg):
            continue
        uid = take.get("take_uid")
        if not uid:
            continue
        if rel and uid not in rel:
            continue
        uids.append(str(uid))
    return sorted(set(uids))


def load_relations_annotations(cfg: Dict[str, Any], split: str) -> Dict[str, dict]:
    path = relations_path(cfg, split)
    if not path.exists():
        return {}
    with path.open() as f:
        data = json.load(f)
    return unwrap_relations_dict(data)


def _relations_take_uids(cfg: Dict[str, Any]) -> Dict[str, str]:
    """Map take_uid → relations split name (train/val)."""
    out: Dict[str, str] = {}
    for split in ("train", "val"):
        for take_uid in load_relations_annotations(cfg, split):
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
        "slam/gopro_calibs.csv",
    ):
        if (take_path / rel).exists():
            return True
    return any(take_path.rglob("gopro_calibs.csv"))


def _exo_cams_for_take(take_path: Path, prefix: str = "cam") -> List[str]:
    video_dir = take_path / "frame_aligned_videos"
    if not video_dir.is_dir():
        return []
    cams: List[str] = []
    for mp4 in sorted(video_dir.glob(f"{prefix}*.mp4")):
        cams.append(mp4.stem)
    return cams


def build_cooking_takes_manifest(cfg: Dict[str, Any]) -> Path:
    """Write ``manifests/scenario_takes.json`` (and Cooking ``cooking_takes.json``)."""
    root = egoexo_root(cfg)
    takes_path = takes_json_path(cfg)
    if not takes_path.exists():
        raise FileNotFoundError(f"Missing takes.json: {takes_path}")

    all_takes = load_takes_json(cfg)
    takes_by_uid = {t["take_uid"]: t for t in all_takes}
    rel_by_split = {
        split: load_relations_annotations(cfg, split) for split in ("train", "val")
    }
    rel_uids: Dict[str, str] = {}
    rel_blobs: Dict[str, dict] = {}
    for split, anns in rel_by_split.items():
        for uid, blob in anns.items():
            rel_uids[uid] = split
            rel_blobs[uid] = blob

    records: List[dict] = []
    n_in_takes = sum(1 for t in all_takes if is_scenario_take(t, cfg))
    stats = {
        "total_takes_json": len(all_takes),
        "scenario": scenario_name(cfg),
        "scenario_slug": scenario_slug(cfg),
        "scenario_in_takes_json": n_in_takes,
        "cooking_in_takes_json": n_in_takes,
        "relations_takes": len(rel_uids),
        "in_relations": 0,
        "with_mps": 0,
        "with_calib": 0,
        "with_exo_video": 0,
        "with_vrs": 0,
        "selected": 0,
        "missing_from_takes_json": 0,
    }

    for take_uid, rel_split in rel_uids.items():
        blob = rel_blobs.get(take_uid) or {}
        take = takes_by_uid.get(take_uid)
        if take is None:
            stats["missing_from_takes_json"] += 1
            take = {
                "take_uid": take_uid,
                "take_name": blob.get("take_name"),
                "parent_task_name": blob.get("scenario"),
            }
        if not is_scenario_take(
            take, cfg, extra_parts=(blob.get("scenario"), blob.get("take_name"))
        ):
            continue
        stats["in_relations"] += 1

        take_name = take.get("take_name") or blob.get("take_name")
        if not take_name:
            continue
        take_path = root / "takes" / take_name
        has_mps = _take_has_mps(take_path)
        has_calib = _take_has_gopro_calib(take_path)
        has_vrs = find_aria_vrs(take_path) is not None
        exo_cams = _exo_cams_for_take(take_path, prefix=cfg["filters"].get("exo_cam_prefix", "cam"))

        if has_mps:
            stats["with_mps"] += 1
        if has_calib:
            stats["with_calib"] += 1
        if exo_cams:
            stats["with_exo_video"] += 1
        if has_vrs:
            stats["with_vrs"] += 1

        if not (has_mps and has_calib and exo_cams and has_vrs):
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
    payload = {"stats": stats, "takes": records}
    manifest_path = takes_manifest_path(cfg)
    with manifest_path.open("w") as f:
        json.dump(payload, f, indent=2)
    if scenario_slug(cfg) == "cooking":
        legacy = out_dir / "cooking_takes.json"
        with legacy.open("w") as f:
            json.dump(payload, f, indent=2)

    label = scenario_name(cfg)
    print(
        f"[index_takes] selected {len(records)} {label} takes "
        f"(relations={stats['in_relations']}, mps={stats['with_mps']}, "
        f"calib={stats['with_calib']}, exo={stats['with_exo_video']}, "
        f"vrs={stats['with_vrs']})"
    )
    return manifest_path


def load_cooking_takes(cfg: Dict[str, Any]) -> List[dict]:
    path = resolve_takes_manifest(cfg)
    if not path.exists():
        build_cooking_takes_manifest(cfg)
        path = resolve_takes_manifest(cfg)
    with path.open() as f:
        payload = json.load(f)
    return payload["takes"]
