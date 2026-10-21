"""Post-build dataset report: disk, frame funnel, CSV/QA stats."""

from __future__ import annotations

import ast
import csv
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from egoexo4d.paths import (
    annotations_dir,
    egoexo_root,
    images_dir,
    manifests_dir,
    output_root,
    resolve_takes_manifest,
)
from egoexo4d.qa import _gaze_in_box


def _load_takes_payload(cfg: dict) -> Optional[dict]:
    path = resolve_takes_manifest(cfg)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def format_bytes(n: int | float) -> str:
    value = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024.0 or unit == "TiB":
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TiB"


def pct(num: int | float, den: int | float) -> Optional[float]:
    if den in (0, 0.0):
        return None
    return 100.0 * float(num) / float(den)


def pct_str(num: int | float, den: int | float) -> str:
    value = pct(num, den)
    return "n/a" if value is None else f"{value:.1f}%"


def path_bytes(path: Path) -> int:
    """Apparent size of a file or directory (prefers ``du -sb``)."""
    if not path.exists():
        return 0
    try:
        out = subprocess.check_output(
            ["du", "-sb", str(path)],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return int(out.split()[0])
    except (OSError, subprocess.CalledProcessError, ValueError, IndexError):
        pass
    if path.is_file():
        try:
            return int(path.stat().st_size)
        except OSError:
            return 0
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            try:
                total += child.stat().st_size
            except OSError:
                continue
    return total


def _dir_breakdown(root: Path, names: Sequence[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for name in names:
        child = root / name
        if child.exists():
            out[name] = path_bytes(child)
    return out


def collect_disk_stats(cfg: dict) -> dict:
    raw = egoexo_root(cfg)
    data = output_root(cfg)
    takes_root = raw / "takes"

    selected_names: set[str] = set()
    payload = _load_takes_payload(cfg)
    if payload:
        selected_names = {
            str(t.get("take_name")) for t in payload.get("takes", []) if t.get("take_name")
        }

    selected_take_bytes = 0
    other_take_bytes = 0
    n_selected_dirs = 0
    n_other_dirs = 0
    if takes_root.is_dir():
        for child in takes_root.iterdir():
            if not child.is_dir():
                continue
            size = path_bytes(child)
            if child.name in selected_names:
                selected_take_bytes += size
                n_selected_dirs += 1
            else:
                other_take_bytes += size
                n_other_dirs += 1

    images = images_dir(cfg)
    jpeg_sizes: List[int] = []
    n_jpegs = 0
    if images.is_dir():
        for jpeg in images.rglob("*.jpg"):
            n_jpegs += 1
            try:
                jpeg_sizes.append(int(jpeg.stat().st_size))
            except OSError:
                continue
    jpeg_total = sum(jpeg_sizes)
    jpeg_mean = int(jpeg_total / len(jpeg_sizes)) if jpeg_sizes else 0

    raw_total = path_bytes(raw)
    data_total = path_bytes(data)
    return {
        "raw_root": str(raw),
        "raw_bytes": raw_total,
        "raw_human": format_bytes(raw_total),
        "raw_breakdown": _dir_breakdown(
            raw, ("takes", "annotations", "captures", "metadata")
        ),
        "selected_take_dirs": n_selected_dirs,
        "selected_take_bytes": selected_take_bytes,
        "selected_take_human": format_bytes(selected_take_bytes),
        "other_take_dirs": n_other_dirs,
        "other_take_bytes": other_take_bytes,
        "other_take_human": format_bytes(other_take_bytes),
        "extracted_root": str(data),
        "extracted_bytes": data_total,
        "extracted_human": format_bytes(data_total),
        "extracted_breakdown": {
            name: path_bytes(data / name)
            for name in (
                "images",
                "annotations",
                "manifests",
                "label-embeds",
                "viz_samples",
                "viz_ego_exo",
            )
            if (data / name).exists()
        },
        "extracted_over_raw_pct": pct(data_total, raw_total),
        "jpeg_count": n_jpegs,
        "jpeg_bytes": jpeg_total,
        "jpeg_mean_bytes": jpeg_mean,
        "jpeg_mean_human": format_bytes(jpeg_mean),
    }


def collect_take_stats(cfg: dict) -> dict:
    path = resolve_takes_manifest(cfg)
    if not path.exists():
        return {"error": f"missing {path}"}
    payload = json.loads(path.read_text())
    takes = payload.get("takes", [])
    cams: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    n_cams: Counter[int] = Counter()
    for rec in takes:
        exo = rec.get("exo_cams") or []
        n_cams[len(exo)] += 1
        split_counts[str(rec.get("relations_split", "unknown"))] += 1
        for cam in exo:
            cams[str(cam)] += 1
    return {
        "index_stats": payload.get("stats", {}),
        "selected_takes": len(takes),
        "relations_split_counts": dict(split_counts),
        "exo_cams_histogram": dict(cams),
        "num_exo_cams_per_take": {str(k): v for k, v in sorted(n_cams.items())},
    }


def count_mp4_frames(video_path: Path) -> int:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return max(n, 0)


def collect_raw_frame_counts(cfg: dict) -> dict:
    """Sum exo ``cam*.mp4`` frame counts for selected complete takes."""
    from tqdm import tqdm

    path = resolve_takes_manifest(cfg)
    if not path.exists():
        return {"error": f"missing {path.name}"}
    takes = json.loads(path.read_text()).get("takes", [])
    prefix = str(cfg.get("filters", {}).get("exo_cam_prefix", "cam"))
    n_videos = 0
    n_missing = 0
    n_frames = 0
    per_cam: Counter[str] = Counter()
    jobs: List[tuple[str, Path]] = []
    for rec in takes:
        take_path = Path(rec["take_path"])
        video_dir = take_path / "frame_aligned_videos"
        cams = rec.get("exo_cams") or []
        if not cams and video_dir.is_dir():
            cams = [p.stem for p in sorted(video_dir.glob(f"{prefix}*.mp4"))]
        for cam in cams:
            mp4 = video_dir / f"{cam}.mp4"
            jobs.append((str(cam), mp4))

    for cam, mp4 in tqdm(jobs, desc="count_raw_frames"):
        if not mp4.exists():
            n_missing += 1
            continue
        n_videos += 1
        frames = count_mp4_frames(mp4)
        n_frames += frames
        per_cam[cam] += frames

    return {
        "selected_takes": len(takes),
        "exo_videos": n_videos,
        "missing_videos": n_missing,
        "raw_exo_frames": n_frames,
        "raw_exo_frames_by_cam": dict(per_cam),
        "note": (
            "Raw counts are every GoPro frame in selected takes (~30 fps). "
            "Relations masks and gaze fusion only use ~1 fps timestamps."
        ),
    }


def collect_relations_frame_stats(cfg: dict) -> dict:
    parquet = manifests_dir(cfg) / "frame_index.parquet"
    if not parquet.exists():
        return {"error": "missing frame_index.parquet"}
    import pandas as pd

    df = pd.read_parquet(
        parquet,
        columns=["take_uid", "cam", "frame_idx", "label_text", "track_name"],
    )
    unique = df.drop_duplicates(["take_uid", "cam", "frame_idx"])
    return {
        "mask_rows": int(len(df)),
        "unique_annotated_frames": int(len(unique)),
        "unique_takes": int(df["take_uid"].nunique()),
        "unique_labels": int(df["label_text"].nunique()),
        "unique_tracks": int(df["track_name"].nunique()),
    }


def _parse_head_box(cell: str) -> Optional[List[float]]:
    try:
        boxes = ast.literal_eval(cell)
        if boxes and len(boxes[0]) == 4:
            return [float(v) for v in boxes[0]]
    except (SyntaxError, ValueError, TypeError, IndexError):
        return None
    return None


def _parse_image_path(image_path: str) -> tuple[str, str, Optional[int]]:
    parts = Path(image_path).parts
    take_uid = parts[0] if parts else ""
    cam = parts[1] if len(parts) > 1 else ""
    frame_idx: Optional[int] = None
    if len(parts) > 2:
        try:
            frame_idx = int(Path(parts[2]).stem)
        except ValueError:
            frame_idx = None
    return take_uid, cam, frame_idx


def collect_csv_stats(cfg: dict) -> dict:
    csv_path = annotations_dir(cfg) / "train_verified.csv"
    img_root = images_dir(cfg)
    if not csv_path.exists():
        return {"error": f"missing {csv_path}"}

    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    images = {r["image_path"] for r in rows}
    labels = Counter(r["label_text"] for r in rows)
    cams: Counter[str] = Counter()
    take_uids: set[str] = set()
    frames_by_take: Dict[str, List[int]] = defaultdict(list)
    missing = 0
    gaze_in_box = 0
    head_top_third = 0
    head_ok = 0
    head_areas: List[float] = []
    obj_areas: List[float] = []

    for row in rows:
        take_uid, cam, frame_idx = _parse_image_path(row["image_path"])
        take_uids.add(take_uid)
        if cam:
            cams[cam] += 1
        if frame_idx is not None:
            frames_by_take[take_uid].append(frame_idx)
        if not (img_root / row["image_path"]).exists():
            missing += 1
        if _gaze_in_box(row):
            gaze_in_box += 1
        x1, y1, x2, y2 = (
            float(row["box_x1"]),
            float(row["box_y1"]),
            float(row["box_x2"]),
            float(row["box_y2"]),
        )
        obj_areas.append(max(0.0, (x2 - x1) * (y2 - y1)))
        head = _parse_head_box(row.get("head_boxes", ""))
        if head is not None:
            hx1, hy1, hx2, hy2 = head
            head_ok += 1
            head_areas.append(max(0.0, (hx2 - hx1) * (hy2 - hy1)))
            # Head in the top third of the object box (YOLO crop heuristic).
            obj_h = max(1e-6, y2 - y1)
            if hy2 <= y1 + obj_h / 3.0 or hy1 <= y1 + obj_h * 0.5:
                head_top_third += 1

    spans = []
    for take_uid, idxs in frames_by_take.items():
        spans.append(
            {
                "take_uid": take_uid,
                "n_rows": len(idxs),
                "frame_min": min(idxs),
                "frame_max": max(idxs),
            }
        )
    span_widths = [s["frame_max"] - s["frame_min"] for s in spans] if spans else []

    selected_uids: set[str] = set()
    payload = _load_takes_payload(cfg)
    if payload:
        selected_uids = {
            str(t["take_uid"])
            for t in payload.get("takes", [])
            if t.get("take_uid")
        }
    zero_tier_a = sorted(selected_uids - take_uids)

    return {
        "csv": str(csv_path),
        "rows": len(rows),
        "unique_images": len(images),
        "unique_labels": len(labels),
        "unique_takes": len(take_uids),
        "missing_images": missing,
        "gaze_in_box_rows": gaze_in_box,
        "gaze_in_box_pct": pct(gaze_in_box, len(rows)),
        "cams": dict(cams),
        "head_boxes_parsed": head_ok,
        "head_mean_area_px": (sum(head_areas) / len(head_areas)) if head_areas else 0.0,
        "object_mean_area_px": (sum(obj_areas) / len(obj_areas)) if obj_areas else 0.0,
        "head_near_top_of_object_pct": pct(head_top_third, head_ok),
        "top_labels": labels.most_common(15),
        "label_frequency_buckets": {
            "eq_1_to_4": sum(1 for n in labels.values() if 1 <= n < 5),
            "eq_5_to_9": sum(1 for n in labels.values() if 5 <= n < 10),
            "eq_10_to_49": sum(1 for n in labels.values() if 10 <= n < 50),
            "ge_50": sum(1 for n in labels.values() if n >= 50),
        },
        "label_count_min": min(labels.values()) if labels else 0,
        "label_count_max": max(labels.values()) if labels else 0,
        "takes_with_zero_tier_a_rows": len(zero_tier_a),
        "zero_tier_a_take_uids_sample": zero_tier_a[:20],
        "mean_frame_span": (
            float(sum(span_widths) / len(span_widths)) if span_widths else 0.0
        ),
    }


def collect_split_stats(cfg: dict) -> dict:
    from egoexo4d.qa import _load_split_rows

    csv_dir = annotations_dir(cfg)
    splits = {name: _load_split_rows(name, csv_dir) for name in ("train", "val", "test")}
    out: Dict[str, Any] = {"splits": {}}
    image_sets: Dict[str, set[str]] = {}
    for name, rows in splits.items():
        images = {r["image_path"] for r in rows}
        image_sets[name] = images
        labels = Counter(r["label_text"] for r in rows)
        out["splits"][name] = {
            "rows": len(rows),
            "unique_images": len(images),
            "unique_labels": len(labels),
            "gaze_in_box_pct": pct(
                sum(1 for r in rows if _gaze_in_box(r)), len(rows)
            ),
        }
    leakage = {
        "train_val": len(image_sets.get("train", set()) & image_sets.get("val", set())),
        "train_test": len(image_sets.get("train", set()) & image_sets.get("test", set())),
        "val_test": len(image_sets.get("val", set()) & image_sets.get("test", set())),
    }
    out["leakage"] = leakage
    out["leakage_ok"] = all(v == 0 for v in leakage.values())
    return out


def collect_vocab_stats(cfg: dict) -> dict:
    vocab_path = output_root(cfg) / "vocab2id.json"
    embed_path = output_root(cfg) / "label-embeds" / "vocab_roberta_embeddings.pt"
    info: Dict[str, Any] = {
        "vocab_path": str(vocab_path),
        "vocab_exists": vocab_path.exists(),
        "vocab_size": 0,
        "embed_path": str(embed_path),
        "embed_exists": embed_path.exists(),
        "embed_bytes": path_bytes(embed_path) if embed_path.exists() else 0,
    }
    if vocab_path.exists():
        vocab = json.loads(vocab_path.read_text())
        info["vocab_size"] = len(vocab)
    if embed_path.exists():
        try:
            import torch

            try:
                payload = torch.load(
                    embed_path, map_location="cpu", weights_only=False
                )
            except TypeError:
                payload = torch.load(embed_path, map_location="cpu")
            if isinstance(payload, dict):
                labels = payload.get("labels") or []
                embeddings = payload.get("embeddings")
                info["embed_n_labels"] = len(labels)
                if hasattr(embeddings, "shape"):
                    info["embed_shape"] = list(embeddings.shape)
                info["embed_matches_vocab"] = info["embed_n_labels"] == info["vocab_size"]
        except Exception as exc:
            info["embed_load_error"] = f"{type(exc).__name__}: {exc}"
    return info


def _yields(frames: dict) -> dict:
    rel = frames.get("relations_annotated_frames") or 0
    gaze = frames.get("frames_with_gaze") or 0
    hit = frames.get("frames_gaze_in_object_mask") or 0
    exported = frames.get("exported_images") or 0
    raw = frames.get("raw_exo_frames") or 0
    return {
        "annotated_over_raw_pct": pct(rel, raw),
        "gaze_over_annotated_pct": pct(gaze, rel),
        "in_mask_over_gaze_pct": pct(hit, gaze),
        "exported_over_annotated_pct": pct(exported, rel),
        "exported_over_raw_pct": pct(exported, raw),
    }


def assemble_frame_section(
    raw: dict,
    relations: dict,
    fusion: dict,
    csv_stats: dict,
) -> dict:
    funnel = (fusion or {}).get("frame_funnel") or {}
    exported = csv_stats.get("unique_images", 0)
    jpeg_ok = csv_stats.get("unique_images", 0) - csv_stats.get("missing_images", 0)
    frames = {
        "raw_exo_frames": raw.get("raw_exo_frames", 0),
        "raw_exo_videos": raw.get("exo_videos", 0),
        "relations_mask_rows": relations.get("mask_rows", 0),
        "relations_annotated_frames": relations.get("unique_annotated_frames", 0),
        "frames_with_gaze": funnel.get("unique_frames_with_gaze"),
        "frames_gaze_in_object_mask": funnel.get("unique_frames_gaze_in_object_mask"),
        "frames_kept_after_head_filter": funnel.get("unique_frames_kept"),
        "exported_images": exported,
        "exported_images_on_disk": jpeg_ok,
        "dataset_rows": csv_stats.get("rows", 0),
        "fusion_frame_funnel": funnel,
        "raw_note": raw.get("note"),
    }
    frames["yields"] = _yields(frames)
    return frames


def load_build_stats(cfg: dict) -> dict:
    path = manifests_dir(cfg) / "build_stats.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _previous_raw_videos(cfg: dict) -> dict | None:
    path = manifests_dir(cfg) / "final_report.json"
    if not path.exists():
        return None
    try:
        prev = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    raw = prev.get("raw_videos")
    if isinstance(raw, dict) and raw.get("raw_exo_frames") and not raw.get("skipped"):
        return raw
    return None


def build_report(cfg: dict, *, skip_raw_frames: bool = False) -> dict:
    disk = collect_disk_stats(cfg)
    takes = collect_take_stats(cfg)
    fusion = load_build_stats(cfg)
    relations = collect_relations_frame_stats(cfg)
    csv_stats = collect_csv_stats(cfg)
    splits = collect_split_stats(cfg)
    vocab = collect_vocab_stats(cfg)
    if skip_raw_frames:
        raw = _previous_raw_videos(cfg) or {
            "skipped": True,
            "raw_exo_frames": 0,
            "exo_videos": 0,
        }
    else:
        raw = collect_raw_frame_counts(cfg)
    frames = assemble_frame_section(raw, relations, fusion, csv_stats)
    return {
        "egoexo_root": str(egoexo_root(cfg)),
        "output_root": str(output_root(cfg)),
        "disk": disk,
        "takes": takes,
        "raw_videos": raw,
        "relations_index": relations,
        "fusion": {
            "drop_reasons": fusion.get("drop_reasons", {}),
            "prune": fusion.get("prune", {}),
            "frame_funnel": fusion.get("frame_funnel", {}),
            "projector_error_takes": len(fusion.get("projector_errors") or {}),
            "tier_a_rows_pre_prune": fusion.get("tier_a_rows"),
            "unique_images_pre_prune": fusion.get("unique_images"),
            "frame_export": fusion.get("frame_export", {}),
        },
        "frames": frames,
        "dataset": csv_stats,
        "splits": splits,
        "vocab": vocab,
    }


def format_report(report: dict) -> str:
    disk = report.get("disk") or {}
    takes = report.get("takes") or {}
    index_stats = takes.get("index_stats") or {}
    frames = report.get("frames") or {}
    yields = frames.get("yields") or {}
    fusion = report.get("fusion") or {}
    dataset = report.get("dataset") or {}
    splits = report.get("splits") or {}
    vocab = report.get("vocab") or {}
    lines = [
        "========== Ego-Exo4D build report ==========",
        "",
        "DISK",
        f"  raw dump:     {disk.get('raw_human', '?')}  ({disk.get('raw_root', '')})",
        f"  extracted:    {disk.get('extracted_human', '?')}  ({disk.get('extracted_root', '')})",
        f"  extracted/raw: {pct_str(disk.get('extracted_bytes', 0), disk.get('raw_bytes', 0))}",
        f"  complete take dirs: {format_bytes(disk.get('selected_take_bytes', 0))} "
        f"({disk.get('selected_take_dirs', 0)} dirs)",
        f"  other take dirs:    {format_bytes(disk.get('other_take_bytes', 0))} "
        f"({disk.get('other_take_dirs', 0)} dirs; incomplete / leftover)",
        f"  JPEGs: {disk.get('jpeg_count', 0)} files, mean {disk.get('jpeg_mean_human', '?')}",
        "",
        "TAKES",
        f"  {(index_stats.get('scenario') or 'scenario')} in takes.json: "
        f"{index_stats.get('scenario_in_takes_json', index_stats.get('cooking_in_takes_json', '?'))}",
        f"  {(index_stats.get('scenario') or 'scenario')} in Relations:  "
        f"{index_stats.get('in_relations', '?')}",
        f"  with MPS/calib/VRS/exo: "
        f"{index_stats.get('with_mps', '?')}/"
        f"{index_stats.get('with_calib', '?')}/"
        f"{index_stats.get('with_vrs', '?')}/"
        f"{index_stats.get('with_exo_video', '?')}",
        f"  selected complete:     {takes.get('selected_takes', index_stats.get('selected', '?'))}",
        f"  relations split (selected): {takes.get('relations_split_counts', {})}",
        f"  zero Tier-A selected takes: {dataset.get('takes_with_zero_tier_a_rows', '?')}",
        "",
        "FRAMES  (unique take+cam+frame unless noted)",
        f"  raw exo video frames:           {frames.get('raw_exo_frames', '?')}",
        f"  Relations-annotated (~1 fps):   {frames.get('relations_annotated_frames', '?')}  "
        f"({pct_str(frames.get('relations_annotated_frames') or 0, frames.get('raw_exo_frames') or 0)} of raw)",
        f"  with gaze coordinates:          {frames.get('frames_with_gaze', '?')}  "
        f"({pct_str(frames.get('frames_with_gaze') or 0, frames.get('relations_annotated_frames') or 0)} of annotated)",
        f"  gaze in object mask (Tier-A):   {frames.get('frames_gaze_in_object_mask', '?')}  "
        f"({pct_str(frames.get('frames_gaze_in_object_mask') or 0, frames.get('frames_with_gaze') or 0)} of gazed)",
        f"  kept after head/YOLO:           {frames.get('frames_kept_after_head_filter', '?')}",
        f"  exported dataset images:        {frames.get('exported_images', '?')}  "
        f"(on disk {frames.get('exported_images_on_disk', '?')})",
        f"  dataset CSV rows:               {frames.get('dataset_rows', '?')}",
        f"  yield annotated/raw:            {_fmt_pct(yields.get('annotated_over_raw_pct'))}",
        f"  yield exported/annotated:       {_fmt_pct(yields.get('exported_over_annotated_pct'))}",
        "",
        "FUSION DROPS (row-level, one Relations object per row)",
        f"  {fusion.get('drop_reasons', {})}",
        f"  prune: {fusion.get('prune', {})}",
        f"  projector_error_takes: {fusion.get('projector_error_takes', 0)}",
        f"  frame_export: {fusion.get('frame_export', {})}",
        "",
        "DATASET",
        f"  unique labels: {dataset.get('unique_labels', '?')}  "
        f"count range {dataset.get('label_count_min', '?')}–{dataset.get('label_count_max', '?')}",
        f"  gaze in bbox:  {_fmt_pct(dataset.get('gaze_in_box_pct'))}",
        f"  missing JPEGs: {dataset.get('missing_images', '?')}",
        f"  cams: {dataset.get('cams', {})}",
        f"  top labels: {dataset.get('top_labels', [])[:8]}",
        f"  label tail: {dataset.get('label_frequency_buckets', {})}",
        "",
        "SPLITS",
        f"  {splits.get('splits', {})}",
        f"  leakage_ok={splits.get('leakage_ok')} {splits.get('leakage', {})}",
        "",
        "VOCAB / EMBEDS",
        f"  vocab_size={vocab.get('vocab_size')} embed_exists={vocab.get('embed_exists')} "
        f"shape={vocab.get('embed_shape')} match={vocab.get('embed_matches_vocab')}",
        "",
        f"Wrote JSON next to manifests.",
    ]
    return "\n".join(lines)


def _fmt_pct(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.1f}%"


def write_report(cfg: dict, report: dict) -> Path:
    out = manifests_dir(cfg) / "final_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(report, f, indent=2, default=str)
    text_path = manifests_dir(cfg) / "final_report.txt"
    text_path.write_text(format_report(report) + "\n")
    return out
