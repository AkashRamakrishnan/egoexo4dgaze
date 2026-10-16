"""Extract exocentric JPEG frames referenced by Tier-A rows."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

import cv2
from tqdm import tqdm

from egoexo4d.paths import egoexo_root, images_dir


def _unique_frame_jobs(rows: Iterable[dict]) -> List[Tuple[str, str, str, int]]:
    seen: Set[tuple] = set()
    jobs: List[Tuple[str, str, str, int]] = []
    for row in rows:
        key = (row["take_uid"], row["take_name"], row["cam"], int(row["frame_idx"]))
        if key in seen:
            continue
        seen.add(key)
        jobs.append(key)
    return jobs


def _extract_one(
    root: Path,
    out_root: Path,
    take_uid: str,
    take_name: str,
    cam: str,
    frame_idx: int,
    jpeg_quality: int,
) -> Tuple[str, bool, str]:
    rel = f"{take_uid}/{cam}/{frame_idx:06d}.jpg"
    out_path = out_root / rel
    if out_path.exists():
        return rel, True, "exists"

    video_path = root / "takes" / take_name / "frame_aligned_videos" / f"{cam}.mp4"
    if not video_path.exists():
        return rel, False, "missing_video"

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return rel, False, "read_failed"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok_write = cv2.imwrite(
        str(out_path),
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
    )
    return rel, bool(ok_write), "written" if ok_write else "write_failed"


def export_frames(cfg: Dict[str, Any], rows: List[dict]) -> dict:
    root = egoexo_root(cfg)
    out_root = images_dir(cfg)
    export_cfg = cfg.get("export", {})
    jpeg_quality = int(export_cfg.get("jpeg_quality", 92))
    num_workers = int(export_cfg.get("num_workers", 4))

    jobs = _unique_frame_jobs(rows)
    stats = {"total": len(jobs), "ok": 0, "failed": 0, "skipped": 0}

    if num_workers <= 1:
        iterator = (
            _extract_one(root, out_root, *job, jpeg_quality) for job in jobs
        )
        for rel, ok, reason in tqdm(iterator, total=len(jobs), desc="export_frames"):
            if ok:
                stats["ok"] += 1
                if reason == "exists":
                    stats["skipped"] += 1
            else:
                stats["failed"] += 1
        return stats

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {
            pool.submit(_extract_one, root, out_root, *job, jpeg_quality): job
            for job in jobs
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="export_frames"):
            _rel, ok, reason = fut.result()
            if ok:
                stats["ok"] += 1
                if reason == "exists":
                    stats["skipped"] += 1
            else:
                stats["failed"] += 1
    return stats
