"""Video helpers for take dumps (frame-aligned ego / exo mp4s)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

ARIA_RGB_TOKEN = "214"
ARIA_SLAM_LEFT_TOKEN = "1201"
ARIA_SLAM_RIGHT_TOKEN = "1202"

CALIB_BY_TOKEN = {
    ARIA_RGB_TOKEN: "camera-rgb",
    ARIA_SLAM_LEFT_TOKEN: "camera-slam-left",
    ARIA_SLAM_RIGHT_TOKEN: "camera-slam-right",
}


def read_video_frame(video_path: Path, frame_idx: int) -> Optional[np.ndarray]:
    """Return a BGR frame or None."""
    if not video_path.exists():
        return None
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    return frame


def find_aria_rgb_video(take_path: Path) -> Optional[Path]:
    """Best Aria RGB (214) stream under ``frame_aligned_videos/``."""
    video_dir = Path(take_path) / "frame_aligned_videos"
    if not video_dir.is_dir():
        return None
    ranked: list[tuple[int, Path]] = []
    for path in video_dir.glob("*.mp4"):
        stem = path.stem.lower()
        if stem.startswith("cam"):
            continue
        score = 0
        if ARIA_RGB_TOKEN in stem:
            score = 3
        elif "rgb" in stem:
            score = 2
        elif stem.startswith("aria"):
            score = 1
        if score:
            ranked.append((score, path))
    if not ranked:
        return None
    ranked.sort(key=lambda item: (-item[0], item[1].name))
    return ranked[0][1]


def calib_label_for_video(video_path: Path) -> str:
    stem = video_path.stem.lower()
    for token, label in CALIB_BY_TOKEN.items():
        if token in stem:
            return label
    return "camera-rgb"


def scale_gaze(
    xy: Tuple[float, float],
    calib_wh: Tuple[float, float],
    frame_wh: Tuple[int, int],
) -> Tuple[float, float]:
    x, y = xy
    cal_w, cal_h = calib_wh
    frame_w, frame_h = frame_wh
    if cal_w > 1 and cal_h > 1:
        x = x * (frame_w / cal_w)
        y = y * (frame_h / cal_h)
    return x, y
