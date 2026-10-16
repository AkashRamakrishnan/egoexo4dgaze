"""Project MPS eye gaze to exocentric GoPro views."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def _first_existing(root: Path, relatives: Tuple[str, ...]) -> Optional[Path]:
    for rel in relatives:
        path = root / rel
        if path.exists():
            return path
    return None


def find_gopro_calibs(take_path: Path) -> Path:
    found = _first_existing(
        take_path,
        (
            "trajectory/gopro_calibs.csv",
            "take_trajectory/gopro_calibs.csv",
            "mps/trajectory/gopro_calibs.csv",
            "slam/gopro_calibs.csv",
        ),
    )
    if found is not None:
        return found
    hits = sorted(Path(take_path).rglob("gopro_calibs.csv"))
    if hits:
        return hits[0]
    raise FileNotFoundError(f"No gopro_calibs.csv under {take_path}")


def _vrs_rank(path: Path) -> Tuple[int, str]:
    name = path.name.lower()
    if "noimagestream" in name:
        return (0, name)
    if name.startswith("aria"):
        return (1, name)
    return (2, name)


def _collect_vrs(root: Path) -> List[Path]:
    if not root.is_dir():
        return []
    found = list(root.glob("*.vrs"))
    if found:
        return found
    return list(root.rglob("*.vrs"))


def find_aria_vrs(take_path: Path) -> Optional[Path]:
    """Factory CPF calib is in take-level VRS (usually ``aria01_noimagestreams.vrs``)."""
    take_path = Path(take_path)
    candidates = _collect_vrs(take_path)
    capture = re.sub(r"_\d+$", "", take_path.name)
    capture_dir = take_path.parent.parent / "captures" / capture
    candidates.extend(_collect_vrs(capture_dir))
    if not candidates:
        return None
    uniq = {p.resolve(): p for p in candidates}
    return sorted(uniq.values(), key=_vrs_rank)[0]


@dataclass
class ExoCamera:
    name: str
    camera: object
    transform_world_cam: object


class GazeProjector:
    """Per-take cache for MPS gaze → exo 2D projection."""

    def __init__(
        self,
        take_path: Path,
        *,
        prefer_personalized: bool = True,
        calib_min_quality: float = 1.0,
    ):
        self.take_path = Path(take_path)
        self.prefer_personalized = prefer_personalized
        self.calib_min_quality = calib_min_quality
        self._mps = None
        self._device_calib = None
        self._T_device_CPF = None
        self._exo_cams: Dict[str, ExoCamera] = {}
        self._pose_timestamps: Optional[np.ndarray] = None
        self._video_fps: Dict[str, float] = {}
        self._init()

    def _init(self) -> None:
        try:
            import projectaria_tools.core.mps as mps
            from projectaria_tools.core.calibration import (
                CameraCalibration,
                KANNALA_BRANDT_K3,
            )
        except ImportError as exc:
            raise ImportError(
                "projectaria-tools is required for gaze projection. "
                "Install with: pip install 'egoexo4d[gaze]'"
            ) from exc

        self._mps = self._make_mps_provider(mps)
        self._load_device_calibration()
        calib_path = find_gopro_calibs(self.take_path)
        static_calibrations = mps.read_static_camera_calibrations(str(calib_path))
        import math

        cam_errors: list[str] = []
        for static_calibration in static_calibrations:
            if float(static_calibration.quality) < self.calib_min_quality:
                continue
            try:
                cam = CameraCalibration(
                    static_calibration.camera_uid,
                    KANNALA_BRANDT_K3,
                    static_calibration.intrinsics,
                    static_calibration.transform_world_cam,
                    static_calibration.width,
                    static_calibration.height,
                    None,
                    math.pi,
                    "",
                )
            except Exception as exc:
                cam_errors.append(f"{static_calibration.camera_uid}: {exc}")
                continue
            self._exo_cams[static_calibration.camera_uid] = ExoCamera(
                name=static_calibration.camera_uid,
                camera=cam,
                transform_world_cam=static_calibration.transform_world_cam,
            )

        if not self._exo_cams and cam_errors:
            raise RuntimeError(
                "Failed to build any exo CameraCalibration: " + "; ".join(cam_errors[:4])
            )

        self._pose_timestamps = self._load_pose_timestamps(mps)

    def _load_device_calibration(self) -> None:
        """Aria 2.x: CPF / RGB calib come from VRS, not MpsDataProvider."""
        from projectaria_tools.core import data_provider

        vrs_path = find_aria_vrs(self.take_path)
        if vrs_path is None:
            raise FileNotFoundError(
                f"No *.vrs under {self.take_path} "
                "(need --parts take_vrs_noimagestream: aria01_noimagestreams.vrs)"
            )
        vrs = data_provider.create_vrs_data_provider(str(vrs_path))
        if vrs is None:
            raise RuntimeError(f"Could not open VRS {vrs_path}")
        calib = vrs.get_device_calibration()
        if calib is None:
            raise RuntimeError(f"No device calibration in {vrs_path}")
        self._device_calib = calib
        self._T_device_CPF = calib.get_transform_device_cpf()

    def _make_mps_provider(self, mps_module):
        paths_provider = mps_module.MpsDataPathsProvider(str(self.take_path))
        data_paths = paths_provider.get_data_paths()
        self._fill_missing_mps_files(data_paths)
        return mps_module.MpsDataProvider(data_paths)

    def _fill_missing_mps_files(self, data_paths) -> None:
        slam = getattr(data_paths, "slam", None)
        if slam is not None:
            current = getattr(slam, "closed_loop_trajectory", "") or ""
            if not current or not Path(str(current)).exists():
                found = _first_existing(
                    self.take_path,
                    (
                        "trajectory/closed_loop_trajectory.csv",
                        "take_trajectory/closed_loop_trajectory.csv",
                        "slam/closed_loop_trajectory.csv",
                        "mps/slam/closed_loop_trajectory.csv",
                        "mps/trajectory/closed_loop_trajectory.csv",
                    ),
                )
                if found is not None:
                    slam.closed_loop_trajectory = str(found)
        eyegaze = getattr(data_paths, "eyegaze", None)
        if eyegaze is not None:
            current = getattr(eyegaze, "general_eyegaze", "") or ""
            if not current or not Path(str(current)).exists():
                found = _first_existing(
                    self.take_path,
                    (
                        "eye_gaze/general_eye_gaze.csv",
                        "mps/eye_gaze/general_eye_gaze.csv",
                    ),
                )
                if found is not None:
                    eyegaze.general_eyegaze = str(found)

    def _load_pose_timestamps(self, mps_module) -> Optional[np.ndarray]:
        for rel in (
            "trajectory/closed_loop_trajectory.csv",
            "take_trajectory/closed_loop_trajectory.csv",
            "mps/trajectory/closed_loop_trajectory.csv",
        ):
            path = self.take_path / rel
            if not path.exists():
                continue
            trajectory = mps_module.read_closed_loop_trajectory(str(path))
            if trajectory:
                return np.array(
                    [p.tracking_timestamp.total_seconds() * 1e9 for p in trajectory],
                    dtype=np.int64,
                )
        getter = getattr(self._mps, "get_closed_loop_trajectory", None)
        if callable(getter):
            trajectory = getter()
            if trajectory:
                return np.array(
                    [p.tracking_timestamp.total_seconds() * 1e9 for p in trajectory],
                    dtype=np.int64,
                )
        return None

    def video_fps(self, cam: str) -> float:
        if cam in self._video_fps:
            return self._video_fps[cam]
        video_path = self.take_path / "frame_aligned_videos" / f"{cam}.mp4"
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        self._video_fps[cam] = float(fps)
        return self._video_fps[cam]

    def frame_to_timestamp_ns(self, frame_idx: int, cam: str) -> Optional[int]:
        if self._pose_timestamps is None or len(self._pose_timestamps) == 0:
            return None
        fps = self.video_fps(cam)
        target = self._pose_timestamps[0] + int((frame_idx / fps) * 1e9)
        idx = int(np.searchsorted(self._pose_timestamps, target))
        idx = min(max(idx, 0), len(self._pose_timestamps) - 1)
        return int(self._pose_timestamps[idx])

    def project_gaze(self, cam: str, frame_idx: int) -> Optional[Tuple[float, float]]:
        """Return (x, y) pixel coords on exo cam, or None if unavailable."""
        if cam not in self._exo_cams:
            return None
        ts = self.frame_to_timestamp_ns(frame_idx, cam)
        if ts is None:
            return None

        mps_provider = self._mps
        pose_info = mps_provider.get_closed_loop_pose(ts)
        if pose_info is None:
            return None

        packed = self._eyegaze_in_device(ts)
        if packed is None:
            return None
        _device_calib, gaze_vector_in_device = packed
        T_world_device = pose_info.transform_world_device
        gaze_vector_in_world = T_world_device @ gaze_vector_in_device

        exo = self._exo_cams[cam]
        gaze_in_cam = exo.transform_world_cam.inverse() @ gaze_vector_in_world
        projection = exo.camera.project(gaze_in_cam)
        if projection is None:
            return None
        return float(projection[0]), float(projection[1])

    def _eyegaze_in_device(self, ts: int):
        import projectaria_tools.core.mps as mps

        mps_provider = self._mps
        if self.prefer_personalized and mps_provider.has_personalized_eyegaze():
            eye_gaze = mps_provider.get_personalized_eyegaze(ts)
        else:
            eye_gaze = mps_provider.get_general_eyegaze(ts)
        if eye_gaze is None:
            return None
        if self._device_calib is None or self._T_device_CPF is None:
            return None
        gaze_vector_in_cpf = mps.get_eyegaze_point_at_depth(
            eye_gaze.yaw, eye_gaze.pitch, eye_gaze.depth
        )
        gaze_vector_in_cpf = np.nan_to_num(gaze_vector_in_cpf)
        return self._device_calib, self._T_device_CPF @ gaze_vector_in_cpf

    def project_ego_gaze(
        self,
        frame_idx: int,
        timestamp_cam: str,
        calib_label: str = "camera-rgb",
    ) -> Optional[Tuple[float, float, Tuple[float, float]]]:
        """Project MPS gaze onto an Aria camera.

        Returns ``(x, y, (calib_w, calib_h))`` in the calibration pixel frame,
        or None if unavailable. Callers should scale to the decoded video size.
        """
        ts = self.frame_to_timestamp_ns(frame_idx, timestamp_cam)
        if ts is None:
            return None
        packed = self._eyegaze_in_device(ts)
        if packed is None:
            return None
        device_calib, gaze_in_device = packed
        rgb_cam = device_calib.get_camera_calib(calib_label)
        if rgb_cam is None:
            return None
        T_device_cam = rgb_cam.get_transform_device_camera()
        gaze_in_cam = T_device_cam.inverse() @ gaze_in_device
        projection = rgb_cam.project(gaze_in_cam)
        if projection is None:
            return None
        try:
            size = rgb_cam.get_image_size()
            cal_wh = (float(size[0]), float(size[1]))
        except Exception:
            cal_wh = (
                float(getattr(rgb_cam, "get_image_width", lambda: 0)() or 0),
                float(getattr(rgb_cam, "get_image_height", lambda: 0)() or 0),
            )
        return float(projection[0]), float(projection[1]), cal_wh


@lru_cache(maxsize=64)
def get_projector(take_path: str, prefer_personalized: bool, calib_min_quality: float) -> GazeProjector:
    return GazeProjector(
        Path(take_path),
        prefer_personalized=prefer_personalized,
        calib_min_quality=calib_min_quality,
    )
