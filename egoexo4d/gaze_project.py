"""Project MPS eye gaze to exocentric GoPro views."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


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

        paths_provider = mps.MpsDataPathsProvider(str(self.take_path))
        data_paths = paths_provider.get_data_paths()
        self._mps = mps.MpsDataProvider(data_paths)

        calib_path = self._find_gopro_calibs()
        static_calibrations = mps.read_static_camera_calibrations(str(calib_path))
        import math

        for static_calibration in static_calibrations:
            if float(static_calibration.quality) < self.calib_min_quality:
                continue
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
            self._exo_cams[static_calibration.camera_uid] = ExoCamera(
                name=static_calibration.camera_uid,
                camera=cam,
                transform_world_cam=static_calibration.transform_world_cam,
            )

        self._pose_timestamps = self._load_pose_timestamps(mps)

    @staticmethod
    def _find_gopro_calibs(take_path: Path) -> Path:
        for rel in (
            "trajectory/gopro_calibs.csv",
            "take_trajectory/gopro_calibs.csv",
            "mps/trajectory/gopro_calibs.csv",
        ):
            p = take_path / rel
            if p.exists():
                return p
        raise FileNotFoundError(f"No gopro_calibs.csv under {take_path}")

    def _find_gopro_calibs(self) -> Path:
        return GazeProjector._find_gopro_calibs(self.take_path)

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

        import projectaria_tools.core.mps as mps

        mps_provider = self._mps
        pose_info = mps_provider.get_closed_loop_pose(ts)
        if pose_info is None:
            return None

        if self.prefer_personalized and mps_provider.has_personalized_eyegaze():
            eye_gaze = mps_provider.get_personalized_eyegaze(ts)
        else:
            eye_gaze = mps_provider.get_general_eyegaze(ts)
        if eye_gaze is None:
            return None

        T_world_device = pose_info.transform_world_device
        device_calib = mps_provider.get_device_calibration()
        if device_calib is None:
            return None
        T_device_CPF = device_calib.get_transform_device_cpf()

        gaze_vector_in_cpf = mps.get_eyegaze_point_at_depth(
            eye_gaze.yaw, eye_gaze.pitch, eye_gaze.depth
        )
        gaze_vector_in_cpf = np.nan_to_num(gaze_vector_in_cpf)
        gaze_vector_in_device = T_device_CPF @ gaze_vector_in_cpf
        gaze_vector_in_world = T_world_device @ gaze_vector_in_device

        exo = self._exo_cams[cam]
        gaze_in_cam = exo.transform_world_cam.inverse() @ gaze_vector_in_world
        projection = exo.camera.project(gaze_in_cam)
        if projection is None:
            return None
        return float(projection[0]), float(projection[1])


@lru_cache(maxsize=64)
def get_projector(take_path: str, prefer_personalized: bool, calib_min_quality: float) -> GazeProjector:
    return GazeProjector(
        Path(take_path),
        prefer_personalized=prefer_personalized,
        calib_min_quality=calib_min_quality,
    )
