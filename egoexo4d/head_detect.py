"""YOLO-based head box estimation on exocentric frames."""

from __future__ import annotations

from functools import lru_cache
from typing import List, Optional, Sequence, Tuple

import numpy as np

from egoexo4d.masks import valid_xyxy


def _person_center(box: Sequence[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _point_in_box(x: float, y: float, box: Sequence[float]) -> bool:
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def _dist_to_box(x: float, y: float, box: Sequence[float]) -> float:
    x1, y1, x2, y2 = box
    cx = min(max(x, x1), x2)
    cy = min(max(y, y1), y2)
    return float((x - cx) ** 2 + (y - cy) ** 2)


def person_to_head_box(
    person_box: Sequence[float], *, head_fraction: float = 0.35
) -> List[float]:
    x1, y1, x2, y2 = (float(v) for v in person_box)
    h = y2 - y1
    head_y2 = y1 + h * head_fraction
    return [x1, y1, x2, head_y2]


class HeadDetector:
    """Lazy-loaded YOLO person detector."""

    def __init__(self, model_name: str = "yolov8n.pt", conf: float = 0.5, head_fraction: float = 0.35):
        self.model_name = model_name
        self.conf = conf
        self.head_fraction = head_fraction
        self._model = None

    def _load(self):
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(self.model_name)

    def detect_head(
        self,
        image_bgr: np.ndarray,
        gaze_xy: Optional[Tuple[float, float]] = None,
    ) -> Optional[List[float]]:
        self._load()
        results = self._model.predict(image_bgr, conf=self.conf, classes=[0], verbose=False)
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return None

        boxes = results[0].boxes.xyxy.cpu().numpy()
        best_idx = 0
        if gaze_xy is not None and len(boxes) > 1:
            gx, gy = gaze_xy
            containing = [
                i for i, b in enumerate(boxes) if _point_in_box(gx, gy, b)
            ]
            if containing:
                best_idx = containing[0]
            else:
                dists = [_dist_to_box(gx, gy, b) for b in boxes]
                best_idx = int(np.argmin(dists))

        person = boxes[best_idx].tolist()
        head = person_to_head_box(person, head_fraction=self.head_fraction)
        return head if valid_xyxy(head) else None


@lru_cache(maxsize=4)
def get_head_detector(model_name: str, conf: float, head_fraction: float) -> HeadDetector:
    return HeadDetector(model_name=model_name, conf=conf, head_fraction=head_fraction)
