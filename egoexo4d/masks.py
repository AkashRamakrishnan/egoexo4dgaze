"""Decode Relations segmentation masks and derive bounding boxes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from pycocotools import mask as mask_utils

from egoexo4d.lzstring import decompress_from_encoded_uri


@dataclass
class MaskAnnotation:
    width: int
    height: int
    encoded_mask: str


def decode_mask_dict(annotation: Dict[str, Any]) -> np.ndarray:
    """Decode a Relations ``annotation[frame]`` object to a binary H×W mask."""
    w = int(annotation["width"])
    h = int(annotation["height"])
    encoded = annotation["encodedMask"]
    return decode_mask_obj(MaskAnnotation(width=w, height=h, encoded_mask=encoded))


def decode_mask_obj(mask: MaskAnnotation) -> np.ndarray:
    decomp_string = decompress_from_encoded_uri(mask.encoded_mask)
    if not decomp_string:
        raise ValueError("Failed to decompress encodedMask")
    rle_obj = {
        "size": [mask.height, mask.width],
        "counts": decomp_string.encode(),
    }
    return mask_utils.decode(rle_obj)


def mask_to_bbox(mask: np.ndarray) -> Tuple[float, float, float, float]:
    """Tight xyxy bbox from a binary mask. Raises if mask is empty."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise ValueError("Empty mask")
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


def point_in_mask(x: float, y: float, mask: np.ndarray) -> bool:
    h, w = mask.shape[:2]
    ix, iy = int(round(x)), int(round(y))
    if ix < 0 or iy < 0 or ix >= w or iy >= h:
        return False
    return bool(mask[iy, ix] > 0)


def valid_xyxy(box: Sequence[float]) -> bool:
    x1, y1, x2, y2 = (float(v) for v in box)
    return x2 > x1 and y2 > y1 and min(x1, y1, x2, y2) >= 0.0


def head_box_json(box: Sequence[float]) -> str:
    x1, y1, x2, y2 = (float(v) for v in box)
    return f"[[{x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f}]]"
