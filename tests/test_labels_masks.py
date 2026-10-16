"""Unit tests for label normalization and mask helpers."""

from __future__ import annotations

import numpy as np

from egoexo4d.labels import is_hand_track, normalize_label, should_keep_track
from egoexo4d.masks import mask_to_bbox, point_in_mask, valid_xyxy


def test_normalize_label():
    assert normalize_label("stainless bowl_0") == "stainless bowl"
    assert normalize_label("knife_12") == "knife"


def test_hand_filter():
    assert is_hand_track("left hand_0")
    assert not should_keep_track("left hand_0", drop_hand_tracks=True)
    assert should_keep_track("stainless bowl_0", drop_hand_tracks=True)


def test_mask_bbox_and_point():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:5, 3:7] = 1
    box = mask_to_bbox(mask)
    assert valid_xyxy(box)
    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2
    assert point_in_mask(cx, cy, mask)
    assert not point_in_mask(0, 0, mask)
