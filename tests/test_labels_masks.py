"""Unit tests for label normalization and mask helpers."""

from __future__ import annotations

import numpy as np

from egoexo4d.index_takes import _is_cooking_take, unwrap_relations_dict
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


def test_unwrap_relations_annotations_wrapper():
    wrapped = {
        "date": "2024",
        "annotations": {
            "uid-1": {"take_name": "sfu_cooking_008_3", "object_masks": {}},
        },
    }
    inner = unwrap_relations_dict(wrapped)
    assert "uid-1" in inner
    assert inner["uid-1"]["take_name"] == "sfu_cooking_008_3"


def test_cooking_id_as_string_and_take_name():
    cfg = {"scenario": {"parent_task_id": 1000, "parent_task_name": "Cooking"}}
    assert _is_cooking_take({"parent_task_id": "1000", "take_name": "x"}, cfg)
    assert _is_cooking_take({"parent_task_name": "Cooking", "take_name": "x"}, cfg)
    assert _is_cooking_take({"take_name": "iiith_cooking_12_1"}, cfg)
    assert not _is_cooking_take({"take_name": "sfu_covid_001_2"}, cfg)
    assert not _is_cooking_take({"take_name": "cmu_bike01_2"}, cfg)
