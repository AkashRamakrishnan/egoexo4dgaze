"""Unit tests for post-build report helpers."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

from egoexo4d.fuse_tier_a import summarize_unique_frame_status
from egoexo4d.report import (
    _parse_image_path,
    assemble_frame_section,
    collect_csv_stats,
    format_bytes,
    format_report,
    pct,
    pct_str,
)


def test_format_bytes_and_pct():
    assert format_bytes(512) == "512 B"
    assert format_bytes(1024).endswith("KiB")
    assert pct(1, 4) == 25.0
    assert pct(1, 0) is None
    assert pct_str(1, 0) == "n/a"
    assert pct_str(1, 4) == "25.0%"


def test_unique_frame_funnel_counts_each_frame_once():
    status = {
        ("t", "cam01", 1): {
            "projector_ok": True,
            "gaze_ok": True,
            "in_mask": False,
            "kept": False,
        },
        ("t", "cam01", 2): {
            "projector_ok": True,
            "gaze_ok": True,
            "in_mask": True,
            "kept": True,
        },
        ("t", "cam01", 3): {
            "projector_ok": False,
            "gaze_ok": False,
            "in_mask": False,
            "kept": False,
        },
    }
    funnel = summarize_unique_frame_status(status)
    assert funnel["unique_relations_frames_processed"] == 3
    assert funnel["unique_frames_with_gaze"] == 2
    assert funnel["unique_frames_gaze_in_object_mask"] == 1
    assert funnel["unique_frames_kept"] == 1


def test_parse_image_path():
    uid, cam, idx = _parse_image_path("abc123/cam02/000042.jpg")
    assert uid == "abc123"
    assert cam == "cam02"
    assert idx == 42


def test_csv_stats_and_format():
    tmp_path = Path(tempfile.mkdtemp())
    images = tmp_path / "images"
    img = images / "uid1" / "cam01"
    img.mkdir(parents=True)
    (img / "000010.jpg").write_bytes(b"x")
    ann = tmp_path / "annotations"
    ann.mkdir()
    csv_path = ann / "train_verified.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_path",
                "gaze_x",
                "gaze_y",
                "box_x1",
                "box_y1",
                "box_x2",
                "box_y2",
                "score",
                "label_text",
                "flag",
                "box_idx",
                "head_boxes",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "image_path": "uid1/cam01/000010.jpg",
                "gaze_x": "15",
                "gaze_y": "25",
                "box_x1": "10",
                "box_y1": "20",
                "box_x2": "40",
                "box_y2": "50",
                "score": "1.00",
                "label_text": "bowl",
                "flag": "in",
                "box_idx": "0",
                "head_boxes": "[[8.00,2.00,22.00,18.00]]",
            }
        )

    cfg = {
        "output_root": str(tmp_path),
        "egoexo_root": str(tmp_path / "raw"),
    }
    stats = collect_csv_stats(cfg)
    assert stats["rows"] == 1
    assert stats["unique_images"] == 1
    assert stats["missing_images"] == 0
    assert stats["gaze_in_box_rows"] == 1
    assert stats["cams"] == {"cam01": 1}

    frames = assemble_frame_section(
        {"raw_exo_frames": 1000, "exo_videos": 1},
        {"mask_rows": 10, "unique_annotated_frames": 8},
        {
            "frame_funnel": {
                "unique_frames_with_gaze": 6,
                "unique_frames_gaze_in_object_mask": 3,
                "unique_frames_kept": 1,
            }
        },
        stats,
    )
    assert frames["raw_exo_frames"] == 1000
    assert frames["frames_with_gaze"] == 6
    assert frames["dataset_rows"] == 1
    text = format_report(
        {
            "disk": {
                "raw_human": "1.00 GiB",
                "extracted_human": "10.00 MiB",
                "raw_bytes": 1024**3,
                "extracted_bytes": 10 * 1024**2,
                "raw_root": "/raw",
                "extracted_root": "/data",
                "selected_take_bytes": 0,
                "other_take_bytes": 0,
                "jpeg_count": 1,
                "jpeg_mean_human": "1.00 KiB",
            },
            "takes": {"index_stats": {"in_relations": 2, "selected": 1}, "selected_takes": 1},
            "frames": frames,
            "fusion": {"drop_reasons": {"gaze_outside_mask": 4}},
            "dataset": stats,
            "splits": {"leakage_ok": True, "leakage": {}, "splits": {}},
            "vocab": {"vocab_size": 1, "embed_exists": False},
        }
    )
    assert "with gaze coordinates" in text
    assert "gaze in object mask" in text
    assert "1.00 GiB" in text


if __name__ == "__main__":
    test_format_bytes_and_pct()
    test_unique_frame_funnel_counts_each_frame_once()
    test_parse_image_path()
    test_csv_stats_and_format()
    print("ok")
