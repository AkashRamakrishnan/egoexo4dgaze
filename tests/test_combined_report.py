"""Combined per-scenario + overall report tables."""

from __future__ import annotations

from egoexo4d.combined_report import (
    build_combined,
    overall_summary,
    scenario_summaries,
)


def _report(*, rows: int, images: int, labels: int, takes: int, name: str) -> dict:
    return {
        "output_root": f"./data_{name.lower()}",
        "takes": {
            "selected_takes": takes,
            "index_stats": {
                "scenario": name,
                "scenario_in_takes_json": takes + 10,
                "in_relations": takes + 5,
                "with_mps": takes,
                "with_calib": takes,
                "with_vrs": takes,
                "with_exo_video": takes,
                "selected": takes,
            },
        },
        "dataset": {
            "rows": rows,
            "unique_images": images,
            "unique_labels": labels,
            "label_count_min": 1,
            "label_count_max": 40,
            "missing_images": 0,
            "gaze_in_box_rows": rows,
            "gaze_in_box_pct": 100.0,
            "takes_with_zero_tier_a_rows": 0,
            "label_frequency_buckets": {"eq_1_to_4": 3},
        },
        "frames": {
            "relations_annotated_frames": images * 2,
            "frames_with_gaze": images,
            "frames_gaze_in_object_mask": images,
            "exported_images": images,
            "dataset_rows": rows,
        },
        "fusion": {"prune": {"dropped_rows": 0, "dropped_labels": 0}},
        "disk": {
            "jpeg_count": images,
            "extracted_bytes": 1024,
            "raw_bytes": 3 * 1024**4,
        },
        "splits": {
            "leakage_ok": True,
            "splits": {
                "train": {"rows": int(rows * 0.7), "unique_images": int(images * 0.7)},
                "val": {"rows": int(rows * 0.15), "unique_images": int(images * 0.15)},
                "test": {"rows": int(rows * 0.15), "unique_images": int(images * 0.15)},
            },
        },
        "vocab": {"vocab_size": labels, "embed_exists": True, "embed_matches_vocab": True},
    }


def test_overall_sums_ok_scenarios_only():
    reports = {
        "cooking": _report(rows=100, images=80, labels=10, takes=5, name="Cooking"),
        "health": _report(rows=50, images=40, labels=8, takes=3, name="Health"),
        "bike_repair": None,
        "music": None,
        "basketball": None,
        "soccer": None,
    }
    payload = build_combined(reports, union_labels=15)
    overall = payload["overall"]
    assert overall["scenarios_ok"] == 2
    assert overall["scenarios_missing"] == 4
    assert overall["csv_rows"] == 150
    assert overall["unique_images"] == 120
    assert overall["selected_takes"] == 8
    assert overall["unique_labels_sum"] == 18
    assert overall["unique_labels_union"] == 15
    assert overall["gaze_in_box_pct"] == 100.0
    text = payload["text"]
    assert "Cooking" in text
    assert "Health" in text
    assert "Overall" in text
    assert "150" in text.replace(",", "")
    html = payload["html"]
    assert "<table>" in html
    assert "2/6" in html


def test_missing_row_does_not_break_table():
    rows = scenario_summaries({"cooking": None})
    cooking = next(r for r in rows if r["slug"] == "cooking")
    assert cooking["status"] == "missing"
    overall = overall_summary(rows)
    assert overall["csv_rows"] == 0
    assert overall["scenarios_ok"] == 0
