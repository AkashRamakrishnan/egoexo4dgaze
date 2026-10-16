"""Dataset QA: coverage, leakage, validity checks."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from egoexo4d.paths import annotations_dir, manifests_dir, output_root


def _load_split_rows(name: str, csv_dir: Path) -> List[dict]:
    path = csv_dir / f"{name}_split.csv"
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _gaze_in_box(row: dict) -> bool:
    gx = float(row["gaze_x"])
    gy = float(row["gaze_y"])
    x1, y1, x2, y2 = (
        float(row["box_x1"]),
        float(row["box_y1"]),
        float(row["box_x2"]),
        float(row["box_y2"]),
    )
    return x1 <= gx <= x2 and y1 <= gy <= y2


def run_qa(cfg: dict) -> dict:
    csv_dir = annotations_dir(cfg)
    splits = {name: _load_split_rows(name, csv_dir) for name in ("train", "val", "test")}

    stats: Dict[str, Any] = {"splits": {}}
    all_images = set()
    for name, rows in splits.items():
        images = {r["image_path"] for r in rows}
        labels = Counter(r["label_text"] for r in rows)
        gaze_in_box = sum(1 for r in rows if _gaze_in_box(r))
        stats["splits"][name] = {
            "rows": len(rows),
            "unique_images": len(images),
            "unique_labels": len(labels),
            "gaze_in_box_pct": (100.0 * gaze_in_box / len(rows)) if rows else 0.0,
            "top_labels": labels.most_common(10),
        }
        all_images.update(images)

    # Leakage check
    train_imgs = {r["image_path"] for r in splits["train"]}
    val_imgs = {r["image_path"] for r in splits["val"]}
    test_imgs = {r["image_path"] for r in splits["test"]}
    leakage = {
        "train_val": len(train_imgs & val_imgs),
        "train_test": len(train_imgs & test_imgs),
        "val_test": len(val_imgs & test_imgs),
    }
    stats["leakage"] = leakage
    stats["leakage_ok"] = all(v == 0 for v in leakage.values())

    build_stats_path = manifests_dir(cfg) / "build_stats.json"
    if build_stats_path.exists():
        with build_stats_path.open() as f:
            stats["build"] = json.load(f)

    stats_path = manifests_dir(cfg) / "qa_report.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w") as f:
        json.dump(stats, f, indent=2)

    print(f"[qa] report -> {stats_path}")
    print(f"[qa] leakage_ok={stats['leakage_ok']} {leakage}")
    return stats
