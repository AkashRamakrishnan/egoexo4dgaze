"""Export GazeFollow-style CSV from Tier-A rows."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from egoexo4d.masks import head_box_json
from egoexo4d.paths import annotations_dir


CSV_FIELDS = [
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
]


def prune_rare_labels(rows: List[dict], min_count: int) -> tuple[List[dict], dict]:
    if min_count <= 1:
        return rows, {"dropped_rows": 0, "dropped_labels": 0}

    counts = Counter(r["label_text"] for r in rows)
    keep = {label for label, c in counts.items() if c >= min_count}
    kept = [r for r in rows if r["label_text"] in keep]
    return kept, {
        "dropped_rows": len(rows) - len(kept),
        "dropped_labels": len(counts) - len(keep),
        "min_count": min_count,
    }


def rows_to_gazefollow(rows: List[dict]) -> List[dict]:
    out: List[dict] = []
    for row in rows:
        out.append(
            {
                "image_path": row["image_path"],
                "gaze_x": f"{float(row['gaze_x']):.2f}",
                "gaze_y": f"{float(row['gaze_y']):.2f}",
                "box_x1": f"{float(row['box_x1']):.2f}",
                "box_y1": f"{float(row['box_y1']):.2f}",
                "box_x2": f"{float(row['box_x2']):.2f}",
                "box_y2": f"{float(row['box_y2']):.2f}",
                "score": f"{float(row.get('score', 1.0)):.2f}",
                "label_text": row["label_text"],
                "flag": row.get("flag", "in"),
                "box_idx": str(int(row.get("box_idx", 0))),
                "head_boxes": head_box_json(
                    [
                        row["head_x1"],
                        row["head_y1"],
                        row["head_x2"],
                        row["head_y2"],
                    ]
                ),
            }
        )
    return out


def write_train_verified_csv(rows: List[dict], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gazefollow_rows = rows_to_gazefollow(rows)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(gazefollow_rows)
    print(f"[export_csv] wrote {len(gazefollow_rows)} rows -> {out_path}")
    return out_path


def export_verified_csv(cfg: Dict[str, Any], rows: List[dict]) -> Path:
    min_count = int(cfg.get("filters", {}).get("min_label_count", 5))
    pruned, prune_stats = prune_rare_labels(rows, min_count)
    out_path = annotations_dir(cfg) / "train_verified.csv"
    write_train_verified_csv(pruned, out_path)
    return out_path
