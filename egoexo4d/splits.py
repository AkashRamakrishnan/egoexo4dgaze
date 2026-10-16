"""Image-level train/val/test splits (GazeFollow-style)."""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from egoexo4d.export_csv import CSV_FIELDS
from egoexo4d.paths import annotations_dir

SPLIT_NAMES = ("train", "val", "test")


def create_image_level_splits(
    verified_csv: Path,
    out_dir: Path,
    ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
    overwrite: bool = False,
) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_paths = {name: out_dir / f"{name}_split.csv" for name in SPLIT_NAMES}
    if not overwrite and all(p.exists() for p in out_paths.values()):
        return out_paths

    with verified_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if not fieldnames or "image_path" not in fieldnames:
        raise ValueError(f"Invalid CSV (missing image_path): {verified_csv}")

    by_image: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        by_image[row["image_path"]].append(row)

    images = sorted(by_image.keys())
    rng = random.Random(seed)
    rng.shuffle(images)

    n = len(images)
    r_train, r_val, _ = ratios
    n_train = int(round(n * r_train))
    n_val = int(round(n * r_val))
    n_test = n - n_train - n_val
    if min(n_train, n_val, n_test) <= 0 and n >= 3:
        n_train = max(1, n_train)
        n_val = max(1, n_val)
        n_test = max(1, n - n_train - n_val)

    split_images = {
        "train": images[:n_train],
        "val": images[n_train : n_train + n_val],
        "test": images[n_train + n_val :],
    }

    s_train, s_val, s_test = (set(split_images[k]) for k in SPLIT_NAMES)
    assert s_train.isdisjoint(s_val) and s_train.isdisjoint(s_test) and s_val.isdisjoint(s_test)

    for name, img_list in split_images.items():
        out_rows = [row for ip in img_list for row in by_image[ip]]
        with out_paths[name].open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames or CSV_FIELDS)
            writer.writeheader()
            writer.writerows(out_rows)

    print(
        f"[splits] images train/val/test = "
        f"{len(split_images['train'])}/{len(split_images['val'])}/{len(split_images['test'])}"
    )
    return out_paths


def run_splits(cfg: dict) -> Dict[str, Path]:
    split_cfg = cfg.get("splits", {})
    ratios = tuple(split_cfg.get("ratios", [0.7, 0.15, 0.15]))
    seed = int(split_cfg.get("seed", 42))
    verified = annotations_dir(cfg) / "train_verified.csv"
    return create_image_level_splits(
        verified_csv=verified,
        out_dir=annotations_dir(cfg),
        ratios=ratios,  # type: ignore[arg-type]
        seed=seed,
        overwrite=True,
    )
