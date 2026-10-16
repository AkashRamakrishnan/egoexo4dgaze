#!/usr/bin/env python3
"""Orchestrate Tier-A fusion, frame export, CSV, splits, and vocab."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from egoexo4d.export_csv import export_verified_csv, prune_rare_labels
from egoexo4d.export_frames import export_frames
from egoexo4d.fuse_tier_a import fuse_tier_a_rows, save_build_stats
from egoexo4d.index_relations import build_frame_index
from egoexo4d.index_takes import build_cooking_takes_manifest
from egoexo4d.paths import load_config, manifests_dir
from egoexo4d.qa import run_qa
from egoexo4d.splits import run_splits
from egoexo4d.vocab import run_vocab


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--skip-index", action="store_true")
    parser.add_argument("--skip-frames", action="store_true")
    parser.add_argument("--skip-qa", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if not args.skip_index:
        build_cooking_takes_manifest(cfg)
        build_frame_index(cfg)

    rows, stats = fuse_tier_a_rows(cfg)
    print(f"[build] fuse drop_reasons={stats.get('drop_reasons')}")
    if stats.get("projector_errors"):
        print(f"[build] projector_errors={stats['projector_errors']}")
    min_count = int(cfg.get("filters", {}).get("min_label_count", 5))
    rows, prune_stats = prune_rare_labels(rows, min_count)
    stats["prune"] = prune_stats
    save_build_stats(cfg, stats)

    tier_a_path = manifests_dir(cfg) / "tier_a_rows.json"
    with tier_a_path.open("w") as f:
        json.dump(rows, f)
    print(f"[build] cached {len(rows)} tier-A rows -> {tier_a_path}")

    if not args.skip_frames and not cfg.get("build", {}).get("skip_frame_export", False):
        frame_stats = export_frames(cfg, rows)
        stats["frame_export"] = frame_stats
        save_build_stats(cfg, stats)

    export_verified_csv(cfg, rows)
    run_splits(cfg)
    run_vocab(cfg)

    if not args.skip_qa:
        run_qa(cfg)

    print("[build] complete")


if __name__ == "__main__":
    main()
