#!/usr/bin/env python3
"""Check that an Ego-Exo4D download is ready for scripts/02_build_index.py."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from egoexo4d.index_takes import build_cooking_takes_manifest  # noqa: E402
from egoexo4d.paths import egoexo_root, load_config, relations_path, takes_json_path  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument(
        "--min-selected",
        type=int,
        default=1,
        help="Minimum complete scenario takes (MPS + calib + exo video + Aria VRS).",
    )
    p.add_argument(
        "--require-val",
        action="store_true",
        help="Fail if relations_val.json is missing (full MVP). Sample QA does not need this.",
    )
    args = p.parse_args()
    cfg = load_config(args.config)
    root = egoexo_root(cfg)

    ok = True
    required_top = [
        takes_json_path(cfg),
        relations_path(cfg, "train"),
    ]
    if args.require_val:
        required_top.append(relations_path(cfg, "val"))
    print(f"Ego-Exo root: {root}")
    for path in required_top:
        if path.exists():
            print(f"  OK  {path.relative_to(root)}")
        else:
            print(f"  MISSING  {path}")
            ok = False

    val_path = relations_path(cfg, "val")
    if not args.require_val and not val_path.exists():
        print(f"  optional missing  {val_path} (ok for sample QA)")

    if not takes_json_path(cfg).exists() or not relations_path(cfg, "train").exists():
        print(
            "\nNeed metadata + Relations annotations before sample/build.\n"
            "  EGOEXO_ROOT=<root> ./scripts/01_download.sh --metadata-only\n"
        )
        return 1

    manifest = build_cooking_takes_manifest(cfg)
    payload = json.loads(manifest.read_text())
    stats = payload.get("stats", {})
    n = len(payload.get("takes", []))
    label = stats.get("scenario") or "scenario"
    print(f"\n{label} manifest: {manifest}")
    print(f"  selected complete takes: {n} (need >= {args.min_selected})")
    print(f"  stats: {json.dumps(stats, indent=2)}")

    if n < args.min_selected:
        print(
            f"\nNot enough complete {label} takes "
            "(each needs Relations + MPS gaze + GoPro calib + exo cam*.mp4 "
            "+ aria*.vrs / aria01_noimagestreams.vrs).\n"
            "VRS is --parts take_vrs_noimagestream (not included in --parts takes).\n"
            "  EGOEXO_ROOT=<root> CONFIG=<config> ./scripts/01_download.sh --vrs-only\n"
        )
        ok = False
    else:
        print(f"\nReady for sample QA / index ({n} complete {label} take(s) on disk).")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
