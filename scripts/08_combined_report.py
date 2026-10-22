#!/usr/bin/env python3
"""Combine per-scenario final_report.json files into overall tables + HTML."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from egoexo4d.combined_report import (
    build_combined,
    load_reports_from_disk,
    rebuild_reports,
    write_combined,
)
from egoexo4d.paths import REPO_ROOT


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--rebuild",
        action="store_true",
        help="Recompute each scenario's final_report.json from CSVs (slow).",
    )
    p.add_argument(
        "--count-raw-frames",
        action="store_true",
        help="With --rebuild, count frames in cam*.mp4 (slow).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "combined_report",
        help="Output prefix (writes .txt, .html, .json).",
    )
    args = p.parse_args()
    reports = (
        rebuild_reports(skip_raw_frames=not args.count_raw_frames)
        if args.rebuild
        else load_reports_from_disk()
    )
    payload = build_combined(reports)
    paths = write_combined(payload, args.out)
    print(payload["text"])
    print(f"[combined] text -> {paths['txt']}")
    print(f"[combined] html -> {paths['html']}")
    print(f"[combined] json -> {paths['json']}")
    missing = [r["name"] for r in payload["scenarios"] if r.get("status") != "ok"]
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
