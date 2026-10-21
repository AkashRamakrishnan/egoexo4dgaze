#!/usr/bin/env python3
"""Print disk + frame-funnel + dataset stats after a Cooking MVP build."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from egoexo4d.paths import load_config, manifests_dir
from egoexo4d.report import build_report, format_report, write_report


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument(
        "--skip-raw-frames",
        action="store_true",
        help="Skip opening cam*.mp4 for frame counts (disk/CSV still collected).",
    )
    args = p.parse_args()
    cfg = load_config(args.config)
    report = build_report(cfg, skip_raw_frames=args.skip_raw_frames)
    out = write_report(cfg, report)
    text = format_report(report)
    print(text)
    print(f"[report] JSON -> {out}")
    print(f"[report] text -> {manifests_dir(cfg) / 'final_report.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
