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
    args = p.parse_args()
    cfg = load_config(args.config)
    root = egoexo_root(cfg)

    ok = True
    required_top = [
        takes_json_path(cfg),
        relations_path(cfg, "train"),
        relations_path(cfg, "val"),
    ]
    print(f"Ego-Exo root: {root}")
    for path in required_top:
        if path.exists():
            print(f"  OK  {path.relative_to(root)}")
        else:
            print(f"  MISSING  {path}")
            ok = False

    if not takes_json_path(cfg).exists():
        print(
            "\nRun:  EGOEXO_ROOT=<root> ./scripts/01_download.sh\n"
            "Ensure AWS is configured (aws configure or AWS_PROFILE)."
        )
        return 1

    manifest = build_cooking_takes_manifest(cfg)
    payload = json.loads(manifest.read_text())
    stats = payload.get("stats", {})
    n = len(payload.get("takes", []))
    print(f"\nCooking manifest: {manifest}")
    print(f"  selected takes: {n}")
    print(f"  stats: {json.dumps(stats, indent=2)}")

    if n == 0:
        print(
            "\nNo takes passed filters (need Relations + MPS gaze + GoPro calib + exo cam*.mp4).\n"
            "Re-run download with take media parts, e.g.:\n"
            "  egoexo -o <root> --release v2 --parts annotations metadata takes "
            "captures take_eye_gaze take_trajectory --benchmarks relations "
            "--splits train val -y"
        )
        ok = False

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
