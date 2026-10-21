#!/usr/bin/env python3
"""Build scenario take manifest and Relations frame index."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from egoexo4d.index_relations import build_frame_index
from egoexo4d.index_takes import build_cooking_takes_manifest
from egoexo4d.paths import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    build_cooking_takes_manifest(cfg)
    build_frame_index(cfg)


if __name__ == "__main__":
    main()
