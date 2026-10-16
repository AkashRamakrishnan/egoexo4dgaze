#!/usr/bin/env python3
"""Print Cooking take UIDs (Relations intersection when annotations are present)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from egoexo4d.index_takes import cooking_uids_for_download
from egoexo4d.paths import load_config, relations_path, takes_json_path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    cfg = load_config(args.config)

    takes_path = takes_json_path(cfg)
    if not takes_path.exists():
        print(f"MISSING takes.json: {takes_path}", file=sys.stderr)
        return 1

    uids = cooking_uids_for_download(cfg)
    rel_train = relations_path(cfg, "train")
    rel_val = relations_path(cfg, "val")
    print(
        f"[cooking_uids] {len(uids)} UIDs "
        f"(takes.json={takes_path.exists()} "
        f"rel_train={rel_train.exists()} rel_val={rel_val.exists()})",
        file=sys.stderr,
    )
    text = "\n".join(uids) + ("\n" if uids else "")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    else:
        sys.stdout.write(text)
    return 0 if uids else 1


if __name__ == "__main__":
    raise SystemExit(main())
