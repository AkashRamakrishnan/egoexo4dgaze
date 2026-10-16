"""Build vocab2id.json from exported label_text values."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict

from egoexo4d.paths import output_root


def build_vocab_from_csv(csv_path: Path, out_path: Path | None = None) -> Path:
    with csv_path.open(newline="") as f:
        labels = sorted({row["label_text"] for row in csv.DictReader(f)})
    vocab: Dict[str, int] = {label: i for i, label in enumerate(labels)}
    out = out_path or csv_path.parent.parent / "vocab2id.json"
    with out.open("w") as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)
    print(f"[vocab] {len(vocab)} labels -> {out}")
    return out


def run_vocab(cfg: dict) -> Path:
    csv_path = output_root(cfg) / "annotations" / "train_verified.csv"
    out_path = output_root(cfg) / "vocab2id.json"
    return build_vocab_from_csv(csv_path, out_path)
