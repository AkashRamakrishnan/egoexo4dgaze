#!/usr/bin/env bash
# Index, build GazeDETR CSVs. CONFIG selects the scenario (default: Cooking).
# Cooking with min_label_count=1 re-fuses rare labels; existing JPEGs are skipped.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG="${CONFIG:-configs/cooking_mvp.yaml}"

echo "[pipeline] config=$CONFIG"
python scripts/verify_download.py --config "$CONFIG"

python scripts/02_build_index.py --config "$CONFIG"
python scripts/03_build_dataset.py --config "$CONFIG"

./scripts/05_precompute_embeddings.sh

python scripts/07_final_report.py --config "$CONFIG"

REPORT="$(python - <<PY
from pathlib import Path
import sys
sys.path.insert(0, "${ROOT}")
from egoexo4d.paths import load_config, manifests_dir
cfg = load_config("${CONFIG}" if "${CONFIG}".startswith("/") else "${ROOT}/${CONFIG}")
print(manifests_dir(cfg) / "final_report.txt")
PY
)"

echo "[pipeline] done. Visual QA: python scripts/04_visualize.py --total 40 --config $CONFIG"
echo "[pipeline] report: ${REPORT}"
