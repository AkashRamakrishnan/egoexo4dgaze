#!/usr/bin/env bash
# Index, build GazeDETR CSVs, optional smoke limits via CONFIG=configs/cooking_mvp_smoke.yaml
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG="${CONFIG:-configs/cooking_mvp.yaml}"

echo "[pipeline] config=$CONFIG"
python scripts/verify_download.py --config "$CONFIG"

python scripts/02_build_index.py --config "$CONFIG"
python scripts/03_build_dataset.py --config "$CONFIG"

./scripts/05_precompute_embeddings.sh

echo "[pipeline] done. Visual QA: python scripts/04_visualize.py --total 40 --config $CONFIG"
