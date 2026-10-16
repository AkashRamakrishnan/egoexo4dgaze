#!/usr/bin/env bash
# Index + Tier-A fuse (cap 40 labeled images) + ego/exo QA plots.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG="${CONFIG:-configs/cooking_sample40.yaml}"
TOTAL="${TOTAL:-40}"

echo "[sample40] config=$CONFIG total=$TOTAL EGOEXO_ROOT=${EGOEXO_ROOT:-<yaml>}"
python scripts/verify_download.py --config "$CONFIG" --min-selected 1
python scripts/02_build_index.py --config "$CONFIG"
python scripts/03_build_dataset.py --config "$CONFIG" --skip-index --skip-qa
python scripts/06_visualize_ego_exo.py --config "$CONFIG" --total "$TOTAL"

echo "[sample40] done. Plots under output_root/viz_ego_exo/"
