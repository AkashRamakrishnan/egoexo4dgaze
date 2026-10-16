#!/usr/bin/env bash
# Precompute RoBERTa vocab embeddings for GazeDETR retrieval training.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-configs/cooking_mvp.yaml}"

DATA="$(python - <<PY
from pathlib import Path
import sys
sys.path.insert(0, "${ROOT}")
from egoexo4d.paths import load_config, output_root
cfg = load_config("${ROOT}/${CONFIG}")
print(output_root(cfg))
PY
)"

VOCAB="${DATA}/vocab2id.json"
OUT="${DATA}/label-embeds/vocab_roberta_embeddings.pt"

if [[ ! -f "${VOCAB}" ]]; then
  echo "Missing vocab: ${VOCAB} (run 03_build_dataset.py first)"
  exit 1
fi

mkdir -p "$(dirname "${OUT}")"

python "${ROOT}/../GazeDETR/scripts/precompute_vocab_roberta_embeddings.py" \
  --vocab_path "${VOCAB}" \
  --output "${OUT}"

echo "Wrote ${OUT}"
