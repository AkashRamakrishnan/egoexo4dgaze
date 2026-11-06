#!/bin/bash
#SBATCH --job-name=egoexo4d_viz_paper
#SBATCH --partition=main,dmb
#SBATCH --gres=gpu:0
#SBATCH -c 4
#SBATCH --mem=32gb
#SBATCH --mail-type=END,FAIL,BEGIN
#SBATCH --time=08:00:00
#SBATCH --output=logs/egoexo4d_viz_paper_%j.out
#SBATCH --error=logs/egoexo4d_viz_paper_%j.err

# Paper-style ego|exo samples: no captions, matched height, object mask + bbox
# + label. For each take: PER_BUCKET gaze-in-mask and PER_BUCKET gaze-out-of-mask.
# Needs take mp4s, MPS, and manifests/frame_index.parquet.
#
#   sbatch --export=ALL,EGOEXO_ROOT=/deepstore/datasets/dmb/ComputerVision/information_retrieval/egotogaze/egoexo4d_raw \
#     run_viz_ego_exo_paper.sbatch
#
# Limit takes / change counts:
#   sbatch --export=ALL,EGOEXO_ROOT=...,TAKES=12,PER_BUCKET=10,OUT=./viz_ego_exo_paper \
#     run_viz_ego_exo_paper.sbatch

hostname

export HTTP_PROXY="${HTTP_PROXY:-http://proxy.utwente.nl:3128}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://proxy.utwente.nl:3128}"
export http_proxy="${http_proxy:-$HTTP_PROXY}"
export https_proxy="${https_proxy:-$HTTPS_PROXY}"
export PYTHONUNBUFFERED=1

mkdir -p logs
cd "$SLURM_SUBMIT_DIR"

CONFIG="${CONFIG:-configs/cooking_mvp.yaml}"
EGOEXO_ROOT="${EGOEXO_ROOT:-/deepstore/datasets/dmb/ComputerVision/information_retrieval/egotogaze/egoexo4d_raw}"
PER_BUCKET="${PER_BUCKET:-10}"
OUT="${OUT:-./viz_ego_exo_paper}"
export CONFIG
export EGOEXO_ROOT

echo "CONFIG=$CONFIG"
echo "EGOEXO_ROOT=$EGOEXO_ROOT"
echo "EGOEXO_OUTPUT_ROOT=${EGOEXO_OUTPUT_ROOT:-<from yaml>}"
echo "TAKES=${TAKES:-all} PER_BUCKET=$PER_BUCKET OUT=$OUT"

pip install -e ".[gaze]" -q 2>/dev/null || pip install -e . -q

ARGS=(
  --config "$CONFIG"
  --paper
  --per-bucket "$PER_BUCKET"
  --out "$OUT"
)
if [[ -n "${TAKES:-}" ]]; then
  ARGS+=(--takes "$TAKES")
fi

python scripts/06_visualize_ego_exo.py "${ARGS[@]}"

echo "Paper ego/exo viz complete -> $OUT"
