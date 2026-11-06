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
# + label. For each scenario: PER_BUCKET gaze-in-mask and PER_BUCKET gaze-out-of-mask
# (distinct takes when possible). Needs take mp4s, MPS, and each scenario's
# manifests/frame_index.parquet.
#
#   sbatch --export=ALL,EGOEXO_ROOT=/deepstore/datasets/dmb/ComputerVision/information_retrieval/egotogaze/egoexo4d_raw \
#     run_viz_ego_exo_paper.sbatch
#
# One scenario only:
#   sbatch --export=ALL,EGOEXO_ROOT=...,ALL_SCENARIOS=0,CONFIG=configs/cooking_mvp.yaml \
#     run_viz_ego_exo_paper.sbatch

hostname

export HTTP_PROXY="${HTTP_PROXY:-http://proxy.utwente.nl:3128}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://proxy.utwente.nl:3128}"
export http_proxy="${http_proxy:-$HTTP_PROXY}"
export https_proxy="${https_proxy:-$HTTPS_PROXY}"
export PYTHONUNBUFFERED=1

mkdir -p logs
cd "$SLURM_SUBMIT_DIR"

EGOEXO_ROOT="${EGOEXO_ROOT:-/deepstore/datasets/dmb/ComputerVision/information_retrieval/egotogaze/egoexo4d_raw}"
PER_BUCKET="${PER_BUCKET:-10}"
OUT="${OUT:-./viz_ego_exo_paper}"
ALL_SCENARIOS="${ALL_SCENARIOS:-1}"
export EGOEXO_ROOT

echo "EGOEXO_ROOT=$EGOEXO_ROOT"
echo "ALL_SCENARIOS=$ALL_SCENARIOS PER_BUCKET=$PER_BUCKET OUT=$OUT"

ARGS=(
  --paper
  --per-bucket "$PER_BUCKET"
  --out "$OUT"
)
if [[ "$ALL_SCENARIOS" == "1" ]]; then
  ARGS+=(--all-scenarios)
else
  ARGS+=(--config "${CONFIG:-configs/cooking_mvp.yaml}")
fi

python scripts/06_visualize_ego_exo.py "${ARGS[@]}"

echo "Paper ego/exo viz complete -> $OUT"
