#!/bin/bash
#SBATCH --job-name=egoexo4d_build
#SBATCH --partition=main,dmb
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64gb
#SBATCH --mail-type=END,FAIL,BEGIN
#SBATCH --time=24:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/egoexo4d_build_%A_%a.out
#SBATCH --error=logs/egoexo4d_build_%A_%a.err

# Build GazeFollow CSVs + frames for every Relations scenario.
# Task 0 = Cooking: re-fuses with min_label_count=1 so previously pruned
# rare labels are written back (existing JPEGs are skipped).
#
# After downloads finish:
#   sbatch --export=ALL,EGOEXO_ROOT=/deepstore/datasets/dmb/ComputerVision/information_retrieval/egotogaze/egoexo4d_raw \
#     run_build_all_scenarios.sbatch
#
# Cooking only (recover rare labels):
#   sbatch --array=0 --export=ALL,EGOEXO_ROOT=... run_build_all_scenarios.sbatch
#
# Do not pip-install here: ego4d extras previously broke numpy/sklearn in vlamma.

hostname
module load nvidia/cuda-10.1

export HTTP_PROXY="${HTTP_PROXY:-http://proxy.utwente.nl:3128}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://proxy.utwente.nl:3128}"
export http_proxy="${http_proxy:-$HTTP_PROXY}"
export https_proxy="${https_proxy:-$HTTPS_PROXY}"
export PYTHONUNBUFFERED=1

mkdir -p logs

EGOEXO_ROOT="${EGOEXO_ROOT:-/deepstore/datasets/dmb/ComputerVision/information_retrieval/egotogaze/egoexo4d_raw}"
export EGOEXO_ROOT

CONFIGS=(
  configs/cooking_mvp.yaml
  configs/health.yaml
  configs/bike_repair.yaml
  configs/music.yaml
  configs/basketball.yaml
  configs/soccer.yaml
)
CONFIG="${CONFIG:-${CONFIGS[$SLURM_ARRAY_TASK_ID]}}"
export CONFIG

echo "EGOEXO_ROOT=$EGOEXO_ROOT"
echo "CONFIG=$CONFIG"
echo "ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"

cd "$SLURM_SUBMIT_DIR"

./scripts/run_after_download.sh

python scripts/04_visualize.py --total 40 --config "$CONFIG" || true

python scripts/07_final_report.py --config "$CONFIG" --skip-raw-frames

echo "Build job complete for $CONFIG"
