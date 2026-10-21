#!/bin/bash
#SBATCH --job-name=egoexo4d_dl
#SBATCH --partition=main,dmb
#SBATCH --gres=gpu:0
#SBATCH -c 8
#SBATCH --mem=16gb
#SBATCH --mail-type=END,FAIL,BEGIN
#SBATCH --time=48:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/egoexo4d_dl_%A_%a.out
#SBATCH --error=logs/egoexo4d_dl_%A_%a.err

# Download Relations media for every official scenario (shared EGOEXO_ROOT).
# Task 0 = Cooking (mostly a no-op if that dump is already complete).
#
#   sbatch --export=ALL,EGOEXO_ROOT=/deepstore/datasets/dmb/ComputerVision/information_retrieval/egotogaze/egoexo4d_raw \
#     run_download_all_scenarios.sbatch
#
# One scenario:  sbatch --array=2 --export=ALL,EGOEXO_ROOT=... run_download_all_scenarios.sbatch
# Missing VRS:   sbatch --export=ALL,EGOEXO_ROOT=...,DOWNLOAD_ARGS=--vrs-only ...

hostname
module load nvidia/cuda-10.1 2>/dev/null || true

export HTTP_PROXY="${HTTP_PROXY:-http://proxy.utwente.nl:3128}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://proxy.utwente.nl:3128}"
export http_proxy="${http_proxy:-$HTTP_PROXY}"
export https_proxy="${https_proxy:-$HTTPS_PROXY}"
export PYTHONUNBUFFERED=1

mkdir -p logs

EGOEXO_ROOT="${EGOEXO_ROOT:-/deepstore/datasets/dmb/ComputerVision/information_retrieval/egotogaze/egoexo4d_raw}"
export EGOEXO_ROOT
DOWNLOAD_ARGS="${DOWNLOAD_ARGS:-}"

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
echo "DOWNLOAD_ARGS=${DOWNLOAD_ARGS:-<scenario UIDs>}"
echo "AWS_PROFILE=${AWS_PROFILE:-<default>}"

cd "$SLURM_SUBMIT_DIR"
# shellcheck disable=SC2086
./scripts/01_download.sh ${DOWNLOAD_ARGS}

echo "Download job complete for $CONFIG"
