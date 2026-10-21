#!/bin/bash
#SBATCH --job-name=egoexo4d_download
#SBATCH --partition=main,dmb
#SBATCH --gres=gpu:0
#SBATCH -c 8
#SBATCH --mem=16gb
#SBATCH --mail-type=END,FAIL,BEGIN
#SBATCH --time=48:00:00
#SBATCH --output=logs/egoexo4d_download_%j.out
#SBATCH --error=logs/egoexo4d_download_%j.err

# Download Relations takes for one scenario (default: Cooking).
# All six scenarios:  sbatch run_download_all_scenarios.sbatch
# Metadata+annotations first, then media for that scenario's UIDs. Resume-safe.
#
#   sbatch --export=ALL,EGOEXO_ROOT=/deepstore/datasets/dmb/ComputerVision/information_retrieval/egotogaze/egoexo4d_raw \
#     run_download_egoexo4d.sbatch
#
# Health (example):    sbatch --export=ALL,CONFIG=configs/health.yaml,EGOEXO_ROOT=... ...
# Full Relations:      sbatch --export=ALL,DOWNLOAD_ARGS=--all-relations ...
# Missing Aria VRS:    sbatch --export=ALL,DOWNLOAD_ARGS=--vrs-only ...

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
CONFIG="${CONFIG:-configs/cooking_mvp.yaml}"
export CONFIG
DOWNLOAD_ARGS="${DOWNLOAD_ARGS:-}"

echo "EGOEXO_ROOT=$EGOEXO_ROOT"
echo "CONFIG=$CONFIG"
echo "DOWNLOAD_ARGS=${DOWNLOAD_ARGS:-<scenario UIDs>}"
echo "AWS_PROFILE=${AWS_PROFILE:-<default>}"

pip install -q 'ego4d>=1.7.1' awscli 2>/dev/null || true

cd "$SLURM_SUBMIT_DIR"
# shellcheck disable=SC2086
./scripts/01_download.sh ${DOWNLOAD_ARGS}

echo "Download job complete."
