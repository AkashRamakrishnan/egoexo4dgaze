#!/bin/bash
#SBATCH --job-name=egoexo4d_sample40
#SBATCH --partition=main,dmb
#SBATCH --gres=gpu:0
#SBATCH -c 8
#SBATCH --mem=32gb
#SBATCH --mail-type=END,FAIL,BEGIN
#SBATCH --time=06:00:00
#SBATCH --output=logs/egoexo4d_sample40_%j.out
#SBATCH --error=logs/egoexo4d_sample40_%j.err

# Verify the Cooking pipeline on 40 random images that receive a Tier-A label.
# No GPU (YOLO heads skipped). Requires a finished download dump.
#
#   sbatch run_sample40_egoexo4d.sbatch
#
# Uses complete Cooking takes already on disk (does not download).
#
#   sbatch --export=ALL,EGOEXO_ROOT=/deepstore/datasets/dmb/ComputerVision/information_retrieval/egotogaze/egoexo4d_raw \
#     run_sample40_egoexo4d.sbatch

hostname

export HTTP_PROXY="${HTTP_PROXY:-http://proxy.utwente.nl:3128}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://proxy.utwente.nl:3128}"
export http_proxy="${http_proxy:-$HTTP_PROXY}"
export https_proxy="${https_proxy:-$HTTPS_PROXY}"
export PYTHONUNBUFFERED=1

mkdir -p logs
cd "$SLURM_SUBMIT_DIR"

CONFIG="${CONFIG:-configs/cooking_sample40.yaml}"
EGOEXO_ROOT="${EGOEXO_ROOT:-/deepstore/datasets/dmb/ComputerVision/information_retrieval/egotogaze/egoexo4d_raw}"
export CONFIG
export EGOEXO_ROOT

echo "CONFIG=$CONFIG"
echo "EGOEXO_ROOT=$EGOEXO_ROOT"
echo "EGOEXO_OUTPUT_ROOT=${EGOEXO_OUTPUT_ROOT:-<from yaml>}"

pip install -e ".[gaze]" -q 2>/dev/null || pip install -e . -q

./scripts/run_sample40.sh

echo "Sample40 job complete."
