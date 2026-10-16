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

# Download Relations Cooking subset (train+val) to cluster storage.
#
# Before submit:
#   aws configure   # on the login node, or export AWS_PROFILE
#   Set EGOEXO_ROOT to a path with enough disk (see README scale estimates).
#
# Submit from repo root:
#   sbatch run_download_egoexo4d.sbatch

hostname
module load nvidia/cuda-10.1 2>/dev/null || true

export HTTP_PROXY="${HTTP_PROXY:-http://proxy.utwente.nl:3128}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://proxy.utwente.nl:3128}"
export http_proxy="${http_proxy:-$HTTP_PROXY}"
export https_proxy="${https_proxy:-$HTTPS_PROXY}"
export PYTHONUNBUFFERED=1

mkdir -p logs

# Large download — override on submit:  sbatch --export=ALL,EGOEXO_ROOT=/data/... run_download_egoexo4d.sbatch
EGOEXO_ROOT="${EGOEXO_ROOT:-$PWD/_downloads/egoexo4d}"
export EGOEXO_ROOT

echo "EGOEXO_ROOT=$EGOEXO_ROOT"
echo "AWS_PROFILE=${AWS_PROFILE:-<default>}"

pip install -q 'ego4d>=1.7.1' awscli 2>/dev/null || true

cd "$SLURM_SUBMIT_DIR"
./scripts/01_download.sh

echo "Download job complete."
