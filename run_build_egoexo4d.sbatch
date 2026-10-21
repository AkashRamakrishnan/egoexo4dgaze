#!/bin/bash
#SBATCH --job-name=egoexo4d_build
#SBATCH --partition=main,dmb
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64gb
#SBATCH --mail-type=END,FAIL,BEGIN
#SBATCH --time=24:00:00
#SBATCH --output=logs/egoexo4d_build_%j.out
#SBATCH --error=logs/egoexo4d_build_%j.err

# Build one scenario's CSV + frames after download (Tier-A fusion, YOLO heads, vocab).
# All six scenarios:  sbatch run_build_all_scenarios.sbatch
# Cooking rare labels: re-run this job after min_label_count: 1 (existing JPEGs skipped).
#
# Requires: completed 01_download.sh (verify_download.py passes).
# Optional smoke:  sbatch --export=ALL,CONFIG=configs/cooking_mvp_smoke.yaml run_build_egoexo4d.sbatch
#
# Submit:
#   sbatch run_build_egoexo4d.sbatch

hostname
module load nvidia/cuda-10.1

export HTTP_PROXY="${HTTP_PROXY:-http://proxy.utwente.nl:3128}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://proxy.utwente.nl:3128}"
export http_proxy="${http_proxy:-$HTTP_PROXY}"
export https_proxy="${https_proxy:-$HTTPS_PROXY}"
export PYTHONUNBUFFERED=1

mkdir -p logs

cd "$SLURM_SUBMIT_DIR"
CONFIG="${CONFIG:-configs/cooking_mvp.yaml}"
export CONFIG

pip install -e ".[gaze]" -q 2>/dev/null || pip install -e . -q

./scripts/run_after_download.sh

./scripts/05_precompute_embeddings.sh

python scripts/04_visualize.py --total 40 --config "$CONFIG" || true

# Refresh report so viz_* dirs are included in extracted disk stats.
python scripts/07_final_report.py --config "$CONFIG" --skip-raw-frames

echo "Build job complete."
