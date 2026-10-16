#!/usr/bin/env bash
# Download Ego-Exo4D metadata, annotations, and take media for Cooking MVP.
#
# Prerequisites:
#   pip install 'ego4d>=1.7.1' awscli
#   aws configure          # access key from Ego-Exo4D license approval
#   # optional: export AWS_PROFILE=egoexo
#
# Usage:
#   EGOEXO_ROOT=./_downloads/egoexo4d ./scripts/01_download.sh
#   COOKING_UID=<uid> EGOEXO_ROOT=... ./scripts/01_download.sh --smoke
#
# After download:
#   ./scripts/run_after_download.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EGOEXO_ROOT="${EGOEXO_ROOT:-${ROOT}/_downloads/egoexo4d}"
RELEASE="${EGOEXO_RELEASE:-v2}"
S3_PROFILE="${AWS_PROFILE:-${S3_PROFILE:-}}"
SMOKE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) SMOKE=1; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

mkdir -p "${EGOEXO_ROOT}"

if [[ "${SKIP_AWS_DIAG:-0}" != 1 ]]; then
  "${ROOT}/scripts/00_diagnose_aws.sh" || {
    echo "[01_download] AWS/S3 check failed. Fix credentials, then retry."
    echo "  ./scripts/00_diagnose_aws.sh"
    exit 1
  }
fi

if ! command -v egoexo >/dev/null 2>&1; then
  echo "Missing egoexo CLI. Install: pip install 'ego4d>=1.7.1'"
  exit 1
fi

EGOEXO_ARGS=( -o "${EGOEXO_ROOT}" --release "${RELEASE}" -y )
if [[ -n "${S3_PROFILE}" ]]; then
  EGOEXO_ARGS+=( --s3_profile "${S3_PROFILE}" )
fi

echo "[01_download] output=${EGOEXO_ROOT} release=${RELEASE} profile=${S3_PROFILE:-default}"

# Relations benchmark needs take-linked parts (not annotations alone).
egoexo "${EGOEXO_ARGS[@]}" \
  --parts annotations metadata takes captures take_eye_gaze take_trajectory \
  --benchmarks relations \
  --splits train val

if [[ "${SMOKE}" -eq 1 ]]; then
  COOKING_UID="${COOKING_UID:-}"
  if [[ -z "${COOKING_UID}" ]]; then
    echo "[01_download] smoke: set COOKING_UID from ${EGOEXO_ROOT}/takes.json for an extra take"
  else
    echo "[01_download] smoke: fetching take ${COOKING_UID}"
    egoexo "${EGOEXO_ARGS[@]}" \
      --uids "${COOKING_UID}" \
      --parts takes captures take_eye_gaze take_trajectory
  fi
fi

echo "[01_download] verifying layout..."
python "${ROOT}/scripts/verify_download.py" \
  --config "${ROOT}/configs/cooking_mvp.yaml" || {
  echo "[01_download] verify failed — check AWS credentials and disk space."
  exit 1
}

echo "[01_download] done. Next: ./scripts/run_after_download.sh"
