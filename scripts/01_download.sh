#!/usr/bin/env bash
# Download Ego-Exo4D **Cooking** takes (Relations train/val), not the full dump.
#
# Phase 1: metadata + annotations (small)
# Phase 2: videos/gaze/trajectory only for Cooking UIDs
#
#   EGOEXO_ROOT=./_downloads/egoexo4d ./scripts/01_download.sh
#   ./scripts/01_download.sh --metadata-only    # annotations + takes.json only
#   ./scripts/01_download.sh --vrs-only         # aria01_noimagestreams.vrs for cooking UIDs
#   ./scripts/01_download.sh --all-relations    # old behaviour: every Relations take
#   ./scripts/01_download.sh --smoke            # one cooking take of media
#
# Resume-safe: re-running skips files egoexo already wrote.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EGOEXO_ROOT="${EGOEXO_ROOT:-${ROOT}/_downloads/egoexo4d}"
RELEASE="${EGOEXO_RELEASE:-v2}"
S3_PROFILE="${AWS_PROFILE:-${S3_PROFILE:-}}"
SMOKE=0
METADATA_ONLY=0
ALL_RELATIONS=0
VRS_ONLY=0
UID_BATCH="${UID_BATCH:-40}"
# takes = frame-aligned mp4 only. CPF / RGB factory calib is in take_vrs_noimagestream
# (aria01_noimagestreams.vrs). Do not add take_vrs (full Aria video, ~12TB).
MEDIA_PARTS=(takes captures take_eye_gaze take_trajectory take_vrs_noimagestream)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) SMOKE=1; shift ;;
    --metadata-only) METADATA_ONLY=1; shift ;;
    --all-relations) ALL_RELATIONS=1; shift ;;
    --vrs-only) VRS_ONLY=1; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ "${VRS_ONLY}" -eq 1 ]]; then
  MEDIA_PARTS=(take_vrs_noimagestream)
fi

mkdir -p "${EGOEXO_ROOT}"
export EGOEXO_ROOT

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

if [[ "${VRS_ONLY}" -eq 1 ]]; then
  echo "[01_download] --vrs-only: skip metadata; fetching take_vrs_noimagestream"
else
  echo "[01_download] phase 1/2: annotations + metadata"
  egoexo "${EGOEXO_ARGS[@]}" \
    --parts annotations metadata \
    --benchmarks relations \
    --splits train val
fi

if [[ "${METADATA_ONLY}" -eq 1 ]]; then
  echo "[01_download] --metadata-only: skipping take media"
  python "${ROOT}/scripts/list_cooking_uids.py" --config "${ROOT}/configs/cooking_mvp.yaml" >/dev/null
  exit 0
fi

download_uids() {
  local -a uids=("$@")
  if [[ ${#uids[@]} -eq 0 ]]; then
    echo "[01_download] no UIDs in this batch"
    return 0
  fi
  local i
  for ((i = 0; i < ${#uids[@]}; i += UID_BATCH)); do
    local batch=("${uids[@]:i:UID_BATCH}")
    echo "[01_download] media batch $((i / UID_BATCH + 1)): ${#batch[@]} UIDs"
    egoexo "${EGOEXO_ARGS[@]}" \
      --uids "${batch[@]}" \
      --parts "${MEDIA_PARTS[@]}"
  done
}

if [[ "${ALL_RELATIONS}" -eq 1 ]]; then
  echo "[01_download] --all-relations: downloading every Relations train/val take (large)"
  egoexo "${EGOEXO_ARGS[@]}" \
    --parts "${MEDIA_PARTS[@]}" \
    --benchmarks relations \
    --splits train val
else
  UID_FILE="$(mktemp)"
  trap 'rm -f "${UID_FILE}"' EXIT
  python "${ROOT}/scripts/list_cooking_uids.py" \
    --config "${ROOT}/configs/cooking_mvp.yaml" \
    --out "${UID_FILE}"
  mapfile -t COOKING_UIDS < "${UID_FILE}"
  echo "[01_download] phase 2/2: Cooking media (${MEDIA_PARTS[*]}) for ${#COOKING_UIDS[@]} takes"
  if [[ ${#COOKING_UIDS[@]} -eq 0 ]]; then
    echo "[01_download] no Cooking UIDs in takes.json / Relations. Aborting media download."
    exit 1
  fi

  if [[ "${SMOKE}" -eq 1 ]]; then
    COOKING_UID="${COOKING_UID:-${COOKING_UIDS[0]:-}}"
    if [[ -z "${COOKING_UID}" ]]; then
      echo "[01_download] smoke: no cooking UID found"
      exit 1
    fi
    echo "[01_download] smoke: fetching take ${COOKING_UID}"
    download_uids "${COOKING_UID}"
  else
    download_uids "${COOKING_UIDS[@]}"
  fi
fi

echo "[01_download] verifying layout..."
python "${ROOT}/scripts/verify_download.py" \
  --config "${ROOT}/configs/cooking_mvp.yaml" || {
  echo "[01_download] verify failed — check AWS credentials and disk space."
  exit 1
}

echo "[01_download] done. Next: ./scripts/run_after_download.sh"
