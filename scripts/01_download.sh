#!/usr/bin/env bash
# Download Ego-Exo4D Relations takes for one scenario (default: Cooking).
#
# Phase 1: metadata + annotations (small; shared across scenarios)
# Phase 2: videos/gaze/trajectory only for that scenario's UIDs
#
#   EGOEXO_ROOT=./_downloads/egoexo4d ./scripts/01_download.sh
#   CONFIG=configs/health.yaml ./scripts/01_download.sh
#   ./scripts/01_download.sh --scenario bike_repair
#   ./scripts/01_download.sh --metadata-only    # annotations + takes.json only
#   ./scripts/01_download.sh --vrs-only         # aria01_noimagestreams.vrs
#   ./scripts/01_download.sh --all-relations    # every Relations take
#   ./scripts/01_download.sh --smoke            # one take of media
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
SCENARIO="${SCENARIO:-}"
CONFIG="${CONFIG:-}"
# takes = frame-aligned mp4 only. CPF / RGB factory calib is in take_vrs_noimagestream
# (aria01_noimagestreams.vrs). Do not add take_vrs (full Aria video, ~12TB).
MEDIA_PARTS=(takes captures take_eye_gaze take_trajectory take_vrs_noimagestream)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) SMOKE=1; shift ;;
    --metadata-only) METADATA_ONLY=1; shift ;;
    --all-relations) ALL_RELATIONS=1; shift ;;
    --vrs-only) VRS_ONLY=1; shift ;;
    --scenario) SCENARIO="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ "${VRS_ONLY}" -eq 1 ]]; then
  MEDIA_PARTS=(take_vrs_noimagestream)
fi

if [[ -n "${SCENARIO}" && -z "${CONFIG}" ]]; then
  export SCENARIO
  CONFIG="$(PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}" python - <<'PY'
from egoexo4d.scenarios import SCENARIOS
import os
slug = os.environ["SCENARIO"]
if slug not in SCENARIOS:
    raise SystemExit(f"Unknown scenario {slug!r}; expected {tuple(SCENARIOS)}")
print(SCENARIOS[slug]["config"])
PY
)"
fi
CONFIG="${CONFIG:-${ROOT}/configs/cooking_mvp.yaml}"
if [[ "${CONFIG}" != /* ]]; then
  CONFIG="${ROOT}/${CONFIG}"
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
echo "[01_download] config=${CONFIG}"

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
  python "${ROOT}/scripts/list_scenario_uids.py" --config "${CONFIG}" >/dev/null
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
  python "${ROOT}/scripts/list_scenario_uids.py" \
    --config "${CONFIG}" \
    --out "${UID_FILE}"
  mapfile -t SCENARIO_UIDS < "${UID_FILE}"
  echo "[01_download] phase 2/2: scenario media (${MEDIA_PARTS[*]}) for ${#SCENARIO_UIDS[@]} takes"
  if [[ ${#SCENARIO_UIDS[@]} -eq 0 ]]; then
    echo "[01_download] no scenario UIDs in takes.json / Relations. Aborting media download."
    exit 1
  fi

  if [[ "${SMOKE}" -eq 1 ]]; then
    SMOKE_UID="${COOKING_UID:-${SCENARIO_UID:-${SCENARIO_UIDS[0]:-}}}"
    if [[ -z "${SMOKE_UID}" ]]; then
      echo "[01_download] smoke: no UID found"
      exit 1
    fi
    echo "[01_download] smoke: fetching take ${SMOKE_UID}"
    download_uids "${SMOKE_UID}"
  else
    download_uids "${SCENARIO_UIDS[@]}"
  fi
fi

echo "[01_download] verifying layout..."
python "${ROOT}/scripts/verify_download.py" \
  --config "${CONFIG}" || {
  echo "[01_download] verify failed — check AWS credentials and disk space."
  exit 1
}

echo "[01_download] done. Next: CONFIG=${CONFIG} ./scripts/run_after_download.sh"
