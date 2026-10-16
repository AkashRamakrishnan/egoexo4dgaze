#!/usr/bin/env bash
# Quick checks before 01_download.sh (same error as "could not get manifests for all parts").
set -euo pipefail

RELEASE="${EGOEXO_RELEASE:-v2}"
MANIFEST="s3://ego4d-consortium-sharing/egoexo-public/${RELEASE}/metadata/manifest.json"

echo "=== Ego-Exo4D download diagnostics ==="
echo "Release: ${RELEASE}"
echo "Test object: ${MANIFEST}"
echo

if ! command -v egoexo >/dev/null 2>&1; then
  echo "FAIL: egoexo not found.  pip install 'ego4d>=1.7.1'"
  exit 1
fi
echo "OK  egoexo: $(command -v egoexo)"
pip show ego4d 2>/dev/null | awk '/^Version:/{print "    ego4d", $0}' || true

if command -v aws >/dev/null 2>&1; then
  echo
  echo "AWS identity:"
  if aws sts get-caller-identity 2>/dev/null; then
    echo "OK  aws CLI sees credentials (profile=${AWS_PROFILE:-default})"
  else
    echo "FAIL  aws sts get-caller-identity — run: aws configure"
    echo "      Keys must be from Ego-Exo4D license approval (~/.aws/credentials [default])"
  fi
  echo
  echo "S3 manifest probe:"
  if aws s3 ls "s3://ego4d-consortium-sharing/egoexo-public/${RELEASE}/metadata/" 2>&1; then
    echo "OK  can list metadata prefix"
  else
    echo "FAIL  cannot list egoexo-public bucket (403/InvalidAccessKeyId = wrong or expired keys,"
    echo "      or Ego-Exo4D license not activated — renew at ego4d.dev, not only Ego4D)"
  fi
else
  echo
  echo "WARN  aws CLI not installed (pip install awscli). egoexo uses boto3 from ~/.aws/credentials."
  if [[ -f "${HOME}/.aws/credentials" ]]; then
    echo "OK  ${HOME}/.aws/credentials exists"
  else
    echo "FAIL  missing ${HOME}/.aws/credentials — run aws configure"
  fi
fi

echo
echo "Python S3 exists() check (same as egoexo downloader):"
python <<PY
import os
import sys
try:
    from iopath.common.file_io import PathManager
    from iopath.common.s3 import S3PathHandler
except ImportError as e:
    print("SKIP", e)
    sys.exit(0)
profile = os.environ.get("AWS_PROFILE") or None
pm = PathManager()
pm.register_handler(S3PathHandler(profile=profile))
path = "${MANIFEST}"
try:
    ok = pm.exists(path)
    print("manifest exists:" if ok else "manifest MISSING (no S3 access or wrong release):", path)
except Exception as e:
    print("ERROR", type(e).__name__, e)
PY

echo
echo "If manifest is MISSING, fix AWS/Ego-Exo4D access before re-running ./scripts/01_download.sh"
echo "Docs: https://docs.ego-exo4d-data.org/download/#setup-aws-client"

# Propagate failure for 01_download.sh (set -e friendly).
python <<PY
import os, sys
from iopath.common.file_io import PathManager
from iopath.common.s3 import S3PathHandler
profile = os.environ.get("AWS_PROFILE") or None
pm = PathManager()
pm.register_handler(S3PathHandler(profile=profile))
path = "${MANIFEST}"
sys.exit(0 if pm.exists(path) else 1)
PY
