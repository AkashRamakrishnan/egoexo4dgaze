# Ego-Exo4D download access status

**Last updated:** 2026-10-13  
**Blocker:** AWS credentials authenticate, but **Ego-Exo4D S3 data is not provisioned** for this account yet.

## Current state

| Item | Status |
|------|--------|
| Pipeline code | Ready (`01_download.sh` → `verify_download.py` → `run_after_download.sh`, cluster sbatch) |
| Local tools | `egoexo` (ego4d **1.7.3**), `aws` CLI, `~/.aws/credentials` |
| AWS STS | **OK** — `aws sts get-caller-identity` succeeds |
| Ego-Exo4D S3 | **Blocked** — cannot read manifests under `s3://ego4d-consortium-sharing/egoexo-public/v2/` |

## Diagnostic (2026-10-13)

Run: `./scripts/00_diagnose_aws.sh`

- **Identity:** `arn:aws:iam::215061535154:user/akash-ramakrishnan-edb6`
- **`aws s3 ls`** on `egoexo-public/v2/metadata/`: **AccessDenied** (`s3:ListBucket` not allowed on `ego4d-consortium-sharing`)
- **Python / iopath `exists()`** on `…/v2/metadata/manifest.json`: **MISSING** (same as `egoexo` downloader — no manifest → `could not get manifests for all parts`)

This is **not** a wrong `--parts` / `relations` benchmark issue; manifests never load when S3 denies access.

## User action in flight

- **New Ego-Exo4D credentials / license provisioning requested** — waiting on approval or CVDF policy attach (typical turnaround per docs: ~48h; keys may expire ~14 days after issue).

## When access is granted

1. `aws configure` with the **new** access key + secret (profile `default` or set `AWS_PROFILE`).
2. `./scripts/00_diagnose_aws.sh` — must pass (manifest reachable).
3. `./scripts/01_download.sh` — note printed **Fetching: … GiB** size.
4. `./scripts/run_after_download.sh` (or `CONFIG=configs/cooking_mvp_smoke.yaml` for one take).
5. Point `GazeDETR/configs/mdetr_egoexo4d_sem_retrieval.yaml` at built `data/` + RoBERTa embeddings.

## References

- Setup: https://docs.ego-exo4d-data.org/download/#setup-aws-client
- Ego-Exo4D license (separate from Ego4D): https://ego4d.dev/request/ego-exo-4d
- Support: info@ego4d-data.org (if AccessDenied persists after renewal)
