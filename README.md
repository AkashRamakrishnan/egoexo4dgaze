# egoexo4d

Build **exocentric**, **Tier-A** (gaze inside Relations mask) datasets for every Ego-Exo4D Relations scenario, compatible with [GazeDETR](../GazeDETR)'s GazeFollow CSV contract.

Scenarios: **Cooking**, **Health**, **Bike Repair**, **Music**, **Basketball**, **Soccer**. Cooking is the default config; the others share the same raw dump and write to `data_<slug>/`.

**Labels:** every object label is kept (`filters.min_label_count: 1`). An earlier Cooking build that dropped labels with fewer than 5 rows cannot recover those labels from the pruned CSV — re-run the Cooking **build** (not download). Fusion re-reads Relations + gaze; existing JPEGs are skipped.

## Prerequisites

```bash
cd egoexo4d
pip install -e ".[gaze]"
pip install 'ego4d>=1.7.1' awscli   # provides the egoexo downloader CLI
aws configure                     # keys from Ego-Exo4D license approval email
# optional: export AWS_PROFILE=your-profile
```

Use **Python 3.10–3.11** for `projectaria-tools` (gaze projection). The downloader can run under 3.12+.

Optional: `yolov8n.pt` is fetched automatically by Ultralytics on first head-detection run.

## Quick start

### 1. Download Ego-Exo4D (Relations + takes)

Shared dump (`EGOEXO_ROOT`). Relations train/val JSON is downloaded once; media is fetched per scenario UID list (resume-safe).

```bash
EGOEXO_ROOT=./_downloads/egoexo4d ./scripts/01_download.sh
python scripts/verify_download.py   # optional sanity check
```

All six scenarios on the cluster (Cooking is mostly a no-op if that dump is already complete):

```bash
export EGOEXO_ROOT=/path/with/disk/egoexo4d
sbatch run_download_all_scenarios.sbatch
```

One scenario:

```bash
CONFIG=configs/health.yaml ./scripts/01_download.sh
# or
./scripts/01_download.sh --scenario bike_repair
```

Gaze projection needs Aria VRS factory calib (`aria01_noimagestreams.vrs`). That is **`--parts take_vrs_noimagestream`**, not `--parts takes` (MP4s only). If videos/gaze/trajectory are already on disk:

```bash
sbatch --export=ALL,DOWNLOAD_ARGS=--vrs-only run_download_all_scenarios.sbatch
```

Edit `egoexo_root` in the scenario YAML if your download root differs.

### 2–3. Index + build dataset

Cooking (also recovers previously pruned rare labels):

```bash
./scripts/run_after_download.sh
```

All scenarios (task 0 = Cooking rebuild):

```bash
sbatch run_build_all_scenarios.sbatch
```

One other scenario:

```bash
CONFIG=configs/soccer.yaml ./scripts/run_after_download.sh
```

Smoke (one Cooking take, output under `data_smoke/`):

```bash
CONFIG=configs/cooking_mvp_smoke.yaml ./scripts/run_after_download.sh
```

`run_after_download.sh` runs index + Tier-A fusion + export + splits + vocab + RoBERTa embeddings + a final disk/frame report (`scripts/07_final_report.py`).

| Path | Description |
|------|-------------|
| `data/images/` (or `data_<slug>/images/`) | Exocentric JPEG frames |
| `…/annotations/train_verified.csv` | GazeFollow-style rows |
| `…/annotations/{train,val,test}_split.csv` | Image-level splits |
| `…/vocab2id.json` | Label vocabulary |
| `…/manifests/build_stats.json` | Pipeline statistics |
| `…/manifests/final_report.json` | Disk, frame funnel, CSV/split/vocab summary |
| `…/manifests/final_report.txt` | Human-readable copy of the same report |
| `combined_report.{txt,html,json}` | Per-scenario + overall tables (`scripts/08_combined_report.py`) |

### 4. RoBERTa vocab embeddings (for GazeDETR retrieval)

```bash
python ../GazeDETR/scripts/precompute_vocab_roberta_embeddings.py \
  --vocab_path ./data/vocab2id.json \
  --output ./data/label-embeds/vocab_roberta_embeddings.pt
```

Point `--vocab_path` at `data_<slug>/vocab2id.json` for non-Cooking builds.

### 5. Combined overview (all scenarios)

After the six builds finish:

```bash
python scripts/08_combined_report.py
```

Prints per-scenario and overall tables, and writes `combined_report.txt`, `combined_report.html`, and `combined_report.json`. Open the HTML file for a dashboard. Scenarios without `final_report.json` show as missing and are omitted from totals.

Rebuild each scenario report from CSVs first (optional):

```bash
python scripts/08_combined_report.py --rebuild
```

### 6. Visual QA

```bash
python scripts/04_visualize.py --total 40
CONFIG=configs/health.yaml python scripts/04_visualize.py --total 40 --config configs/health.yaml
```

## GazeDETR training

Use `GazeDETR/configs/mdetr_egoexo4d_sem_retrieval.yaml` with `data.dataset: egoexo4d`.

```bash
cd ../GazeDETR
python train_mdetr.py --config configs/mdetr_egoexo4d_sem_retrieval.yaml \
  --face_feats 6drepnet360 --scene_conditioning sight_conic --load_face_weights --no-loc_only
```

Cluster: `sbatch run_egoexo4d_retrieval.sbatch`

## Pipeline overview

1. Filter scenario takes (`parent_task_id` 1000/2000/4000/5000/6000/8000) present in Relations train/val JSON
2. Index exocentric Relations masks (`cam01`–`cam05`, ~1 fps frame keys)
3. Project MPS eye gaze to exo via `projectaria_tools`
4. **Tier-A**: keep rows where gaze pixel falls inside decoded mask
5. Estimate head boxes with YOLO person detections (upper-body crop)
6. Export frames + GazeFollow CSV + image-level splits + vocab (all labels)

## Config

See [`configs/cooking_mvp.yaml`](configs/cooking_mvp.yaml) and `configs/{health,bike_repair,music,basketball,soccer}.yaml`. Key knobs:

- `scenario.slug` / `parent_task_id` — which Relations subset
- `output_root` — extracted dataset (`./data` for Cooking, `./data_<slug>` otherwise)
- `filters.drop_hand_tracks` — exclude `*hand*` tracks
- `filters.min_label_count` — `1` keeps rare labels (set higher only if you want to drop them)
- `build.max_takes` — limit takes for smoke tests (`1`)
- `build.skip_head_detection` — skip YOLO (uses placeholder head boxes)

## Target scale

Cooking MVP Tier-A: **~15–25k rows**, **~8–15k** unique exo frames (depends on download coverage and MPS availability). Other scenarios vary with take count.
