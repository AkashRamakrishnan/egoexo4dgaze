# egoexo4d

Build a **Cooking-only**, **exocentric**, **Tier-A** (gaze inside Relations mask) dataset compatible with [GazeDETR](../GazeDETR)'s GazeFollow CSV contract.

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

### 1. Download Ego-Exo4D (Cooking Relations + takes)

```bash
EGOEXO_ROOT=./_downloads/egoexo4d ./scripts/01_download.sh
python scripts/verify_download.py   # optional sanity check
```

Cluster:

```bash
export EGOEXO_ROOT=/path/with/disk/egoexo4d
sbatch run_download_egoexo4d.sbatch
```

For a single-take smoke test after the main download:

```bash
COOKING_UID=<take_uid> EGOEXO_ROOT=./_downloads/egoexo4d ./scripts/01_download.sh --smoke
```

Edit `egoexo_root` in `configs/cooking_mvp.yaml` if your download root differs.

### 2–3. Index + build dataset

```bash
./scripts/run_after_download.sh
```

Or smoke (one take, output under `data_smoke/`):

```bash
CONFIG=configs/cooking_mvp_smoke.yaml ./scripts/run_after_download.sh
```

Cluster build (GPU for YOLO head boxes):

```bash
sbatch run_build_egoexo4d.sbatch
```

`run_after_download.sh` runs index + Tier-A fusion + export + splits + vocab + RoBERTa embeddings.

Manual steps equivalent to the script:

```bash
python scripts/02_build_index.py
python scripts/03_build_dataset.py
./scripts/05_precompute_embeddings.sh
```

Outputs:

| Path | Description |
|------|-------------|
| `data/images/` | Exocentric JPEG frames |
| `data/annotations/train_verified.csv` | GazeFollow-style rows |
| `data/annotations/{train,val,test}_split.csv` | Image-level splits |
| `data/vocab2id.json` | Label vocabulary |
| `data/manifests/build_stats.json` | Pipeline statistics |

### 4. RoBERTa vocab embeddings (for GazeDETR retrieval)

```bash
python ../GazeDETR/scripts/precompute_vocab_roberta_embeddings.py \
  --vocab_path ./data/vocab2id.json \
  --output ./data/label-embeds/vocab_roberta_embeddings.pt
```

### 5. Visual QA

```bash
python scripts/04_visualize.py --total 40
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

1. Filter **Cooking** takes (`parent_task_id: 1000`) present in Relations train/val JSON
2. Index exocentric Relations masks (`cam01`–`cam05`, ~1 fps frame keys)
3. Project MPS eye gaze to exo via `projectaria_tools`
4. **Tier-A**: keep rows where gaze pixel falls inside decoded mask
5. Estimate head boxes with YOLO person detections (upper-body crop)
6. Export frames + GazeFollow CSV + image-level splits + vocab

## Config

See [`configs/cooking_mvp.yaml`](configs/cooking_mvp.yaml). Key knobs:

- `filters.drop_hand_tracks` — exclude `*hand*` tracks
- `filters.min_label_count` — drop rare labels before CSV export
- `build.max_takes` — limit takes for smoke tests (`1`)
- `build.skip_head_detection` — skip YOLO (uses placeholder head boxes)

## Target scale

Cooking MVP Tier-A: **~15–25k rows**, **~8–15k** unique exo frames (depends on download coverage and MPS availability).
