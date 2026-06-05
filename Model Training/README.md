# Crack Segmentation Pipeline

YOLOv8-Seg transfer learning pipeline for drone-based crack detection.
Two modes: **smoke** (Mac M3, 300 images, ~3 min) and **full** (RunPod GPU, ~39k images).

## Quick Start

### Smoke Test (Mac M3)
```bash
# 1. Download datasets (requires Kaggle API key at ~/.kaggle/kaggle.json)
make download

# 2. Convert masks to YOLO-Seg polygons
make convert

# 3. Remove blurry/mislabelled images
make clean-data

# 4. Build 300-image subset
make subset

# 5. Run smoke train inside Docker
make smoke
```

### Full Training (RunPod)

1. Push image to registry or build on RunPod pod:
   ```bash
   make build-runpod
   docker tag crack-seg:runpod <your-registry>/crack-seg:runpod
   docker push <your-registry>/crack-seg:runpod
   ```
2. On RunPod: mount `/workspace/data` and `/workspace/runs` volumes, then:
   ```bash
   docker run --gpus all -e TRAIN_MODE=full \
     -v /path/to/data:/workspace/data \
     -v /path/to/runs:/workspace/runs \
     crack-seg:runpod
   ```

## Pipeline Stages

| Stage | Script | What it does |
|-------|--------|-------------|
| Download | `src/download_data.py` | Fetch CrackSeg9k (Kaggle) + OmniCrack30k (HuggingFace), validate manifest |
| Convert | `src/convert_annotations.py` | Binary mask PNG → YOLO-Seg polygon .txt, QC overlays to `data/yolo/_qc/` |
| Clean | `src/clean_data.py` | Remove blurry images, empty masks, detect orphan files, write cleaning report |
| Subset | `src/make_subset.py` | Stratified 300-image sample → train/val/test splits for smoke test |
| Train | `src/train.py` | Single entrypoint; reads `TRAIN_MODE` env var + config YAML |
| Validate | `src/validate.py` | Run `model.val()` on a split, print Mask mAP50 and mAP50-95 |
| Test | `src/test_predict.py` | Inference on test split, save overlays and `summary.json` |

## Configs

Training configs live in `src/config.py` (`SMOKE_CONFIG`, `SANITY_CONFIG`,
`FULL_CONFIG`) — selected via `--mode`. These are the source of truth read by
`train.py`.

- smoke — 300 images, 3 epochs, imgsz=320, batch=4
- full  — ~39k images, 150 epochs, imgsz=640, batch=16, SGD, augmentations

To resume an interrupted full run from the last checkpoint:
`make full-resume` (or `python src/train.py --mode full --resume`).

## Running Tests

```bash
make test
```

## Directory Structure

```
.
├── docker/
│   ├── Dockerfile.local      # CPU/MPS, Apple Silicon smoke test
│   ├── Dockerfile.runpod     # CUDA base, full training
│   ├── docker-compose.yml    # local convenience wrapper
│   └── entrypoint_runpod.sh  # CUDA check before training starts
├── configs/
│   ├── smoke.yaml
│   └── full.yaml
├── data/
│   ├── raw/                  # downloaded datasets (gitignored)
│   ├── yolo/
│   │   ├── images/{train,val,test}/
│   │   ├── labels/{train,val,test}/
│   │   ├── labels_raw/{crackseg9k,omnicrack30k}/
│   │   └── _qc/              # QC overlay images
│   └── data.yaml             # ultralytics dataset config
├── src/
│   ├── download_data.py
│   ├── convert_annotations.py
│   ├── clean_data.py
│   ├── make_subset.py
│   ├── train.py
│   ├── validate.py
│   └── test_predict.py
├── tests/
│   ├── test_convert_annotations.py
│   ├── test_clean_data.py
│   └── test_make_subset.py
├── runs/                     # ultralytics outputs (gitignored)
├── requirements.txt
├── Makefile
├── .dockerignore
└── README.md
```
