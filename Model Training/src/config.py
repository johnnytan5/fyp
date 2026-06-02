"""Central configuration. Edit MODE here or pass --mode to scripts."""
from pathlib import Path

# ---------------------------------------------------------------------------
# Mode — change to "full" for production training
# ---------------------------------------------------------------------------
MODE: str = "smoke"

# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------
DATA_RAW = Path("data/raw")
DATA_YOLO = Path("data/yolo")
DATA_YAML = Path("data/data.yaml")
CONFIG_DIR = Path("configs")
RUNS_DIR = Path("runs")
LABELS_RAW = Path("data/yolo/labels_raw")
QC_DIR = Path("data/yolo/_qc")
EVAL_DIR = Path("data/yolo/_eval_conversion")

# Dataset paths
CRACKSEG9K = DATA_RAW / "crackseg9k"
CRACKSEG9K_IMAGES = CRACKSEG9K / "images"
CRACKSEG9K_MASKS = CRACKSEG9K / "masks"

# OmniCrack30k — pre-split: images/{training,validation,test} + annotations/{training,validation,test}
OMNICRACK30K = DATA_RAW / "omnicrack30k"

# Label output paths
CRACKSEG9K_LABELS = LABELS_RAW / "crackseg9k"
OMNICRACK_LABELS = LABELS_RAW / "omnicrack30k"

# HuggingFace dataset (public, no auth)
HF_DATASET = "rimvydasrub/crackseg9k"

# ---------------------------------------------------------------------------
# Conversion parameters
# ---------------------------------------------------------------------------
MIN_AREA = 100
EPSILON_RATIO = 0.0005
QC_SAMPLES = 20

# ---------------------------------------------------------------------------
# Cleaning parameters
# ---------------------------------------------------------------------------
BLUR_THRESHOLD = 100.0

# ---------------------------------------------------------------------------
# Subset parameters
# ---------------------------------------------------------------------------
SUBSET_SIZE = 300
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Smoke test download — only pulls ~N_SMOKE_FILES from HuggingFace (no API key)
# ---------------------------------------------------------------------------
N_SMOKE_FILES = 250  # images to download for smoke (no Kaggle key needed)

# ---------------------------------------------------------------------------
# Training configurations
# ---------------------------------------------------------------------------
SMOKE_CONFIG = {
    "mode": "smoke",
    "base_model": "yolov8s-seg.pt",
    "epochs": 3,
    "batch": 4,
    "imgsz": 320,
    "patience": 5,
    "project": "runs/smoke",
    "name": "exp",
    "workers": 2,
}

FULL_CONFIG = {
    "mode": "full",
    "base_model": "yolov8s-seg.pt",
    "epochs": 200,
    "batch": 16,
    "imgsz": 640,
    "patience": 50,
    "optimizer": "SGD",
    "project": "runs/full",
    "name": "exp",
    "workers": 8,
    "mosaic": 1.0,
    "fliplr": 0.5,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "scale": 0.5,
}

SANITY_CONFIG = {
    "mode": "sanity",
    "base_model": "yolov8s-seg.pt",
    "epochs": 5,
    "batch": 16,
    "imgsz": 640,
    "patience": 10,
    "project": "runs/sanity",
    "name": "exp",
    "workers": 8,
}


def get_config(mode: str | None = None) -> dict:
    """Return training config for the given mode."""
    m = mode or MODE
    configs = {"smoke": SMOKE_CONFIG, "sanity": SANITY_CONFIG, "full": FULL_CONFIG}
    if m not in configs:
        raise ValueError(f"Invalid mode: {m}. Must be one of {set(configs.keys())}.")
    return configs[m]
