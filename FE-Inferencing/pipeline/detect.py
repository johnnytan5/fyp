"""Crack detection for the pipeline: U-Net (ResNet34) semantic segmentation,
Experiment D -- best test F1/clIoU in the ablation (Model Training/docs).

Returns raw detection geometry (polygons in the input frame's pixel
coordinates) so the stitching stages can project detections onto the facade.
The YOLOv8-Seg + SAHI path (Experiment B) stays in infer.py for app.py.
"""
from pathlib import Path

import cv2
import numpy as np
import torch

WEIGHTS_PATH = Path(__file__).parent.parent / "weights" / "unet_best.pt"
TILE, OVERLAP, BATCH = 640, 0.2, 8  # trained at 640; 20% overlap like the old SAHI tiles
# Pixel probability cut-off. 0.5 = the operating point of the test-set F1 (69.8).
CONF_THRESHOLD = 0.5
MIN_AREA_PX = 30  # drop speckle components
_MEAN, _STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])  # ImageNet, as in training


def cache_path(frame: Path, conf: float) -> Path:
    """Per-frame detection cache, keyed by model + conf so a change never reuses stale results."""
    return frame.with_name(f"{frame.stem}.dets_unet_c{conf:.2f}.pkl")


def load_detector(conf: float = CONF_THRESHOLD, weights: Path = WEIGHTS_PATH) -> dict:
    """U-Net on Apple GPU (MPS) / CUDA when present, else CPU (~0.8 s vs ~5 s per 1080p frame)."""
    import segmentation_models_pytorch as smp
    if not weights.exists():
        raise FileNotFoundError(f"U-Net weights not found: {weights}. Copy Experiment D best.pt there "
                                "(HF johnnytan5/2nd-run-fyp-train, D-unet-640-150epoch/run/weights/best.pt)")
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    net = smp.Unet(encoder_name="resnet34", encoder_weights=None, classes=1)
    net.load_state_dict(torch.load(weights, map_location="cpu", weights_only=False)["model"])
    return {"net": net.eval().to(dev), "dev": dev, "conf": conf}


def _starts(n: int) -> list[int]:
    step = int(TILE * (1 - OVERLAP))
    s = list(range(0, max(n - TILE, 0) + 1, step))
    return s + [n - TILE] if n > TILE and s[-1] + TILE < n else s


def crack_probability(image: np.ndarray, det: dict) -> np.ndarray:
    """Tiled inference; overlapping tiles are averaged. Returns HxW float32 in [0, 1]."""
    h, w = image.shape[:2]
    ph, pw = max(h, TILE), max(w, TILE)  # pad frames smaller than one tile
    x = np.zeros((ph, pw, 3), np.float32)
    x[:h, :w] = (cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0 - _MEAN) / _STD
    tiles = [(y, x0) for y in _starts(ph) for x0 in _starts(pw)]
    prob, cnt = np.zeros((ph, pw), np.float32), np.zeros((ph, pw), np.float32)
    for i in range(0, len(tiles), BATCH):
        b = np.stack([x[y:y + TILE, x0:x0 + TILE].transpose(2, 0, 1) for y, x0 in tiles[i:i + BATCH]])
        with torch.no_grad():
            p = torch.sigmoid(det["net"](torch.from_numpy(b).to(det["dev"])))[:, 0].cpu().numpy()
        for (y, x0), pp in zip(tiles[i:i + BATCH], p):
            prob[y:y + TILE, x0:x0 + TILE] += pp
            cnt[y:y + TILE, x0:x0 + TILE] += 1
    return (prob / cnt)[:h, :w]


def detect_cracks(image: np.ndarray, det: dict) -> list[dict]:
    """One detection per connected crack region:
    {"bbox": (x1, y1, x2, y2), "mask": Nx2 float polygon, "score": mean crack probability}
    -- coordinates in `image`'s own pixel space (not yet projected onto the facade).
    """
    prob = crack_probability(image, det)
    binary = (prob >= det["conf"]).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < MIN_AREA_PX:
            continue
        comp = (labels[y:y + h, x:x + w] == i).astype(np.uint8)
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        poly = max(contours, key=len).reshape(-1, 2).astype(np.float32) + (x, y)
        out.append({"bbox": (float(x), float(y), float(x + w), float(y + h)), "mask": poly,
                    "score": float(prob[y:y + h, x:x + w][comp > 0].mean())})
    return out


def _self_check() -> None:
    """Needs real weights. Blank frame -> nothing; drawn dark line on grey wall -> well-formed dets."""
    det = load_detector()
    assert detect_cracks(np.full((700, 900, 3), 255, np.uint8), det) == []
    img = np.full((700, 900, 3), 170, np.uint8)
    cv2.line(img, (100, 100), (800, 600), (40, 40, 40), 3)
    for d in detect_cracks(img, det):
        assert set(d) == {"bbox", "mask", "score"} and d["mask"].shape[1] == 2 and d["score"] >= det["conf"]
    print(f"[detect] self-check OK on {det['dev']}")


if __name__ == "__main__":
    _self_check()
