"""Pixel-level segmentation evaluation for crack models.

Runs a YOLOv8-Seg model over a split, unions its instance masks into a binary
crack map per image, and compares against the ORIGINAL ground-truth masks
(polarity-corrected) — not the polygon labels. Reports the metrics the crack
literature uses, which (unlike COCO mAP) are fair to thin structures:

  - Pixel IoU            overlap of crack pixels
  - clIoU (tolerant)     IoU allowing a few-px offset (OmniCrack30k headline)
  - F1 / Dice            harmonic of pixel precision/recall
  - clDice               centerline connectivity (for length/width measurement)
  - Pixel Precision/Recall

Evaluated per dataset (overall / omnicrack30k / crackseg9k) across a confidence
grid, with micro (pixel-pooled, primary) and macro (per-image mean) aggregation.

Usage:
  python src/eval_segmentation.py --weights runs/full/exp/weights/best.pt \
      --split test --tolerance 4 [--max-images N] [--imgsz 640]
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import torch
_orig_load = torch.load
torch.load = lambda f, **kw: _orig_load(f, **{"weights_only": False, **kw})
from ultralytics import YOLO

from config import DATA_YOLO

EPS = 1e-9
DEFAULT_CONFS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
REPORT_CONF = 0.25


# --------------------------------------------------------------------------- #
# Ground-truth resolution: test image symlink -> source mask + polarity
# --------------------------------------------------------------------------- #
def resolve_gt(image_path: Path) -> tuple[str, Path, bool] | None:
    """Map a (symlinked) dataset image to (dataset, gt_mask_path, invert)."""
    real = Path(image_path).resolve()
    parts = real.parts
    if "crackseg9k" in parts:
        # .../crackseg9k/images/<stem>.png -> .../crackseg9k/masks/<stem>.png
        mask = real.parent.parent / "masks" / real.name
        return "crackseg9k", mask, False
    if "omnicrack30k" in parts:
        # .../omnicrack30k/images/<split>/<stem>.png
        #   -> .../omnicrack30k/annotations/<split>/<stem>.png
        split = real.parent.name
        root = real.parent.parent.parent  # omnicrack30k/
        mask = root / "annotations" / split / real.name
        return "omnicrack30k", mask, True
    return None


def load_gt(mask_path: Path, invert: bool, shape: tuple[int, int]) -> np.ndarray:
    """Binary GT crack map (crack=True) at the given (h, w)."""
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return np.zeros(shape, dtype=bool)
    if invert:
        m = 255 - m
    if m.shape[:2] != shape:
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return m > 127


# --------------------------------------------------------------------------- #
# Prediction: union instance masks (filtered by confidence) -> binary map
# --------------------------------------------------------------------------- #
def predict_masks(model: YOLO, image_path: Path, imgsz: int, device: str,
                  min_conf: float, max_det: int):
    """Return (orig_h, orig_w, polygons[list of (P,2) in orig coords], confs[N]).

    Uses polygon contours (masks.xy), not full-res mask tensors: memory stays
    bounded by imgsz regardless of image size (retina masks on multi-thousand-px
    OmniCrack images OOM the GPU). Polygons are rasterized per-conf downstream.
    """
    res = model.predict(str(image_path), conf=min_conf, imgsz=imgsz, max_det=max_det,
                        device=device, verbose=False)[0]
    h, w = res.orig_shape
    if res.masks is None or len(res.masks) == 0:
        return h, w, [], np.zeros((0,))
    polys = [p for p in res.masks.xy]                      # each (P,2), orig-image coords
    confs = res.boxes.conf.cpu().numpy()
    return h, w, polys, confs


def union_at_conf(polys: list, confs: np.ndarray, conf: float,
                  shape: tuple[int, int]) -> np.ndarray:
    canvas = np.zeros(shape, dtype=np.uint8)
    for p, c in zip(polys, confs):
        if c >= conf and len(p) >= 3:
            cv2.fillPoly(canvas, [p.astype(np.int32)], 1)
    return canvas.astype(bool)


# --------------------------------------------------------------------------- #
# Metric accumulators
# --------------------------------------------------------------------------- #
class Accum:
    """Micro (pixel-pooled) counters + macro per-image lists, for one bucket."""
    def __init__(self):
        # micro standard
        self.tp = self.fp = self.fn = 0
        # micro tolerant (clIoU)
        self.ttp = self.tfp = self.tfn = 0
        # micro clDice
        self.sp_hit = self.sp_tot = self.sg_hit = self.sg_tot = 0
        # macro (per crack-bearing image)
        self.iou_list: list[float] = []
        self.f1_list: list[float] = []
        self.cldice_list: list[float] = []

    def update(self, pred, gt, gt_dil, pred_dil, sp, sg):
        inter = np.logical_and(pred, gt).sum()
        psum, gsum = pred.sum(), gt.sum()
        tp = int(inter); fp = int(psum - inter); fn = int(gsum - inter)
        self.tp += tp; self.fp += fp; self.fn += fn

        ttp = int(np.logical_and(pred, gt_dil).sum())
        self.ttp += ttp
        self.tfp += int(psum - ttp)
        self.tfn += int(gsum - np.logical_and(gt, pred_dil).sum())

        self.sp_hit += int(np.logical_and(sp, gt).sum()); self.sp_tot += int(sp.sum())
        self.sg_hit += int(np.logical_and(sg, pred).sum()); self.sg_tot += int(sg.sum())

        if gsum > 0:  # macro only over images that contain a crack
            union = tp + fp + fn
            self.iou_list.append(tp / (union + EPS))
            self.f1_list.append(2 * tp / (2 * tp + fp + fn + EPS))
            self.cldice_list.append(_cldice_img(sp, sg, pred, gt))

    def report(self) -> dict:
        tp, fp, fn = self.tp, self.fp, self.fn
        iou = tp / (tp + fp + fn + EPS)
        f1 = 2 * tp / (2 * tp + fp + fn + EPS)
        prec = tp / (tp + fp + EPS); rec = tp / (tp + fn + EPS)
        cliou = self.ttp / (self.ttp + self.tfp + self.tfn + EPS)
        tprec = self.sp_hit / (self.sp_tot + EPS)
        tsens = self.sg_hit / (self.sg_tot + EPS)
        cldice = 2 * tprec * tsens / (tprec + tsens + EPS)
        return {
            "IoU": iou, "clIoU": cliou, "F1": f1, "clDice": cldice,
            "P": prec, "R": rec,
            "IoU_macro": float(np.mean(self.iou_list)) if self.iou_list else 0.0,
            "F1_macro": float(np.mean(self.f1_list)) if self.f1_list else 0.0,
            "clDice_macro": float(np.mean(self.cldice_list)) if self.cldice_list else 0.0,
            "n_crack_imgs": len(self.iou_list),
        }


def _cldice_img(sp, sg, pred, gt) -> float:
    if gt.sum() == 0 and pred.sum() == 0:
        return 1.0
    if sp.sum() == 0 or sg.sum() == 0:
        return 0.0
    tprec = np.logical_and(sp, gt).sum() / (sp.sum() + EPS)
    tsens = np.logical_and(sg, pred).sum() / (sg.sum() + EPS)
    return float(2 * tprec * tsens / (tprec + tsens + EPS))


# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--imgsz", default=640, type=int)
    parser.add_argument("--tolerance", default=4, type=int, help="clIoU buffer in px")
    parser.add_argument("--confs", type=float, nargs="+", default=DEFAULT_CONFS)
    parser.add_argument("--device", default="0")
    parser.add_argument("--max-det", type=int, default=100, help="cap instances per image")
    parser.add_argument("--max-images", type=int, default=0, help="0 = all")
    parser.add_argument("--out-dir", default="results", type=Path)
    parser.add_argument("--tag", type=str, default=None,
                        help="label for output filenames (default: derived from weights path project name)")
    args = parser.parse_args()

    try:
        from skimage.morphology import skeletonize
    except ImportError:
        sys.exit("scikit-image required for clDice. Install: pip install scikit-image")

    img_dir = DATA_YOLO / "images" / args.split
    images = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    if args.max_images:
        images = images[: args.max_images]
    if not images:
        sys.exit(f"No images in {img_dir}")

    print(f"[eval] {args.weights} on {args.split}: {len(images)} images | "
          f"confs={args.confs} | tolerance={args.tolerance}px")
    model = YOLO(str(args.weights))

    # derive tag from weights path if not given explicitly
    tag = args.tag or Path(args.weights).resolve().parent.parent.parent.name  # project dir name
    # e.g. runs/full_yolo11/exp/weights/best.pt -> full_yolo11
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                  (2 * args.tolerance + 1, 2 * args.tolerance + 1))

    # accumulators[(dataset, conf)] and [("overall", conf)]
    acc: dict[tuple[str, float], Accum] = defaultdict(Accum)
    skipped = 0

    for i, img_path in enumerate(images):
        info = resolve_gt(img_path)
        if info is None:
            skipped += 1
            continue
        dataset, mask_path, invert = info
        min_conf = min(args.confs)
        h, w, polys, confs = predict_masks(model, img_path, args.imgsz, args.device,
                                           min_conf, args.max_det)
        gt = load_gt(mask_path, invert, (h, w))
        gt_dil = cv2.dilate(gt.astype(np.uint8), k).astype(bool)
        sg = skeletonize(gt) if gt.any() else np.zeros_like(gt)

        for conf in args.confs:
            pred = union_at_conf(polys, confs, conf, (h, w))
            pred_dil = cv2.dilate(pred.astype(np.uint8), k).astype(bool)
            sp = skeletonize(pred) if pred.any() else np.zeros_like(pred)
            acc[(dataset, conf)].update(pred, gt, gt_dil, pred_dil, sp, sg)
            acc[("overall", conf)].update(pred, gt, gt_dil, pred_dil, sp, sg)

        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(images)}")

    # ---- assemble results ----
    datasets = ["overall", "omnicrack30k", "crackseg9k"]
    results: dict = {"weights": str(args.weights), "split": args.split,
                     "tolerance": args.tolerance, "skipped": skipped, "by": {}}
    # find best-F1 conf per dataset (micro)
    for ds in datasets:
        rows = {c: acc[(ds, c)].report() for c in args.confs if (ds, c) in acc}
        if not rows:
            continue
        best_conf = max(rows, key=lambda c: rows[c]["F1"])
        results["by"][ds] = {"per_conf": rows, "best_f1_conf": best_conf,
                             "report_conf": REPORT_CONF}

    _print_tables(results, args.confs)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"seg_eval_{tag}_{args.split}"
    (args.out_dir / f"{stem}.json").write_text(json.dumps(results, indent=2))
    _write_csv(args.out_dir / f"{stem}.csv", results, args.confs)
    print(f"\nSaved {args.out_dir/stem}.json and .csv")


def _fmt(m: dict) -> str:
    return (f"IoU {m['IoU']:.3f}  clIoU {m['clIoU']:.3f}  F1 {m['F1']:.3f}  "
            f"clDice {m['clDice']:.3f}  P {m['P']:.3f}  R {m['R']:.3f}")


def _print_tables(results: dict, confs: list[float]) -> None:
    for ds, d in results["by"].items():
        print(f"\n=== {ds}  (micro; {d['per_conf'][confs[0]]['n_crack_imgs']} crack imgs) ===")
        for c in confs:
            if c not in d["per_conf"]:
                continue
            tag = []
            if abs(c - d["report_conf"]) < 1e-9: tag.append("<-default")
            if c == d["best_f1_conf"]: tag.append("<-bestF1")
            print(f"  conf {c:.2f}: {_fmt(d['per_conf'][c])} {' '.join(tag)}")
        bm = d["per_conf"][d["best_f1_conf"]]
        print(f"  macro@bestF1: IoU {bm['IoU_macro']:.3f}  F1 {bm['F1_macro']:.3f}  "
              f"clDice {bm['clDice_macro']:.3f}")


def _write_csv(path: Path, results: dict, confs: list[float]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "conf", "IoU", "clIoU", "F1", "clDice", "P", "R",
                    "IoU_macro", "F1_macro", "clDice_macro", "is_default", "is_bestF1"])
        for ds, d in results["by"].items():
            for c in confs:
                if c not in d["per_conf"]:
                    continue
                m = d["per_conf"][c]
                w.writerow([ds, c, f"{m['IoU']:.4f}", f"{m['clIoU']:.4f}", f"{m['F1']:.4f}",
                            f"{m['clDice']:.4f}", f"{m['P']:.4f}", f"{m['R']:.4f}",
                            f"{m['IoU_macro']:.4f}", f"{m['F1_macro']:.4f}",
                            f"{m['clDice_macro']:.4f}",
                            int(abs(c - d["report_conf"]) < 1e-9), int(c == d["best_f1_conf"])])


if __name__ == "__main__":
    main()
