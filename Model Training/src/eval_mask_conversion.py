"""Evaluate mask-to-polygon conversion quality with visual comparison report."""
import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from convert_annotations import mask_to_yolo_polygons, MIN_AREA, EPSILON_RATIO


def polygons_to_mask(polygons: list[str], h: int, w: int) -> np.ndarray:
    """Reconstruct a binary mask from YOLO-Seg polygon lines."""
    mask = np.zeros((h, w), dtype=np.uint8)
    for line in polygons:
        parts = line.split()
        coords = np.array(parts[1:], dtype=float).reshape(-1, 2)
        pts = (coords * [w, h]).astype(np.int32)
        cv2.fillPoly(mask, [pts], 255)
    return mask


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Intersection over Union between two binary masks."""
    a_bin = (mask_a > 127).astype(np.uint8)
    b_bin = (mask_b > 127).astype(np.uint8)
    intersection = np.logical_and(a_bin, b_bin).sum()
    union = np.logical_or(a_bin, b_bin).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection / union)


def make_comparison_image(
    orig_mask: np.ndarray,
    recon_mask: np.ndarray,
    diff_mask: np.ndarray,
    filename: str,
    iou: float,
) -> np.ndarray:
    """Build 3-panel visualisation: original | reconstructed | diff overlay."""
    h, w = orig_mask.shape

    # Convert grayscale to BGR for coloured overlays
    orig_bgr = cv2.cvtColor(orig_mask, cv2.COLOR_GRAY2BGR)
    recon_bgr = cv2.cvtColor(recon_mask, cv2.COLOR_GRAY2BGR)

    # Diff panel: green=reconstruction, red=original only
    diff_bgr = np.zeros((h, w, 3), dtype=np.uint8)
    diff_bgr[:, :, 1] = recon_mask  # green channel = recon
    diff_bgr[:, :, 2] = orig_mask   # red channel   = original

    # Labels
    label_y = 20
    for panel, title in [(orig_bgr, "Original Mask"), (recon_bgr, "Polygon Mask"), (diff_bgr, f"Diff (IoU={iou:.4f})")]:
        cv2.putText(panel, title, (5, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Footer with filename
    panel_w = w
    footer = np.zeros((30, panel_w * 3, 3), dtype=np.uint8)
    cv2.putText(footer, filename, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    row = np.hstack([orig_bgr, recon_bgr, diff_bgr])
    return np.vstack([row, footer])


def evaluate_samples(
    mask_dir: Path,
    label_dir: Path,
    out_dir: Path,
    n: int,
    min_area: int = MIN_AREA,
    seed: int = 42,
) -> dict:
    """Select N random masks, compare originals vs reconstructions, save report."""
    random.seed(seed)
    mask_files = sorted(mask_dir.glob("*.png"))
    if not mask_files:
        print(f"No masks found in {mask_dir}")
        return {}

    samples = random.sample(mask_files, min(n, len(mask_files)))
    out_dir.mkdir(parents=True, exist_ok=True)

    ious = []
    for mask_path in samples:
        stem = mask_path.stem
        orig = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if orig is None:
            continue
        h, w = orig.shape

        # Re-read with the same pipeline used in conversion
        _, binary = cv2.threshold(orig, 127, 255, cv2.THRESH_BINARY)
        polygons = mask_to_yolo_polygons(mask_path, min_area)

        recon = polygons_to_mask(polygons, h, w)
        iou = compute_iou(binary, recon)
        ious.append(iou)

        comparison = make_comparison_image(binary, recon, np.zeros_like(binary), f"{stem}  IoU={iou:.4f}", iou)
        cv2.imwrite(str(out_dir / f"{stem}_comparison.png"), comparison)

    mean_iou = np.mean(ious) if ious else 0.0
    summary = {
        "samples": len(ious),
        "mean_iou": float(mean_iou),
        "min_iou": float(min(ious)) if ious else 0.0,
        "max_iou": float(max(ious)) if ious else 0.0,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20, help="Number of masks to sample")
    parser.add_argument("--mask-dir", type=Path, help="Directory containing mask PNGs")
    parser.add_argument("--label-dir", type=Path, help="Directory containing converted .txt labels")
    parser.add_argument("--out-dir", default="data/yolo/_eval_conversion", type=Path)
    parser.add_argument("--min-area", default=MIN_AREA, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--all", action="store_true", help="Evaluate both CrackSeg9k and OmniCrack30k")
    args = parser.parse_args()

    if args.all:
        datasets = [
            ("crackseg9k", "data/raw/crackseg9k/masks", "data/yolo/labels_raw/crackseg9k"),
            ("omnicrack30k", "data/raw/omnicrack30k/masks", "data/yolo/labels_raw/omnicrack30k"),
        ]
        for name, mask_dir, label_dir in datasets:
            print(f"\n=== {name} ===")
            summary = evaluate_samples(
                Path(mask_dir), Path(label_dir),
                args.out_dir / name, args.n, args.min_area, args.seed,
            )
            if summary:
                print(f"  Samples: {summary['samples']}")
                print(f"  Mean IoU: {summary['mean_iou']:.4f}")
                print(f"  Min IoU:  {summary['min_iou']:.4f}")
                print(f"  Max IoU:  {summary['max_iou']:.4f}")
            else:
                print("  No masks found (run make convert first?)")
    elif args.mask_dir and args.label_dir:
        summary = evaluate_samples(
            args.mask_dir, args.label_dir, args.out_dir, args.n, args.min_area, args.seed,
        )
        if summary:
            print(f"Samples: {summary['samples']}")
            print(f"Mean IoU: {summary['mean_iou']:.4f}")
            print(f"Min IoU:  {summary['min_iou']:.4f}")
            print(f"Max IoU:  {summary['max_iou']:.4f}")
        else:
            print("No masks found.")
    else:
        print("Specify --mask-dir and --label-dir, or use --all for both datasets.")
        return

    print(f"\nComparison images saved to {args.out_dir}")


if __name__ == "__main__":
    main()
