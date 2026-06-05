"""Evaluate mask-to-polygon conversion quality.

Two metrics:
  - miss_rate: of masks that contain a crack, the fraction that produced an
    EMPTY label (crack silently dropped). The key health number.
  - mean_iou: overlap between the original crack mask and the polygon
    reconstruction (how faithful the kept polygons are).

Sampling is reproducible: either a seeded random sample, or --sequential which
picks evenly-spaced masks by position in the sorted file list (no RNG, same
masks every run regardless of seed).
"""
import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from convert_annotations import mask_to_yolo_polygons, MIN_CRACK_PIXELS


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


def make_comparison_image(orig_mask, recon_mask, filename: str, iou: float) -> np.ndarray:
    """Build 3-panel visualisation: original | reconstructed | diff overlay."""
    h, w = orig_mask.shape
    orig_bgr = cv2.cvtColor(orig_mask, cv2.COLOR_GRAY2BGR)
    recon_bgr = cv2.cvtColor(recon_mask, cv2.COLOR_GRAY2BGR)
    diff_bgr = np.zeros((h, w, 3), dtype=np.uint8)
    diff_bgr[:, :, 1] = recon_mask  # green = polygon recon
    diff_bgr[:, :, 2] = orig_mask   # red   = original mask
    for panel, title in [(orig_bgr, "Original (crack=white)"),
                         (recon_bgr, "Polygon Mask"),
                         (diff_bgr, f"Diff IoU={iou:.3f}")]:
        cv2.putText(panel, title, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    row = np.hstack([orig_bgr, recon_bgr, diff_bgr])
    footer = np.zeros((26, w * 3, 3), dtype=np.uint8)
    cv2.putText(footer, filename, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    return np.vstack([row, footer])


def select_masks(mask_files: list[Path], n: int, seed: int, sequential: bool) -> list[Path]:
    """Pick n masks. sequential=True -> evenly-spaced by position (reproducible,
    no RNG). Otherwise a seeded random sample."""
    if n <= 0 or n >= len(mask_files):
        return mask_files
    if sequential:
        idx = sorted(set(np.linspace(0, len(mask_files) - 1, n).round().astype(int).tolist()))
        return [mask_files[i] for i in idx]
    random.seed(seed)
    return random.sample(mask_files, n)


def evaluate_samples(
    mask_dir: Path,
    out_dir: Path | None,
    n: int,
    invert: bool,
    min_pixels: int = MIN_CRACK_PIXELS,
    seed: int = 42,
    sequential: bool = False,
) -> dict:
    """Compute miss_rate + mean IoU over a sample of masks. Saves comparison
    images if out_dir is given."""
    mask_files = sorted(mask_dir.glob("*.png"))
    if not mask_files:
        print(f"  No masks found in {mask_dir}")
        return {}

    samples = select_masks(mask_files, n, seed, sequential)
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    ious: list[float] = []
    has_crack = missed = background = 0

    for mask_path in samples:
        orig = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if orig is None:
            continue
        h, w = orig.shape
        src = (255 - orig) if invert else orig          # crack = white after this
        crack_px = int((src > 127).sum())

        polygons = mask_to_yolo_polygons(mask_path, min_pixels, invert=invert)

        if crack_px == 0:
            background += 1                              # genuine no-crack image
        else:
            has_crack += 1
            if len(polygons) == 0:
                missed += 1                              # crack present, label empty

        _, binary = cv2.threshold(src, 127, 255, cv2.THRESH_BINARY)
        recon = polygons_to_mask(polygons, h, w)
        ious.append(compute_iou(binary, recon))

        if out_dir is not None:
            iou = ious[-1]
            img = make_comparison_image(binary, recon, f"{mask_path.stem} IoU={iou:.3f}", iou)
            cv2.imwrite(str(out_dir / f"{mask_path.stem}_cmp.png"), img)

    return {
        "samples": len(ious),
        "background": background,
        "has_crack": has_crack,
        "missed": missed,
        "miss_rate": (missed / has_crack) if has_crack else 0.0,
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "min_iou": float(np.min(ious)) if ious else 0.0,
    }


def _print(name: str, s: dict) -> None:
    if not s:
        print(f"  {name}: no masks (run convert first?)")
        return
    print(f"  {name}: samples={s['samples']} background={s['background']} has_crack={s['has_crack']}")
    print(f"    miss_rate = {s['missed']}/{s['has_crack']} = {s['miss_rate']*100:.2f}%   "
          f"mean_IoU = {s['mean_iou']:.4f}  (min {s['min_iou']:.3f})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=300, help="Masks to sample per dataset (<=0 = all)")
    parser.add_argument("--mask-dir", type=Path, help="Mask PNG dir (manual mode)")
    parser.add_argument("--invert", action="store_true",
                        help="Crack is the dark class (OmniCrack30k); flip before eval")
    parser.add_argument("--out-dir", default="data/yolo/_eval_conversion", type=Path)
    parser.add_argument("--no-images", action="store_true", help="Numbers only, skip comparison PNGs")
    parser.add_argument("--min-pixels", default=MIN_CRACK_PIXELS, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--sequential", action="store_true",
                        help="Sample evenly-spaced by position (reproducible, ignores --seed)")
    parser.add_argument("--all", action="store_true",
                        help="Eval CrackSeg9k + all OmniCrack30k splits with correct polarity")
    args = parser.parse_args()

    out = None if args.no_images else args.out_dir

    if args.all:
        targets = [
            ("crackseg9k", Path("data/raw/crackseg9k/masks"), False),
            ("omni/training", Path("data/raw/omnicrack30k/annotations/training"), True),
            ("omni/validation", Path("data/raw/omnicrack30k/annotations/validation"), True),
            ("omni/test", Path("data/raw/omnicrack30k/annotations/test"), True),
        ]
        for name, mdir, inv in targets:
            sub = (out / name.replace("/", "_")) if out is not None else None
            _print(name, evaluate_samples(mdir, sub, args.n, inv, args.min_pixels, args.seed, args.sequential))
    elif args.mask_dir:
        _print(str(args.mask_dir),
               evaluate_samples(args.mask_dir, out, args.n, args.invert, args.min_pixels, args.seed, args.sequential))
    else:
        print("Specify --mask-dir [--invert], or --all for both datasets.")
        return

    if out is not None:
        print(f"\nComparison images saved under {args.out_dir}")


if __name__ == "__main__":
    main()
