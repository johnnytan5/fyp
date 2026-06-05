"""Convert binary/semantic mask PNGs to YOLO-Seg polygon .txt files."""
import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from config import MIN_AREA, EPSILON_RATIO, DILATE_ITER, QC_SAMPLES, CRACKSEG9K, \
    OMNICRACK30K, CRACKSEG9K_LABELS, OMNICRACK_LABELS, QC_DIR


def mask_to_yolo_polygons(
    mask_path: Path, min_area: int = MIN_AREA, invert: bool = False,
    dilate_iter: int = DILATE_ITER,
) -> list[str]:
    """Return YOLO-Seg lines for one binary mask PNG. Class is always 0 (crack).

    Mask polarity differs by dataset: CrackSeg9k stores white crack on black bg
    (correct), OmniCrack30k stores black crack on white bg (inverted). Pass
    invert=True for datasets where the crack is the dark class so it becomes
    foreground before contouring.

    Thin (≈1px) cracks enclose almost no area, so cv2.contourArea would filter
    them out and approxPolyDP would collapse them to <3 points — dropping ~a
    third of OmniCrack30k cracks. dilate_iter>0 widens them first so they form a
    valid polygon with real area.
    """
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(mask_path)

    if invert:
        mask = 255 - mask

    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    if dilate_iter > 0:
        binary = cv2.dilate(binary, np.ones((3, 3), np.uint8), iterations=dilate_iter)
    h, w = binary.shape

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lines = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        epsilon = EPSILON_RATIO * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) < 3:
            continue
        pts = approx.reshape(-1, 2)
        coords = " ".join(f"{x/w:.6f} {y/h:.6f}" for x, y in pts)
        lines.append(f"0 {coords}")

    return lines


def _label(panel: np.ndarray, text: str) -> np.ndarray:
    panel = panel.copy()
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(panel, text, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
    return panel


def save_qc_overlay(
    image_path: Path, mask_path: Path, polygons: list[str], qc_out_dir: Path,
    invert: bool = False,
) -> None:
    """Write a 3-panel QC image: original | polygon overlay | ground-truth mask.

    The mask panel is polarity-corrected (crack = white) so a crack visible there
    with no green in the overlay panel is an obvious missed label.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return
    h, w = img.shape[:2]

    # Panel 2: polygon overlay
    overlay = img.copy()
    for line in polygons:
        parts = line.split()
        coords = np.array(parts[1:], dtype=float).reshape(-1, 2)
        pts = (coords * [w, h]).astype(np.int32)
        cv2.polylines(overlay, [pts], True, (0, 255, 0), 2)
        cv2.fillPoly(overlay, [pts], (0, 200, 0))
    blended = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)

    # Panel 3: ground-truth mask (crack = white)
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        mask_bgr = np.zeros((h, w, 3), dtype=np.uint8)
    else:
        if invert:
            m = 255 - m
        _, mbin = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
        if mbin.shape[:2] != (h, w):
            mbin = cv2.resize(mbin, (w, h), interpolation=cv2.INTER_NEAREST)
        mask_bgr = cv2.cvtColor(mbin, cv2.COLOR_GRAY2BGR)

    combined = np.hstack([
        _label(img, "ORIGINAL"),
        _label(blended, f"POLYGONS n={len(polygons)}"),
        _label(mask_bgr, "GT MASK (crack=white)"),
    ])
    qc_out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(qc_out_dir / image_path.name), combined)


def convert_dataset(
    image_dir: Path,
    mask_dir: Path,
    label_out_dir: Path,
    qc_out_dir: Path,
    min_area: int = MIN_AREA,
    invert: bool = False,
) -> int:
    """Convert all masks in mask_dir; returns count of converted files."""
    if not mask_dir.exists():
        print(f"  Skipping (no mask dir): {mask_dir}")
        return 0

    label_out_dir.mkdir(parents=True, exist_ok=True)
    mask_files = sorted(mask_dir.glob("*.png"))
    if not mask_files:
        print(f"  No masks found in {mask_dir}")
        return 0

    qc_candidates = set(random.sample(mask_files, min(QC_SAMPLES, len(mask_files))))
    converted = 0

    for mask_path in mask_files:
        stem = mask_path.stem
        polygons = mask_to_yolo_polygons(mask_path, min_area, invert=invert)
        (label_out_dir / f"{stem}.txt").write_text("\n".join(polygons))

        if mask_path in qc_candidates:
            img_path = image_dir / f"{stem}.jpg"
            if not img_path.exists():
                img_path = image_dir / f"{stem}.png"
            if img_path.exists():
                save_qc_overlay(img_path, mask_path, polygons, qc_out_dir, invert=invert)
        converted += 1

    return converted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-area", default=MIN_AREA, type=int)
    args = parser.parse_args()

    total = 0

    # CrackSeg9k — flat images/ + masks/
    if (CRACKSEG9K / "images").exists() and (CRACKSEG9K / "masks").exists():
        n = convert_dataset(
            CRACKSEG9K / "images", CRACKSEG9K / "masks",
            CRACKSEG9K_LABELS, QC_DIR / "crackseg9k",
            args.min_area, invert=False,  # white crack on black bg — correct as-is
        )
        print(f"CrackSeg9k: converted {n}")
        total += n
    else:
        print("CrackSeg9k: not downloaded (skipping)")

    # OmniCrack30k — pre-split: annotations/{training,validation,test} + images/{training,validation,test}
    omni_root = OMNICRACK30K
    if (omni_root / "images").exists() and (omni_root / "annotations").exists():
        for split in ["training", "validation", "test"]:
            img_dir = omni_root / "images" / split
            mask_dir = omni_root / "annotations" / split
            label_dir = OMNICRACK_LABELS / split
            if img_dir.exists() and mask_dir.exists():
                n = convert_dataset(
                    img_dir, mask_dir,
                    label_dir, QC_DIR / "omnicrack30k" / split,
                    args.min_area, invert=True,  # black crack on white bg — flip
                )
                print(f"OmniCrack30k/{split}: converted {n}")
                total += n
    else:
        print("OmniCrack30k: not extracted (skipping)")

    print(f"Total converted: {total}")


if __name__ == "__main__":
    main()
