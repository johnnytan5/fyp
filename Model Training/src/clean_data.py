"""Remove blurry/mislabelled images; verify image-label alignment."""
import argparse
import csv
from pathlib import Path

import cv2

BLUR_THRESHOLD = 100.0  # Laplacian variance below this = blurry


def laplacian_variance(image_path: Path) -> float:
    """Return variance of Laplacian as sharpness metric."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def is_near_empty_mask(label_path: Path) -> bool:
    """Return True if label file exists but has no polygon lines."""
    if not label_path.exists():
        return False
    return label_path.read_text().strip() == ""


def find_orphans(image_dir: Path, label_dir: Path) -> list[dict]:
    """Return orphan entries: image without label or label without image."""
    img_stems = {f.stem for f in image_dir.glob("*") if f.suffix in {".jpg", ".png"}}
    lbl_stems = {f.stem for f in label_dir.glob("*.txt")}
    orphans = []
    for stem in img_stems - lbl_stems:
        orphans.append({"filename": f"{stem}.*", "reason": "image_without_label"})
    for stem in lbl_stems - img_stems:
        orphans.append({"filename": f"{stem}.txt", "reason": "label_without_image"})
    return orphans


def clean_dataset(
    image_dir: Path,
    label_dir: Path,
    report_rows: list,
    blur_threshold: float = BLUR_THRESHOLD,
) -> None:
    """Delete blurry images and images with empty masks; append removals to report_rows."""
    images = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png"))
    for img_path in images:
        label_path = label_dir / f"{img_path.stem}.txt"
        var = laplacian_variance(img_path)
        if var < blur_threshold:
            report_rows.append({"filename": img_path.name, "reason": f"blurry (var={var:.1f})"})
            img_path.unlink(missing_ok=True)
            label_path.unlink(missing_ok=True)
            continue
        if is_near_empty_mask(label_path):
            report_rows.append({"filename": img_path.name, "reason": "empty_mask"})
            img_path.unlink(missing_ok=True)
            label_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--label-dir", required=True, type=Path)
    parser.add_argument("--report", default="data/cleaning_report.csv", type=Path)
    parser.add_argument("--blur-threshold", default=BLUR_THRESHOLD, type=float)
    args = parser.parse_args()

    report_rows: list[dict] = []
    clean_dataset(args.image_dir, args.label_dir, report_rows, args.blur_threshold)

    orphans = find_orphans(args.image_dir, args.label_dir)
    report_rows.extend(orphans)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "reason"])
        writer.writeheader()
        writer.writerows(report_rows)

    remaining = find_orphans(args.image_dir, args.label_dir)
    if remaining:
        print(f"WARNING: {len(remaining)} orphans remain after cleaning")
        for o in remaining:
            print(f"  {o}")
    else:
        print("No orphans remain.")
    print(f"Removed {len(report_rows)} files. Report: {args.report}")


if __name__ == "__main__":
    main()
