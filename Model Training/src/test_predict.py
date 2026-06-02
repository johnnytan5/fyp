"""Run inference on test split; save overlays and metrics summary.json."""
import argparse
import json
import sys
from pathlib import Path

import torch
_orig_load = torch.load
torch.load = lambda f, **kw: _orig_load(f, **{'weights_only': False, **kw})

from ultralytics import YOLO

DATA_YAML = Path("data/data.yaml")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--test-images", default="data/yolo/images/test", type=Path)
    parser.add_argument("--out-dir", default="runs/test_predictions", type=Path)
    parser.add_argument("--conf", default=0.25, type=float)
    args = parser.parse_args()

    if not args.weights.exists():
        print(f"ERROR: weights not found: {args.weights}")
        sys.exit(1)

    model = YOLO(str(args.weights))
    results = model.predict(
        source=str(args.test_images),
        save=True,
        save_txt=True,
        conf=args.conf,
        project=str(args.out_dir.parent),
        name=args.out_dir.name,
        exist_ok=True,
    )

    total = len(results)
    with_detections = sum(1 for r in results if r.masks is not None and len(r.masks) > 0)
    summary = {
        "total_images": total,
        "images_with_detections": with_detections,
        "detection_rate": with_detections / total if total > 0 else 0.0,
        "conf_threshold": args.conf,
        "weights": str(args.weights),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Test predictions → {args.out_dir}")
    print(f"  Total:          {total}")
    print(f"  With detections:{with_detections}")
    print(f"  Detection rate: {summary['detection_rate']:.2%}")


if __name__ == "__main__":
    main()
