"""Run model.val() on a split and print Mask mAP50."""
import argparse
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
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    args = parser.parse_args()

    if not args.weights.exists():
        print(f"ERROR: weights not found: {args.weights}")
        sys.exit(1)

    model = YOLO(str(args.weights))
    results = model.val(data=str(DATA_YAML), split=args.split)

    seg = getattr(results, "seg", None)
    map50    = seg.map50 if seg is not None else results.box.map50
    map5095  = seg.map   if seg is not None else results.box.map

    print(f"Validation ({args.split}):")
    print(f"  Mask mAP50:    {map50:.4f}")
    print(f"  Mask mAP50-95: {map5095:.4f}")


if __name__ == "__main__":
    main()
