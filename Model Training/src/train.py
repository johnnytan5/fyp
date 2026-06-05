"""Single training entrypoint. Reads mode from config.py or --mode flag."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
_orig_load = torch.load
torch.load = lambda f, **kw: _orig_load(f, **{'weights_only': False, **kw})

from ultralytics import YOLO

from config import get_config, MODE, DATA_YAML


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default=MODE, choices=["smoke", "sanity", "full"])
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from last.pt in the run dir (e.g. after a pod dies). "
             "Ignores config args — uses the checkpoint's saved args/epoch.",
    )
    args = parser.parse_args()

    print(f"[train] mode={args.mode} resume={args.resume}")
    cfg = get_config(args.mode)

    last_weights = Path(cfg["project"]) / cfg["name"] / "weights" / "last.pt"

    if args.resume:
        if not last_weights.exists():
            raise FileNotFoundError(
                f"--resume given but no checkpoint at {last_weights}. "
                f"Run without --resume to start fresh."
            )
        print(f"[train] resuming from {last_weights}")
        model = YOLO(str(last_weights))
        model.train(resume=True)
    else:
        model = YOLO(cfg["base_model"])
        train_args: dict = {
            "data": str(DATA_YAML),
            "epochs": cfg["epochs"],
            "batch": cfg["batch"],
            "imgsz": cfg["imgsz"],
            "patience": cfg["patience"],
            "project": cfg["project"],
            "name": cfg["name"],
            "exist_ok": True,
            "workers": cfg.get("workers", 4),
        }
        if "optimizer" in cfg:
            train_args["optimizer"] = cfg["optimizer"]
        for aug_key in ("mosaic", "fliplr", "hsv_h", "hsv_s", "hsv_v", "scale"):
            if aug_key in cfg:
                train_args[aug_key] = cfg[aug_key]

        model.train(**train_args)

    best_weights = Path(cfg["project"]) / cfg["name"] / "weights" / "best.pt"
    if not best_weights.exists():
        print(f"WARNING: best.pt not found at {best_weights} — skipping validation.")
        return

    best_model = YOLO(str(best_weights))
    val_results = best_model.val(data=str(DATA_YAML))

    seg = getattr(val_results, "seg", None)
    map50 = seg.map50 if seg is not None else val_results.box.map50
    print(f"[train] Mask mAP50 = {map50:.4f}")


if __name__ == "__main__":
    main()
