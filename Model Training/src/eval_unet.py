"""Pixel-level segmentation evaluation for the U-Net slots (D, E).

Counterpart to eval_segmentation.py for YOLO models. Reuses its tested metric
math (Accum, GT resolution/polarity, clIoU/clDice) unchanged -- only the
prediction step differs: U-Net has no per-instance confidence to sweep like
YOLO's box scores, so the sweep axis here is the sigmoid probability
threshold on the dense per-pixel output instead.

Usage:
  python src/eval_unet.py --weights runs/full_unet/exp/weights/best.pt --split test
  python src/eval_unet.py --weights runs/full_unet_1280/exp/weights/best.pt --split test
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from config import DATA_YOLO
from eval_segmentation import (
    resolve_gt, load_gt, Accum, _cldice_img, _print_tables, _write_csv,
    EPS, DEFAULT_CONFS,
)
from train_unet import IMAGENET_MEAN, IMAGENET_STD


def load_unet(weights: Path, device: str):
    """Load a train_unet.py checkpoint. Returns (model, imgsz) -- imgsz is read
    from the checkpoint's own saved cfg, so it can't be mismatched by mistake."""
    try:
        import segmentation_models_pytorch as smp
    except ImportError:
        sys.exit("pip install segmentation-models-pytorch")

    ckpt = torch.load(weights, map_location=device, weights_only=False)
    imgsz = ckpt["cfg"]["imgsz"]
    tag = Path(ckpt["cfg"]["project"]).name  # e.g. "full_unet" / "full_unet_1280"

    model = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                     in_channels=3, classes=1).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, imgsz, tag


@torch.no_grad()
def predict_prob(model, image_path: Path, imgsz: int, device: str):
    """Return (h, w, prob) -- prob is the sigmoid output resized to native
    resolution, computed once per image (thresholds are applied cheaply after)."""
    img_bgr = cv2.imread(str(image_path))
    h, w = img_bgr.shape[:2]
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_r = cv2.resize(img, (imgsz, imgsz))
    img_t = torch.from_numpy(img_r.transpose(2, 0, 1).copy()).float() / 255.0
    img_t = ((img_t - IMAGENET_MEAN) / IMAGENET_STD).unsqueeze(0).to(device)

    logits = model(img_t)
    prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
    if (h, w) != (imgsz, imgsz):
        prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
    return h, w, prob


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--tolerance", default=4, type=int, help="clIoU buffer in px")
    parser.add_argument("--threshs", type=float, nargs="+", default=DEFAULT_CONFS,
                        help="sigmoid probability thresholds to sweep (like YOLO's --confs)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-images", type=int, default=0, help="0 = all")
    parser.add_argument("--out-dir", default="results", type=Path)
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

    model, imgsz, tag = load_unet(args.weights, args.device)
    print(f"[eval_unet] {args.weights} (tag={tag}, imgsz={imgsz}) on {args.split}: "
          f"{len(images)} images | threshs={args.threshs} | tolerance={args.tolerance}px")

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                  (2 * args.tolerance + 1, 2 * args.tolerance + 1))

    from collections import defaultdict
    acc: dict[tuple[str, float], Accum] = defaultdict(Accum)
    skipped = 0

    for i, img_path in enumerate(images):
        info = resolve_gt(img_path)
        if info is None:
            skipped += 1
            continue
        dataset, mask_path, invert = info
        h, w, prob = predict_prob(model, img_path, imgsz, args.device)
        gt = load_gt(mask_path, invert, (h, w))
        gt_dil = cv2.dilate(gt.astype(np.uint8), k).astype(bool)
        sg = skeletonize(gt) if gt.any() else np.zeros_like(gt)

        for thr in args.threshs:
            pred = prob > thr
            pred_dil = cv2.dilate(pred.astype(np.uint8), k).astype(bool)
            sp = skeletonize(pred) if pred.any() else np.zeros_like(pred)
            acc[(dataset, thr)].update(pred, gt, gt_dil, pred_dil, sp, sg)
            acc[("overall", thr)].update(pred, gt, gt_dil, pred_dil, sp, sg)

        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(images)}")

    datasets = ["overall", "omnicrack30k", "crackseg9k"]
    results: dict = {"weights": str(args.weights), "tag": tag, "imgsz": imgsz,
                     "split": args.split, "tolerance": args.tolerance,
                     "skipped": skipped, "by": {}}
    for ds in datasets:
        rows = {c: acc[(ds, c)].report() for c in args.threshs if (ds, c) in acc}
        if not rows:
            continue
        best_thr = max(rows, key=lambda c: rows[c]["F1"])
        results["by"][ds] = {"per_conf": rows, "best_f1_conf": best_thr,
                             "report_conf": args.threshs[len(args.threshs) // 2]}

    _print_tables(results, args.threshs)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"seg_eval_{tag}_{args.split}"
    import json
    (args.out_dir / f"{stem}.json").write_text(json.dumps(results, indent=2))
    _write_csv(args.out_dir / f"{stem}.csv", results, args.threshs)
    print(f"\nSaved {args.out_dir/stem}.json and .csv")


if __name__ == "__main__":
    main()
