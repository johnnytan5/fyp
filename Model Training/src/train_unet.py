"""Train a semantic U-Net (ResNet34 encoder) with Focal Tversky loss.

Counterpart to train.py's YOLO runs — same data/split, but a semantic model +
an imbalance-aware loss instead of instance-seg + BCE. Isolates two FYP2
levers at once: architecture family and loss function.

Usage:
  python src/train_unet.py [--resume]
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent))

from config import DATA_YOLO, UNET_CONFIG, UNET_1280_CONFIG

CONFIGS = {"unet": UNET_CONFIG, "unet-1280": UNET_1280_CONFIG}
from eval_segmentation import resolve_gt, load_gt  # reuse symlink + polarity logic

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class CrackSegDataset(Dataset):
    """Image + polarity-corrected binary mask, resized to imgsz. Light aug on train."""

    def __init__(self, split: str, imgsz: int, augment: bool):
        img_dir = DATA_YOLO / "images" / split
        self.paths = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
        if not self.paths:
            raise FileNotFoundError(f"No images in {img_dir} — run make_subset.py first")
        self.imgsz = imgsz
        self.augment = augment

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        img_path = self.paths[i]
        img_bgr = cv2.imread(str(img_path))
        h, w = img_bgr.shape[:2]

        info = resolve_gt(img_path)
        if info is None:
            mask = np.zeros((h, w), dtype=bool)
        else:
            _, mask_path, invert = info
            mask = load_gt(mask_path, invert, (h, w))

        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.imgsz, self.imgsz))
        m = cv2.resize(mask.astype(np.uint8), (self.imgsz, self.imgsz),
                       interpolation=cv2.INTER_NEAREST).astype(np.float32)

        if self.augment:
            if np.random.rand() < 0.5:            # fliplr, matches YOLO config
                img = np.ascontiguousarray(img[:, ::-1])
                m = np.ascontiguousarray(m[:, ::-1])
            if np.random.rand() < 0.5:             # brightness jitter, ~ hsv_v
                img = np.clip(img.astype(np.float32) * np.random.uniform(0.7, 1.3), 0, 255)

        img_t = torch.from_numpy(img.transpose(2, 0, 1).copy()).float() / 255.0
        img_t = (img_t - IMAGENET_MEAN) / IMAGENET_STD
        mask_t = torch.from_numpy(m).unsqueeze(0)
        return img_t, mask_t


def focal_tversky_loss(logits: torch.Tensor, target: torch.Tensor,
                       alpha: float = 0.3, beta: float = 0.7,
                       gamma: float = 1.33, eps: float = 1e-6) -> torch.Tensor:
    """alpha<beta weights false negatives harder -> pushes recall (the metric
    we're bottlenecked on). gamma>1 focuses the loss on hard/thin pixels."""
    p = torch.sigmoid(logits)
    tp = (p * target).sum()
    fp = (p * (1 - target)).sum()
    fn = ((1 - p) * target).sum()
    tversky = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return (1 - tversky) ** gamma


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    tp = fp = fn = 0.0
    loss_sum = n = 0
    for img, mask in loader:
        img, mask = img.to(device), mask.to(device)
        logits = model(img)
        loss_sum += focal_tversky_loss(logits, mask).item(); n += 1
        pred = (torch.sigmoid(logits) > 0.5).float()
        tp += (pred * mask).sum().item()
        fp += (pred * (1 - mask)).sum().item()
        fn += ((1 - pred) * mask).sum().item()
    iou = tp / (tp + fp + fn + 1e-9)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-9)
    return {"loss": loss_sum / max(n, 1), "iou": iou, "dice": dice}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="unet", choices=list(CONFIGS))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    try:
        import segmentation_models_pytorch as smp
    except ImportError:
        sys.exit("pip install segmentation-models-pytorch")

    cfg = CONFIGS[args.config]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train_unet] device={device} resume={args.resume} cfg={cfg}")

    out_dir = Path(cfg["project"]) / cfg["name"] / "weights"
    out_dir.mkdir(parents=True, exist_ok=True)
    last_ckpt = out_dir / "last.pt"
    best_ckpt = out_dir / "best.pt"

    train_ds = CrackSegDataset("train", cfg["imgsz"], augment=True)
    val_ds = CrackSegDataset("val", cfg["imgsz"], augment=False)
    train_loader = DataLoader(train_ds, batch_size=cfg["batch"], shuffle=True,
                              num_workers=cfg["workers"], pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch"], shuffle=False,
                            num_workers=cfg["workers"], pin_memory=True)
    print(f"[train_unet] train={len(train_ds)} val={len(val_ds)}")

    model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet",
                     in_channels=3, classes=1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                            factor=0.5, patience=5)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    start_epoch, best_dice, no_improve = 0, -1.0, 0
    if args.resume:
        if not last_ckpt.exists():
            raise FileNotFoundError(f"--resume given but no checkpoint at {last_ckpt}")
        ck = torch.load(last_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        start_epoch, best_dice, no_improve = ck["epoch"] + 1, ck["best_dice"], ck["no_improve"]
        print(f"[train_unet] resumed from epoch {start_epoch}, best_dice={best_dice:.4f}")

    for epoch in range(start_epoch, cfg["epochs"]):
        model.train()
        t0 = time.time(); train_loss = 0.0; n = 0
        for img, mask in train_loader:
            img, mask = img.to(device), mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                logits = model(img)
                loss = focal_tversky_loss(logits, mask)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            train_loss += loss.item(); n += 1

        val = evaluate(model, val_loader, device)
        scheduler.step(val["dice"])
        dt = time.time() - t0
        print(f"[train_unet] epoch {epoch+1}/{cfg['epochs']}  "
              f"train_loss={train_loss/max(n,1):.4f}  val_loss={val['loss']:.4f}  "
              f"val_iou={val['iou']:.4f}  val_dice={val['dice']:.4f}  ({dt:.0f}s)")

        improved = val["dice"] > best_dice
        if improved:
            best_dice, no_improve = val["dice"], 0
        else:
            no_improve += 1

        ckpt = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "epoch": epoch,
                "best_dice": best_dice, "no_improve": no_improve, "cfg": cfg}
        torch.save(ckpt, last_ckpt)
        if improved:
            torch.save(ckpt, best_ckpt)
        elif no_improve >= cfg["patience"]:
            print(f"[train_unet] early stop: no improvement for {cfg['patience']} epochs")
            break

    print(f"[train_unet] done. best_dice={best_dice:.4f}  weights at {out_dir}")


if __name__ == "__main__":
    main()
