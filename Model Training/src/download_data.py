"""Download CrackSeg9k from HuggingFace into data/raw/."""
import argparse
import base64
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import CRACKSEG9K, CRACKSEG9K_IMAGES, CRACKSEG9K_MASKS, \
    N_SMOKE_FILES, MODE as DEFAULT_MODE


def _save_image(data, path: Path) -> None:
    """Save image data (str, bytes, or PIL Image) to disk as PNG."""
    if isinstance(data, str):
        data = base64.b64decode(data)
    if isinstance(data, bytes):
        from PIL import Image as PILImage
        img = PILImage.open(io.BytesIO(data))
        img.save(str(path))
    else:
        data.save(str(path))


def download_crackseg9k(dest_images: Path, dest_masks: Path, n: int | None = None) -> int:
    """Download CrackSeg9k from HuggingFace datasets. Returns count saved."""
    from datasets import load_dataset

    mode_str = f" (first {n} rows)" if n else " (all rows)"
    print(f"Loading crackseg9k from HuggingFace{mode_str}...")

    dataset = load_dataset("rimvydasrub/crackseg9k", split="train", streaming=(n is not None))

    dest_images.mkdir(parents=True, exist_ok=True)
    dest_masks.mkdir(parents=True, exist_ok=True)

    count = 0
    for i, row in enumerate(dataset):
        if n and i >= n:
            break
        stem = f"{i:05d}"
        _save_image(row["image"], dest_images / f"{stem}.png")
        _save_image(row["mask"], dest_masks / f"{stem}.png")
        count += 1
        if count % 50 == 0:
            print(f"\r  {count} pairs saved", end="", flush=True)

    print(f"\r  {count} pairs saved")
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default=DEFAULT_MODE, choices=["smoke", "full"])
    args = parser.parse_args()

    print(f"[download] mode={args.mode}")

    if args.mode == "smoke":
        n = download_crackseg9k(CRACKSEG9K_IMAGES, CRACKSEG9K_MASKS, n=N_SMOKE_FILES)
    else:
        n = download_crackseg9k(CRACKSEG9K_IMAGES, CRACKSEG9K_MASKS)

    print(f"Download complete: {n} pairs → {CRACKSEG9K}")


if __name__ == "__main__":
    main()
