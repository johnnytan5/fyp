"""Create stratified 300-image smoke-test subset from multiple sources."""
import argparse
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import SUBSET_SIZE, TRAIN_RATIO, VAL_RATIO, RANDOM_SEED


def stratified_sample(
    sources: list[tuple[Path, Path]],
    n: int,
    seed: int = RANDOM_SEED,
) -> list[tuple[Path, Path]]:
    """Return n (image, label) pairs sampled proportionally across sources."""
    random.seed(seed)
    per_source = n // len(sources)
    remainder = n % len(sources)

    sampled = []
    for i, (img_dir, lbl_dir) in enumerate(sources):
        images = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
        count = min(per_source + (1 if i < remainder else 0), len(images))
        if count == 0:
            continue
        for img in random.sample(images, count):
            sampled.append((img, lbl_dir / f"{img.stem}.txt"))

    return sampled


def copy_split(pairs: list[tuple[Path, Path]], img_out: Path, lbl_out: Path) -> None:
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    for img, lbl in pairs:
        shutil.copy2(img, img_out / img.name)
        if lbl.exists():
            shutil.copy2(lbl, lbl_out / lbl.name)
        else:
            (lbl_out / lbl.name).write_text("")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs="+", required=True,
                        help="Pairs of IMAGE_DIR:LABEL_DIR, e.g. dir1/imgs:dir1/lbls dir2/imgs:dir2/lbls")
    parser.add_argument("--out-dir", default="data/yolo", type=Path)
    parser.add_argument("--n", default=SUBSET_SIZE, type=int)
    parser.add_argument("--all", action="store_true", help="Use all images (no sampling)")
    parser.add_argument("--seed", default=RANDOM_SEED, type=int)
    args = parser.parse_args()

    sources = []
    for pair in args.sources:
        img_dir, lbl_dir = pair.split(":")
        img_p = Path(img_dir)
        lbl_p = Path(lbl_dir)
        if img_p.exists():
            n_files = len(list(img_p.glob("*")))
            print(f"  {img_dir}: {n_files} images")
            sources.append((img_p, lbl_p))
        else:
            print(f"  {img_dir}: NOT FOUND (skipping)")

    if not sources:
        print("ERROR: no sources found. Run 'make convert' first.")
        return

    if args.all:
        # Use all images — collect every pair
        pairs = []
        for img_dir, lbl_dir in sources:
            for img in sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png")):
                pairs.append((img, lbl_dir / f"{img.stem}.txt"))
    else:
        pairs = stratified_sample(sources, args.n, args.seed)

    random.seed(args.seed + 1)
    random.shuffle(pairs)
    n_train = int(len(pairs) * TRAIN_RATIO)
    n_val = int(len(pairs) * VAL_RATIO)

    splits = {
        "train": pairs[:n_train],
        "val":   pairs[n_train:n_train + n_val],
        "test":  pairs[n_train + n_val:],
    }
    for split_name, split_pairs in splits.items():
        copy_split(
            split_pairs,
            args.out_dir / "images" / split_name,
            args.out_dir / "labels" / split_name,
        )
        print(f"{split_name}: {len(split_pairs)} images")

    print(f"Subset ready in {args.out_dir}")


if __name__ == "__main__":
    main()
