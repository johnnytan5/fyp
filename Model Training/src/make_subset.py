"""Create stratified 300-image smoke-test subset from multiple sources."""
import argparse
import os
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


def collect_pairs(img_dir: Path, lbl_dir: Path) -> list[tuple[Path, Path]]:
    """Return all (image, label) pairs in a source directory."""
    return [
        (img, lbl_dir / f"{img.stem}.txt")
        for img in sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    ]


def ratio_split(pairs: list[tuple[Path, Path]], seed: int) -> dict[str, list]:
    """Shuffle and split pairs into train/val/test by TRAIN_RATIO / VAL_RATIO."""
    p = list(pairs)
    random.seed(seed)
    random.shuffle(p)
    n_train = int(len(p) * TRAIN_RATIO)
    n_val = int(len(p) * VAL_RATIO)
    return {
        "train": p[:n_train],
        "val":   p[n_train:n_train + n_val],
        "test":  p[n_train + n_val:],
    }


def _place(src: Path, dst: Path, link: bool) -> None:
    """Copy or relative-symlink src -> dst, replacing any existing entry."""
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    if link:
        rel = os.path.relpath(src.resolve(), dst.parent.resolve())
        os.symlink(rel, dst)
    else:
        shutil.copy2(src, dst)


def copy_split(
    pairs: list[tuple[Path, Path]], img_out: Path, lbl_out: Path, name: str,
    link: bool = False,
) -> None:
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    total = len(pairs)
    verb = "linked" if link else "copied"
    for i, (img, lbl) in enumerate(pairs):
        _place(img, img_out / img.name, link)
        if lbl.exists():
            _place(lbl, lbl_out / lbl.name, link)
        else:
            # Missing label = legitimate negative (no crack); write a real empty file.
            (lbl_out / lbl.name).write_text("")
        if (i + 1) % 2000 == 0:
            print(f"  {name} {verb} {i+1}/{total}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sources", nargs="+", required=True,
        help="IMAGE_DIR:LABEL_DIR[:SPLIT] entries. SPLIT is one of "
             "train/val/test/auto (default auto). With --all, a fixed SPLIT "
             "sends that source straight to that split (e.g. respect a dataset's "
             "official split); 'auto' sources are pooled and ratio-split.",
    )
    parser.add_argument("--out-dir", default="data/yolo", type=Path)
    parser.add_argument("--n", default=SUBSET_SIZE, type=int)
    parser.add_argument("--all", action="store_true", help="Use all images (no sampling)")
    parser.add_argument("--link", action="store_true",
                        help="Symlink images/labels instead of copying (fast, no duplicate storage). "
                             "Uses relative links so they survive volume re-mounts.")
    parser.add_argument("--seed", default=RANDOM_SEED, type=int)
    args = parser.parse_args()

    valid_tags = {"train", "val", "test", "auto"}
    sources = []  # (img_p, lbl_p, split_tag)
    for entry in args.sources:
        parts = entry.split(":")
        img_dir, lbl_dir = parts[0], parts[1]
        tag = parts[2] if len(parts) > 2 else "auto"
        if tag not in valid_tags:
            raise ValueError(f"Bad split tag '{tag}' in '{entry}'. Use one of {valid_tags}.")
        img_p, lbl_p = Path(img_dir), Path(lbl_dir)
        if img_p.exists():
            n_files = len(list(img_p.glob("*")))
            print(f"  {img_dir} [{tag}]: {n_files} images")
            sources.append((img_p, lbl_p, tag))
        else:
            print(f"  {img_dir}: NOT FOUND (skipping)")

    if not sources:
        print("ERROR: no sources found. Run 'make convert' first.")
        return

    splits: dict[str, list] = {"train": [], "val": [], "test": []}

    if args.all:
        # Fixed-split sources go straight to their split (respect official splits);
        # 'auto' sources are pooled and ratio-split into train/val/test.
        auto_pool: list[tuple[Path, Path]] = []
        for img_p, lbl_p, tag in sources:
            pairs = collect_pairs(img_p, lbl_p)
            if tag == "auto":
                auto_pool.extend(pairs)
            else:
                splits[tag].extend(pairs)
        if auto_pool:
            for name, part in ratio_split(auto_pool, args.seed).items():
                splits[name].extend(part)
    else:
        # Smoke subset: stratified sample across sources, then ratio-split.
        sources_2 = [(img_p, lbl_p) for img_p, lbl_p, _ in sources]
        sampled = stratified_sample(sources_2, args.n, args.seed)
        splits = ratio_split(sampled, args.seed + 1)

    for split_name, split_pairs in splits.items():
        copy_split(
            split_pairs,
            args.out_dir / "images" / split_name,
            args.out_dir / "labels" / split_name,
            split_name,
            link=args.link,
        )
        print(f"{split_name}: {len(split_pairs)} images")

    print(f"Subset ready in {args.out_dir}")


if __name__ == "__main__":
    main()
