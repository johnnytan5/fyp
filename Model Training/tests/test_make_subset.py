import pytest
from pathlib import Path
from src.make_subset import stratified_sample


def make_fake_dataset(root: Path, n: int, prefix: str) -> tuple[Path, Path]:
    img_dir = root / prefix / "images"
    lbl_dir = root / prefix / "labels"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)
    for i in range(n):
        (img_dir / f"{prefix}_{i:04d}.jpg").touch()
        (lbl_dir / f"{prefix}_{i:04d}.txt").touch()
    return img_dir, lbl_dir


def test_total_count(tmp_path):
    src1 = make_fake_dataset(tmp_path, 200, "ds1")
    src2 = make_fake_dataset(tmp_path, 200, "ds2")
    pairs = stratified_sample([src1, src2], n=100)
    assert len(pairs) == 100


def test_balanced_across_sources(tmp_path):
    src1 = make_fake_dataset(tmp_path, 300, "ds1")
    src2 = make_fake_dataset(tmp_path, 300, "ds2")
    pairs = stratified_sample([src1, src2], n=100)
    ds1_count = sum(1 for img, _ in pairs if "ds1" in img.name)
    ds2_count = sum(1 for img, _ in pairs if "ds2" in img.name)
    assert abs(ds1_count - 50) <= 1
    assert abs(ds2_count - 50) <= 1


def test_reproducible_with_seed(tmp_path):
    src1 = make_fake_dataset(tmp_path, 300, "ds1")
    src2 = make_fake_dataset(tmp_path, 300, "ds2")
    a = sorted(img.name for img, _ in stratified_sample([src1, src2], n=100, seed=42))
    b = sorted(img.name for img, _ in stratified_sample([src1, src2], n=100, seed=42))
    assert a == b


def test_different_seeds_differ(tmp_path):
    src1 = make_fake_dataset(tmp_path, 300, "ds1")
    src2 = make_fake_dataset(tmp_path, 300, "ds2")
    a = sorted(img.name for img, _ in stratified_sample([src1, src2], n=100, seed=42))
    b = sorted(img.name for img, _ in stratified_sample([src1, src2], n=100, seed=99))
    assert a != b
