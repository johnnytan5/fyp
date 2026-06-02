import cv2
import numpy as np
import pytest
from pathlib import Path
from src.clean_data import laplacian_variance, is_near_empty_mask, find_orphans


def make_blurry_image(path: Path) -> None:
    img = np.ones((100, 100, 3), dtype=np.uint8) * 128  # flat gray = near-zero variance
    cv2.imwrite(str(path), img)


def make_sharp_image(path: Path) -> None:
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(img, (10, 10), (90, 90), (255, 255, 255), 2)
    cv2.imwrite(str(path), img)


def test_blurry_image_low_variance(tmp_path):
    p = tmp_path / "blurry.jpg"
    make_blurry_image(p)
    assert laplacian_variance(p) < 10.0


def test_sharp_image_high_variance(tmp_path):
    p = tmp_path / "sharp.jpg"
    make_sharp_image(p)
    assert laplacian_variance(p) > 10.0


def test_empty_label_detected(tmp_path):
    lbl = tmp_path / "empty.txt"
    lbl.write_text("")
    assert is_near_empty_mask(lbl) is True


def test_nonempty_label_not_flagged(tmp_path):
    lbl = tmp_path / "filled.txt"
    lbl.write_text("0 0.1 0.2 0.3 0.4 0.5 0.6")
    assert is_near_empty_mask(lbl) is False


def test_find_orphans_image_without_label(tmp_path):
    img_dir = tmp_path / "images"
    lbl_dir = tmp_path / "labels"
    img_dir.mkdir()
    lbl_dir.mkdir()
    (img_dir / "a.jpg").touch()
    (lbl_dir / "a.txt").touch()
    (img_dir / "b.jpg").touch()  # no label
    orphans = find_orphans(img_dir, lbl_dir)
    assert any(o["reason"] == "image_without_label" for o in orphans)


def test_no_orphans_when_aligned(tmp_path):
    img_dir = tmp_path / "images"
    lbl_dir = tmp_path / "labels"
    img_dir.mkdir()
    lbl_dir.mkdir()
    for name in ["a", "b", "c"]:
        (img_dir / f"{name}.jpg").touch()
        (lbl_dir / f"{name}.txt").touch()
    assert find_orphans(img_dir, lbl_dir) == []
