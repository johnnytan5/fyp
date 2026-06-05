import cv2
import numpy as np
import pytest
from pathlib import Path
from src.convert_annotations import mask_to_yolo_polygons


def make_mask(tmp_path: Path, draw_fn) -> Path:
    mask = np.zeros((100, 100), dtype=np.uint8)
    draw_fn(mask)
    p = tmp_path / "mask.png"
    cv2.imwrite(str(p), mask)
    return p


def test_single_rectangle_produces_one_line(tmp_path):
    def draw(m):
        cv2.rectangle(m, (10, 10), (50, 50), 255, -1)
    mask_path = make_mask(tmp_path, draw)
    lines = mask_to_yolo_polygons(mask_path, min_pixels=50)
    assert len(lines) == 1
    parts = lines[0].split()
    assert parts[0] == "0"
    assert len(parts) >= 7  # class + at least 3 xy pairs


def test_coords_normalized(tmp_path):
    def draw(m):
        cv2.rectangle(m, (5, 5), (90, 90), 255, -1)
    mask_path = make_mask(tmp_path, draw)
    lines = mask_to_yolo_polygons(mask_path, min_pixels=50)
    assert lines
    coords = [float(v) for v in lines[0].split()[1:]]
    assert all(0.0 <= c <= 1.0 for c in coords)


def test_small_contour_filtered(tmp_path):
    def draw(m):
        cv2.rectangle(m, (45, 45), (50, 50), 255, -1)  # ~25px²
    mask_path = make_mask(tmp_path, draw)
    lines = mask_to_yolo_polygons(mask_path, min_pixels=100)
    assert lines == []


def test_empty_mask_returns_empty(tmp_path):
    mask = np.zeros((100, 100), dtype=np.uint8)
    p = tmp_path / "empty.png"
    cv2.imwrite(str(p), mask)
    assert mask_to_yolo_polygons(p) == []


def test_multiple_contours(tmp_path):
    def draw(m):
        cv2.rectangle(m, (5, 5), (30, 30), 255, -1)
        cv2.rectangle(m, (60, 60), (90, 90), 255, -1)
    mask_path = make_mask(tmp_path, draw)
    lines = mask_to_yolo_polygons(mask_path, min_pixels=50)
    assert len(lines) == 2
