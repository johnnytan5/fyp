"""Connected-component merging for the final facade's detections.

SAHI's tile-based slicing splits one continuous crack into several adjacent
box/mask detections wherever it crosses a tile boundary (seen repeatedly in
this project's testing -- e.g. one diagonal crack producing 3-4 separate
"crack" boxes strung along its length). Rasterize every detection onto one
binary canvas, morphologically CLOSE (bridge small gaps between fragments of
the same crack) then OPEN (remove isolated noise specks), then re-extract
connected components as the final, de-duplicated crack list.
"""
import sys
from typing import Any

import cv2
import numpy as np

CLOSE_KERNEL_PX = 15  # ponytail: tunable -- bridges gaps between tile-split fragments
OPEN_KERNEL_PX = 3    # ponytail: tunable -- removes small noise specks


def _rasterize(detections: list[dict], shape: tuple[int, int]) -> np.ndarray:
    canvas = np.zeros(shape[:2], dtype=np.uint8)
    for d in detections:
        if d.get("mask") is not None:
            pts = d["mask"].astype(np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(canvas, [pts], 255)
        else:
            x1, y1, x2, y2 = [int(round(v)) for v in d["bbox"]]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), 255, -1)
    return canvas


def merge_detections_cca(
    detections: list[dict],
    canvas_shape: tuple[int, int],
    close_kernel_px: int = CLOSE_KERNEL_PX,
    open_kernel_px: int = OPEN_KERNEL_PX,
) -> tuple[list[dict], np.ndarray]:
    """Returns (merged_detections, cleaned_binary_mask).

    Each merged detection: {"bbox", "contour" (Nx2 float array), "area_px",
    "score" (max of the raw detections whose mask/box overlaps this
    component -- conservative, doesn't dilute a strong finding by a weaker
    overlapping one)}.
    """
    raster = _rasterize(detections, canvas_shape)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel_px, close_kernel_px))
    closed = cv2.morphologyEx(raster, cv2.MORPH_CLOSE, close_kernel)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel_px, open_kernel_px))
    cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, open_kernel)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)

    def _rects_overlap(a, b) -> bool:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        return ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1

    merged = []
    for label in range(1, n_labels):  # 0 = background
        component_mask = (labels == label).astype(np.uint8) * 255
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)

        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        component_bbox = (x, y, x + w, y + h)

        # attribute by bbox overlap, not centroid containment -- a raw
        # detection's centroid can sit outside its own footprint for
        # L-shaped/diagonal masks, wrongly excluding a real contributor
        contributing_scores = [
            d["score"] for d in detections if _rects_overlap(d["bbox"], component_bbox)
        ]
        score = max(contributing_scores) if contributing_scores else 0.0

        merged.append({
            "bbox": (float(x), float(y), float(x + w), float(y + h)),
            "contour": contour.reshape(-1, 2).astype(np.float32),
            "area_px": area,
            "score": score,
        })

    return merged, cleaned


def draw_merged(image: np.ndarray, merged: list[dict], color=(0, 255, 0)) -> np.ndarray:
    out = image.copy()
    for d in merged:
        pts = d["contour"].astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], isClosed=True, color=color, thickness=2)
        x1, y1, _, _ = [int(round(v)) for v in d["bbox"]]
        cv2.putText(out, f"crack {d['score']:.2f} ({d['area_px']}px)", (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return out


def _self_check() -> None:
    """ponytail: two boxes with a small gap between them -- closing should
    bridge them into ONE merged component, not two -- while a box far away
    stays separate."""
    shape = (200, 200)
    detections = [
        {"bbox": (20, 90, 60, 110), "mask": None, "score": 0.5},
        {"bbox": (65, 90, 100, 110), "mask": None, "score": 0.8},  # 5px gap from the first
        {"bbox": (150, 20, 170, 40), "mask": None, "score": 0.3},  # far away, stays separate
    ]
    merged, cleaned = merge_detections_cca(detections, shape, close_kernel_px=15, open_kernel_px=3)
    assert len(merged) == 2, f"expected 2 components (1 merged pair + 1 separate), got {len(merged)}"
    bridged = max(merged, key=lambda d: d["area_px"])
    assert bridged["score"] == 0.8, "bridged component should take the max of its contributing scores"
    print("[cca] self-check OK", file=sys.stderr)


if __name__ == "__main__":
    _self_check()
