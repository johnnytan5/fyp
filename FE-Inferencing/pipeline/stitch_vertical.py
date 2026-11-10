"""Vertical (across-level) stitching: combine each level's already-stitched
panorama into one full facade image.

Reuses stitch_horizontal's cv2.detail pipeline unchanged, rather than
reviving the earlier telemetry-constrained translation-only approach from
this project's exploration phase -- that approach depended on DJI's `D`
telemetry field for real-world sizing, which was later proven to be
distance-from-home-point, not distance-to-wall, and unusable for that
purpose. The vertical problem has the same shape as the horizontal one
(planar images, partial overlap), so: rotate each level panorama 90{deg},
stitch left-to-right (now "top-to-bottom" in the original orientation) with
the exact same bundle-adjusted pipeline, rotate the result back.

Known caveat carried over from earlier testing on blurry footage: a
level-to-level pair with weak/degenerate overlap can still get dropped by
the same strict confidence threshold that protects the horizontal case from
corrupting its whole bundle adjustment (see stitch_horizontal's history) --
expect this to need the same kind of validation pass once real (sharper)
footage is available, same as horizontal did.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

from stitch_horizontal import stitch_horizontal

# ponytail: tuned on DJI_0649 -- 0.6 adds level 6 cleanly; 0.4 links all 12 levels but
# tilts/misaligns the upper floors (plain wall = weak matches). Horizontal keeps 1.0.
VERTICAL_CONF_THRESH = 0.6


def _rotate_point_90cw(pt: tuple[float, float], src_h: float) -> tuple[float, float]:
    """(x, y) in an src_h-tall image -> its position after cv2.ROTATE_90_CLOCKWISE."""
    x, y = pt
    return (src_h - y, x)


def _rotate_point_90ccw(pt: tuple[float, float], src_w: float) -> tuple[float, float]:
    """(x, y) in an src_w-wide image -> its position after cv2.ROTATE_90_COUNTERCLOCKWISE."""
    x, y = pt
    return (y, src_w - x)


def _rotate_bbox(bbox, rotate_fn, dim) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    corners = [rotate_fn((x1, y1), dim), rotate_fn((x2, y1), dim),
               rotate_fn((x2, y2), dim), rotate_fn((x1, y2), dim)]
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    return (min(xs), min(ys), max(xs), max(ys))


def _rotate_detections(detections: list[dict], rotate_fn, dim: float) -> list[dict]:
    out = []
    for d in detections:
        mask = None
        if d["mask"] is not None:
            mask = np.array([rotate_fn(tuple(p), dim) for p in d["mask"]], dtype=np.float32)
        out.append({"bbox": _rotate_bbox(d["bbox"], rotate_fn, dim), "mask": mask, "score": d["score"]})
    return out


def stitch_vertical(level_panoramas: list[np.ndarray], level_detections: list[list[dict]],
                     tmp_dir: Path) -> dict:
    """level_panoramas: one stitched panorama per level, bottom-to-top order.
    level_detections[i]: detections already in level_panoramas[i]'s own pixel
    coordinates (stitch_horizontal's output). Returns the same shape as
    stitch_horizontal: {"panorama", "detections", "dropped"} (dropped indices
    here refer to whole LEVELS that couldn't be connected)."""
    tmp_dir.mkdir(parents=True, exist_ok=True)

    rotated_paths = []
    rotated_detections = []
    for i, (pano, dets) in enumerate(zip(level_panoramas, level_detections)):
        src_h = pano.shape[0]
        rotated = cv2.rotate(pano, cv2.ROTATE_90_CLOCKWISE)
        p = tmp_dir / f"level_{i:02d}_rot.jpg"
        cv2.imwrite(str(p), rotated)
        rotated_paths.append(p)
        rotated_detections.append(_rotate_detections(dets, _rotate_point_90cw, src_h))

    result = stitch_horizontal(rotated_paths, rotated_detections, conf_thresh=VERTICAL_CONF_THRESH)
    rotated_pano = result["panorama"]
    src_w = rotated_pano.shape[1]

    final_panorama = cv2.rotate(rotated_pano, cv2.ROTATE_90_COUNTERCLOCKWISE)
    final_detections = _rotate_detections(result["detections"], _rotate_point_90ccw, src_w)

    return {"panorama": final_panorama, "detections": final_detections, "dropped": result["dropped"]}


def _self_check() -> None:
    """ponytail: verifies the CW-then-CCW round trip is the identity, on
    both a point and a bbox -- this is the exact failure mode that would
    silently misplace every overlay if the rotation formulas had a sign or
    axis error."""
    h, w = 480, 640
    pt = (100.0, 50.0)
    rotated = _rotate_point_90cw(pt, h)
    back = _rotate_point_90ccw(rotated, h)  # rotated image's width == original height
    assert abs(back[0] - pt[0]) < 1e-6 and abs(back[1] - pt[1]) < 1e-6, (pt, rotated, back)

    bbox = (10.0, 20.0, 100.0, 200.0)
    r_bbox = _rotate_bbox(bbox, _rotate_point_90cw, h)
    back_bbox = _rotate_bbox(r_bbox, _rotate_point_90ccw, h)
    assert all(abs(a - b) < 1e-6 for a, b in zip(bbox, back_bbox)), (bbox, r_bbox, back_bbox)

    print("[stitch_vertical] self-check OK (rotation round-trip is identity)", file=sys.stderr)


if __name__ == "__main__":
    _self_check()
