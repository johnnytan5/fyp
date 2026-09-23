"""Thin wrapper around FE-Inferencing's existing SAHI crack-detection model
that returns raw detection geometry (boxes/masks in the input frame's pixel
coordinates), instead of infer.py's run_inference() which only returns a
pre-rendered annotated image. The stitching stages need the raw geometry to
project detections into panorama coordinates.
"""
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from infer import CONF_THRESHOLD, SAHI_OVERLAP, SAHI_SLICE_SIZE, load_sahi_model  # noqa: E402


def detect_cracks(image: np.ndarray, sahi_model: Any, slice_size: int = SAHI_SLICE_SIZE) -> list[dict]:
    """Run SAHI-sliced inference. Returns a list of detections, each:
    {"bbox": (x1, y1, x2, y2), "mask": Nx2 float array or None, "score": float}
    -- all coordinates in `image`'s own pixel space (not yet projected onto
    any panorama).
    """
    from sahi.predict import get_sliced_prediction

    result = get_sliced_prediction(
        image, sahi_model,
        slice_height=slice_size, slice_width=slice_size,
        overlap_height_ratio=SAHI_OVERLAP, overlap_width_ratio=SAHI_OVERLAP,
        verbose=0,
    )

    detections = []
    for pred in result.object_prediction_list:
        bbox = tuple(pred.bbox.to_xyxy())
        mask = None
        if pred.mask is not None and pred.mask.segmentation:
            # segmentation: list of polygon parts, each a flat [x1,y1,x2,y2,...] list.
            # Keep only the largest part if there are several (rare for thin cracks).
            parts = pred.mask.segmentation
            largest = max(parts, key=len)
            mask = np.array(largest, dtype=np.float32).reshape(-1, 2)
        detections.append({"bbox": bbox, "mask": mask, "score": float(pred.score.value)})
    return detections


def _self_check() -> None:
    """ponytail: smoke test only -- needs the real model weights, so just
    verifies the model loads and returns a well-formed (possibly empty)
    list on a blank image, not full detection-accuracy validation."""
    import sys as _sys
    model = load_sahi_model(conf=CONF_THRESHOLD)
    blank = np.zeros((640, 640, 3), dtype=np.uint8)
    dets = detect_cracks(blank, model)
    assert isinstance(dets, list)
    for d in dets:
        assert set(d.keys()) == {"bbox", "mask", "score"}
        assert len(d["bbox"]) == 4
    print(f"[detect] self-check OK ({len(dets)} detections on blank image)", file=_sys.stderr)


if __name__ == "__main__":
    _self_check()
