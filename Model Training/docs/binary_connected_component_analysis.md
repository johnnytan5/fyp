# Post-SAHI Connected-Component Analysis (CCA) Pipeline

> **Status:** TEMPORARY design note — not yet implemented. Parked until after model retrain.
> **Goal:** Fix SAHI's crack over-counting (1 crack → 20 detections) via post-processing.

---

## Problem

YOLOv8-Seg is trained on small image patches (256–640 px), so full-resolution
building photos must be sliced (SAHI) to keep cracks at a detectable pixel scale.
Side effect: **one physical crack spanning multiple tiles is reported as many
separate detections**. Example observed: 2–3 real cracks → reported as ~20.

Root cause: SAHI counts **detections** (per-tile bounding boxes), not **physical
crack instances**. A crack crossing 4 tiles yields 4+ boxes.

---

## Idea

The per-tile masks, when unioned onto the full image, form **one connected blob**
per physical crack. Count the blobs, not the boxes. This is classic
**connected-component labeling** (CCA) — `cv2.connectedComponentsWithStats`.

Not image classification, not detection. Pure post-processing on the merged
binary mask. Model-agnostic — no retrain needed.

---

## Pipeline

```
SAHI instance masks (full-image placed)
   → UNION into one binary mask
   → morphological CLOSE        (bridge gaps at tile seams)
   → connectedComponentsWithStats(connectivity=8)
   → filter components by min_area   (drop speckle)
   → each surviving component = 1 crack
        • count   = number of components
        • mask    = labels == i      (clean per-crack mask)
        • bbox    = stats[i] left/top/width/height
        • centroid (for label number)
```

### Reference implementation

```python
import cv2
import numpy as np


def merge_count(masks, shape, close_ksize=9, min_area=200):
    """Collapse SAHI fragment masks into physical crack instances.

    masks: list of (H, W) bool/uint8, each already placed on the full image.
    Returns (count, cracks) where each crack = {bbox, mask, centroid}.
    """
    union = np.zeros(shape, np.uint8)
    for m in masks:
        union |= (m > 0).astype(np.uint8)

    # Close small gaps where one crack broke across a tile boundary.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
    union = cv2.morphologyEx(union, cv2.MORPH_CLOSE, k)

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(union, connectivity=8)

    cracks = []
    for i in range(1, n):  # 0 = background
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        x, y, w, h = stats[i, :4]
        cracks.append({
            "bbox": (x, y, x + w, y + h),
            "mask": labels == i,
            "centroid": tuple(centroids[i].astype(int)),
        })
    return len(cracks), cracks
```

### Visualization — distinct color per crack

```python
def colorize(image, cracks):
    """Paint each crack a distinct, stable color + number it."""
    vis = image.copy()
    n = max(len(cracks), 1)
    for idx, c in enumerate(cracks, start=1):
        hue = int(idx * 180 / n) % 180                       # evenly spaced hues
        color = cv2.cvtColor(np.uint8([[[hue, 200, 255]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
        vis[c["mask"]] = color
        cv2.putText(vis, str(idx), c["centroid"],
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return cv2.addWeighted(image, 0.5, vis, 0.5, 0)
```

Result: 2–3 cracks, 2–3 distinct colors, numbered, count matches the picture.

---

## Tuning knobs

| Param | Effect | Risk if too high |
|-------|--------|------------------|
| `close_ksize` | bridges wider tile-seam gaps | fuses two nearby *separate* cracks → undercount |
| `connectivity` | 8 = diagonal touch connects | (keep 8 for diagonal cracks) |
| `min_area` | drops speckle false-positives | drops genuine tiny cracks |

Start: `close_ksize=9`, `connectivity=8`, `min_area=200`. Tune on real images.

---

## Caveats (state these in FYP defense)

1. **Post-processing heuristic, not model output.** The segmentation model still
   emits fragments; the *pipeline* stitches them. Don't claim "the model counts
   cracks" — the pipeline does. Legit CV technique, but be transparent.
2. **Touching/crossing cracks merge into one component.** Topologically correct,
   but two physically distinct cracks that intersect count as 1. Separating them
   needs skeletonization + graph analysis (out of scope, overkill for FYP).
3. **Branching (Y-shaped) crack = 1 component = 1 crack.** Usually desired.
4. **Merged bbox of a diagonal crack is large + mostly empty wall.** Prefer the
   mask overlay over the bbox for crack viz. Draw bbox only if a metric needs it.

---

## Where to wire (later)

In [infer.py](infer.py) `_infer_sahi`:
- After `get_sliced_prediction`, collect each pred's full-image mask.
- Call `merge_count(...)` → replace `len(preds)` count.
- Call `colorize(...)` → replace `visualize_object_predictions` output.

Drop-in, model-agnostic. Pairs with the retrained (un-poisoned) model.

---

## Order of work

1. Retrain model on corrected labels (priority — fixes mask quality).
2. Then wire CCA post-processing on top.

Better masks first → cleaner unions → cleaner component separation.
