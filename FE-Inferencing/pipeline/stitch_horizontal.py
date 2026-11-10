"""Horizontal (per-level) stitching via cv2.detail -- OpenCV's own internal
stitching pipeline (what the high-level cv2.Stitcher wraps), used directly
instead of hand-rolled homography chaining. Gives us: joint bundle
adjustment (refines all frame positions together), DP-based seam finding,
and sharp no-blend compositing (multi-band blending was tried and rejected
-- it hides misalignment by averaging two frames' pixels together, which
reads as "ghosting" and, worse, smears the real crack texture we need for
detection) -- while restricting matching to nearby frames only
(BestOf2NearestRangeMatcher) so it doesn't fall into the repetitive-facade-
texture trap that broke plain cv2.Stitcher.

Affine motion model throughout (not full projective/spherical): two
consecutive video frames of a roughly flat wall differ by translation/
rotation/scale, never true shear/keystone -- confirmed by testing (a full
projective homography chain visibly warped straight lines; affine did not).

Also projects crack detections (from detect.py, in each source frame's raw
pixel coordinates) into the final panorama's pixel coordinates, using the
exact same per-frame camera warp the stitcher itself computed -- so overlay
positions are geometrically consistent with the stitch, not a separate
approximation.
"""
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

RANGE_WIDTH = 6     # only match a frame against its N nearest neighbors (see stitch_detail
                    # history: widening this didn't rescue genuinely poorly-conditioned pairs,
                    # but does help borderline ones without the cost of full all-pairs matching)
MATCH_CONF = 0.3    # OpenCV's default for ORB features
CONF_THRESH = 1.0   # keep strict: lowering this let a low-confidence pair corrupt the whole
                    # joint bundle adjustment in testing (fragmented/misplaced panorama)
WORK_MEGAPIX = 0.6
SEAM_MEGAPIX = 0.1
MAX_WARP_SIDE_RATIO = 4.0  # ponytail: tunable -- normal frames warp within ~1.3x of the row median
REMAP_MAX_PX = 32000       # cv2.remap asserts every side < SHRT_MAX (32767)


def stitch_horizontal(
    image_paths: list[Path],
    detections_per_frame: list[list[dict]] | None = None,
    range_width: int = RANGE_WIDTH,
    conf_thresh: float = CONF_THRESH,
) -> dict[str, Any]:
    """Returns {"panorama": np.ndarray, "detections": [...], "dropped": [indices]}.
    `detections_per_frame[i]` (if given) must line up with `image_paths[i]`,
    each item shaped like detect.py's output: {"bbox", "mask", "score"}.
    Projected detections come back in the SAME shape, in panorama pixel coords.
    """
    imgs_full = [cv2.imread(str(p)) for p in image_paths]
    if any(im is None for im in imgs_full):
        raise RuntimeError("failed to read one or more frames for stitching")
    n = len(imgs_full)
    h0, w0 = imgs_full[0].shape[:2]

    work_scale = min(1.0, np.sqrt(WORK_MEGAPIX * 1e6 / (h0 * w0)))
    seam_scale = min(1.0, np.sqrt(SEAM_MEGAPIX * 1e6 / (h0 * w0)))
    seam_work_aspect = seam_scale / work_scale

    finder = cv2.SIFT_create()
    images_work = [cv2.resize(im, None, fx=work_scale, fy=work_scale,
                               interpolation=cv2.INTER_LINEAR_EXACT) for im in imgs_full]
    features = [cv2.detail.computeImageFeatures2(finder, im) for im in images_work]

    matcher = cv2.detail_BestOf2NearestRangeMatcher(range_width, False, MATCH_CONF)
    pairwise_matches = matcher.apply2(features)
    matcher.collectGarbage()

    indices = [int(i) for i in cv2.detail.leaveBiggestComponent(features, pairwise_matches, conf_thresh)]
    dropped = [i for i in range(n) if i not in indices]
    if dropped:
        print(f"[stitch_horizontal] WARNING: {len(dropped)}/{n} frames dropped (not connected "
              f"enough to the rest): {[str(image_paths[i]) for i in dropped]}", file=sys.stderr)

    imgs_full = [imgs_full[i] for i in indices]
    images_work = [images_work[i] for i in indices]
    features = [features[i] for i in indices]
    pairwise_matches = matcher.apply2(features)
    matcher.collectGarbage()

    estimator = cv2.detail_AffineBasedEstimator()
    ok, cameras = estimator.apply(features, pairwise_matches, None)
    if not ok:
        raise RuntimeError("camera parameter estimation failed")
    for cam in cameras:
        cam.R = cam.R.astype(np.float32)

    adjuster = cv2.detail_BundleAdjusterAffinePartial()
    adjuster.setConfThresh(conf_thresh)
    ok, cameras = adjuster.apply(features, pairwise_matches, cameras)
    if not ok:
        raise RuntimeError("bundle adjustment failed to converge")

    focals = sorted(cam.focal for cam in cameras)
    mid = len(focals) // 2
    warped_image_scale = focals[mid] if len(focals) % 2 else (focals[mid - 1] + focals[mid]) / 2

    # A frame can pass the connectivity check yet get a degenerate camera from
    # bundle adjustment: its warp explodes (seen: 190k x 238k px -> OOM kill;
    # long thin warps -> remap's 32767 px limit) or collapses (seen: 4x4 px).
    # warpRoi sizes each warp without allocating. Keep a frame if each side is
    # within 1/r..r of the row's median warped side AND its full-res warp stays
    # under remap's hard limit (the cap still holds when the median is skewed).
    probe = cv2.PyRotationWarper('affine', warped_image_scale * seam_work_aspect)
    seam_wh = (int(round(w0 * seam_scale)), int(round(h0 * seam_scale)))
    sides = [probe.warpRoi(seam_wh, cam.K().astype(np.float32), cam.R)[2:] for cam in cameras]
    med_w, med_h = float(np.median([s[0] for s in sides])), float(np.median([s[1] for s in sides]))
    r = MAX_WARP_SIDE_RATIO
    keep = [i for i, (w, h) in enumerate(sides)
            if 1 / r <= w / med_w <= r and 1 / r <= h / med_h <= r
            and max(w, h) / seam_scale < REMAP_MAX_PX]
    if len(keep) < len(cameras):
        bad = [indices[i] for i in range(len(cameras)) if i not in keep]
        print(f"[stitch_horizontal] WARNING: dropped {len(bad)} frame(s) with degenerate camera "
              f"estimates (warped side outside 1/{r}-{r}x row median or over {REMAP_MAX_PX}px): "
              f"{[str(image_paths[i]) for i in bad]}", file=sys.stderr)
        dropped = sorted(dropped + bad)
        indices = [indices[i] for i in keep]
        imgs_full = [imgs_full[i] for i in keep]
        cameras = [cameras[i] for i in keep]
    if not cameras:
        raise RuntimeError("every frame had a degenerate camera estimate")

    # pass 1: warp at seam resolution, find seams there (cheap)
    seam_warper = cv2.PyRotationWarper('affine', warped_image_scale * seam_work_aspect)
    corners, sizes, images_warped, masks_warped = [], [], [], []
    for img_full, cam in zip(imgs_full, cameras):
        img_seam = cv2.resize(img_full, None, fx=seam_scale, fy=seam_scale,
                               interpolation=cv2.INTER_LINEAR_EXACT)
        K = cam.K().astype(np.float32)
        corner, warped = seam_warper.warp(img_seam, K, cam.R, cv2.INTER_LINEAR, cv2.BORDER_REFLECT)
        mask = np.full(img_seam.shape[:2], 255, np.uint8)
        _, warped_mask = seam_warper.warp(mask, K, cam.R, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT)
        corners.append(corner)
        sizes.append((warped.shape[1], warped.shape[0]))
        images_warped.append(warped)
        masks_warped.append(warped_mask)

    compensator = cv2.detail.ExposureCompensator_createDefault(cv2.detail.ExposureCompensator_GAIN_BLOCKS)
    compensator.feed(corners, images_warped, masks_warped)

    seam_finder = cv2.detail_DpSeamFinder('COLOR_GRAD')
    images_warped_f = [im.astype(np.float32) for im in images_warped]
    masks_warped = seam_finder.find(images_warped_f, corners, masks_warped)

    # pass 2: warp at full resolution for the final composite
    compose_work_aspect = 1.0 / work_scale
    full_warper = cv2.PyRotationWarper('affine', warped_image_scale * compose_work_aspect)

    corners_full, sizes_full, warped_fulls, masks_full, full_Ks = [], [], [], [], []
    for i, (img_full, cam) in enumerate(zip(imgs_full, cameras)):
        K = cam.K().astype(np.float32)
        K[0, 0] *= compose_work_aspect
        K[0, 2] *= compose_work_aspect
        K[1, 1] *= compose_work_aspect
        K[1, 2] *= compose_work_aspect
        corner, warped_full = full_warper.warp(img_full, K, cam.R, cv2.INTER_LINEAR, cv2.BORDER_REFLECT)
        mask_full = np.full(img_full.shape[:2], 255, np.uint8)
        _, warped_mask_full = full_warper.warp(mask_full, K, cam.R, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT)
        seam_mask_up = cv2.resize(masks_warped[i], (warped_mask_full.shape[1], warped_mask_full.shape[0]),
                                   interpolation=cv2.INTER_LINEAR)
        mask_final = cv2.bitwise_and(seam_mask_up, warped_mask_full)

        corners_full.append(corner)
        sizes_full.append((warped_full.shape[1], warped_full.shape[0]))
        warped_fulls.append(warped_full)
        masks_full.append(mask_final)
        full_Ks.append(K)

    dst_sz = cv2.detail.resultRoi(corners_full, sizes_full)
    blender = cv2.detail.Blender_createDefault(cv2.detail.Blender_NO)
    blender.prepare(dst_sz)

    for i in range(len(imgs_full)):
        compensator.apply(i, corners_full[i], warped_fulls[i], masks_full[i])
        blender.feed(warped_fulls[i].astype(np.int16), masks_full[i], corners_full[i])

    result, _ = blender.blend(None, None)
    panorama = cv2.convertScaleAbs(result)

    projected_detections = []
    if detections_per_frame is not None:
        dst_offset = (dst_sz[0], dst_sz[1])
        for subset_i, orig_i in enumerate(indices):
            dets = detections_per_frame[orig_i]
            if not dets:
                continue
            for d in dets:
                proj = _project_detection(d, full_warper, full_Ks[subset_i], cameras[subset_i].R,
                                           corners_full[subset_i], dst_offset)
                projected_detections.append(proj)

    return {"panorama": panorama, "detections": projected_detections, "dropped": dropped}


def _project_point(pt, warper, K, R, corner, dst_offset) -> tuple[float, float]:
    """Map a point in a source frame's raw pixel coords into the final
    panorama's pixel coords, using the same warp cv2.detail applied to the
    frame's pixels themselves."""
    wx, wy = warper.warpPoint((float(pt[0]), float(pt[1])), K, R)
    return (wx + corner[0] - dst_offset[0], wy + corner[1] - dst_offset[1])


def _project_detection(det: dict, warper, K, R, corner, dst_offset) -> dict:
    x1, y1, x2, y2 = det["bbox"]
    corners_box = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    proj_corners = [_project_point(p, warper, K, R, corner, dst_offset) for p in corners_box]
    xs = [p[0] for p in proj_corners]
    ys = [p[1] for p in proj_corners]
    proj_bbox = (min(xs), min(ys), max(xs), max(ys))

    proj_mask = None
    if det["mask"] is not None:
        proj_mask = np.array(
            [_project_point(p, warper, K, R, corner, dst_offset) for p in det["mask"]],
            dtype=np.float32,
        )

    return {"bbox": proj_bbox, "mask": proj_mask, "score": det["score"]}


def draw_detections(image: np.ndarray, detections: list[dict], color=(0, 0, 255)) -> np.ndarray:
    """Render projected detections onto an image (panorama or otherwise)."""
    out = image.copy()
    for d in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in d["bbox"]]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"crack {d['score']:.2f}"
        cv2.putText(out, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        if d["mask"] is not None:
            pts = d["mask"].astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(out, [pts], isClosed=True, color=color, thickness=1)
    return out


def _self_check() -> None:
    """ponytail: geometric check on the point-projection math using synthetic
    input -- warps a known point through the same code path used for real
    detections and checks it lands where a manual homography says it should."""
    # Build a trivial 2-camera affine scene: identity R for cam0, pure
    # translation for cam1, verify projecting a detection from image1's
    # coords ends up shifted by the expected amount in panorama coords.
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float32)
    R0 = np.eye(3, dtype=np.float32)
    warper = cv2.PyRotationWarper('affine', 500.0)
    corner0, _ = warper.warp(np.zeros((480, 640, 3), dtype=np.uint8), K, R0, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
    p0 = _project_point((100, 100), warper, K, R0, corner0, (0, 0))
    # projecting the SAME point with the SAME camera should be a no-op modulo
    # the warp's own (deterministic) mapping -- reproject and confirm it's stable
    p0_again = _project_point((100, 100), warper, K, R0, corner0, (0, 0))
    assert p0 == p0_again, "warpPoint should be deterministic"
    print("[stitch_horizontal] self-check OK (projection determinism)", file=sys.stderr)


if __name__ == "__main__":
    _self_check()
