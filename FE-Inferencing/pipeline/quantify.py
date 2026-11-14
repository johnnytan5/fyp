"""Crack dimension quantification: length and width in millimetres per crack.

Scale. SfM reconstructs the facade up to an unknown scale. The drone's
altimeter supplies it: every frame's telemetry height (metres) is regressed
against the same camera's height along the wall in the SfM model (units),
h = s * up + b, and s converts model units to metres. That also measures the
camera-to-wall distance, which nobody had to tape. On J17: stills 4.80 m
(R^2 0.9999, 55 photos over 10.5 m of climb), July video 6.83 m (R^2 0.998,
SRT height has 0.1 m steps). Both flights then give the same floor-to-floor
height (3.2 / 3.1 m), which is the independent check.
Fallback when no usable telemetry (fewer than MIN_HEIGHT_FRAMES, under
MIN_HEIGHT_SPAN_M of climb, or a poor fit): assume WALL_DISTANCE_M stand-off.
Each frame keeps its own SfM distance and its own SfM-calibrated focal
length, so per-frame GSD = distance / f_px.

Camera: DJI Mavic Air 2, EXIF model FC3170. 1/2" Sony IMX586, 8000x6000 at
0.8 um -> 6.4 x 4.8 mm sensor, 4.5 mm lens (24 mm equiv, f/2.8, 84 deg FOV).
spec_f_px() is the spec-sheet focal, used only as a cross-check: the 4K
video is a sensor crop (SfM gave f = 3080 px vs 2700 px from the spec at
3840 px wide), so the spec sheet alone would mis-scale video by ~14%.

Length: skeleton of the merged facade crack (continuous across frames),
traced as 1 per orthogonal step and sqrt(2) per diagonal step, times facade
mm/px, plus one width (the skeleton stops half a width short of each tip;
without this, small blobs read as 0 mm).

Width: 2 * distance_transform - 1 at each skeleton pixel, measured on each
contributing detection's mask in its own SOURCE frame (finest pixels, single
view). Not on the facade mask: that is a union of several views plus a 15 px
morphological close, which would inflate width. Reported as the median over
views of each view's mean and 95th-percentile width.

Resolution limit: at 5 m a 3840 px-wide frame has ~1.8 mm pixels (the GSD).
A real hairline crack (0.1-1 mm) images as a 1-3 px blurred dark line, so
widths under 2 px (Nyquist: two samples across) cannot be measured and are
reported as an upper bound "< 2 GSD" (width_resolved=False). Above that,
optical blur and the mask edge still bias width upwards, so treat widths as
upper estimates and validate against a crack-width gauge on site. Length is
unaffected (it spans many pixels). Finer width needs closer flight or the
full 8000 px stills (0.89 mm/px at 5 m).

On J17 the widest "cracks" were a black aircon pipe (~35 mm), lettering and
floor-slab joints: implausible width is itself a false-positive signal.

Severity: BRE Digest 251 damage categories, by crack width ("<=" when the
width is only an upper bound).
"""
import sys

import cv2
import numpy as np

WALL_DISTANCE_M = 5.0            # fallback stand-off when there is no usable telemetry
MIN_HEIGHT_FRAMES = 6            # telemetry scale needs this many frames with a height...
MIN_HEIGHT_SPAN_M = 1.0          # ...spread over at least this much climb...
MIN_HEIGHT_R2 = 0.98             # ...and a near-linear fit; else fall back to WALL_DISTANCE_M
CAMERA = {"name": "DJI Mavic Air 2 (FC3170)", "focal_mm": 4.5, "sensor_w_mm": 6.4, "sensor_h_mm": 4.8}
IMPLAUSIBLE_WIDTH_MM = 25.0      # BRE 5 on a facade: almost always a pipe/joint/edge; kept, but flagged
MIN_RESOLVED_WIDTH_PX = 2.0      # Nyquist: narrower than 2 px is below the GSD -> upper bound only

# BRE Digest 251 (damage to low-rise buildings), upper width bound in mm -> category
BRE_CATEGORIES = [(0.1, 0, "negligible (hairline)"), (1.0, 1, "very slight"), (5.0, 2, "slight"),
                  (15.0, 3, "moderate"), (25.0, 4, "severe"), (float("inf"), 5, "very severe")]


def spec_f_px(image_w_px: int) -> float:
    """Spec-sheet focal length in pixels for an image spanning the full sensor width."""
    return CAMERA["focal_mm"] / CAMERA["sensor_w_mm"] * image_w_px


def bre_category(width_mm: float) -> tuple[int, str]:
    for upper, cat, desc in BRE_CATEGORIES:
        if width_mm <= upper:
            return cat, desc
    raise ValueError(width_mm)


def fit_height_scale(geom: dict, heights: dict[str, float]) -> dict | None:
    """Metres per SfM unit from telemetry heights (frame name -> metres). None if unusable."""
    names = [n for n in heights if n in geom["frame_up"]]
    if len(names) < MIN_HEIGHT_FRAMES:
        return None
    x = np.array([geom["frame_up"][n] for n in names])
    y = np.array([heights[n] for n in names])
    if np.ptp(y) < MIN_HEIGHT_SPAN_M:
        return None
    A = np.column_stack([x, np.ones_like(x)])
    (s, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ [s, b]
    r2 = float(1 - res.var() / y.var())
    if s <= 0 or r2 < MIN_HEIGHT_R2:
        return None
    return {"m_per_unit": float(s), "r2": r2, "rms_m": float(res.std()), "n": len(names), "span_m": float(np.ptp(y))}


def scale_from_geom(geom: dict, wall_distance_m: float = WALL_DISTANCE_M,
                    heights: dict[str, float] | None = None) -> dict:
    """mm per pixel for the facade and for every source frame, from ortho.build_facade's geom.
    Telemetry heights (frame name -> m) set the scale when usable; else assume wall_distance_m.
    Plain floats/dicts so it can be stored in the wizard's project state."""
    med = float(np.median(list(geom["frame_dist"].values())))
    fit = fit_height_scale(geom, heights) if heights else None
    if fit:
        mm_per_unit = fit["m_per_unit"] * 1000
        wall_distance_m = med * fit["m_per_unit"]
    else:
        mm_per_unit = wall_distance_m * 1000 / med
    return {
        "source": "telemetry" if fit else "assumed", "fit": fit,
        "wall_distance_m": wall_distance_m,
        "facade_mm_per_px": float(geom["res"] * mm_per_unit),
        "frame_mm_per_px": {nm: float(d * mm_per_unit / geom["frame_f_px"][nm])
                            for nm, d in geom["frame_dist"].items()},
        "frame_distance_m": {nm: float(d * mm_per_unit / 1000) for nm, d in geom["frame_dist"].items()},
    }


def skeleton_length_px(skel: np.ndarray) -> float:
    s = skel.astype(bool)
    orth = (s[:, :-1] & s[:, 1:]).sum() + (s[:-1, :] & s[1:, :]).sum()
    d1 = s[:-1, :-1] & s[1:, 1:] & ~(s[:-1, 1:] | s[1:, :-1])  # diagonal links not already joined
    d2 = s[:-1, 1:] & s[1:, :-1] & ~(s[:-1, :-1] | s[1:, 1:])  # through an orthogonal corner pixel
    return float(orth + np.sqrt(2) * (d1.sum() + d2.sum()))


def _skeleton(mask: np.ndarray) -> np.ndarray:
    from skimage.morphology import skeletonize
    return skeletonize(mask > 0)


def widths_px(mask: np.ndarray) -> np.ndarray:
    """Local full width at every skeleton pixel of a binary mask."""
    m = (mask > 0).astype(np.uint8)
    dt = cv2.distanceTransform(np.pad(m, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    return 2 * dt[_skeleton(m)] - 1


def _polygon_mask(poly: np.ndarray) -> np.ndarray:
    """Rasterise a polygon into a tight local mask (origin at its bbox corner)."""
    pts = np.round(poly).astype(np.int32)
    x0, y0 = pts.min(0)
    pts -= (x0, y0)
    m = np.zeros((pts[:, 1].max() + 1, pts[:, 0].max() + 1), np.uint8)
    cv2.fillPoly(m, [pts.reshape(-1, 1, 2)], 255)
    return m


def quantify_cracks(merged: list[dict], cleaned: np.ndarray, facade_dets: list[dict], scale: dict) -> list[dict]:
    """One row per merged crack (cca.merge_detections_cca output, same `cleaned` mask).
    facade_dets are the detections that were merged (with "frame"/"src_mask" from ortho)."""
    _, labels = cv2.connectedComponents(cleaned, connectivity=8)
    f_mm = scale["facade_mm_per_px"]
    rows = []
    for c in merged:
        x1, y1, x2, y2 = (int(v) for v in c["bbox"])
        comp = (labels[y1:y2, x1:x2] == c["label"]).astype(np.uint8)
        cw = widths_px(comp)
        length_mm = (skeleton_length_px(_skeleton(comp)) + (float(np.median(cw)) if cw.size else 0.0)) * f_mm

        # width from each contributing view's own mask, at that frame's GSD
        means, p95s, means_px = [], [], []
        for d in facade_dets:
            if d.get("src_mask") is None or len(d["src_mask"]) < 3:
                continue
            fx = int(np.clip((d["bbox"][0] + d["bbox"][2]) / 2, 0, labels.shape[1] - 1))
            fy = int(np.clip((d["bbox"][1] + d["bbox"][3]) / 2, 0, labels.shape[0] - 1))
            bx1, by1, bx2, by2 = (int(v) for v in d["bbox"])
            if labels[fy, fx] != c["label"] and not (labels[max(by1, 0):by2, max(bx1, 0):bx2] == c["label"]).any():
                continue
            w = widths_px(_polygon_mask(np.asarray(d["src_mask"])))
            if w.size == 0:
                continue
            mm = scale["frame_mm_per_px"].get(d.get("frame"), f_mm)
            means.append(float(w.mean()) * mm)
            p95s.append(float(np.percentile(w, 95)) * mm)
            means_px.append(float(w.mean()))
        width_mm = float(np.median(means)) if means else float("nan")
        max_width_mm = float(np.median(p95s)) if p95s else float("nan")
        resolved = bool(means_px) and float(np.median(means_px)) >= MIN_RESOLVED_WIDTH_PX
        if means and not resolved:  # report the resolution limit itself as the bound
            gsd = float(np.median([scale["frame_mm_per_px"].get(d.get("frame"), f_mm) for d in facade_dets]))
            width_mm = max(width_mm, MIN_RESOLVED_WIDTH_PX * gsd)
        cat, desc = bre_category(max_width_mm) if p95s else (None, "unknown")
        rows.append({
            "length_mm": round(length_mm, 1),
            "width_mm": round(width_mm, 1),
            "max_width_mm": round(max_width_mm, 1),
            "width_resolved": resolved,
            "area_mm2": round(c["area_px"] * f_mm ** 2, 1),
            "bre_category": cat, "bre_desc": desc, "n_views": len(means),
        })
    return rows


def review_order(found: list[tuple[dict, dict, str]]) -> list[tuple[dict, dict, str]]:
    """Review order, shared by the table and the labelled image so crack #N means the same
    crack in both: on-wall before off-wall, plausible widths before "verify" ones, then
    confidence tier (likely-real first), then widest first. Widest-first alone put window
    frames, pipes and slab joints at #1-#5 on J17."""
    from cca import tier
    rank = {"high": 0, "medium": 1, "low": 2}
    return sorted(found, key=lambda x: (x[2] != "wall", bool(x[1]["max_width_mm"] > IMPLAUSIBLE_WIDTH_MM),
                                        rank[tier(x[0]["score"])], -np.nan_to_num(x[1]["max_width_mm"]),
                                        -x[0]["score"]))


def crack_table(found: list[tuple[dict, dict, str]]) -> list[dict]:
    """found: (merged crack, quantify row, location) -> one display row per crack, in review_order."""
    from cca import tier
    return [{"crack #": i + 1, "location": loc, "tier": tier(c["score"]), "score": round(c["score"], 2),
             "length (mm)": r["length_mm"],
             # text columns: "<" bounds and "n/a" sit beside numbers (keeps the table Arrow/CSV-safe)
             "width (mm)": "n/a" if not r["n_views"] else str(r["width_mm"]) if r["width_resolved"]
             else f"< {r['width_mm']}",
             "max width (mm)": "n/a" if not r["n_views"] else str(r["max_width_mm"]),
             "BRE cat": "n/a" if not r["n_views"] else str(r["bre_category"]) if r["width_resolved"]
             else f"<= {r['bre_category']}",
             "BRE": "n/a" if not r["n_views"] else r["bre_desc"] if r["width_resolved"] else f"at most {r['bre_desc']}",
             "views": r["n_views"],
             "note": "verify: unusually wide, may be a pipe/joint/edge" if r["max_width_mm"] > IMPLAUSIBLE_WIDTH_MM
             else "",
             "bbox": tuple(round(v) for v in c["bbox"])}
            for i, (c, r, loc) in enumerate(review_order(found))]


def scale_note(scale: dict) -> str:
    """One sentence on where the mm scale came from, for logs and the report."""
    fit = scale.get("fit")
    if fit:
        return (f"Drone-to-wall distance {scale['wall_distance_m']:.2f} m, measured from the drone's altimeter "
                f"(R\u00b2 {fit['r2']:.4f}, {fit['n']} frames over {fit['span_m']:.1f} m of climb, "
                f"{fit['rms_m'] * 100:.0f} cm residual).")
    return (f"Drone-to-wall distance ASSUMED to be {scale['wall_distance_m']:.2f} m (no usable altimeter data); "
            f"all sizes scale linearly with this value.")


def quantify_and_report(merged_wall, cleaned_wall, dets_wall, merged_off, cleaned_off, dets_off, geom,
                        image, out_dir, heights: dict[str, float] | None = None, title: str = "Facade crack report"):
    """Quantify on-wall + off-wall cracks and write cracks.csv, facade_labelled.jpg, report.html."""
    from report import write_report
    scale = scale_from_geom(geom, heights=heights)
    sfm_f = float(np.median(list(geom["frame_f_px"].values())))
    print(f"[quantify] {scale_note(scale)} Facade {scale['facade_mm_per_px']:.2f} mm/px, frames median "
          f"{np.median(list(scale['frame_mm_per_px'].values())):.2f} mm/px; SfM focal {sfm_f:.0f} px vs "
          f"{CAMERA['name']} spec {spec_f_px(3840):.0f} px at 3840 px wide", file=sys.stderr)
    found = [(c, r, "wall") for c, r in zip(merged_wall, quantify_cracks(merged_wall, cleaned_wall, dets_wall, scale))]
    found += [(c, r, "off-wall") for c, r in zip(merged_off, quantify_cracks(merged_off, cleaned_off, dets_off, scale))]
    out = write_report(out_dir, image, found, scale, title)
    print(f"[quantify] {len(out['table'])} cracks -> {out['html']}", file=sys.stderr)
    return out


def _self_check() -> None:
    """ponytail: synthetic cracks with known size -> length, width, scale and BRE must come back."""
    # straight band 5 px wide, 200 px long
    m = np.zeros((40, 260), np.uint8)
    m[18:23, 30:230] = 255
    w = widths_px(m)
    assert abs(np.median(w) - 5) <= 0.5, np.median(w)
    L = skeleton_length_px(_skeleton(m))
    assert 190 <= L <= 201, L  # tips lose ~half a width each
    # thin (1 px) line must be flagged as below resolution
    tc = np.zeros((40, 200), np.uint8); tc[20, 20:180] = 255
    trow = quantify_cracks([{"bbox": (20.0, 20.0, 180.0, 21.0), "label": 1, "area_px": 160}], tc,
                           [{"bbox": (20, 20, 180, 21), "frame": "b",
                             "src_mask": np.array([[20, 20], [179, 20], [179, 20.4], [20, 20.4]], np.float32)}],
                           {"facade_mm_per_px": 2.0, "frame_mm_per_px": {"b": 2.0}})[0]
    assert not trow["width_resolved"] and trow["width_mm"] >= 4.0, trow  # bound = 2 px * 2 mm
    # 45-degree line: length must use sqrt(2) per step, not 1
    d = np.zeros((120, 120), np.uint8)
    cv2.line(d, (10, 10), (110, 110), 255, 3)
    Ld = skeleton_length_px(_skeleton(d))
    assert abs(Ld - 100 * np.sqrt(2)) < 8, Ld
    # scale: median camera distance 2.0 units -> 5 m, so 1 unit = 2500 mm
    geom = {"res": 0.001, "frame_dist": {"a": 1.8, "b": 2.0, "c": 2.2}, "frame_f_px": {"a": 2700, "b": 2700, "c": 2700},
            "frame_up": {"a": 0.0, "b": 1.0, "c": 2.0}}
    s = scale_from_geom(geom, 5.0)
    assert abs(s["facade_mm_per_px"] - 2.5) < 1e-9 and abs(s["frame_mm_per_px"]["b"] - 5000 / 2700) < 1e-9
    assert abs(s["frame_distance_m"]["a"] - 4.5) < 1e-9
    # telemetry scale: heights = 2.5 m per unit along "up" -> wall distance = 2.0 units * 2.5 = 5 m
    rng = np.random.default_rng(0)
    up = {f"f{i}": float(u) for i, u in enumerate(np.linspace(0, 4, 20))}
    g2 = {"res": 0.001, "frame_up": up, "frame_dist": {k: 2.0 for k in up}, "frame_f_px": {k: 2700.0 for k in up}}
    hts = {k: 1.0 + 2.5 * u + rng.normal(0, 0.03) for k, u in up.items()}
    s2 = scale_from_geom(g2, heights=hts)
    assert s2["source"] == "telemetry" and abs(s2["wall_distance_m"] - 5.0) < 0.05, s2["wall_distance_m"]
    flat = {k: 3.0 for k in up}  # no climb -> unusable -> falls back to the assumed distance
    assert scale_from_geom(g2, 4.0, heights=flat)["source"] == "assumed"
    assert scale_from_geom(g2, 4.0, heights=flat)["wall_distance_m"] == 4.0
    # spec sheet: 3840 px wide -> 2700 px focal -> 1.85 mm/px at 5 m
    assert abs(spec_f_px(3840) - 2700) < 1e-9
    assert bre_category(0.05)[0] == 0 and bre_category(3)[0] == 2 and bre_category(30)[0] == 5
    # end to end: one crack, one view, facade == frame scale
    cleaned = np.zeros((60, 300), np.uint8)
    cleaned[28:33, 40:240] = 255
    merged = [{"bbox": (40.0, 28.0, 240.0, 33.0), "label": 1, "area_px": int((cleaned > 0).sum())}]
    poly = np.array([[40, 28], [239, 28], [239, 32], [40, 32]], np.float32)
    dets = [{"bbox": (40, 28, 240, 33), "frame": "b", "src_mask": poly}]
    row = quantify_cracks(merged, cleaned, dets, {"facade_mm_per_px": 2.0, "frame_mm_per_px": {"b": 2.0}})[0]
    assert 390 <= row["length_mm"] <= 412 and abs(row["width_mm"] - 10) <= 1 and row["n_views"] == 1, row
    assert row["width_resolved"] and row["bre_category"] == 3, row  # 10 mm: moderate (5-15)
    print("[quantify] self-check OK", file=sys.stderr)


if __name__ == "__main__":
    _self_check()
