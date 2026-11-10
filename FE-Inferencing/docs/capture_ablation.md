# Capture Ablation — Drone Footage → Facade Stitch

Living reference for how the **capture method** affects the stitched facade.
The question: is the current stitch quality limited by *how the footage is
captured* (blur, exposure, row spacing) rather than by the stitching code?
And would a Litchi waypoint mission (paid app, grant line item) actually help?

Fill in each result table after its run. Keep everything in §2 identical
across runs so only the capture variable changes.

---

## 1. Baseline (run B0) — existing footage, `DJI_0649.MP4`

| Item | Value |
|---|---|
| Date flown | 2026-07-12 |
| Building / wall | J17 (Jabatan Fizik), main facade |
| Drone | DJI Mini 2 (confirm) |
| Capture | Manual S-pattern, continuous 4K video, **auto exposure** |
| Code version | `stitch_horizontal.py` degenerate-camera filter + `VERTICAL_CONF_THRESH = 0.6` (2026-10-16) |
| Output | `~/Desktop/J17_facade_stitch/` |

Per-level results (row heights from SRT `H`):

| Level | Height (m) | Step from prev (m) | Frames | Dropped frames | Detections |
|---|---|---|---|---|---|
| 0 | 1.30 | – | 15 | 0, 1 | 29 |
| 1 | 3.00 | 1.70 | 13 | – | 6 |
| 2 | 3.60 | 0.60 | 15 | 0 (degenerate camera) | 12 |
| 3 | 4.50 | 0.90 | 15 | – | 8 |
| 4 | 5.40 | 0.90 | 15 | – | 6 |
| 5 | 6.30 | 0.90 | 14 | – | 9 |
| 6 | 7.10 | 0.80 | 14 | 0 | 2 |
| 7 | 8.30 | 1.20 | 12 | – | 5 |
| 8 | 9.30 | 1.00 | 12 | – | 9 |
| 9 | 11.90 | **2.60** | 13 | 10, 11, 12 | 21 |
| 10 | 12.90 | 1.00 | 13 | 0 | 6 |
| 11 | 14.30 | 1.40 | 14 | – | 51 |

Facade: **7 / 12 levels joined** (0–6), 9674 × 5569 px, 72 detections.

Vertical match confidence between neighbouring levels (inliers in brackets),
at the default work resolution — threshold for joining is 0.6:

| 0-1 | 1-2 | 2-3 | 3-4 | 4-5 | 5-6 | 6-7 | 7-8 | 8-9 | 9-10 | 10-11 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1.18 (78) | 1.14 (110) | 0.87 (75) | 1.12 (89) | 1.05 (81) | 0.87 (47) | 0.42 (20) | 0.52 (25) | **0.13 (4)** | 0.45 (17) | 0.62 (25) |

Observed problems:
1. **Upper levels 7–11 not joined.** Plain painted wall → few features; plus
   the 2.6 m climb between levels 8→9 left almost no overlap.
2. **Patchy diagonal light/dark blocks.** Auto exposure shifts between frames;
   no blending (deliberate — blending smears crack texture).
3. **Dropped frames at row starts.** Near-zero motion at the start of a pass.
4. **False-positive detections** on trees and the black canvas border
   (model/filtering issue — not a capture variable, out of scope here).

Rejected tuning: `VERTICAL_CONF_THRESH = 0.4` joins all 12 levels but tilts and
misaligns the upper floors. Lowering thresholds can't fix weak captures.

---

## 2. Fixed controls (same for every run)

| Control | Setting |
|---|---|
| Drone | DJI Mavic Air 2 |
| Wall | J17, same facade as B0 |
| Standoff from wall | ___ m (≥ 5 m) — measure with laser meter, keep identical |
| Time of day | ___ (same light for both runs, ideally back-to-back) |
| Row spacing (vertical step) | ~1.0 m, never skip a row |
| Lowest row | 2–3 m |
| Gimbal pitch | 0° (level) |
| Heading | Perpendicular to wall, fixed |
| White balance | Locked (manual) |
| Weather | Note wind; abort if gusty |

---

## 3. Experiment E1 — video, locked exposure, slower flight

**Hypothesis:** locked exposure removes the patchy blocks; faster shutter +
slower flight reduces motion blur and rolling-shutter skew → more matches
on plain wall → more levels join.

| Setting | Value |
|---|---|
| Mode | 4K video |
| Exposure | Manual: shutter ≥ 1/500 s, ISO set for correct brightness |
| Shutter / ISO / fps used | ___ / ___ / ___ |
| Flight speed | 0.5–1 m/s |
| Row start | Hover 2–3 s before moving |
| Pipeline | `pipeline/run_pipeline.py <video> --work-dir ...` (unchanged) |

Results:

| Metric | B0 | E1 |
|---|---|---|
| Levels joined / total | 7 / 12 | |
| Weakest neighbour-level confidence | 0.13 | |
| Total dropped frames | 8 | |
| Visible exposure blocks (Y/N) | Y | |
| Misalignment / tilt (Y/N) | N | |
| Detections on final facade | 72 | |
| Notes | | |

---

## 4. Experiment E2 — manual stills, mimicking a waypoint grid

**Hypothesis:** hovering stills remove blur and give even, known overlap —
the main benefit a Litchi mission would provide. If E2 doesn't beat E1,
Litchi isn't worth buying.

| Setting | Value |
|---|---|
| Mode | Photo, 12 MP |
| Exposure | Manual, same as E1 |
| Horizontal overlap | ~70% between consecutive photos |
| Vertical overlap | ~60% between rows |
| Frame footprint at standoff | ___ m wide × ___ m tall (measure before flying) |
| Horizontal step | ___ m (= 0.3 × footprint width) |
| Capture | Settle hover → shoot → move; heading fixed |
| Row marking | Photograph hand-written row number before each row |
| Pipeline | Needs image-folder entry point (one folder per row) — not built yet |

Results:

| Metric | B0 | E1 | E2 |
|---|---|---|---|
| Levels joined / total | 7 / 12 | | |
| Weakest neighbour-level confidence | 0.13 | | |
| Total dropped frames | 8 | | |
| Visible exposure blocks (Y/N) | Y | | |
| Misalignment / tilt (Y/N) | N | | |
| Detections on final facade | 72 | | |
| Flight time | – | | |
| Batteries used | – | | |
| Notes | | | |

---

## 5. Decision rule

- **E1 ≈ E2, both good** → keep video capture; skip Litchi.
- **E2 clearly better than E1** → stills matter; Litchi automates E2 → justify it in the budget.
- **Both still fail on upper rows** → the plain wall is the limit, not capture.
  Next step is code: place rows using known grid spacing where image matching fails.
