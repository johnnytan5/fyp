"""Still-photo entry point: a folder of DJI hover-and-shoot JPGs (manual or
waypoint grid) -> rows grouped by the drone's recorded height -> the same
detect -> horizontal stitch -> vertical stitch chain as run_pipeline.py.

No video, no SRT: every DJI photo carries its own RelativeAltitude in XMP.
"""
import pickle
import re
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))  # for infer.py
from detect import detect_cracks  # noqa: E402
from cca import draw_merged, merge_detections_cca  # noqa: E402
from infer import CONF_THRESHOLD, load_sahi_model  # noqa: E402
from ortho import build_facade  # noqa: E402
from stitch_horizontal import draw_detections, stitch_horizontal  # noqa: E402
from stitch_vertical import stitch_vertical  # noqa: E402

ROW_GAP_M = 0.5        # height change that starts a new row (rows are ~1-1.5 m apart)
MAX_WIDTH_PX = 3840    # ponytail: 48 MP stills downscaled to 4K-frame width; full res OOMs the warp
_ALT_RE = re.compile(rb'drone-dji:RelativeAltitude="([+-]?[\d.]+)"')


def relative_altitude(path: Path) -> float:
    m = _ALT_RE.search(path.read_bytes()[:300_000])
    if not m:
        raise RuntimeError(f"no DJI RelativeAltitude in {path.name}")
    return float(m.group(1))


def group_rows(heights: list[float], gap_m: float = ROW_GAP_M) -> list[list[int]]:
    """Consecutive (capture-order) photos within gap_m of the row's first photo
    share a row. Returns index lists, bottom row first."""
    rows: list[list[int]] = []
    for i, h in enumerate(heights):
        if rows and abs(h - heights[rows[-1][0]]) <= gap_m:
            rows[-1].append(i)
        else:
            rows.append([i])
    return sorted(rows, key=lambda r: heights[r[0]])


def run_stills(photo_dir: Path, work_dir: Path, conf_threshold: float = CONF_THRESHOLD,
               stitch: str = "ortho") -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    photos = sorted(photo_dir.glob("*.JPG")) + sorted(photo_dir.glob("*.jpg"))
    heights = [relative_altitude(p) for p in photos]
    rows = group_rows(heights)
    model = load_sahi_model(conf=conf_threshold)

    panos, pano_dets = [], []
    all_frames, all_dets = [], []
    for ri, idxs in enumerate(rows):
        row_dir = work_dir / f"frames_row{ri}"
        row_dir.mkdir(exist_ok=True)
        frame_paths, dets_per_frame = [], []
        for i in idxs:
            out = row_dir / photos[i].name
            cache = out.with_suffix(".dets.pkl")  # reruns skip detection (same as run_pipeline)
            frame_paths.append(out)
            if out.exists() and cache.exists():
                dets_per_frame.append(pickle.loads(cache.read_bytes()))
                continue
            img = cv2.imread(str(photos[i]))
            s = min(1.0, MAX_WIDTH_PX / img.shape[1])
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            dets = detect_cracks(img, model)
            cache.write_bytes(pickle.dumps(dets))
            dets_per_frame.append(dets)

        if stitch == "ortho":  # one facade from all rows, built after the loop
            all_frames += frame_paths
            all_dets += dets_per_frame
            print(f"[run_stills] row {ri} ({heights[idxs[0]]:.1f}m): {len(idxs)} photos, "
                  f"{sum(map(len, dets_per_frame))} raw detections", file=sys.stderr)
            continue

        try:
            stitched = stitch_horizontal(frame_paths, dets_per_frame)
        except RuntimeError as e:
            print(f"[run_stills] row {ri} ({heights[idxs[0]]:.1f}m): SKIPPED -- {e}", file=sys.stderr)
            continue
        panos.append(stitched["panorama"])
        pano_dets.append(stitched["detections"])
        cv2.imwrite(str(work_dir / f"row_{ri:02d}_annotated.jpg"),
                    draw_detections(stitched["panorama"], stitched["detections"]))
        print(f"[run_stills] row {ri} ({heights[idxs[0]]:.1f}m): {len(idxs)} photos, "
              f"{len(stitched['detections'])} detections, dropped {stitched['dropped']}", file=sys.stderr)

    if stitch == "ortho":
        o = build_facade(work_dir, [str(f.relative_to(work_dir)) for f in all_frames], all_dets, work_dir)
        facade, dets = o["image"], o["detections"]
        cv2.imwrite(str(work_dir / "facade_ortho.jpg"), facade)
    else:
        v = stitch_vertical(panos, pano_dets, work_dir / "vertical_rotate_tmp")
        facade, dets = v["panorama"], v["detections"]
        print(f"[run_stills] dropped rows {v['dropped']}", file=sys.stderr)
    cv2.imwrite(str(work_dir / "facade_annotated.jpg"), draw_detections(facade, dets))
    merged, _ = merge_detections_cca(dets, facade.shape)
    cv2.imwrite(str(work_dir / "facade_cca.jpg"), draw_merged(facade, merged))
    print(f"[run_stills] facade: {facade.shape}, CCA: {len(dets)} detections -> {len(merged)} cracks "
          f"-> {work_dir}", file=sys.stderr)


def _self_check() -> None:
    # S-pattern: row changes mark on height jumps; small baro jitter stays in-row
    heights = [1.7, 1.6, 1.7, 2.8, 2.8, 2.9, 4.4, 4.4]
    assert group_rows(heights) == [[0, 1, 2], [3, 4, 5], [6, 7]], group_rows(heights)
    print("[run_stills] self-check OK", file=sys.stderr)


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-check"]:
        _self_check()
    else:
        import argparse
        ap = argparse.ArgumentParser()
        ap.add_argument("photo_dir", type=Path)
        ap.add_argument("--work-dir", type=Path, required=True)
        ap.add_argument("--stitch", choices=["ortho", "panorama"], default="ortho")
        a = ap.parse_args()
        run_stills(a.photo_dir, a.work_dir, stitch=a.stitch)
