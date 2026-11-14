"""Module 1 showcase: drone footage -> sampled keyframes -> 2D full-facade image.

Stitching only: no crack detection, so the detection model is never loaded.
Video: altitude telemetry splits the flight into levels, frames are sampled
every STEP_M of travel. Photos: grouped into rows by altitude, downscaled to
the same 3840 px width the pipeline uses. Both then go through ortho.py
(SfM + wall-plane orthomosaic). Results are cached per input under
projects/module1/, so a rerun or a screenshot session is instant.
"""
import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT))

from ortho import build_facade  # noqa: E402
from quantify import scale_from_geom  # noqa: E402
from run_stills import MAX_WIDTH_PX, group_rows, relative_altitude  # noqa: E402
from sampling import STEP_M, extract_frames, sample_frames  # noqa: E402
from srt_levels import Sample, detect_levels, extract_telemetry_srt, parse_telemetry, split_video_into_levels  # noqa: E402

THUMB_W = 360  # px, sampled-frame thumbnails shown in the grid

st.set_page_config(page_title="Module 1 · Facade Stitching", layout="wide")
st.title("Module 1 · 2D Facade Stitching")
st.caption("Drone footage → keyframe sampling → structure-from-motion → wall-plane orthomosaic. "
           "No crack detection on this page.")


def _local_path(raw: str) -> Path | None:
    # ponytail: same as app_pipeline._local_path (importing that would run the wizard)
    raw = raw.strip().strip("'\"").replace("\\ ", " ")
    return Path(raw).expanduser() if raw else None


def _work_dir(src: Path) -> Path:
    tag = hashlib.sha1(str(src.resolve()).encode()).hexdigest()[:8]
    d = ROOT / "projects" / "module1" / f"{src.stem}_{tag}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _thumb(path: Path) -> str:
    t = path.with_name(path.stem + ".thumb.jpg")
    if not t.exists():
        img = cv2.imread(str(path))
        s = THUMB_W / img.shape[1]
        cv2.imwrite(str(t), cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA))
    return str(t)


def sample_video(video: Path, srt: Path | None, step_m: float, wd: Path) -> list[dict]:
    samples = parse_telemetry(srt.read_text() if srt else extract_telemetry_srt(video))
    levels = detect_levels(samples)
    clips = split_video_into_levels(video, levels, wd / "levels")
    groups = []
    for lv, clip in zip(levels, clips):
        rel = [Sample(s.t - lv.start_s, s.h, s.h_speed, s.v_speed) for s in samples if lv.start_s <= s.t <= lv.end_s]
        paths = extract_frames(clip, sample_frames(rel, step_m=step_m), wd / f"frames_level{lv.index}")
        ts, hs = [s.t for s in rel], [s.h for s in rel]
        frames = []
        for p in paths:
            t = float(p.stem.rsplit("_t", 1)[1])  # extract_frames names f<i>_t<seconds>.jpg
            frames.append({"path": str(p), "h": float(np.interp(t, ts, hs)), "caption": f"t = {lv.start_s + t:.1f} s"})
        groups.append({"label": f"Level {lv.index} · {lv.height_m:.1f} m", "frames": frames})
    return groups


def sample_photos(folder: Path, wd: Path) -> list[dict]:
    photos = sorted(folder.glob("*.JPG")) + sorted(folder.glob("*.jpg"))
    heights = [relative_altitude(p) for p in photos]
    groups = []
    for ri, idxs in enumerate(group_rows(heights)):
        row_dir = wd / f"frames_row{ri}"
        row_dir.mkdir(exist_ok=True)
        frames = []
        for i in idxs:
            out = row_dir / photos[i].name
            if not out.exists():
                img = cv2.imread(str(photos[i]))
                s = min(1.0, MAX_WIDTH_PX / img.shape[1])
                cv2.imwrite(str(out), cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA),
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
            frames.append({"path": str(out), "h": heights[i], "caption": photos[i].name})
        groups.append({"label": f"Row {ri} · {heights[idxs[0]]:.1f} m", "frames": frames})
    return groups


# ---------- 1. Input ----------
st.header("1 · Drone footage")
kind = st.radio("Input", ["Drone video", "Drone photos (folder)"], horizontal=True, key="m1_kind")
src = _local_path(st.text_input("Video file" if kind == "Drone video" else "Photo folder",
                                key="m1_src", placeholder="drag the file or folder into this box"))
srt = None
if kind == "Drone video":
    srt = _local_path(st.text_input("Telemetry .srt (optional: only if the video has no embedded telemetry)",
                                    key="m1_srt"))
    step_m = st.slider("Sample a keyframe every … metres of travel", 0.25, 3.0, float(STEP_M), 0.25)

ok = src is not None and (src.is_file() if kind == "Drone video" else src.is_dir()) and (srt is None or srt.is_file())
if src and not ok:
    st.error("Path not found, or not a " + ("video file" if kind == "Drone video" else "folder"))

if ok and st.button("Sample keyframes", type="primary"):
    wd = _work_dir(src)
    with st.spinner("Splitting into levels and sampling keyframes..." if kind == "Drone video"
                    else "Grouping photos into rows by altitude..."):
        groups = sample_video(src, srt, step_m, wd) if kind == "Drone video" else sample_photos(src, wd)
    st.session_state.update(m1_groups=groups, m1_wd=str(wd), m1_facade=None)

# ---------- 2. Sampled keyframes ----------
groups = st.session_state.get("m1_groups")
if groups:
    st.header("2 · Sampled keyframes")
    n = sum(len(g["frames"]) for g in groups)
    a, b, c = st.columns(3)
    a.metric("Levels / rows", len(groups))
    b.metric("Keyframes", n)
    c.metric("Height covered", f"{min(f['h'] for g in groups for f in g['frames']):.1f} – "
                               f"{max(f['h'] for g in groups for f in g['frames']):.1f} m")
    for gi, g in enumerate(groups):
        with st.expander(f"{g['label']} · {len(g['frames'])} keyframes", expanded=gi == 0):
            cols = st.columns(5)
            for i, f in enumerate(g["frames"]):
                cols[i % 5].image(_thumb(Path(f["path"])), caption=f["caption"], use_container_width=True)

    # ---------- 3. Facade ----------
    st.header("3 · Stitched facade")
    if st.button("Build facade", type="primary"):
        wd = Path(st.session_state["m1_wd"])
        frames = [f for g in groups for f in g["frames"]]
        names = [str(Path(f["path"]).relative_to(wd)) for f in frames]
        with st.spinner(f"Reconstructing {len(names)} keyframes in 3D and projecting onto the wall "
                        f"(a few minutes the first time, cached after)..."):
            o = build_facade(wd, names, [[] for _ in names], wd)
        out = wd / "facade_ortho.jpg"
        cv2.imwrite(str(out), o["image"])
        scale = scale_from_geom(o["geom"], heights={nm: f["h"] for nm, f in zip(names, frames)})
        st.session_state["m1_facade"] = {"path": str(out), "stats": o["stats"], "unregistered": o["unregistered"],
                                         "mm_per_px": scale["facade_mm_per_px"],
                                         "distance_m": scale["wall_distance_m"], "measured": bool(scale["fit"])}

    fac = st.session_state.get("m1_facade")
    if fac:
        s = fac["stats"]
        w, h = s["facade_px"]
        m = st.columns(5)
        m[0].metric("Keyframes placed", f"{s['registered']} / {s['total']}")
        m[1].metric("3D points", f"{s['points3D']:,}")
        m[2].metric("Reprojection error", f"{s['reproj_err_px']:.2f} px")
        m[3].metric("Facade size", f"{w * fac['mm_per_px'] / 1000:.1f} × {h * fac['mm_per_px'] / 1000:.1f} m",
                    f"{w} × {h} px", delta_color="off")
        m[4].metric("Drone-to-wall distance", f"{fac['distance_m']:.2f} m",
                    "measured (altimeter)" if fac["measured"] else "assumed", delta_color="off")
        if fac["unregistered"]:
            st.warning(f"{len(fac['unregistered'])} keyframe(s) could not be placed: {fac['unregistered']}")
        st.image(fac["path"], use_container_width=True,
                 caption="Full-facade orthomosaic: every keyframe projected onto the fitted wall plane; "
                         "each pixel taken from the most head-on keyframe.")
        st.download_button("⬇ Facade image (JPG)", Path(fac["path"]).read_bytes(), "facade.jpg", "image/jpeg")
