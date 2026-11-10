"""Multi-stage drone-facade crack pipeline: level split -> sampling ->
inference -> stitching -> CCA merge, as a 5-tab Streamlit wizard.

State persists per-project via pipeline/project_store.py (local SQLite +
files on disk), so a project can be closed and reopened later without
re-running any already-completed stage.
"""
import sys
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "pipeline"))
sys.path.insert(0, str(Path(__file__).parent))

import pipeline.project_store as store  # noqa: E402
from cca import draw_merged, merge_detections_cca  # noqa: E402
from detect import detect_cracks  # noqa: E402
from infer import CONF_THRESHOLD, load_sahi_model  # noqa: E402
from ortho import build_facade  # noqa: E402
from sampling import DWELL_MIN_SECONDS, DWELL_VELOCITY_EPS, STEP_M, extract_frames, sample_frames  # noqa: E402
from srt_levels import Sample, detect_levels, extract_telemetry_srt, parse_telemetry, split_video_into_levels  # noqa: E402
from stitch_horizontal import draw_detections, stitch_horizontal  # noqa: E402
from stitch_vertical import stitch_vertical  # noqa: E402

st.set_page_config(page_title="Crack Pipeline", layout="wide")
st.title("🏗️ Drone Facade Crack Pipeline")

STATE_KEYS = [
    "video_path", "sampling_mode", "top_to_bottom", "step_m", "dwell_velocity_eps",
    "dwell_min_seconds", "conf_threshold", "levels", "frame_paths_per_level",
    "detections_per_frame", "level_pano_paths", "level_pano_detections",
    "level_dropped_frames", "facade_path", "facade_detections", "dropped_levels",
    "cca_close_px", "cca_open_px", "cca_merged", "stitch_method", "ortho_stats",
]


def _persist(project_name: str) -> None:
    state = {k: st.session_state.get(k) for k in STATE_KEYS}
    store.save_state(project_name, state)


def _hydrate(loaded: dict) -> None:
    for k in STATE_KEYS:
        if k in loaded:
            st.session_state[k] = loaded[k]


# ---------- Sidebar: project selection ----------
with st.sidebar:
    st.subheader("Project")
    projects = store.list_projects()
    project_names = [p["name"] for p in projects]
    choice = st.selectbox("Open project", ["<new project>"] + project_names)

    if choice == "<new project>":
        new_name = st.text_input("New project name")
        if st.button("Create project") and new_name:
            store.create_project(new_name)
            st.session_state.clear()
            st.session_state["project_name"] = new_name
            st.rerun()
    elif st.session_state.get("project_name") != choice:
        loaded = store.load_state(choice) or {}
        st.session_state.clear()
        st.session_state["project_name"] = choice
        _hydrate(loaded)
        st.rerun()

    if "project_name" in st.session_state:
        st.success(f"Project: {st.session_state['project_name']}")
        if st.button("🗑️ Delete this project"):
            store.delete_project(st.session_state["project_name"])
            st.session_state.clear()
            st.rerun()

if "project_name" not in st.session_state:
    st.info("Create or select a project in the sidebar to begin.")
    st.stop()

project_name = st.session_state["project_name"]
work_dir = store.project_work_dir(project_name)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["1. Level Split", "2. Sampling", "3. Inference", "4. Stitching", "5. CCA & Findings"]
)

# ---------- Tab 1: Level split ----------
with tab1:
    st.header("Split video into levels")
    uploaded = st.file_uploader("Upload drone video (.mp4)", type=["mp4", "mov"])
    top_to_bottom = st.checkbox("Flight order: top to bottom",
                                 value=st.session_state.get("top_to_bottom", False))

    if uploaded and st.button("Split into levels", key="btn_split"):
        video_path = work_dir / "input_video.mp4"
        video_path.write_bytes(uploaded.getvalue())
        with st.spinner("Parsing telemetry and splitting..."):
            srt_text = extract_telemetry_srt(video_path)
            samples = parse_telemetry(srt_text)
            levels_obj = detect_levels(samples, top_to_bottom=top_to_bottom)
            level_video_paths = split_video_into_levels(video_path, levels_obj, work_dir / "levels")
        st.session_state["video_path"] = str(video_path)
        st.session_state["top_to_bottom"] = top_to_bottom
        st.session_state["levels"] = [
            {"index": lv.index, "start_s": lv.start_s, "end_s": lv.end_s, "height_m": lv.height_m,
             "video_path": str(p)}
            for lv, p in zip(levels_obj, level_video_paths)
        ]
        _persist(project_name)
        st.rerun()

    if st.session_state.get("levels"):
        st.success(f"{len(st.session_state['levels'])} levels detected")
        for lv in st.session_state["levels"]:
            st.write(f"**Level {lv['index']}** — height {lv['height_m']:.2f}m "
                      f"({lv['start_s']:.1f}s–{lv['end_s']:.1f}s)")
            st.video(lv["video_path"])

# ---------- Tab 2: Sampling ----------
with tab2:
    st.header("Sample frames from each level")
    if not st.session_state.get("levels"):
        st.info("Complete level splitting first.")
    else:
        mode = st.radio("Sampling mode", ["dynamic", "dwell"],
                         index=0 if st.session_state.get("sampling_mode", "dynamic") == "dynamic" else 1,
                         help="dynamic: sample by distance traveled (continuous flight). "
                              "dwell: sample from near-zero-speed hover pauses (waypoint mission).")
        if mode == "dynamic":
            step_m = st.slider("Step distance (m)", 0.2, 3.0, st.session_state.get("step_m", STEP_M), 0.1)
            vel_eps = st.session_state.get("dwell_velocity_eps", DWELL_VELOCITY_EPS)
            min_dwell = st.session_state.get("dwell_min_seconds", DWELL_MIN_SECONDS)
        else:
            vel_eps = st.slider("Dwell velocity threshold (m/s)", 0.05, 0.5,
                                 st.session_state.get("dwell_velocity_eps", DWELL_VELOCITY_EPS), 0.01)
            min_dwell = st.slider("Min dwell duration (s)", 0.2, 2.0,
                                   st.session_state.get("dwell_min_seconds", DWELL_MIN_SECONDS), 0.1)
            step_m = st.session_state.get("step_m", STEP_M)

        if st.button("Sample frames", key="btn_sample"):
            srt_text = extract_telemetry_srt(Path(st.session_state["video_path"]))
            samples = parse_telemetry(srt_text)
            frame_paths_per_level = {}
            with st.spinner("Extracting frames..."):
                for lv in st.session_state["levels"]:
                    level_samples = [s for s in samples if lv["start_s"] <= s.t <= lv["end_s"]]
                    rel_samples = [Sample(s.t - lv["start_s"], s.h, s.h_speed, s.v_speed)
                                   for s in level_samples]
                    timestamps = sample_frames(rel_samples, mode=mode, step_m=step_m,
                                                velocity_eps=vel_eps, min_dwell_s=min_dwell)
                    frame_paths = extract_frames(Path(lv["video_path"]), timestamps,
                                                  work_dir / f"frames_level{lv['index']}")
                    frame_paths_per_level[str(lv["index"])] = [str(p) for p in frame_paths]
            st.session_state["sampling_mode"] = mode
            st.session_state["step_m"] = step_m
            st.session_state["dwell_velocity_eps"] = vel_eps
            st.session_state["dwell_min_seconds"] = min_dwell
            st.session_state["frame_paths_per_level"] = frame_paths_per_level
            _persist(project_name)
            st.rerun()

        if st.session_state.get("frame_paths_per_level"):
            level_choice = st.selectbox("View level", [lv["index"] for lv in st.session_state["levels"]],
                                         key="sample_view_level")
            paths = st.session_state["frame_paths_per_level"].get(str(level_choice), [])
            st.write(f"{len(paths)} frames sampled")
            cols = st.columns(5)
            for i, p in enumerate(paths):
                cols[i % 5].image(p, use_container_width=True)

# ---------- Tab 3: Inference ----------
with tab3:
    st.header("Run crack detection")
    if not st.session_state.get("frame_paths_per_level"):
        st.info("Complete sampling first.")
    else:
        conf = st.slider("Confidence threshold", 0.05, 0.95,
                          st.session_state.get("conf_threshold", CONF_THRESHOLD), 0.05)
        if st.button("Run inference", key="btn_infer"):
            model = load_sahi_model(conf=conf)
            detections_per_frame = {}
            with st.spinner("Running detection (SAHI-tiled)..."):
                for level_idx_str, paths in st.session_state["frame_paths_per_level"].items():
                    dets = []
                    for p in paths:
                        img = cv2.imread(p)
                        dets.append(detect_cracks(img, model) if img is not None else [])
                    detections_per_frame[level_idx_str] = dets
            st.session_state["conf_threshold"] = conf
            st.session_state["detections_per_frame"] = detections_per_frame
            _persist(project_name)
            st.rerun()

        if st.session_state.get("detections_per_frame"):
            level_choice = st.selectbox("View level", [lv["index"] for lv in st.session_state["levels"]],
                                         key="infer_view_level")
            paths = st.session_state["frame_paths_per_level"].get(str(level_choice), [])
            dets_list = st.session_state["detections_per_frame"].get(str(level_choice), [])
            total = sum(len(d) for d in dets_list)
            st.write(f"{total} detections across {len(paths)} frames")
            cols = st.columns(3)
            for i, (p, dets) in enumerate(zip(paths, dets_list)):
                img = cv2.imread(p)
                annotated = draw_detections(img, dets)
                cols[i % 3].image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                                   caption=f"{len(dets)} dets", use_container_width=True)

# ---------- Tab 4: Stitching ----------
with tab4:
    st.header("Stitch levels and combine into a facade")
    if not st.session_state.get("detections_per_frame"):
        st.info("Complete inference first.")
    else:
        methods = ["ortho", "panorama"]
        method = st.radio("Method", methods,
                           index=methods.index(st.session_state.get("stitch_method", "ortho")),
                           help="ortho: COLMAP 3D reconstruction projected onto the wall plane "
                                "(straight, complete facade; slower). panorama: legacy 2D "
                                "homography stitcher (fast, tears on parallax).")
        if method == "ortho" and st.button("Build facade", key="btn_ortho"):
            names, all_dets = [], []
            for lv in st.session_state["levels"]:
                idx_str = str(lv["index"])
                for fp, d in zip(st.session_state["frame_paths_per_level"][idx_str],
                                 st.session_state["detections_per_frame"][idx_str]):
                    names.append(str(Path(fp).relative_to(work_dir)))
                    all_dets.append(d)
            with st.spinner(f"Reconstructing {len(names)} frames in 3D (several minutes)..."):
                o = build_facade(work_dir, names, all_dets, work_dir)
            fp = work_dir / "facade_ortho.jpg"
            cv2.imwrite(str(fp), o["image"])
            st.session_state.update(
                stitch_method="ortho", facade_path=str(fp), facade_detections=o["detections"],
                ortho_stats={**o["stats"], "unregistered": o["unregistered"]},
                level_pano_paths={}, level_pano_detections={}, level_dropped_frames={},
                dropped_levels=[], cca_merged=None)
            _persist(project_name)
            st.rerun()

        if method == "panorama" and st.button("Stitch", key="btn_stitch"):
            level_pano_paths, level_pano_detections, level_dropped_frames = {}, {}, {}
            panoramas_ordered, detections_ordered = [], []
            with st.spinner("Stitching each level horizontally..."):
                for lv in st.session_state["levels"]:
                    idx_str = str(lv["index"])
                    frame_paths = [Path(p) for p in st.session_state["frame_paths_per_level"][idx_str]]
                    dets_per_frame = st.session_state["detections_per_frame"][idx_str]
                    result = stitch_horizontal(frame_paths, dets_per_frame)
                    pano_path = work_dir / f"level_{lv['index']:02d}_panorama.jpg"
                    cv2.imwrite(str(pano_path), result["panorama"])
                    level_pano_paths[idx_str] = str(pano_path)
                    level_pano_detections[idx_str] = result["detections"]
                    level_dropped_frames[idx_str] = result["dropped"]
                    panoramas_ordered.append(result["panorama"])
                    detections_ordered.append(result["detections"])

            facade_path, facade_detections, dropped_levels = None, [], []
            if len(panoramas_ordered) >= 2:
                with st.spinner("Stitching levels vertically into the full facade..."):
                    vresult = stitch_vertical(panoramas_ordered, detections_ordered,
                                               work_dir / "vertical_rotate_tmp")
                fp = work_dir / "facade_panorama.jpg"
                cv2.imwrite(str(fp), vresult["panorama"])
                facade_path, facade_detections, dropped_levels = str(fp), vresult["detections"], vresult["dropped"]
            elif panoramas_ordered:
                facade_path = level_pano_paths[str(st.session_state["levels"][0]["index"])]
                facade_detections = detections_ordered[0]

            st.session_state["level_pano_paths"] = level_pano_paths
            st.session_state["level_pano_detections"] = level_pano_detections
            st.session_state["level_dropped_frames"] = level_dropped_frames
            st.session_state["facade_path"] = facade_path
            st.session_state["facade_detections"] = facade_detections
            st.session_state["dropped_levels"] = dropped_levels
            st.session_state["stitch_method"] = "panorama"
            st.session_state["cca_merged"] = None
            _persist(project_name)
            st.rerun()

        if st.session_state.get("facade_path") and st.session_state.get("stitch_method") == "ortho":
            stats = st.session_state.get("ortho_stats") or {}
            st.success(f"{stats.get('registered')}/{stats.get('total')} frames placed in 3D · "
                       f"{stats.get('points3D')} points · reprojection error "
                       f"{stats.get('reproj_err_px', 0):.2f}px · facade "
                       f"{stats.get('facade_px', ('?', '?'))[0]}x{stats.get('facade_px', ('?', '?'))[1]}px")
            if stats.get("unregistered"):
                st.warning(f"{len(stats['unregistered'])} frame(s) not placed: {stats['unregistered']}")
            facade_img = cv2.imread(st.session_state["facade_path"])
            facade_annotated = draw_detections(facade_img, st.session_state["facade_detections"])
            st.image(cv2.cvtColor(facade_annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
        elif st.session_state.get("facade_path"):
            st.subheader("Per-level panoramas")
            level_choice = st.selectbox("View level", [lv["index"] for lv in st.session_state["levels"]],
                                         key="stitch_view_level")
            idx_str = str(level_choice)
            pano_path = st.session_state["level_pano_paths"].get(idx_str)
            if pano_path:
                img = cv2.imread(pano_path)
                dets = st.session_state["level_pano_detections"].get(idx_str, [])
                annotated = draw_detections(img, dets)
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
                dropped = st.session_state["level_dropped_frames"].get(idx_str, [])
                if dropped:
                    st.warning(f"{len(dropped)} frame(s) couldn't be connected: {dropped}")

            st.subheader("Combined facade")
            facade_img = cv2.imread(st.session_state["facade_path"])
            facade_annotated = draw_detections(facade_img, st.session_state["facade_detections"])
            st.image(cv2.cvtColor(facade_annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
            if st.session_state.get("dropped_levels"):
                st.warning(f"Level(s) not connected to the facade: {st.session_state['dropped_levels']}")

# ---------- Tab 5: CCA & findings ----------
with tab5:
    st.header("Merge overlapping detections (connected-component analysis)")
    if not st.session_state.get("facade_path"):
        st.info("Complete stitching first.")
    else:
        close_px = st.slider("Closing kernel (px) — bridges gaps from tile-split cracks",
                              3, 51, st.session_state.get("cca_close_px", 15), 2)
        open_px = st.slider("Opening kernel (px) — removes noise specks",
                             1, 21, st.session_state.get("cca_open_px", 3), 2)
        if st.button("Merge detections", key="btn_cca"):
            facade_img = cv2.imread(st.session_state["facade_path"])
            merged, _cleaned_mask = merge_detections_cca(
                st.session_state["facade_detections"], facade_img.shape, close_px, open_px)
            st.session_state["cca_close_px"] = close_px
            st.session_state["cca_open_px"] = open_px
            st.session_state["cca_merged"] = merged
            _persist(project_name)
            st.rerun()

        if st.session_state.get("cca_merged"):
            raw_count = len(st.session_state["facade_detections"])
            merged_count = len(st.session_state["cca_merged"])
            st.success(f"{raw_count} raw detections → {merged_count} merged cracks")

            facade_img = cv2.imread(st.session_state["facade_path"])
            annotated = draw_merged(facade_img, st.session_state["cca_merged"])
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

            rows = [{"crack #": i + 1, "score": round(d["score"], 2), "area (px)": d["area_px"],
                     "bbox": tuple(round(v) for v in d["bbox"])}
                    for i, d in enumerate(st.session_state["cca_merged"])]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
