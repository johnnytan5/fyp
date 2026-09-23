"""Orchestrator: video -> per-level split -> sample -> detect -> horizontal
stitch+overlay -> vertical stitch+overlay -> one crack-annotated facade
image. This is the entry point the Streamlit frontend will call.

Detections are carried as coordinate data through both stitching stages and
only rendered (drawn as boxes/masks) once, on the final facade image -- see
stitch_horizontal.py's module docstring for why (avoids compounding
distortion from drawing then re-warping an annotated image).
"""
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))  # for infer.py
from detect import detect_cracks  # noqa: E402
from infer import CONF_THRESHOLD, load_sahi_model  # noqa: E402
from sampling import DWELL_MIN_SECONDS, DWELL_VELOCITY_EPS, STEP_M, extract_frames, sample_frames  # noqa: E402
from srt_levels import Sample, extract_telemetry_srt, parse_telemetry, detect_levels, split_video_into_levels  # noqa: E402
from stitch_horizontal import draw_detections, stitch_horizontal  # noqa: E402
from stitch_vertical import stitch_vertical  # noqa: E402


@dataclass
class LevelResult:
    level_index: int
    height_m: float
    n_frames: int
    panorama: Any = None
    detections: list = field(default_factory=list)
    dropped_frames: list = field(default_factory=list)


@dataclass
class PipelineResult:
    facade: Any = None
    facade_detections: list = field(default_factory=list)
    dropped_levels: list = field(default_factory=list)
    per_level: list[LevelResult] = field(default_factory=list)


def run_pipeline(
    video_path: Path,
    work_dir: Path,
    sampling_mode: Literal["dynamic", "dwell"] = "dynamic",
    top_to_bottom: bool = False,
    step_m: float = STEP_M,
    dwell_velocity_eps: float = DWELL_VELOCITY_EPS,
    dwell_min_seconds: float = DWELL_MIN_SECONDS,
    conf_threshold: float = CONF_THRESHOLD,
    save_per_level_annotated: bool = True,
    max_levels: int | None = None,
) -> PipelineResult:
    work_dir.mkdir(parents=True, exist_ok=True)

    srt_text = extract_telemetry_srt(video_path)
    samples = parse_telemetry(srt_text)
    levels = detect_levels(samples, top_to_bottom=top_to_bottom)
    if not levels:
        raise RuntimeError("no levels detected in this video's telemetry")
    if max_levels is not None:
        levels = levels[:max_levels]

    level_video_paths = split_video_into_levels(video_path, levels, work_dir / "levels")
    model = load_sahi_model(conf=conf_threshold)

    per_level_results: list[LevelResult] = []
    for level, level_video in zip(levels, level_video_paths):
        level_samples = [s for s in samples if level.start_s <= s.t <= level.end_s]
        rel_samples = [Sample(s.t - level.start_s, s.h, s.h_speed, s.v_speed) for s in level_samples]
        timestamps = sample_frames(rel_samples, mode=sampling_mode, step_m=step_m,
                                    velocity_eps=dwell_velocity_eps, min_dwell_s=dwell_min_seconds)
        frame_paths = extract_frames(level_video, timestamps, work_dir / f"frames_level{level.index}")

        detections_per_frame = []
        for fp in frame_paths:
            img = cv2.imread(str(fp))
            detections_per_frame.append(detect_cracks(img, model) if img is not None else [])

        stitched = stitch_horizontal(frame_paths, detections_per_frame)
        lvl_result = LevelResult(
            level_index=level.index, height_m=level.height_m, n_frames=len(frame_paths),
            panorama=stitched["panorama"], detections=stitched["detections"],
            dropped_frames=stitched["dropped"],
        )
        per_level_results.append(lvl_result)

        if save_per_level_annotated:
            annotated = draw_detections(lvl_result.panorama, lvl_result.detections)
            cv2.imwrite(str(work_dir / f"level_{level.index:02d}_annotated.jpg"), annotated)

        print(f"[run_pipeline] level {level.index} ({level.height_m:.2f}m): "
              f"{len(frame_paths)} frames, {len(lvl_result.detections)} detections, "
              f"dropped frames {lvl_result.dropped_frames}", file=sys.stderr)

    connectable = [lr for lr in per_level_results if lr.panorama is not None]
    if len(connectable) < 2:
        result = PipelineResult(
            facade=connectable[0].panorama if connectable else None,
            facade_detections=connectable[0].detections if connectable else [],
            per_level=per_level_results,
        )
    else:
        vresult = stitch_vertical(
            [lr.panorama for lr in connectable],
            [lr.detections for lr in connectable],
            work_dir / "vertical_rotate_tmp",
        )
        result = PipelineResult(
            facade=vresult["panorama"], facade_detections=vresult["detections"],
            dropped_levels=vresult["dropped"], per_level=per_level_results,
        )

    if result.facade is not None:
        final_annotated = draw_detections(result.facade, result.facade_detections)
        cv2.imwrite(str(work_dir / "facade_annotated.jpg"), final_annotated)
        print(f"[run_pipeline] facade: {result.facade.shape}, "
              f"{len(result.facade_detections)} total detections, "
              f"dropped levels {result.dropped_levels} -> {work_dir / 'facade_annotated.jpg'}",
              file=sys.stderr)

    return result


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--sampling-mode", choices=["dynamic", "dwell"], default="dynamic")
    parser.add_argument("--top-to-bottom", action="store_true")
    parser.add_argument("--max-levels", type=int, default=None)
    args = parser.parse_args()
    run_pipeline(args.video, args.work_dir, sampling_mode=args.sampling_mode,
                 top_to_bottom=args.top_to_bottom, max_levels=args.max_levels)


if __name__ == "__main__":
    main()
