"""Parse DJI embedded SRT telemetry, detect scan-row/level segments, and
physically split the source video into one file per level.

Each level = one horizontal pass at roughly constant height. Levels are
separated by short rise bursts (drone stepping up ~half a floor height for
vertical-stitch overlap). Detected from the barometric height field's
derivative (V.S), not GPS altitude, which jitters too much to use.

`D` (distance) in this telemetry is distance-from-home-point, NOT
distance-to-wall -- confirmed by it swinging ~4.7x monotonically within
every single level, alternating direction level to level (a boustrophedon
sweep signature, not what a wall-standoff sensor would ever show). Don't
use it for real-world sizing.
"""
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

VS_RISE_THRESHOLD = 0.3     # m/s -- V.S above this counts as "rising"
MIN_LEVEL_SECONDS = 3       # discard flying segments shorter than this (noise)
MIN_RISE_SECONDS = 1        # discard rise blips shorter than this (noise)

TELEMETRY_RE = re.compile(
    r"(\d\d:\d\d:\d\d,\d+) --> .*?\n"
    r".*?H (-?[\d.]+)m, H\.S (-?[\d.]+)m/s, V\.S (-?[\d.]+)m/s"
)


@dataclass
class Sample:
    t: float  # seconds
    h: float  # barometric relative height, meters
    h_speed: float
    v_speed: float


@dataclass
class Level:
    index: int
    start_s: float
    end_s: float
    height_m: float  # median height during this level


def _timestamp_to_seconds(ts: str) -> float:
    h, m, rest = ts.split(":")
    s, ms = rest.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def extract_telemetry_srt(video_path: Path) -> str:
    """Extract the embedded DJI subtitle telemetry track as SRT text."""
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", str(video_path), "-map", "0:s:0", "-c:s", "srt", "-f", "srt", "-"],
        capture_output=True, text=True,
    )
    if not result.stdout.strip():
        raise RuntimeError(f"No telemetry subtitle extracted (stderr: {result.stderr[-500:]})")
    return result.stdout


def parse_telemetry(srt_text: str) -> list[Sample]:
    samples = []
    for ts, h, h_speed, v_speed in TELEMETRY_RE.findall(srt_text):
        samples.append(Sample(_timestamp_to_seconds(ts), float(h), float(h_speed), float(v_speed)))
    return samples


def detect_levels(
    samples: list[Sample],
    vs_threshold: float = VS_RISE_THRESHOLD,
    min_level_s: float = MIN_LEVEL_SECONDS,
    min_rise_s: float = MIN_RISE_SECONDS,
    top_to_bottom: bool = False,
) -> list[Level]:
    """Classify each sample as RISING or FLYING, merge into level segments."""
    if not samples:
        return []

    labels = ["RISE" if s.v_speed > vs_threshold else "FLY" for s in samples]

    segments: list[tuple[str, int, int]] = []  # (label, start_idx, end_idx inclusive)
    seg_start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[seg_start]:
            segments.append((labels[seg_start], seg_start, i - 1))
            seg_start = i

    def seg_duration(seg):
        _, a, b = seg
        return samples[b].t - samples[a].t

    filtered = [seg for seg in segments if
                (seg[0] == "FLY" and seg_duration(seg) >= min_level_s) or
                (seg[0] == "RISE" and seg_duration(seg) >= min_rise_s)]

    levels = []
    for label, a, b in filtered:
        if label != "FLY":
            continue
        heights = sorted(s.h for s in samples[a:b + 1])
        median_h = heights[len(heights) // 2]
        levels.append(Level(len(levels), samples[a].t, samples[b].t, median_h))

    if top_to_bottom:
        levels = list(reversed(levels))
        for i, lv in enumerate(levels):
            lv.index = i
    return levels


def split_video_into_levels(video_path: Path, levels: list[Level], out_dir: Path) -> list[Path]:
    """Cut the source video into one file per level via stream-copy (fast,
    lossless). Stream-copy only cuts on keyframe boundaries, so actual split
    points can land ~1-2s off the requested timestamp -- acceptable here
    since each level already excludes a rise-transition buffer on both ends.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_paths = []
    for level in levels:
        out_path = out_dir / f"level_{level.index:02d}.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-ss", str(level.start_s), "-to", str(level.end_s),
             "-i", str(video_path), "-c", "copy", "-avoid_negative_ts", "make_zero",
             str(out_path)],
            capture_output=True,
        )
        out_paths.append(out_path)
    return out_paths


def parse_and_split(video_path: Path, out_dir: Path, top_to_bottom: bool = False,
                     vs_threshold: float = VS_RISE_THRESHOLD,
                     min_level_s: float = MIN_LEVEL_SECONDS) -> tuple[list[Level], list[Path]]:
    """Convenience entry point: telemetry -> levels -> split video files."""
    srt_text = extract_telemetry_srt(video_path)
    samples = parse_telemetry(srt_text)
    levels = detect_levels(samples, vs_threshold, min_level_s, top_to_bottom=top_to_bottom)
    paths = split_video_into_levels(video_path, levels, out_dir)
    return levels, paths


def _self_check() -> None:
    """ponytail: minimal runnable check, no test framework."""
    synth = (
        [Sample(t, 1.0, 1.0, 0.0) for t in range(5)] +
        [Sample(t, 1.0 + 0.5 * (t - 4), 0.0, 0.5) for t in range(5, 7)] +
        [Sample(t, 2.0, 1.0, 0.0) for t in range(7, 12)]
    )
    levels = detect_levels(synth, vs_threshold=0.3, min_level_s=3, min_rise_s=1)
    assert len(levels) == 2, f"expected 2 levels, got {len(levels)}"
    assert levels[0].height_m == 1.0 and levels[1].height_m == 2.0
    assert levels[0].index == 0 and levels[1].index == 1
    reversed_levels = detect_levels(synth, top_to_bottom=True)
    assert reversed_levels[0].height_m == 2.0 and reversed_levels[0].index == 0
    print("[srt_levels] self-check OK", file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--self-check":
        _self_check()
    else:
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("video", type=Path)
        parser.add_argument("--out-dir", type=Path, required=True)
        parser.add_argument("--top-to-bottom", action="store_true")
        args = parser.parse_args()
        levels, paths = parse_and_split(args.video, args.out_dir, args.top_to_bottom)
        for lv, p in zip(levels, paths):
            print(f"level {lv.index}: {lv.start_s:.1f}s-{lv.end_s:.1f}s, "
                  f"height {lv.height_m:.2f}m -> {p}")
