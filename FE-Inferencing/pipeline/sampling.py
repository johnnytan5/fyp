"""Sample frame timestamps from a level video, two selectable strategies:

- dynamic: sample every STEP_M meters of horizontal travel (integrating H.S,
  which is real telemetry). Works on continuous-pan footage. Frames will
  have motion blur proportional to flight speed / shutter speed -- see
  dwell mode for the fix.

- dwell: assumes the footage was flown as a waypoint mission with a brief
  hover at each sample point (e.g. via generate_litchi_mission.py's
  stay-then-shoot pattern). Finds near-zero-H.S segments and picks one
  frame from the middle of each -- these frames have ~zero motion blur
  regardless of shutter speed, since the drone isn't moving when captured.

Both are driven off the same telemetry Sample list from srt_levels.py.
"""
import subprocess
import sys
from pathlib import Path

from srt_levels import Sample

STEP_M = 1.0            # ponytail: tunable -- meters of horizontal travel between dynamic samples
DWELL_VELOCITY_EPS = 0.15  # ponytail: tunable -- m/s below this counts as "stopped"; calibrate
                           # once real waypoint-mission footage is available
DWELL_MIN_SECONDS = 0.5   # ponytail: tunable -- minimum dwell duration to count (filters noise)


def sample_dynamic(level_samples: list[Sample], step_m: float = STEP_M) -> list[float]:
    """Timestamps spaced by step_m of H.S-integrated travel, interpolating
    within a tick so fast motion still yields sub-second-spaced samples
    (telemetry is only 1Hz -- without interpolation, sampling could never
    be denser than 1 sample/second no matter how small step_m is)."""
    if len(level_samples) < 2:
        return [s.t for s in level_samples]

    timestamps = [level_samples[0].t]
    accumulated_m = 0.0
    for a, b in zip(level_samples, level_samples[1:]):
        dt = b.t - a.t
        dist = a.h_speed * dt
        if dist <= 0:
            continue
        pos_in_interval = 0.0
        while accumulated_m + (dist - pos_in_interval) >= step_m:
            need = step_m - accumulated_m
            frac = (pos_in_interval + need) / dist
            timestamps.append(a.t + frac * dt)
            pos_in_interval += need
            accumulated_m = 0.0
        accumulated_m += dist - pos_in_interval
    if timestamps[-1] != level_samples[-1].t:
        timestamps.append(level_samples[-1].t)
    return timestamps


def sample_dwell(level_samples: list[Sample], velocity_eps: float = DWELL_VELOCITY_EPS,
                  min_dwell_s: float = DWELL_MIN_SECONDS) -> list[float]:
    """One timestamp per near-zero-H.S segment (the middle of the dwell),
    for waypoint-mission footage with deliberate hover-and-shoot stops."""
    if not level_samples:
        return []

    labels = ["DWELL" if s.h_speed < velocity_eps else "MOVE" for s in level_samples]
    segments: list[tuple[str, int, int]] = []
    seg_start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[seg_start]:
            segments.append((labels[seg_start], seg_start, i - 1))
            seg_start = i

    timestamps = []
    for label, a, b in segments:
        if label != "DWELL":
            continue
        if level_samples[b].t - level_samples[a].t < min_dwell_s:
            continue
        mid_idx = (a + b) // 2
        timestamps.append(level_samples[mid_idx].t)
    return timestamps


def sample_frames(level_samples: list[Sample], mode: str = "dynamic",
                   step_m: float = STEP_M, velocity_eps: float = DWELL_VELOCITY_EPS,
                   min_dwell_s: float = DWELL_MIN_SECONDS) -> list[float]:
    if mode == "dynamic":
        return sample_dynamic(level_samples, step_m)
    if mode == "dwell":
        return sample_dwell(level_samples, velocity_eps, min_dwell_s)
    raise ValueError(f"unknown sampling mode: {mode!r} (expected 'dynamic' or 'dwell')")


SEEK_END_MARGIN_S = 0.15  # ponytail: a timestamp right at a clip's end can seek past the
                          # last decodable frame (confirmed: sampling always includes the
                          # level's exact end_s, which sits ~0.01-0.1s from a stream-copied
                          # clip's true last frame) -- stay this far clear of the end


def _video_duration_s(video_path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
        capture_output=True, text=True,
    )
    return float(result.stdout.strip())


def extract_frame(video_path: Path, t: float, out_path: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-ss", str(t), "-i", str(video_path), "-frames:v", "1", "-q:v", "3", str(out_path)],
        capture_output=True,
    )


def extract_frames(video_path: Path, timestamps: list[float], out_dir: Path,
                    prefix: str = "f") -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    duration = _video_duration_s(video_path)
    safe_end = max(0.0, duration - SEEK_END_MARGIN_S)
    paths = []
    for i, t in enumerate(timestamps):
        t_safe = min(t, safe_end)
        p = out_dir / f"{prefix}{i:02d}_t{t_safe:.2f}.jpg"
        extract_frame(video_path, t_safe, p)
        if not p.exists():
            print(f"[sampling] WARNING: frame extraction failed at t={t_safe:.2f}s "
                  f"(video duration {duration:.3f}s), skipping", file=sys.stderr)
            continue
        paths.append(p)
    return paths


def _self_check() -> None:
    """ponytail: minimal runnable check, no test framework."""
    # dynamic: 2 m/s for 3 ticks = 6m travel; step_m=1.0 -> 6 evenly-spaced samples
    synth = [Sample(float(t), 1.0, 2.0, 0.0) for t in range(4)]
    ts = sample_dynamic(synth, step_m=1.0)
    assert ts[0] == 0.0 and ts[-1] == 3.0, ts
    gaps = [b - a for a, b in zip(ts, ts[1:])]
    assert all(abs(g - 0.5) < 1e-9 for g in gaps[:-1]), gaps

    # dwell: move, stop for 2s, move, stop for 2s -> 2 dwell samples
    dwell_synth = (
        [Sample(float(t), 1.0, 2.0, 0.0) for t in range(0, 3)] +      # MOVE 0-2
        [Sample(float(t), 1.0, 0.0, 0.0) for t in range(3, 6)] +      # DWELL 3-5
        [Sample(float(t), 1.0, 2.0, 0.0) for t in range(6, 9)] +      # MOVE 6-8
        [Sample(float(t), 1.0, 0.0, 0.0) for t in range(9, 12)]       # DWELL 9-11
    )
    dwell_ts = sample_dwell(dwell_synth, velocity_eps=0.15, min_dwell_s=0.5)
    assert len(dwell_ts) == 2, dwell_ts
    assert 3.0 <= dwell_ts[0] <= 5.0 and 9.0 <= dwell_ts[1] <= 11.0, dwell_ts

    print("[sampling] self-check OK", file=sys.stderr)


if __name__ == "__main__":
    _self_check()
