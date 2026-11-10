"""Convert a decoded flight-log SRT (imperial units, one total speed, starts at
log start) into the DJI-video telemetry format srt_levels.py parses, trimmed
and re-timed to one video clip.

Needed for drones (e.g. Mavic Air 2) whose videos have no embedded telemetry
track. Input blocks look like:
    Height ATO: 4.6 ft / Speed: 1.2 mph / Lat/Long: 3.12,101.65
Output lines look like:
    GPS (101.65, 3.12, 0), D 0.00m, H 1.40m, H.S 0.50m/s, V.S 0.00m/s

Alignment: --video-start-s is where the clip starts in LOG seconds. Find it by
matching photo EXIF times/heights against the log, or video creation_time
minus log start (both in local time).
"""
import math
import re
import sys
from pathlib import Path

FT_TO_M = 0.3048
MPH_TO_MS = 0.44704
VS_HALF_WINDOW_S = 1.0  # ponytail: central difference over +-1 s; height is only 0.1 ft resolution

_BLOCK_RE = re.compile(r"(\d\d):(\d\d):(\d\d),(\d+) -->")


def parse_log_srt(text: str) -> list[dict]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    recs = []
    for block in text.strip().split("\n\n"):
        t = _BLOCK_RE.search(block)
        h = re.search(r"Height ATO: (-?[\d.]+) ft", block)
        s = re.search(r"Speed: (-?[\d.]+) mph", block)
        ll = re.search(r"Lat/Long: (-?[\d.]+),(-?[\d.]+)", block)
        if not (t and h and s and ll):
            continue
        recs.append({
            "t": int(t[1]) * 3600 + int(t[2]) * 60 + int(t[3]) + int(t[4]) / 1000,
            "h": float(h[1]) * FT_TO_M, "speed": float(s[1]) * MPH_TO_MS,
            "lat": float(ll[1]), "lon": float(ll[2]),
        })
    return recs


def add_speeds(recs: list[dict]) -> None:
    """V.S = central height difference (positive = climbing); H.S = horizontal
    part of the total speed."""
    for i, r in enumerate(recs):
        lo = next(j for j in range(i, -1, -1) if r["t"] - recs[j]["t"] >= VS_HALF_WINDOW_S or j == 0)
        hi = next(j for j in range(i, len(recs)) if recs[j]["t"] - r["t"] >= VS_HALF_WINDOW_S or j == len(recs) - 1)
        dt = recs[hi]["t"] - recs[lo]["t"]
        r["vs"] = (recs[hi]["h"] - recs[lo]["h"]) / dt if dt > 0 else 0.0
        r["hs"] = math.sqrt(max(0.0, r["speed"] ** 2 - r["vs"] ** 2))


def _fmt(t: float) -> str:
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def to_dji_srt(recs: list[dict], video_start_s: float, duration_s: float) -> str:
    clip = [r for r in recs if video_start_s <= r["t"] <= video_start_s + duration_s]
    out = []
    for n, (r, nxt) in enumerate(zip(clip, clip[1:] + [None]), start=1):
        t0 = r["t"] - video_start_s
        t1 = (nxt["t"] - video_start_s) if nxt else t0 + 0.5
        out.append(f"{n}\n{_fmt(t0)} --> {_fmt(t1)}\n"
                   f"GPS ({r['lon']:.6f}, {r['lat']:.6f}, 0), D 0.00m, H {r['h']:.2f}m, "
                   f"H.S {r['hs']:.2f}m/s, V.S {r['vs']:.2f}m/s\n")
    return "\n".join(out)


def _self_check() -> None:
    import srt_levels
    log = ("1\n00:00:00,000 --> 00:00:00,500\nLat/Long: 3.1,101.6 \nHeight ATO: 0.0 ft\nSpeed: 0.0 mph\n\n"
           "2\n00:00:00,500 --> 00:00:01,000\nLat/Long: 3.1,101.6 \nHeight ATO: 3.3 ft\nSpeed: 2.2 mph\n\n"
           "3\n00:00:01,000 --> 00:00:01,500\nLat/Long: 3.1,101.6 \nHeight ATO: 6.6 ft\nSpeed: 4.5 mph\n")
    recs = parse_log_srt(log)
    add_speeds(recs)
    assert abs(recs[2]["h"] - 2.01) < 0.01, recs[2]["h"]
    assert recs[1]["vs"] > 1.5, "climbing ~2 m/s should give positive V.S"
    srt = to_dji_srt(recs, video_start_s=0.5, duration_s=1.0)
    samples = srt_levels.parse_telemetry(srt)
    assert len(samples) == 2 and samples[0].t == 0.0, samples  # re-timed to clip start, parser reads it
    print("[flightlog_to_srt] self-check OK", file=sys.stderr)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    if sys.argv[1:] == ["--self-check"]:
        _self_check()
    else:
        import argparse
        ap = argparse.ArgumentParser()
        ap.add_argument("log_srt", type=Path)
        ap.add_argument("--video-start-s", type=float, required=True, help="clip start, in log seconds")
        ap.add_argument("--duration-s", type=float, required=True)
        ap.add_argument("--out", type=Path, required=True)
        a = ap.parse_args()
        recs = parse_log_srt(a.log_srt.read_text(errors="ignore"))
        add_speeds(recs)
        a.out.write_text(to_dji_srt(recs, a.video_start_s, a.duration_s))
        print(f"[flightlog_to_srt] {len(recs)} log records -> {a.out}", file=sys.stderr)
