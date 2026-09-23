"""Local persistence for pipeline projects: SQLite for state/metadata,
plain files on disk for video/images (which the pipeline already writes as
part of normal operation -- the DB just points at them). Lets the Streamlit
app reopen a project later without re-running split/sample/infer/stitch.

One row per project, state stored as one JSON blob -- simpler than a fully
normalized schema, and entirely adequate for a single-user local tool with
no concurrent-writer or complex-query needs. Detection dicts contain numpy
mask arrays, which aren't JSON-serializable, so they're converted to plain
lists on save and back to arrays on load.
"""
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

DB_PATH = Path(__file__).parent.parent / "projects" / "projects.db"
PROJECTS_ROOT = Path(__file__).parent.parent / "projects"


def _connect() -> sqlite3.Connection:
    PROJECTS_ROOT.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS projects (
            name TEXT PRIMARY KEY,
            work_dir TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            state_json TEXT NOT NULL
        )
    """)
    return conn


def _serialize_detections(detections: list[dict]) -> list[dict]:
    out = []
    for d in detections:
        item = dict(d)
        if item.get("mask") is not None:
            item["mask"] = item["mask"].tolist()
        if item.get("contour") is not None:
            item["contour"] = item["contour"].tolist()
        out.append(item)
    return out


def _deserialize_detections(detections: list[dict]) -> list[dict]:
    out = []
    for d in detections:
        item = dict(d)
        if item.get("mask") is not None:
            item["mask"] = np.array(item["mask"], dtype=np.float32)
        if item.get("contour") is not None:
            item["contour"] = np.array(item["contour"], dtype=np.float32)
        out.append(item)
    return out


# Keys in the state dict whose values are lists of detection dicts (mask
# arrays needing serialization), keyed by an arbitrary structure -- see
# serialize_state/deserialize_state below for how each is walked.
_PER_LEVEL_DETECTION_KEYS = ["level_pano_detections"]                    # {level_index: [dets]}
_PER_LEVEL_PER_FRAME_DETECTION_KEYS = ["detections_per_frame"]           # {level_index: [[dets], [dets], ...]}
_FLAT_DETECTION_LIST_KEYS = ["facade_detections", "cca_merged"]         # [dets]


def serialize_state(state: dict[str, Any]) -> str:
    out = dict(state)
    for key in _PER_LEVEL_DETECTION_KEYS:
        if out.get(key) is not None:
            out[key] = {k: _serialize_detections(v) for k, v in out[key].items()}
    for key in _PER_LEVEL_PER_FRAME_DETECTION_KEYS:
        if out.get(key) is not None:
            out[key] = {k: [_serialize_detections(f) for f in v] for k, v in out[key].items()}
    for key in _FLAT_DETECTION_LIST_KEYS:
        if out.get(key) is not None:
            out[key] = _serialize_detections(out[key])
    return json.dumps(out)


def deserialize_state(state_json: str) -> dict[str, Any]:
    out = json.loads(state_json)
    for key in _PER_LEVEL_DETECTION_KEYS:
        if out.get(key) is not None:
            out[key] = {k: _deserialize_detections(v) for k, v in out[key].items()}
    for key in _PER_LEVEL_PER_FRAME_DETECTION_KEYS:
        if out.get(key) is not None:
            out[key] = {k: [_deserialize_detections(f) for f in v] for k, v in out[key].items()}
    for key in _FLAT_DETECTION_LIST_KEYS:
        if out.get(key) is not None:
            out[key] = _deserialize_detections(out[key])
    return out


def list_projects() -> list[dict]:
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT name, work_dir, created_at, updated_at FROM projects ORDER BY updated_at DESC"
        ).fetchall()
        return [{"name": r[0], "work_dir": r[1], "created_at": r[2], "updated_at": r[3]} for r in rows]
    finally:
        conn.close()


def project_work_dir(name: str) -> Path:
    return PROJECTS_ROOT / name


def create_project(name: str) -> dict:
    work_dir = project_work_dir(name)
    work_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    state = {"video_path": None}
    conn = _connect()
    try:
        conn.execute(
            "INSERT INTO projects (name, work_dir, created_at, updated_at, state_json) VALUES (?, ?, ?, ?, ?)",
            (name, str(work_dir), now, now, json.dumps(state)),
        )
        conn.commit()
    finally:
        conn.close()
    return state


def save_state(name: str, state: dict[str, Any]) -> None:
    now = datetime.now(timezone.utc).isoformat()
    conn = _connect()
    try:
        conn.execute(
            "UPDATE projects SET state_json = ?, updated_at = ? WHERE name = ?",
            (serialize_state(state), now, name),
        )
        conn.commit()
    finally:
        conn.close()


def load_state(name: str) -> dict[str, Any] | None:
    conn = _connect()
    try:
        row = conn.execute("SELECT state_json FROM projects WHERE name = ?", (name,)).fetchone()
        return deserialize_state(row[0]) if row else None
    finally:
        conn.close()


def delete_project(name: str) -> None:
    import shutil
    conn = _connect()
    try:
        conn.execute("DELETE FROM projects WHERE name = ?", (name,))
        conn.commit()
    finally:
        conn.close()
    work_dir = project_work_dir(name)
    if work_dir.exists():
        shutil.rmtree(work_dir)


def _self_check() -> None:
    """ponytail: round-trips a state dict containing a numpy mask through
    serialize/deserialize and a real sqlite save/load, using a throwaway
    project name so it doesn't collide with real projects."""
    import sys
    test_name = "__self_check_project__"
    delete_project(test_name)  # clean slate if a previous failed run left it behind

    create_project(test_name)
    state = {
        "video_path": "/tmp/fake.mp4",
        "facade_detections": [
            {"bbox": (1.0, 2.0, 3.0, 4.0), "mask": np.array([[1.0, 2.0], [3.0, 4.0]]), "score": 0.9},
            {"bbox": (5.0, 6.0, 7.0, 8.0), "mask": None, "score": 0.5},
        ],
        "level_pano_detections": {"0": [{"bbox": (0, 0, 1, 1), "mask": None, "score": 0.1}]},
    }
    save_state(test_name, state)
    loaded = load_state(test_name)

    assert loaded["video_path"] == "/tmp/fake.mp4"
    assert isinstance(loaded["facade_detections"][0]["mask"], np.ndarray)
    assert loaded["facade_detections"][0]["mask"].tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert loaded["facade_detections"][1]["mask"] is None
    assert loaded["level_pano_detections"]["0"][0]["score"] == 0.1

    names = [p["name"] for p in list_projects()]
    assert test_name in names

    delete_project(test_name)
    assert test_name not in [p["name"] for p in list_projects()]

    print("[project_store] self-check OK", file=sys.stderr)


if __name__ == "__main__":
    _self_check()
