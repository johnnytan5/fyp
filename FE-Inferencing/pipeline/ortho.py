"""Photogrammetry facade: frames -> COLMAP structure-from-motion -> wall-plane
orthomosaic, with crack detections projected onto it.

Replaces the 2D panorama stitcher (stitch_horizontal/stitch_vertical), which
models each frame pair with one homography. That is only exact for a rotating
camera or a flat scene; a drone translates along a facade with protrusions, so
parallax tore windows, tilted the building and dropped levels.

Here COLMAP recovers every camera in 3D jointly (bundle adjustment), RANSAC
finds the wall plane in the point cloud, and each frame is warped onto that
plane with a homography derived from its known pose:

    facade pixel (x, y) -> world X = O + x*res*right - y*res*up
    H_i = K [R_i a | R_i b | R_i O + t_i],  a = res*right, b = -res*up

Each facade pixel takes its colour from the frame whose optical axis hits the
wall nearest that pixel (most head-on view, least parallax; no blending, so
crack texture stays sharp). Crack detections run on the original frames and
are mapped with the same H_i^-1, so they land exactly on the wall. A
detection is kept if it lands on filmed wall; copies of one crack seen in
overlapping frames are merged afterwards by CCA (cca.py).

Rebuilt from the "J17 Facade Photogrammetry" write-up (18 Oct 2026).
"""
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

MAX_EXTRACT_PX = 2000    # SIFT on full 4K frames x threads exhausted 16 GB RAM once
NUM_THREADS = 4
MAX_FEATURES = 8192
MAX_FACADE_PX = 6000     # ponytail: output size cap; raise for full native detail (memory cost)
PLANE_ITERS = 3000
PLANE_THR_FRAC = 0.01    # inlier band = 1% of the cloud's median spread (model has no units)


def _sfm_child(image_root: Path, names_file: Path, work_dir: Path) -> None:
    """Runs in its own process (see reconstruct). SIFT -> exhaustive matching +
    geometric verification -> incremental SfM, then exports the largest model
    as plain numpy arrays to work_dir/model.npz."""
    import pycolmap

    image_names = names_file.read_text().splitlines()
    db = work_dir / "db.db"
    db.unlink(missing_ok=True)
    out = work_dir / "sparse"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir()

    ext = pycolmap.FeatureExtractionOptions()
    ext.max_image_size = MAX_EXTRACT_PX
    ext.num_threads = NUM_THREADS
    ext.sift.max_num_features = MAX_FEATURES
    pycolmap.extract_features(db, image_root, image_names=image_names,
                              camera_mode=pycolmap.CameraMode.SINGLE, extraction_options=ext)

    mo = pycolmap.FeatureMatchingOptions()
    mo.num_threads = NUM_THREADS
    # ponytail: exhaustive is n^2 pairs (~13k for 164 frames); switch to
    # match_sequential / vocab tree for surveys of several hundred frames
    pycolmap.match_exhaustive(db, matching_options=mo)

    recs = pycolmap.incremental_mapping(db, image_root, out)
    if not recs:
        raise RuntimeError("SfM registered no images")
    rec = max(recs.values(), key=lambda r: r.num_reg_images())

    imgs = sorted((im for im in rec.images.values() if im.has_pose), key=lambda im: im.name)
    poses = [_pose(im) for im in imgs]
    intr = [_intrinsics(im.camera) for im in imgs]
    np.savez(work_dir / "model.npz",
             xyz=np.array([p.xyz for p in rec.points3D.values()]),
             names=np.array([im.name for im in imgs]),
             R=np.array([R for R, _ in poses]), t=np.array([t for _, t in poses]),
             K=np.array([K for K, _ in intr]), dist=np.array([d for _, d in intr]),
             reproj_err=rec.compute_mean_reprojection_error(), n_models=len(recs))


def reconstruct(image_root: Path, image_names: list[str], work_dir: Path) -> dict:
    """COLMAP in a child process: torch (detection) and pycolmap each bundle
    libomp, and macOS aborts ("OMP: Error #15") when both load in one process.
    Cached as work_dir/model.npz, reused only for the same frame list."""
    import subprocess

    work_dir.mkdir(parents=True, exist_ok=True)
    npz = work_dir / "model.npz"
    names_file = work_dir / "images.txt"
    names_txt = "\n".join(image_names)
    if not npz.exists() or not names_file.exists() or names_file.read_text() != names_txt:
        npz.unlink(missing_ok=True)
        names_file.write_text(names_txt)
        subprocess.run([sys.executable, __file__, "--sfm", str(image_root), str(names_file), str(work_dir)],
                       check=True)
    m = dict(np.load(npz))
    m["names"] = [str(x) for x in m["names"]]
    print(f"[ortho] SfM: {len(m['names'])}/{len(image_names)} frames registered, {len(m['xyz'])} 3D points, "
          f"mean reproj err {float(m['reproj_err']):.2f}px ({int(m['n_models'])} model(s))", file=sys.stderr)
    return m


def fit_wall_plane(pts: np.ndarray, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RANSAC plane, then SVD refit on inliers. Returns (normal, centroid, inlier_mask).
    Fixed seed so the facade image is identical on every run."""
    rng = np.random.default_rng(seed)
    scale = np.median(np.linalg.norm(pts - np.median(pts, 0), axis=1))
    thr = PLANE_THR_FRAC * scale
    best_inl = None
    for _ in range(PLANE_ITERS):
        a, b, c = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-12:
            continue  # collinear sample
        n /= np.linalg.norm(n)
        inl = np.abs((pts - a) @ n) < thr
        if best_inl is None or inl.sum() > best_inl.sum():
            best_inl = inl
    P = pts[best_inl]
    centroid = P.mean(0)
    n = np.linalg.svd(P - centroid)[2][-1]
    return n, centroid, best_inl


def _intrinsics(cam) -> tuple[np.ndarray, np.ndarray]:
    K = np.asarray(cam.calibration_matrix(), dtype=np.float64)
    dist = np.zeros(5)
    if cam.model_name in ("SIMPLE_RADIAL", "RADIAL"):
        dist[0] = cam.params[3]
        if cam.model_name == "RADIAL":
            dist[1] = cam.params[4]
    return K, dist


def _pose(img) -> tuple[np.ndarray, np.ndarray]:
    cfw = img.cam_from_world() if callable(img.cam_from_world) else img.cam_from_world
    return np.asarray(cfw.rotation.matrix()), np.asarray(cfw.translation)


def plane_to_camera(R: np.ndarray, t: np.ndarray, centroid, right, up, u0, v1, res) -> np.ndarray:
    """[R a | R b | R O + t]: facade pixel (x, y, 1) -> camera coords. H_i = K @ this."""
    A = np.column_stack([right * res, -up * res, centroid + u0 * right + v1 * up])
    return R @ A + np.column_stack([np.zeros((3, 2)), t])


def build_facade(image_root: Path, image_names: list[str], detections: list[list[dict]],
                 work_dir: Path, max_facade_px: int = MAX_FACADE_PX) -> dict:
    """Returns {"image", "detections" (facade coords), "registered", "unregistered", "stats"}."""
    m = reconstruct(image_root, image_names, work_dir / "sfm")

    pts = m["xyz"]
    n, centroid, inl = fit_wall_plane(pts)
    poses = list(zip(m["R"], m["t"]))
    centers = np.array([-R.T @ t for R, t in poses])
    if np.mean((centers - centroid) @ n) < 0:
        n = -n  # normal faces the cameras
    up = np.mean([R.T @ np.array([0, -1.0, 0]) for R, _ in poses], 0)  # image-up in world, averaged
    up -= (up @ n) * n
    up /= np.linalg.norm(up)
    right = np.cross(up, n)
    dist_wall = float(np.median((centers - centroid) @ n))

    # facade extent = union of every frame's footprint on the wall (image corners
    # ray-cast onto the plane), so everything filmed is kept, base and roofline included
    uv = []
    for (R, t), K in zip(poses, m["K"]):
        c = -R.T @ t
        w_img, h_img = 2 * K[0, 2], 2 * K[1, 2]
        for px in ((0, 0), (w_img, 0), (w_img, h_img), (0, h_img)):
            ray = R.T @ np.linalg.solve(K, [px[0], px[1], 1.0])
            s = ((centroid - c) @ n) / (ray @ n)
            if s > 0:
                hit = c + s * ray - centroid
                uv.append((hit @ right, hit @ up))
    uv = np.array(uv)
    u0, v0 = uv.min(0)
    u1, v1 = uv.max(0)
    focal = float(np.mean(m["K"][:, [0, 1], [0, 1]]))
    res = max(dist_wall / focal, max(u1 - u0, v1 - v0) / max_facade_px)
    W, H = int(np.ceil((u1 - u0) / res)), int(np.ceil((v1 - v0) / res))

    canvas = np.zeros((H, W, 3), np.uint8)
    owner = np.full((H, W), -1, np.int32)
    best_d = np.full((H, W), np.inf, np.float32)
    yy = np.arange(H, dtype=np.float32)[:, None]
    xx = np.arange(W, dtype=np.float32)[None, :]

    name_to_idx = {name: i for i, name in enumerate(image_names)}
    facade_dets: list[dict] = []
    per_img = []  # (img_idx, K, dist, Hm)
    for k, (name, (R, t), K, dist) in enumerate(zip(m["names"], poses, m["K"], m["dist"])):
        Hm = K @ plane_to_camera(R, t, centroid, right, up, u0, v1, res)
        photo = cv2.imread(str(image_root / name))
        if photo is None:
            continue
        photo = cv2.undistort(photo, K, dist)
        Hinv = np.linalg.inv(Hm)
        warped = cv2.warpPerspective(photo, Hinv, (W, H), flags=cv2.INTER_LINEAR)
        valid = cv2.warpPerspective(np.full(photo.shape[:2], 255, np.uint8), Hinv, (W, H),
                                    flags=cv2.INTER_NEAREST) > 0

        # where this camera's optical axis meets the wall = centre of its Voronoi cell
        c = -R.T @ t
        axis = R.T @ np.array([0, 0, 1.0])
        s = ((centroid - c) @ n) / (axis @ n)
        hit = c + s * axis - centroid
        hx = (hit @ right - u0) / res
        hy = (v1 - hit @ up) / res
        d = np.hypot(xx - hx, yy - hy)
        take = valid & (d < best_d)
        canvas[take] = warped[take]
        best_d[take] = d[take]
        owner[take] = k
        per_img.append((k, name, K, dist, Hinv))
        del warped, valid, d, take

    for k, name, K, dist, Hinv in per_img:
        for det in detections[name_to_idx[name]]:
            fd = _project_detection(det, K, dist, Hinv)
            cx, cy = (np.asarray(fd["bbox"][:2]) + np.asarray(fd["bbox"][2:])) / 2
            xi, yi = int(cx), int(cy)
            # keep anything landing on filmed wall. Copies of one crack from overlapping
            # frames land on the same spot, so CCA merges them into one count. (Keeping
            # only the owner view's copy dropped 138/143 here: the detector is rarely
            # consistent across neighbouring frames, so the owner frame often misses it.)
            if 0 <= xi < W and 0 <= yi < H and owner[yi, xi] >= 0:
                facade_dets.append(fd)

    registered = set(m["names"])
    stats = {
        "registered": len(registered), "total": len(image_names),
        "points3D": len(pts), "reproj_err_px": float(m["reproj_err"]),
        "plane_inliers": int(inl.sum()), "facade_px": (W, H),
    }
    print(f"[ortho] wall plane {inl.sum()}/{len(pts)} points; facade {W}x{H}px; "
          f"{len(facade_dets)}/{sum(len(d) for d in detections)} detections kept "
          f"(on filmed wall)", file=sys.stderr)
    return {"image": canvas, "detections": facade_dets, "stats": stats,
            "unregistered": [nm for nm in image_names if nm not in registered]}


def _project_detection(det: dict, K: np.ndarray, dist: np.ndarray, Hinv: np.ndarray) -> dict:
    if det.get("mask") is not None:
        pts = np.asarray(det["mask"], np.float64)
    else:
        x1, y1, x2, y2 = det["bbox"]
        pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float64)
    und = cv2.undistortPoints(pts.reshape(-1, 1, 2), K, dist, P=K)  # raw frame -> undistorted pixels
    fp = cv2.perspectiveTransform(und, Hinv).reshape(-1, 2).astype(np.float32)
    x1, y1 = fp.min(0)
    x2, y2 = fp.max(0)
    return {"bbox": (float(x1), float(y1), float(x2), float(y2)),
            "mask": fp if det.get("mask") is not None else None, "score": det["score"]}


def _self_check() -> None:
    """ponytail: synthetic wall + camera -- plane fit must recover the wall
    despite 30% clutter, and H must send a facade pixel to the same photo
    pixel as projecting its 3D point directly."""
    rng = np.random.default_rng(1)
    wall = np.column_stack([rng.uniform(-5, 5, 700), rng.uniform(0, 8, 700), np.zeros(700)])
    clutter = rng.uniform([-5, 0, 0.3], [5, 8, 3], (300, 3))
    n, centroid, inl = fit_wall_plane(np.vstack([wall, clutter]))
    assert abs(abs(n[2]) - 1) < 1e-3 and inl[:700].all() and inl[700:].mean() < 0.05, "plane fit"

    n = np.array([0, 0, 1.0])
    up, right = np.array([0, 1.0, 0]), np.cross([0, 1.0, 0], n)
    K = np.array([[1000, 0, 960], [0, 1000, 540], [0, 0, 1.0]])
    ang = np.deg2rad(10)
    yaw = np.array([[np.cos(ang), 0, np.sin(ang)], [0, 1, 0], [-np.sin(ang), 0, np.cos(ang)]])
    R = yaw @ np.diag([1, -1, -1.0])  # camera looks at the wall (-z), image y = world down
    t = -R @ np.array([0.5, 4, 3.0])
    u0, v1, res = -5.0, 8.0, 0.01
    Hm = K @ plane_to_camera(R, t, np.zeros(3), right, up, u0, v1, res)
    x, y = 420.0, 310.0
    X = (u0 + x * res) * right + (v1 - y * res) * up
    p = K @ (R @ X + t)
    q = Hm @ np.array([x, y, 1.0])
    assert np.allclose(p[:2] / p[2], q[:2] / q[2], atol=1e-6), "facade homography"
    print("[ortho] self-check OK", file=sys.stderr)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--sfm"]:
        _sfm_child(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
    else:
        _self_check()
