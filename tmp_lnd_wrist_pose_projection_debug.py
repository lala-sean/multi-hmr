#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
import trimesh
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_LND_ROOT = Path("/mnt/iMVR/daiyun/Dataset/LND/TRAIN")
DEFAULT_OUT_DIR = Path("/mnt/iMVR/daiyun/shuojue-temp/lnd_wrist_pose_debug")
DEFAULT_WRIST_MESH = REPO_ROOT / "submodules/gaussian-mesh-splatting/instrument_mesh/transformed_wrist.obj"


# Maps repo wrist local coordinates in millimetres to LND joint.stl coordinates in millimetres.
# Estimated by rigid ICP between the repo wrist mesh after Instrument.bias2wrist and LND/TRAIN/joint.stl.
R_LND_FROM_REPO = np.array(
    [
        [-3.965562107231e-04, -9.999972555239e-01, -2.309044784927e-03],
        [-9.999996414411e-01, 3.948273259691e-04, 7.491522775273e-04],
        [-7.482385475190e-04, 2.309341037986e-03, -9.999970535372e-01],
    ],
    dtype=np.float64,
)
T_LND_FROM_REPO_MM = np.array(
    [3.687278000000e-04, -1.340887690810e-01, 2.029389171000e-03],
    dtype=np.float64,
)


def load_intrinsics(config_path: Path) -> np.ndarray:
    text = config_path.read_text()
    match = re.search(r"data:\s*\[([^\]]+)\]", text, re.S)
    if not match:
        raise ValueError(f"Could not find camera_matrix data in {config_path}")
    vals = [float(v) for v in re.split(r"[,\s]+", match.group(1).strip()) if v]
    if len(vals) != 9:
        raise ValueError(f"Expected 9 intrinsics values, got {len(vals)} from {config_path}")
    return np.asarray(vals, dtype=np.float64).reshape(3, 3)


def repo_bias2wrist() -> np.ndarray:
    bias2world = np.eye(4, dtype=np.float64)
    bias2world[:3, :3] = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64).T
    flip_wrist = np.eye(4, dtype=np.float64)
    flip_wrist[:3, :3] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64).T
    return flip_wrist @ bias2world


def load_repo_wrist_mesh(mesh_path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load_mesh(str(mesh_path), force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
    mesh = mesh.copy()
    mesh.apply_transform(repo_bias2wrist())
    return mesh


def matrix_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    m = np.asarray(R, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    q /= np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    return q


def convert_lnd_pose_to_repo(T_cam_lnd_3x4: np.ndarray) -> dict:
    R_cam_lnd = T_cam_lnd_3x4[:, :3].astype(np.float64)
    t_cam_lnd_mm = T_cam_lnd_3x4[:, 3].astype(np.float64)
    R_cam_repo = R_cam_lnd @ R_LND_FROM_REPO
    t_cam_repo_mm = R_cam_lnd @ T_LND_FROM_REPO_MM + t_cam_lnd_mm
    return {
        "rot": matrix_to_quat_wxyz(R_cam_repo).astype(np.float32),
        "rot_mat": R_cam_repo.astype(np.float64),
        "trans": (t_cam_repo_mm / 1000.0).astype(np.float32),
        "trans_mm": t_cam_repo_mm.astype(np.float64),
        "alpha": 0.0,
        "theta_l": 0.0,
        "theta_r": 0.0,
    }


def project(points_cam_mm: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = points_cam_mm[:, 2]
    valid = z > 1e-6
    uvw = points_cam_mm[valid] @ K.T
    uv = uvw[:, :2] / uvw[:, 2:3]
    return uv, valid


def rasterize_mesh(vertices_cam_mm: np.ndarray, faces: np.ndarray, K: np.ndarray, h: int, w: int) -> np.ndarray:
    uv_all = np.full((vertices_cam_mm.shape[0], 2), np.nan, dtype=np.float64)
    uv_valid, valid = project(vertices_cam_mm, K)
    uv_all[valid] = uv_valid
    face_valid = valid[faces].all(axis=1)
    faces_ok = faces[face_valid]
    if faces_ok.size == 0:
        return np.zeros((h, w), dtype=np.uint8)
    depth = vertices_cam_mm[faces_ok, 2].mean(axis=1)
    order = np.argsort(depth)[::-1]
    mask = np.zeros((h, w), dtype=np.uint8)
    for face in faces_ok[order]:
        pts = np.round(uv_all[face]).astype(np.int32)
        if (
            pts[:, 0].max() < 0
            or pts[:, 1].max() < 0
            or pts[:, 0].min() >= w
            or pts[:, 1].min() >= h
        ):
            continue
        cv2.fillConvexPoly(mask, pts, 255)
    return mask


def overlay_panel(rgb: np.ndarray, gt_mask: np.ndarray, pred_mask: np.ndarray, uv: np.ndarray) -> np.ndarray:
    overlay = rgb.copy()
    gt = gt_mask > 0
    pred = pred_mask > 0
    both = gt & pred
    gt_only = gt & ~pred
    pred_only = pred & ~gt
    overlay[gt_only] = (0.45 * overlay[gt_only] + 0.55 * np.array([255, 60, 60])).astype(np.uint8)
    overlay[pred_only] = (0.45 * overlay[pred_only] + 0.55 * np.array([60, 170, 255])).astype(np.uint8)
    overlay[both] = (0.35 * overlay[both] + 0.65 * np.array([40, 220, 110])).astype(np.uint8)
    for u, v in uv[:: max(1, len(uv) // 2000)]:
        if 0 <= int(round(u)) < rgb.shape[1] and 0 <= int(round(v)) < rgb.shape[0]:
            cv2.circle(overlay, (int(round(u)), int(round(v))), 1, (255, 255, 0), -1)
    gt_rgb = np.dstack([gt_mask, np.zeros_like(gt_mask), np.zeros_like(gt_mask)])
    pred_rgb = np.dstack([np.zeros_like(pred_mask), pred_mask, pred_mask])
    return np.concatenate([rgb, overlay, gt_rgb, pred_rgb], axis=1)


def bbox_from_mask(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lnd-root", type=Path, default=DEFAULT_LND_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--wrist-mesh", type=Path, default=DEFAULT_WRIST_MESH)
    parser.add_argument("--frames", type=str, default="1,100,300,600,900,1147")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    K = load_intrinsics(args.lnd_root / "config.yaml")
    mesh = load_repo_wrist_mesh(args.wrist_mesh)
    vertices_repo_m = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    frame_ids = [int(x) for x in args.frames.split(",") if x.strip()]

    summary = []
    converted = {}
    for frame_id in frame_ids:
        rgb_path = args.lnd_root / "image" / f"{frame_id}.png"
        pose_path = args.lnd_root / "pose" / f"{frame_id}.npy"
        mask_path = args.lnd_root / "mask_original" / f"{frame_id}.png"
        rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
        gt_mask = np.asarray(Image.open(mask_path).convert("L"))
        h, w = rgb.shape[:2]
        T_cam_lnd = np.load(pose_path).astype(np.float64)
        pose_repo = convert_lnd_pose_to_repo(T_cam_lnd)
        vertices_cam_m = vertices_repo_m @ pose_repo["rot_mat"].T + pose_repo["trans"].reshape(1, 3)
        vertices_cam_mm = vertices_cam_m * 1000.0
        uv, valid = project(vertices_cam_mm, K)
        pred_mask = rasterize_mesh(vertices_cam_mm, faces, K, h, w)
        inter = np.logical_and(pred_mask > 0, gt_mask > 0).sum()
        union = np.logical_or(pred_mask > 0, gt_mask > 0).sum()
        iou = float(inter / union) if union > 0 else 0.0
        panel = overlay_panel(rgb, gt_mask, pred_mask, uv)
        out_png = args.out_dir / f"frame_{frame_id:04d}_repo_wrist_projection.png"
        Image.fromarray(panel).save(out_png)
        converted[str(frame_id)] = {
            "rot_wxyz": pose_repo["rot"].tolist(),
            "trans_m": pose_repo["trans"].tolist(),
            "alpha": 0.0,
            "theta_l": 0.0,
            "theta_r": 0.0,
        }
        summary.append(
            {
                "frame": frame_id,
                "image": str(rgb_path),
                "pose": str(pose_path),
                "overlay": str(out_png),
                "iou_repo_wrist_vs_lnd_mask": iou,
                "gt_bbox_xyxy": bbox_from_mask(gt_mask),
                "repo_projection_bbox_xyxy": bbox_from_mask(pred_mask),
                "repo_rot_wxyz": pose_repo["rot"].tolist(),
                "repo_trans_m": pose_repo["trans"].tolist(),
                "valid_projected_vertices": int(valid.sum()),
                "total_vertices": int(len(vertices_repo_m)),
            }
        )

    np.savez(
        args.out_dir / "converted_repo_wrist_poses_sample.npz",
        frame_ids=np.asarray(frame_ids, dtype=np.int32),
        rot_wxyz=np.asarray([converted[str(i)]["rot_wxyz"] for i in frame_ids], dtype=np.float32),
        trans_m=np.asarray([converted[str(i)]["trans_m"] for i in frame_ids], dtype=np.float32),
    )
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
