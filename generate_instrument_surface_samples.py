import argparse
import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh
from PIL import Image

ROBOPEPP_ROOT = Path(__file__).resolve().parent
MULTIHMR_ROOT = ROBOPEPP_ROOT.parents[1]
GMS_ROOT = MULTIHMR_ROOT / "submodules" / "gaussian-mesh-splatting"
CAD_ROOT = GMS_ROOT / "instrument_mesh"
DEFAULT_OUT = ROBOPEPP_ROOT / "assets" / "instrument_surface_samples"
DEFAULT_LND_MEMORY = (
    GMS_ROOT
    / "Results2/surgripe_lnd_refine_memory_train/TRAIN/refine_memory_pool.json"
)

if str(ROBOPEPP_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBOPEPP_ROOT))
if str(MULTIHMR_ROOT) not in sys.path:
    sys.path.insert(0, str(MULTIHMR_ROOT))

from instrument_geometry import fk_matrices_np, make_transform_np, project_points_np  # noqa: E402


PART_MESHES = {
    "shaft": "transformed_shaft.obj",
    "wrist": "transformed_wrist.obj",
    "l_gripper": "transformed_gripper_left.obj",
    "r_gripper": "transformed_gripper_right.obj",
}
PART_IDS = {"shaft": 1, "wrist": 2, "l_gripper": 3, "r_gripper": 4}
COLORS = {
    "shaft": (80, 150, 255),
    "wrist": (80, 230, 120),
    "l_gripper": (255, 90, 90),
    "r_gripper": (255, 180, 60),
}


def _load_local_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bias_matrices():
    bias2world = np.eye(4, dtype=np.float64)
    bias2world[:3, :3] = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64).T
    flip_wrist = np.eye(4, dtype=np.float64)
    flip_wrist[:3, :3] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64).T

    shaft2world = np.eye(4, dtype=np.float64)
    wrist2world = np.eye(4, dtype=np.float64)
    l_gripper2world = np.eye(4, dtype=np.float64)
    r_gripper2world = np.eye(4, dtype=np.float64)
    shaft2world[:3, 3] = np.array([-0.2159, 0.0, 0.0])
    l_gripper2world[:3, 3] = np.array([0.009, 0.0, 0.0])
    r_gripper2world[:3, 3] = np.array([0.009, 0.0, 0.0])

    return bias2world, {
        "shaft": np.linalg.inv(shaft2world) @ bias2world,
        "wrist": np.linalg.inv(wrist2world) @ flip_wrist @ bias2world,
        "l_gripper": np.linalg.inv(l_gripper2world) @ bias2world,
        "r_gripper": np.linalg.inv(r_gripper2world) @ bias2world,
    }


def _load_part_mesh(part_name):
    path = CAD_ROOT / PART_MESHES[part_name]
    mesh = trimesh.load_mesh(path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected Trimesh at {path}, got {type(mesh)}")
    return mesh


def _apply_transform(points, T):
    return points @ T[:3, :3].T + T[:3, 3]


def sample_surface_assets(out_dir, samples_per_part, seed):
    rng = np.random.default_rng(int(seed))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bias2world, bias2part = _bias_matrices()

    part_meshes = {}
    for part_name in PART_MESHES:
        mesh = _load_part_mesh(part_name)
        mesh.apply_transform(bias2part[part_name])
        part_meshes[part_name] = mesh

    wrist_vertices = np.asarray(part_meshes["wrist"].vertices, dtype=np.float64)
    canon_scale = float(2.0 * (wrist_vertices.max(axis=0) - wrist_vertices.min(axis=0)).max())

    all_points_norm = []
    all_points_part = []
    all_part_ids = []
    all_names = []
    summary = []
    for part_name, mesh in part_meshes.items():
        n_samples = int(samples_per_part.get(part_name, samples_per_part["default"]))
        state = np.random.get_state()
        np.random.seed(int(rng.integers(0, 2**31 - 1)))
        points_part, face_idx = trimesh.sample.sample_surface(mesh, n_samples)
        np.random.set_state(state)
        points_part = points_part.astype(np.float32)
        normals_part = np.asarray(mesh.face_normals[face_idx], dtype=np.float32)

        part2world = bias2world @ np.linalg.inv(bias2part[part_name])
        points_canon_world = _apply_transform(points_part.astype(np.float64), part2world).astype(np.float32)
        points_norm = (points_canon_world / canon_scale).astype(np.float32)
        normals_canon_world = (normals_part @ part2world[:3, :3].T).astype(np.float32)
        normals_canon_world /= np.linalg.norm(normals_canon_world, axis=1, keepdims=True).clip(1e-8)

        payload = {
            "part_name": np.array(part_name),
            "part_id": np.array(PART_IDS[part_name], dtype=np.int64),
            "points_part_m": points_part,
            "points_canon_world_m": points_canon_world,
            "points_norm": points_norm,
            "normals_part": normals_part,
            "normals_canon_world": normals_canon_world,
            "face_idx": face_idx.astype(np.int64),
            "canon_scale": np.array(canon_scale, dtype=np.float32),
        }
        np.save(out_dir / f"{part_name}_surface_points.npy", payload, allow_pickle=True)
        summary.append(
            f"{part_name}: n={n_samples}, faces={len(mesh.faces)}, area={mesh.area:.8f}, "
            f"part_bounds={mesh.bounds.tolist()}"
        )
        all_points_norm.append(points_norm)
        all_points_part.append(points_part)
        all_part_ids.append(np.full((n_samples,), PART_IDS[part_name], dtype=np.int64))
        all_names.extend([part_name] * n_samples)

    combined = {
        "points_norm": np.concatenate(all_points_norm, axis=0).astype(np.float32),
        "points_part_m": np.concatenate(all_points_part, axis=0).astype(np.float32),
        "part_ids": np.concatenate(all_part_ids, axis=0).astype(np.int64),
        "part_names": np.asarray(all_names),
        "canon_scale": np.array(canon_scale, dtype=np.float32),
    }
    np.save(out_dir / "instrument_surface_points_all.npy", combined, allow_pickle=True)
    (out_dir / "README.txt").write_text(
        "Uniform area-weighted trimesh.sample.sample_surface points.\n"
        "points_part_m are in per-part FK coordinates; points_norm match rendered coord_img[..., :3].\n"
        + "\n".join(summary)
        + "\n",
        encoding="utf-8",
    )
    return combined, part_meshes, canon_scale


def _draw_points(rgb, points_cam, K, color, radius=1):
    out = rgb.copy()
    z = points_cam[:, 2]
    keep = z > 1e-4
    if keep.any():
        uv = project_points_np(points_cam[keep], K)
        h, w = out.shape[:2]
        inside = (
            np.isfinite(uv).all(axis=1)
            & (uv[:, 0] >= 0)
            & (uv[:, 0] < w)
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < h)
        )
        for u, v in uv[inside]:
            cv2.circle(out, (int(round(u)), int(round(v))), int(radius), color, -1, lineType=cv2.LINE_AA)
    return out


def _transform_points_normals(points, normals, T):
    points_cam = points @ T[:3, :3].T + T[:3, 3]
    if normals is None:
        return points_cam, None
    normals_cam = normals @ T[:3, :3].T
    normals_cam /= np.linalg.norm(normals_cam, axis=1, keepdims=True).clip(1e-8)
    return points_cam, normals_cam


def _transform_surface_payload(part_name, payload, transforms):
    points = np.asarray(payload["points_part_m"], dtype=np.float64)
    normals = np.asarray(payload["normals_part"], dtype=np.float64) if "normals_part" in payload else None
    points_cam = np.empty_like(points)
    normals_cam = np.empty_like(normals) if normals is not None else None

    if part_name in ("l_gripper", "r_gripper") and "static_wrist_mask" in payload:
        static = np.asarray(payload["static_wrist_mask"]).astype(bool)
    else:
        static = np.zeros((len(points),), dtype=bool)

    moving = ~static
    if moving.any():
        pts, nrm = _transform_points_normals(
            points[moving],
            normals[moving] if normals is not None else None,
            transforms[part_name],
        )
        points_cam[moving] = pts
        if normals_cam is not None:
            normals_cam[moving] = nrm

    if static.any():
        # These are rear gripper samples embedded in the wrist. Keep their local
        # gripper-joint offset, but do not apply the left/right gripper rotation.
        wrist_to_static = make_transform_np(np.eye(3), [0.009, 0.0, 0.0])
        pts, nrm = _transform_points_normals(
            points[static],
            normals[static] if normals is not None else None,
            transforms["wrist"] @ wrist_to_static,
        )
        points_cam[static] = pts
        if normals_cam is not None:
            normals_cam[static] = nrm

    return points_cam, normals_cam


def _visible_projected_mask(
    points_cam,
    normals_cam,
    K,
    image_shape,
    mesh_depth=None,
    min_depth=1e-4,
    depth_tolerance=0.0008,
):
    h, w = int(image_shape[0]), int(image_shape[1])
    z = points_cam[:, 2]
    uv = project_points_np(points_cam, K)
    u = np.rint(uv[:, 0]).astype(np.int64)
    v = np.rint(uv[:, 1]).astype(np.int64)
    valid = (
        np.isfinite(uv).all(axis=1)
        & (z > float(min_depth))
        & (u >= 0)
        & (u < w)
        & (v >= 0)
        & (v < h)
    )
    if normals_cam is not None:
        valid &= np.sum(normals_cam * points_cam, axis=1) < 0.0
    pix = v * w + u
    if mesh_depth is not None:
        depth = np.asarray(mesh_depth, dtype=np.float64)
        if depth.shape[:2] != (h, w):
            raise ValueError(f"mesh_depth shape {depth.shape[:2]} does not match image shape {(h, w)}")
        depth_flat = depth.reshape(-1)
        valid_idx = np.flatnonzero(valid)
        if len(valid_idx) > 0:
            depth_at_point = depth_flat[pix[valid_idx]]
            keep = (depth_at_point > float(min_depth)) & (z[valid_idx] <= depth_at_point + float(depth_tolerance))
            mesh_valid = np.zeros_like(valid)
            mesh_valid[valid_idx] = keep
            valid &= mesh_valid
    zbuf = np.full((h * w,), np.inf, dtype=np.float64)
    np.minimum.at(zbuf, pix[valid], z[valid])
    visible = np.zeros((len(points_cam),), dtype=bool)
    visible[valid] = z[valid] <= zbuf[pix[valid]] + float(depth_tolerance)
    return visible, uv


def _draw_projected_points(rgb, uv, mask, colors, depths=None, max_points=6400, radius=1):
    out = rgb.copy()
    idx = np.flatnonzero(mask)
    if len(idx) > int(max_points):
        idx = idx[np.linspace(0, len(idx) - 1, int(max_points)).astype(np.int64)]
    if depths is not None and len(idx) > 0:
        z = np.asarray(depths, dtype=np.float64).reshape(-1)
        idx = idx[np.argsort(z[idx])[::-1]]
    h, w = out.shape[:2]
    for i in idx:
        u, v = uv[i]
        if not np.isfinite(u) or not np.isfinite(v):
            continue
        x, y = int(round(float(u))), int(round(float(v)))
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(out, (x, y), int(radius), tuple(int(c) for c in colors[i]), -1, lineType=cv2.LINE_AA)
    return out


def _pose_from_target(target):
    action = target["action"].detach().cpu().numpy().reshape(3)
    return {
        "rot": target["wrist_quat"].detach().cpu().numpy(),
        "trans": target["wrist_trans"].detach().cpu().numpy(),
        "alpha": float(action[0]),
        "theta_l": float(action[1]),
        "theta_r": float(action[2]),
    }


def _load_vis_datasets():
    rarp_mod = _load_local_module("rarp_hcce_crop_surface_vis", ROBOPEPP_ROOT / "datasets" / "rarp_hcce_crop.py")
    lnd_mod = _load_local_module("lnd_hcce_crop_surface_vis", ROBOPEPP_ROOT / "datasets" / "lnd_hcce_crop.py")
    rarp_ds = rarp_mod.RARPCropHCCEDataset(
        "/mnt/nas/share/shuojue/data/needlePuncture_videos",
        "/mnt/nas/share/shuojue/data/needlePuncture_results",
        split="train",
        training=False,
        crop_size=224,
        train_ratio=0.95,
        subsample=50,
        min_dice=(0.8, 0.6, 0.6),
        bbox_padding_frac=0.12,
        color_jitter=False,
        rgb_augmentation=False,
        occlusion_augmentation=False,
        render_on_the_fly=True,
        require_cse=True,
    )
    lnd_ds = lnd_mod.SurgripeLNDHCCECropDataset(
        root="/mnt/iMVR/daiyun/Dataset/LND",
        split="TRAIN",
        training=False,
        crop_size=224,
        memory_path=str(DEFAULT_LND_MEMORY),
        use_memory_pose=True,
        bbox_padding_frac=0.12,
        color_jitter=False,
        rgb_augmentation=False,
        occlusion_augmentation=False,
        render_on_the_fly=True,
        require_cse=True,
    )
    return [("rarp", rarp_ds), ("surgripe_lnd", lnd_ds)]


def visualize_surface_projections(out_dir, combined, max_points_per_part=1600):
    vis_dir = Path(out_dir) / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)
    part_points = {}
    for part_name in PART_MESHES:
        payload = np.load(Path(out_dir) / f"{part_name}_surface_points.npy", allow_pickle=True).item()
        part_points[part_name] = payload["points_part_m"]

    datasets = _load_vis_datasets()
    renderer = None
    if torch.cuda.is_available():
        from instrument_trimesh_renderer import GMSInstrumentTrimeshRenderer

        renderer = GMSInstrumentTrimeshRenderer(torch.device("cuda:0"))

    for ds_name, ds in datasets:
        indices = [0, min(len(ds) - 1, max(1, len(ds) // 2))]
        for ordinal, idx in enumerate(indices):
            _, target = ds[idx]
            rgb = target["orig_rgb"].numpy().astype(np.uint8)
            pose = _pose_from_target(target)
            K = target["K_orig"].numpy().astype(np.float32)
            transforms = fk_matrices_np(
                pose["rot"],
                pose["trans"],
                pose["alpha"],
                pose["theta_l"],
                pose["theta_r"],
            )
            mesh_color, mesh_depth = None, None
            if renderer is not None:
                mesh_color, mesh_depth, _ = renderer.render_pose(pose, K, rgb.shape[:2])

            all_points_cam = []
            all_normals_cam = []
            all_colors = []
            for part_name, pts in part_points.items():
                payload = np.load(Path(out_dir) / f"{part_name}_surface_points.npy", allow_pickle=True).item()
                pts_cam, normals_cam = _transform_surface_payload(part_name, payload, transforms)
                all_points_cam.append(pts_cam)
                all_normals_cam.append(normals_cam)
                all_colors.extend([COLORS[part_name]] * len(pts_cam))

            points_cam = np.concatenate(all_points_cam, axis=0)
            normals_cam = np.concatenate(all_normals_cam, axis=0) if all(n is not None for n in all_normals_cam) else None
            colors = np.asarray(all_colors, dtype=np.uint8)
            visible, uv = _visible_projected_mask(points_cam, normals_cam, K, rgb.shape[:2], mesh_depth=mesh_depth)
            pc = _draw_projected_points(
                rgb,
                uv,
                visible,
                colors,
                depths=points_cam[:, 2],
                max_points=max_points_per_part * len(PART_MESHES),
                radius=1,
            )

            panels = [rgb, pc]
            if mesh_color is not None and mesh_depth is not None:
                support = np.asarray(mesh_depth) > 0
                mesh_overlay = rgb.copy()
                if support.any():
                    mesh_overlay[support] = (
                        rgb[support].astype(np.float32) * 0.2
                        + np.asarray(mesh_color)[support].astype(np.float32) * 0.8
                    ).clip(0, 255).astype(np.uint8)
                panels.append(mesh_overlay)
            panel = np.concatenate(panels, axis=1)
            path = vis_dir / f"{ds_name}_{ordinal:02d}_surface_points_mesh.jpg"
            Image.fromarray(panel).save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT))
    parser.add_argument("--samples_per_part", type=int, default=50000)
    parser.add_argument("--shaft_samples", type=int, default=80000)
    parser.add_argument("--seed", type=int, default=20260705)
    parser.add_argument("--vis", type=int, choices=[0, 1], default=1)
    args = parser.parse_args()

    counts = {
        "default": int(args.samples_per_part),
        "shaft": int(args.shaft_samples),
    }
    combined, _, _ = sample_surface_assets(args.out_dir, counts, args.seed)
    print(f"saved surface samples to {args.out_dir}")
    print(f"combined points: {combined['points_norm'].shape[0]}")
    if int(args.vis):
        visualize_surface_projections(args.out_dir, combined)
        print(f"saved projection visualizations to {Path(args.out_dir) / 'vis'}")


if __name__ == "__main__":
    main()
