import argparse
import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh
from PIL import Image

ROBOPEPP_ROOT = Path(__file__).resolve().parent
MULTIHMR_ROOT = ROBOPEPP_ROOT.parents[1]
GMS_ROOT = MULTIHMR_ROOT / "submodules" / "gaussian-mesh-splatting"
CAD_ROOT = GMS_ROOT / "instrument_mesh"
DEFAULT_OUT = ROBOPEPP_ROOT / "assets" / "instrument_surface_samples_surfemb_x2.13mm_wg1over3_shafttop30mm"
DEFAULT_LND_MEMORY = (
    GMS_ROOT
    / "Results2/surgripe_lnd_refine_memory_train/TRAIN/refine_memory_pool.json"
)

if str(ROBOPEPP_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBOPEPP_ROOT))
if str(MULTIHMR_ROOT) not in sys.path:
    sys.path.insert(0, str(MULTIHMR_ROOT))

from instrument_geometry import (  # noqa: E402
    GRIPPER_JOINT_OFFSET_M,
    SHAFT_WRIST_OFFSET_M,
    SURFEMB_GRIPPER_STATIC_THRESHOLD_M,
    fk_matrices_np,
    make_transform_np,
    project_points_np,
)


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
    shaft2world[:3, 3] = np.array([-SHAFT_WRIST_OFFSET_M, 0.0, 0.0])
    l_gripper2world[:3, 3] = np.array([GRIPPER_JOINT_OFFSET_M, 0.0, 0.0])
    r_gripper2world[:3, 3] = np.array([GRIPPER_JOINT_OFFSET_M, 0.0, 0.0])

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


def _shaft_face_weights(mesh, top_len_m, rear_density):
    face_centers = mesh.triangles_center
    x = face_centers[:, 0]
    x_max = float(x.max())
    top_start = x_max - float(top_len_m)
    # Keep high density at the distal 3cm, then decay smoothly to a sparse
    # but non-zero density toward the long rear shaft.
    t = np.clip((x - float(x.min())) / max(1e-8, top_start - float(x.min())), 0.0, 1.0)
    density = float(rear_density) + (1.0 - float(rear_density)) * (t**2)
    density[x >= top_start] = 1.0
    return mesh.area_faces * density, top_start


def _sample_surface(mesh, n_samples, rng, face_weight=None):
    seed = int(rng.integers(0, 2**31 - 1))
    points, face_idx = trimesh.sample.sample_surface(mesh, int(n_samples), face_weight=face_weight, seed=seed)
    return points.astype(np.float32), face_idx.astype(np.int64)


def sample_surface_assets(
    out_dir,
    shaft_samples,
    wrist_samples,
    gripper_samples,
    static_threshold_m,
    shaft_top_len_m,
    shaft_rear_density,
    seed,
):
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
    counts = {
        "shaft": int(shaft_samples),
        "wrist": int(wrist_samples),
        "l_gripper": int(gripper_samples),
        "r_gripper": int(gripper_samples),
    }

    all_points_norm = []
    all_points_part = []
    all_points_canon = []
    all_part_ids = []
    all_effective = []
    all_static = []
    all_names = []
    summary = []
    shaft_top_start = None

    for part_name, mesh in part_meshes.items():
        face_weight = None
        sampling_note = "uniform_area"
        if part_name == "shaft":
            face_weight, shaft_top_start = _shaft_face_weights(mesh, shaft_top_len_m, shaft_rear_density)
            sampling_note = (
                f"x_biased_top_start={shaft_top_start:.9f}m "
                f"top_len={float(shaft_top_len_m):.3f}m rear_density={float(shaft_rear_density):.3f}"
            )

        points_part, face_idx = _sample_surface(mesh, counts[part_name], rng, face_weight=face_weight)
        normals_part = np.asarray(mesh.face_normals[face_idx], dtype=np.float32)
        part2world = bias2world @ np.linalg.inv(bias2part[part_name])
        points_canon_world = _apply_transform(points_part.astype(np.float64), part2world).astype(np.float32)
        points_norm = (points_canon_world / canon_scale).astype(np.float32)
        normals_canon_world = (normals_part @ part2world[:3, :3].T).astype(np.float32)
        normals_canon_world /= np.linalg.norm(normals_canon_world, axis=1, keepdims=True).clip(1e-8)

        original_id = PART_IDS[part_name]
        static = np.zeros((len(points_part),), dtype=bool)
        effective = np.full((len(points_part),), original_id, dtype=np.int64)
        if part_name in ("l_gripper", "r_gripper"):
            static = points_part[:, 0] < float(static_threshold_m)
            effective[static] = PART_IDS["wrist"]

        payload = {
            "part_name": np.array(part_name),
            "part_id": np.array(original_id, dtype=np.int64),
            "points_part_m": points_part.astype(np.float32),
            "points_canon_world_m": points_canon_world.astype(np.float32),
            "points_norm": points_norm.astype(np.float32),
            "normals_part": normals_part.astype(np.float32),
            "normals_canon_world": normals_canon_world.astype(np.float32),
            "face_idx": face_idx.astype(np.int64),
            "canon_scale": np.array(canon_scale, dtype=np.float32),
            "effective_part_ids": effective.astype(np.int64),
            "static_wrist_mask": static.astype(bool),
            "gripper_static_threshold_x_m": np.array(static_threshold_m, dtype=np.float32),
        }
        np.save(out_dir / f"{part_name}_surface_points.npy", payload, allow_pickle=True)

        all_points_norm.append(points_norm.astype(np.float32))
        all_points_part.append(points_part.astype(np.float32))
        all_points_canon.append(points_canon_world.astype(np.float32))
        all_part_ids.append(np.full((len(points_part),), original_id, dtype=np.int64))
        all_effective.append(effective.astype(np.int64))
        all_static.append(static.astype(bool))
        all_names.extend([part_name] * len(points_part))
        summary.append(
            f"{part_name}: n={len(points_part)}, original_part_id={original_id}, "
            f"effective_wrist_static={int(static.sum())}, moving={int((~static).sum())}, "
            f"sampling={sampling_note}, bounds={mesh.bounds.tolist()}"
        )

    combined = {
        "points_norm": np.concatenate(all_points_norm, axis=0).astype(np.float32),
        "points_part_m": np.concatenate(all_points_part, axis=0).astype(np.float32),
        "points_canon_world_m": np.concatenate(all_points_canon, axis=0).astype(np.float32),
        "part_ids": np.concatenate(all_part_ids, axis=0).astype(np.int64),
        "effective_part_ids": np.concatenate(all_effective, axis=0).astype(np.int64),
        "static_wrist_mask": np.concatenate(all_static, axis=0).astype(bool),
        "part_names": np.asarray(all_names),
        "canon_scale": np.array(canon_scale, dtype=np.float32),
        "gripper_static_threshold_x_m": np.array(static_threshold_m, dtype=np.float32),
        "shaft_top_start_x_m": np.array(shaft_top_start, dtype=np.float32),
        "shaft_top_len_m": np.array(shaft_top_len_m, dtype=np.float32),
        "shaft_rear_density": np.array(shaft_rear_density, dtype=np.float32),
    }
    np.save(out_dir / "instrument_surface_points_all.npy", combined, allow_pickle=True)
    (out_dir / "README.txt").write_text(
        "SurfEmb instrument surface samples.\n"
        f"Gripper static rule: points_part_m[:,0] < {float(static_threshold_m):.9f} m "
        f"({float(static_threshold_m) * 1000.0:.3f} mm) => effective_part_ids=2 and no gripper rotation.\n"
        "Wrist and each gripper are sampled at one third of the previous 50k per-part dense default.\n"
        "Shaft is sampled with area weights biased toward the distal x top region.\n"
        "Visible projection visualizations use full mesh-triangle depth buffer plus front-facing normal filtering.\n"
        + "\n".join(summary)
        + "\n",
        encoding="utf-8",
    )
    return combined


def _transform_points_normals(points, normals, T):
    points_cam = points @ T[:3, :3].T + T[:3, 3]
    normals_cam = normals @ T[:3, :3].T
    normals_cam /= np.linalg.norm(normals_cam, axis=1, keepdims=True).clip(1e-8)
    return points_cam, normals_cam


def _transform_surface_payload(part_name, payload, transforms):
    points = np.asarray(payload["points_part_m"], dtype=np.float64)
    normals = np.asarray(payload["normals_part"], dtype=np.float64)
    points_cam = np.empty_like(points)
    normals_cam = np.empty_like(normals)
    static = np.asarray(payload.get("static_wrist_mask", np.zeros((len(points),), dtype=bool))).astype(bool)
    if part_name not in ("l_gripper", "r_gripper"):
        static[:] = False

    moving = ~static
    if moving.any():
        pts, nrm = _transform_points_normals(points[moving], normals[moving], transforms[part_name])
        points_cam[moving] = pts
        normals_cam[moving] = nrm
    if static.any():
        wrist_to_static = make_transform_np(
            np.eye(3),
            [GRIPPER_JOINT_OFFSET_M, 0.0, 0.0],
        )
        pts, nrm = _transform_points_normals(points[static], normals[static], transforms["wrist"] @ wrist_to_static)
        points_cam[static] = pts
        normals_cam[static] = nrm
    return points_cam, normals_cam


def _visible_projected_mask(points_cam, normals_cam, K, image_shape, mesh_depth, min_depth=1e-4, depth_tolerance=0.0008):
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
        & (np.sum(normals_cam * points_cam, axis=1) < 0.0)
    )
    pix = v * w + u
    depth = np.asarray(mesh_depth, dtype=np.float64)
    if depth.shape[:2] != (h, w):
        raise ValueError(f"mesh_depth shape {depth.shape[:2]} does not match {(h, w)}")
    depth_flat = depth.reshape(-1)
    valid_idx = np.flatnonzero(valid)
    mesh_valid = np.zeros_like(valid)
    if len(valid_idx) > 0:
        depth_at_point = depth_flat[pix[valid_idx]]
        mesh_valid[valid_idx] = (depth_at_point > float(min_depth)) & (
            z[valid_idx] <= depth_at_point + float(depth_tolerance)
        )
    valid &= mesh_valid

    zbuf = np.full((h * w,), np.inf, dtype=np.float64)
    np.minimum.at(zbuf, pix[valid], z[valid])
    visible = np.zeros((len(points_cam),), dtype=bool)
    visible[valid] = z[valid] <= zbuf[pix[valid]] + float(depth_tolerance)
    return visible, uv


def _draw_projected_points(rgb, uv, mask, colors, depths, max_points=7200):
    out = rgb.copy()
    idx = np.flatnonzero(mask)
    if len(idx) > int(max_points):
        idx = idx[np.linspace(0, len(idx) - 1, int(max_points)).astype(np.int64)]
    idx = idx[np.argsort(np.asarray(depths)[idx])[::-1]]
    h, w = out.shape[:2]
    for i in idx:
        u, v = uv[i]
        x, y = int(round(float(u))), int(round(float(v)))
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(out, (x, y), 1, tuple(int(c) for c in colors[i]), -1, lineType=cv2.LINE_AA)
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
    rarp_mod = _load_local_module("rarp_hcce_crop_surface_vis_v2", ROBOPEPP_ROOT / "datasets" / "rarp_hcce_crop.py")
    lnd_mod = _load_local_module("lnd_hcce_crop_surface_vis_v2", ROBOPEPP_ROOT / "datasets" / "lnd_hcce_crop.py")
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


def visualize_surface_projections(out_dir, max_points_per_part=1800, device_idx=0):
    out_dir = Path(out_dir)
    vis_dir = out_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)
    payloads = {
        part_name: np.load(out_dir / f"{part_name}_surface_points.npy", allow_pickle=True).item()
        for part_name in PART_MESHES
    }
    datasets = _load_vis_datasets()
    from instrument_opengl_renderer import InstrumentOpenGLDepthRenderer

    renderer = InstrumentOpenGLDepthRenderer(224, 224, device_idx=int(device_idx))
    for ds_name, ds in datasets:
        indices = [0, min(len(ds) - 1, max(1, len(ds) // 2))]
        for ordinal, idx in enumerate(indices):
            _, target = ds[idx]
            rgb = target["orig_rgb"].numpy().astype(np.uint8)
            pose = _pose_from_target(target)
            K = target["K_orig"].numpy().astype(np.float32)
            mesh_color, mesh_depth, _ = renderer.render_pose(pose, K, rgb.shape[:2])
            transforms = fk_matrices_np(
                pose["rot"],
                pose["trans"],
                pose["alpha"],
                pose["theta_l"],
                pose["theta_r"],
            )

            all_points_cam = []
            all_normals_cam = []
            all_colors = []
            for part_name, payload in payloads.items():
                points_cam, normals_cam = _transform_surface_payload(part_name, payload, transforms)
                all_points_cam.append(points_cam)
                all_normals_cam.append(normals_cam)
                part_colors = np.tile(np.asarray(COLORS[part_name], dtype=np.uint8), (len(points_cam), 1))
                if part_name in ("l_gripper", "r_gripper"):
                    static = np.asarray(payload.get("static_wrist_mask", np.zeros((len(points_cam),), dtype=bool))).astype(bool)
                    part_colors[static] = np.asarray(COLORS["wrist"], dtype=np.uint8)
                all_colors.append(part_colors)
            points_cam = np.concatenate(all_points_cam, axis=0)
            normals_cam = np.concatenate(all_normals_cam, axis=0)
            colors = np.concatenate(all_colors, axis=0).astype(np.uint8)
            visible, uv = _visible_projected_mask(points_cam, normals_cam, K, rgb.shape[:2], mesh_depth)
            pc = _draw_projected_points(
                rgb,
                uv,
                visible,
                colors,
                points_cam[:, 2],
                max_points=max_points_per_part * len(PART_MESHES),
            )

            support = np.asarray(mesh_depth) > 0
            mesh_overlay = rgb.copy()
            if support.any():
                mesh_overlay[support] = (
                    rgb[support].astype(np.float32) * 0.2
                    + np.asarray(mesh_color)[support].astype(np.float32) * 0.8
                ).clip(0, 255).astype(np.uint8)
            panel = np.concatenate([rgb, pc, mesh_overlay], axis=1)
            Image.fromarray(panel).save(vis_dir / f"{ds_name}_{ordinal:02d}_visible_surface_points_mesh.jpg")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT))
    parser.add_argument("--shaft_samples", type=int, default=80000)
    parser.add_argument("--wrist_samples", type=int, default=16667)
    parser.add_argument("--gripper_samples", type=int, default=16667)
    parser.add_argument(
        "--static_threshold_mm",
        type=float,
        default=SURFEMB_GRIPPER_STATIC_THRESHOLD_M * 1000.0,
    )
    parser.add_argument("--shaft_top_len_mm", type=float, default=30.0)
    parser.add_argument("--shaft_rear_density", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--vis", type=int, choices=[0, 1], default=1)
    parser.add_argument("--device_idx", type=int, default=0)
    args = parser.parse_args()

    combined = sample_surface_assets(
        out_dir=args.out_dir,
        shaft_samples=args.shaft_samples,
        wrist_samples=args.wrist_samples,
        gripper_samples=args.gripper_samples,
        static_threshold_m=float(args.static_threshold_mm) / 1000.0,
        shaft_top_len_m=float(args.shaft_top_len_mm) / 1000.0,
        shaft_rear_density=args.shaft_rear_density,
        seed=args.seed,
    )
    print(f"saved surface samples to {args.out_dir}")
    print(f"combined points: {combined['points_norm'].shape[0]}")
    print(f"static gripper as wrist: {int(combined['static_wrist_mask'].sum())}")
    if int(args.vis):
        visualize_surface_projections(args.out_dir, device_idx=args.device_idx)
        print(f"saved projection visualizations to {Path(args.out_dir) / 'vis'}")


if __name__ == "__main__":
    main()
