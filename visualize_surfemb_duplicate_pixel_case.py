import argparse
import random
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial import cKDTree

import train_surfemb_keypoint_crop_rarp_lnd_refinemem as train_module
from instrument_geometry import fk_matrices_np, project_points_np


ROBOPEPP_ROOT = Path(__file__).resolve().parent
DEFAULT_RUN = (
    ROBOPEPP_ROOT
    / "logs/surfemb_keypoint_crop_part206020_fbo4_p2048_b32_gpu0123"
)
PART_COLORS = {
    1: np.array([65, 130, 255], dtype=np.uint8),
    2: np.array([55, 210, 100], dtype=np.uint8),
    3: np.array([245, 80, 80], dtype=np.uint8),
    4: np.array([255, 165, 55], dtype=np.uint8),
}
PART_NAMES = {1: "shaft", 2: "wrist", 3: "left gripper", 4: "right gripper"}


def _load_dataset(checkpoint, dataset_name):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    args = SimpleNamespace(**payload["args"])
    if dataset_name == "rarp_val":
        dataset = train_module.make_rarp_val_dataset(args)
    elif dataset_name == "lnd_val":
        dataset = train_module.make_lnd_val_dataset(args)
    else:
        raise ValueError(f"unsupported dataset: {dataset_name}")
    return dataset


def _part_rgb(mask):
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    rgb[mask == 1] = PART_COLORS[3]
    rgb[mask == 2] = PART_COLORS[2]
    rgb[mask == 3] = PART_COLORS[1]
    return rgb


def _positive_point_geometry(dataset, target):
    dataset_module = train_module._dataset_module
    pose = dataset._pose_from_target(target)
    transforms = fk_matrices_np(
        pose["rot"],
        pose["trans"],
        pose["alpha"],
        pose["theta_l"],
        pose["theta_r"],
    )
    points_norm = []
    points_cam = []
    normals_cam = []
    effective_ids = []
    for part_name in dataset_module.PARTS:
        payload = dataset.surface_payloads[part_name]
        pts_cam, nrm_cam = dataset_module._transform_surface_payload(
            part_name,
            payload,
            transforms,
        )
        points_norm.append(np.asarray(payload["points_norm"], dtype=np.float32))
        points_cam.append(pts_cam)
        normals_cam.append(nrm_cam)
        effective_ids.append(np.asarray(payload["effective_part_ids"], dtype=np.int64))
    return (
        pose,
        np.concatenate(points_norm),
        np.concatenate(points_cam),
        np.concatenate(normals_cam),
        np.concatenate(effective_ids),
    )


def _make_count_overlay(rgb, yx):
    h, w = rgb.shape[:2]
    count = np.zeros((h, w), dtype=np.int32)
    np.add.at(count, (yx[:, 0], yx[:, 1]), 1)
    heat = cv2.applyColorMap(
        np.clip(count / max(1, count.max()) * 255.0, 0, 255).astype(np.uint8),
        cv2.COLORMAP_TURBO,
    )[..., ::-1]
    support = count > 1
    out = rgb.copy()
    out[support] = np.clip(
        0.35 * out[support].astype(np.float32) + 0.65 * heat[support].astype(np.float32),
        0,
        255,
    ).astype(np.uint8)
    return out, count


def visualize_case(dataset, index, seed, out_dir):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    _, target = dataset[index]

    rgb = target["crop_rgb"].numpy()
    yx = target["surfemb_mask_samples"].numpy()
    xyz_pos = target["surfemb_coords_pos"].numpy()
    pos_part_ids = target["surfemb_positive_part_ids"].numpy()
    unique_yx, inverse, counts = np.unique(
        yx,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    max_group = int(np.argmax(counts))
    max_yx = unique_yx[max_group]
    duplicate_rows = np.flatnonzero(inverse == max_group)

    pose, all_norm, all_cam, all_nrm, all_effective_ids = _positive_point_geometry(dataset, target)
    nearest_dist, surface_idx = cKDTree(all_norm).query(xyz_pos[duplicate_rows], k=1)
    if float(np.max(nearest_dist)) > 1e-6:
        raise RuntimeError(f"could not recover exact surface point indices: max distance={nearest_dist.max()}")

    points_cam = all_cam[surface_idx]
    normals_cam = all_nrm[surface_idx]
    recovered_ids = all_effective_ids[surface_idx]
    front_score = np.sum(normals_cam * points_cam, axis=1)
    front_facing = front_score < 0.0

    K_orig = target["K_orig"].numpy()
    M_crop = target["surfemb_M_crop"].numpy()
    orig_h, orig_w = target["orig_size"].numpy().tolist()
    uv_orig = project_points_np(points_cam, K_orig)
    uv_crop = train_module._dataset_module._surf_aug.transform_points(uv_orig, M_crop)
    orig_xy_round = np.rint(uv_orig).astype(np.int64)

    renderer = dataset._get_mesh_renderer()
    mesh_part_orig = renderer.render_pose_mask(pose, K_orig, (orig_h, orig_w))
    mesh_depth = renderer.render_depth(pose, K_orig, (orig_h, orig_w))
    ox = np.clip(orig_xy_round[:, 0], 0, orig_w - 1)
    oy = np.clip(orig_xy_round[:, 1], 0, orig_h - 1)
    depth_at = mesh_depth[oy, ox]
    depth_delta_mm = (points_cam[:, 2] - depth_at) * 1000.0
    mesh_z_pass = (depth_at > dataset.min_depth) & (
        points_cam[:, 2] <= depth_at + dataset.depth_tolerance
    )

    mesh_part_crop = cv2.warpAffine(
        mesh_part_orig,
        M_crop,
        (dataset.crop_size, dataset.crop_size),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    overlay, count_map = _make_count_overlay(rgb, yx)
    v_max, u_max = int(max_yx[0]), int(max_yx[1])

    radius = 12
    x0, x1 = max(0, u_max - radius), min(rgb.shape[1], u_max + radius + 1)
    y0, y1 = max(0, v_max - radius), min(rgb.shape[0], v_max + radius + 1)
    zoom = cv2.resize(
        rgb[y0:y1, x0:x1],
        (320, 320),
        interpolation=cv2.INTER_NEAREST,
    )

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes[0, 0].imshow(overlay)
    axes[0, 0].scatter([u_max], [v_max], s=180, marker="x", c="white", linewidths=3)
    axes[0, 0].set_title(f"Positive multiplicity heatmap; max={len(duplicate_rows)}")

    axes[0, 1].imshow(_part_rgb(mesh_part_crop))
    axes[0, 1].scatter(yx[:, 1], yx[:, 0], s=3, c="white", alpha=0.12)
    axes[0, 1].scatter([u_max], [v_max], s=180, marker="x", c="yellow", linewidths=3)
    axes[0, 1].set_title("OpenGL mesh part z-buffer + sampled positives")

    axes[0, 2].imshow(zoom)
    axes[0, 2].axvline(160, c="yellow", lw=2)
    axes[0, 2].axhline(160, c="yellow", lw=2)
    axes[0, 2].set_title(f"Nearest-neighbor zoom around crop pixel ({u_max}, {v_max})")

    for part_id in sorted(np.unique(recovered_ids)):
        part_mask = recovered_ids == part_id
        color = PART_COLORS[int(part_id)] / 255.0
        axes[1, 0].scatter(
            uv_crop[part_mask, 0] - u_max,
            uv_crop[part_mask, 1] - v_max,
            s=38,
            color=color,
            label=PART_NAMES[int(part_id)],
            alpha=0.8,
        )
    axes[1, 0].axvline(-0.5, c="black", lw=1)
    axes[1, 0].axvline(0.5, c="black", lw=1)
    axes[1, 0].axhline(-0.5, c="black", lw=1)
    axes[1, 0].axhline(0.5, c="black", lw=1)
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_aspect("equal")
    axes[1, 0].legend(loc="best")
    axes[1, 0].set_title("Continuous projections inside the same rounded pixel")
    axes[1, 0].set_xlabel("u - rounded u [pixel]")
    axes[1, 0].set_ylabel("v - rounded v [pixel]")

    colors = np.stack([PART_COLORS[int(v)] for v in recovered_ids]).astype(np.float32) / 255.0
    axes[1, 1].scatter(np.arange(len(duplicate_rows)), depth_delta_mm, c=colors, s=36)
    axes[1, 1].axhline(dataset.depth_tolerance * 1000.0, c="red", ls="--", label="z tolerance")
    axes[1, 1].axhline(0.0, c="black", lw=1)
    axes[1, 1].set_title("Point depth minus mesh z-buffer depth")
    axes[1, 1].set_xlabel("duplicate point")
    axes[1, 1].set_ylabel("depth delta [mm]")
    axes[1, 1].legend(loc="best")

    axes[1, 2].axis("off")
    part_counts = Counter(PART_NAMES[int(v)] for v in recovered_ids)
    original_pixel_count = len(np.unique(orig_xy_round, axis=0))
    summary = [
        f"dataset index: {index}",
        f"video/frame: {target['video_name']} / {target['frame_id']}",
        f"render IoU: {float(target['surfemb_render_iou']):.4f}",
        f"crop pixel (u,v): ({u_max}, {v_max})",
        f"positive points at pixel: {len(duplicate_rows)}",
        f"distinct original pixels: {original_pixel_count}",
        f"front-facing: {int(front_facing.sum())}/{len(front_facing)}",
        f"mesh z-buffer pass: {int(mesh_z_pass.sum())}/{len(mesh_z_pass)}",
        f"camera-z span: {np.ptp(points_cam[:, 2]) * 1000.0:.4f} mm",
        f"depth delta range: [{depth_delta_mm.min():.4f}, {depth_delta_mm.max():.4f}] mm",
        f"part counts: {dict(part_counts)}",
        f"all-positive unique pixel ratio: {len(unique_yx) / len(yx):.4f}",
        f"all-positive max multiplicity: {int(count_map.max())}",
    ]
    axes[1, 2].text(0.0, 1.0, "\n".join(summary), va="top", family="monospace", fontsize=12)
    axes[1, 2].set_title("Geometry audit")

    for ax in axes[0]:
        ax.set_xlim(0, rgb.shape[1] - 1)
        ax.set_ylim(rgb.shape[0] - 1, 0)
    fig.suptitle(
        "SurfEmb duplicate-pixel audit: back-face leakage vs. projection quantization",
        fontsize=16,
    )
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{target['video_name']}_{target['frame_id']}_idx{index:04d}"
    figure_path = out_dir / f"{stem}_duplicate_pixel_audit.png"
    fig.savefig(figure_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    csv_path = out_dir / f"{stem}_duplicate_points.csv"
    rows = np.column_stack(
        [
            duplicate_rows,
            recovered_ids,
            xyz_pos[duplicate_rows],
            uv_orig,
            uv_crop,
            points_cam[:, 2],
            depth_at,
            depth_delta_mm,
            front_score,
            front_facing.astype(np.int64),
            mesh_z_pass.astype(np.int64),
        ]
    )
    np.savetxt(
        csv_path,
        rows,
        delimiter=",",
        header=(
            "positive_row,effective_part_id,xyz_norm_x,xyz_norm_y,xyz_norm_z,"
            "u_orig,v_orig,u_crop,v_crop,z_cam_m,z_mesh_m,depth_delta_mm,"
            "normal_dot_camera_point,front_facing,mesh_zbuffer_pass"
        ),
        comments="",
    )
    print("\n".join(summary), flush=True)
    print(f"figure: {figure_path}", flush=True)
    print(f"csv: {csv_path}", flush=True)
    return figure_path, csv_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_RUN / "checkpoints/last.pt",
    )
    parser.add_argument("--dataset", choices=["rarp_val", "lnd_val"], default="rarp_val")
    parser.add_argument("--index", type=int, default=40)
    parser.add_argument("--seed_base", type=int, default=10000)
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=DEFAULT_RUN / "debug_duplicate_pixel",
    )
    args = parser.parse_args()
    dataset = _load_dataset(args.checkpoint, args.dataset)
    visualize_case(dataset, args.index, args.seed_base + args.index, args.out_dir)


if __name__ == "__main__":
    main()
