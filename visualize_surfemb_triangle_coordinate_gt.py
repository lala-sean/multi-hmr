import argparse
import csv
import os
import random
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import visualize_surfemb_opengl_zbuffer_occlusion as data_vis
from instrument_geometry import fk_matrices_np, project_points_np


ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "logs" / "surfemb_triangle_coordinate_gt_vis"
PART_RGB = {
    0: (0, 0, 0),
    1: (60, 130, 245),
    2: (60, 220, 100),
    3: (245, 145, 45),
}
STATUS_RGB = {
    "visible_front": (40, 240, 80),
    "occluded_front": (255, 55, 55),
    "backface": (40, 205, 255),
    "depth_mismatch": (235, 70, 255),
}


def as_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def add_label(image, lines):
    out = np.asarray(image, dtype=np.uint8).copy()
    bar_h = 7 + 18 * len(lines)
    cv2.rectangle(out, (0, 0), (out.shape[1] - 1, bar_h), (0, 0, 0), -1)
    for row, line in enumerate(lines):
        cv2.putText(
            out,
            str(line),
            (6, 17 + 18 * row),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return out


def color_part_mask(part_ids):
    out = np.zeros((*part_ids.shape, 3), dtype=np.uint8)
    for part_id, color in PART_RGB.items():
        out[part_ids == part_id] = color
    return out


def overlay(rgb, color, mask, alpha=0.58):
    out = rgb.copy()
    mask = np.asarray(mask, dtype=bool)
    out[mask] = np.clip(
        out[mask].astype(np.float32) * (1.0 - alpha)
        + np.asarray(color, dtype=np.float32)[mask] * alpha,
        0,
        255,
    ).astype(np.uint8)
    return out


def draw_point_classes(rgb, u, v, classes, rng, max_points=9000):
    out = rgb.copy()
    for name in ("backface", "occluded_front", "depth_mismatch", "visible_front"):
        indices = np.flatnonzero(classes[name])
        if len(indices) > max_points:
            indices = rng.choice(indices, max_points, replace=False)
        for index in indices:
            cv2.circle(
                out,
                (int(u[index]), int(v[index])),
                1,
                STATUS_RGB[name],
                -1,
                cv2.LINE_AA,
            )
    return out


def classify_surface_points(dataset, target, raster_part, raster_depth, raster_valid, tolerance):
    pose = dataset._pose_from_target(target)
    K_crop = as_numpy(target["K"]).astype(np.float64)
    inst = as_numpy(target["inst_mask"]).astype(bool)
    h, w = inst.shape
    transforms = fk_matrices_np(
        pose["rot"], pose["trans"], pose["alpha"], pose["theta_l"], pose["theta_r"]
    )

    points_cam_all = []
    normals_cam_all = []
    effective_ids_all = []
    static_all = []
    for part_name in data_vis.PARTS:
        payload = dataset.surface_payloads[part_name]
        points_cam, normals_cam = data_vis.dataset_mod._transform_surface_payload(
            part_name, payload, transforms
        )
        points_cam_all.append(points_cam)
        normals_cam_all.append(normals_cam)
        effective = np.asarray(payload["effective_part_ids"], dtype=np.int64).copy()
        effective[effective >= 3] = 3
        effective_ids_all.append(effective)
        static_all.append(
            np.asarray(payload.get("static_wrist_mask", np.zeros(len(points_cam))), dtype=bool)
        )

    points_cam = np.concatenate(points_cam_all, axis=0)
    normals_cam = np.concatenate(normals_cam_all, axis=0)
    effective_ids = np.concatenate(effective_ids_all, axis=0)
    static_mask = np.concatenate(static_all, axis=0)
    uv = project_points_np(points_cam, K_crop)
    z = points_cam[:, 2]
    finite = np.isfinite(uv).all(axis=1) & np.isfinite(z)
    u = np.rint(np.nan_to_num(uv[:, 0], nan=-1e9)).astype(np.int64)
    v = np.rint(np.nan_to_num(uv[:, 1], nan=-1e9)).astype(np.int64)
    inside = finite & (z > 1e-4) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    in_inst = np.zeros(len(z), dtype=bool)
    in_inst[inside] = inst[v[inside], u[inside]]
    candidate = inside & in_inst
    front = np.sum(normals_cam * points_cam, axis=1) < 0.0

    depth_at = np.zeros(len(z), dtype=np.float64)
    part_at = np.zeros(len(z), dtype=np.uint8)
    hit = np.zeros(len(z), dtype=bool)
    depth_at[inside] = raster_depth[v[inside], u[inside]]
    part_at[inside] = raster_part[v[inside], u[inside]]
    hit[inside] = raster_valid[v[inside], u[inside]]
    same_part = part_at == effective_ids
    depth_delta = z - depth_at
    visible_front = candidate & front & hit & same_part & (np.abs(depth_delta) <= tolerance)
    occluded_front = candidate & front & hit & ~visible_front & (depth_delta >= -tolerance)
    depth_mismatch = candidate & front & (~hit | (depth_delta < -tolerance))
    backface = candidate & ~front
    return {
        "u": u,
        "v": v,
        "effective_ids": effective_ids,
        "static_mask": static_mask,
        "visible_front": visible_front,
        "occluded_front": occluded_front,
        "depth_mismatch": depth_mismatch,
        "backface": backface,
    }


def coordinate_rgb(coords, valid, coord_min, coord_max):
    scale = np.maximum(coord_max - coord_min, 1e-8)
    normalized = np.clip((coords - coord_min) / scale, 0.0, 1.0)
    out = (normalized * 255.0).astype(np.uint8)
    out[~valid] = 0
    return out


def make_case(dataset, target, renderer, coord_min, coord_max, rng, tolerance):
    rgb = as_numpy(target["crop_rgb"]).astype(np.uint8)
    inst = as_numpy(target["inst_mask"]).astype(bool)
    pose = dataset._pose_from_target(target)
    K_crop = as_numpy(target["K"]).astype(np.float64)
    coords, raster_part, depth, raster_valid = renderer.render_canonical_coordinates(
        pose, K_crop, rgb.shape[:2]
    )
    supervision = raster_valid & inst
    point_stats = classify_surface_points(
        dataset, target, raster_part, depth, raster_valid, tolerance
    )

    part_color = color_part_mask(raster_part)
    part_overlay = overlay(rgb, part_color, raster_valid)
    coord_color = coordinate_rgb(coords, raster_valid, coord_min, coord_max)
    point_panel = draw_point_classes(
        rgb,
        point_stats["u"],
        point_stats["v"],
        point_stats,
        rng,
    )

    intersection = np.zeros_like(rgb)
    intersection[raster_valid & ~inst] = (255, 55, 55)
    intersection[inst & ~raster_valid] = (255, 220, 50)
    intersection[supervision] = (40, 240, 80)

    positives = rgb.copy()
    pos_yx = as_numpy(target["surfemb_mask_samples"]).astype(np.int64)
    pos_ids = as_numpy(target["surfemb_positive_part_ids"]).astype(np.int64)
    for (y, x), part_id in zip(pos_yx, pos_ids):
        cv2.circle(positives, (int(x), int(y)), 1, PART_RGB[int(part_id)], -1, cv2.LINE_AA)

    support_union = np.count_nonzero(raster_valid | inst)
    render_iou = np.count_nonzero(supervision) / max(1, support_union)
    counts = {
        key: int(np.count_nonzero(point_stats[key]))
        for key in ("visible_front", "occluded_front", "backface", "depth_mismatch")
    }
    static_count = int(np.count_nonzero(point_stats["static_mask"]))
    sampled_counts = {
        int(part_id): int(np.count_nonzero(pos_ids == part_id)) for part_id in (1, 2, 3)
    }
    unique_positive_pixels = int(len(np.unique(pos_yx, axis=0)))

    panels = [
        add_label(rgb, ["SurfEmb crop RGB"]),
        add_label(
            part_overlay,
            ["Triangle effective parts", "blue shaft | green wrist/static | orange moving grip"],
        ),
        add_label(coord_color, ["Raster canonical XYZ", "perspective barycentric interpolation"]),
        add_label(
            point_panel,
            [
                "Surface visibility audit",
                "green visible | red occluded | cyan back | magenta mismatch",
                f"vis={counts['visible_front']} occ={counts['occluded_front']} back={counts['backface']}",
            ],
        ),
        add_label(
            intersection,
            ["Raster x SAM/GT visible mask", f"green supervision | IoU={render_iou:.3f}"],
        ),
        add_label(
            positives,
            [
                "Sampled positive pixels 20/60/20",
                f"n={len(pos_yx)} unique={unique_positive_pixels}",
            ],
        ),
    ]
    return np.concatenate(panels, axis=1), {
        "render_iou": float(render_iou),
        "raster_pixels": int(raster_valid.sum()),
        "supervision_pixels": int(supervision.sum()),
        "static_asset_points": static_count,
        "positive_unique_pixels": unique_positive_pixels,
        "positive_shaft": sampled_counts[1],
        "positive_wrist": sampled_counts[2],
        "positive_gripper": sampled_counts[3],
        **counts,
    }


def make_datasets(args):
    common = SimpleNamespace(
        crop_size=224,
        rarp_subsample=int(args.rarp_subsample),
        lnd_subsample=int(args.lnd_subsample),
        out_dir=str(args.out_dir),
        surface_points_path=str(args.surface_points_path),
        n_pos=int(args.n_pos),
        n_neg=int(args.n_neg),
        crop_scale=1.2,
        max_angle=np.pi,
        offset_scale=1.0,
        min_depth=1e-4,
        depth_tolerance=float(args.depth_tolerance),
        lnd_root=str(args.lnd_root),
        lnd_memory=str(args.lnd_memory),
    )
    return (
        ("rarp_train", data_vis.make_rarp("train", True, common)),
        ("rarp_test", data_vis.make_rarp("test", False, common)),
        ("lnd_train", data_vis.make_lnd("TRAIN", True, True, common)),
    )


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = np.load(args.surface_points_path, allow_pickle=True).item()
    surface_coords = np.asarray(payload["points_norm"], dtype=np.float32)
    coord_min = surface_coords.min(axis=0)
    coord_max = surface_coords.max(axis=0)
    rng = np.random.default_rng(args.seed + 41)
    rows = []
    images = []

    for dataset_name, dataset in make_datasets(args):
        for ordinal in range(min(int(args.samples_per_dataset), len(dataset))):
            np.random.seed(args.seed + ordinal)
            _, target = dataset[ordinal]
            renderer = dataset._get_mesh_renderer()
            panel, stats = make_case(
                dataset,
                target,
                renderer,
                coord_min,
                coord_max,
                rng,
                float(args.depth_tolerance),
            )
            frame_id = str(target.get("frame_id", ordinal))
            path = out_dir / f"{dataset_name}_{ordinal:02d}_frame{frame_id}_triangle_coord.jpg"
            Image.fromarray(panel).save(path, quality=94)
            images.append(Image.fromarray(panel))
            row = {"dataset": dataset_name, "ordinal": ordinal, "frame_id": frame_id, **stats}
            rows.append(row)
            print(f"WROTE {path} {stats}", flush=True)

    if rows:
        with (out_dir / "rasterization_stats.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        sheet = Image.new("RGB", (max(im.width for im in images), sum(im.height for im in images)), (0, 0, 0))
        y = 0
        for image in images:
            sheet.paste(image, (0, y))
            y += image.height
        sheet_path = out_dir / "contact_sheet.jpg"
        sheet.save(sheet_path, quality=92)
        print(f"CONTACT_SHEET {sheet_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT))
    parser.add_argument("--surface_points_path", default=str(data_vis.DEFAULT_SURFACE_POINTS))
    parser.add_argument("--lnd_root", default=data_vis.DEFAULT_LND_ROOT)
    parser.add_argument("--lnd_memory", default=data_vis.DEFAULT_LND_MEMORY)
    parser.add_argument("--samples_per_dataset", type=int, default=2)
    parser.add_argument("--rarp_subsample", type=int, default=1000)
    parser.add_argument("--lnd_subsample", type=int, default=1000)
    parser.add_argument("--n_pos", type=int, default=1024)
    parser.add_argument("--n_neg", type=int, default=1024)
    parser.add_argument("--depth_tolerance", type=float, default=8e-4)
    parser.add_argument("--seed", type=int, default=13)
    main(parser.parse_args())
