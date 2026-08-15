import argparse
import csv
import os
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import instrument_opengl_renderer as renderer_mod
import visualize_surfemb_opengl_zbuffer_occlusion as data_vis
from instrument_geometry import fk_matrices_np, project_points_np


ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "logs" / "surfemb_triangle_coordinate_one_to_one_vis"


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


def xyz_colors(xyz, coord_min, coord_max):
    xyz = np.asarray(xyz, dtype=np.float32)
    scale = np.maximum(coord_max - coord_min, 1e-8)
    return (np.clip((xyz - coord_min) / scale, 0.0, 1.0) * 255.0).astype(np.uint8)


def dense_xyz_color(coords, valid, coord_min, coord_max):
    out = xyz_colors(coords, coord_min, coord_max)
    out[~valid] = 0
    return out


def draw_colored_points(image, uv, colors, radius=1):
    out = image.copy()
    h, w = out.shape[:2]
    for point, color in zip(uv, colors):
        if not np.isfinite(point).all():
            continue
        x, y = np.rint(point).astype(np.int64)
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(out, (int(x), int(y)), int(radius), tuple(int(v) for v in color), -1, cv2.LINE_AA)
    return out


def part_color(part_ids):
    colors = np.array(
        [[0, 0, 0], [60, 130, 245], [60, 220, 100], [245, 145, 45]],
        dtype=np.uint8,
    )
    return colors[np.clip(part_ids, 0, 3)]


def coordinate_candidates(pose, renderer):
    transforms = fk_matrices_np(
        pose["rot"], pose["trans"], pose["alpha"], pose["theta_l"], pose["theta_r"]
    )
    return {
        1: (("shaft", transforms["shaft"]),),
        2: (
            ("wrist", transforms["wrist"]),
            ("l_gripper", transforms["wrist"] @ renderer_mod._wrist_to_gripper_origin()),
            ("r_gripper", transforms["wrist"] @ renderer_mod._wrist_to_gripper_origin()),
        ),
        3: (
            ("l_gripper", transforms["l_gripper"]),
            ("r_gripper", transforms["r_gripper"]),
        ),
    }


def reproject_canonical_points(coords, part_ids, pose, K_crop, renderer, raster_depth, query_yx):
    candidates = coordinate_candidates(pose, renderer)
    projected = []
    depth_errors = []
    chosen_parts = []
    for coord, part_id, (query_y, query_x) in zip(coords, part_ids, query_yx):
        canonical_m = np.asarray(coord, dtype=np.float64) * renderer.canonical_scale
        options = []
        for part_name, draw_transform in candidates[int(part_id)]:
            canonical_to_part = np.linalg.inv(renderer_mod._part_to_canonical_matrix(part_name))
            point_part = renderer_mod._transform_points(canonical_m[None], canonical_to_part)
            point_cam = renderer_mod._transform_points(point_part, draw_transform)
            uv = project_points_np(point_cam, K_crop)[0]
            uv_error = float(np.linalg.norm(uv - np.array([query_x, query_y], dtype=np.float64)))
            depth_error = float(abs(point_cam[0, 2] - raster_depth[query_y, query_x]))
            options.append((uv_error + 100.0 * depth_error, uv, depth_error, part_name))
        _, uv, depth_error, part_name = min(options, key=lambda item: item[0])
        projected.append(uv)
        depth_errors.append(depth_error)
        chosen_parts.append(part_name)
    return np.asarray(projected), np.asarray(depth_errors), np.asarray(chosen_parts)


def duplicate_coordinate_error(query_yx, coords):
    groups = {}
    for yx, coord in zip(query_yx, coords):
        groups.setdefault(tuple(int(v) for v in yx), []).append(coord)
    max_span = 0.0
    repeated = 0
    max_multiplicity = 1
    for values in groups.values():
        max_multiplicity = max(max_multiplicity, len(values))
        if len(values) <= 1:
            continue
        repeated += len(values) - 1
        values = np.asarray(values, dtype=np.float64)
        max_span = max(max_span, float(np.linalg.norm(values - values[0], axis=1).max()))
    return len(groups), repeated, max_multiplicity, max_span


def multiplicity_panel(rgb, query_yx):
    h, w = rgb.shape[:2]
    count = np.zeros((h, w), dtype=np.int32)
    np.add.at(count, (query_yx[:, 0], query_yx[:, 1]), 1)
    out = np.zeros_like(rgb)
    out[count == 1] = (50, 220, 90)
    out[(count >= 2) & (count <= 3)] = (255, 220, 50)
    out[count >= 4] = (255, 60, 60)
    return out


def correspondence_strip(query_panel, reproj_panel, query_uv, reproj_uv, colors, rng, count=24):
    h, w = query_panel.shape[:2]
    strip = np.concatenate((query_panel, reproj_panel), axis=1)
    eligible = np.flatnonzero((query_uv[:, 1] > 48) & (reproj_uv[:, 1] > 48))
    if len(eligible) > count:
        eligible = rng.choice(eligible, count, replace=False)
    for index in eligible:
        start = tuple(np.rint(query_uv[index]).astype(int))
        end_local = np.rint(reproj_uv[index]).astype(int)
        end = (int(end_local[0] + w), int(end_local[1]))
        color = tuple(int(v) for v in colors[index])
        cv2.line(strip, start, end, color, 1, cv2.LINE_AA)
        cv2.circle(strip, start, 2, color, -1, cv2.LINE_AA)
        cv2.circle(strip, end, 2, color, -1, cv2.LINE_AA)
    return add_label(
        strip,
        ["Same-color 2D query -> reconstructed 3D key projection", "24 correspondences; lines should be parallel"],
    )


def make_case(dataset, target, coord_min, coord_max, rng):
    rgb = as_numpy(target["crop_rgb"]).astype(np.uint8)
    pose = dataset._pose_from_target(target)
    K_crop = as_numpy(target["K"]).astype(np.float64)
    renderer = dataset._get_mesh_renderer()
    raster_xyz, raster_part, raster_depth, raster_valid = renderer.render_canonical_coordinates(
        pose, K_crop, rgb.shape[:2]
    )
    query_yx = as_numpy(target["surfemb_mask_samples"]).astype(np.int64)
    sampled_xyz = as_numpy(target["surfemb_coords_pos"]).astype(np.float32)
    sampled_part = as_numpy(target["surfemb_positive_part_ids"]).astype(np.int64)
    query_uv = query_yx[:, ::-1].astype(np.float64)
    sampled_colors = xyz_colors(sampled_xyz, coord_min, coord_max)

    raster_xyz_at_queries = raster_xyz[query_yx[:, 0], query_yx[:, 1]]
    raster_part_at_queries = raster_part[query_yx[:, 0], query_yx[:, 1]].astype(np.int64)
    max_xyz_error = float(np.abs(raster_xyz_at_queries - sampled_xyz).max())
    if not np.array_equal(raster_part_at_queries, sampled_part):
        raise AssertionError("Sampled effective part IDs do not match the triangle raster.")

    reprojected_uv, depth_errors, chosen_parts = reproject_canonical_points(
        sampled_xyz,
        sampled_part,
        pose,
        K_crop,
        renderer,
        raster_depth,
        query_yx,
    )
    uv_errors = np.linalg.norm(reprojected_uv - query_uv, axis=1)
    unique_count, repeats, max_multiplicity, duplicate_xyz_span = duplicate_coordinate_error(
        query_yx, sampled_xyz
    )

    dense = dense_xyz_color(raster_xyz, raster_valid, coord_min, coord_max)
    query_points = draw_colored_points(rgb, query_uv, sampled_colors, radius=1)
    mesh_background = (0.32 * rgb + 0.68 * part_color(raster_part)).astype(np.uint8)
    mesh_background[~raster_valid] = 0
    reproj_points = draw_colored_points(mesh_background, reprojected_uv, sampled_colors, radius=1)
    multiplicity = multiplicity_panel(rgb, query_yx)
    panels = [
        add_label(rgb, ["SurfEmb crop RGB"]),
        add_label(dense, ["Dense triangle XYZ colors", "same global canonical color function"]),
        add_label(query_points, ["Training query pixels", "color = sampled canonical XYZ"]),
        add_label(reproj_points, ["Same 3D keys reprojected", "inverse canonical -> FK -> K_crop"]),
        add_label(
            multiplicity,
            ["Pixel multiplicity", "green=1 yellow=2-3 red>=4", f"duplicate XYZ span={duplicate_xyz_span:.1e}"],
        ),
    ]
    top = np.concatenate(panels, axis=1)
    strip = correspondence_strip(query_points, reproj_points, query_uv, reprojected_uv, sampled_colors, rng)
    bottom = np.zeros((strip.shape[0], top.shape[1], 3), dtype=np.uint8)
    bottom[:, : strip.shape[1]] = strip
    result = np.concatenate((top, bottom), axis=0)
    stats = {
        "n_positive": int(len(query_yx)),
        "unique_pixels": int(unique_count),
        "repeated_samples": int(repeats),
        "max_multiplicity": int(max_multiplicity),
        "max_duplicate_xyz_span": float(duplicate_xyz_span),
        "max_raster_xyz_error": float(max_xyz_error),
        "mean_uv_reprojection_error_px": float(uv_errors.mean()),
        "max_uv_reprojection_error_px": float(uv_errors.max()),
        "max_depth_reprojection_error_m": float(depth_errors.max()),
        "chosen_shaft": int(np.count_nonzero(chosen_parts == "shaft")),
        "chosen_wrist": int(np.count_nonzero(chosen_parts == "wrist")),
        "chosen_static_gripper": int(
            np.count_nonzero((sampled_part == 2) & (chosen_parts != "wrist"))
        ),
        "chosen_moving_gripper": int(np.count_nonzero(sampled_part == 3)),
    }
    return result, stats


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
        depth_tolerance=8e-4,
        lnd_root=str(args.lnd_root),
        lnd_memory=str(args.lnd_memory),
    )
    return (
        ("rarp_train", data_vis.make_rarp("train", True, common)),
        ("rarp_test", data_vis.make_rarp("test", False, common)),
        ("lnd_train", data_vis.make_lnd("TRAIN", True, True, common)),
    )


def main(args):
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed + 73)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = np.load(args.surface_points_path, allow_pickle=True).item()
    surface_xyz = np.asarray(payload["points_norm"], dtype=np.float32)
    coord_min = surface_xyz.min(axis=0)
    coord_max = surface_xyz.max(axis=0)
    rows = []
    images = []
    for dataset_name, dataset in make_datasets(args):
        for ordinal in range(min(args.samples_per_dataset, len(dataset))):
            np.random.seed(args.seed + ordinal)
            _, target = dataset[ordinal]
            image, stats = make_case(dataset, target, coord_min, coord_max, rng)
            frame_id = str(target.get("frame_id", ordinal))
            path = out_dir / f"{dataset_name}_{ordinal:02d}_frame{frame_id}_xyz_one_to_one.jpg"
            Image.fromarray(image).save(path, quality=95)
            images.append(Image.fromarray(image))
            rows.append({"dataset": dataset_name, "ordinal": ordinal, "frame_id": frame_id, **stats})
            print(f"WROTE {path} {stats}", flush=True)

    with (out_dir / "one_to_one_stats.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    sheet = Image.new("RGB", (max(image.width for image in images), sum(image.height for image in images)))
    y = 0
    for image in images:
        sheet.paste(image, (0, y))
        y += image.height
    sheet_path = out_dir / "contact_sheet.jpg"
    sheet.save(sheet_path, quality=93)
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
    parser.add_argument("--seed", type=int, default=29)
    main(parser.parse_args())
