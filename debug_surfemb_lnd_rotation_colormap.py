#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import eval_surfemb_articulated_rarp as surf_eval  # noqa: E402
import eval_surfemb_wrist_lnd as lnd_eval  # noqa: E402
from instrument_geometry import GRIPPER_JOINT_OFFSET_M, fk_matrices_np  # noqa: E402
from instrument_opengl_renderer import _load_part_mesh, _split_gripper_mesh_vertices  # noqa: E402
from surfemb_articulated_pose import (  # noqa: E402
    build_part_probability_inputs,
    encode_surface_keys,
    estimate_part_pose_topk_ransac,
    load_part_surfaces,
    prepare_part_score_context,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_EVAL_CSV = (
    ROOT
    / "logs"
    / "surfemb_wrist_lnd_gtroi_topk_ransac_exclude210_full"
    / "surfemb_per_frame.csv"
)
DEFAULT_OUTPUT = ROOT / "logs" / "surfemb_lnd_rotation_colormap_debug"


def font(size):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def read_csv(path):
    return list(csv.DictReader(Path(path).open(encoding="utf-8")))


def write_csv(path, rows):
    if not rows:
        return
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def contiguous_clusters(frame_ids):
    clusters = []
    for frame_id in sorted(int(value) for value in frame_ids):
        if not clusters or frame_id > clusters[-1][-1] + 1:
            clusters.append([frame_id])
        else:
            clusters[-1].append(frame_id)
    return [(values[0], values[-1], len(values)) for values in clusters]


def rotation_distribution(rows, specs, output_dir):
    summary = []
    selected = {}
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
    for model_index, spec in enumerate(specs):
        model_rows = [
            row
            for row in rows
            if row.get("model") == spec["name"] and row.get("status") == "ok" and int(row["frame_id"]) != 210
        ]
        model_rows.sort(key=lambda row: int(row["frame_id"]))
        values = np.asarray([float(row["canonical_rot_err_deg"]) for row in model_rows], dtype=np.float64)
        if not len(values):
            raise RuntimeError(f"No successful rows for model {spec['name']!r} in evaluation CSV")
        quantiles = {
            "min": float(values.min()),
            "p25": float(np.quantile(values, 0.25)),
            "p50": float(np.quantile(values, 0.50)),
            "p75": float(np.quantile(values, 0.75)),
            "p90": float(np.quantile(values, 0.90)),
            "p95": float(np.quantile(values, 0.95)),
            "p99": float(np.quantile(values, 0.99)),
            "max": float(values.max()),
            "mean": float(values.mean()),
        }
        translation = np.asarray([float(row["canonical_trans_err_mm"]) for row in model_rows])
        inlier_fraction = np.asarray([float(row["topk_inlier_fraction"]) for row in model_rows])
        reprojection = np.asarray([float(row["topk_reprojection_median_px"]) for row in model_rows])
        frame_ids = np.asarray([int(row["frame_id"]) for row in model_rows])
        threshold_counts = {f"count_gt_{threshold}deg": int((values > threshold).sum()) for threshold in (20, 30, 45, 60)}
        clusters = contiguous_clusters(frame_ids[values > 30])
        summary.append(
            {
                "model": spec["name"],
                "count": len(values),
                **quantiles,
                **threshold_counts,
                "corr_rot_translation": float(np.corrcoef(values, translation)[0, 1]),
                "corr_rot_inlier_fraction": float(np.corrcoef(values, inlier_fraction)[0, 1]),
                "corr_rot_reprojection": float(np.corrcoef(values, reprojection)[0, 1]),
                "clusters_gt_30deg": json.dumps(clusters),
            }
        )
        color = colors[model_index % len(colors)]
        bins = np.linspace(0.0, max(90.0, float(values.max()) + 1.0), 37)
        axes[0].hist(values, bins=bins, histtype="step", linewidth=2.0, color=color, label=spec["name"])
        sorted_values = np.sort(values)
        axes[1].plot(sorted_values, np.arange(1, len(values) + 1) / len(values), color=color, linewidth=2.0, label=spec["name"])

        categories = []
        for label, target in (("median", quantiles["p50"]), ("p90", quantiles["p90"])):
            row = min(model_rows, key=lambda item: abs(float(item["canonical_rot_err_deg"]) - target))
            categories.append((label, row))
        ranked = sorted(model_rows, key=lambda row: float(row["canonical_rot_err_deg"]), reverse=True)
        for rank, row in enumerate(ranked[: int(spec["top_cases"])], start=1):
            categories.append((f"top{rank}", row))
        deduplicated = []
        used = set()
        for category, row in categories:
            frame_id = int(row["frame_id"])
            if frame_id in used:
                continue
            used.add(frame_id)
            deduplicated.append((category, row))
        selected[spec["name"]] = deduplicated

    axes[0].set_title("LND TEST rotation error histogram")
    axes[0].set_xlabel("rotation error (degree)")
    axes[0].set_ylabel("frames")
    axes[0].grid(alpha=0.25)
    axes[1].set_title("LND TEST rotation error CDF")
    axes[1].set_xlabel("rotation error (degree)")
    axes[1].set_ylabel("fraction of frames")
    axes[1].set_ylim(0.0, 1.01)
    axes[1].grid(alpha=0.25)
    for axis in axes:
        axis.legend()
    figure.savefig(output_dir / "rotation_error_distribution.png", dpi=180)
    plt.close(figure)
    write_csv(output_dir / "rotation_error_distribution.csv", summary)
    return summary, selected


def fit_affine_coords(surface_root, part_name):
    payload = np.load(Path(surface_root) / f"{part_name}_surface_points.npy", allow_pickle=True).item()
    points = np.asarray(payload["points_part_m"], dtype=np.float64)
    coords = np.asarray(payload["points_norm"], dtype=np.float64)
    design = np.concatenate((points, np.ones((len(points), 1), dtype=np.float64)), axis=1)
    affine, _, _, _ = np.linalg.lstsq(design, coords, rcond=None)
    residual = np.abs(design @ affine - coords).max()
    if residual > 1e-5:
        raise RuntimeError(f"Could not recover {part_name} point-to-key affine: max residual={residual}")
    return affine


def apply_affine(points, affine):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    design = np.concatenate((points, np.ones((len(points), 1), dtype=np.float64)), axis=1)
    return (design @ affine).astype(np.float32)


class EffectiveWristCoordRenderer:
    """Render the effective wrist coordinate image with full-mesh occlusion."""

    def __init__(self, size, surface_root, device_idx=0):
        import moderngl

        self.moderngl = moderngl
        self.size = int(size)
        self.ctx = moderngl.create_context(standalone=True, backend="egl", device_index=int(device_idx))
        self.ctx.disable(moderngl.CULL_FACE)
        self.ctx.enable(moderngl.DEPTH_TEST)
        color = self.ctx.renderbuffer((self.size, self.size), components=4, dtype="f4")
        depth = self.ctx.depth_renderbuffer((self.size, self.size))
        self.fbo = self.ctx.framebuffer(color_attachments=[color], depth_attachment=depth)
        self.color_buffer = color
        self.depth_buffer = depth
        self.program = self.ctx.program(
            vertex_shader="""
                #version 330
                uniform mat4 model;
                uniform vec3 k0;
                uniform vec3 k1;
                uniform vec2 image_size;
                uniform float near_z;
                uniform float far_z;
                in vec3 in_vert;
                in vec3 in_coord;
                out vec3 coord;
                void main() {
                    vec4 cam = model * vec4(in_vert, 1.0);
                    float z = cam.z;
                    float u = dot(k0, cam.xyz) / z;
                    float v = dot(k1, cam.xyz) / z;
                    float x_ndc = 2.0 * (u + 0.5) / image_size.x - 1.0;
                    float y_ndc = 2.0 * (v + 0.5) / image_size.y - 1.0;
                    float a = (far_z + near_z) / (far_z - near_z);
                    float b = -2.0 * far_z * near_z / (far_z - near_z);
                    gl_Position = vec4(x_ndc * z, y_ndc * z, a * z + b, z);
                    coord = in_coord;
                }
            """,
            fragment_shader="""
                #version 330
                uniform float show_coord;
                in vec3 coord;
                out vec4 frag_color;
                void main() {
                    frag_color = show_coord > 0.5 ? vec4(coord, 1.0) : vec4(0.0, 0.0, 0.0, 0.0);
                }
            """,
        )
        affines = {name: fit_affine_coords(surface_root, name) for name in ("wrist", "l_gripper", "r_gripper")}
        self.buffers = []
        self.vaos = []
        self.specs = []

        wrist_mesh = _load_part_mesh("wrist")
        wrist_vertices = np.asarray(wrist_mesh.vertices[wrist_mesh.faces], dtype=np.float32).reshape(-1, 3)
        self.add_geometry("wrist", wrist_vertices, apply_affine(wrist_vertices, affines["wrist"]), "wrist", True)

        shaft_mesh = _load_part_mesh("shaft")
        shaft_vertices = np.asarray(shaft_mesh.vertices[shaft_mesh.faces], dtype=np.float32).reshape(-1, 3)
        self.add_geometry("shaft", shaft_vertices, np.zeros_like(shaft_vertices), "shaft", False)

        for name in ("l_gripper", "r_gripper"):
            mesh = _load_part_mesh(name)
            static_vertices, moving_vertices = _split_gripper_mesh_vertices(mesh)
            self.add_geometry(
                f"{name}_static",
                static_vertices,
                apply_affine(static_vertices, affines[name]),
                "static_wrist",
                True,
            )
            self.add_geometry(f"{name}_moving", moving_vertices, np.zeros_like(moving_vertices), name, False)

    def add_geometry(self, name, vertices, coords, transform_kind, show_coord):
        packed = np.concatenate((vertices, coords), axis=1).astype("f4")
        buffer = self.ctx.buffer(packed.tobytes())
        vao = self.ctx.vertex_array(
            self.program,
            [(buffer, "3f 3f", "in_vert", "in_coord")],
        )
        self.buffers.append(buffer)
        self.vaos.append(vao)
        self.specs.append((name, vao, transform_kind, bool(show_coord)))

    @staticmethod
    def transform_for(kind, transforms):
        if kind == "static_wrist":
            offset = np.eye(4, dtype=np.float64)
            offset[:3, 3] = [GRIPPER_JOINT_OFFSET_M, 0.0, 0.0]
            return transforms["wrist"] @ offset
        return transforms[kind]

    def render(self, pose, K):
        transforms = fk_matrices_np(
            pose["rot"], pose["trans"], pose.get("alpha", 0.0), pose.get("theta_l", 0.0), pose.get("theta_r", 0.0)
        )
        self.fbo.use()
        self.ctx.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)
        K = np.asarray(K, dtype=np.float64).reshape(3, 3)
        self.program["k0"].value = tuple(K[0].astype("f4"))
        self.program["k1"].value = tuple(K[1].astype("f4"))
        self.program["image_size"].value = (float(self.size), float(self.size))
        self.program["near_z"].value = 1e-4
        self.program["far_z"].value = 10.0
        for _, vao, transform_kind, show_coord in self.specs:
            transform = self.transform_for(transform_kind, transforms)
            self.program["model"].value = tuple(transform.T.astype("f4").reshape(-1))
            self.program["show_coord"].value = float(show_coord)
            vao.render(mode=self.moderngl.TRIANGLES)
        rgba = np.frombuffer(self.fbo.read(attachment=0, components=4, dtype="f4"), dtype="f4")
        return rgba.reshape(self.size, self.size, 4).copy()

    def release(self):
        for vao in self.vaos:
            vao.release()
        for buffer in self.buffers:
            buffer.release()
        self.program.release()
        self.fbo.release()
        self.color_buffer.release()
        self.depth_buffer.release()
        self.ctx.release()


def surfemb_embedding_vis(embedding, mask=None, demean=None):
    """Exact channel grouping and normalization from SurfEmb.get_emb_vis()."""
    embedding = embedding.float().clone()
    if demean is not None:
        embedding = embedding - demean
    shape = embedding.shape[:-1]
    if embedding.shape[-1] % 3:
        raise ValueError(f"Embedding dimension {embedding.shape[-1]} is not divisible by 3")
    rgb = embedding.reshape(*shape, 3, -1).mean(dim=-1)
    if mask is not None:
        rgb[~mask] = 0.0
    rgb /= torch.abs(rgb).max().clamp_min(1e-9)
    rgb.mul_(0.5).add_(0.5)
    return np.clip(rgb.cpu().numpy() * 255.0, 0, 255).astype(np.uint8)


@torch.inference_mode()
def render_key_colormap(renderer, model, surface, pose, K_crop, device, chunk_size=65536):
    coord_rgba = renderer.render(pose, K_crop)
    visible = coord_rgba[..., 3] > 0.5
    embedding_dim = int(surface.keys.shape[1])
    embedding = torch.zeros((*visible.shape, embedding_dim), device=device, dtype=torch.float32)
    coords = torch.from_numpy(coord_rgba[..., :3][visible]).to(device=device, dtype=torch.float32)
    chunks = []
    for start in range(0, len(coords), int(chunk_size)):
        chunks.append(model.surface_key_mlp(coords[start : start + int(chunk_size)]).float())
    if chunks:
        embedding[torch.from_numpy(visible).to(device)] = torch.cat(chunks, dim=0)
    visible_tensor = torch.from_numpy(visible).to(device)
    key_mean = surface.keys.float().mean(dim=0)
    return surfemb_embedding_vis(embedding, mask=visible_tensor, demean=key_mean), visible


def project_points(points, transform, K):
    points = np.asarray(points, dtype=np.float64)
    transform = np.asarray(transform, dtype=np.float64)
    camera = points @ transform[:3, :3].T + transform[:3, 3]
    uvw = camera @ np.asarray(K, dtype=np.float64).T
    return uvw[:, :2] / np.clip(uvw[:, 2:], 1e-9, None), camera[:, 2]


def spatial_subset(points, count):
    points = np.asarray(points, dtype=np.float64)
    if len(points) <= int(count):
        return np.arange(len(points), dtype=np.int64)
    center = np.median(points, axis=0)
    selected = [int(np.argmax(np.linalg.norm(points - center, axis=1)))]
    min_distance = np.full((len(points),), np.inf, dtype=np.float64)
    for _ in range(1, int(count)):
        distance = np.linalg.norm(points - points[selected[-1]], axis=1)
        min_distance = np.minimum(min_distance, distance)
        min_distance[selected] = -1.0
        selected.append(int(np.argmax(min_distance)))
    return np.asarray(selected, dtype=np.int64)


def square_roi(mask, point_sets, padding=18):
    points = []
    yx = np.argwhere(np.asarray(mask, dtype=bool))
    if len(yx):
        points.append(yx[:, ::-1].astype(np.float64))
    for values in point_sets:
        values = np.asarray(values, dtype=np.float64).reshape(-1, 2)
        values = values[np.isfinite(values).all(axis=1)]
        if len(values):
            points.append(values)
    if not points:
        return 0, 0, mask.shape[1], mask.shape[0]
    points = np.concatenate(points, axis=0)
    h, w = mask.shape
    x0 = max(0.0, float(points[:, 0].min()) - float(padding))
    y0 = max(0.0, float(points[:, 1].min()) - float(padding))
    x1 = min(float(w), float(points[:, 0].max()) + float(padding) + 1.0)
    y1 = min(float(h), float(points[:, 1].max()) + float(padding) + 1.0)
    side = min(float(max(h, w)), max(x1 - x0, y1 - y0, 24.0))
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    x0, y0 = cx - side * 0.5, cy - side * 0.5
    x0 = min(max(0.0, x0), float(w) - side)
    y0 = min(max(0.0, y0), float(h) - side)
    return int(math.floor(x0)), int(math.floor(y0)), int(math.ceil(x0 + side)), int(math.ceil(y0 + side))


def crop_panel(image, roi, panel_size, interpolation):
    x0, y0, x1, y1 = roi
    crop = np.asarray(image, dtype=np.uint8)[y0:y1, x0:x1]
    return cv2.resize(crop, (int(panel_size), int(panel_size)), interpolation=interpolation)


def transform_uv(uv, roi, panel_size):
    x0, y0, x1, y1 = roi
    scale = float(panel_size) / max(1.0, float(x1 - x0))
    return (np.asarray(uv, dtype=np.float64) - np.asarray([x0, y0], dtype=np.float64)) * scale


def label_panel(image, title, subtitle, header_height=76):
    image = Image.fromarray(np.asarray(image, dtype=np.uint8))
    canvas = Image.new("RGB", (image.width, image.height + int(header_height)), (247, 247, 247))
    canvas.paste(image, (0, int(header_height)))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 7), title, fill=(12, 12, 12), font=font(19))
    draw.text((10, 37), subtitle[:82], fill=(55, 55, 55), font=font(13))
    return np.asarray(canvas)


def line_colors(count):
    if count <= 0:
        return []
    hsv = np.zeros((1, count, 3), dtype=np.uint8)
    hsv[0, :, 0] = np.linspace(0, 179, count, endpoint=False).astype(np.uint8)
    hsv[0, :, 1] = 210
    hsv[0, :, 2] = 245
    return [tuple(int(value) for value in color) for color in cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0]]


def compose_case(rgb, query_map, key_map, gt_mask, query_uv, mesh_uv, title, subtitles, args):
    roi = square_roi(gt_mask, (query_uv, mesh_uv), padding=int(args.roi_padding))
    original = crop_panel(rgb, roi, args.panel_size, cv2.INTER_CUBIC)
    query = crop_panel(query_map, roi, args.panel_size, cv2.INTER_NEAREST)
    key = crop_panel(key_map, roi, args.panel_size, cv2.INTER_NEAREST)
    query_uv_panel = transform_uv(query_uv, roi, args.panel_size)
    mesh_uv_panel = transform_uv(mesh_uv, roi, args.panel_size)
    panels = [
        label_panel(original, "RGB crop", subtitles[0]),
        label_panel(query, "Encoder query colormap", subtitles[1]),
        label_panel(key, "Posed 3D wrist key colormap", subtitles[2]),
    ]
    gap = int(args.column_gap)
    height = panels[0].shape[0]
    separator = np.full((height, gap, 3), 255, dtype=np.uint8)
    canvas = np.concatenate((panels[0], separator, panels[1], separator, panels[2]), axis=1)
    header = 76
    x_query = int(args.panel_size) + gap
    x_mesh = 2 * (int(args.panel_size) + gap)
    colors = line_colors(len(query_uv_panel))
    overlay = canvas.copy()
    for query_point, mesh_point, color in zip(query_uv_panel, mesh_uv_panel, colors):
        p_query = (x_query + int(round(query_point[0])), header + int(round(query_point[1])))
        p_mesh = (x_mesh + int(round(mesh_point[0])), header + int(round(mesh_point[1])))
        cv2.line(overlay, p_query, p_mesh, color, 1, cv2.LINE_AA)
    canvas = cv2.addWeighted(canvas, 0.28, overlay, 0.72, 0.0)
    for query_point, mesh_point, color in zip(query_uv_panel, mesh_uv_panel, colors):
        p_rgb = (int(round(query_point[0])), header + int(round(query_point[1])))
        p_query = (x_query + int(round(query_point[0])), header + int(round(query_point[1])))
        p_mesh = (x_mesh + int(round(mesh_point[0])), header + int(round(mesh_point[1])))
        cv2.circle(canvas, p_rgb, 2, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, p_query, 2, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, p_mesh, 2, color, -1, cv2.LINE_AA)

    banner_h = 54
    output = Image.new("RGB", (canvas.shape[1], canvas.shape[0] + banner_h), (238, 241, 244))
    output.paste(Image.fromarray(canvas), (0, banner_h))
    draw = ImageDraw.Draw(output)
    draw.text((12, 13), title, fill=(8, 8, 8), font=font(22))
    return np.asarray(output), roi


@torch.inference_mode()
def diagnose_case(model, surfaces, renderer, dataset, dataset_index, row, category, spec, args, device):
    _, target = dataset[int(dataset_index)]
    image, K_crop, crop_rgb, M_crop = surf_eval.make_surfemb_crop(target, args)
    part_crop = cv2.warpAffine(
        target["part_mask_orig"].detach().cpu().numpy().astype(np.uint8),
        M_crop,
        (int(args.crop_size), int(args.crop_size)),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    gt_wrist = part_crop == 2
    x = image[None].to(device=device, non_blocking=True)
    K_tensor = torch.from_numpy(K_crop)[None].to(device=device, non_blocking=True)
    with torch.amp.autocast(
        device_type=device.type,
        enabled=device.type == "cuda" and bool(args.amp),
        dtype=torch.bfloat16,
    ):
        output = model(x, K_tensor)

    query_flat, _, K_ds, image_hw, _ = build_part_probability_inputs(
        output["inst_mask_logits"][0].float(),
        output["surfemb_queries"][0].float(),
        surfaces,
        K_crop,
        down_sample_scale=int(args.down_sample_scale),
    )
    gt_wrist_tensor = torch.from_numpy(gt_wrist.astype(np.float32)).to(device)
    gt_wrist_ds = F.max_pool2d(
        gt_wrist_tensor[None, None], int(args.down_sample_scale), int(args.down_sample_scale)
    )[0, 0] > 0
    context = prepare_part_score_context(
        query_flat,
        lnd_eval.probability_input_from_mask(gt_wrist_ds, image_hw),
        surfaces["wrist"],
        image_hw,
    )
    best, diagnostics = estimate_part_pose_topk_ransac(
        context,
        surfaces["wrist"],
        K_ds,
        image_hw,
        pixel_mask=gt_wrist_ds.reshape(-1),
        max_correspondences=int(args.topk_max_correspondences),
        min_correspondences=int(args.topk_min_correspondences),
        min_part_probability=float(args.topk_min_part_probability),
        ransac_iterations=int(args.topk_ransac_iterations),
        ransac_reprojection_error=float(args.topk_ransac_reprojection_error),
        ransac_confidence=float(args.topk_ransac_confidence),
        min_inliers=int(args.topk_min_inliers),
        min_inlier_fraction=float(args.topk_min_inlier_fraction),
        return_correspondences=True,
    )
    if best is None:
        raise RuntimeError(f"top-K RANSAC failed for {spec['name']} frame {target['frame_id']}: {diagnostics}")

    raw_pose = {
        "rot": lnd_eval.matrix_to_quat_wxyz(best.transform[:3, :3]),
        "trans": best.transform[:3, 3].copy(),
        "alpha": 0.0,
        "theta_l": 0.0,
        "theta_r": 0.0,
        "chain_cost": float(-best.score),
        "chain_nfev": 0,
    }
    prediction = surf_eval.canonicalize_prediction(raw_pose, args.canonical_eps)
    gt = surf_eval.target_pose(target)
    trans_error = float(np.linalg.norm(np.asarray(prediction["trans"]) - np.asarray(gt["trans"])) * 1000.0)
    rot_error = surf_eval.rotation_error_deg(prediction["rot"], gt["rot"])

    query_hwc = output["surfemb_queries"][0].float().permute(1, 2, 0)
    query_map = surfemb_embedding_vis(
        query_hwc,
        mask=torch.from_numpy(gt_wrist).to(device),
    )
    key_map, key_visible = render_key_colormap(
        renderer,
        model,
        surfaces["wrist"],
        raw_pose,
        K_crop,
        device,
    )

    selected_pixels = diagnostics["topk_selected_pixel_indices"]
    selected_keys = diagnostics["topk_selected_key_indices"]
    inlier_indices = diagnostics["topk_inlier_selected_indices"]
    inlier_pixels = selected_pixels[inlier_indices]
    inlier_keys = selected_keys[inlier_indices]
    h_ds, w_ds = image_hw
    query_uv = np.stack(
        (
            (inlier_pixels % w_ds + 0.5) * int(args.down_sample_scale) - 0.5,
            (inlier_pixels // w_ds + 0.5) * int(args.down_sample_scale) - 0.5,
        ),
        axis=1,
    )
    mesh_uv, depth = project_points(surfaces["wrist"].points_m[inlier_keys], best.transform, K_crop)
    valid = (
        np.isfinite(query_uv).all(axis=1)
        & np.isfinite(mesh_uv).all(axis=1)
        & (depth > 0.0)
        & (mesh_uv[:, 0] >= 0.0)
        & (mesh_uv[:, 0] < int(args.crop_size))
        & (mesh_uv[:, 1] >= 0.0)
        & (mesh_uv[:, 1] < int(args.crop_size))
    )
    query_uv = query_uv[valid]
    mesh_uv = mesh_uv[valid]
    subset = spatial_subset(query_uv, int(args.num_connections))
    query_uv = query_uv[subset]
    mesh_uv = mesh_uv[subset]

    csv_rot_error = float(row["canonical_rot_err_deg"])
    csv_trans_error = float(row["canonical_trans_err_mm"])
    title = (
        f"{spec['name']} | {category} | LND frame {int(target['frame_id'])} | "
        f"rotation={rot_error:.1f} deg, translation={trans_error:.1f} mm"
    )
    subtitles = (
        f"GT wrist ROI; shown points={len(query_uv)}",
        "original SurfEmb get_emb_vis channel grouping",
        "effective wrist; shaft/moving grippers are z-buffer occluders",
    )
    panel, roi = compose_case(
        crop_rgb,
        query_map,
        key_map,
        gt_wrist,
        query_uv,
        mesh_uv,
        title,
        subtitles,
        args,
    )
    return panel, {
        "model": spec["name"],
        "category": category,
        "dataset_idx": int(dataset_index),
        "frame_id": int(target["frame_id"]),
        "csv_rotation_error_deg": csv_rot_error,
        "csv_translation_error_mm": csv_trans_error,
        "rerun_rotation_error_deg": float(rot_error),
        "rerun_translation_error_mm": float(trans_error),
        "gt_wrist_pixels": int(gt_wrist.sum()),
        "mesh_key_pixels": int(key_visible.sum()),
        "ransac_correspondences": int(diagnostics["topk_correspondences"]),
        "ransac_inliers": int(diagnostics["topk_inliers"]),
        "ransac_inlier_fraction": float(diagnostics["topk_inlier_fraction"]),
        "shown_connections": int(len(query_uv)),
        "roi_xyxy": json.dumps([int(value) for value in roi]),
    }


def make_contact_sheet(paths, output_path, max_width=2100):
    images = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        if image.width > int(max_width):
            height = int(round(image.height * int(max_width) / image.width))
            image = image.resize((int(max_width), height), Image.Resampling.LANCZOS)
        images.append(image)
    if not images:
        return
    gap = 14
    sheet = Image.new(
        "RGB",
        (max(image.width for image in images), sum(image.height for image in images) + gap * (len(images) - 1)),
        (255, 255, 255),
    )
    y = 0
    for image in images:
        sheet.paste(image, (0, y))
        y += image.height + gap
    sheet.save(output_path, quality=92)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_csv", default=str(DEFAULT_EVAL_CSV))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--model", action="append", default=None, help="NAME=CHECKPOINT")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--egl_device", type=int, default=0)
    parser.add_argument("--top_cases_per_model", type=int, default=4)
    parser.add_argument("--num_connections", type=int, default=24)
    parser.add_argument("--panel_size", type=int, default=560)
    parser.add_argument("--column_gap", type=int, default=28)
    parser.add_argument("--roi_padding", type=int, default=16)
    parser.add_argument("--lnd_root", default="/mnt/iMVR/daiyun/Dataset/LND")
    parser.add_argument("--surface_root", default=str(surf_eval.DEFAULT_SURFACE_ROOT))
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--surfemb_crop_scale", type=float, default=1.2)
    parser.add_argument("--bbox_padding_frac", type=float, default=0.12)
    parser.add_argument("--canonical_eps", type=float, default=0.08)
    parser.add_argument("--surface_keys_per_part", type=int, default=4096)
    parser.add_argument("--mask_keys_per_part", type=int, default=512)
    parser.add_argument("--surface_seed", type=int, default=2026)
    parser.add_argument("--down_sample_scale", type=int, default=3)
    parser.add_argument("--topk_max_correspondences", type=int, default=512)
    parser.add_argument("--topk_min_correspondences", type=int, default=12)
    parser.add_argument("--topk_min_part_probability", type=float, default=0.05)
    parser.add_argument("--topk_ransac_iterations", type=int, default=2000)
    parser.add_argument("--topk_ransac_reprojection_error", type=float, default=3.0)
    parser.add_argument("--topk_ransac_confidence", type=float, default=0.999)
    parser.add_argument("--topk_min_inliers", type=int, default=8)
    parser.add_argument("--topk_min_inlier_fraction", type=float, default=0.6)
    parser.add_argument("--amp", type=int, choices=(0, 1), default=1)
    return parser


def main(args):
    cv2.setNumThreads(0)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = surf_eval.parse_model_specs(args.model or list(surf_eval.DEFAULT_MODELS))
    for spec in specs:
        spec["top_cases"] = int(args.top_cases_per_model)
    rows = read_csv(args.eval_csv)
    distribution_summary, selected = rotation_distribution(rows, specs, output_dir)

    dataset = lnd_eval.build_dataset(args)
    frame_to_index = {int(sample[0]): index for index, sample in enumerate(dataset.samples)}
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    renderer = EffectiveWristCoordRenderer(args.crop_size, args.surface_root, device_idx=args.egl_device)
    all_rows = []
    all_paths = []
    try:
        for spec in specs:
            model, checkpoint_iter, _ = surf_eval.load_model(spec, device)
            surfaces = load_part_surfaces(
                args.surface_root,
                keys_per_part=int(args.surface_keys_per_part),
                seed=int(args.surface_seed),
            )
            encode_surface_keys(model, surfaces, device, mask_keys_per_part=int(args.mask_keys_per_part))
            model_paths = []
            for category, row in selected[spec["name"]]:
                frame_id = int(row["frame_id"])
                if frame_id not in frame_to_index:
                    raise KeyError(f"Frame {frame_id} is absent from LND TEST dataset")
                panel, result = diagnose_case(
                    model,
                    surfaces,
                    renderer,
                    dataset,
                    frame_to_index[frame_id],
                    row,
                    category,
                    spec,
                    args,
                    device,
                )
                result["checkpoint_iter"] = int(checkpoint_iter)
                path = output_dir / f"{spec['name']}_{category}_frame{frame_id:03d}.jpg"
                Image.fromarray(panel).save(path, quality=95)
                result["visualization"] = str(path)
                all_rows.append(result)
                all_paths.append(path)
                model_paths.append(path)
                print(
                    f"saved {path.name}: r={result['rerun_rotation_error_deg']:.2f}deg "
                    f"t={result['rerun_translation_error_mm']:.2f}mm "
                    f"inliers={result['ransac_inliers']}",
                    flush=True,
                )
            make_contact_sheet(model_paths, output_dir / f"{spec['name']}_contact_sheet.jpg")
            del model, surfaces
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        renderer.release()

    write_csv(output_dir / "selected_case_summary.csv", all_rows)
    make_contact_sheet(all_paths, output_dir / "all_models_contact_sheet.jpg")
    report = [
        "# LND SurfEmb rotation correspondence debug",
        "",
        f"- evaluation CSV: {Path(args.eval_csv).resolve()}",
        "- split: TEST (non-occ), frame 210 excluded by the evaluation CSV.",
        "- matching ROI: GT wrist = SAM wrist intersect rendered visible mask.",
        "- colormap: exact SurfEmb get_emb_vis channel grouping; no PCA.",
        "- right panel: predicted-pose effective wrist coordinate render with full-mesh z-buffer occlusion.",
        "- lines: spatial subset of actual multi-point RANSAC inlier correspondences.",
        "",
        "## Rotation distribution",
        "",
    ]
    for item in distribution_summary:
        report.append(
            f"- {item['model']}: mean={item['mean']:.3f}, p50={item['p50']:.3f}, "
            f"p90={item['p90']:.3f}, p95={item['p95']:.3f}, p99={item['p99']:.3f}, max={item['max']:.3f} deg"
        )
        report.append(
            f"  - counts: >20={item['count_gt_20deg']}, >30={item['count_gt_30deg']}, "
            f">45={item['count_gt_45deg']}, >60={item['count_gt_60deg']} of {item['count']}"
        )
        report.append(
            f"  - correlations: rot/trans={item['corr_rot_translation']:.3f}, "
            f"rot/inlier_fraction={item['corr_rot_inlier_fraction']:.3f}, "
            f"rot/reprojection={item['corr_rot_reprojection']:.3f}"
        )
        report.append(f"  - contiguous >30 deg frame clusters: {item['clusters_gt_30deg']}")
    (output_dir / "README.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"distribution={output_dir / 'rotation_error_distribution.png'}", flush=True)
    print(f"contact_sheet={output_dir / 'all_models_contact_sheet.jpg'}", flush=True)


if __name__ == "__main__":
    main(build_parser().parse_args())
