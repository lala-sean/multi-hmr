#!/usr/bin/env python3
import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

import debug_surfemb_lnd_rotation_colormap as vis_base
import eval_surfemb_articulated_rarp as surf_eval
import eval_surfemb_wrist_lnd as lnd_eval
from surfemb_articulated_pose import (
    PART_NAMES,
    build_part_probability_inputs,
    encode_surface_keys,
    estimate_part_pose_topk_ransac,
    load_part_surfaces,
    prepare_part_score_context,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_EVAL_ROOT = ROOT / "logs" / "surfemb_current_last_lnd_eval_20260804_iter21000_22000"
DEFAULT_OUTPUT = DEFAULT_EVAL_ROOT / "full_gtroi_vs_wrist_correspondence_vis"
DEFAULT_FRAMES = (339, 344, 170, 254, 358, 342)

GREEN = (34, 197, 94)
RED = (239, 68, 68)
CYAN = (34, 211, 238)
ORANGE = (249, 115, 22)
YELLOW = (250, 204, 21)
WHITE = (248, 250, 252)
INK = (15, 23, 42)


def read_rows(path):
    with Path(path).open(encoding="utf-8") as handle:
        return {int(row["frame_id"]): row for row in csv.DictReader(handle) if row["status"] == "ok"}


def write_rows(path, rows):
    if not rows:
        return
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def pose_from_row(row):
    return {
        "rot": np.asarray(
            [row["canonical_qw"], row["canonical_qx"], row["canonical_qy"], row["canonical_qz"]],
            dtype=np.float64,
        ),
        "trans": np.asarray(
            [row["canonical_tx_m"], row["canonical_ty_m"], row["canonical_tz_m"]],
            dtype=np.float64,
        ),
        "alpha": 0.0,
        "theta_l": 0.0,
        "theta_r": 0.0,
    }


def pose_transform(pose):
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = surf_eval.quat_wxyz_to_matrix_np(pose["rot"])
    transform[:3, 3] = np.asarray(pose["trans"], dtype=np.float64)
    return transform


def resize_mask(mask, size):
    return cv2.resize(np.asarray(mask, dtype=np.uint8), (size, size), interpolation=cv2.INTER_NEAREST) > 0


def draw_contour(image, mask, color, thickness=2):
    output = np.asarray(image, dtype=np.uint8).copy()
    contours, _ = cv2.findContours(np.asarray(mask, dtype=np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(output, contours, -1, tuple(int(value) for value in color), int(thickness), cv2.LINE_AA)
    return output


def fill_mask(image, mask, color, alpha):
    output = np.asarray(image, dtype=np.uint8).copy()
    overlay = output.copy()
    overlay[np.asarray(mask, dtype=bool)] = np.asarray(color, dtype=np.uint8)
    return cv2.addWeighted(output, 1.0 - float(alpha), overlay, float(alpha), 0.0)


def dashed_line(image, p0, p1, color, thickness=2, segments=8):
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    for index in range(0, int(segments), 2):
        a = p0 + (p1 - p0) * (index / float(segments))
        b = p0 + (p1 - p0) * ((index + 1) / float(segments))
        cv2.line(
            image,
            tuple(np.rint(a).astype(int)),
            tuple(np.rint(b).astype(int)),
            color,
            int(thickness),
            cv2.LINE_AA,
        )


def draw_pose_axes(image, pose, K, predicted=False):
    transform = pose_transform(pose)
    points = np.asarray(
        [[0.0, 0.0, 0.0], [0.004, 0.0, 0.0], [0.0, 0.004, 0.0], [0.0, 0.0, 0.004]],
        dtype=np.float64,
    )
    uv, depth = vis_base.project_points(points, transform, K)
    if not np.isfinite(uv).all() or np.any(depth <= 0.0):
        return None
    origin = tuple(np.rint(uv[0]).astype(int))
    colors = ((245, 158, 11), (244, 63, 94), (168, 85, 247)) if predicted else ((34, 211, 238), (34, 197, 94), (59, 130, 246))
    for endpoint, color in zip(uv[1:], colors):
        if predicted:
            dashed_line(image, uv[0], endpoint, color, thickness=2)
        else:
            cv2.line(image, origin, tuple(np.rint(endpoint).astype(int)), color, 2, cv2.LINE_AA)
    cv2.circle(image, origin, 4, ORANGE if predicted else GREEN, -1, cv2.LINE_AA)
    return np.asarray(uv[0], dtype=np.float64)


def labeled_panel(image, title, subtitle, width, image_height):
    image = Image.fromarray(np.asarray(image, dtype=np.uint8)).resize((int(width), int(image_height)), Image.Resampling.LANCZOS)
    header = 64
    canvas = Image.new("RGB", (int(width), int(image_height) + header), (248, 250, 252))
    canvas.paste(image, (0, header))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), title, fill=INK, font=vis_base.font(19))
    draw.text((12, 35), subtitle[:100], fill=(71, 85, 105), font=vis_base.font(13))
    return canvas


def subset_status_points(query_uv, inlier_mask, max_inliers, max_outliers):
    selected = []
    for state, count in ((True, max_inliers), (False, max_outliers)):
        indices = np.flatnonzero(np.asarray(inlier_mask, dtype=bool) == state)
        if len(indices) == 0:
            continue
        local = vis_base.spatial_subset(np.asarray(query_uv)[indices], min(int(count), len(indices)))
        selected.extend(indices[local].tolist())
    return np.asarray(selected, dtype=np.int64)


def similarity_scale(source, target):
    """Least-squares 2D similarity scale from source points to target points."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if len(source) < 2:
        return float("nan")
    source_centered = source - source.mean(axis=0, keepdims=True)
    target_centered = target - target.mean(axis=0, keepdims=True)
    denominator = float(np.square(source_centered).sum())
    if denominator <= 1e-12:
        return float("nan")
    singular_values = np.linalg.svd(source_centered.T @ target_centered, compute_uv=False)
    return float(singular_values.sum() / denominator)


def make_colormap_pair(query_map, key_map, query_uv, gt_uv, inlier_mask, width, height):
    gap = 24
    half = (int(width) - gap) // 2
    query = cv2.resize(query_map, (half, int(height)), interpolation=cv2.INTER_NEAREST)
    key = cv2.resize(key_map, (half, int(height)), interpolation=cv2.INTER_NEAREST)
    canvas = np.full((int(height), int(width), 3), 248, dtype=np.uint8)
    canvas[:, :half] = query
    canvas[:, half + gap : half + gap + half] = key
    sx = half / float(query_map.shape[1])
    sy = int(height) / float(query_map.shape[0])
    overlay = canvas.copy()
    for q, g, is_inlier in zip(query_uv, gt_uv, inlier_mask):
        p0 = (int(round(q[0] * sx)), int(round(q[1] * sy)))
        p1 = (half + gap + int(round(g[0] * sx)), int(round(g[1] * sy)))
        color = GREEN if bool(is_inlier) else RED
        cv2.line(overlay, p0, p1, color, 1, cv2.LINE_AA)
    canvas = cv2.addWeighted(canvas, 0.42, overlay, 0.58, 0.0)
    for q, g, is_inlier in zip(query_uv, gt_uv, inlier_mask):
        p0 = (int(round(q[0] * sx)), int(round(q[1] * sy)))
        p1 = (half + gap + int(round(g[0] * sx)), int(round(g[1] * sy)))
        color = GREEN if bool(is_inlier) else RED
        cv2.circle(canvas, p0, 2, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, p1, 3, color, 1, cv2.LINE_AA)
    cv2.putText(canvas, "query embedding", (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.52, WHITE, 1, cv2.LINE_AA)
    cv2.putText(canvas, "GT posed key embedding", (half + gap + 8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.52, WHITE, 1, cv2.LINE_AA)
    return canvas


def make_correspondence_overlay(rgb, query_uv, gt_uv, inlier_mask):
    image = np.asarray(rgb, dtype=np.uint8).copy()
    dark = np.zeros_like(image)
    image = cv2.addWeighted(image, 0.72, dark, 0.28, 0.0)
    overlay = image.copy()
    for q, g, is_inlier in zip(query_uv, gt_uv, inlier_mask):
        color = GREEN if bool(is_inlier) else RED
        cv2.line(overlay, tuple(np.rint(q).astype(int)), tuple(np.rint(g).astype(int)), color, 1, cv2.LINE_AA)
    image = cv2.addWeighted(image, 0.35, overlay, 0.65, 0.0)
    for q, g, is_inlier in zip(query_uv, gt_uv, inlier_mask):
        color = GREEN if bool(is_inlier) else RED
        cv2.circle(image, tuple(np.rint(q).astype(int)), 2, color, -1, cv2.LINE_AA)
        cv2.circle(image, tuple(np.rint(g).astype(int)), 3, color, 1, cv2.LINE_AA)
    cv2.putText(image, "dot=query | ring=GT key", (7, image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.43, WHITE, 1, cv2.LINE_AA)
    return image


def make_pose_overlay(rgb, gt_mask, gt_mesh_mask, pred_mesh_mask, gt_pose, pred_pose, K):
    image = fill_mask(rgb, gt_mask, GREEN, 0.10)
    image = fill_mask(image, pred_mesh_mask, ORANGE, 0.10)
    image = draw_contour(image, gt_mask, GREEN, 2)
    image = draw_contour(image, gt_mesh_mask, CYAN, 1)
    image = draw_contour(image, pred_mesh_mask, ORANGE, 2)
    gt_origin = draw_pose_axes(image, gt_pose, K, predicted=False)
    pred_origin = draw_pose_axes(image, pred_pose, K, predicted=True)
    if gt_origin is not None and pred_origin is not None:
        cv2.arrowedLine(
            image,
            tuple(np.rint(gt_origin).astype(int)),
            tuple(np.rint(pred_origin).astype(int)),
            YELLOW,
            2,
            cv2.LINE_AA,
            tipLength=0.18,
        )
        cv2.putText(image, "GT", tuple(np.rint(gt_origin + [5, -5]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, GREEN, 1, cv2.LINE_AA)
        cv2.putText(image, "P", tuple(np.rint(pred_origin + [5, 12]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, ORANGE, 1, cv2.LINE_AA)
    return image


def build_column(result, args):
    width = int(args.column_width)
    square = int(args.square_height)
    pair_height = int(args.pair_height)
    row_gap = 12
    rgb = result["crop_rgb"]
    input_overlay = fill_mask(rgb, result["gt_wrist"], GREEN, 0.08)
    input_overlay = draw_contour(input_overlay, result["gt_wrist"], GREEN, 2)
    input_overlay = draw_contour(input_overlay, result["matching_mask"], YELLOW, 2)

    pair = make_colormap_pair(
        result["query_map"],
        result["gt_key_map"],
        result["shown_query_uv"],
        result["shown_gt_uv"],
        result["shown_inlier_mask"],
        width,
        pair_height,
    )
    corr = make_correspondence_overlay(
        rgb,
        result["shown_query_uv"],
        result["shown_gt_uv"],
        result["shown_inlier_mask"],
    )
    pose = make_pose_overlay(
        rgb,
        result["gt_wrist"],
        result["gt_mesh_mask"],
        result["pred_mesh_mask"],
        result["gt_pose"],
        result["pred_pose"],
        result["K_crop"],
    )

    panels = [
        labeled_panel(
            input_overlay,
            "Input crop + matching ROI",
            f"green=GT wrist, yellow={result['roi_label']}, ROI@74={result['matching_area_ds']} px",
            width,
            square,
        ),
        labeled_panel(
            pair,
            "Original SurfEmb embedding colors",
            f"green=inlier, red=outlier; {result['shown_count']} actual top-K pairs shown",
            width,
            pair_height,
        ),
        labeled_panel(
            corr,
            "Correspondence error against GT pose",
            f"GT residual={result['gt_inlier_residual_median_px']:.2f}px, pair scale={result['gt_inlier_pair_scale']:.3f}, pred/GT z={result['pred_gt_z_ratio']:.3f}",
            width,
            square,
        ),
        labeled_panel(
            pose,
            "GT and fitted wrist pose",
            f"green=GT mask, cyan=GT mesh, orange=fit; origin arrow; dt=({result['dt_x_mm']:+.1f},{result['dt_y_mm']:+.1f},{result['dt_z_mm']:+.1f})mm",
            width,
            square,
        ),
    ]
    column_header = 96
    total_height = column_header + sum(panel.height for panel in panels) + row_gap * (len(panels) - 1)
    column = Image.new("RGB", (width, total_height), (241, 245, 249))
    draw = ImageDraw.Draw(column)
    draw.text((14, 10), result["display_name"], fill=INK, font=vis_base.font(25))
    draw.text(
        (14, 46),
        f"{result['crop_label']} | t={result['csv_trans_err_mm']:.2f}mm  r={result['csv_rot_err_deg']:.2f}deg",
        fill=(51, 65, 85),
        font=vis_base.font(16),
    )
    draw.text(
        (14, 72),
        f"top-K={result['csv_correspondences']}  inliers={result['csv_inliers']} ({result['csv_inlier_fraction'] * 100:.1f}%)",
        fill=(71, 85, 105),
        font=vis_base.font(13),
    )
    y = column_header
    for panel in panels:
        column.paste(panel, (0, y))
        y += panel.height + row_gap
    return column


def compose_frame(full, wrist, args):
    left = build_column(full, args)
    right = build_column(wrist, args)
    if left.height != right.height:
        raise RuntimeError("Full and wrist-only columns have different heights")
    gap = 20
    banner = 96
    output = Image.new("RGB", (left.width + right.width + gap, left.height + banner), (226, 232, 240))
    output.paste(left, (0, banner))
    output.paste(right, (left.width + gap, banner))
    draw = ImageDraw.Draw(output)
    rot_gain = full["csv_rot_err_deg"] - wrist["csv_rot_err_deg"]
    trans_change = wrist["csv_trans_err_mm"] - full["csv_trans_err_mm"]
    draw.text((16, 10), f"LND frame {full['frame_id']}: Full vs Wrist-only SurfEmb", fill=INK, font=vis_base.font(27))
    draw.text(
        (16, 50),
        f"Wrist-only rotation gain {rot_gain:+.2f} deg; translation change {trans_change:+.2f} mm (positive = worse)",
        fill=(51, 65, 85),
        font=vis_base.font(17),
    )
    return output


@torch.inference_mode()
def diagnose_model(model, surfaces, renderer, target, row, config, args, device):
    args.crop_mask_source = config["crop_source"]
    args.predicted_wrist_mask_mode = config["mask_mode"]
    image, K_crop, crop_rgb, M_crop = lnd_eval.make_eval_crop(target, args)
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
    query_flat, part_inputs, K_ds, image_hw, object_prob = build_part_probability_inputs(
        output["inst_mask_logits"][0].float(),
        output["surfemb_queries"][0].float(),
        surfaces,
        K_crop,
        down_sample_scale=int(args.down_sample_scale),
    )
    if config["roi_source"] == "gt_wrist":
        gt_wrist_tensor = torch.as_tensor(gt_wrist, device=device, dtype=torch.float32)
        gt_wrist_roi = F.max_pool2d(
            gt_wrist_tensor[None, None],
            int(args.down_sample_scale),
            int(args.down_sample_scale),
        )[0, 0] > 0
        if tuple(gt_wrist_roi.shape) != tuple(image_hw):
            raise ValueError(f"GT wrist ROI shape {tuple(gt_wrist_roi.shape)} != output shape {tuple(image_hw)}")
        context_input = lnd_eval.probability_input_from_mask(gt_wrist_roi, image_hw)
        matching_roi = gt_wrist_roi.reshape(-1)
    elif config["mask_mode"] == "binary_head":
        context_input = lnd_eval.probability_input_from_binary_logits(
            output["inst_mask_logits"][0], int(args.down_sample_scale)
        )
        matching_roi = context_input["prob"] >= float(args.topk_min_object_probability)
    else:
        context_input = part_inputs["wrist"]
        part_probability = torch.stack([part_inputs[name]["prob"] for name in PART_NAMES], dim=1)
        wrist_index = PART_NAMES.index("wrist")
        matching_roi = (
            (part_probability.argmax(dim=1) == wrist_index)
            & (object_prob >= float(args.topk_min_object_probability))
        )
    context = prepare_part_score_context(query_flat, context_input, surfaces["wrist"], image_hw)
    cv2.setRNGSeed(int(args.ransac_seed) + int(row["frame_id"]))
    best, diagnostics = estimate_part_pose_topk_ransac(
        context,
        surfaces["wrist"],
        K_ds,
        image_hw,
        pixel_mask=matching_roi,
        max_correspondences=int(args.topk_max_correspondences),
        min_correspondences=int(args.topk_min_correspondences),
        min_part_probability=float(args.topk_min_part_probability),
        margin_power=float(args.topk_margin_power),
        ransac_iterations=int(args.topk_ransac_iterations),
        ransac_reprojection_error=float(args.topk_ransac_reprojection_error),
        ransac_confidence=float(args.topk_ransac_confidence),
        min_inliers=int(args.topk_min_inliers),
        min_inlier_fraction=0.0,
        return_correspondences=True,
    )
    if best is None:
        raise RuntimeError(f"RANSAC failed for {config['name']} frame {row['frame_id']}: {diagnostics}")

    pred_pose = pose_from_row(row)
    gt_pose = surf_eval.target_pose(target)
    pred_transform = pose_transform(pred_pose)
    gt_transform = pose_transform(gt_pose)
    selected_pixels = np.asarray(diagnostics["topk_selected_pixel_indices"], dtype=np.int64)
    selected_keys = np.asarray(diagnostics["topk_selected_key_indices"], dtype=np.int64)
    inlier_indices = np.asarray(diagnostics["topk_inlier_selected_indices"], dtype=np.int64)
    inlier_mask = np.zeros(len(selected_pixels), dtype=bool)
    inlier_mask[inlier_indices] = True
    h_ds, w_ds = image_hw
    query_uv = np.stack(
        (
            (selected_pixels % w_ds + 0.5) * int(args.down_sample_scale) - 0.5,
            (selected_pixels // w_ds + 0.5) * int(args.down_sample_scale) - 0.5,
        ),
        axis=1,
    )
    key_points = surfaces["wrist"].points_m[selected_keys]
    gt_uv, gt_depth = vis_base.project_points(key_points, gt_transform, K_crop)
    pred_uv, pred_depth = vis_base.project_points(key_points, pred_transform, K_crop)
    valid = (
        np.isfinite(query_uv).all(axis=1)
        & np.isfinite(gt_uv).all(axis=1)
        & np.isfinite(pred_uv).all(axis=1)
        & (gt_depth > 0.0)
        & (pred_depth > 0.0)
        & (gt_uv[:, 0] >= 0.0)
        & (gt_uv[:, 0] < int(args.crop_size))
        & (gt_uv[:, 1] >= 0.0)
        & (gt_uv[:, 1] < int(args.crop_size))
    )
    query_uv, gt_uv, pred_uv, inlier_mask = query_uv[valid], gt_uv[valid], pred_uv[valid], inlier_mask[valid]
    shown = subset_status_points(query_uv, inlier_mask, args.num_inliers, args.num_outliers)
    shown_query_uv = query_uv[shown]
    shown_gt_uv = gt_uv[shown]
    shown_pred_uv = pred_uv[shown]
    shown_inlier_mask = inlier_mask[shown]

    query_hwc = output["surfemb_queries"][0].float().permute(1, 2, 0)
    matching_mask = resize_mask(matching_roi.reshape(image_hw).detach().cpu().numpy(), int(args.crop_size))
    query_map = vis_base.surfemb_embedding_vis(
        query_hwc,
        mask=torch.from_numpy(matching_mask).to(device),
    )
    gt_key_map, _ = vis_base.render_key_colormap(
        renderer,
        model,
        surfaces["wrist"],
        gt_pose,
        K_crop,
        device,
    )
    gt_key_map[~gt_wrist] = 0
    gt_rgba = renderer.render(gt_pose, K_crop)
    pred_rgba = renderer.render(pred_pose, K_crop)
    gt_mesh_mask = gt_rgba[..., 3] > 0.5
    pred_mesh_mask = pred_rgba[..., 3] > 0.5

    gt_residual = query_uv - gt_uv
    gt_inlier_residual = np.linalg.norm(gt_residual[inlier_mask], axis=1)
    gt_inlier_bias = gt_residual[inlier_mask].mean(axis=0) if np.any(inlier_mask) else np.asarray([np.nan, np.nan])
    gt_inlier_pair_scale = similarity_scale(gt_uv[inlier_mask], query_uv[inlier_mask])
    dt_mm = (np.asarray(pred_pose["trans"]) - np.asarray(gt_pose["trans"])) * 1000.0
    return {
        "name": config["name"],
        "display_name": config["display_name"],
        "crop_label": config["crop_label"],
        "roi_label": config["roi_label"],
        "frame_id": int(row["frame_id"]),
        "crop_rgb": crop_rgb,
        "M_crop": M_crop,
        "K_crop": K_crop,
        "gt_wrist": gt_wrist,
        "matching_mask": matching_mask,
        "matching_area_ds": int(matching_roi.sum().item()),
        "query_map": query_map,
        "gt_key_map": gt_key_map,
        "shown_query_uv": shown_query_uv,
        "shown_gt_uv": shown_gt_uv,
        "shown_pred_uv": shown_pred_uv,
        "shown_inlier_mask": shown_inlier_mask,
        "shown_count": int(len(shown)),
        "gt_mesh_mask": gt_mesh_mask,
        "pred_mesh_mask": pred_mesh_mask,
        "gt_pose": gt_pose,
        "pred_pose": pred_pose,
        "csv_trans_err_mm": float(row["canonical_trans_err_mm"]),
        "csv_rot_err_deg": float(row["canonical_rot_err_deg"]),
        "csv_correspondences": int(row["topk_correspondences"]),
        "csv_inliers": int(row["topk_inliers"]),
        "csv_inlier_fraction": float(row["topk_inlier_fraction"]),
        "rerun_correspondences": int(diagnostics["topk_correspondences"]),
        "rerun_inliers": int(diagnostics["topk_inliers"]),
        "gt_inlier_residual_median_px": float(np.median(gt_inlier_residual)) if len(gt_inlier_residual) else float("nan"),
        "gt_inlier_bias_x_px": float(gt_inlier_bias[0]),
        "gt_inlier_bias_y_px": float(gt_inlier_bias[1]),
        "gt_inlier_pair_scale": gt_inlier_pair_scale,
        "pred_gt_z_ratio": float(pred_pose["trans"][2] / gt_pose["trans"][2]),
        "dt_x_mm": float(dt_mm[0]),
        "dt_y_mm": float(dt_mm[1]),
        "dt_z_mm": float(dt_mm[2]),
    }


def contact_sheet(paths, output_path, thumb_width=1040):
    images = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        height = int(round(image.height * int(thumb_width) / image.width))
        images.append(image.resize((int(thumb_width), height), Image.Resampling.LANCZOS))
    if not images:
        return
    gap = 14
    canvas = Image.new("RGB", (int(thumb_width), sum(image.height for image in images) + gap * (len(images) - 1)), WHITE)
    y = 0
    for image in images:
        canvas.paste(image, (0, y))
        y += image.height + gap
    canvas.save(output_path, quality=92)


def evaluation_stats(rows):
    values = list(rows.values())
    return {
        "matching_area_mean": float(np.mean([float(row["matching_wrist_area_ds"]) for row in values])),
        "matching_area_median": float(np.median([float(row["matching_wrist_area_ds"]) for row in values])),
        "correspondences_mean": float(np.mean([float(row["topk_correspondences"]) for row in values])),
        "correspondences_median": float(np.median([float(row["topk_correspondences"]) for row in values])),
        "inlier_fraction_mean": float(np.mean([float(row["topk_inlier_fraction"]) for row in values])),
        "fit_reprojection_mean": float(np.mean([float(row["topk_reprojection_median_px"]) for row in values])),
    }


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full_checkpoint", default=str(DEFAULT_EVAL_ROOT / "checkpoints/full_last_iter21000.pt"))
    parser.add_argument("--wrist_checkpoint", default=str(DEFAULT_EVAL_ROOT / "checkpoints/wristonly_last_iter22000.pt"))
    parser.add_argument("--full_csv", default=str(DEFAULT_EVAL_ROOT / "full_instance_crop_gt_wrist_roi/surfemb_per_frame.csv"))
    parser.add_argument("--wrist_csv", default=str(DEFAULT_EVAL_ROOT / "wrist_crop/surfemb_per_frame.csv"))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--frame_ids", nargs="*", type=int, default=list(DEFAULT_FRAMES))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--egl_device", type=int, default=0)
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
    parser.add_argument("--topk_min_object_probability", type=float, default=0.5)
    parser.add_argument("--topk_margin_power", type=float, default=0.0)
    parser.add_argument("--topk_ransac_iterations", type=int, default=2000)
    parser.add_argument("--topk_ransac_reprojection_error", type=float, default=3.0)
    parser.add_argument("--topk_ransac_confidence", type=float, default=0.999)
    parser.add_argument("--topk_min_inliers", type=int, default=8)
    parser.add_argument("--ransac_seed", type=int, default=20260804)
    parser.add_argument("--num_inliers", type=int, default=24)
    parser.add_argument("--num_outliers", type=int, default=12)
    parser.add_argument("--column_width", type=int, default=590)
    parser.add_argument("--square_height", type=int, default=500)
    parser.add_argument("--pair_height", type=int, default=285)
    parser.add_argument("--amp", type=int, choices=(0, 1), default=1)
    return parser


def main(args):
    cv2.setNumThreads(0)
    torch.set_num_threads(2)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    full_rows = read_rows(args.full_csv)
    wrist_rows = read_rows(args.wrist_csv)
    frame_ids = [int(frame) for frame in args.frame_ids]
    missing = [frame for frame in frame_ids if frame not in full_rows or frame not in wrist_rows]
    if missing:
        raise KeyError(f"Frames missing from evaluation CSVs: {missing}")

    dataset = lnd_eval.build_dataset(args)
    frame_to_index = {int(sample[0]): index for index, sample in enumerate(dataset.samples)}
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    renderer = vis_base.EffectiveWristCoordRenderer(args.crop_size, args.surface_root, device_idx=args.egl_device)
    configs = [
        {
            "name": "resnet_full",
            "display_name": "Full ResNet iter21000 + GT wrist ROI",
            "crop_label": "instance crop + GT wrist matching ROI",
            "checkpoint": args.full_checkpoint,
            "crop_source": "instance",
            "roi_source": "gt_wrist",
            "roi_label": "GT wrist ROI",
            "mask_mode": "part_competition",
            "rows": full_rows,
        },
        {
            "name": "resnet_wristonly",
            "display_name": "Wrist-only ResNet iter22000",
            "crop_label": "wrist crop",
            "checkpoint": args.wrist_checkpoint,
            "crop_source": "wrist",
            "roi_source": "predicted",
            "roi_label": "pred binary ROI",
            "mask_mode": "binary_head",
            "rows": wrist_rows,
        },
    ]
    results = {frame: {} for frame in frame_ids}
    try:
        for config in configs:
            spec = surf_eval.parse_model_specs([f"{config['name']}={config['checkpoint']}"])[0]
            model, checkpoint_iter, _ = surf_eval.load_model(spec, device)
            surfaces = load_part_surfaces(
                args.surface_root,
                keys_per_part=int(args.surface_keys_per_part),
                seed=int(args.surface_seed),
            )
            encode_surface_keys(model, surfaces, device, mask_keys_per_part=int(args.mask_keys_per_part))
            for frame_id in frame_ids:
                if frame_id not in frame_to_index:
                    raise KeyError(f"Frame {frame_id} is absent from LND TEST")
                _, target = dataset[frame_to_index[frame_id]]
                result = diagnose_model(
                    model,
                    surfaces,
                    renderer,
                    target,
                    config["rows"][frame_id],
                    config,
                    args,
                    device,
                )
                result["checkpoint_iter"] = int(checkpoint_iter)
                results[frame_id][config["name"]] = result
                print(
                    f"{config['name']} frame={frame_id} t={result['csv_trans_err_mm']:.2f}mm "
                    f"r={result['csv_rot_err_deg']:.2f}deg area={result['matching_area_ds']} "
                    f"shown={result['shown_count']}",
                    flush=True,
                )
            del model, surfaces
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        renderer.release()

    paths = []
    rows = []
    for frame_id in frame_ids:
        full = results[frame_id]["resnet_full"]
        wrist = results[frame_id]["resnet_wristonly"]
        image = compose_frame(full, wrist, args)
        path = output_dir / f"frame{frame_id:03d}_full_vs_wrist.jpg"
        image.save(path, quality=95)
        paths.append(path)
        for item in (full, wrist):
            rows.append(
                {
                    "frame_id": frame_id,
                    "model": item["name"],
                    "checkpoint_iter": item["checkpoint_iter"],
                    "crop": item["crop_label"],
                    "trans_err_mm": item["csv_trans_err_mm"],
                    "rot_err_deg": item["csv_rot_err_deg"],
                    "matching_area_ds": item["matching_area_ds"],
                    "csv_correspondences": item["csv_correspondences"],
                    "csv_inliers": item["csv_inliers"],
                    "csv_inlier_fraction": item["csv_inlier_fraction"],
                    "rerun_correspondences": item["rerun_correspondences"],
                    "rerun_inliers": item["rerun_inliers"],
                    "gt_inlier_residual_median_px": item["gt_inlier_residual_median_px"],
                    "gt_inlier_bias_x_px": item["gt_inlier_bias_x_px"],
                    "gt_inlier_bias_y_px": item["gt_inlier_bias_y_px"],
                    "gt_inlier_pair_scale": item["gt_inlier_pair_scale"],
                    "pred_gt_z_ratio": item["pred_gt_z_ratio"],
                    "dt_x_mm": item["dt_x_mm"],
                    "dt_y_mm": item["dt_y_mm"],
                    "dt_z_mm": item["dt_z_mm"],
                    "visualization": str(path),
                }
            )
    write_rows(output_dir / "selected_case_metrics.csv", rows)
    contact_sheet(paths, output_dir / "contact_sheet.jpg")
    full_stats = evaluation_stats(full_rows)
    wrist_stats = evaluation_stats(wrist_rows)
    report = [
        "# Full versus wrist-only SurfEmb correspondence visualization",
        "",
        "- Columns: Full ResNet instance crop with GT wrist matching ROI versus Wrist-only ResNet wrist crop with predicted binary matching ROI.",
        "- Embedding colors: exact original SurfEmb get_emb_vis channel grouping; no PCA.",
        "- Green/red pairs: actual top-K RANSAC inliers/outliers. Filled circle is the query pixel; ring is the selected 3D key projected with GT pose.",
        "- Pose overlay: green SAM GT wrist, cyan GT-pose mesh, orange fitted mesh, yellow GT-to-predicted origin arrow.",
        "- Frame 342 is a control where both rotation and translation improve; the others emphasize rotation gain with flat or worse translation.",
        "",
        "## All 372 evaluation frames",
        "",
        f"- Full: matching area mean/median={full_stats['matching_area_mean']:.1f}/{full_stats['matching_area_median']:.1f} pixels at 74x74; correspondence mean/median={full_stats['correspondences_mean']:.1f}/{full_stats['correspondences_median']:.1f}.",
        f"- Wrist-only: matching area mean/median={wrist_stats['matching_area_mean']:.1f}/{wrist_stats['matching_area_median']:.1f} pixels at 74x74; correspondence mean/median={wrist_stats['correspondences_mean']:.1f}/{wrist_stats['correspondences_median']:.1f}.",
        f"- Wrist-only provides {wrist_stats['matching_area_mean'] / full_stats['matching_area_mean']:.2f}x more wrist pixels and {wrist_stats['correspondences_mean'] / full_stats['correspondences_mean']:.2f}x more selected correspondences on average.",
        f"- Fit-space inlier fraction is Full={full_stats['inlier_fraction_mean']:.3f}, Wrist-only={wrist_stats['inlier_fraction_mean']:.3f}; low reprojection to a fitted pose does not imply low error to GT pose.",
        "",
        "## Selected cases",
        "",
    ]
    for frame_id in frame_ids:
        full = results[frame_id]["resnet_full"]
        wrist = results[frame_id]["resnet_wristonly"]
        report.append(
            f"- frame {frame_id}: rot {full['csv_rot_err_deg']:.2f}->{wrist['csv_rot_err_deg']:.2f} deg, "
            f"trans {full['csv_trans_err_mm']:.2f}->{wrist['csv_trans_err_mm']:.2f} mm, "
            f"matching area {full['matching_area_ds']}->{wrist['matching_area_ds']} pixels at 74x74."
        )
    (output_dir / "README.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"contact_sheet={output_dir / 'contact_sheet.jpg'}", flush=True)


if __name__ == "__main__":
    main(build_parser().parse_args())
