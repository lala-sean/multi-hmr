#!/usr/bin/env python3
"""Visualize the exact dense correspondences rejected by wrist-only PnP/RANSAC."""

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

import debug_surfemb_lnd_rotation_colormap as debug_vis
import eval_surfemb_articulated_rarp as surf_eval
import eval_surfemb_wrist_lnd as lnd_eval
from instrument_geometry import fk_matrices_np
from surfemb_articulated_pose import (
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
    / "surfemb_resnet_wristonly_lnd_best_last_predmask_20260804"
    / "surfemb_per_frame.csv"
)
DEFAULT_CHECKPOINT = (
    ROOT
    / "logs"
    / "surfemb_resnet_wristonly_lnd_best_last_predmask_20260804"
    / "checkpoints"
    / "last_iter17000.pt"
)
DEFAULT_OUTPUT = ROOT / "logs" / "surfemb_resnet_wristonly_lnd_failure_correspondences_iter17000"


def failed_rows(path, model_name, requested_frame_ids):
    with Path(path).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    requested = None if not requested_frame_ids else set(int(value) for value in requested_frame_ids)
    selected = [
        row
        for row in rows
        if row.get("model") == model_name
        and row.get("status") != "ok"
        and int(row["frame_id"]) != 210
        and (requested is None or int(row["frame_id"]) in requested)
    ]
    selected.sort(key=lambda row: int(row["frame_id"]))
    if not selected:
        raise RuntimeError(f"No failed rows found for model={model_name!r} in {path}")
    return selected


def resize_mask(mask, size):
    return cv2.resize(
        np.asarray(mask, dtype=np.uint8),
        (int(size), int(size)),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)


def draw_mask_contours(rgb, gt_mask, pred_mask):
    image = np.asarray(rgb, dtype=np.uint8).copy()
    for mask, color in ((gt_mask, (40, 220, 90)), (pred_mask, (255, 80, 190))):
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, color, 1, cv2.LINE_AA)
    return image


def choose_connections(query_uv, valid_indices, inlier_flags, max_inliers, max_outliers):
    selected = []
    for state, count in ((True, max_inliers), (False, max_outliers)):
        local = np.flatnonzero(inlier_flags == state)
        if not len(local):
            continue
        subset = debug_vis.spatial_subset(query_uv[local], int(count))
        selected.extend(local[subset].tolist())
    if not selected:
        return np.empty((0,), dtype=np.int64)
    return valid_indices[np.asarray(selected, dtype=np.int64)]


def compose_case(
    crop_rgb,
    query_map,
    key_map,
    gt_mask,
    pred_mask,
    query_uv,
    mesh_uv,
    inlier_flags,
    title,
    diagnostics,
    args,
):
    roi = debug_vis.square_roi(
        gt_mask | pred_mask,
        (query_uv, mesh_uv),
        padding=int(args.roi_padding),
    )
    panels = [
        debug_vis.label_panel(
            debug_vis.crop_panel(
                draw_mask_contours(crop_rgb, gt_mask, pred_mask),
                roi,
                args.panel_size,
                cv2.INTER_CUBIC,
            ),
            "Wrist-only input crop",
            "green=GT wrist, magenta=predicted wrist mask",
        ),
        debug_vis.label_panel(
            debug_vis.crop_panel(query_map, roi, args.panel_size, cv2.INTER_NEAREST),
            "Predicted query embedding",
            "original SurfEmb get_emb_vis channel grouping",
        ),
        debug_vis.label_panel(
            debug_vis.crop_panel(key_map, roi, args.panel_size, cv2.INTER_NEAREST),
            "GT-posed 3D wrist embedding",
            "mesh triangle z-buffer; effective wrist includes x<2.13 mm gripper base",
        ),
    ]
    gap = int(args.column_gap)
    separator = np.full((panels[0].shape[0], gap, 3), 255, dtype=np.uint8)
    canvas = np.concatenate((panels[0], separator, panels[1], separator, panels[2]), axis=1)
    query_panel = debug_vis.transform_uv(query_uv, roi, args.panel_size)
    mesh_panel = debug_vis.transform_uv(mesh_uv, roi, args.panel_size)
    x_query = int(args.panel_size) + gap
    x_mesh = 2 * (int(args.panel_size) + gap)
    header = 76
    colors = debug_vis.line_colors(len(query_panel))

    overlay = canvas.copy()
    for q, m, color in zip(query_panel, mesh_panel, colors):
        p0 = (x_query + int(round(q[0])), header + int(round(q[1])))
        p1 = (x_mesh + int(round(m[0])), header + int(round(m[1])))
        cv2.line(overlay, p0, p1, color, 1, cv2.LINE_AA)
    canvas = cv2.addWeighted(canvas, 0.34, overlay, 0.66, 0.0)
    for q, m, is_inlier, color in zip(query_panel, mesh_panel, inlier_flags, colors):
        ring = (20, 210, 70) if bool(is_inlier) else (245, 55, 45)
        p_rgb = (int(round(q[0])), header + int(round(q[1])))
        p_query = (x_query + int(round(q[0])), header + int(round(q[1])))
        p_mesh = (x_mesh + int(round(m[0])), header + int(round(m[1])))
        for point in (p_rgb, p_query, p_mesh):
            cv2.circle(canvas, point, 3, ring, -1, cv2.LINE_AA)
            cv2.circle(canvas, point, 1, color, -1, cv2.LINE_AA)

    banner_height = 64
    output = Image.new("RGB", (canvas.shape[1], canvas.shape[0] + banner_height), (238, 241, 244))
    output.paste(Image.fromarray(canvas), (0, banner_height))
    draw = ImageDraw.Draw(output)
    draw.text((12, 8), title, fill=(8, 8, 8), font=debug_vis.font(21))
    subtitle = (
        f"selected={diagnostics['topk_correspondences']}  "
        f"RANSAC inliers={diagnostics['topk_inliers']} "
        f"({100.0 * diagnostics['topk_inlier_fraction']:.1f}%, reject <60%)  "
        f"shown: green ring=inlier, red ring=outlier"
    )
    draw.text((12, 36), subtitle, fill=(45, 45, 45), font=debug_vis.font(14))
    return np.asarray(output), roi


@torch.inference_mode()
def visualize_frame(model, surfaces, renderer, dataset, row, args, device):
    dataset_index = int(row["dataset_idx"])
    _, target = dataset[dataset_index]
    image, K_crop, crop_rgb, M_crop = lnd_eval.make_eval_crop(target, args)
    part_mask = target["part_mask_orig"].detach().cpu().numpy()
    gt_wrist = cv2.warpAffine(
        (part_mask == 2).astype(np.uint8),
        M_crop,
        (int(args.crop_size), int(args.crop_size)),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)

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
    probability_input = lnd_eval.probability_input_from_binary_logits(
        output["inst_mask_logits"][0],
        int(args.down_sample_scale),
    )
    matching_roi = probability_input["prob"] >= float(args.topk_min_object_probability)
    context = prepare_part_score_context(query_flat, probability_input, surfaces["wrist"], image_hw)
    _, diagnostics = estimate_part_pose_topk_ransac(
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
        min_inlier_fraction=float(args.topk_min_inlier_fraction),
        return_correspondences=True,
    )
    required = (
        "topk_selected_pixel_indices",
        "topk_selected_key_indices",
        "topk_inlier_selected_indices",
    )
    if any(name not in diagnostics for name in required):
        raise RuntimeError(f"No RANSAC correspondences for frame {target['frame_id']}: {diagnostics}")

    query_hwc = output["surfemb_queries"][0].float().permute(1, 2, 0)
    pred_wrist = torch.sigmoid(output["inst_mask_logits"][0].float()) >= 0.5
    query_map = debug_vis.surfemb_embedding_vis(query_hwc, mask=pred_wrist)
    pred_wrist_np = pred_wrist.cpu().numpy()
    gt_pose = surf_eval.target_pose(target)
    key_map, key_visible = debug_vis.render_key_colormap(
        renderer,
        model,
        surfaces["wrist"],
        gt_pose,
        K_crop,
        device,
    )

    selected_pixels = diagnostics["topk_selected_pixel_indices"]
    selected_keys = diagnostics["topk_selected_key_indices"]
    inlier_indices = diagnostics["topk_inlier_selected_indices"]
    h_ds, w_ds = image_hw
    query_uv_all = np.stack(
        (
            (selected_pixels % w_ds + 0.5) * int(args.down_sample_scale) - 0.5,
            (selected_pixels // w_ds + 0.5) * int(args.down_sample_scale) - 0.5,
        ),
        axis=1,
    )
    wrist_transform = fk_matrices_np(
        gt_pose["rot"],
        gt_pose["trans"],
        gt_pose.get("alpha", 0.0),
        gt_pose.get("theta_l", 0.0),
        gt_pose.get("theta_r", 0.0),
    )["wrist"]
    mesh_uv_all, depth = debug_vis.project_points(
        surfaces["wrist"].points_m[selected_keys],
        wrist_transform,
        K_crop,
    )
    valid = (
        np.isfinite(query_uv_all).all(axis=1)
        & np.isfinite(mesh_uv_all).all(axis=1)
        & (depth > 0.0)
        & (mesh_uv_all[:, 0] >= 0.0)
        & (mesh_uv_all[:, 0] < int(args.crop_size))
        & (mesh_uv_all[:, 1] >= 0.0)
        & (mesh_uv_all[:, 1] < int(args.crop_size))
    )
    valid_indices = np.flatnonzero(valid)
    all_inlier_flags = np.zeros((len(selected_pixels),), dtype=bool)
    all_inlier_flags[inlier_indices] = True
    shown_indices = choose_connections(
        query_uv_all[valid],
        valid_indices,
        all_inlier_flags[valid],
        args.max_inlier_connections,
        args.max_outlier_connections,
    )
    if not len(shown_indices):
        raise RuntimeError(f"No projectable correspondences for frame {target['frame_id']}")

    query_uv = query_uv_all[shown_indices]
    mesh_uv = mesh_uv_all[shown_indices]
    shown_inliers = all_inlier_flags[shown_indices]
    frame_id = int(target["frame_id"])
    title = f"LND TEST frame {frame_id} | rejected wrist-only top-K correspondence fit"
    panel, roi = compose_case(
        crop_rgb,
        query_map,
        key_map,
        gt_wrist,
        pred_wrist_np,
        query_uv,
        mesh_uv,
        shown_inliers,
        title,
        diagnostics,
        args,
    )
    return panel, {
        "dataset_idx": dataset_index,
        "frame_id": frame_id,
        "checkpoint_iter": int(args.checkpoint_iter),
        "topk_candidates": int(diagnostics["topk_candidates"]),
        "topk_correspondences": int(diagnostics["topk_correspondences"]),
        "topk_inliers": int(diagnostics["topk_inliers"]),
        "topk_inlier_fraction": float(diagnostics["topk_inlier_fraction"]),
        "matching_wrist_area_ds": int(matching_roi.sum().item()),
        "gt_wrist_pixels": int(gt_wrist.sum()),
        "pred_wrist_pixels": int(pred_wrist_np.sum()),
        "gt_key_visible_pixels": int(key_visible.sum()),
        "shown_inlier_connections": int(shown_inliers.sum()),
        "shown_outlier_connections": int((~shown_inliers).sum()),
        "roi_xyxy": json.dumps([int(value) for value in roi]),
    }


def make_contact_pages(paths, output_dir, rows_per_page=3, max_width=2100):
    pages = []
    for start in range(0, len(paths), int(rows_per_page)):
        page_paths = paths[start : start + int(rows_per_page)]
        images = []
        for path in page_paths:
            image = Image.open(path).convert("RGB")
            if image.width > int(max_width):
                height = int(round(image.height * int(max_width) / image.width))
                image = image.resize((int(max_width), height), Image.Resampling.LANCZOS)
            images.append(image)
        gap = 12
        page = Image.new(
            "RGB",
            (max(image.width for image in images), sum(image.height for image in images) + gap * (len(images) - 1)),
            (255, 255, 255),
        )
        y = 0
        for image in images:
            page.paste(image, (0, y))
            y += image.height + gap
        page_path = output_dir / f"contact_sheet_page_{start // int(rows_per_page):02d}.jpg"
        page.save(page_path, quality=92)
        pages.append(page_path)
    return pages


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_csv", default=str(DEFAULT_EVAL_CSV))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--model_name", default="resnet_wristonly_last")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--frame_ids", nargs="*", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--egl_device", type=int, default=0)
    parser.add_argument("--lnd_root", default="/mnt/iMVR/daiyun/Dataset/LND")
    parser.add_argument("--surface_root", default=str(surf_eval.DEFAULT_SURFACE_ROOT))
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--surfemb_crop_scale", type=float, default=1.2)
    parser.add_argument("--crop_mask_source", choices=("instance", "wrist"), default="wrist")
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
    parser.add_argument("--topk_min_inlier_fraction", type=float, default=0.6)
    parser.add_argument("--max_inlier_connections", type=int, default=16)
    parser.add_argument("--max_outlier_connections", type=int, default=16)
    parser.add_argument("--panel_size", type=int, default=560)
    parser.add_argument("--column_gap", type=int, default=28)
    parser.add_argument("--roi_padding", type=int, default=10)
    parser.add_argument("--rows_per_page", type=int, default=3)
    parser.add_argument("--amp", type=int, choices=(0, 1), default=1)
    return parser


def main(args):
    cv2.setNumThreads(0)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = failed_rows(args.eval_csv, args.model_name, args.frame_ids)
    dataset = lnd_eval.build_dataset(args)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    spec = surf_eval.parse_model_specs([f"{args.model_name}={args.checkpoint}"])[0]
    model, checkpoint_iter, _ = surf_eval.load_model(spec, device)
    args.checkpoint_iter = checkpoint_iter
    surfaces = load_part_surfaces(
        args.surface_root,
        keys_per_part=int(args.surface_keys_per_part),
        seed=int(args.surface_seed),
    )
    encode_surface_keys(model, surfaces, device, mask_keys_per_part=int(args.mask_keys_per_part))
    renderer = debug_vis.EffectiveWristCoordRenderer(
        args.crop_size,
        args.surface_root,
        device_idx=args.egl_device,
    )

    results = []
    paths = []
    try:
        for row in rows:
            panel, result = visualize_frame(model, surfaces, renderer, dataset, row, args, device)
            path = output_dir / f"frame{result['frame_id']:03d}_wrist_correspondences.jpg"
            Image.fromarray(panel).save(path, quality=95)
            result["visualization"] = str(path)
            results.append(result)
            paths.append(path)
            print(
                f"saved frame={result['frame_id']} corr={result['topk_correspondences']} "
                f"inliers={result['topk_inliers']} fraction={result['topk_inlier_fraction']:.3f}",
                flush=True,
            )
    finally:
        renderer.release()

    debug_vis.write_csv(output_dir / "failure_correspondence_summary.csv", results)
    pages = make_contact_pages(paths, output_dir, args.rows_per_page)
    report = [
        "# Wrist-only LND failed-frame correspondence visualizations",
        "",
        f"- checkpoint: {Path(args.checkpoint).resolve()} (iter {checkpoint_iter})",
        f"- source evaluation: {Path(args.eval_csv).resolve()}",
        f"- failed frames: {[int(row['frame_id']) for row in rows]}",
        "- crop and matching protocol exactly follow wrist-only evaluation: wrist crop, binary-head predicted wrist ROI, top-K unique 3D keys, EPNP RANSAC.",
        "- right panel is rendered under GT wrist pose. Each line therefore shows where the selected 3D key should project for that query pixel.",
        "- green endpoint ring: RANSAC inlier; red endpoint ring: RANSAC outlier. Line colors only identify individual pairs.",
        f"- contact sheets: {[path.name for path in pages]}",
    ]
    (output_dir / "README.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"output={output_dir}", flush=True)


if __name__ == "__main__":
    main(build_parser().parse_args())
