#!/usr/bin/env python3
"""Show detected 2D sources, their GT key locations, and full OpenGL meshes."""

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

import debug_surfemb_lnd_rotation_colormap as vis_base
import eval_surfemb_articulated_rarp as surf_eval
import eval_surfemb_wrist_lnd as lnd_eval
import visualize_surfemb_full_vs_wrist_lnd as comparison_vis
from instrument_opengl_renderer import InstrumentOpenGLDepthRenderer
from surfemb_articulated_pose import encode_surface_keys, load_part_surfaces


ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = (
    ROOT
    / "logs"
    / "surfemb_training_progress_lnd_20260805"
    / "checkpoints"
    / "wristonly_last_iter38000.pt"
)
DEFAULT_EVAL_CSV = (
    ROOT
    / "logs"
    / "surfemb_training_progress_lnd_20260805"
    / "wrist_last38k_topk_ransac_50k_acceptall"
    / "surfemb_per_frame.csv"
)
DEFAULT_SURFACE_ROOT = ROOT / "assets" / "instrument_surface_samples_surfemb_x2.13mm_dense50kkeys"
DEFAULT_OUTPUT = (
    ROOT
    / "logs"
    / "surfemb_training_progress_lnd_20260805"
    / "wrist_last38k_source_gt_fullmesh_vis"
)
DEFAULT_FRAMES = (209, 272, 341, 352, 303, 344)

PART_COLORS = {
    1: (249, 115, 22),
    2: (34, 197, 94),
    3: (59, 130, 246),
}


def affine_points(points, matrix):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    homogeneous = np.concatenate((points, np.ones((len(points), 1), dtype=np.float64)), axis=1)
    return homogeneous @ np.asarray(matrix, dtype=np.float64).reshape(2, 3).T


def overlay_part_mask(rgb, part_mask, alpha=0.42):
    output = np.asarray(rgb, dtype=np.uint8).copy()
    blended = output.copy()
    for part_id, color in PART_COLORS.items():
        support = np.asarray(part_mask) == int(part_id)
        blended[support] = np.asarray(color, dtype=np.uint8)
    support = np.asarray(part_mask) > 0
    output[support] = np.clip(
        output[support].astype(np.float32) * (1.0 - float(alpha))
        + blended[support].astype(np.float32) * float(alpha),
        0,
        255,
    ).astype(np.uint8)
    for part_id, color in PART_COLORS.items():
        contours, _ = cv2.findContours(
            (np.asarray(part_mask) == int(part_id)).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(output, contours, -1, color, 2, cv2.LINE_AA)
    return output


def draw_source_target_links(image, source_uv, target_uv, inlier_mask, pair_colors, target_kind):
    output = np.asarray(image, dtype=np.uint8).copy()
    overlay = output.copy()
    for source, target, color in zip(source_uv, target_uv, pair_colors):
        if not np.isfinite(source).all() or not np.isfinite(target).all():
            continue
        cv2.line(
            overlay,
            tuple(np.rint(source).astype(int)),
            tuple(np.rint(target).astype(int)),
            color,
            1,
            cv2.LINE_AA,
        )
    output = cv2.addWeighted(output, 0.42, overlay, 0.58, 0.0)
    for source, target, is_inlier, color in zip(source_uv, target_uv, inlier_mask, pair_colors):
        if not np.isfinite(source).all() or not np.isfinite(target).all():
            continue
        source_point = tuple(np.rint(source).astype(int))
        target_point = tuple(np.rint(target).astype(int))
        status_color = comparison_vis.GREEN if bool(is_inlier) else comparison_vis.RED
        cv2.circle(output, source_point, 4, status_color, 1, cv2.LINE_AA)
        cv2.circle(output, source_point, 2, color, -1, cv2.LINE_AA)
        if target_kind == "gt":
            cv2.circle(output, target_point, 5, comparison_vis.CYAN, 2, cv2.LINE_AA)
            cv2.circle(output, target_point, 1, color, -1, cv2.LINE_AA)
        else:
            cv2.drawMarker(
                output,
                target_point,
                comparison_vis.ORANGE,
                markerType=cv2.MARKER_TILTED_CROSS,
                markerSize=8,
                thickness=2,
                line_type=cv2.LINE_AA,
            )
    return output


def letterbox(image, size):
    image = np.asarray(image, dtype=np.uint8)
    h, w = image.shape[:2]
    scale = min(float(size) / float(w), float(size) / float(h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    x0 = (int(size) - new_w) // 2
    y0 = (int(size) - new_h) // 2
    canvas = np.full((int(size), int(size), 3), 20, dtype=np.uint8)
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas, scale, np.asarray([x0, y0], dtype=np.float64)


def label_and_join(images, titles, subtitles, gap):
    panels = [
        vis_base.label_panel(image, title, subtitle)
        for image, title, subtitle in zip(images, titles, subtitles)
    ]
    separator = np.full((panels[0].shape[0], int(gap), 3), 255, dtype=np.uint8)
    return np.concatenate((panels[0], separator, panels[1], separator, panels[2]), axis=1)


def compose_crop_row(result, args, pair_colors):
    panel_size = int(args.panel_size)
    gap = int(args.column_gap)
    source_uv = np.asarray(result["shown_query_uv"], dtype=np.float64)
    gt_uv = np.asarray(result["shown_gt_uv"], dtype=np.float64)
    pred_uv = np.asarray(result["shown_pred_uv"], dtype=np.float64)
    status = np.asarray(result["shown_inlier_mask"], dtype=bool)
    roi = vis_base.square_roi(
        np.asarray(result["gt_wrist"], dtype=bool),
        (source_uv, gt_uv, pred_uv),
        padding=int(args.roi_padding),
    )
    source_panel = vis_base.crop_panel(result["crop_rgb"], roi, panel_size, cv2.INTER_CUBIC)
    source_panel = draw_source_target_links(
        source_panel,
        vis_base.transform_uv(source_uv, roi, panel_size),
        vis_base.transform_uv(gt_uv, roi, panel_size),
        status,
        pair_colors,
        "gt",
    )
    query_panel = vis_base.crop_panel(result["query_map"], roi, panel_size, cv2.INTER_NEAREST)
    key_panel = vis_base.crop_panel(result["gt_key_map"], roi, panel_size, cv2.INTER_NEAREST)
    row = label_and_join(
        (source_panel, query_panel, key_panel),
        (
            "Detected source -> GT correspondence",
            "Encoder query feature colormap",
            "GT-posed 3D key colormap",
        ),
        (
            "filled dot=source; cyan ring=where selected 3D key should be under GT",
            "same detected source pixels; original SurfEmb color grouping",
            "triangle-rendered effective wrist coordinates; full-mesh z-buffer",
        ),
        gap,
    )

    source_panel_uv = vis_base.transform_uv(source_uv, roi, panel_size)
    gt_panel_uv = vis_base.transform_uv(gt_uv, roi, panel_size)
    header = 76
    x_query = panel_size + gap
    x_key = 2 * (panel_size + gap)
    line_layer = row.copy()
    for source, gt_point, color in zip(source_panel_uv, gt_panel_uv, pair_colors):
        p0 = (x_query + int(round(source[0])), header + int(round(source[1])))
        p1 = (x_key + int(round(gt_point[0])), header + int(round(gt_point[1])))
        cv2.line(line_layer, p0, p1, color, 1, cv2.LINE_AA)
    row = cv2.addWeighted(row, 0.34, line_layer, 0.66, 0.0)
    for source, gt_point, is_inlier, color in zip(source_panel_uv, gt_panel_uv, status, pair_colors):
        status_color = comparison_vis.GREEN if bool(is_inlier) else comparison_vis.RED
        p_source = (x_query + int(round(source[0])), header + int(round(source[1])))
        p_gt = (x_key + int(round(gt_point[0])), header + int(round(gt_point[1])))
        cv2.circle(row, p_source, 4, status_color, 1, cv2.LINE_AA)
        cv2.circle(row, p_source, 2, color, -1, cv2.LINE_AA)
        cv2.circle(row, p_gt, 5, comparison_vis.CYAN, 2, cv2.LINE_AA)
        cv2.circle(row, p_gt, 1, color, -1, cv2.LINE_AA)
    return row


def compose_full_frame_row(result, target, full_renderer, args, pair_colors):
    panel_size = int(args.panel_size)
    gap = int(args.column_gap)
    rgb = target["orig_rgb"].detach().cpu().numpy().astype(np.uint8)
    K_orig = target["K_orig"].detach().cpu().numpy().astype(np.float64)
    gt_part_mask = full_renderer.render_pose_mask(result["gt_pose"], K_orig, rgb.shape[:2])
    pred_part_mask = full_renderer.render_pose_mask(result["pred_pose"], K_orig, rgb.shape[:2])
    gt_mesh = overlay_part_mask(rgb, gt_part_mask)
    pred_mesh = overlay_part_mask(rgb, pred_part_mask)

    inverse_crop = cv2.invertAffineTransform(np.asarray(result["M_crop"], dtype=np.float64))
    source_uv = affine_points(result["shown_query_uv"], inverse_crop)
    gt_uv = affine_points(result["shown_gt_uv"], inverse_crop)
    pred_uv = affine_points(result["shown_pred_uv"], inverse_crop)
    status = np.asarray(result["shown_inlier_mask"], dtype=bool)

    source_gt = draw_source_target_links(
        rgb, source_uv, gt_uv, status, pair_colors, "gt"
    )
    gt_mesh = draw_source_target_links(
        gt_mesh, source_uv, gt_uv, status, pair_colors, "gt"
    )
    pred_mesh = draw_source_target_links(
        pred_mesh, source_uv, pred_uv, status, pair_colors, "pred"
    )
    full_roi = vis_base.square_roi(
        (gt_part_mask > 0) | (pred_part_mask > 0),
        (source_uv, gt_uv, pred_uv),
        padding=int(args.full_frame_roi_padding),
    )
    boxed = [
        vis_base.crop_panel(image, full_roi, panel_size, cv2.INTER_CUBIC)
        for image in (source_gt, gt_mesh, pred_mesh)
    ]
    row = label_and_join(
        boxed,
        (
            "Original-frame ROI: source -> GT key",
            "GT full instrument OpenGL mesh",
            "Predicted full instrument OpenGL mesh",
        ),
        (
            "one shared ROI contains all visible GT/pred OpenGL mesh pixels",
            "blue=shaft, green=wrist, orange=grippers; alpha/theta fixed to zero",
            "same complete four-part mesh under predicted wrist pose",
        ),
        gap,
    )

    # Connect identical selected 3D keys across the full GT and predicted mesh panels.
    gt_panel_uv = vis_base.transform_uv(gt_uv, full_roi, panel_size)
    pred_panel_uv = vis_base.transform_uv(pred_uv, full_roi, panel_size)
    header = 76
    x_gt = panel_size + gap
    x_pred = 2 * (panel_size + gap)
    line_layer = row.copy()
    for gt_point, pred_point, color in zip(gt_panel_uv, pred_panel_uv, pair_colors):
        p0 = (x_gt + int(round(gt_point[0])), header + int(round(gt_point[1])))
        p1 = (x_pred + int(round(pred_point[0])), header + int(round(pred_point[1])))
        cv2.line(line_layer, p0, p1, color, 1, cv2.LINE_AA)
    row = cv2.addWeighted(row, 0.34, line_layer, 0.66, 0.0)
    for gt_point, pred_point, color in zip(gt_panel_uv, pred_panel_uv, pair_colors):
        p_gt = (x_gt + int(round(gt_point[0])), header + int(round(gt_point[1])))
        p_pred = (x_pred + int(round(pred_point[0])), header + int(round(pred_point[1])))
        cv2.circle(row, p_gt, 5, comparison_vis.CYAN, 2, cv2.LINE_AA)
        cv2.circle(row, p_gt, 1, color, -1, cv2.LINE_AA)
        cv2.drawMarker(
            row,
            p_pred,
            comparison_vis.ORANGE,
            markerType=cv2.MARKER_TILTED_CROSS,
            markerSize=8,
            thickness=2,
            line_type=cv2.LINE_AA,
        )
    return row, {
        "gt_full_mesh_pixels": int((gt_part_mask > 0).sum()),
        "pred_full_mesh_pixels": int((pred_part_mask > 0).sum()),
        "gt_shaft_pixels": int((gt_part_mask == 3).sum()),
        "gt_wrist_pixels": int((gt_part_mask == 2).sum()),
        "gt_gripper_pixels": int((gt_part_mask == 1).sum()),
        "full_frame_roi_xyxy": str(tuple(int(value) for value in full_roi)),
    }


def compose_case(result, target, full_renderer, args):
    pair_colors = vis_base.line_colors(len(result["shown_query_uv"]))
    crop_row = compose_crop_row(result, args, pair_colors)
    full_row, full_stats = compose_full_frame_row(result, target, full_renderer, args, pair_colors)
    banner = 98
    gap = 14
    output = Image.new(
        "RGB",
        (crop_row.shape[1], banner + crop_row.shape[0] + gap + full_row.shape[0]),
        (230, 235, 241),
    )
    output.paste(Image.fromarray(crop_row), (0, banner))
    output.paste(Image.fromarray(full_row), (0, banner + crop_row.shape[0] + gap))
    draw = ImageDraw.Draw(output)
    draw.text(
        (14, 8),
        f"LND TEST frame {result['frame_id']} | ResNet wrist-only iter38000 | original 50k Top-K RANSAC",
        fill=comparison_vis.INK,
        font=vis_base.font(24),
    )
    draw.text(
        (14, 42),
        f"translation={result['csv_trans_err_mm']:.2f} mm   rotation={result['csv_rot_err_deg']:.2f} deg   "
        f"inliers={result['csv_inliers']}/{result['csv_correspondences']} ({100.0 * result['csv_inlier_fraction']:.1f}%)",
        fill=(51, 65, 85),
        font=vis_base.font(16),
    )
    draw.text(
        (14, 69),
        "filled source dot -> cyan GT ring; orange cross=the same selected 3D key under predicted pose; "
        "green/red source ring=RANSAC inlier/outlier",
        fill=(71, 85, 105),
        font=vis_base.font(13),
    )
    return output, full_stats


def make_contact_pages(paths, output_dir, max_width=1700):
    pages = []
    for page_index, path in enumerate(paths):
        image = Image.open(path).convert("RGB")
        if image.width > int(max_width):
            height = int(round(image.height * int(max_width) / image.width))
            image = image.resize((int(max_width), height), Image.Resampling.LANCZOS)
        page_path = output_dir / f"contact_sheet_page_{page_index:02d}.jpg"
        image.save(page_path, quality=94)
        pages.append(page_path)
    return pages


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--eval_csv", default=str(DEFAULT_EVAL_CSV))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--frame_ids", nargs="*", type=int, default=list(DEFAULT_FRAMES))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--egl_device", type=int, default=0)
    parser.add_argument("--lnd_root", default="/mnt/iMVR/daiyun/Dataset/LND")
    parser.add_argument("--surface_root", default=str(DEFAULT_SURFACE_ROOT))
    parser.add_argument("--surface_keys_per_part", type=int, default=50000)
    parser.add_argument("--mask_keys_per_part", type=int, default=512)
    parser.add_argument("--surface_seed", type=int, default=2026)
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--surfemb_crop_scale", type=float, default=1.2)
    parser.add_argument("--crop_mask_source", choices=("instance", "wrist"), default="wrist")
    parser.add_argument("--bbox_padding_frac", type=float, default=0.12)
    parser.add_argument("--canonical_eps", type=float, default=0.08)
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
    parser.add_argument("--num_inliers", type=int, default=12)
    parser.add_argument("--num_outliers", type=int, default=6)
    parser.add_argument("--panel_size", type=int, default=560)
    parser.add_argument("--column_gap", type=int, default=28)
    parser.add_argument("--roi_padding", type=int, default=10)
    parser.add_argument("--full_frame_roi_padding", type=int, default=40)
    parser.add_argument("--amp", type=int, choices=(0, 1), default=1)
    return parser


def main(args):
    cv2.setNumThreads(0)
    torch.set_num_threads(2)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = comparison_vis.read_rows(args.eval_csv)
    frame_ids = [int(value) for value in args.frame_ids]
    dataset = lnd_eval.build_dataset(args)
    frame_to_index = {int(sample[0]): index for index, sample in enumerate(dataset.samples)}
    missing = [frame_id for frame_id in frame_ids if frame_id not in rows or frame_id not in frame_to_index]
    if missing:
        raise KeyError(f"Frames unavailable for visualization: {missing}")

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    spec = surf_eval.parse_model_specs([f"resnet_wristonly_iter38000={args.checkpoint}"])[0]
    model, checkpoint_iter, _ = surf_eval.load_model(spec, device)
    surfaces = load_part_surfaces(
        args.surface_root,
        keys_per_part=int(args.surface_keys_per_part),
        seed=int(args.surface_seed),
    )
    encode_surface_keys(model, surfaces, device, mask_keys_per_part=int(args.mask_keys_per_part))
    key_renderer = vis_base.EffectiveWristCoordRenderer(
        args.crop_size, args.surface_root, device_idx=int(args.egl_device)
    )
    full_renderer = InstrumentOpenGLDepthRenderer(
        args.crop_size, args.crop_size, device_idx=int(args.egl_device)
    )
    config = {
        "name": "resnet_wristonly_iter38000",
        "display_name": "Wrist-only ResNet iter38000",
        "crop_label": "wrist crop",
        "crop_source": "wrist",
        "roi_source": "predicted",
        "roi_label": "predicted binary wrist ROI",
        "mask_mode": "binary_head",
    }

    paths = []
    metrics = []
    try:
        for frame_id in frame_ids:
            _, target = dataset[frame_to_index[frame_id]]
            result = comparison_vis.diagnose_model(
                model, surfaces, key_renderer, target, rows[frame_id], config, args, device
            )
            image, full_stats = compose_case(result, target, full_renderer, args)
            path = output_dir / f"frame{frame_id:03d}_source_gt_fullmesh.jpg"
            image.save(path, quality=96)
            paths.append(path)
            metrics.append(
                {
                    "frame_id": frame_id,
                    "checkpoint_iter": int(checkpoint_iter),
                    "trans_err_mm": result["csv_trans_err_mm"],
                    "rot_err_deg": result["csv_rot_err_deg"],
                    "shown_sources": int(len(result["shown_query_uv"])),
                    **full_stats,
                    "visualization": str(path),
                }
            )
            print(
                f"frame={frame_id} sources={len(result['shown_query_uv'])} "
                f"full_mesh_pixels={full_stats['gt_full_mesh_pixels']} saved={path}",
                flush=True,
            )
    finally:
        full_renderer.release()
        key_renderer.release()

    comparison_vis.write_rows(output_dir / "selected_failure_metrics.csv", metrics)
    pages = make_contact_pages(paths, output_dir)
    report = [
        "# SurfEmb iter38000 source-to-GT and full-mesh failure visualization",
        "",
        f"- checkpoint: {Path(args.checkpoint).resolve()}",
        f"- evaluation: {Path(args.eval_csv).resolve()}",
        f"- frames: {frame_ids}",
        "- Filled colored dot: detected 2D source pixel.",
        "- Cyan ring: selected 3D key projected with the GT wrist pose.",
        "- Orange cross: the same selected 3D key projected with the predicted wrist pose.",
        "- Full OpenGL mesh panels render shaft, wrist, static gripper bases, and moving grippers with shared triangle z-buffer.",
        "- LND TEST supplies wrist pose only, so alpha/theta_l/theta_r are fixed to zero in both full-mesh renders.",
        f"- pages: {[path.name for path in pages]}",
    ]
    (output_dir / "README.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"output={output_dir}", flush=True)


if __name__ == "__main__":
    main(build_parser().parse_args())
