#!/usr/bin/env python3
"""Visualize iter38k LND failures in feature and pose-projection space."""

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
    / "wrist_last38k_failure_mesh_correspondence_vis"
)
DEFAULT_FRAMES = (209, 272, 341, 352, 303, 344)


def make_mesh_images(result):
    rgb = np.asarray(result["crop_rgb"], dtype=np.uint8)
    gt = comparison_vis.fill_mask(rgb, result["gt_mesh_mask"], comparison_vis.CYAN, 0.20)
    gt = comparison_vis.draw_contour(gt, result["gt_mesh_mask"], comparison_vis.CYAN, 2)
    pred = comparison_vis.fill_mask(rgb, result["pred_mesh_mask"], comparison_vis.ORANGE, 0.20)
    pred = comparison_vis.draw_contour(pred, result["pred_mesh_mask"], comparison_vis.ORANGE, 2)
    both = comparison_vis.fill_mask(rgb, result["gt_mesh_mask"], comparison_vis.CYAN, 0.12)
    both = comparison_vis.fill_mask(both, result["pred_mesh_mask"], comparison_vis.ORANGE, 0.12)
    both = comparison_vis.draw_contour(both, result["gt_mesh_mask"], comparison_vis.CYAN, 2)
    both = comparison_vis.draw_contour(both, result["pred_mesh_mask"], comparison_vis.ORANGE, 2)
    return both, gt, pred


def concatenate_panels(panels, gap):
    separator = np.full((panels[0].shape[0], int(gap), 3), 255, dtype=np.uint8)
    return np.concatenate((panels[0], separator, panels[1], separator, panels[2]), axis=1)


def draw_feature_row(result, roi, args):
    panel_size = int(args.panel_size)
    gap = int(args.column_gap)
    input_image = comparison_vis.fill_mask(
        result["crop_rgb"], result["gt_wrist"], comparison_vis.GREEN, 0.08
    )
    input_image = comparison_vis.draw_contour(
        input_image, result["gt_wrist"], comparison_vis.GREEN, 2
    )
    input_image = comparison_vis.draw_contour(
        input_image, result["matching_mask"], comparison_vis.YELLOW, 1
    )
    images = (
        vis_base.crop_panel(input_image, roi, panel_size, cv2.INTER_CUBIC),
        vis_base.crop_panel(result["query_map"], roi, panel_size, cv2.INTER_NEAREST),
        vis_base.crop_panel(result["gt_key_map"], roi, panel_size, cv2.INTER_NEAREST),
    )
    panels = [
        vis_base.label_panel(images[0], "RGB wrist crop", "green=GT wrist; yellow=predicted matching ROI"),
        vis_base.label_panel(images[1], "Encoder query colormap", "original SurfEmb embedding color grouping"),
        vis_base.label_panel(images[2], "GT-posed 3D key colormap", "effective wrist triangle render with full-mesh z-buffer"),
    ]
    canvas = concatenate_panels(panels, gap)
    query_uv = vis_base.transform_uv(result["shown_query_uv"], roi, panel_size)
    gt_uv = vis_base.transform_uv(result["shown_gt_uv"], roi, panel_size)
    colors = vis_base.line_colors(len(query_uv))
    header = 76
    x_query = panel_size + gap
    x_gt = 2 * (panel_size + gap)
    overlay = canvas.copy()
    for query_point, gt_point, color in zip(query_uv, gt_uv, colors):
        p0 = (x_query + int(round(query_point[0])), header + int(round(query_point[1])))
        p1 = (x_gt + int(round(gt_point[0])), header + int(round(gt_point[1])))
        cv2.line(overlay, p0, p1, color, 1, cv2.LINE_AA)
    canvas = cv2.addWeighted(canvas, 0.30, overlay, 0.70, 0.0)
    for query_point, gt_point, is_inlier, color in zip(
        query_uv, gt_uv, result["shown_inlier_mask"], colors
    ):
        ring = comparison_vis.GREEN if bool(is_inlier) else comparison_vis.RED
        points = (
            (int(round(query_point[0])), header + int(round(query_point[1]))),
            (x_query + int(round(query_point[0])), header + int(round(query_point[1]))),
            (x_gt + int(round(gt_point[0])), header + int(round(gt_point[1]))),
        )
        for point in points:
            cv2.circle(canvas, point, 3, ring, 1, cv2.LINE_AA)
            cv2.circle(canvas, point, 1, color, -1, cv2.LINE_AA)
    return canvas


def draw_mesh_row(result, roi, args):
    panel_size = int(args.panel_size)
    gap = int(args.column_gap)
    both, gt_mesh, pred_mesh = make_mesh_images(result)
    images = [
        vis_base.crop_panel(both, roi, panel_size, cv2.INTER_CUBIC),
        vis_base.crop_panel(gt_mesh, roi, panel_size, cv2.INTER_CUBIC),
        vis_base.crop_panel(pred_mesh, roi, panel_size, cv2.INTER_CUBIC),
    ]
    gt_uv = vis_base.transform_uv(result["shown_gt_uv"], roi, panel_size)
    pred_uv = vis_base.transform_uv(result["shown_pred_uv"], roi, panel_size)
    inlier_mask = np.asarray(result["shown_inlier_mask"], dtype=bool)

    # The first panel overlays displacement of identical canonical surface keys.
    overlay = images[0].copy()
    for gt_point, pred_point, is_inlier in zip(gt_uv, pred_uv, inlier_mask):
        color = comparison_vis.GREEN if bool(is_inlier) else comparison_vis.RED
        cv2.line(
            overlay,
            tuple(np.rint(gt_point).astype(int)),
            tuple(np.rint(pred_point).astype(int)),
            color,
            1,
            cv2.LINE_AA,
        )
        cv2.circle(overlay, tuple(np.rint(gt_point).astype(int)), 2, comparison_vis.CYAN, -1, cv2.LINE_AA)
        cv2.circle(overlay, tuple(np.rint(pred_point).astype(int)), 2, comparison_vis.ORANGE, -1, cv2.LINE_AA)
    images[0] = cv2.addWeighted(images[0], 0.30, overlay, 0.70, 0.0)

    panels = [
        vis_base.label_panel(
            images[0],
            "GT-to-predicted mesh displacement",
            "same canonical key: cyan=GT, orange=pred; green/red=RANSAC status",
        ),
        vis_base.label_panel(images[1], "GT wrist mesh projection", "cyan triangle-rendered effective wrist"),
        vis_base.label_panel(images[2], "Predicted wrist mesh projection", "orange triangle-rendered effective wrist"),
    ]
    canvas = concatenate_panels(panels, gap)
    colors = vis_base.line_colors(len(gt_uv))
    header = 76
    x_gt = panel_size + gap
    x_pred = 2 * (panel_size + gap)
    line_layer = canvas.copy()
    for gt_point, pred_point, color in zip(gt_uv, pred_uv, colors):
        p0 = (x_gt + int(round(gt_point[0])), header + int(round(gt_point[1])))
        p1 = (x_pred + int(round(pred_point[0])), header + int(round(pred_point[1])))
        cv2.line(line_layer, p0, p1, color, 1, cv2.LINE_AA)
    canvas = cv2.addWeighted(canvas, 0.30, line_layer, 0.70, 0.0)
    for gt_point, pred_point, is_inlier, color in zip(gt_uv, pred_uv, inlier_mask, colors):
        ring = comparison_vis.GREEN if bool(is_inlier) else comparison_vis.RED
        p_gt = (x_gt + int(round(gt_point[0])), header + int(round(gt_point[1])))
        p_pred = (x_pred + int(round(pred_point[0])), header + int(round(pred_point[1])))
        for point in (p_gt, p_pred):
            cv2.circle(canvas, point, 3, ring, 1, cv2.LINE_AA)
            cv2.circle(canvas, point, 1, color, -1, cv2.LINE_AA)
    return canvas


def compose_case(result, args):
    mask = (
        np.asarray(result["gt_wrist"], dtype=bool)
        | np.asarray(result["gt_mesh_mask"], dtype=bool)
        | np.asarray(result["pred_mesh_mask"], dtype=bool)
    )
    roi = vis_base.square_roi(
        mask,
        (
            result["shown_query_uv"],
            result["shown_gt_uv"],
            result["shown_pred_uv"],
        ),
        padding=int(args.roi_padding),
    )
    feature_row = draw_feature_row(result, roi, args)
    mesh_row = draw_mesh_row(result, roi, args)
    row_gap = 14
    banner = 92
    output = Image.new(
        "RGB",
        (feature_row.shape[1], banner + feature_row.shape[0] + row_gap + mesh_row.shape[0]),
        (231, 236, 242),
    )
    output.paste(Image.fromarray(feature_row), (0, banner))
    output.paste(Image.fromarray(mesh_row), (0, banner + feature_row.shape[0] + row_gap))
    draw = ImageDraw.Draw(output)
    draw.text(
        (14, 9),
        f"LND TEST frame {result['frame_id']} | wrist-only ResNet iter38000 | original 50k Top-K RANSAC",
        fill=comparison_vis.INK,
        font=vis_base.font(24),
    )
    draw.text(
        (14, 43),
        f"translation={result['csv_trans_err_mm']:.2f} mm   rotation={result['csv_rot_err_deg']:.2f} deg   "
        f"inliers={result['csv_inliers']}/{result['csv_correspondences']} ({100.0 * result['csv_inlier_fraction']:.1f}%)",
        fill=(51, 65, 85),
        font=vis_base.font(16),
    )
    draw.text(
        (14, 68),
        f"pred-GT translation: ({result['dt_x_mm']:+.1f}, {result['dt_y_mm']:+.1f}, {result['dt_z_mm']:+.1f}) mm; "
        "colored lines identify the same correspondence/key across panels",
        fill=(71, 85, 105),
        font=vis_base.font(13),
    )
    return output, roi


def make_contact_pages(paths, output_dir, rows_per_page=2, max_width=1700):
    pages = []
    for start in range(0, len(paths), int(rows_per_page)):
        images = []
        for path in paths[start : start + int(rows_per_page)]:
            image = Image.open(path).convert("RGB")
            if image.width > int(max_width):
                height = int(round(image.height * int(max_width) / image.width))
                image = image.resize((int(max_width), height), Image.Resampling.LANCZOS)
            images.append(image)
        gap = 14
        page = Image.new(
            "RGB",
            (max(image.width for image in images), sum(image.height for image in images) + gap * (len(images) - 1)),
            (255, 255, 255),
        )
        y = 0
        for image in images:
            page.paste(image, (0, y))
            y += image.height + gap
        path = output_dir / f"contact_sheet_page_{start // int(rows_per_page):02d}.jpg"
        page.save(path, quality=93)
        pages.append(path)
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
    parser.add_argument("--num_inliers", type=int, default=16)
    parser.add_argument("--num_outliers", type=int, default=8)
    parser.add_argument("--panel_size", type=int, default=520)
    parser.add_argument("--column_gap", type=int, default=28)
    parser.add_argument("--roi_padding", type=int, default=10)
    parser.add_argument("--rows_per_page", type=int, default=2)
    parser.add_argument("--amp", type=int, choices=(0, 1), default=1)
    return parser


def main(args):
    cv2.setNumThreads(0)
    torch.set_num_threads(2)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = comparison_vis.read_rows(args.eval_csv)
    frame_ids = [int(value) for value in args.frame_ids]
    missing = [frame_id for frame_id in frame_ids if frame_id not in rows]
    if missing:
        raise KeyError(f"Frames absent from evaluation CSV: {missing}")

    dataset = lnd_eval.build_dataset(args)
    frame_to_index = {int(sample[0]): index for index, sample in enumerate(dataset.samples)}
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
    renderer = vis_base.EffectiveWristCoordRenderer(
        args.crop_size,
        args.surface_root,
        device_idx=int(args.egl_device),
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
                model,
                surfaces,
                renderer,
                target,
                rows[frame_id],
                config,
                args,
                device,
            )
            result["checkpoint_iter"] = int(checkpoint_iter)
            image, roi = compose_case(result, args)
            path = output_dir / f"frame{frame_id:03d}_iter38k_failure_correspondence.jpg"
            image.save(path, quality=96)
            paths.append(path)
            metrics.append(
                {
                    "frame_id": frame_id,
                    "checkpoint_iter": int(checkpoint_iter),
                    "trans_err_mm": result["csv_trans_err_mm"],
                    "rot_err_deg": result["csv_rot_err_deg"],
                    "topk_correspondences": result["csv_correspondences"],
                    "topk_inliers": result["csv_inliers"],
                    "topk_inlier_fraction": result["csv_inlier_fraction"],
                    "gt_inlier_residual_median_px": result["gt_inlier_residual_median_px"],
                    "dt_x_mm": result["dt_x_mm"],
                    "dt_y_mm": result["dt_y_mm"],
                    "dt_z_mm": result["dt_z_mm"],
                    "roi_xyxy": str(tuple(int(value) for value in roi)),
                    "visualization": str(path),
                }
            )
            print(
                f"frame={frame_id} t={result['csv_trans_err_mm']:.2f}mm "
                f"r={result['csv_rot_err_deg']:.2f}deg saved={path}",
                flush=True,
            )
    finally:
        renderer.release()

    comparison_vis.write_rows(output_dir / "selected_failure_metrics.csv", metrics)
    pages = make_contact_pages(paths, output_dir, args.rows_per_page)
    report = [
        "# SurfEmb iter38000 LND failure correspondence visualization",
        "",
        f"- checkpoint: {Path(args.checkpoint).resolve()}",
        f"- evaluation: {Path(args.eval_csv).resolve()}",
        f"- frames: {frame_ids}",
        "- Row 1: input crop, encoder query colormap, and GT-posed effective-wrist key colormap.",
        "- Row 2: direct GT-to-predicted displacement, GT mesh, and predicted mesh.",
        "- Cross-panel mesh lines connect the same canonical 3D surface key under GT and predicted poses.",
        "- Green/red rings denote rerun Top-K RANSAC inlier/outlier status; line hue identifies a pair.",
        f"- contact sheets: {[path.name for path in pages]}",
    ]
    (output_dir / "README.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"output={output_dir}", flush=True)


if __name__ == "__main__":
    main(build_parser().parse_args())
