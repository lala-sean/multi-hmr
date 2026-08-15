#!/usr/bin/env python3
"""Visualize source-to-GT matches with complete effective-wrist OpenGL meshes."""

import argparse
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
import visualize_surfemb_iter38k_source_gt_fullmesh as link_vis
from instrument_geometry import fk_matrices_np
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
    / "wrist_last38k_source_gt_wristmesh_vis"
)
DEFAULT_FRAMES = (209, 272, 341, 352, 303, 344)


def render_complete_wrist(renderer, pose, K, image_shape, base_color, edge_color):
    transforms = fk_matrices_np(
        pose["rot"],
        pose["trans"],
        pose.get("alpha", 0.0),
        pose.get("theta_l", 0.0),
        pose.get("theta_r", 0.0),
    )
    mesh_rgb, depth, valid = renderer.render_candidate_mesh_visualization(
        transforms,
        K,
        image_shape,
        include_parts=("wrist", "static_wrist"),
        base_color=np.asarray(base_color, dtype=np.float32) / 255.0,
        edge_color=np.asarray(edge_color, dtype=np.float32) / 255.0,
    )
    mask = valid & (depth > 0.0)
    return mesh_rgb, mask, depth


def composite_mesh(rgb, mesh_rgb, mask, contour_color, alpha=0.82):
    output = np.asarray(rgb, dtype=np.uint8).copy()
    mask = np.asarray(mask, dtype=bool)
    if mask.any():
        output[mask] = np.clip(
            output[mask].astype(np.float32) * (1.0 - float(alpha))
            + np.asarray(mesh_rgb, dtype=np.float32)[mask] * float(alpha),
            0,
            255,
        ).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(output, contours, -1, contour_color, 2, cv2.LINE_AA)
    return output


def overlay_gt_pred_meshes(rgb, gt_rgb, gt_mask, pred_rgb, pred_mask):
    output = composite_mesh(rgb, gt_rgb, gt_mask, comparison_vis.CYAN, alpha=0.56)
    output = composite_mesh(output, pred_rgb, pred_mask, comparison_vis.ORANGE, alpha=0.56)
    output = comparison_vis.draw_contour(output, gt_mask, comparison_vis.CYAN, 2)
    output = comparison_vis.draw_contour(output, pred_mask, comparison_vis.ORANGE, 2)
    return output


def draw_gt_pred_links(image, gt_uv, pred_uv, pair_colors):
    output = np.asarray(image, dtype=np.uint8).copy()
    line_layer = output.copy()
    for gt_point, pred_point, color in zip(gt_uv, pred_uv, pair_colors):
        if not np.isfinite(gt_point).all() or not np.isfinite(pred_point).all():
            continue
        cv2.line(
            line_layer,
            tuple(np.rint(gt_point).astype(int)),
            tuple(np.rint(pred_point).astype(int)),
            color,
            1,
            cv2.LINE_AA,
        )
    output = cv2.addWeighted(output, 0.38, line_layer, 0.62, 0.0)
    for gt_point, pred_point, color in zip(gt_uv, pred_uv, pair_colors):
        if not np.isfinite(gt_point).all() or not np.isfinite(pred_point).all():
            continue
        cv2.circle(
            output,
            tuple(np.rint(gt_point).astype(int)),
            5,
            comparison_vis.CYAN,
            2,
            cv2.LINE_AA,
        )
        cv2.circle(output, tuple(np.rint(gt_point).astype(int)), 1, color, -1, cv2.LINE_AA)
        cv2.drawMarker(
            output,
            tuple(np.rint(pred_point).astype(int)),
            comparison_vis.ORANGE,
            markerType=cv2.MARKER_TILTED_CROSS,
            markerSize=8,
            thickness=2,
            line_type=cv2.LINE_AA,
        )
    return output


def compose_wrist_mesh_row(result, renderer, args, pair_colors):
    rgb = np.asarray(result["crop_rgb"], dtype=np.uint8)
    K = np.asarray(result["K_crop"], dtype=np.float64)
    gt_rgb, gt_mask, gt_depth = render_complete_wrist(
        renderer,
        result["gt_pose"],
        K,
        rgb.shape[:2],
        comparison_vis.CYAN,
        (5, 42, 51),
    )
    pred_rgb, pred_mask, pred_depth = render_complete_wrist(
        renderer,
        result["pred_pose"],
        K,
        rgb.shape[:2],
        comparison_vis.ORANGE,
        (67, 24, 4),
    )
    gt_mesh = composite_mesh(rgb, gt_rgb, gt_mask, comparison_vis.CYAN)
    pred_mesh = composite_mesh(rgb, pred_rgb, pred_mask, comparison_vis.ORANGE)
    comparison = overlay_gt_pred_meshes(rgb, gt_rgb, gt_mask, pred_rgb, pred_mask)

    source_uv = np.asarray(result["shown_query_uv"], dtype=np.float64)
    gt_uv = np.asarray(result["shown_gt_uv"], dtype=np.float64)
    pred_uv = np.asarray(result["shown_pred_uv"], dtype=np.float64)
    status = np.asarray(result["shown_inlier_mask"], dtype=bool)
    gt_mesh = link_vis.draw_source_target_links(
        gt_mesh, source_uv, gt_uv, status, pair_colors, "gt"
    )
    pred_mesh = link_vis.draw_source_target_links(
        pred_mesh, source_uv, pred_uv, status, pair_colors, "pred"
    )
    comparison = draw_gt_pred_links(comparison, gt_uv, pred_uv, pair_colors)

    roi = vis_base.square_roi(
        gt_mask | pred_mask,
        (source_uv, gt_uv, pred_uv),
        padding=int(args.roi_padding),
    )
    panel_size = int(args.panel_size)
    images = [
        vis_base.crop_panel(image, roi, panel_size, cv2.INTER_CUBIC)
        for image in (gt_mesh, pred_mesh, comparison)
    ]
    row = link_vis.label_and_join(
        images,
        (
            "Complete GT wrist OpenGL mesh",
            "Complete predicted wrist OpenGL mesh",
            "GT vs predicted wrist mesh",
        ),
        (
            "shaded triangle faces + wire edges; wrist OBJ + x<2.13 mm static bases",
            "shaded triangle faces + wire edges; no shaft/moving-gripper draw",
            "cyan=GT mesh, orange=pred mesh; lines join identical canonical keys",
        ),
        int(args.column_gap),
    )
    return row, {
        "gt_wrist_mesh_pixels": int(gt_mask.sum()),
        "pred_wrist_mesh_pixels": int(pred_mask.sum()),
        "wrist_mesh_iou": float(np.count_nonzero(gt_mask & pred_mask) / max(1, np.count_nonzero(gt_mask | pred_mask))),
        "wrist_mesh_roi_xyxy": str(tuple(int(value) for value in roi)),
    }


def compose_case(result, renderer, args):
    pair_colors = vis_base.line_colors(len(result["shown_query_uv"]))
    crop_row = link_vis.compose_crop_row(result, args, pair_colors)
    mesh_row, mesh_stats = compose_wrist_mesh_row(result, renderer, args, pair_colors)
    banner = 98
    gap = 14
    output = Image.new(
        "RGB",
        (crop_row.shape[1], banner + crop_row.shape[0] + gap + mesh_row.shape[0]),
        (230, 235, 241),
    )
    output.paste(Image.fromarray(crop_row), (0, banner))
    output.paste(Image.fromarray(mesh_row), (0, banner + crop_row.shape[0] + gap))
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
        "filled dot=detected source; cyan ring=its selected 3D key under GT; orange cross=same key under predicted pose",
        fill=(71, 85, 105),
        font=vis_base.font(13),
    )
    return output, mesh_stats


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
    parser.add_argument("--num_inliers", type=int, default=14)
    parser.add_argument("--num_outliers", type=int, default=6)
    parser.add_argument("--panel_size", type=int, default=560)
    parser.add_argument("--column_gap", type=int, default=28)
    parser.add_argument("--roi_padding", type=int, default=10)
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
    wrist_renderer = InstrumentOpenGLDepthRenderer(
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
            image, mesh_stats = compose_case(result, wrist_renderer, args)
            path = output_dir / f"frame{frame_id:03d}_source_gt_wristmesh.jpg"
            image.save(path, quality=96)
            paths.append(path)
            metrics.append(
                {
                    "frame_id": frame_id,
                    "checkpoint_iter": int(checkpoint_iter),
                    "trans_err_mm": result["csv_trans_err_mm"],
                    "rot_err_deg": result["csv_rot_err_deg"],
                    "shown_sources": int(len(result["shown_query_uv"])),
                    **mesh_stats,
                    "visualization": str(path),
                }
            )
            print(
                f"frame={frame_id} sources={len(result['shown_query_uv'])} "
                f"gt_wrist_pixels={mesh_stats['gt_wrist_mesh_pixels']} "
                f"mesh_iou={mesh_stats['wrist_mesh_iou']:.3f} saved={path}",
                flush=True,
            )
    finally:
        wrist_renderer.release()
        key_renderer.release()

    comparison_vis.write_rows(output_dir / "selected_failure_metrics.csv", metrics)
    pages = link_vis.make_contact_pages(paths, output_dir)
    report = [
        "# SurfEmb iter38000 source-to-GT complete-wrist visualization",
        "",
        f"- checkpoint: {Path(args.checkpoint).resolve()}",
        f"- evaluation: {Path(args.eval_csv).resolve()}",
        f"- frames: {frame_ids}",
        "- OpenGL draw set: wrist OBJ plus x<2.13 mm static gripper-base triangles.",
        "- Shaft and moving-gripper triangles are excluded from these wrist-mesh panels.",
        "- Filled dot: detected source pixel; cyan ring: selected key at GT pose; orange cross: same key at predicted pose.",
        f"- pages: {[path.name for path in pages]}",
    ]
    (output_dir / "README.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"output={output_dir}", flush=True)


if __name__ == "__main__":
    main(build_parser().parse_args())
