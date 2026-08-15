#!/usr/bin/env python3
"""Diagnose crop-scale and correspondence-scale effects on LND wrist pose."""

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr

import eval_surfemb_articulated_rarp as surf_eval
import eval_surfemb_wrist_lnd as lnd_eval
from instrument_geometry import quat_wxyz_to_matrix_np
from surfemb_articulated_pose import (
    build_part_probability_inputs,
    downsample_intrinsics,
    encode_surface_keys,
    estimate_part_pose_topk_ransac,
    load_part_surfaces,
    prepare_part_score_context,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "logs" / "surfemb_correspondence_scale_diagnostic_lnd_20260805"
DEFAULT_EXPERIMENTS = (
    (
        "old_point_resnet_full26k",
        ROOT / "logs/surfemb_resnet_crop224_rarp_lnd_refinemem_b56_gpu4567/checkpoints/best_val_total.pt",
        "instance",
    ),
    (
        "triangle_resnet_full26k",
        ROOT / "logs/surfemb_training_progress_lnd_20260804/checkpoints/full_last_iter26000.pt",
        "instance",
    ),
    (
        "triangle_resnet_full36k",
        ROOT / "logs/surfemb_training_progress_lnd_20260805/checkpoints/full_last_iter36000.pt",
        "instance",
    ),
    (
        "triangle_resnet_wrist38k",
        ROOT / "logs/surfemb_training_progress_lnd_20260805/checkpoints/wristonly_last_iter38000.pt",
        "wrist",
    ),
)


def make_crop(target, crop_source, crop_size=224, crop_scale=1.2):
    rgb = target["orig_rgb"].cpu().numpy().astype(np.uint8)
    inst = target["inst_mask_orig"].cpu().numpy().astype(bool)
    part = target["part_mask_orig"].cpu().numpy()
    crop_mask = inst if crop_source == "instance" else inst & (part == 2)
    if not crop_mask.any():
        raise RuntimeError(f"Empty {crop_source} crop mask")
    M = surf_eval._surf_aug.random_rotated_mask_crop_matrix(
        crop_mask,
        int(crop_size),
        crop_scale=float(crop_scale),
        max_angle=0.0,
        offset_scale=0.0,
        ensure_full_mask=True,
    )
    crop = cv2.warpAffine(
        rgb,
        M,
        (int(crop_size), int(crop_size)),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    wrist_crop = cv2.warpAffine(
        (inst & (part == 2)).astype(np.uint8),
        M,
        (int(crop_size), int(crop_size)),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)
    K = surf_eval._surf_aug.matrix3_from_affine(M) @ target["K_orig"].cpu().numpy()
    return surf_eval._surf_aug.imagenet_tensor(crop), K.astype(np.float32), wrist_crop, M


def project(points_m, quat_wxyz, trans_m, K):
    R = quat_wxyz_to_matrix_np(np.asarray(quat_wxyz, dtype=np.float64))
    points_cam = np.asarray(points_m, dtype=np.float64) @ R.T + np.asarray(trans_m, dtype=np.float64)
    uvw = points_cam @ np.asarray(K, dtype=np.float64).T
    return uvw[:, :2] / uvw[:, 2:3]


def downsample_pixels_to_original(points_ds, M, scale):
    points_crop = (np.asarray(points_ds, dtype=np.float64) + 0.5) * float(scale) - 0.5
    inverse = cv2.invertAffineTransform(np.asarray(M, dtype=np.float64))
    return cv2.transform(points_crop[None].astype(np.float64), inverse)[0]


def similarity_metrics(gt_uv, observed_uv):
    gt_uv = np.asarray(gt_uv, dtype=np.float64)
    observed_uv = np.asarray(observed_uv, dtype=np.float64)
    gt_mean = gt_uv.mean(axis=0)
    observed_mean = observed_uv.mean(axis=0)
    X = gt_uv - gt_mean
    Y = observed_uv - observed_mean
    gt_energy = float(np.sum(X * X))
    obs_energy = float(np.sum(Y * Y))
    if gt_energy <= 1e-12 or obs_energy <= 1e-12:
        return None
    U, singular, Vt = np.linalg.svd(X.T @ Y)
    R2 = Vt.T @ U.T
    if np.linalg.det(R2) < 0.0:
        Vt[-1] *= -1.0
        R2 = Vt.T @ U.T
        singular[-1] *= -1.0
    scale = float(np.sum(singular) / gt_energy)
    fitted = scale * (X @ R2.T) + observed_mean
    residual = np.linalg.norm(observed_uv - fitted, axis=1)
    direct = np.linalg.norm(observed_uv - gt_uv, axis=1)
    return {
        "center_dx_px": float(observed_mean[0] - gt_mean[0]),
        "center_dy_px": float(observed_mean[1] - gt_mean[1]),
        "center_err_px": float(np.linalg.norm(observed_mean - gt_mean)),
        "span_ratio": float(math.sqrt(obs_energy / gt_energy)),
        "similarity_scale": scale,
        "similarity_rot_deg": float(math.degrees(math.atan2(R2[1, 0], R2[0, 0]))),
        "local_residual_mean_px": float(residual.mean()),
        "local_residual_median_px": float(np.median(residual)),
        "direct_residual_median_px": float(np.median(direct)),
    }


def hull_area(points):
    points = np.asarray(points, dtype=np.float32)
    if len(points) < 3:
        return 0.0
    return float(cv2.contourArea(cv2.convexHull(points)))


def surface_coverage(points, all_points):
    points = np.asarray(points, dtype=np.float64)
    all_points = np.asarray(all_points, dtype=np.float64)
    selected_diag = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
    total_diag = np.linalg.norm(all_points.max(axis=0) - all_points.min(axis=0))
    selected_rms = np.sqrt(np.mean(np.sum((points - points.mean(axis=0)) ** 2, axis=1)))
    total_rms = np.sqrt(np.mean(np.sum((all_points - all_points.mean(axis=0)) ** 2, axis=1)))
    return float(selected_diag / total_diag), float(selected_rms / total_rms)


def finite_corr(x, y, kind="pearson"):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3 or np.std(x[valid]) < 1e-12 or np.std(y[valid]) < 1e-12:
        return float("nan")
    fn = pearsonr if kind == "pearson" else spearmanr
    return float(fn(x[valid], y[valid]).statistic)


def summarize(rows):
    metrics = [
        "wrist_patch_fraction",
        "selected_count",
        "inlier_count",
        "selected_span_ratio",
        "selected_center_err_px",
        "selected_local_residual_median_px",
        "selected_direct_residual_median_px",
        "selected_original_center_err_px",
        "selected_original_local_residual_median_px",
        "selected_original_direct_residual_median_px",
        "selected_key_bbox_coverage",
        "selected_key_rms_coverage",
        "selected_gt_hull_fraction",
        "trans_err_mm",
        "signed_z_err_mm",
        "rot_err_deg",
    ]
    out = {"count": len(rows)}
    for key in metrics:
        values = np.asarray([row.get(key, np.nan) for row in rows], dtype=np.float64)
        values = values[np.isfinite(values)]
        out[key] = {
            "mean": float(values.mean()) if len(values) else float("nan"),
            "median": float(np.median(values)) if len(values) else float("nan"),
        }
    out["correlations"] = {
        "span_ratio_vs_signed_z_pearson": finite_corr(
            [r["selected_span_ratio"] for r in rows], [r["signed_z_err_mm"] for r in rows]
        ),
        "span_ratio_vs_signed_z_spearman": finite_corr(
            [r["selected_span_ratio"] for r in rows], [r["signed_z_err_mm"] for r in rows], "spearman"
        ),
        "local_residual_vs_rotation_pearson": finite_corr(
            [r["selected_original_local_residual_median_px"] for r in rows],
            [r["rot_err_deg"] for r in rows],
        ),
        "local_residual_vs_rotation_spearman": finite_corr(
            [r["selected_original_local_residual_median_px"] for r in rows],
            [r["rot_err_deg"] for r in rows],
            "spearman",
        ),
        "patch_fraction_vs_rotation_spearman": finite_corr(
            [r["wrist_patch_fraction"] for r in rows], [r["rot_err_deg"] for r in rows], "spearman"
        ),
    }
    return out


def save_plot(rows, output):
    labels = list(dict.fromkeys(row["experiment"] for row in rows))
    colors = ["#6b7280", "#7c3aed", "#2563eb", "#16a34a"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    box_metrics = [
        ("selected_span_ratio", "Assigned span ratio (observed / GT)"),
        ("selected_original_local_residual_median_px", "Local residual in original image (px)"),
    ]
    for ax, (key, title) in zip(axes[0], box_metrics):
        values = [[r[key] for r in rows if r["experiment"] == label] for label in labels]
        bp = ax.boxplot(values, labels=labels, showfliers=False, patch_artist=True)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=15)
        ax.grid(alpha=0.25)
    for label, color in zip(labels, colors):
        subset = [r for r in rows if r["experiment"] == label]
        axes[1, 0].scatter(
            [r["selected_span_ratio"] for r in subset],
            [r["signed_z_err_mm"] for r in subset],
            s=13,
            alpha=0.65,
            color=color,
            label=label,
        )
        axes[1, 1].scatter(
            [r["selected_original_local_residual_median_px"] for r in subset],
            [r["rot_err_deg"] for r in subset],
            s=13,
            alpha=0.65,
            color=color,
            label=label,
        )
    axes[1, 0].axvline(1.0, color="black", linewidth=0.8)
    axes[1, 0].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 0].set(xlabel="Assigned span ratio", ylabel="Signed z error (mm)", title="Scale bias vs depth bias")
    axes[1, 1].set(
        xlabel="Local residual in original image (px)",
        ylabel="Rotation error (deg)",
        title="Local correspondence error vs rotation",
    )
    for ax in axes[1]:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.savefig(output, dpi=180)
    plt.close(fig)


@torch.inference_mode()
def main(args):
    cv2.setNumThreads(0)
    torch.set_num_threads(int(args.cpu_threads))
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    dataset_args = argparse.Namespace(
        lnd_root=args.lnd_root,
        crop_size=args.crop_size,
        canonical_eps=args.canonical_eps,
        bbox_padding_frac=0.12,
    )
    dataset = lnd_eval.build_dataset(dataset_args)
    valid_indices = [i for i, sample in enumerate(dataset.samples) if int(sample[0]) != 210]
    positions = np.linspace(0, len(valid_indices) - 1, int(args.num_frames)).round().astype(int)
    indices = [valid_indices[i] for i in np.unique(positions)]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for experiment, checkpoint, crop_source in DEFAULT_EXPERIMENTS:
        spec = surf_eval.parse_model_specs([f"{experiment}={checkpoint}"])[0]
        model, checkpoint_iter, _ = surf_eval.load_model(spec, device)
        surfaces = load_part_surfaces(args.surface_root, keys_per_part=args.surface_keys_per_part, seed=args.surface_seed)
        encode_surface_keys(model, surfaces, device, mask_keys_per_part=args.mask_keys_per_part)
        wrist_surface = surfaces["wrist"]
        print(f"[{experiment}] iter={checkpoint_iter} crop={crop_source} frames={len(indices)}", flush=True)

        for count, dataset_index in enumerate(indices, 1):
            _, target = dataset[dataset_index]
            image, K_crop, wrist_crop, M_crop = make_crop(
                target, crop_source, args.crop_size, args.crop_scale
            )
            x = image[None].to(device)
            K_tensor = torch.from_numpy(K_crop)[None].to(device)
            with torch.amp.autocast(device.type, enabled=device.type == "cuda", dtype=torch.bfloat16):
                prediction = model(x, K_tensor)
            query_flat, _, K_ds, image_hw, _ = build_part_probability_inputs(
                prediction["inst_mask_logits"][0].float(),
                prediction["surfemb_queries"][0].float(),
                surfaces,
                K_crop,
                down_sample_scale=args.down_sample_scale,
            )
            gt_wrist_roi = F.max_pool2d(
                torch.as_tensor(wrist_crop, device=device, dtype=torch.float32)[None, None],
                args.down_sample_scale,
                args.down_sample_scale,
            )[0, 0] > 0
            context = prepare_part_score_context(
                query_flat,
                lnd_eval.probability_input_from_mask(gt_wrist_roi, image_hw),
                wrist_surface,
                image_hw,
            )
            best, diagnostic = estimate_part_pose_topk_ransac(
                context,
                wrist_surface,
                K_ds,
                image_hw,
                pixel_mask=gt_wrist_roi.reshape(-1),
                max_correspondences=args.max_correspondences,
                min_correspondences=12,
                min_part_probability=0.05,
                ransac_iterations=2000,
                ransac_reprojection_error=3.0,
                ransac_confidence=0.999,
                min_inliers=8,
                min_inlier_fraction=0.6,
                return_correspondences=True,
            )
            if best is None:
                print(f"  skip frame={target['frame_id']} status={diagnostic['topk_status']}", flush=True)
                continue

            selected_pixels = diagnostic["topk_selected_pixel_indices"]
            selected_keys = diagnostic["topk_selected_key_indices"]
            h, w = image_hw
            observed_uv = np.stack((selected_pixels % w, selected_pixels // w), axis=1).astype(np.float64)
            gt = surf_eval.target_pose(target)
            gt_uv = project(wrist_surface.points_m[selected_keys], gt["rot"], gt["trans"], K_ds)
            finite = np.isfinite(gt_uv).all(axis=1)
            selected_metrics = similarity_metrics(gt_uv[finite], observed_uv[finite])
            if selected_metrics is None:
                continue
            observed_original = downsample_pixels_to_original(
                observed_uv[finite], M_crop, args.down_sample_scale
            )
            gt_original = downsample_pixels_to_original(gt_uv[finite], M_crop, args.down_sample_scale)
            original_metrics = similarity_metrics(gt_original, observed_original)
            if original_metrics is None:
                continue
            selected_points = wrist_surface.points_m[selected_keys[finite]]
            bbox_coverage, rms_coverage = surface_coverage(selected_points, wrist_surface.points_m)

            raw = {
                "rot": lnd_eval.matrix_to_quat_wxyz(best.transform[:3, :3]),
                "trans": best.transform[:3, 3],
                "alpha": 0.0,
                "theta_l": 0.0,
                "theta_r": 0.0,
            }
            canonical = surf_eval.canonicalize_prediction(raw, args.canonical_eps)
            trans_delta_mm = (canonical["trans"] - gt["trans"]) * 1000.0
            row = {
                "experiment": experiment,
                "checkpoint_iter": checkpoint_iter,
                "crop_source": crop_source,
                "dataset_idx": dataset_index,
                "frame_id": int(target["frame_id"]),
                "wrist_patch_fraction": float(wrist_crop.mean()),
                "wrist_area_ds": int(gt_wrist_roi.sum().item()),
                "selected_count": int(len(selected_pixels)),
                "inlier_count": int(diagnostic["topk_inliers"]),
                "selected_key_bbox_coverage": bbox_coverage,
                "selected_key_rms_coverage": rms_coverage,
                "selected_gt_hull_fraction": hull_area(gt_uv[finite]) / max(1.0, float(gt_wrist_roi.sum().item())),
                "trans_err_mm": float(np.linalg.norm(trans_delta_mm)),
                "signed_x_err_mm": float(trans_delta_mm[0]),
                "signed_y_err_mm": float(trans_delta_mm[1]),
                "signed_z_err_mm": float(trans_delta_mm[2]),
                "rot_err_deg": surf_eval.rotation_error_deg(canonical["rot"], gt["rot"]),
            }
            row.update({f"selected_{key}": value for key, value in selected_metrics.items()})
            row.update({f"selected_original_{key}": value for key, value in original_metrics.items()})
            rows.append(row)
            if count == 1 or count % 16 == 0 or count == len(indices):
                print(f"  {count}/{len(indices)} frame={row['frame_id']}", flush=True)
        del model, surfaces
        torch.cuda.empty_cache()

    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (output_dir / "per_frame.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        experiment: summarize([row for row in rows if row["experiment"] == experiment])
        for experiment, _, _ in DEFAULT_EXPERIMENTS
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")
    save_plot(rows, output_dir / "diagnostic.png")
    print(json.dumps(summary, indent=2, allow_nan=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--lnd_root", default="/mnt/iMVR/daiyun/Dataset/LND")
    parser.add_argument("--surface_root", default=str(surf_eval.DEFAULT_SURFACE_ROOT))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_frames", type=int, default=64)
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--crop_scale", type=float, default=1.2)
    parser.add_argument("--down_sample_scale", type=int, default=3)
    parser.add_argument("--surface_keys_per_part", type=int, default=4096)
    parser.add_argument("--mask_keys_per_part", type=int, default=512)
    parser.add_argument("--surface_seed", type=int, default=2026)
    parser.add_argument("--max_correspondences", type=int, default=512)
    parser.add_argument("--canonical_eps", type=float, default=0.08)
    parser.add_argument("--cpu_threads", type=int, default=2)
    main(parser.parse_args())
