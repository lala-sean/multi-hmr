#!/usr/bin/env python3
"""Diagnose LND wrist-only depth error without changing evaluation code."""

import argparse
import csv
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

import eval_surfemb_articulated_rarp as surf_eval
import eval_surfemb_wrist_lnd as lnd_eval
from instrument_geometry import quat_wxyz_to_matrix_np
from surfemb_articulated_pose import downsample_intrinsics, load_part_surfaces


ROOT = Path(__file__).resolve().parent
DEFAULT_ROOT = ROOT / "logs" / "surfemb_current_last_lnd_eval_20260804_iter21000_22000"


def read_rows(path):
    with Path(path).open(encoding="utf-8") as handle:
        return {int(row["frame_id"]): row for row in csv.DictReader(handle) if row["status"] == "ok"}


def project(points, rotation, translation, K):
    camera = np.asarray(points, dtype=np.float64) @ rotation.T + np.asarray(translation, dtype=np.float64)
    uvw = camera @ np.asarray(K, dtype=np.float64).T
    uv = uvw[:, :2] / uvw[:, 2:]
    return uv, camera[:, 2]


def affine_points(points, M):
    points = np.asarray(points, dtype=np.float64)
    return np.concatenate([points, np.ones((len(points), 1), dtype=np.float64)], axis=1) @ np.asarray(M).T


def corrcoef(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3 or np.std(x[valid]) == 0.0 or np.std(y[valid]) == 0.0:
        return float("nan")
    return float(np.corrcoef(x[valid], y[valid])[0, 1])


def component_stats(delta_mm):
    delta_mm = np.asarray(delta_mm, dtype=np.float64)
    names = ("x", "y", "z")
    return {
        name: {
            "signed_mean_mm": float(delta_mm[:, index].mean()),
            "abs_mean_mm": float(np.abs(delta_mm[:, index]).mean()),
            "abs_median_mm": float(np.median(np.abs(delta_mm[:, index]))),
            "abs_p90_mm": float(np.percentile(np.abs(delta_mm[:, index]), 90)),
        }
        for index, name in enumerate(names)
    }


def make_plot(path, full_delta, wrist_delta, wrist_rows, wrist_crop_fx):
    ids = sorted(wrist_rows)
    inlier = np.asarray([float(wrist_rows[frame]["topk_inlier_fraction"]) for frame in ids])
    reproj = np.asarray([float(wrist_rows[frame]["topk_reprojection_median_px"]) for frame in ids])
    dz = wrist_delta[:, 2]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    axes[0].boxplot(
        [np.abs(full_delta[:, 0]), np.abs(wrist_delta[:, 0]), np.abs(full_delta[:, 1]),
         np.abs(wrist_delta[:, 1]), np.abs(full_delta[:, 2]), np.abs(wrist_delta[:, 2])],
        labels=["F |dx|", "W |dx|", "F |dy|", "W |dy|", "F |dz|", "W |dz|"],
        showfliers=False,
    )
    axes[0].set_ylabel("absolute translation error (mm)")
    axes[0].set_title("Error components")
    axes[0].grid(axis="y", alpha=0.25)

    scatter = axes[1].scatter(inlier, np.abs(dz), c=reproj, s=15, cmap="viridis", alpha=0.75)
    axes[1].set_xlabel("RANSAC inlier fraction (fit-space)")
    axes[1].set_ylabel("wrist-only |dz| (mm)")
    axes[1].set_title("Many inliers do not guarantee correct depth")
    axes[1].grid(alpha=0.25)
    fig.colorbar(scatter, ax=axes[1], label="fit reprojection median (74px map)")

    axes[2].scatter(wrist_crop_fx, dz, s=15, alpha=0.7, color="#2563eb")
    axes[2].axhline(0.0, color="black", linewidth=1)
    axes[2].set_xlabel("wrist crop focal length fx (pixels)")
    axes[2].set_ylabel("signed wrist-only dz (mm)")
    axes[2].set_title("Depth bias versus crop scale")
    axes[2].grid(alpha=0.25)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full_csv", default=str(DEFAULT_ROOT / "full_instance_crop_gt_wrist_roi/surfemb_per_frame.csv"))
    parser.add_argument("--wrist_csv", default=str(DEFAULT_ROOT / "wrist_crop/surfemb_per_frame.csv"))
    parser.add_argument("--selected_csv", default=str(DEFAULT_ROOT / "full_gtroi_vs_wrist_correspondence_vis/selected_case_metrics.csv"))
    parser.add_argument("--output_dir", default=str(DEFAULT_ROOT / "wristonly_depth_intrinsics_debug"))
    parser.add_argument("--lnd_root", default="/mnt/iMVR/daiyun/Dataset/LND")
    parser.add_argument("--surface_root", default=str(surf_eval.DEFAULT_SURFACE_ROOT))
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--surfemb_crop_scale", type=float, default=1.2)
    parser.add_argument("--bbox_padding_frac", type=float, default=0.12)
    parser.add_argument("--canonical_eps", type=float, default=0.08)
    parser.add_argument("--down_sample_scale", type=int, default=3)
    parser.add_argument("--exclude_frame_ids", nargs="*", type=int, default=[210])
    return parser


def main(args):
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    full_rows = read_rows(args.full_csv)
    wrist_rows = read_rows(args.wrist_csv)
    frame_ids = sorted((set(full_rows) & set(wrist_rows)) - set(args.exclude_frame_ids))

    dataset_args = lnd_eval.build_parser().parse_args([])
    dataset_args.lnd_root = args.lnd_root
    dataset_args.crop_size = args.crop_size
    dataset_args.bbox_padding_frac = args.bbox_padding_frac
    dataset_args.canonical_eps = args.canonical_eps
    dataset_args.surfemb_crop_scale = args.surfemb_crop_scale
    dataset = lnd_eval.build_dataset(dataset_args)
    frame_to_index = {int(sample[0]): index for index, sample in enumerate(dataset.samples)}

    surfaces = load_part_surfaces(args.surface_root, keys_per_part=4096, seed=2026)
    wrist_points = np.asarray(surfaces["wrist"].points_m, dtype=np.float64)
    wrist_points = wrist_points[np.linspace(0, len(wrist_points) - 1, min(256, len(wrist_points))).astype(int)]

    full_delta = []
    wrist_delta = []
    wrist_crop_fx = []
    gt_z_mm = []
    affine_errors = []
    downsample_errors = []
    csv_total_error = []
    for frame_id in frame_ids:
        _, target = dataset[frame_to_index[frame_id]]
        gt = surf_eval.target_pose(target)
        gt_t = np.asarray(gt["trans"], dtype=np.float64)
        gt_z_mm.append(float(gt_t[2] * 1000.0))
        rotation = quat_wxyz_to_matrix_np(gt["rot"])
        K_orig = target["K_orig"].detach().cpu().numpy().astype(np.float64)

        deltas = []
        for rows in (full_rows, wrist_rows):
            row = rows[frame_id]
            pred_t = np.asarray([row["canonical_tx_m"], row["canonical_ty_m"], row["canonical_tz_m"]], dtype=np.float64)
            deltas.append((pred_t - gt_t) * 1000.0)
            csv_total_error.append(abs(np.linalg.norm(deltas[-1]) - float(row["canonical_trans_err_mm"])))
        full_delta.append(deltas[0])
        wrist_delta.append(deltas[1])

        dataset_args.crop_mask_source = "wrist"
        _, K_crop, _, M_crop = lnd_eval.make_eval_crop(target, dataset_args)
        wrist_crop_fx.append(float(K_crop[0, 0]))
        uv_orig, depth = project(wrist_points, rotation, gt_t, K_orig)
        uv_affine = affine_points(uv_orig, M_crop)
        uv_crop, _ = project(wrist_points, rotation, gt_t, K_crop)
        valid = depth > 0.0
        affine_errors.append(float(np.max(np.linalg.norm(uv_affine[valid] - uv_crop[valid], axis=1))))

        K_ds = downsample_intrinsics(K_crop, args.down_sample_scale)
        uv_ds_expected = (uv_crop + 0.5) / float(args.down_sample_scale) - 0.5
        uv_ds, _ = project(wrist_points, rotation, gt_t, K_ds)
        downsample_errors.append(float(np.max(np.linalg.norm(uv_ds_expected[valid] - uv_ds[valid], axis=1))))

    full_delta = np.asarray(full_delta)
    wrist_delta = np.asarray(wrist_delta)
    wrist_crop_fx = np.asarray(wrist_crop_fx)
    gt_z_mm = np.asarray(gt_z_mm)
    inlier = np.asarray([float(wrist_rows[frame]["topk_inlier_fraction"]) for frame in frame_ids])
    reproj = np.asarray([float(wrist_rows[frame]["topk_reprojection_median_px"]) for frame in frame_ids])

    selected = []
    selected_path = Path(args.selected_csv)
    if selected_path.exists():
        with selected_path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["model"] == "resnet_wristonly":
                    selected.append(
                        {
                            "frame_id": int(row["frame_id"]),
                            "ransac_inlier_fraction": float(row["csv_inlier_fraction"]),
                            "fit_reprojection_median_px_ds": float(wrist_rows[int(row["frame_id"])]["topk_reprojection_median_px"]),
                            "gt_correspondence_residual_median_px_crop": float(row["gt_inlier_residual_median_px"]),
                            "dz_mm": float(row["dt_z_mm"]),
                        }
                    )

    report = {
        "num_frames": len(frame_ids),
        "intrinsics_checks": {
            "max_affine_projection_error_px": float(max(affine_errors)),
            "max_downsample_projection_error_px": float(max(downsample_errors)),
            "max_csv_translation_reconstruction_error_mm": float(max(csv_total_error)),
            "formula_crop": "K_crop = affine_3x3 @ K_orig",
            "formula_downsample": "c_ds = (c_crop + 0.5) / 3 - 0.5",
        },
        "full_gt_roi_translation_components": component_stats(full_delta),
        "wrist_only_translation_components": component_stats(wrist_delta),
        "wrist_only_correlations": {
            "abs_dz_vs_ransac_inlier_fraction": corrcoef(np.abs(wrist_delta[:, 2]), inlier),
            "abs_dz_vs_fit_reprojection_median": corrcoef(np.abs(wrist_delta[:, 2]), reproj),
            "signed_dz_vs_crop_fx": corrcoef(wrist_delta[:, 2], wrist_crop_fx),
            "abs_dz_vs_crop_fx": corrcoef(np.abs(wrist_delta[:, 2]), wrist_crop_fx),
            "crop_fx_vs_gt_z": corrcoef(wrist_crop_fx, gt_z_mm),
            "signed_dz_vs_gt_z": corrcoef(wrist_delta[:, 2], gt_z_mm),
        },
        "wrist_only_depth_scale": {
            "gt_z_mean_mm": float(gt_z_mm.mean()),
            "signed_relative_dz_mean_percent": float(np.mean(wrist_delta[:, 2] / gt_z_mm) * 100.0),
            "signed_relative_dz_median_percent": float(np.median(wrist_delta[:, 2] / gt_z_mm) * 100.0),
            "pred_z_vs_gt_z_slope": float(np.polyfit(gt_z_mm, gt_z_mm + wrist_delta[:, 2], 1)[0]),
            "pred_z_vs_gt_z_intercept_mm": float(np.polyfit(gt_z_mm, gt_z_mm + wrist_delta[:, 2], 1)[1]),
        },
        "selected_correspondence_examples": selected,
    }
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    make_plot(output_dir / "depth_intrinsics_diagnostic.png", full_delta, wrist_delta, wrist_rows, wrist_crop_fx)
    print(json.dumps(report, indent=2), flush=True)
    print(f"plot={output_dir / 'depth_intrinsics_diagnostic.png'}", flush=True)


if __name__ == "__main__":
    main(build_parser().parse_args())
