#!/usr/bin/env python3
"""RARP ablation for wrist-only, wrist+gripper, and gripper-only SurfEmb fits."""

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from eval_surfemb_articulated_rarp import (  # noqa: E402
    DEFAULT_MODELS,
    DEFAULT_SURFACE_ROOT,
    build_dataset,
    canonicalize_prediction,
    load_model,
    make_surfemb_crop,
    parse_model_specs,
    rotation_error_deg,
    stat,
    target_pose,
)
from instrument_geometry import fk_matrices_np  # noqa: E402
from instrument_opengl_renderer import InstrumentOpenGLDepthRenderer  # noqa: E402
from surfemb_articulated_pose import (  # noqa: E402
    PART_NAMES,
    _matrix_to_quat_wxyz,
    articulated_add_mm,
    build_part_probability_inputs,
    encode_surface_keys,
    estimate_chain_joint_angle,
    estimate_part_pose_topk_ransac,
    fit_kinematic_chain,
    load_part_surfaces,
    part_pose_errors,
    prepare_part_score_context,
    refine_part_pose_with_triangle_visibility,
)


ROOT = Path(__file__).resolve().parent
GRIPPER_NAMES = ("l_gripper", "r_gripper")


class Pose:
    """Compatibility shim for legacy RARP memory_pool.pth pickles."""

    def __setstate__(self, state):
        self.__dict__.update(state if isinstance(state, dict) else {})


def _angle_error_deg(pred, gt):
    delta = math.atan2(math.sin(float(pred) - float(gt)), math.cos(float(pred) - float(gt)))
    return abs(math.degrees(delta))


def _pose_from_wrist_transform(transform, theta_l=0.0, theta_r=0.0, alpha=0.0):
    transform = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    return {
        "rot": _matrix_to_quat_wxyz(transform[:3, :3]),
        "trans": transform[:3, 3].copy(),
        "alpha": float(alpha),
        "theta_l": float(theta_l),
        "theta_r": float(theta_r),
        "chain_cost": float("nan"),
        "chain_nfev": 0,
    }


def _wrist_metrics(prefix, pose, gt):
    return {
        f"{prefix}_trans_err_mm": float(
            np.linalg.norm(np.asarray(pose["trans"]) - np.asarray(gt["trans"])) * 1000.0
        ),
        f"{prefix}_rot_err_deg": rotation_error_deg(pose["rot"], gt["rot"]),
    }


def _part_transform_metrics(prefix, transform, gt_transform, points):
    transform = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    gt_transform = np.asarray(gt_transform, dtype=np.float64).reshape(4, 4)
    relative = transform[:3, :3] @ gt_transform[:3, :3].T
    cosine = np.clip((float(np.trace(relative)) - 1.0) * 0.5, -1.0, 1.0)
    pred_points = np.asarray(points, dtype=np.float64) @ transform[:3, :3].T + transform[:3, 3]
    gt_points = np.asarray(points, dtype=np.float64) @ gt_transform[:3, :3].T + gt_transform[:3, 3]
    return {
        f"{prefix}_trans_err_mm": float(np.linalg.norm(transform[:3, 3] - gt_transform[:3, 3]) * 1000.0),
        f"{prefix}_rot_err_deg": float(np.degrees(np.arccos(cosine))),
        f"{prefix}_add_mm": float(np.linalg.norm(pred_points - gt_points, axis=1).mean() * 1000.0),
    }


def _crop_part_mask(target, crop_matrix, crop_size):
    part_orig = target["part_mask_orig"].detach().cpu().numpy().astype(np.uint8)
    return cv2.warpAffine(
        part_orig,
        np.asarray(crop_matrix, dtype=np.float32),
        (int(crop_size), int(crop_size)),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _part_rois(part_inputs, object_prob, threshold):
    probabilities = torch.stack([part_inputs[name]["prob"] for name in PART_NAMES], dim=1)
    labels = probabilities.argmax(dim=1)
    object_roi = object_prob >= float(threshold)
    return {
        name: (labels == PART_NAMES.index(name)) & object_roi
        for name in ("wrist", *GRIPPER_NAMES)
    }


def _fit_part(context, surface, roi, K, image_hw, args, return_correspondences=False):
    return estimate_part_pose_topk_ransac(
        context,
        surface,
        K,
        image_hw,
        pixel_mask=roi,
        max_correspondences=int(args.topk_max_correspondences),
        min_correspondences=int(args.topk_min_correspondences),
        min_part_probability=float(args.topk_min_part_probability),
        margin_power=float(args.topk_margin_power),
        ransac_iterations=int(args.topk_ransac_iterations),
        ransac_reprojection_error=float(args.topk_ransac_reprojection_error),
        ransac_confidence=float(args.topk_ransac_confidence),
        min_inliers=int(args.topk_min_inliers),
        min_inlier_fraction=float(args.topk_min_inlier_fraction),
        return_correspondences=return_correspondences,
    )


@torch.inference_mode()
def evaluate_output(model, surfaces, output, K_crop, gt, part_crop, renderer, args):
    query_flat, part_inputs, K_ds, image_hw, object_prob = build_part_probability_inputs(
        output["inst_mask_logits"].float(),
        output["surfemb_queries"].float(),
        surfaces,
        K_crop,
        down_sample_scale=int(args.down_sample_scale),
    )
    contexts = {
        name: prepare_part_score_context(query_flat, part_inputs[name], surfaces[name], image_hw)
        for name in PART_NAMES
    }
    rois = _part_rois(part_inputs, object_prob, args.topk_min_object_probability)
    row = {f"pred_{name}_area_ds": int(rois[name].sum().item()) for name in rois}
    h, w = image_hw
    scale = int(args.down_sample_scale)
    ys = np.minimum(np.arange(h) * scale + scale // 2, part_crop.shape[0] - 1)
    xs = np.minimum(np.arange(w) * scale + scale // 2, part_crop.shape[1] - 1)
    gt_part_ds = np.asarray(part_crop, dtype=np.uint8)[np.ix_(ys, xs)]
    row["gt_wrist_area_ds"] = int(np.count_nonzero(gt_part_ds == 2))
    row["gt_gripper_area_ds"] = int(np.count_nonzero(gt_part_ds == 1))

    initial_hypotheses = {}
    initial_diagnostics = {}
    for name in ("wrist", *GRIPPER_NAMES):
        hypothesis, diag = _fit_part(
            contexts[name], surfaces[name], rois[name], K_ds, image_hw, args, return_correspondences=True
        )
        initial_hypotheses[name] = hypothesis
        initial_diagnostics[name] = diag
        row[f"initial_{name}_pnp_status"] = str(diag.get("topk_status", "unknown"))
        for key in (
            "topk_candidates",
            "topk_correspondences",
            "topk_inliers",
            "topk_inlier_fraction",
            "topk_confidence_median",
            "topk_reprojection_median_px",
        ):
            if key in diag:
                row[f"initial_{name}_{key}"] = float(diag[key])

    initial_wrist = initial_hypotheses["wrist"]
    if initial_wrist is None:
        row["status"] = "initial_wrist_pnp_failed"
        return row

    gt_transforms = fk_matrices_np(gt["rot"], gt["trans"], gt["alpha"], gt["theta_l"], gt["theta_r"])
    initial_chain_angles = {}
    for name in ("shaft", *GRIPPER_NAMES):
        try:
            initial_chain_angles[name] = estimate_chain_joint_angle(
                contexts[name],
                initial_wrist.transform,
                name,
                K_ds,
                image_hw,
                pose_batch_size=int(args.pose_batch_size),
                coarse_steps=int(args.coarse_steps),
                fine_steps=int(args.fine_steps),
            )["angle"]
        except RuntimeError:
            initial_chain_angles[name] = 0.0
    initial_pose = _pose_from_wrist_transform(
        initial_wrist.transform,
        initial_chain_angles["l_gripper"],
        initial_chain_angles["r_gripper"],
        initial_chain_angles["shaft"],
    )
    row.update(_wrist_metrics("initial_wrist_only", canonicalize_prediction(initial_pose, args.canonical_eps), gt))
    candidate_transforms = fk_matrices_np(
        initial_pose["rot"],
        initial_pose["trans"],
        initial_pose["alpha"],
        initial_pose["theta_l"],
        initial_pose["theta_r"],
    )
    for name in GRIPPER_NAMES:
        if initial_hypotheses[name] is not None:
            candidate_transforms[name] = initial_hypotheses[name].transform
            row.update(
                _part_transform_metrics(
                    f"initial_independent_{name}",
                    initial_hypotheses[name].transform,
                    gt_transforms[name],
                    surfaces[name].points_m,
                )
            )

    _, raster_part_ids, raster_depth, raster_valid = renderer.render_candidate_part_visibility(
        candidate_transforms,
        K_ds,
        image_hw,
    )
    hypotheses = {}
    for name in ("wrist", *GRIPPER_NAMES):
        initial = initial_hypotheses[name]
        if initial is None:
            hypotheses[name] = None
            row[f"{name}_pnp_status"] = "initial_pnp_failed"
            continue
        refined, visibility_diag = refine_part_pose_with_triangle_visibility(
            contexts[name],
            surfaces[name],
            name,
            initial,
            initial_diagnostics[name],
            K_ds,
            image_hw,
            raster_part_ids,
            raster_depth,
            raster_valid,
            depth_tolerance=float(args.visibility_depth_tolerance),
            min_correspondences=int(args.visibility_min_correspondences),
            ransac_iterations=int(args.topk_ransac_iterations),
            ransac_reprojection_error=float(args.topk_ransac_reprojection_error),
            ransac_confidence=float(args.topk_ransac_confidence),
            min_inliers=int(args.topk_min_inliers),
            min_inlier_fraction=float(args.topk_min_inlier_fraction),
        )
        hypotheses[name] = refined
        row[f"{name}_pnp_status"] = str(visibility_diag["visibility_status"])
        for key, value in visibility_diag.items():
            if isinstance(value, (int, float)):
                row[f"{name}_{key}"] = value

    wrist_hypothesis = hypotheses["wrist"]
    if wrist_hypothesis is None:
        row["status"] = "visibility_refined_wrist_pnp_failed"
        return row

    pred_angles = {}
    oracle_angles = {}
    for name in GRIPPER_NAMES:
        try:
            pred_angles[name] = estimate_chain_joint_angle(
                contexts[name],
                wrist_hypothesis.transform,
                name,
                K_ds,
                image_hw,
                pose_batch_size=int(args.pose_batch_size),
                coarse_steps=int(args.coarse_steps),
                fine_steps=int(args.fine_steps),
            )
            row[f"predwrist_{name}_angle_status"] = "ok"
        except RuntimeError as exc:
            row[f"predwrist_{name}_angle_status"] = str(exc)
        try:
            oracle_angles[name] = estimate_chain_joint_angle(
                contexts[name],
                gt_transforms["wrist"],
                name,
                K_ds,
                image_hw,
                pose_batch_size=int(args.pose_batch_size),
                coarse_steps=int(args.coarse_steps),
                fine_steps=int(args.fine_steps),
            )
            row[f"oraclewrist_{name}_angle_status"] = "ok"
        except RuntimeError as exc:
            row[f"oraclewrist_{name}_angle_status"] = str(exc)

    baseline = _pose_from_wrist_transform(
        wrist_hypothesis.transform,
        pred_angles.get("l_gripper", {}).get("angle", 0.0),
        pred_angles.get("r_gripper", {}).get("angle", 0.0),
    )
    baseline = canonicalize_prediction(baseline, args.canonical_eps)
    row["status"] = "ok"
    row.update(_wrist_metrics("wrist_only", baseline, gt))

    for name in GRIPPER_NAMES:
        theta_key = "theta_l" if name == "l_gripper" else "theta_r"
        if name in pred_angles:
            row[f"predwrist_{theta_key}_err_deg"] = _angle_error_deg(pred_angles[name]["angle"], gt[theta_key])
        if name in oracle_angles:
            row[f"oraclewrist_{theta_key}_err_deg"] = _angle_error_deg(oracle_angles[name]["angle"], gt[theta_key])
        if hypotheses[name] is not None:
            row.update(
                _part_transform_metrics(
                    f"independent_{name}", hypotheses[name].transform, gt_transforms[name], surfaces[name].points_m
                )
            )

    if all(hypotheses[name] is not None for name in GRIPPER_NAMES):
        selected = {name: hypotheses[name] for name in ("wrist", *GRIPPER_NAMES)}
        joint = canonicalize_prediction(fit_kinematic_chain(selected), args.canonical_eps)
        row["wg_joint_status"] = "ok"
        row.update(_wrist_metrics("wg_joint_wrist", joint, gt))
        row["wg_joint_theta_l_err_deg"] = _angle_error_deg(joint["theta_l"], gt["theta_l"])
        row["wg_joint_theta_r_err_deg"] = _angle_error_deg(joint["theta_r"], gt["theta_r"])
        row.update({f"wg_joint_{key}": value for key, value in part_pose_errors(joint, gt).items()})
        add_metrics = articulated_add_mm(joint, gt, surfaces)
        for name in ("wrist", *GRIPPER_NAMES):
            row[f"wg_joint_{name}_add_mm"] = add_metrics[f"{name}_add_mm"]
        row["wg_joint_wg_add_mm"] = float(
            np.mean([row[f"wg_joint_{name}_add_mm"] for name in ("wrist", *GRIPPER_NAMES)])
        )
        row["wg_minus_wrist_trans_mm"] = row["wg_joint_wrist_trans_err_mm"] - row["wrist_only_trans_err_mm"]
        row["wg_minus_wrist_rot_deg"] = row["wg_joint_wrist_rot_err_deg"] - row["wrist_only_rot_err_deg"]
    else:
        failed = [name for name in GRIPPER_NAMES if hypotheses[name] is None]
        row["wg_joint_status"] = "missing_" + "+".join(failed)
    return row


def _write_csv(path, rows):
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _metric_stats(rows, keys):
    return {key: stat([row.get(key, float("nan")) for row in rows]) for key in keys}


def _paired_refinement_stat(rows, initial_key, refined_key):
    pairs = [
        (float(row[initial_key]), float(row[refined_key]))
        for row in rows
        if initial_key in row
        and refined_key in row
        and math.isfinite(float(row[initial_key]))
        and math.isfinite(float(row[refined_key]))
    ]
    initial = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
    refined = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
    return {
        "count": int(len(pairs)),
        "initial": stat(initial),
        "refined": stat(refined),
        "delta_refined_minus_initial": stat(refined - initial),
        "refined_win_rate": float(np.mean(refined < initial)) if len(pairs) else float("nan"),
    }


def _summarize_model(rows, spec):
    selected = [row for row in rows if row["model"] == spec["name"]]
    wrist_ok = [row for row in selected if row.get("status") == "ok"]
    initial_wrist_ok = [row for row in selected if "initial_wrist_only_trans_err_mm" in row]
    joint_ok = [row for row in wrist_ok if row.get("wg_joint_status") == "ok"]
    independent = {
        name: [row for row in wrist_ok if row.get(f"{name}_pnp_status") == "ok"] for name in GRIPPER_NAMES
    }
    initial_independent = {
        name: [
            row
            for row in selected
            if row.get(f"initial_{name}_pnp_status") == "ok"
            and f"initial_independent_{name}_trans_err_mm" in row
        ]
        for name in GRIPPER_NAMES
    }
    wrist_keys = ["wrist_only_trans_err_mm", "wrist_only_rot_err_deg"]
    joint_keys = [
        "wg_joint_wrist_trans_err_mm",
        "wg_joint_wrist_rot_err_deg",
        "wg_joint_theta_l_err_deg",
        "wg_joint_theta_r_err_deg",
        "wg_joint_l_gripper_add_mm",
        "wg_joint_r_gripper_add_mm",
        "wg_joint_wg_add_mm",
    ]
    angle_keys = [
        "predwrist_theta_l_err_deg",
        "predwrist_theta_r_err_deg",
        "oraclewrist_theta_l_err_deg",
        "oraclewrist_theta_r_err_deg",
    ]
    result = {
        "checkpoint": spec["path"],
        "total": len(selected),
        "wrist_success": len(wrist_ok),
        "wrist_success_rate": len(wrist_ok) / max(1, len(selected)),
        "initial_wrist_success": len(initial_wrist_ok),
        "initial_wrist_only": _metric_stats(
            initial_wrist_ok,
            ["initial_wrist_only_trans_err_mm", "initial_wrist_only_rot_err_deg"],
        ),
        "visibility_refinement_paired": {
            "wrist_trans_err_mm": _paired_refinement_stat(
                wrist_ok, "initial_wrist_only_trans_err_mm", "wrist_only_trans_err_mm"
            ),
            "wrist_rot_err_deg": _paired_refinement_stat(
                wrist_ok, "initial_wrist_only_rot_err_deg", "wrist_only_rot_err_deg"
            ),
        },
        "full_wrist_gripper_success": len(joint_ok),
        "full_wrist_gripper_success_rate": len(joint_ok) / max(1, len(selected)),
        "wrist_only_all_successes": _metric_stats(wrist_ok, wrist_keys),
        "paired_wrist_only": _metric_stats(joint_ok, wrist_keys),
        "wrist_gripper_joint": _metric_stats(joint_ok, joint_keys),
        "joint_minus_wrist": _metric_stats(joint_ok, ["wg_minus_wrist_trans_mm", "wg_minus_wrist_rot_deg"]),
        "joint_wrist_win_rate": {
            "translation": float(np.mean([row["wg_minus_wrist_trans_mm"] < 0.0 for row in joint_ok])) if joint_ok else float("nan"),
            "rotation": float(np.mean([row["wg_minus_wrist_rot_deg"] < 0.0 for row in joint_ok])) if joint_ok else float("nan"),
        },
        "chain_angle_diagnostics": _metric_stats(wrist_ok, angle_keys),
        "initial_independent_gripper": {},
        "independent_gripper": {},
        "support": _metric_stats(
            wrist_ok,
            [
                "gt_wrist_area_ds",
                "gt_gripper_area_ds",
                "pred_wrist_area_ds",
                "pred_l_gripper_area_ds",
                "pred_r_gripper_area_ds",
            ],
        ),
    }
    for name in GRIPPER_NAMES:
        initial_part_rows = initial_independent[name]
        result["initial_independent_gripper"][name] = {
            "success": len(initial_part_rows),
            "success_rate": len(initial_part_rows) / max(1, len(selected)),
            "metrics": _metric_stats(
                initial_part_rows,
                [
                    f"initial_independent_{name}_trans_err_mm",
                    f"initial_independent_{name}_rot_err_deg",
                    f"initial_independent_{name}_add_mm",
                ],
            ),
        }
        part_rows = independent[name]
        result["independent_gripper"][name] = {
            "success": len(part_rows),
            "success_rate": len(part_rows) / max(1, len(selected)),
            "metrics": _metric_stats(
                part_rows,
                [
                    f"independent_{name}_trans_err_mm",
                    f"independent_{name}_rot_err_deg",
                    f"independent_{name}_add_mm",
                    f"{name}_visibility_input_inliers",
                    f"{name}_visibility_kept",
                    f"{name}_visibility_keep_fraction",
                    f"{name}_visibility_refit_inliers",
                    f"{name}_visibility_refit_inlier_fraction",
                ],
            ),
        }
        result["visibility_refinement_paired"][f"{name}_trans_err_mm"] = _paired_refinement_stat(
            part_rows,
            f"initial_independent_{name}_trans_err_mm",
            f"independent_{name}_trans_err_mm",
        )
        result["visibility_refinement_paired"][f"{name}_rot_err_deg"] = _paired_refinement_stat(
            part_rows,
            f"initial_independent_{name}_rot_err_deg",
            f"independent_{name}_rot_err_deg",
        )
        result["visibility_refinement_paired"][f"{name}_add_mm"] = _paired_refinement_stat(
            part_rows,
            f"initial_independent_{name}_add_mm",
            f"independent_{name}_add_mm",
        )
    return result


def _write_markdown(path, summary):
    lines = ["# SurfEmb wrist + gripper RARP ablation", ""]
    for name, result in summary["models"].items():
        lines += [f"## {name}", ""]
        lines.append(
            f"- Wrist PnP: {result['wrist_success']}/{result['total']}; full W+G chain: "
            f"{result['full_wrist_gripper_success']}/{result['total']}."
        )
        refinement = result["visibility_refinement_paired"]
        lines.append(
            "- Paired initial -> visibility-refined wrist median: "
            f"translation {refinement['wrist_trans_err_mm']['initial']['median']:.3f} -> "
            f"{refinement['wrist_trans_err_mm']['refined']['median']:.3f} mm; rotation "
            f"{refinement['wrist_rot_err_deg']['initial']['median']:.3f} -> "
            f"{refinement['wrist_rot_err_deg']['refined']['median']:.3f} deg."
        )
        paired = result["paired_wrist_only"]
        joint = result["wrist_gripper_joint"]
        lines.append(
            "- Paired wrist-only -> W+G joint median: "
            f"translation {paired['wrist_only_trans_err_mm']['median']:.3f} -> "
            f"{joint['wg_joint_wrist_trans_err_mm']['median']:.3f} mm; rotation "
            f"{paired['wrist_only_rot_err_deg']['median']:.3f} -> "
            f"{joint['wg_joint_wrist_rot_err_deg']['median']:.3f} deg."
        )
        for part in GRIPPER_NAMES:
            initial_item = result["initial_independent_gripper"][part]
            item = result["independent_gripper"][part]
            metrics = item["metrics"]
            paired_trans = refinement[f"{part}_trans_err_mm"]
            paired_rot = refinement[f"{part}_rot_err_deg"]
            paired_add = refinement[f"{part}_add_mm"]
            lines.append(
                f"- {part} initial -> visibility-refined PnP: "
                f"{initial_item['success']} -> {item['success']}/{result['total']}; median t "
                f"{paired_trans['initial']['median']:.3f} -> {paired_trans['refined']['median']:.3f} mm "
                f"(win {paired_trans['refined_win_rate']:.1%}), median r "
                f"{paired_rot['initial']['median']:.3f} -> {paired_rot['refined']['median']:.3f} deg "
                f"(win {paired_rot['refined_win_rate']:.1%}), median ADD "
                f"{paired_add['initial']['median']:.3f} -> {paired_add['refined']['median']:.3f} mm."
            )
        angles = result["chain_angle_diagnostics"]
        lines.append(
            "- Predicted-wrist / GT-wrist constrained angle median: "
            f"left {angles['predwrist_theta_l_err_deg']['median']:.3f} / "
            f"{angles['oraclewrist_theta_l_err_deg']['median']:.3f} deg; right "
            f"{angles['predwrist_theta_r_err_deg']['median']:.3f} / "
            f"{angles['oraclewrist_theta_r_err_deg']['median']:.3f} deg."
        )
        lines.append("")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def worker_main(rank, chunks, specs, args):
    cv2.setNumThreads(0)
    torch.set_num_threads(max(1, int(args.cpu_threads_per_worker)))
    device = torch.device(args.devices[rank])
    if device.type == "cuda":
        torch.cuda.set_device(device)
    renderer = InstrumentOpenGLDepthRenderer(
        int(args.crop_size),
        int(args.crop_size),
        device_idx=0 if device.index is None else int(device.index),
    )
    dataset = build_dataset(args)
    indices = chunks[rank]
    rows = []
    batch_size = max(1, int(args.inference_batch_size))
    for model_index, spec in enumerate(specs):
        model, checkpoint_iter, _ = load_model(spec, device)
        surfaces = load_part_surfaces(args.surface_root, args.surface_keys_per_part, args.surface_seed)
        encode_surface_keys(model, surfaces, device, args.mask_keys_per_part)
        print(f"[worker {rank}] {spec['name']} iter={checkpoint_iter} n={len(indices)}", flush=True)
        for start in range(0, len(indices), batch_size):
            samples = []
            for dataset_idx in indices[start : start + batch_size]:
                _, target = dataset[int(dataset_idx)]
                image, K_crop, _, crop_matrix = make_surfemb_crop(target, args)
                samples.append(
                    {
                        "dataset_idx": int(dataset_idx),
                        "target": target,
                        "image": image,
                        "K_crop": K_crop,
                        "part_crop": _crop_part_mask(target, crop_matrix, args.crop_size),
                        "gt": target_pose(target),
                    }
                )
            images = torch.stack([sample["image"] for sample in samples]).to(device, non_blocking=True)
            intrinsics = torch.from_numpy(np.stack([sample["K_crop"] for sample in samples])).to(
                device, non_blocking=True
            )
            forward_start = time.perf_counter()
            with torch.inference_mode(), torch.amp.autocast(
                device_type="cuda", enabled=device.type == "cuda" and bool(args.amp), dtype=torch.bfloat16
            ):
                outputs = model(images, intrinsics)
            forward_per_sample = (time.perf_counter() - forward_start) / max(1, len(samples))
            for sample_index, sample in enumerate(samples):
                target = sample["target"]
                row = {
                    "model": spec["name"],
                    "checkpoint_iter": checkpoint_iter,
                    "dataset_idx": sample["dataset_idx"],
                    "video": str(target["video_name"]),
                    "frame_id": str(target["frame_id"]),
                    "instance_id": int(target["instance_id"].item()),
                    "forward_sec": float(forward_per_sample),
                }
                solve_start = time.perf_counter()
                try:
                    row.update(
                        evaluate_output(
                            model,
                            surfaces,
                            {
                                "inst_mask_logits": outputs["inst_mask_logits"][sample_index],
                                "surfemb_queries": outputs["surfemb_queries"][sample_index],
                            },
                            sample["K_crop"],
                            sample["gt"],
                            sample["part_crop"],
                            renderer,
                            args,
                        )
                    )
                except Exception as exc:
                    row["status"] = f"{type(exc).__name__}: {exc}"
                    if args.fail_fast:
                        raise
                row["solve_sec"] = float(time.perf_counter() - solve_start)
                rows.append(row)
            done = min(start + len(samples), len(indices))
            if start == 0 or done == len(indices) or done % int(args.print_freq) < len(samples):
                print(f"[worker {rank}] {spec['name']} {done}/{len(indices)}", flush=True)
        del model, surfaces, outputs, images, intrinsics
        torch.cuda.empty_cache()
    renderer.release()
    output = Path(args.output_dir) / "workers" / f"worker_{rank:02d}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2, allow_nan=True), encoding="utf-8")


def main(args):
    args.output_dir = str(Path(args.output_dir).resolve())
    args.dataset_cache_dir = str(Path(args.dataset_cache_dir).resolve())
    specs = parse_model_specs(args.model or list(DEFAULT_MODELS))
    dataset = build_dataset(args)
    total = len(dataset) if args.max_samples <= 0 else min(len(dataset), int(args.max_samples))
    indices = list(range(total))
    chunks = [indices[offset:: len(args.devices)] for offset in range(len(args.devices))]
    chunks = [chunk for chunk in chunks if chunk]
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print(f"dataset={dataset} selected={total} models={[spec['name'] for spec in specs]}", flush=True)
    if len(chunks) == 1:
        worker_main(0, chunks, specs, args)
    else:
        mp.spawn(worker_main, args=(chunks, specs, args), nprocs=len(chunks), join=True)
    rows = []
    for rank in range(len(chunks)):
        rows.extend(json.loads((Path(args.output_dir) / "workers" / f"worker_{rank:02d}.json").read_text()))
    rows.sort(key=lambda row: (row["dataset_idx"], row["model"]))
    _write_csv(Path(args.output_dir) / "per_instance.csv", rows)
    summary = {
        "models": {spec["name"]: _summarize_model(rows, spec) for spec in specs},
        "config": {
            "dataset": "needleGrasping/test",
            "samples": total,
            "devices": args.devices,
            "fit": "independent top-K RANSAC per part, followed by wrist+l/r-gripper kinematic-chain fit",
            "surface_root": str(Path(args.surface_root).resolve()),
            "surface_keys_per_part": args.surface_keys_per_part,
            "down_sample_scale": args.down_sample_scale,
            "visibility_depth_tolerance_m": args.visibility_depth_tolerance,
            "visibility_min_correspondences": args.visibility_min_correspondences,
        },
    }
    (Path(args.output_dir) / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True))
    _write_markdown(Path(args.output_dir) / "summary.md", summary)
    print(json.dumps(summary["models"], indent=2, allow_nan=True), flush=True)
    print(f"summary={Path(args.output_dir) / 'summary.md'}", flush=True)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", default=None)
    parser.add_argument(
        "--output_dir", default=str(ROOT / "logs" / "surfemb_wrist_gripper_ablation_rarp")
    )
    parser.add_argument("--devices", nargs="+", default=["cuda:0"])
    parser.add_argument("--needle_dataset_root", default="/mnt/nas/share/shuojue/data/needleGrasping_videos")
    parser.add_argument("--needle_pose_root", default="/mnt/nas/share/shuojue/data/needleGrasping_results")
    parser.add_argument(
        "--dataset_cache_dir",
        default=str(ROOT / "logs/robopepp_rarp_lnd_refinemem_eval_keypoint_trimesh/dataset_cache"),
    )
    parser.add_argument("--surface_root", default=str(DEFAULT_SURFACE_ROOT))
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--surfemb_crop_scale", type=float, default=1.2)
    parser.add_argument("--bbox_padding_frac", type=float, default=0.12)
    parser.add_argument("--train_ratio", type=float, default=0.95)
    parser.add_argument("--min_dice_shaft", type=float, default=0.8)
    parser.add_argument("--min_dice_wrist", type=float, default=0.6)
    parser.add_argument("--min_dice_gripper", type=float, default=0.6)
    parser.add_argument("--canonicalize_pose_symmetry", type=int, choices=(0, 1), default=1)
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
    parser.add_argument("--visibility_depth_tolerance", type=float, default=0.0008)
    parser.add_argument("--visibility_min_correspondences", type=int, default=8)
    parser.add_argument("--pose_batch_size", type=int, default=64)
    parser.add_argument("--coarse_steps", type=int, default=73)
    parser.add_argument("--fine_steps", type=int, default=17)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--amp", type=int, choices=(0, 1), default=1)
    parser.add_argument("--inference_batch_size", type=int, default=32)
    parser.add_argument("--cpu_threads_per_worker", type=int, default=2)
    parser.add_argument("--print_freq", type=int, default=32)
    parser.add_argument("--fail_fast", type=int, choices=(0, 1), default=0)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
