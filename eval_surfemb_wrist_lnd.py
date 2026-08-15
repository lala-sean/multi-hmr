#!/usr/bin/env python3
import argparse
import csv
import gc
import json
import math
import os
import sys
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from scipy.spatial.transform import Rotation

ORIGINAL_SURFEMB_SNAPSHOT = Path(__file__).resolve().parents[3] / "surfemb_original_debug_20260810"
if not ORIGINAL_SURFEMB_SNAPSHOT.is_dir():
    raise FileNotFoundError(f"Missing read-only original SurfEmb snapshot: {ORIGINAL_SURFEMB_SNAPSHOT}")
sys.path.insert(0, str(ORIGINAL_SURFEMB_SNAPSHOT))
from surfemb import utils as original_surfemb_utils

import eval_surfemb_articulated_rarp as surf_eval
from surfemb_articulated_pose import (
    PART_NAMES,
    build_part_probability_inputs,
    encode_surface_keys,
    estimate_part_pose_from_context,
    estimate_part_pose_coarse_fine_ransac,
    estimate_part_pose_spatial_topk_ransac,
    estimate_part_pose_topk_roi_ransac,
    estimate_part_pose_topk_ransac,
    load_part_surfaces,
    prepare_original_surfemb_score_context,
    prepare_part_score_context,
    refine_original_surfemb_pose,
)
from instrument_opengl_renderer import InstrumentOpenGLDepthRenderer


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "logs" / "surfemb_wrist_lnd_test"
DEFAULT_HCCE_CSV = ROOT / "logs" / "lnd_test_eval_sam3_wristonly_iter44000" / "per_frame.csv"
DEFAULT_ROBO_CSV = ROOT / "logs" / "surgripe_lnd_test_full_robopepp_only_rotate_last" / "per_instance.csv"

_lnd_module = surf_eval.load_local_module(
    "surfemb_wrist_eval_lnd_dataset",
    ROOT / "datasets" / "surgripe_lnd_instrument.py",
)
RoboPEPPSurgripeLNDInstrument = _lnd_module.RoboPEPPSurgripeLNDInstrument


def matrix_to_quat_wxyz(matrix):
    xyzw = Rotation.from_matrix(np.asarray(matrix, dtype=np.float64)).as_quat()
    quat = np.asarray([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64)
    return -quat if quat[0] < 0.0 else quat


def build_dataset(args):
    return RoboPEPPSurgripeLNDInstrument(
        root=args.lnd_root,
        split="TEST",
        training=False,
        crop_size=int(args.crop_size),
        memory_path=None,
        use_memory_pose=False,
        canonicalize_pose_symmetry=True,
        canonical_eps=float(args.canonical_eps),
        color_jitter=False,
        rgb_augmentation=False,
        occlusion_augmentation=False,
        bbox_padding_frac=float(args.bbox_padding_frac),
        bbox_jitter=False,
        bbox_shift=False,
        subsample=1,
    )


@lru_cache(maxsize=None)
def load_detector_bbox_map(detection_folder):
    folder = Path(detection_folder)
    bboxes = np.load(folder / "bboxes.npy")
    scene_ids = np.load(folder / "scene_ids.npy")
    view_ids = np.load(folder / "view_ids.npy")
    return {
        (int(scene_id), int(view_id)): np.asarray(bbox).round().astype(np.float32)
        for bbox, scene_id, view_id in zip(bboxes, scene_ids, view_ids)
    }


def make_eval_crop(target, args):
    rgb = target["orig_rgb"].detach().cpu().numpy().astype(np.uint8)
    inst_mask = target["inst_mask_orig"].detach().cpu().numpy().astype(bool)
    if args.crop_mask_source == "detector":
        frame_id = int(target["frame_id"])
        detector_bbox = load_detector_bbox_map(args.detection_folder).get((2, frame_id - 1))
        if detector_bbox is None:
            raise KeyError(f"Missing scene 2/view {frame_id - 1} detector bbox")
        left, top, right, bottom = detector_bbox
        cx, cy = (left + right) / 2.0, (top + bottom) / 2.0
        size = int(args.crop_size) / max(bottom - top, right - left) / float(args.surfemb_crop_scale)
        M = np.asarray(
            [
                [size, 0.0, -cx * size + int(args.crop_size) / 2.0],
                [0.0, size, -cy * size + int(args.crop_size) / 2.0],
            ],
            dtype=np.float32,
        )
    elif args.crop_mask_source == "wrist":
        part_mask = target["part_mask_orig"].detach().cpu().numpy()
        crop_mask = inst_mask & (part_mask == 2)
    else:
        crop_mask = inst_mask
    if args.crop_mask_source != "detector":
        if not np.any(crop_mask):
            raise RuntimeError(f"Empty {args.crop_mask_source} crop mask")
        M = surf_eval._surf_aug.random_rotated_mask_crop_matrix(
            crop_mask,
            int(args.crop_size),
            crop_scale=float(args.surfemb_crop_scale),
            max_angle=0.0,
            offset_scale=0.0,
            ensure_full_mask=True,
        )

    K_orig = target["K_orig"].detach().cpu().numpy().astype(np.float32)
    crop = cv2.warpAffine(
        rgb,
        M,
        (int(args.crop_size), int(args.crop_size)),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    K_crop = surf_eval._surf_aug.matrix3_from_affine(M) @ K_orig
    return surf_eval._surf_aug.imagenet_tensor(crop), K_crop.astype(np.float32), crop, M


def probability_input_from_mask(mask, image_hw):
    h, w = image_hw
    probability = torch.where(
        mask.reshape(-1),
        torch.full((h * w,), 0.995, device=mask.device),
        torch.full((h * w,), 1e-7, device=mask.device),
    )
    log_probability = probability.log().reshape(h, w)
    neg_log_probability = torch.log1p(-probability).reshape(h, w)
    return {
        "prob": probability,
        "mask_log_prob": F.max_pool2d(log_probability[None, None], 3, 1, 1)[0, 0].reshape(-1),
        "neg_mask_log_prob": F.max_pool2d(neg_log_probability[None, None], 3, 1, 1)[0, 0].reshape(-1),
    }


def probability_input_from_binary_logits(mask_logits, down_sample_scale):
    scale = int(down_sample_scale)
    logits = mask_logits.float()[None, None]
    log_probability = F.max_pool2d(F.logsigmoid(logits), scale)[0, 0]
    probability = log_probability.exp()
    neg_log_probability = torch.log1p(-probability.clamp(max=1.0 - 1e-7))
    return {
        "prob": probability.reshape(-1),
        "mask_log_prob": F.max_pool2d(log_probability[None, None], 3, 1, 1)[0, 0].reshape(-1),
        "neg_mask_log_prob": F.max_pool2d(neg_log_probability[None, None], 3, 1, 1)[0, 0].reshape(-1),
    }


@torch.inference_mode()
def predict_wrist(model, surfaces, image, K_crop, gt_wrist_crop, device, args, seed, renderer=None):
    x = image[None].to(device=device, non_blocking=True)
    K_tensor = torch.from_numpy(K_crop)[None].to(device=device, non_blocking=True)
    with torch.amp.autocast(
        device_type=device.type,
        enabled=device.type == "cuda" and bool(args.amp),
        dtype=torch.bfloat16,
    ):
        if bool(args.rotation_ensemble):
            if not hasattr(model, "cnn"):
                raise TypeError("rotation ensemble requires the original SurfEmb ResNet CNN")
            rotated = original_surfemb_utils.rotate_batch(x[0])
            dense = model.cnn(rotated)
            dense = original_surfemb_utils.rotate_batch_back(dense).mean(dim=0, keepdim=True)
            output = {
                "inst_mask_logits": dense[:, 0],
                "surfemb_queries": dense[:, 1:],
                "dense_logits": dense,
            }
        else:
            output = model(x, K_tensor)

    if args.pose_estimator == "original_surfemb":
        if args.wrist_roi_source == "predicted":
            if args.predicted_wrist_mask_mode != "binary_head":
                raise ValueError("original_surfemb predicted ROI requires the wrist-only binary head")
            original_mask_logits = output["inst_mask_logits"][0].float()
        else:
            if gt_wrist_crop is None:
                raise ValueError("original_surfemb GT wrist ROI requires gt_wrist_crop")
            gt_wrist_full = torch.as_tensor(gt_wrist_crop, device=device, dtype=torch.bool)
            foreground_logit = math.log(0.995 / 0.005)
            background_logit = math.log(1e-7 / (1.0 - 1e-7))
            original_mask_logits = torch.where(
                gt_wrist_full,
                torch.full_like(gt_wrist_full, foreground_logit, dtype=torch.float32),
                torch.full_like(gt_wrist_full, background_logit, dtype=torch.float32),
            )
        context, K_ds, image_hw, object_prob = prepare_original_surfemb_score_context(
            original_mask_logits,
            output["surfemb_queries"][0].float(),
            surfaces["wrist"],
            K_crop,
            down_sample_scale=int(args.down_sample_scale),
            similarity=args.correspondence_similarity,
            temperature=float(args.correspondence_temperature),
        )
        query_flat = part_inputs = None
    else:
        query_flat, part_inputs, K_ds, image_hw, object_prob = build_part_probability_inputs(
            output["inst_mask_logits"][0].float(),
            output["surfemb_queries"][0].float(),
            surfaces,
            K_crop,
            down_sample_scale=int(args.down_sample_scale),
            similarity=args.correspondence_similarity,
            temperature=float(args.correspondence_temperature),
        )
    gt_wrist_roi = None
    if args.pose_estimator == "original_surfemb":
        context_input = None
    elif args.wrist_roi_source == "gt_wrist":
        if gt_wrist_crop is None:
            raise ValueError("gt_wrist_crop is required for wrist_roi_source=gt_wrist")
        gt_wrist = torch.as_tensor(gt_wrist_crop, device=device, dtype=torch.float32)
        gt_wrist_roi = F.max_pool2d(
            gt_wrist[None, None],
            int(args.down_sample_scale),
            int(args.down_sample_scale),
        )[0, 0] > 0
        if tuple(gt_wrist_roi.shape) != tuple(image_hw):
            raise ValueError(f"GT wrist ROI shape {tuple(gt_wrist_roi.shape)} != output shape {tuple(image_hw)}")
        context_input = probability_input_from_mask(gt_wrist_roi, image_hw)
    elif args.predicted_wrist_mask_mode == "binary_head":
        context_input = probability_input_from_binary_logits(
            output["inst_mask_logits"][0],
            int(args.down_sample_scale),
        )
        if context_input["prob"].numel() != int(image_hw[0] * image_hw[1]):
            raise ValueError("Binary wrist probability map does not match query output resolution")
    else:
        context_input = part_inputs["wrist"]
    if args.pose_estimator != "original_surfemb":
        context = prepare_part_score_context(
            query_flat,
            context_input,
            surfaces["wrist"],
            image_hw,
            similarity=args.correspondence_similarity,
            temperature=float(args.correspondence_temperature),
        )
    estimator_diagnostics = {}
    if args.pose_estimator in (
        "topk_ransac",
        "topk_roi_ransac",
        "spatial_topk_ransac",
        "coarse_fine_ransac",
    ):
        if args.wrist_roi_source == "gt_wrist":
            matching_roi = gt_wrist_roi.reshape(-1)
        elif args.predicted_wrist_mask_mode == "binary_head":
            matching_roi = context_input["prob"] >= float(args.topk_min_object_probability)
        else:
            part_probability = torch.stack([part_inputs[name]["prob"] for name in PART_NAMES], dim=1)
            wrist_index = PART_NAMES.index("wrist")
            matching_roi = (
                (part_probability.argmax(dim=1) == wrist_index)
                & (object_prob >= float(args.topk_min_object_probability))
            )
        if args.pose_estimator == "coarse_fine_ransac":
            estimator = estimate_part_pose_coarse_fine_ransac
        elif args.pose_estimator == "spatial_topk_ransac":
            estimator = estimate_part_pose_spatial_topk_ransac
        elif args.pose_estimator == "topk_roi_ransac":
            estimator = estimate_part_pose_topk_roi_ransac
        else:
            estimator = estimate_part_pose_topk_ransac
        estimator_kwargs = {
            "pixel_mask": matching_roi,
            "max_correspondences": int(args.topk_max_correspondences),
            "min_correspondences": int(args.topk_min_correspondences),
            "min_part_probability": float(args.topk_min_part_probability),
            "margin_power": float(args.topk_margin_power),
            "ransac_iterations": int(args.topk_ransac_iterations),
            "ransac_reprojection_error": float(args.topk_ransac_reprojection_error),
            "ransac_confidence": float(args.topk_ransac_confidence),
            "min_inliers": int(args.topk_min_inliers),
            "min_inlier_fraction": float(args.topk_min_inlier_fraction),
        }
        if args.pose_estimator == "coarse_fine_ransac":
            estimator_kwargs = {
                "pixel_mask": matching_roi,
                "coarse_max_correspondences": int(args.coarse_max_correspondences),
                "coarse_keys_per_pixel": int(args.coarse_keys_per_pixel),
                "coarse_min_inlier_fraction": float(args.coarse_min_inlier_fraction),
                "fine_neighbors_per_inlier": int(args.fine_neighbors_per_inlier),
                "fine_max_correspondences": int(args.fine_max_correspondences),
                "min_correspondences": int(args.topk_min_correspondences),
                "min_part_probability": float(args.topk_min_part_probability),
                "margin_power": float(args.topk_margin_power),
                "ransac_iterations": int(args.topk_ransac_iterations),
                "coarse_ransac_reprojection_error": float(
                    args.topk_ransac_reprojection_error
                ),
                "fine_ransac_reprojection_error": float(
                    args.fine_ransac_reprojection_error
                ),
                "ransac_confidence": float(args.topk_ransac_confidence),
                "min_inliers": int(args.topk_min_inliers),
                "min_inlier_fraction": float(args.topk_min_inlier_fraction),
            }
        elif args.pose_estimator == "spatial_topk_ransac":
            estimator_kwargs.update(
                grid_size=int(args.spatial_grid_size),
                max_per_grid_cell=int(args.spatial_max_per_grid_cell),
                fps_2d_weight=float(args.spatial_fps_2d_weight),
                fps_3d_weight=float(args.spatial_fps_3d_weight),
                fps_confidence_power=float(args.spatial_fps_confidence_power),
                min_inlier_hull_fraction=float(args.spatial_min_inlier_hull_fraction),
                min_inlier_axis_span=float(args.spatial_min_inlier_axis_span),
                min_inlier_3d_span=float(args.spatial_min_inlier_3d_span),
                max_reprojection_median=float(args.spatial_max_reprojection_median),
            )
        elif args.pose_estimator == "topk_roi_ransac":
            estimator_kwargs["roi_refit_max_iterations"] = int(args.roi_refit_max_iterations)
        best, estimator_diagnostics = estimator(
            context,
            surfaces["wrist"],
            K_ds,
            image_hw,
            **estimator_kwargs,
        )
        estimator_diagnostics["matching_wrist_area_ds"] = int(matching_roi.sum().item())
        if best is None:
            raise RuntimeError(f"No valid wrist {args.pose_estimator} pose: {estimator_diagnostics}")
        hypothesis_count = 1
    else:
        hypotheses = estimate_part_pose_from_context(
            context,
            surfaces["wrist"],
            K_ds,
            image_hw,
            max_poses=int(args.max_poses),
            max_pose_evaluations=int(args.max_pose_evaluations),
            pose_batch_size=int(args.pose_batch_size),
            top_k=int(args.top_k),
            alpha=float(args.corr_alpha),
            dist_2d_min=float(args.dist_2d_min),
            seed=int(seed),
        )
        if not hypotheses:
            raise RuntimeError("No valid wrist AP3P hypothesis")
        best = hypotheses[0]
        hypothesis_count = len(hypotheses)
    run_bfgs_refine = (
        (args.pose_estimator == "original_surfemb" and bool(args.original_refine))
        or (args.pose_estimator == "topk_ransac" and bool(args.topk_bfgs_refine))
    )
    if run_bfgs_refine:
        if renderer is None:
            raise RuntimeError("SurfEmb BFGS refinement requires a triangle renderer")
        initial_raw = {
            "rot": matrix_to_quat_wxyz(best.transform[:3, :3]),
            "trans": best.transform[:3, 3].astype(np.float64),
            "alpha": 0.0,
            "theta_l": 0.0,
            "theta_r": 0.0,
        }
        initial_canonical = surf_eval.canonicalize_prediction(initial_raw, args.canonical_eps)
        for index, name in enumerate(("qw", "qx", "qy", "qz")):
            estimator_diagnostics[f"pre_bfgs_{name}"] = float(initial_canonical["rot"][index])
        for index, name in enumerate(("tx_m", "ty_m", "tz_m")):
            estimator_diagnostics[f"pre_bfgs_{name}"] = float(initial_canonical["trans"][index])
        initial_score = float(best.score)
        with torch.inference_mode(False), torch.enable_grad():
            best, refine_diagnostics = refine_original_surfemb_pose(
                best,
                output["inst_mask_logits"][0].detach().clone(),
                output["surfemb_queries"][0].detach().clone(),
                model,
                surfaces["wrist"],
                K_crop,
                renderer,
                device,
                denominator_keys=int(args.original_refine_denominator_keys),
                min_visible_pixels=int(args.original_refine_min_visible_pixels),
                max_iterations=(
                    None
                    if int(
                        args.topk_bfgs_max_iterations
                        if args.pose_estimator == "topk_ransac"
                        else args.original_refine_max_iterations
                    )
                    <= 0
                    else int(
                        args.topk_bfgs_max_iterations
                        if args.pose_estimator == "topk_ransac"
                        else args.original_refine_max_iterations
                    )
                ),
                optimization_float64=bool(args.bfgs_refine_float64),
                rotation_units_per_radian=float(args.bfgs_rotation_units_per_radian),
            )
        estimator_diagnostics["bfgs_initial_score"] = initial_score
        estimator_diagnostics.update(
            {
                key.replace("original_refine", "bfgs_refine"): value
                for key, value in refine_diagnostics.items()
            }
        )
    raw = {
        "rot": matrix_to_quat_wxyz(best.transform[:3, :3]),
        "trans": best.transform[:3, 3].astype(np.float64),
        "alpha": 0.0,
        "theta_l": 0.0,
        "theta_r": 0.0,
        "chain_cost": float(-best.score),
        "chain_nfev": 0,
    }
    canonical = surf_eval.canonicalize_prediction(raw, args.canonical_eps)
    diagnostics = {
        "pose_estimator": args.pose_estimator,
        "wrist_roi_source": args.wrist_roi_source,
        "predicted_wrist_mask_mode": args.predicted_wrist_mask_mode,
        "crop_mask_source": args.crop_mask_source,
        "pred_mask_area_ds": int((object_prob > 0.5).sum().item()),
        "wrist_hypotheses": int(hypothesis_count),
        "wrist_score": float(best.score),
        "wrist_mask_score": float(best.mask_score),
        "wrist_coord_score": float(best.coord_score),
    }
    diagnostics.update(estimator_diagnostics)
    return raw, canonical, diagnostics


def add_pose_fields(row, prefix, pose, gt):
    row[f"{prefix}_trans_err_mm"] = float(
        np.linalg.norm(np.asarray(pose["trans"]) - np.asarray(gt["trans"])) * 1000.0
    )
    row[f"{prefix}_rot_err_deg"] = surf_eval.rotation_error_deg(pose["rot"], gt["rot"])
    quat = np.asarray(pose["rot"], dtype=np.float64)
    trans = np.asarray(pose["trans"], dtype=np.float64)
    for i, name in enumerate(("qw", "qx", "qy", "qz")):
        row[f"{prefix}_{name}"] = float(quat[i])
    for i, name in enumerate(("tx_m", "ty_m", "tz_m")):
        row[f"{prefix}_{name}"] = float(trans[i])


def worker_main(rank, chunks, specs, args):
    cv2.setNumThreads(0)
    torch.set_num_threads(max(1, int(args.cpu_threads_per_worker)))
    device = torch.device(args.devices[rank])
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dataset = build_dataset(args)
    rows = []
    indices = chunks[rank]
    renderer = None
    run_bfgs_refine = (
        (args.pose_estimator == "original_surfemb" and bool(args.original_refine))
        or (args.pose_estimator == "topk_ransac" and bool(args.topk_bfgs_refine))
    )
    if run_bfgs_refine:
        renderer = InstrumentOpenGLDepthRenderer(
            int(args.crop_size),
            int(args.crop_size),
            device_idx=(0 if device.index is None else int(device.index)),
        )

    for model_index, spec in enumerate(specs):
        model, checkpoint_iter, _ = surf_eval.load_model(spec, device)
        surfaces = load_part_surfaces(
            args.surface_root,
            keys_per_part=int(args.surface_keys_per_part),
            seed=int(args.surface_seed),
        )
        encode_surface_keys(model, surfaces, device, mask_keys_per_part=int(args.mask_keys_per_part))
        for local_index, dataset_index in enumerate(indices):
            _, target = dataset[int(dataset_index)]
            image, K_crop, _, M_crop = make_eval_crop(target, args)
            gt_wrist_crop = cv2.warpAffine(
                (target["part_mask_orig"].detach().cpu().numpy() == 2).astype(np.uint8),
                M_crop,
                (int(args.crop_size), int(args.crop_size)),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            ).astype(bool)
            gt = surf_eval.target_pose(target)
            row = {
                "model": spec["name"],
                "checkpoint_iter": int(checkpoint_iter),
                "dataset_idx": int(dataset_index),
                "frame_id": int(target["frame_id"]),
                "image_path": str(target["img_path"]),
                "status": "ok",
            }
            try:
                raw, canonical, diagnostics = predict_wrist(
                    model,
                    surfaces,
                    image,
                    K_crop,
                    gt_wrist_crop,
                    device,
                    args,
                    seed=int(args.pose_seed) + int(dataset_index) * 1009 + model_index * 10000019,
                    renderer=renderer,
                )
                add_pose_fields(row, "raw", raw, gt)
                add_pose_fields(row, "canonical", canonical, gt)
                row["prediction_sym_flipped"] = bool(canonical.get("pose_sym_flipped", False))
                row.update(diagnostics)
                if "pre_bfgs_qw" in diagnostics:
                    pre_bfgs = {
                        "rot": np.asarray(
                            [diagnostics[f"pre_bfgs_{name}"] for name in ("qw", "qx", "qy", "qz")],
                            dtype=np.float64,
                        ),
                        "trans": np.asarray(
                            [diagnostics[f"pre_bfgs_{name}"] for name in ("tx_m", "ty_m", "tz_m")],
                            dtype=np.float64,
                        ),
                    }
                    add_pose_fields(row, "pre_bfgs", pre_bfgs, gt)
            except Exception as exc:
                row["status"] = f"{type(exc).__name__}: {exc}"
                if int(args.fail_fast):
                    raise
            rows.append(row)
            if local_index == 0 or (local_index + 1) % int(args.print_freq) == 0 or local_index + 1 == len(indices):
                print(
                    f"[worker {rank}] {spec['name']} {local_index + 1}/{len(indices)} "
                    f"frame={row['frame_id']} status={row['status']}",
                    flush=True,
                )
        del model, surfaces
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if renderer is not None:
        renderer.release()

    output = Path(args.output_dir) / "workers" / f"worker_{rank:02d}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2, allow_nan=True), encoding="utf-8")


def stat(values):
    values = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=np.float64)
    if len(values) == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
            "max": float("nan"),
            "rmse": float("nan"),
        }
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
        "rmse": float(np.sqrt(np.mean(values * values))),
    }


def write_csv(path, rows):
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv_index(path):
    rows = list(csv.DictReader(Path(path).open(encoding="utf-8")))
    return {int(row["frame_id"]): row for row in rows}


def float_or_nan(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def combined_rows(rows, specs, args):
    by_key = {(row["model"], int(row["frame_id"])): row for row in rows}
    hcce = read_csv_index(args.historical_hcce_csv)
    robo = read_csv_index(args.historical_robopepp_csv)
    excluded = set(int(value) for value in args.exclude_frame_ids)
    frame_ids = sorted((set(hcce) & set(robo)) - excluded)
    expected = [frame_id for frame_id in range(1, 374) if frame_id not in excluded]
    if frame_ids != expected:
        raise RuntimeError(f"Historical LND frame IDs differ from expected set: n={len(frame_ids)}")
    out = []
    for frame_id in frame_ids:
        row = {
            "frame_id": frame_id,
            "hcce_wrist_fit_trans_err_mm": float(hcce[frame_id]["hcce_fit_trans_err_m"]) * 1000.0,
            "hcce_wrist_fit_rot_err_deg": float(hcce[frame_id]["hcce_fit_rot_err_deg"]),
            "hcce_direct_trans_err_mm": float(hcce[frame_id]["hcce_direct_trans_err_m"]) * 1000.0,
            "hcce_direct_rot_err_deg": float(hcce[frame_id]["hcce_direct_rot_err_deg"]),
            "robopepp_direct_trans_err_mm": float(robo[frame_id]["direct_trans_err_mm"]),
            "robopepp_direct_rot_err_deg": float(robo[frame_id]["direct_rot_err_deg"]),
        }
        for spec in specs:
            source = by_key.get((spec["name"], frame_id))
            prefix = spec["name"]
            row[f"{prefix}_status"] = "missing" if source is None else source["status"]
            for mode in ("raw", "canonical"):
                for metric in ("trans_err_mm", "rot_err_deg"):
                    row[f"{prefix}_{mode}_{metric}"] = (
                        float("nan") if source is None else float_or_nan(source.get(f"{mode}_{metric}"))
                    )
            row[f"{prefix}_prediction_sym_flipped"] = (
                False if source is None else bool(source.get("prediction_sym_flipped", False))
            )
        out.append(row)
    return out


def summarize(rows, specs, combined, args):
    if args.pose_estimator == "topk_ransac":
        pose_protocol = (
            "wrist-only high-confidence top-K dense correspondences + multi-point EPNP RANSAC + LM"
            + (" + triangle-visible dense BFGS refinement" if bool(args.topk_bfgs_refine) else "")
            + "; "
            f"{args.wrist_roi_source} wrist ROI; no shaft/gripper fitting"
        )
    elif args.pose_estimator == "spatial_topk_ransac":
        pose_protocol = (
            "wrist-only confidence-ranked correspondences + 2D grid stratification + joint 2D/3D "
            "weighted FPS + EPNP RANSAC + LM + inlier coverage checks; "
            f"{args.wrist_roi_source} wrist ROI; no fallback; no shaft/gripper fitting"
        )
    elif args.pose_estimator == "coarse_fine_ransac":
        pose_protocol = (
            f"wrist-only {args.coarse_max_correspondences}-pair coarse Top-K RANSAC + LM; "
            f"{args.fine_neighbors_per_inlier} nearest 2D/3D neighbors per coarse inlier + "
            "local SurfEmb rematching + second EPNP RANSAC + LM; "
            f"{args.wrist_roi_source} wrist ROI; no fallback; no shaft/gripper fitting"
        )
    elif args.pose_estimator == "topk_roi_ransac":
        pose_protocol = (
            "wrist-only original Top-K EPNP RANSAC + projected-inlier ROI filtering + iterative LM; "
            f"{args.wrist_roi_source} wrist ROI; no fallback; no shaft/gripper fitting"
        )
    elif args.pose_estimator == "original_surfemb":
        roi_description = (
            "predicted binary wrist mask"
            if args.wrist_roi_source == "predicted"
            else "GT wrist segmentation"
        )
        pose_protocol = (
            "wrist-only original SurfEmb probability inversion sampling + AP3P hypotheses + "
            "mask/correspondence likelihood scoring"
            + (" + triangle-visible BFGS refinement" if bool(args.original_refine) else "")
            + f"; {roi_description}; no shaft/gripper fitting"
        )
    else:
        pose_protocol = "wrist-only SurfEmb probability AP3P; no SAM part labels; no shaft/gripper fitting"
    result = {
        "dataset": "SurgRIPE-LND/TEST",
        "num_frames": len(combined),
        "protocol": {
            "crop": (
                "original ycbv_ori_old detector bbox + "
                "SurfEmb RandomRotatedMaskCrop(scale=1.2, angle=0, offset=0, use_bbox=True)"
                if args.crop_mask_source == "detector"
                else (
                    f"SAM {args.crop_mask_source} mask + "
                    "SurfEmb RandomRotatedMaskCrop(scale=1.2, angle=0, offset=0)"
                )
            ),
            "pose": pose_protocol,
            "pose_estimator": args.pose_estimator,
            "correspondence_similarity": args.correspondence_similarity,
            "correspondence_temperature": float(args.correspondence_temperature),
            "rotation_ensemble": bool(args.rotation_ensemble),
            "wrist_roi_source": args.wrist_roi_source,
            "predicted_wrist_mask_mode": args.predicted_wrist_mask_mode,
            "excluded_frame_ids": [int(value) for value in args.exclude_frame_ids],
            "translation_unit": "mm",
            "rotation_unit": "degree",
            "historical_hcce_csv": str(args.historical_hcce_csv),
            "historical_hcce_checkpoint": "hcce_crop224_rarp_lnd_refinemem_bs56_gpu0234/checkpoints/iter0044000.pt",
            "historical_robopepp_csv": str(args.historical_robopepp_csv),
            "historical_robopepp_checkpoint": "robopepp_instrument_pose_rarp_lnd_refinemem_rotate_bs56_gpu0123/checkpoints/last.pt",
        },
        "surfemb": {},
        "historical": {},
        "paired_success": {},
    }
    for spec in specs:
        selected = [row for row in rows if row["model"] == spec["name"]]
        ok = [row for row in selected if row["status"] == "ok"]
        item = {
            "checkpoint": spec["path"],
            "checkpoint_iter": next((row["checkpoint_iter"] for row in selected), -1),
            "total": len(selected),
            "success": len(ok),
            "status_counts": {},
            "prediction_sym_flipped": int(sum(bool(row.get("prediction_sym_flipped", False)) for row in ok)),
            "metrics": {},
        }
        for row in selected:
            item["status_counts"][row["status"]] = item["status_counts"].get(row["status"], 0) + 1
        for mode in ("raw", "canonical"):
            for metric in ("trans_err_mm", "rot_err_deg"):
                key = f"{mode}_{metric}"
                item["metrics"][key] = stat([row.get(key, float("nan")) for row in ok])
        result["surfemb"][spec["name"]] = item

    historical = {
        "hcce_wrist_fit": ("hcce_wrist_fit_trans_err_mm", "hcce_wrist_fit_rot_err_deg"),
        "hcce_direct": ("hcce_direct_trans_err_mm", "hcce_direct_rot_err_deg"),
        "robopepp_direct": ("robopepp_direct_trans_err_mm", "robopepp_direct_rot_err_deg"),
    }
    for name, (trans_key, rot_key) in historical.items():
        result["historical"][name] = {
            "trans_err_mm": stat([row[trans_key] for row in combined]),
            "rot_err_deg": stat([row[rot_key] for row in combined]),
        }
    for spec in specs:
        prefix = spec["name"]
        paired = [row for row in combined if row[f"{prefix}_status"] == "ok"]
        result["paired_success"][prefix] = {
            "count": len(paired),
            "surfemb_canonical": {
                "trans_err_mm": stat([row[f"{prefix}_canonical_trans_err_mm"] for row in paired]),
                "rot_err_deg": stat([row[f"{prefix}_canonical_rot_err_deg"] for row in paired]),
            },
            "hcce_wrist_fit": {
                "trans_err_mm": stat([row["hcce_wrist_fit_trans_err_mm"] for row in paired]),
                "rot_err_deg": stat([row["hcce_wrist_fit_rot_err_deg"] for row in paired]),
            },
            "robopepp_direct": {
                "trans_err_mm": stat([row["robopepp_direct_trans_err_mm"] for row in paired]),
                "rot_err_deg": stat([row["robopepp_direct_rot_err_deg"] for row in paired]),
            },
        }
    return result


def write_summary(path, summary):
    lines = [
        "# SurfEmb wrist-only LND TEST evaluation",
        "",
        f"- frames: {summary['num_frames']}",
        f"- wrist matching ROI: {summary['protocol']['wrist_roi_source']}.",
        f"- predicted wrist mask mode: {summary['protocol']['predicted_wrist_mask_mode']}.",
        f"- pose estimator: {summary['protocol']['pose_estimator']}.",
        f"- TEST crop: {summary['protocol']['crop']}.",
        "- units: translation mm; rotation degree.",
        "",
        "| method | mode | success | trans mean / median / rmse | rot mean / median / rmse |",
        "|---|---|---:|---:|---:|",
    ]
    for name, item in summary["surfemb"].items():
        for mode in ("raw", "canonical"):
            trans = item["metrics"][f"{mode}_trans_err_mm"]
            rot = item["metrics"][f"{mode}_rot_err_deg"]
            lines.append(
                f"| {name} | {mode} | {item['success']}/{item['total']} | "
                f"{trans['mean']:.3f} / {trans['median']:.3f} / {trans['rmse']:.3f} | "
                f"{rot['mean']:.3f} / {rot['median']:.3f} / {rot['rmse']:.3f} |"
            )
    for name, item in summary["historical"].items():
        trans, rot = item["trans_err_mm"], item["rot_err_deg"]
        lines.append(
            f"| {name} | historical | {trans['count']}/{summary['num_frames']} | "
            f"{trans['mean']:.3f} / {trans['median']:.3f} / {trans['rmse']:.3f} | "
            f"{rot['mean']:.3f} / {rot['median']:.3f} / {rot['rmse']:.3f} |"
        )
    lines.extend(["", "## Paired successful subsets", ""])
    for name, item in summary["paired_success"].items():
        lines.append(f"- {name}: common frames={item['count']}")
        for method in ("surfemb_canonical", "hcce_wrist_fit", "robopepp_direct"):
            trans = item[method]["trans_err_mm"]
            rot = item[method]["rot_err_deg"]
            lines.append(
                f"  - {method}: trans mean/median={trans['mean']:.3f}/{trans['median']:.3f} mm; "
                f"rot mean/median={rot['mean']:.3f}/{rot['median']:.3f} deg"
            )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", default=None, help="NAME=CHECKPOINT")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1", "cuda:2", "cuda:3"])
    parser.add_argument("--lnd_root", default="/mnt/iMVR/daiyun/Dataset/LND")
    parser.add_argument(
        "--detection_folder",
        default="/mnt/iMVR/daiyun/surfemb/data/detection_results/ycbv_ori_old",
    )
    parser.add_argument("--historical_hcce_csv", default=str(DEFAULT_HCCE_CSV))
    parser.add_argument("--historical_robopepp_csv", default=str(DEFAULT_ROBO_CSV))
    parser.add_argument("--surface_root", default=str(surf_eval.DEFAULT_SURFACE_ROOT))
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--surfemb_crop_scale", type=float, default=1.2)
    parser.add_argument(
        "--crop_mask_source",
        choices=("instance", "wrist", "detector"),
        default="instance",
    )
    parser.add_argument("--bbox_padding_frac", type=float, default=0.12)
    parser.add_argument("--canonical_eps", type=float, default=0.08)
    parser.add_argument("--surface_keys_per_part", type=int, default=4096)
    parser.add_argument("--mask_keys_per_part", type=int, default=512)
    parser.add_argument("--surface_seed", type=int, default=2026)
    parser.add_argument("--pose_seed", type=int, default=20260802)
    parser.add_argument("--down_sample_scale", type=int, default=3)
    parser.add_argument(
        "--correspondence_similarity",
        choices=("raw_dot", "cosine"),
        default="raw_dot",
    )
    parser.add_argument("--correspondence_temperature", type=float, default=1.0)
    parser.add_argument(
        "--pose_estimator",
        choices=(
            "topk_ransac",
            "topk_roi_ransac",
            "spatial_topk_ransac",
            "coarse_fine_ransac",
            "probability_ap3p",
            "original_surfemb",
        ),
        default="topk_ransac",
    )
    parser.add_argument(
        "--wrist_roi_source",
        choices=("predicted", "gt_wrist"),
        default="predicted",
    )
    parser.add_argument(
        "--predicted_wrist_mask_mode",
        choices=("part_competition", "binary_head"),
        default="part_competition",
    )
    parser.add_argument("--exclude_frame_ids", nargs="*", type=int, default=[])
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
    parser.add_argument("--coarse_max_correspondences", type=int, default=4096)
    parser.add_argument("--coarse_keys_per_pixel", type=int, default=4)
    parser.add_argument("--coarse_min_inlier_fraction", type=float, default=0.0)
    parser.add_argument("--fine_neighbors_per_inlier", type=int, default=20)
    parser.add_argument("--fine_max_correspondences", type=int, default=4096)
    parser.add_argument("--fine_ransac_reprojection_error", type=float, default=2.0)
    parser.add_argument("--roi_refit_max_iterations", type=int, default=5)
    parser.add_argument("--spatial_grid_size", type=int, default=8)
    parser.add_argument("--spatial_max_per_grid_cell", type=int, default=12)
    parser.add_argument("--spatial_fps_2d_weight", type=float, default=0.5)
    parser.add_argument("--spatial_fps_3d_weight", type=float, default=0.5)
    parser.add_argument("--spatial_fps_confidence_power", type=float, default=0.25)
    parser.add_argument("--spatial_min_inlier_hull_fraction", type=float, default=0.15)
    parser.add_argument("--spatial_min_inlier_axis_span", type=float, default=0.25)
    parser.add_argument("--spatial_min_inlier_3d_span", type=float, default=0.20)
    parser.add_argument("--spatial_max_reprojection_median", type=float, default=1.5)
    parser.add_argument("--topk_bfgs_refine", type=int, choices=[0, 1], default=0)
    parser.add_argument("--topk_bfgs_max_iterations", type=int, default=50)
    parser.add_argument("--bfgs_refine_float64", type=int, choices=[0, 1], default=0)
    parser.add_argument("--bfgs_rotation_units_per_radian", type=float, default=1.0)
    parser.add_argument("--max_poses", type=int, default=4096)
    parser.add_argument("--max_pose_evaluations", type=int, default=512)
    parser.add_argument("--pose_batch_size", type=int, default=64)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--corr_alpha", type=float, default=1.5)
    parser.add_argument("--dist_2d_min", type=float, default=0.1)
    parser.add_argument("--original_refine", type=int, choices=[0, 1], default=0)
    parser.add_argument("--original_refine_denominator_keys", type=int, default=4096)
    parser.add_argument("--original_refine_min_visible_pixels", type=int, default=20)
    parser.add_argument("--original_refine_max_iterations", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--amp", type=int, choices=[0, 1], default=1)
    parser.add_argument("--rotation_ensemble", type=int, choices=[0, 1], default=0)
    parser.add_argument("--cpu_threads_per_worker", type=int, default=2)
    parser.add_argument("--print_freq", type=int, default=20)
    parser.add_argument("--fail_fast", type=int, choices=[0, 1], default=0)
    return parser


def main(args):
    args.output_dir = str(Path(args.output_dir).resolve())
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    specs = surf_eval.parse_model_specs(args.model or list(surf_eval.DEFAULT_MODELS))
    dataset = build_dataset(args)
    excluded = set(int(value) for value in args.exclude_frame_ids)
    dataset_frame_ids = {int(sample[0]) for sample in dataset.samples}
    unknown_exclusions = sorted(excluded - dataset_frame_ids)
    if unknown_exclusions:
        raise ValueError(f"Excluded frame IDs are absent from LND TEST: {unknown_exclusions}")
    all_indices = [index for index, sample in enumerate(dataset.samples) if int(sample[0]) not in excluded]
    full_total = len(all_indices)
    indices = all_indices if int(args.max_samples) <= 0 else all_indices[: int(args.max_samples)]
    total = len(indices)
    chunks = [indices[index :: len(args.devices)] for index in range(len(args.devices))]
    chunks = [chunk for chunk in chunks if chunk]
    args.devices = list(args.devices[: len(chunks)])
    print(
        f"dataset={dataset} selected={total} excluded={sorted(excluded)} "
        f"crop_mask_source={args.crop_mask_source} wrist_roi_source={args.wrist_roi_source} "
        f"predicted_wrist_mask_mode={args.predicted_wrist_mask_mode} workers={len(chunks)}",
        flush=True,
    )
    if len(chunks) == 1:
        worker_main(0, chunks, specs, args)
    else:
        mp.spawn(worker_main, args=(chunks, specs, args), nprocs=len(chunks), join=True)

    rows = []
    for rank in range(len(chunks)):
        rows.extend(json.loads((Path(args.output_dir) / "workers" / f"worker_{rank:02d}.json").read_text()))
    rows.sort(key=lambda row: (int(row["frame_id"]), str(row["model"])))
    write_csv(Path(args.output_dir) / "surfemb_per_frame.csv", rows)
    if total != full_total:
        print(f"smoke_complete rows={len(rows)}; historical comparison requires all selected frames", flush=True)
        return
    combined = combined_rows(rows, specs, args)
    write_csv(Path(args.output_dir) / "comparison_per_frame.csv", combined)
    summary = summarize(rows, specs, combined, args)
    (Path(args.output_dir) / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")
    write_summary(Path(args.output_dir) / "summary.md", summary)
    print(json.dumps(summary, indent=2, allow_nan=True), flush=True)
    print(f"summary={Path(args.output_dir) / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main(build_parser().parse_args())
