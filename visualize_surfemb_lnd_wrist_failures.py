#!/usr/bin/env python3
"""Diagnose SurfEmb wrist-only pose failures on SurgRIPE-LND TEST."""

import argparse
import csv
import json
import math
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import eval_surfemb_articulated_rarp as surf_eval
import eval_surfemb_wrist_lnd as lnd_eval
from instrument_geometry import quat_wxyz_to_matrix_np
from instrument_opengl_renderer import InstrumentOpenGLDepthRenderer
from surfemb_articulated_pose import (
    PART_NAMES,
    _score_pose_batch,
    build_part_probability_inputs,
    encode_surface_keys,
    estimate_part_pose_from_context,
    load_part_surfaces,
    prepare_part_score_context,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS = ROOT / "logs" / "surfemb_wrist_lnd_test_full" / "surfemb_per_frame.csv"
DEFAULT_OUT = ROOT / "logs" / "surfemb_wrist_lnd_failure_analysis"
DEFAULT_FRAMES = "62,347,103,78,132,210,341"
FRAME_LABELS = {
    62: "success",
    347: "rotation ambiguity",
    103: "hard subsequence",
    78: "model contrast",
    132: "severe correspondence failure",
    210: "missing wrist annotation",
    341: "small wrist / extreme failure",
}


def font(size):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def as_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def pose_from_transform(transform):
    return {
        "rot": lnd_eval.matrix_to_quat_wxyz(transform[:3, :3]),
        "trans": np.asarray(transform[:3, 3], dtype=np.float64),
        "alpha": 0.0,
        "theta_l": 0.0,
        "theta_r": 0.0,
    }


def pose_errors(pose, gt):
    if pose is None:
        return float("nan"), float("nan")
    return (
        float(np.linalg.norm(np.asarray(pose["trans"]) - np.asarray(gt["trans"])) * 1000.0),
        float(surf_eval.rotation_error_deg(pose["rot"], gt["rot"])),
    )


def probability_input(prob, h, w):
    prob = prob.reshape(-1).float().clamp(1e-7, 1.0 - 1e-7)
    log_prob = prob.log().reshape(h, w)
    neg_log_prob = torch.log1p(-prob).reshape(h, w)
    return {
        "prob": prob,
        "mask_log_prob": F.max_pool2d(log_prob[None, None], 3, 1, 1)[0, 0].reshape(-1),
        "neg_mask_log_prob": F.max_pool2d(neg_log_prob[None, None], 3, 1, 1)[0, 0].reshape(-1),
    }


def score_pose(context, surface, pose, K_ds, image_hw):
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = quat_wxyz_to_matrix_np(pose["rot"])
    transform[:3, 3] = np.asarray(pose["trans"], dtype=np.float64)
    score, mask_score, coord_score = _score_pose_batch(
        transform[None, :3],
        context.points,
        context.corr_log_score,
        context.mask_log_prob,
        context.neg_mask_log_prob,
        K_ds,
        image_hw,
    )
    return float(score[0]), float(mask_score[0]), float(coord_score[0])


def fit_context(context, surface, K_ds, image_hw, args, seed):
    hypotheses = estimate_part_pose_from_context(
        context,
        surface,
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
        return None, None
    pose = pose_from_transform(hypotheses[0].transform)
    return surf_eval.canonicalize_prediction(pose, args.canonical_eps), hypotheses[0]


def project_points(points, pose, K):
    points = np.asarray(points, dtype=np.float64)
    camera = points @ quat_wxyz_to_matrix_np(pose["rot"]).T + np.asarray(pose["trans"])
    uvw = camera @ np.asarray(K, dtype=np.float64).T
    uv = uvw[:, :2] / np.clip(uvw[:, 2:], 1e-9, None)
    return uv, camera[:, 2]


def top1_ransac_pose(query_flat, surface, pixel_mask, K_ds, image_hw, args):
    h, w = image_hw
    pixel_idx = torch.nonzero(pixel_mask.reshape(-1), as_tuple=False).reshape(-1)
    if len(pixel_idx) < 6:
        return None, {"top1_pnp_points": int(len(pixel_idx)), "top1_pnp_inliers": 0}
    logits = query_flat[pixel_idx] @ surface.keys.T
    confidence, key_idx = logits.max(dim=1)
    order = confidence.argsort(descending=True).cpu().numpy()
    key_np = key_idx.cpu().numpy()
    pixel_np = pixel_idx.cpu().numpy()
    unique = []
    seen = set()
    for index in order:
        key = int(key_np[index])
        if key in seen:
            continue
        seen.add(key)
        unique.append(int(index))
        if len(unique) >= int(args.top1_pnp_max_points):
            break
    if len(unique) < 6:
        return None, {"top1_pnp_points": int(len(unique)), "top1_pnp_inliers": 0}
    unique = np.asarray(unique, dtype=np.int64)
    selected_pixel = pixel_np[unique]
    object_points = np.asarray(surface.points_m[key_np[unique]], dtype=np.float64)
    image_points = np.stack((selected_pixel % w, selected_pixel // w), axis=1).astype(np.float64)
    try:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points,
            image_points,
            np.asarray(K_ds, dtype=np.float64),
            None,
            iterationsCount=int(args.top1_pnp_iters),
            reprojectionError=float(args.top1_pnp_reproj_error),
            confidence=0.999,
            flags=cv2.SOLVEPNP_EPNP,
        )
    except cv2.error:
        ok, inliers = False, None
    n_inliers = 0 if inliers is None else int(len(inliers))
    diagnostics = {"top1_pnp_points": int(len(unique)), "top1_pnp_inliers": n_inliers}
    if not ok or n_inliers < 6:
        return None, diagnostics
    inlier_idx = inliers.reshape(-1)
    try:
        rvec, tvec = cv2.solvePnPRefineLM(
            object_points[inlier_idx],
            image_points[inlier_idx],
            np.asarray(K_ds, dtype=np.float64),
            None,
            rvec,
            tvec,
        )
    except cv2.error:
        pass
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    pose = surf_eval.canonicalize_prediction(pose_from_transform(transform), args.canonical_eps)
    return pose, diagnostics


@torch.inference_mode()
def diagnose(model, surfaces, image, K_crop, crop_rgb, inst_crop, part_crop, gt, args, seed):
    device = next(model.parameters()).device
    x = image[None].to(device=device, non_blocking=True)
    K_tensor = torch.from_numpy(K_crop)[None].to(device=device, non_blocking=True)
    with torch.amp.autocast(
        device_type=device.type,
        enabled=device.type == "cuda" and bool(args.amp),
        dtype=torch.bfloat16,
    ):
        output = model(x, K_tensor)

    mask_logits = output["inst_mask_logits"][0].float()
    queries = output["surfemb_queries"][0].float()
    query_flat, part_inputs, K_ds, image_hw, object_prob = build_part_probability_inputs(
        mask_logits,
        queries,
        surfaces,
        K_crop,
        down_sample_scale=int(args.down_sample_scale),
    )
    h, w = image_hw
    wrist = surfaces["wrist"]
    learned_context = prepare_part_score_context(query_flat, part_inputs["wrist"], wrist, image_hw)
    pred, hypothesis = fit_context(learned_context, wrist, K_ds, image_hw, args, seed)

    wrist_gt = torch.from_numpy((part_crop == 2).astype(np.float32)).to(device)
    wrist_gt_ds = F.max_pool2d(
        wrist_gt[None, None], int(args.down_sample_scale), int(args.down_sample_scale)
    )[0, 0]
    gt_ds = wrist_gt_ds > 0
    oracle_prob = torch.where(
        wrist_gt_ds.reshape(-1) > 0,
        torch.full((h * w,), 0.995, device=device),
        torch.full((h * w,), 1e-7, device=device),
    )
    oracle_context = prepare_part_score_context(
        query_flat,
        probability_input(oracle_prob, h, w),
        wrist,
        image_hw,
    )
    if int(gt_ds.sum().item()) > 0:
        oracle_pred, oracle_hypothesis = fit_context(
            oracle_context,
            wrist,
            K_ds,
            image_hw,
            args,
            seed + 70000001,
        )
        oracle_score_gt = score_pose(oracle_context, wrist, gt, K_ds, image_hw)
        top1_pred, top1_diagnostics = top1_ransac_pose(
            query_flat,
            wrist,
            gt_ds,
            K_ds,
            image_hw,
            args,
        )
    else:
        oracle_pred, oracle_hypothesis = None, None
        oracle_score_gt = (float("nan"), float("nan"), float("nan"))
        top1_pred = None
        top1_diagnostics = {"top1_pnp_points": 0, "top1_pnp_inliers": 0}

    learned_score_gt = score_pose(learned_context, wrist, gt, K_ds, image_hw)
    object_map = object_prob.reshape(h, w)
    wrist_prob = part_inputs["wrist"]["prob"].reshape(h, w)
    part_prob = torch.stack([part_inputs[name]["prob"] for name in PART_NAMES], dim=0).reshape(4, h, w)
    part_argmax = part_prob.argmax(dim=0)

    wrist_mass = wrist_prob.sum().clamp_min(1e-9)
    wrist_mass_precision = float(wrist_prob[gt_ds].sum() / wrist_mass)
    gt_count = int(gt_ds.sum().item())
    if gt_count:
        top_idx = torch.topk(wrist_prob.reshape(-1), k=min(gt_count, h * w)).indices
        topk_hit = float(gt_ds.reshape(-1)[top_idx].float().mean())
        wrist_part_recall = float((part_argmax[gt_ds] == 1).float().mean())
        conditional = learned_context.corr_prob[gt_ds.reshape(-1)] / wrist_prob[gt_ds].unsqueeze(1).clamp_min(1e-12)
        conditional = conditional.clamp_min(1e-12)
        corr_max_prob_median = float(conditional.max(dim=1).values.median())
        corr_effective_keys_median = float(torch.exp(-(conditional * conditional.log()).sum(dim=1)).median())
    else:
        topk_hit = float("nan")
        wrist_part_recall = float("nan")
        corr_max_prob_median = float("nan")
        corr_effective_keys_median = float("nan")

    inst_gt = torch.from_numpy(np.asarray(inst_crop, dtype=np.float32)).to(device)
    inst_gt_ds = F.max_pool2d(
        inst_gt[None, None], int(args.down_sample_scale), int(args.down_sample_scale)
    )[0, 0] > 0
    inst_pred_ds = object_map > 0.5
    object_iou = float(
        (inst_gt_ds & inst_pred_ds).sum()
        / (inst_gt_ds | inst_pred_ds).sum().clamp_min(1)
    )

    corr_canvas = np.asarray(crop_rgb, dtype=np.uint8).copy()
    corr_candidates = torch.nonzero(
        wrist_prob.reshape(-1) > torch.quantile(wrist_prob, 0.96), as_tuple=False
    ).reshape(-1)
    if len(corr_candidates) > int(args.corr_vis_points):
        choose = torch.linspace(0, len(corr_candidates) - 1, int(args.corr_vis_points), device=device).round().long()
        corr_candidates = corr_candidates[choose]
    corr_residuals = []
    if len(corr_candidates):
        key_idx = (query_flat[corr_candidates] @ wrist.keys.T).argmax(dim=1)
        key_points = wrist.points_m[key_idx.cpu().numpy()]
        uv_gt, depth_gt = project_points(key_points, gt, K_crop)
        yy = torch.div(corr_candidates, w, rounding_mode="floor").cpu().numpy()
        xx = (corr_candidates % w).cpu().numpy()
        uv_query = np.stack(
            [
                (xx + 0.5) * int(args.down_sample_scale) - 0.5,
                (yy + 0.5) * int(args.down_sample_scale) - 0.5,
            ],
            axis=1,
        )
        for start, end, depth in zip(uv_query, uv_gt, depth_gt):
            if depth <= 0 or not np.isfinite(end).all():
                continue
            residual = float(np.linalg.norm(start - end))
            corr_residuals.append(residual)
            color = (30, 220, 70) if residual < 6 else ((250, 190, 20) if residual < 15 else (245, 55, 55))
            p0 = tuple(np.rint(start).astype(int))
            p1 = tuple(np.rint(end).astype(int))
            cv2.line(corr_canvas, p0, p1, color, 1, cv2.LINE_AA)
            cv2.circle(corr_canvas, p0, 2, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(corr_canvas, p1, 2, color, -1, cv2.LINE_AA)

    object_pred = torch.sigmoid(mask_logits).cpu().numpy()
    wrist_prob_up = cv2.resize(wrist_prob.cpu().numpy(), (crop_rgb.shape[1], crop_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    part_argmax_up = cv2.resize(
        part_argmax.cpu().numpy().astype(np.uint8),
        (crop_rgb.shape[1], crop_rgb.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    t_err, r_err = pose_errors(pred, gt)
    ot_err, or_err = pose_errors(oracle_pred, gt)
    pt_err, pr_err = pose_errors(top1_pred, gt)
    return {
        "pred": pred,
        "oracle_pred": oracle_pred,
        "top1_pred": top1_pred,
        "hypothesis": hypothesis,
        "oracle_hypothesis": oracle_hypothesis,
        "object_pred": object_pred,
        "wrist_prob": wrist_prob_up,
        "part_argmax": part_argmax_up,
        "corr_canvas": corr_canvas,
        "trans_err_mm": t_err,
        "rot_err_deg": r_err,
        "oracle_trans_err_mm": ot_err,
        "oracle_rot_err_deg": or_err,
        "top1_trans_err_mm": pt_err,
        "top1_rot_err_deg": pr_err,
        "object_iou": object_iou,
        "wrist_mass_precision": wrist_mass_precision,
        "wrist_topk_hit": topk_hit,
        "wrist_part_recall": wrist_part_recall,
        "corr_max_prob_median": corr_max_prob_median,
        "corr_effective_keys_median": corr_effective_keys_median,
        "wrist_gt_area": int((part_crop == 2).sum()),
        "corr_resid_median_px": float(np.median(corr_residuals)) if corr_residuals else float("nan"),
        "corr_resid_p90_px": float(np.quantile(corr_residuals, 0.9)) if corr_residuals else float("nan"),
        "learned_score_gt": learned_score_gt[0],
        "learned_score_pred": float(hypothesis.score) if hypothesis else float("nan"),
        "oracle_score_gt": oracle_score_gt[0],
        "oracle_score_pred": float(oracle_hypothesis.score) if oracle_hypothesis else float("nan"),
        **top1_diagnostics,
    }


def resize_square(rgb, size):
    rgb = np.asarray(rgb, dtype=np.uint8)
    h, w = rgb.shape[:2]
    scale = min(float(size) / max(w, 1), float(size) / max(h, 1))
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    out = np.zeros((size, size, 3), dtype=np.uint8)
    x0, y0 = (size - nw) // 2, (size - nh) // 2
    out[y0 : y0 + nh, x0 : x0 + nw] = resized
    return out


def title_tile(rgb, title, subtitle, size):
    image = Image.fromarray(resize_square(rgb, size))
    header = 70
    canvas = Image.new("RGB", (size, size + header), (247, 247, 247))
    canvas.paste(image, (0, header))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 5), title, fill=(15, 15, 15), font=font(17))
    draw.text((8, 32), subtitle[:62], fill=(65, 65, 65), font=font(12))
    return np.asarray(canvas)


def overlay_sam_parts(rgb, part):
    out = np.asarray(rgb, dtype=np.uint8).copy()
    colors = {1: (240, 70, 190), 2: (40, 230, 80), 3: (70, 140, 240)}
    for label, color in colors.items():
        mask = np.asarray(part) == label
        out[mask] = np.clip(out[mask].astype(np.float32) * 0.42 + np.asarray(color) * 0.58, 0, 255).astype(np.uint8)
    return out


def mask_mismatch_tile(rgb, gt, prob):
    out = np.asarray(rgb, dtype=np.uint8).copy()
    pred = np.asarray(prob) > 0.5
    gt = np.asarray(gt, dtype=bool)
    colors = np.zeros_like(out)
    colors[gt & pred] = (250, 220, 40)
    colors[gt & ~pred] = (40, 225, 80)
    colors[~gt & pred] = (245, 55, 70)
    support = gt | pred
    out[support] = np.clip(out[support].astype(np.float32) * 0.35 + colors[support] * 0.65, 0, 255).astype(np.uint8)
    return out


def wrist_probability_tile(rgb, probability, wrist_gt, part_argmax):
    heat = cv2.applyColorMap(np.uint8(np.clip(probability, 0, 1) * 255), cv2.COLORMAP_TURBO)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    out = np.clip(np.asarray(rgb) * 0.38 + heat * 0.62, 0, 255).astype(np.uint8)
    contours, _ = cv2.findContours(np.asarray(wrist_gt, dtype=np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (255, 255, 255), 2, cv2.LINE_AA)
    wrong = (np.asarray(part_argmax) != 1) & np.asarray(wrist_gt, dtype=bool)
    out[wrong] = np.clip(out[wrong].astype(np.float32) * 0.3 + np.array([255, 20, 20]) * 0.7, 0, 255).astype(np.uint8)
    return out


def wrist_mesh_comparison(rgb, renderer, gt, pred, K):
    gt_mask = renderer.render_pose_mask(gt, K, rgb.shape[:2]) == 2
    pred_mask = np.zeros_like(gt_mask)
    if pred is not None:
        pred_mask = renderer.render_pose_mask(pred, K, rgb.shape[:2]) == 2
    out = np.asarray(rgb, dtype=np.uint8).copy()
    colors = np.zeros_like(out)
    colors[gt_mask & pred_mask] = (250, 220, 40)
    colors[gt_mask & ~pred_mask] = (40, 230, 80)
    colors[~gt_mask & pred_mask] = (245, 60, 190)
    support = gt_mask | pred_mask
    out[support] = np.clip(out[support].astype(np.float32) * 0.28 + colors[support] * 0.72, 0, 255).astype(np.uint8)
    intersection = np.count_nonzero(gt_mask & pred_mask)
    union = np.count_nonzero(gt_mask | pred_mask)
    return out, float(intersection / max(1, union))


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


def global_plots(results_csv, output_path):
    rows = list(csv.DictReader(Path(results_csv).open(encoding="utf-8")))
    colors = {"resnet": "#167d9a", "dino_multihead": "#d1495b"}
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    for model in ("resnet", "dino_multihead"):
        selected = [row for row in rows if row["model"] == model and row["status"] == "ok"]
        frame = np.asarray([int(row["frame_id"]) for row in selected])
        trans = np.asarray([float(row["canonical_trans_err_mm"]) for row in selected])
        rot = np.asarray([float(row["canonical_rot_err_deg"]) for row in selected])
        area = np.asarray([float(row["pred_mask_area_ds"]) for row in selected])
        score = np.asarray([float(row["wrist_score"]) for row in selected])
        severity = trans / 10.0 + rot / 15.0
        axes[0, 0].plot(frame, trans, color=colors[model], alpha=0.75, linewidth=1.3, label=model)
        axes[0, 1].plot(frame, rot, color=colors[model], alpha=0.75, linewidth=1.3, label=model)
        axes[1, 0].scatter(area, severity, s=13, alpha=0.52, color=colors[model], label=model)
        axes[1, 1].scatter(score, severity, s=13, alpha=0.52, color=colors[model], label=model)
    for axis in axes[0]:
        axis.axvspan(76, 150, color="#f4a261", alpha=0.16, label="hard frames 76-150")
        axis.axvline(210, color="#6c757d", linestyle="--", linewidth=1)
    axes[0, 0].set(title="LND TEST translation error by frame", xlabel="frame", ylabel="translation error (mm)")
    axes[0, 1].set(title="LND TEST rotation error by frame", xlabel="frame", ylabel="rotation error (degree)")
    axes[1, 0].set(title="Small predicted support correlates with failure", xlabel="predicted object pixels at 74x74", ylabel="pose severity")
    axes[1, 1].set(title="Low wrist hypothesis score correlates with failure", xlabel="best wrist score", ylabel="pose severity")
    for axis in axes.reshape(-1):
        axis.grid(alpha=0.22)
        axis.legend(loc="best")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main(args):
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = surf_eval.parse_model_specs(args.model or list(surf_eval.DEFAULT_MODELS))
    frames = [int(value) for value in args.frames.split(",") if value.strip()]
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    dataset = lnd_eval.build_dataset(args)
    frame_to_index = {int(dataset.samples[index][0]): index for index in range(len(dataset))}
    missing = sorted(set(frames) - set(frame_to_index))
    if missing:
        raise ValueError(f"Frames absent from LND TEST: {missing}")

    renderer = InstrumentOpenGLDepthRenderer(device_idx=int(args.opengl_device))
    rows = []
    model_panel_paths = {}
    for model_index, spec in enumerate(specs):
        model, checkpoint_iter, _ = surf_eval.load_model(spec, device)
        surfaces = load_part_surfaces(
            args.surface_root,
            keys_per_part=int(args.surface_keys_per_part),
            seed=int(args.surface_seed),
        )
        encode_surface_keys(model, surfaces, device, mask_keys_per_part=int(args.mask_keys_per_part))
        panels = []
        paths = []
        for frame_id in frames:
            dataset_idx = frame_to_index[frame_id]
            _, target = dataset[dataset_idx]
            image, K_crop, crop_rgb, M_crop = surf_eval.make_surfemb_crop(target, args)
            part_orig = as_numpy(target["part_mask_orig"]).astype(np.uint8)
            part_crop = cv2.warpAffine(
                part_orig,
                M_crop,
                (int(args.crop_size), int(args.crop_size)),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            inst_crop = cv2.warpAffine(
                as_numpy(target["inst_mask_orig"]).astype(np.uint8),
                M_crop,
                (int(args.crop_size), int(args.crop_size)),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            ).astype(bool)
            gt = surf_eval.target_pose(target)
            diag = diagnose(
                model,
                surfaces,
                image,
                K_crop,
                crop_rgb,
                inst_crop,
                part_crop,
                gt,
                args,
                seed=int(args.pose_seed) + dataset_idx * 1009 + model_index * 10000019,
            )
            rgb = as_numpy(target["orig_rgb"]).astype(np.uint8)
            K_orig = as_numpy(target["K_orig"]).astype(np.float32)
            standard_mesh, standard_iou = wrist_mesh_comparison(rgb, renderer, gt, diag["pred"], K_orig)
            oracle_mesh, oracle_iou = wrist_mesh_comparison(rgb, renderer, gt, diag["oracle_pred"], K_orig)
            top1_mesh, top1_iou = wrist_mesh_comparison(rgb, renderer, gt, diag["top1_pred"], K_orig)
            tiles = [
                title_tile(
                    overlay_sam_parts(rgb, part_orig),
                    "RGB + SAM parts",
                    "green=wrist, pink=gripper, blue=shaft",
                    args.panel_size,
                ),
                title_tile(
                    mask_mismatch_tile(crop_rgb, inst_crop, diag["object_pred"]),
                    "Binary mask",
                    f"IoU={diag['object_iou']:.3f}; yellow=overlap",
                    args.panel_size,
                ),
                title_tile(
                    wrist_probability_tile(crop_rgb, diag["wrist_prob"], part_crop == 2, diag["part_argmax"]),
                    "Learned wrist probability",
                    f"mass-in-GT={diag['wrist_mass_precision']:.2f} topK-hit={diag['wrist_topk_hit']:.2f}",
                    args.panel_size,
                ),
                title_tile(
                    diag["corr_canvas"],
                    "Top-1 pixel->3D wrist",
                    f"GT reproj residual med/p90={diag['corr_resid_median_px']:.1f}/{diag['corr_resid_p90_px']:.1f}px",
                    args.panel_size,
                ),
                title_tile(
                    standard_mesh,
                    "Standard SurfEmb fit",
                    f"t={diag['trans_err_mm']:.1f}mm r={diag['rot_err_deg']:.1f}deg meshIoU={standard_iou:.2f}",
                    args.panel_size,
                ),
                title_tile(
                    oracle_mesh,
                    "GT-wrist-ROI oracle fit",
                    f"t={diag['oracle_trans_err_mm']:.1f}mm r={diag['oracle_rot_err_deg']:.1f}deg meshIoU={oracle_iou:.2f}",
                    args.panel_size,
                ),
                title_tile(
                    top1_mesh,
                    "GT ROI + top-1 RANSAC",
                    f"t={diag['top1_trans_err_mm']:.1f}mm r={diag['top1_rot_err_deg']:.1f}deg inliers={diag['top1_pnp_inliers']}",
                    args.panel_size,
                ),
            ]
            row_panel = np.concatenate(tiles, axis=1)
            banner_h = 45
            panel_image = Image.fromarray(row_panel)
            canvas = Image.new("RGB", (panel_image.width, panel_image.height + banner_h), (28, 28, 28))
            canvas.paste(panel_image, (0, banner_h))
            ImageDraw.Draw(canvas).text(
                (10, 10),
                f"{spec['name']}  frame={frame_id}  {FRAME_LABELS.get(frame_id, 'representative')}",
                fill=(255, 255, 255),
                font=font(19),
            )
            path = output_dir / f"{spec['name']}_frame{frame_id:06d}_wrist_diagnostic.jpg"
            canvas.save(path, quality=94)
            panels.append(canvas)
            paths.append(path)
            row = {
                "model": spec["name"],
                "checkpoint_iter": int(checkpoint_iter),
                "frame_id": frame_id,
                "category": FRAME_LABELS.get(frame_id, "representative"),
                "standard_mesh_iou": standard_iou,
                "oracle_mesh_iou": oracle_iou,
                "top1_mesh_iou": top1_iou,
            }
            for key, value in diag.items():
                if isinstance(value, (bool, int, float, str)):
                    row[key] = value
            rows.append(row)
            print(
                f"{spec['name']} frame={frame_id} standard={diag['trans_err_mm']:.2f}mm/{diag['rot_err_deg']:.2f}deg "
                f"oracle={diag['oracle_trans_err_mm']:.2f}mm/{diag['oracle_rot_err_deg']:.2f}deg",
                flush=True,
            )
        width = max(panel.width for panel in panels)
        contact = Image.new("RGB", (width, sum(panel.height for panel in panels)), "white")
        top = 0
        for panel in panels:
            contact.paste(panel, (0, top))
            top += panel.height
        contact_path = output_dir / f"{spec['name']}_selected_wrist_failures_contact_sheet.jpg"
        contact.save(contact_path, quality=93)
        model_panel_paths[spec["name"]] = str(contact_path)
        del model, surfaces
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_csv(output_dir / "selected_case_diagnostics.csv", rows)
    global_plots(args.results_csv, output_dir / "global_error_diagnostics.png")
    summary = {
        "frames": frames,
        "models": [spec["name"] for spec in specs],
        "contact_sheets": model_panel_paths,
        "diagnostics_csv": str(output_dir / "selected_case_diagnostics.csv"),
        "global_plot": str(output_dir / "global_error_diagnostics.png"),
    }
    (output_dir / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def build_parser():
    parser = lnd_eval.build_parser()
    parser.set_defaults(
        output_dir=str(DEFAULT_OUT),
        devices=["cuda:0"],
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--opengl_device", type=int, default=0)
    parser.add_argument("--frames", default=DEFAULT_FRAMES)
    parser.add_argument("--results_csv", default=str(DEFAULT_RESULTS))
    parser.add_argument("--panel_size", type=int, default=300)
    parser.add_argument("--corr_vis_points", type=int, default=48)
    parser.add_argument("--top1_pnp_max_points", type=int, default=512)
    parser.add_argument("--top1_pnp_iters", type=int, default=2000)
    parser.add_argument("--top1_pnp_reproj_error", type=float, default=3.0)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
