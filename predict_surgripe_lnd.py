import argparse
import csv
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from PIL import Image

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

ROBOPEPP_ROOT = Path(__file__).resolve().parent
MULTIHMR_ROOT = ROBOPEPP_ROOT.parents[1]
if str(ROBOPEPP_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBOPEPP_ROOT))
if str(MULTIHMR_ROOT) not in sys.path:
    sys.path.insert(0, str(MULTIHMR_ROOT))

import compare_crop_hcce_robopepp_rarp as cmp  # noqa: E402
from instrument_trimesh_renderer import GMSInstrumentTrimeshRenderer  # noqa: E402


def _load_local_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


surgripe_lnd = _load_local_module(
    "robopepp_surgripe_lnd_dataset",
    ROBOPEPP_ROOT / "datasets" / "surgripe_lnd.py",
)
surgripe_lnd_instrument = _load_local_module(
    "robopepp_surgripe_lnd_instrument_dataset",
    ROBOPEPP_ROOT / "datasets" / "surgripe_lnd_instrument.py",
)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = []
    seen = set()
    preferred = [
        "dataset",
        "split",
        "frame_id",
        "ordinal",
        "robopepp_pnp_status",
        "hcce_direct_status",
        "hcce_fit_status",
        "hcce_kp_pnp_status",
        "vis_path",
    ]
    for key in preferred:
        if any(key in row for row in rows):
            seen.add(key)
            fieldnames.append(key)
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def finite_values(rows, key):
    vals = []
    for row in rows:
        try:
            value = float(row.get(key, float("nan")))
        except (TypeError, ValueError):
            value = float("nan")
        if math.isfinite(value):
            vals.append(value)
    return np.asarray(vals, dtype=np.float64)


def stat(rows, key):
    vals = finite_values(rows, key)
    if vals.size == 0:
        return {"count": 0, "mean": float("nan"), "median": float("nan"), "rmse": float("nan")}
    return {
        "count": int(vals.size),
        "mean": float(vals.mean()),
        "median": float(np.median(vals)),
        "rmse": float(np.sqrt(np.mean(vals * vals))),
    }


def status_counts(rows):
    out = {}
    for key in ("robopepp_pnp_status", "hcce_direct_status", "hcce_fit_status", "hcce_kp_pnp_status"):
        counts = {}
        for row in rows:
            value = str(row.get(key, "missing"))
            counts[value] = counts.get(value, 0) + 1
        out[key] = counts
    return out


def make_fit_args(args, hcce_meta):
    return SimpleNamespace(
        inst_thresh=float(args.inst_thresh),
        hcce_bits=int(hcce_meta.get("hcce_bits", args.hcce_bits)),
        hcce_coord_min=float(hcce_meta.get("hcce_coord_min", args.hcce_coord_min)),
        hcce_coord_max=float(hcce_meta.get("hcce_coord_max", args.hcce_coord_max)),
        hcce_bit_thresh=float(args.hcce_bit_thresh),
        max_points_per_part=int(args.max_points_per_part),
        point_select="random",
        fit_seg_source="pred",
        surface_snap_method=str(args.surface_snap_method),
        surface_k_faces=int(args.surface_k_faces),
        hcce_axis_scale=args.hcce_axis_scale,
        shaft_raw_x_min=args.shaft_raw_x_min,
        min_wrist_points=int(args.min_wrist_points),
        min_total_points=int(args.min_total_points),
        min_shaft_points=int(args.min_shaft_points),
        min_pnp_inliers=int(args.min_pnp_inliers),
        pnp_iters=int(args.pnp_iters),
        pnp_reproj_error=float(args.pnp_reproj_error),
        pnp_confidence=float(args.pnp_confidence),
        freeze_wrist_after_pnp=0,
        hcce_fit_mode=str(args.hcce_fit_mode),
        optim_strategy="decoupled",
        optim_parts="wrist_gripper",
        optim_loss=str(args.optim_loss),
        optim_f_scale=float(args.optim_f_scale),
        optim_max_nfev=int(args.optim_max_nfev),
        gripper_no_wrist_pose=int(args.gripper_no_wrist_pose),
        min_depth=1e-4,
        behind_camera_penalty=1e4,
        render_seg_metrics=0,
        render_seg_metrics_limit=-1,
        render_min_depth=1e-4,
        render_draw_margin=20.0,
        overlay_alpha=float(args.overlay_alpha),
    )


def target_like_from_dataset(target):
    return {
        "orig_rgb": target["orig_rgb"],
        "crop_rgb": target["crop_rgb"],
        "gt_part_orig": target["gt_part_orig"],
        "gt_part_crop": target["gt_part_crop"],
        "K_orig": target["K_orig"],
        "K_crop": target["K_crop"],
        "bbox_min": target["bbox_min"],
        "bbox_max": target["bbox_max"],
        "scale": target["scale"],
        "pad": target["pad"],
        "resized_size": target["resized_size"],
        "keypoints_crop": target["keypoints_crop"],
        "keypoints_valid": target["keypoints_valid"],
    }


def _to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def resized_size_from_bbox(bbox_min, bbox_max, crop_size):
    width = float(bbox_max[0] - bbox_min[0])
    height = float(bbox_max[1] - bbox_min[1])
    if width > height:
        new_w = int(crop_size)
        new_h = int(crop_size * height / max(width, 1e-6))
    else:
        new_h = int(crop_size)
        new_w = int(crop_size * width / max(height, 1e-6))
    return np.asarray([max(1, new_w), max(1, new_h)], dtype=np.int32)


def pose_from_instrument_target(target):
    action = _to_numpy(target["action"]).astype(np.float64).reshape(3)
    quat = _to_numpy(target["wrist_quat"]).astype(np.float64).reshape(4)
    quat = quat / max(np.linalg.norm(quat), 1e-12)
    if quat[0] < 0.0:
        quat = -quat
    return {
        "rot": quat,
        "trans": _to_numpy(target["wrist_trans"]).astype(np.float64).reshape(3),
        "alpha": float(action[0]),
        "theta_l": float(action[1]),
        "theta_r": float(action[2]),
    }


def target_like_from_instrument_dataset(target, crop_size):
    bbox_min = _to_numpy(target["bbox_min"]).astype(np.float32)
    bbox_max = _to_numpy(target["bbox_max"]).astype(np.float32)
    gt_part_orig = _to_numpy(target["part_mask_orig"]).astype(np.uint8)
    gt_part_crop = cmp.crop_resize_pad_map(
        gt_part_orig,
        bbox_min,
        bbox_max,
        int(crop_size),
        cv2.INTER_NEAREST,
        value=0,
    ).astype(np.uint8)
    frame_id = int(str(target["frame_id"]))
    return {
        "dataset": "surgripe_lnd",
        "split": "TEST",
        "frame_id": frame_id,
        "image_path": str(target["img_path"]),
        "mask_path": str(target["img_path"]),
        "crop_source": "sam3_inst_part",
        "orig_rgb": _to_numpy(target["orig_rgb"]).astype(np.uint8),
        "crop_rgb": _to_numpy(target["crop_rgb"]).astype(np.uint8),
        "wrist_mask_orig": (gt_part_orig == 2).astype(np.uint8),
        "wrist_mask_crop": (gt_part_crop == 2).astype(np.uint8),
        "gt_part_orig": gt_part_orig,
        "gt_part_crop": gt_part_crop,
        "K_orig": _to_numpy(target["K_orig"]).astype(np.float32),
        "K_crop": _to_numpy(target["K"]).astype(np.float32),
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
        "scale": _to_numpy(target["scale"]).astype(np.float32),
        "pad": _to_numpy(target["pad"]).astype(np.float32),
        "resized_size": resized_size_from_bbox(bbox_min, bbox_max, int(crop_size)),
        "gt_pose": pose_from_instrument_target(target),
        "keypoints_orig": _to_numpy(target["keypoints_orig"]).astype(np.float32),
        "keypoints_crop": _to_numpy(target["keypoints_crop"]).astype(np.float32),
        "keypoints_valid": _to_numpy(target["keypoints_valid"]).astype(bool),
        "keypoints_valid_orig": _to_numpy(target["keypoints_valid_orig"]).astype(bool),
    }


def crop_map_to_orig(target, crop_map, interpolation):
    return surgripe_lnd.unproject_crop_map_to_orig(
        crop_map,
        target["orig_rgb"].shape[:2],
        target["bbox_min"],
        target["bbox_max"],
        target["resized_size"],
        target["pad"],
        interpolation=interpolation,
    )


def recrop_with_gt_mesh(renderer, target, to_tensor, args):
    rgb = target["orig_rgb"]
    mesh_part = renderer.render_pose_mask(
        target["gt_pose"],
        target["K_orig"],
        rgb.shape[:2],
        min_depth=1e-4,
        draw_margin=20.0,
    ).astype(np.uint8)
    if not np.any(mesh_part > 0):
        raise RuntimeError(f"Empty GT repo mesh render mask for LND frame {target['frame_id']}")
    bbox_min, bbox_max = surgripe_lnd._bbox_from_mask(
        mesh_part > 0,
        rgb.shape[:2],
        bbox_scale=float(args.mesh_bbox_scale),
        min_crop_size=int(args.mesh_min_crop_size),
        margin_px=int(args.mesh_bbox_margin_px),
    )
    crop_rgb, scale_xy, pad_xy, resized_size = surgripe_lnd.crop_resize_pad_array(
        rgb,
        bbox_min,
        bbox_max,
        int(args.crop_size),
        cv2.INTER_LINEAR,
        value=0,
    )
    wrist_crop, _, _, _ = surgripe_lnd.crop_resize_pad_array(
        target["wrist_mask_orig"].astype(np.uint8),
        bbox_min,
        bbox_max,
        int(args.crop_size),
        cv2.INTER_NEAREST,
        value=0,
    )
    mesh_part_crop, _, _, _ = surgripe_lnd.crop_resize_pad_array(
        mesh_part,
        bbox_min,
        bbox_max,
        int(args.crop_size),
        cv2.INTER_NEAREST,
        value=0,
    )
    K_crop = surgripe_lnd.crop_resize_pad_intrinsics(
        target["K_orig"],
        bbox_min=bbox_min,
        scale_xy=scale_xy,
        pad_xy=pad_xy,
    )
    kp_crop = target["keypoints_orig"].copy()
    kp_crop[:, 0] = (kp_crop[:, 0] - bbox_min[0]) * scale_xy[0] + pad_xy[0]
    kp_crop[:, 1] = (kp_crop[:, 1] - bbox_min[1]) * scale_xy[1] + pad_xy[1]
    kp_valid = (
        target["keypoints_valid_orig"].astype(bool)
        & (kp_crop[:, 0] >= 0.0)
        & (kp_crop[:, 0] < float(args.crop_size))
        & (kp_crop[:, 1] >= 0.0)
        & (kp_crop[:, 1] < float(args.crop_size))
    )
    target = dict(target)
    target.update(
        {
            "crop_source": "gt_repo_mesh",
            "crop_rgb": crop_rgb.astype(np.uint8),
            "wrist_mask_crop": wrist_crop.astype(np.uint8),
            "gt_mesh_part_orig": mesh_part.astype(np.uint8),
            "gt_mesh_part_crop": mesh_part_crop.astype(np.uint8),
            "K_crop": K_crop.astype(np.float32),
            "bbox_min": bbox_min.astype(np.float32),
            "bbox_max": bbox_max.astype(np.float32),
            "scale": np.asarray(scale_xy, dtype=np.float32),
            "pad": np.asarray(pad_xy, dtype=np.float32),
            "resized_size": np.asarray(resized_size, dtype=np.int32),
            "keypoints_crop": kp_crop.astype(np.float32),
            "keypoints_valid": kp_valid.astype(bool),
        }
    )
    image_tensor = to_tensor(Image.fromarray(crop_rgb.astype(np.uint8)))
    return image_tensor, target


def decode_hcce_seg_full(hcce_out, target, inst_thresh):
    inst_prob_crop = torch.sigmoid(hcce_out["inst_mask_logits"][0].detach().float().cpu()).numpy()
    inst_crop = (inst_prob_crop >= float(inst_thresh)).astype(np.uint8)
    part_crop = cmp.model_part_mask_crop(hcce_out, inst_thresh)
    inst_prob_full = crop_map_to_orig(target, inst_prob_crop.astype(np.float32), cv2.INTER_LINEAR)
    inst_full = crop_map_to_orig(target, inst_crop, cv2.INTER_NEAREST).astype(bool)
    part_full = crop_map_to_orig(target, part_crop.astype(np.uint8), cv2.INTER_NEAREST).astype(np.uint8)
    return inst_prob_full, inst_full, part_full


def overlay_binary(rgb, mask, color=(0, 220, 255), alpha=0.55):
    out = rgb.copy()
    mask = np.asarray(mask).astype(bool)
    if mask.any():
        out[mask] = np.clip(
            out[mask].astype(np.float32) * (1.0 - float(alpha))
            + np.asarray(color, dtype=np.float32) * float(alpha),
            0,
            255,
        ).astype(np.uint8)
    return out


def draw_crop_box(rgb, target):
    out = rgb.copy()
    x0, y0 = np.round(target["bbox_min"]).astype(int)
    x1, y1 = np.round(target["bbox_max"]).astype(int)
    cv2.rectangle(out, (x0, y0), (x1, y1), (255, 220, 0), 2, cv2.LINE_AA)
    ys, xs = np.where(target["wrist_mask_orig"].astype(bool))
    if len(xs):
        cv2.rectangle(out, (int(xs.min()), int(ys.min())), (int(xs.max()), int(ys.max())), (0, 255, 0), 2, cv2.LINE_AA)
    return out


def draw_crop_keypoints(
    crop_rgb,
    target,
    pred_xy=None,
    pred_scores=None,
    title="keypoints",
    pred_color=(255, 80, 40),
    panel_size=None,
):
    draw_scale = 1.0
    if panel_size is None:
        panel = crop_rgb.copy()
    else:
        panel = cv2.resize(crop_rgb, (int(panel_size), int(panel_size)), interpolation=cv2.INTER_LINEAR)
        draw_scale = float(panel_size) / float(crop_rgb.shape[1])
    gt_xy = target.get("keypoints_crop")
    valid = target.get("keypoints_valid")
    if gt_xy is not None:
        for i, xy in enumerate(np.asarray(gt_xy)):
            if valid is not None and not bool(valid[i]):
                continue
            x, y = np.round(np.asarray(xy) * draw_scale).astype(int)
            if 0 <= x < panel.shape[1] and 0 <= y < panel.shape[0]:
                cv2.circle(panel, (x, y), 4, (50, 240, 80), -1, lineType=cv2.LINE_AA)
                cv2.putText(panel, str(i), (x + 5, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 240, 80), 1, cv2.LINE_AA)
    if pred_xy is not None:
        pred_xy = np.asarray(pred_xy)
        pred_scores = np.ones((len(pred_xy),), dtype=np.float32) if pred_scores is None else np.asarray(pred_scores)
        for i, xy in enumerate(pred_xy):
            if not np.isfinite(xy).all():
                continue
            x, y = np.round(np.asarray(xy) * draw_scale).astype(int)
            if 0 <= x < panel.shape[1] and 0 <= y < panel.shape[0]:
                radius = 5 if float(pred_scores[i]) > 0.01 else 3
                cv2.circle(panel, (x, y), radius, pred_color, 2, lineType=cv2.LINE_AA)
                cv2.putText(panel, str(i), (x + 5, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, pred_color, 1, cv2.LINE_AA)
    return cmp.add_title(panel, title)


def render_pose_or_failed(renderer, rgb, pose, K, title, panel_size, scale, pad_x, pad_y, alpha):
    if pose is None:
        panel = rgb.copy()
        cv2.putText(panel, "failed", (16, 46), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 40, 40), 2, cv2.LINE_AA)
    else:
        panel = renderer.render_pose_overlay(rgb, pose, K, alpha=alpha)
    return cmp.add_title(cmp.pad_rgb_to_square(panel, panel_size, scale, pad_x, pad_y), title)


def concat_panels_grid(panels, cols=4):
    if not panels:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    cols = max(1, int(cols))
    rows = int(math.ceil(len(panels) / float(cols)))
    h = max(panel.shape[0] for panel in panels)
    w = max(panel.shape[1] for panel in panels)
    canvas = np.full((rows * h, cols * w, 3), 255, dtype=np.uint8)
    for i, panel in enumerate(panels):
        y = (i // cols) * h
        x = (i % cols) * w
        canvas[y : y + panel.shape[0], x : x + panel.shape[1]] = panel
    return canvas


def make_visual(target, renderer, poses, row, inst_full, part_full, args, hm_cache=None):
    rgb = target["orig_rgb"]
    panel_size = int(args.panel_size)
    scale, pad_x, pad_y = cmp.square_image_geometry(rgb, panel_size)
    panels = []
    panels.append(cmp.add_title(cmp.pad_rgb_to_square(draw_crop_box(rgb, target), panel_size, scale, pad_x, pad_y), "rgb + wrist/crop bbox"))
    panels.append(
        cmp.add_title(
            cmp.pad_rgb_to_square(overlay_binary(rgb, target["wrist_mask_orig"], color=(0, 255, 0)), panel_size, scale, pad_x, pad_y),
            "gt wrist mask only",
        )
    )
    if target.get("gt_part_orig") is not None:
        panels.append(
            cmp.add_title(
                cmp.pad_rgb_to_square(cmp.overlay_part_mask(rgb, target["gt_part_orig"]), panel_size, scale, pad_x, pad_y),
                "SAM3 GT part seg",
            )
        )
    panels.append(
        cmp.add_title(
            cmp.pad_rgb_to_square(overlay_binary(rgb, inst_full, color=(0, 220, 255)), panel_size, scale, pad_x, pad_y),
            "HCCE inst seg",
        )
    )
    panels.append(
        cmp.add_title(
            cmp.pad_rgb_to_square(cmp.overlay_part_mask(rgb, part_full), panel_size, scale, pad_x, pad_y),
            "HCCE part seg",
        )
    )
    hm_cache = hm_cache or {}
    panels.append(draw_crop_keypoints(target["crop_rgb"], target, None, None, "crop GT keypoints", panel_size=panel_size))
    if "robopepp" in hm_cache:
        xy, scores = hm_cache["robopepp"]
        panels.append(draw_crop_keypoints(target["crop_rgb"], target, xy, scores, "crop RoboPEPP heatmap", (255, 90, 30), panel_size=panel_size))
    if "hcce" in hm_cache:
        xy, scores = hm_cache["hcce"]
        panels.append(draw_crop_keypoints(target["crop_rgb"], target, xy, scores, "crop HCCE heatmap", (40, 130, 255), panel_size=panel_size))
    panels.append(
        render_pose_or_failed(
            renderer,
            rgb,
            target["gt_pose"],
            target["K_orig"],
            "GT repo mesh",
            panel_size,
            scale,
            pad_x,
            pad_y,
            float(args.overlay_alpha),
        )
    )
    for key, title in (
        ("hcce_direct", "HCCE direct"),
        ("hcce_fit", "HCCE fit"),
        ("hcce_kp_pnp", "HCCE kp PnP"),
        ("robopepp_pnp", "RoboPEPP PnP"),
    ):
        rot = row.get(f"{key}_rot_err_deg")
        trans = row.get(f"{key}_trans_err_m")
        title_i = title
        try:
            rot_f, trans_f = float(rot), float(trans)
            if math.isfinite(rot_f) and math.isfinite(trans_f):
                title_i = f"{title} t={trans_f:.3f} r={rot_f:.1f}"
        except (TypeError, ValueError):
            pass
        panels.append(
            render_pose_or_failed(
                renderer,
                rgb,
                poses.get(key),
                target["K_orig"],
                title_i,
                panel_size,
                scale,
                pad_x,
                pad_y,
                float(args.overlay_alpha),
            )
        )
    return concat_panels_grid(panels, cols=int(args.vis_cols))


def evaluate_sample(idx, image_tensor, target, models, renderer, cad, device, args, rng):
    x = image_tensor.unsqueeze(0).to(device, non_blocking=True)
    K_crop_t = torch.from_numpy(target["K_crop"]).unsqueeze(0).to(device)
    row = {
        "dataset": "surgripe_lnd",
        "split": str(target["split"]),
        "frame_id": int(target["frame_id"]),
        "ordinal": int(idx),
        "image_path": target["image_path"],
        "mask_path": target["mask_path"],
        "has_pose_gt": 1,
        "crop_source": str(target.get("crop_source", "wrist_mask")),
        "bbox_min_x": float(target["bbox_min"][0]),
        "bbox_min_y": float(target["bbox_min"][1]),
        "bbox_max_x": float(target["bbox_max"][0]),
        "bbox_max_y": float(target["bbox_max"][1]),
    }
    with torch.inference_mode(), torch.amp.autocast(
        device_type="cuda",
        enabled=(device.type == "cuda"),
        dtype=torch.bfloat16,
    ):
        robo_out = models["robopepp"](x, K_crop_t, masks_enc=None, masks_pred=None)
        hcce_out = models["hcce"](x, K_crop_t)

    fit_args = make_fit_args(args, models["hcce_meta"])
    target_like = target_like_from_dataset(target)
    poses = {}
    hm_cache = {}
    try:
        poses["robopepp_pnp"], robo_hm, robo_scores, _ = cmp.pnp_from_heatmap_output(
            robo_out,
            target["K_crop"],
            float(args.pnp_min_score),
        )
        row["robopepp_pnp_status"] = "ok"
        row["robopepp_hm_score_mean"] = float(np.mean(robo_scores))
        row["robopepp_hm_score_min"] = float(np.min(robo_scores))
        hm_cache["robopepp"] = (robo_hm, robo_scores)
    except Exception as exc:
        poses["robopepp_pnp"] = None
        row["robopepp_pnp_status"] = f"{type(exc).__name__}: {exc}"

    try:
        poses["hcce_direct"] = cmp.pose_from_output(hcce_out)
        row["hcce_direct_status"] = "ok"
    except Exception as exc:
        poses["hcce_direct"] = None
        row["hcce_direct_status"] = f"{type(exc).__name__}: {exc}"

    try:
        poses["hcce_fit"], fit_extra = cmp.fit_pose_from_hcce(
            hcce_out,
            cad,
            models["hcce_meta"],
            target_like,
            fit_args,
            rng,
        )
        row["hcce_fit_status"] = "ok"
        row.update(fit_extra)
    except Exception as exc:
        poses["hcce_fit"] = None
        row["hcce_fit_status"] = f"{type(exc).__name__}: {exc}"
        if int(args.fail_fast):
            raise

    try:
        poses["hcce_kp_pnp"], hcce_hm, hcce_scores, _ = cmp.pnp_from_heatmap_output(
            hcce_out,
            target["K_crop"],
            float(args.pnp_min_score),
        )
        row["hcce_kp_pnp_status"] = "ok"
        row["hcce_kp_hm_score_mean"] = float(np.mean(hcce_scores))
        row["hcce_kp_hm_score_min"] = float(np.min(hcce_scores))
        hm_cache["hcce"] = (hcce_hm, hcce_scores)
    except Exception as exc:
        poses["hcce_kp_pnp"] = None
        row["hcce_kp_pnp_status"] = f"{type(exc).__name__}: {exc}"

    for prefix, cache_key in (("robopepp_pnp", "robopepp"), ("hcce_kp_pnp", "hcce")):
        if cache_key not in hm_cache:
            row[f"{prefix}_hm_crop_rmse_px"] = float("nan")
            continue
        hm, _ = hm_cache[cache_key]
        valid = target["keypoints_valid"]
        if valid is not None and np.any(valid):
            diff = hm[valid] - target["keypoints_crop"][valid]
            row[f"{prefix}_hm_crop_rmse_px"] = float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))
        else:
            row[f"{prefix}_hm_crop_rmse_px"] = float("nan")

    for key, pose in poses.items():
        cmp.add_pose_errors(row, pose, target["gt_pose"], key)
        cmp.add_keypoint_reprojection(row, pose, target_like, key)

    inst_prob_full, inst_full, part_full = decode_hcce_seg_full(hcce_out, target, float(args.inst_thresh))
    row["hcce_inst_area_full"] = int(inst_full.sum())
    row["hcce_inst_prob_mean_in_pred"] = float(inst_prob_full[inst_full].mean()) if inst_full.any() else float("nan")
    row["gt_wrist_area"] = int(target["wrist_mask_orig"].sum())

    if int(args.vis_limit) != 0 and idx < int(args.vis_limit):
        canvas = make_visual(target, renderer, poses, row, inst_full, part_full, args, hm_cache=hm_cache)
        vis_path = Path(args.output_dir) / "vis" / f"frame_{int(target['frame_id']):06d}_compare.jpg"
        vis_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(canvas).save(vis_path)
        row["vis_path"] = str(vis_path)
    else:
        row["vis_path"] = ""
    return row


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lnd_root", type=str, default="/mnt/iMVR/daiyun/Dataset/LND")
    parser.add_argument("--split", type=str, default="TEST")
    parser.add_argument("--output_dir", type=str, default=str(ROBOPEPP_ROOT / "logs/surgripe_lnd_test_compare"))
    parser.add_argument("--robopepp_checkpoint", type=str, default=str(cmp.DEFAULT_ROBOPEPP_CKPT))
    parser.add_argument("--hcce_checkpoint", type=str, default=str(cmp.DEFAULT_HCCE_CKPT))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--frame_ids", nargs="*", type=int, default=None)
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--chunk_rank", type=int, default=0)
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--bbox_scale", type=float, default=1.8)
    parser.add_argument("--min_crop_size", type=int, default=120)
    parser.add_argument("--bbox_margin_px", type=int, default=16)
    parser.add_argument("--crop_source", choices=["gt_mesh", "wrist_mask", "sam3"], default="gt_mesh")
    parser.add_argument("--bbox_padding_frac", type=float, default=0.12)
    parser.add_argument("--mesh_bbox_scale", type=float, default=1.0)
    parser.add_argument("--mesh_min_crop_size", type=int, default=80)
    parser.add_argument("--mesh_bbox_margin_px", type=int, default=2)
    parser.add_argument("--panel_size", type=int, default=540)
    parser.add_argument("--vis_cols", type=int, default=4)
    parser.add_argument("--vis_limit", type=int, default=24)
    parser.add_argument("--inst_thresh", type=float, default=0.5)
    parser.add_argument("--hcce_bits", type=int, default=8)
    parser.add_argument("--hcce_coord_min", type=float, default=-1.0)
    parser.add_argument("--hcce_coord_max", type=float, default=1.0)
    parser.add_argument("--hcce_bit_thresh", type=float, default=0.5)
    parser.add_argument("--max_points_per_part", type=int, default=1200)
    parser.add_argument("--surface_snap_method", choices=["surface", "vertex"], default="surface")
    parser.add_argument("--surface_k_faces", type=int, default=0)
    parser.add_argument("--hcce_axis_scale", type=str, default=None)
    parser.add_argument("--shaft_raw_x_min", type=float, default=-0.5)
    parser.add_argument("--min_wrist_points", type=int, default=12)
    parser.add_argument("--min_total_points", type=int, default=24)
    parser.add_argument("--min_shaft_points", type=int, default=24)
    parser.add_argument("--min_pnp_inliers", type=int, default=8)
    parser.add_argument("--pnp_iters", type=int, default=300)
    parser.add_argument("--pnp_reproj_error", type=float, default=8.0)
    parser.add_argument("--pnp_confidence", type=float, default=0.99)
    parser.add_argument("--pnp_min_score", type=float, default=0.0)
    parser.add_argument("--optim_loss", choices=["linear", "soft_l1", "huber", "cauchy", "arctan"], default="soft_l1")
    parser.add_argument("--optim_f_scale", type=float, default=8.0)
    parser.add_argument("--optim_max_nfev", type=int, default=200)
    parser.add_argument("--hcce_fit_mode", choices=["full", "wrist_only"], default="full")
    parser.add_argument("--gripper_no_wrist_pose", type=int, choices=[0, 1], default=0)
    parser.add_argument("--overlay_alpha", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--print_freq", type=int, default=25)
    parser.add_argument("--fail_fast", type=int, choices=[0, 1], default=0)
    return parser


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = cmp.configure_device(args.device)
    if args.crop_source == "sam3":
        dataset = surgripe_lnd_instrument.RoboPEPPSurgripeLNDInstrument(
            root=args.lnd_root,
            split=args.split,
            training=False,
            crop_size=args.crop_size,
            memory_path=None,
            use_memory_pose=False,
            canonicalize_pose_symmetry=True,
            bbox_padding_frac=float(args.bbox_padding_frac),
            bbox_jitter=False,
            bbox_shift=False,
            frame_ids=args.frame_ids,
        )
    else:
        dataset = surgripe_lnd.SurgripeLNDDataset(
            root=args.lnd_root,
            split=args.split,
            crop_size=args.crop_size,
            bbox_scale=args.bbox_scale,
            min_crop_size=args.min_crop_size,
            margin_px=args.bbox_margin_px,
            frame_ids=args.frame_ids,
        )
    indices = list(range(len(dataset)))
    if int(args.max_frames) > 0:
        indices = indices[: int(args.max_frames)]
    if int(args.num_chunks) > 1:
        if not (0 <= int(args.chunk_rank) < int(args.num_chunks)):
            raise ValueError(f"chunk_rank must be in [0, num_chunks), got {args.chunk_rank}/{args.num_chunks}")
        indices = indices[int(args.chunk_rank) :: int(args.num_chunks)]
    print(f"[config] output_dir={output_dir}", flush=True)
    print(f"[config] split={args.split} selected_frames={len(indices)} dataset_len={len(dataset)}", flush=True)
    print(f"[config] chunk_rank={args.chunk_rank} num_chunks={args.num_chunks}", flush=True)
    print(f"[config] robopepp_checkpoint={args.robopepp_checkpoint}", flush=True)
    print(f"[config] hcce_checkpoint={args.hcce_checkpoint}", flush=True)

    robo_model, _ = cmp.load_robopepp_model(Path(args.robopepp_checkpoint), device)
    hcce_model, hcce_meta = cmp.load_hcce_model(Path(args.hcce_checkpoint), device)
    renderer = GMSInstrumentTrimeshRenderer(device)
    cad = cmp.InstrumentCAD(cmp.CAD_ROOT)
    models = {"robopepp": robo_model, "hcce": hcce_model, "hcce_meta": hcce_meta}
    rng = np.random.default_rng(int(args.seed))
    rows = []
    for n, idx in enumerate(indices):
        try:
            image_tensor, target = dataset[idx]
            if args.crop_source == "sam3":
                target = target_like_from_instrument_dataset(target, int(args.crop_size))
            elif args.crop_source == "gt_mesh":
                image_tensor, target = recrop_with_gt_mesh(renderer, target, dataset.to_tensor, args)
            row = evaluate_sample(n, image_tensor, target, models, renderer, cad, device, args, rng)
        except Exception as exc:
            row = {
                "dataset": "surgripe_lnd",
                "split": args.split,
                "ordinal": n,
                "status": f"{type(exc).__name__}: {exc}",
            }
            print(f"[error] idx={idx}: {row['status']}", flush=True)
            if int(args.fail_fast):
                raise
        rows.append(row)
        if n == 0 or (n + 1) % int(args.print_freq) == 0 or n + 1 == len(indices):
            print(f"[progress] {n + 1}/{len(indices)}", flush=True)

    csv_path = output_dir / "per_frame.csv"
    write_csv(csv_path, rows)
    summary = {
        "num_rows": len(rows),
        "status_counts": status_counts(rows),
        "metrics": {
            key: stat(rows, key)
            for key in (
                "robopepp_pnp_trans_err_m",
                "robopepp_pnp_rot_err_deg",
                "hcce_direct_trans_err_m",
                "hcce_direct_rot_err_deg",
                "hcce_fit_trans_err_m",
                "hcce_fit_rot_err_deg",
                "hcce_kp_pnp_trans_err_m",
                "hcce_kp_pnp_rot_err_deg",
            )
        },
        "csv": str(csv_path),
        "vis_dir": str(output_dir / "vis"),
        "note": "LND GT provides wrist SE3 only; articulation GT is set to zero for mesh sanity and joint error fields are not meaningful.",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")
    lines = [
        "# Surgripe LND TEST Compare",
        "",
        f"- rows: {len(rows)}",
        f"- csv: `{csv_path}`",
        f"- vis: `{output_dir / 'vis'}`",
        "- note: LND GT has wrist SE3 only; articulation GT is set to zero for GT mesh sanity.",
        "",
        "## Status",
        json.dumps(summary["status_counts"], indent=2, sort_keys=True),
        "",
        "## Metrics",
    ]
    for key, value in summary["metrics"].items():
        lines.append(f"- {key}: mean={value['mean']:.6g}, median={value['median']:.6g}, rmse={value['rmse']:.6g}, n={value['count']}")
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[done] csv={csv_path}", flush=True)
    print(f"[done] summary={output_dir / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main(build_parser().parse_args())
