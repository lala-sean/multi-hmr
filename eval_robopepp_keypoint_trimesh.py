import argparse
import csv
import importlib.util
import json
import math
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
import torchvision.transforms as tv_transforms
from PIL import Image

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

ROBOPEPP_ROOT = Path(__file__).resolve().parent
MULTIHMR_ROOT = ROBOPEPP_ROOT.parents[1]
if str(ROBOPEPP_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBOPEPP_ROOT))
if str(MULTIHMR_ROOT) not in sys.path:
    sys.path.insert(0, str(MULTIHMR_ROOT))

import compare_crop_hcce_robopepp_rarp as cmp  # noqa: E402
from instrument_geometry import (  # noqa: E402
    KEYPOINT_NAMES,
    crop_resize_pad_intrinsics,
    instrument_keypoints_camera_np,
    project_points_np,
    rarp_intrinsics,
)
from instrument_trimesh_renderer import GMSInstrumentTrimeshRenderer  # noqa: E402
from pose_pnp import pose_from_keypoints_pnp  # noqa: E402
from predict_instrument_pose import (  # noqa: E402
    add_title,
    concat_panels,
    heatmap_argmax,
    load_model as load_robopepp_model,
    overlay_part_mask,
    pad_rgb_to_square,
    square_image_geometry,
)


DEFAULT_CKPT = ROBOPEPP_ROOT / "logs/robopepp_instrument_pose_rarp_lnd_refinemem_bs56_gpu1567/checkpoints/last.pt"
DEFAULT_OUT = ROBOPEPP_ROOT / "logs/robopepp_rarp_lnd_refinemem_eval_keypoint_trimesh"
DEFAULT_SUTURE_REF_VIS = (
    MULTIHMR_ROOT
    / "eval_outputs/suturepulling_best55000_ref_current_best_three_exp_full_stride4/densepart_h2/vis/suturePulling"
)


class Pose:
    def __setstate__(self, state):
        self.__dict__.update(state if isinstance(state, dict) else {})


def load_local_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rarp_module = load_local_module("eval_rarp_instrument", ROBOPEPP_ROOT / "datasets" / "rarp_instrument.py")
lnd_module = load_local_module("eval_lnd_instrument", ROBOPEPP_ROOT / "datasets" / "surgripe_lnd_instrument.py")
RoboPEPPRARPInstrument = rarp_module.RoboPEPPRARPInstrument
RoboPEPPSurgripeLNDInstrument = lnd_module.RoboPEPPSurgripeLNDInstrument


def norm_frame_id(frame_id):
    return f"{int(frame_id):05d}"


def read_csv_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = []
    seen = set()
    preferred = [
        "dataset",
        "video",
        "frame_id",
        "instance_id",
        "ordinal",
        "status",
        "pnp_status",
        "direct_status",
        "has_pose_gt",
        "vis_path",
    ]
    for key in preferred:
        if any(key in row for row in rows):
            keys.append(key)
            seen.add(key)
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def finite_values(rows, key):
    vals = []
    for row in rows:
        try:
            val = float(row.get(key, float("nan")))
        except (TypeError, ValueError):
            val = float("nan")
        if math.isfinite(val):
            vals.append(val)
    return np.asarray(vals, dtype=np.float64)


def stat(rows, key):
    vals = finite_values(rows, key)
    if vals.size == 0:
        return {"count": 0, "mean": float("nan"), "median": float("nan"), "std": float("nan"), "rmse": float("nan")}
    return {
        "count": int(vals.size),
        "mean": float(vals.mean()),
        "median": float(np.median(vals)),
        "std": float(vals.std()),
        "rmse": float(np.sqrt(np.mean(vals * vals))),
    }


def status_counts(rows, key):
    out = {}
    for row in rows:
        value = str(row.get(key, "missing"))
        out[value] = out.get(value, 0) + 1
    return out


def crop_points_to_original(points_crop, target_like):
    points = np.asarray(points_crop, dtype=np.float64).copy()
    scale = np.asarray(target_like["scale"], dtype=np.float64).reshape(2)
    pad = np.asarray(target_like["pad"], dtype=np.float64).reshape(2)
    bbox_min = np.asarray(target_like["bbox_min"], dtype=np.float64).reshape(2)
    points[:, 0] = (points[:, 0] - pad[0]) / scale[0] + bbox_min[0]
    points[:, 1] = (points[:, 1] - pad[1]) / scale[1] + bbox_min[1]
    return points.astype(np.float32)


def pose_action_array(pose):
    return np.asarray([pose["alpha"], pose["theta_l"], pose["theta_r"]], dtype=np.float64)


def pose_from_target_tensor(target):
    return cmp.pose_from_target(target)


def tensor_target_like(target):
    out = {
        "orig_rgb": target["orig_rgb"].detach().cpu().numpy().astype(np.uint8),
        "crop_rgb": target["crop_rgb"].detach().cpu().numpy().astype(np.uint8),
        "gt_part_orig": target["part_mask_orig"].detach().cpu().numpy().astype(np.uint8),
        "K_orig": target["K_orig"].detach().cpu().numpy().astype(np.float32),
        "K_crop": target["K"].detach().cpu().numpy().astype(np.float32),
        "bbox_min": target["bbox_min"].detach().cpu().numpy().astype(np.float32),
        "bbox_max": target["bbox_max"].detach().cpu().numpy().astype(np.float32),
        "scale": target["scale"].detach().cpu().numpy().astype(np.float32),
        "pad": target["pad"].detach().cpu().numpy().astype(np.float32),
        "keypoints_crop": target["keypoints_crop"].detach().cpu().numpy().astype(np.float32),
        "keypoints_orig": target["keypoints_orig"].detach().cpu().numpy().astype(np.float32),
        "keypoints_valid": target["keypoints_valid"].detach().cpu().numpy().astype(bool),
        "keypoints_valid_orig": target["keypoints_valid_orig"].detach().cpu().numpy().astype(bool),
    }
    out["gt_part_crop"] = crop_resize_pad_map(
        out["gt_part_orig"],
        out["bbox_min"],
        out["bbox_max"],
        out["crop_rgb"].shape[0],
        cv2.INTER_NEAREST,
        value=0,
    )
    return out


def resize_longer_side(arr, crop_size, interpolation):
    h, w = arr.shape[:2]
    if w > h:
        new_w = int(crop_size)
        new_h = max(1, int(crop_size * h / w))
    else:
        new_h = int(crop_size)
        new_w = max(1, int(crop_size * w / h))
    return cv2.resize(arr, (new_w, new_h), interpolation=interpolation), (new_w, new_h)


def pad_2d(arr, crop_size, value=0):
    h, w = arr.shape[:2]
    pad_h = (int(crop_size) - h) // 2
    pad_w = (int(crop_size) - w) // 2
    if arr.ndim == 2:
        padding = ((pad_h, int(crop_size) - h - pad_h), (pad_w, int(crop_size) - w - pad_w))
    else:
        padding = ((pad_h, int(crop_size) - h - pad_h), (pad_w, int(crop_size) - w - pad_w), (0, 0))
    if arr.ndim == 3 and value == "edge":
        return np.pad(arr, padding, mode="edge"), (pad_w, pad_h)
    return np.pad(arr, padding, mode="constant", constant_values=value), (pad_w, pad_h)


def crop_resize_pad_map(arr, bbox_min, bbox_max, crop_size, interpolation, value=0):
    x0, y0 = np.asarray(bbox_min, dtype=np.float32).astype(np.int64)
    x1, y1 = np.ceil(np.asarray(bbox_max, dtype=np.float32)).astype(np.int64)
    crop = arr[y0:y1, x0:x1]
    resized, _ = resize_longer_side(crop, crop_size, interpolation)
    square, _ = pad_2d(resized, crop_size, value=value)
    return square


def bbox_from_mask(mask, width, height, padding_frac):
    ys, xs = np.where(mask)
    if xs.size == 0:
        raise RuntimeError("empty mask bbox")
    bbox_min = np.array([float(xs.min()), float(ys.min())], dtype=np.float32)
    bbox_max = np.array([float(xs.max() + 1), float(ys.max() + 1)], dtype=np.float32)
    side = float(max(bbox_max[0] - bbox_min[0], bbox_max[1] - bbox_min[1]))
    pad = np.array([side * float(padding_frac), side * float(padding_frac)], dtype=np.float32)
    bbox_min = bbox_min - pad
    bbox_max = bbox_max + pad
    bbox_min = np.clip(bbox_min, [0.0, 0.0], [float(width - 1), float(height - 1)])
    bbox_max = np.clip(bbox_max, [1.0, 1.0], [float(width), float(height)])
    return bbox_min, bbox_max


def crop_from_part_mask(rgb, K_orig, gt_part, crop_size, padding_frac):
    h, w = rgb.shape[:2]
    bbox_min, bbox_max = bbox_from_mask(gt_part > 0, w, h, padding_frac)
    x0, y0 = bbox_min.astype(np.int64)
    x1, y1 = np.ceil(bbox_max).astype(np.int64)
    crop = rgb[y0:y1, x0:x1]
    crop_resized, (new_w, new_h) = resize_longer_side(crop, crop_size, cv2.INTER_LINEAR)
    crop_square, (pad_w, pad_h) = pad_2d(crop_resized, crop_size, value="edge")
    scale_x = float(new_w) / float(bbox_max[0] - bbox_min[0])
    scale_y = float(new_h) / float(bbox_max[1] - bbox_min[1])
    K_crop = crop_resize_pad_intrinsics(K_orig, bbox_min, (scale_x, scale_y), (pad_w, pad_h))
    gt_part_crop = crop_resize_pad_map(gt_part, bbox_min, bbox_max, crop_size, cv2.INTER_NEAREST, value=0).astype(np.uint8)
    return {
        "orig_rgb": rgb.astype(np.uint8),
        "crop_rgb": crop_square.astype(np.uint8),
        "gt_part_orig": gt_part.astype(np.uint8),
        "gt_part_crop": gt_part_crop,
        "K_orig": K_orig.astype(np.float32),
        "K_crop": K_crop.astype(np.float32),
        "bbox_min": bbox_min.astype(np.float32),
        "bbox_max": bbox_max.astype(np.float32),
        "scale": np.array([scale_x, scale_y], dtype=np.float32),
        "pad": np.array([pad_w, pad_h], dtype=np.float32),
        "keypoints_crop": None,
        "keypoints_orig": None,
        "keypoints_valid": None,
        "keypoints_valid_orig": None,
    }


def load_suture_reference_items(args):
    vis_dir = Path(args.suture_ref_vis_dir)
    pattern = re.compile(r"^(suturePulling_\d+_video\d+)_(\d+)_inst(\d+)\.jpg$")
    by_video = {}
    for path in sorted(vis_dir.glob("*.jpg")):
        match = pattern.match(path.name)
        if not match:
            continue
        video, frame_id, inst = match.groups()
        by_video.setdefault(video, []).append((norm_frame_id(frame_id), int(inst)))
    if args.suture_video:
        video = args.suture_video
    else:
        video = sorted(by_video)[0]
    frame_inst = sorted(set(by_video.get(video, [])), key=lambda x: (int(x[0]), x[1]))
    if int(args.max_suture_samples) > 0:
        frame_inst = frame_inst[: int(args.max_suture_samples)]
    dataset, index = cmp.build_suture_dataset_index(args)
    rows = []
    for i, (frame_id, inst) in enumerate(frame_inst):
        actual_instance_id = int(inst)
        try:
            _, annot = dataset[index[(video, frame_id)]]
            ref_slot = int(inst) - 1
            actual_instance_id = int(annot["instruments"][ref_slot]["instance_id"])
        except Exception:
            actual_instance_id = int(inst)
        rows.append(
            {
                "dataset": "suturePulling",
                "video": video,
                "frame_id": frame_id,
                "instance_id": inst,
                "actual_instance_id": actual_instance_id,
                "ordinal": i,
            }
        )
    return rows, {"video": video, "num_items": len(rows), "ref_vis_dir": str(vis_dir)}


def build_needle_dataset(args):
    return RoboPEPPRARPInstrument(
        args.needle_dataset_root,
        args.needle_pose_root,
        split="test",
        training=False,
        crop_size=int(args.crop_size),
        train_ratio=float(args.train_ratio),
        subsample=1,
        min_dice=(args.min_dice_shaft, args.min_dice_wrist, args.min_dice_gripper),
        canonicalize_pose_symmetry=bool(args.canonicalize_pose_symmetry),
        canonical_eps=float(args.canonical_eps),
        bbox_padding_frac=float(args.bbox_padding_frac),
        cache_dir=args.dataset_cache_dir,
    )


def build_lnd_dataset(args):
    return RoboPEPPSurgripeLNDInstrument(
        root=args.lnd_root,
        split="TEST",
        training=False,
        crop_size=int(args.crop_size),
        memory_path=None,
        use_memory_pose=False,
        canonicalize_pose_symmetry=bool(args.canonicalize_pose_symmetry),
        canonical_eps=float(args.canonical_eps),
        heatmap_sigma=2.0,
        bbox_padding_frac=float(args.bbox_padding_frac),
        color_jitter=False,
        rgb_augmentation=False,
        occlusion_augmentation=False,
        occlusion_prob=0.0,
    )


def build_items(args):
    items = []
    meta = {}
    if "needleGrasping" in args.datasets:
        ds = build_needle_dataset(args)
        for idx, (video, frame_id, inst, _) in enumerate(ds.samples):
            if int(args.max_needle_samples) > 0 and len([x for x in items if x["dataset"] == "needleGrasping"]) >= int(args.max_needle_samples):
                break
            items.append(
                {
                    "dataset": "needleGrasping",
                    "video": video,
                    "frame_id": norm_frame_id(frame_id),
                    "instance_id": int(inst),
                    "ordinal": idx,
                    "dataset_idx": idx,
                }
            )
        meta["needleGrasping"] = {"dataset_len": len(ds), "selected_items": sum(x["dataset"] == "needleGrasping" for x in items)}
    if "suturePulling" in args.datasets:
        suture_items, suture_meta = load_suture_reference_items(args)
        items.extend(suture_items)
        meta["suturePulling"] = suture_meta
    if "surgripe_lnd" in args.datasets:
        ds = build_lnd_dataset(args)
        max_n = len(ds) if int(args.max_lnd_samples) <= 0 else min(len(ds), int(args.max_lnd_samples))
        base_ord = 0
        for idx in range(max_n):
            frame_id = int(ds.samples[idx][0])
            items.append(
                {
                    "dataset": "surgripe_lnd",
                    "video": "surgripe_lnd_TEST",
                    "frame_id": str(frame_id),
                    "instance_id": 0,
                    "ordinal": base_ord + idx,
                    "dataset_idx": idx,
                }
            )
        meta["surgripe_lnd"] = {"dataset_len": len(ds), "selected_items": max_n}
    return items, meta


def draw_points(rgb, points, valid, color, marker="circle", prefix=""):
    out = rgb.copy()
    if points is None:
        return out
    if valid is None:
        valid = np.ones((len(points),), dtype=bool)
    for i, (xy, ok) in enumerate(zip(points, valid)):
        if not bool(ok) or not np.isfinite(xy).all():
            continue
        x, y = np.round(xy).astype(int)
        if not (0 <= x < out.shape[1] and 0 <= y < out.shape[0]):
            continue
        if marker == "cross":
            cv2.drawMarker(out, (x, y), color, cv2.MARKER_CROSS, 15, 2, cv2.LINE_AA)
        else:
            cv2.circle(out, (x, y), 5, color, -1, lineType=cv2.LINE_AA)
        cv2.putText(out, f"{prefix}{i}", (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    return out


def render_or_fail(renderer, rgb, pose, K, title, panel_size, scale, pad_x, pad_y, alpha):
    if pose is None:
        panel = rgb.copy()
        cv2.putText(panel, "failed", (16, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 40, 40), 2, cv2.LINE_AA)
    else:
        panel = renderer.render_pose_overlay(rgb, pose, K, alpha=alpha)
    return add_title(pad_rgb_to_square(panel, panel_size, scale, pad_x, pad_y), title)


def make_visual(target_like, poses, pred, renderer, row, args, gt_pose=None):
    rgb = target_like["orig_rgb"]
    panel_size = int(args.panel_size)
    scale, pad_x, pad_y = square_image_geometry(rgb, panel_size)
    rgb_sq = pad_rgb_to_square(rgb, panel_size, scale, pad_x, pad_y)
    seg_sq = cmp.pad_mask_to_square(target_like["gt_part_orig"], panel_size, scale, pad_x, pad_y)
    panels = [add_title(rgb_sq, "rgb"), add_title(overlay_part_mask(rgb_sq, seg_sq), "gt/SAM part seg")]

    valid_orig = target_like.get("keypoints_valid_orig")
    valid_crop = target_like.get("keypoints_valid")
    pnp_valid_orig = pred.get("pnp_fk_valid")
    if pnp_valid_orig is not None and valid_orig is not None:
        pnp_valid_orig = np.asarray(pnp_valid_orig, dtype=bool) & np.asarray(valid_orig, dtype=bool)
    pnp_valid_crop = pred.get("pnp_fk_valid")
    if pnp_valid_crop is not None and valid_crop is not None:
        pnp_valid_crop = np.asarray(pnp_valid_crop, dtype=bool) & np.asarray(valid_crop, dtype=bool)

    kp_orig = rgb.copy()
    kp_orig = draw_points(kp_orig, target_like.get("keypoints_orig"), valid_orig, (40, 220, 70), prefix="g")
    kp_orig = draw_points(kp_orig, pred["hm_orig"], valid_orig, (255, 70, 220), marker="cross", prefix="h")
    kp_orig = draw_points(kp_orig, pred.get("pnp_fk_orig"), pnp_valid_orig, (60, 210, 255), marker="circle", prefix="p")
    panels.append(add_title(pad_rgb_to_square(kp_orig, panel_size, scale, pad_x, pad_y), "keypoints: gt/hm/pnpFK"))

    if gt_pose is not None:
        panels.append(render_or_fail(renderer, rgb, gt_pose, target_like["K_orig"], "GT trimesh", panel_size, scale, pad_x, pad_y, args.overlay_alpha))
    panels.append(render_or_fail(renderer, rgb, poses.get("pnp"), target_like["K_orig"], "RoboPEPP keypoint-PnP trimesh", panel_size, scale, pad_x, pad_y, args.overlay_alpha))
    panels.append(render_or_fail(renderer, rgb, poses.get("direct"), target_like["K_orig"], "RoboPEPP direct trimesh", panel_size, scale, pad_x, pad_y, args.overlay_alpha))

    crop = target_like["crop_rgb"]
    crop_panel = crop.copy()
    crop_panel = draw_points(crop_panel, target_like.get("keypoints_crop"), valid_crop, (40, 220, 70), prefix="g")
    crop_panel = draw_points(crop_panel, pred["hm_crop"], valid_crop, (255, 70, 220), marker="cross", prefix="h")
    crop_panel = draw_points(crop_panel, pred.get("pnp_fk_crop"), pnp_valid_crop, (60, 210, 255), prefix="p")
    crop_panel = cv2.resize(crop_panel, (panel_size, panel_size), interpolation=cv2.INTER_LINEAR)
    panels.append(add_title(crop_panel, "crop keypoints"))
    return concat_panels(panels)


def add_pose_metrics(row, prefix, pred_pose, gt_pose, lnd_units=False):
    if pred_pose is None or gt_pose is None:
        return
    trans_m = float(np.linalg.norm(np.asarray(pred_pose["trans"], dtype=np.float64) - np.asarray(gt_pose["trans"], dtype=np.float64)))
    rot_deg = cmp.rotation_error_deg(pred_pose["rot"], gt_pose["rot"])
    row[f"{prefix}_trans_err_m"] = trans_m
    row[f"{prefix}_rot_err_deg"] = rot_deg
    row[f"{prefix}_trans_err_mm"] = trans_m * 1000.0
    if not lnd_units:
        gt_action = pose_action_array(gt_pose)
        pred_action = pose_action_array(pred_pose)
        err_deg = np.degrees(np.abs(pred_action - gt_action))
        row[f"{prefix}_joint_mae_deg"] = float(err_deg.mean())
        row[f"{prefix}_alpha_err_deg"] = float(err_deg[0])
        row[f"{prefix}_theta_l_err_deg"] = float(err_deg[1])
        row[f"{prefix}_theta_r_err_deg"] = float(err_deg[2])


def add_keypoint_metrics(row, target_like, pred):
    valid = target_like.get("keypoints_valid")
    if valid is None or target_like.get("keypoints_crop") is None or not np.any(valid):
        return
    hm_crop = pred["hm_crop"]
    hm_orig = pred["hm_orig"]
    gt_crop = target_like["keypoints_crop"]
    gt_orig = target_like["keypoints_orig"]
    row["hm_crop_rmse_px"] = float(np.sqrt(np.mean(np.sum((hm_crop[valid] - gt_crop[valid]) ** 2, axis=1))))
    row["hm_orig_rmse_px"] = float(np.sqrt(np.mean(np.sum((hm_orig[valid] - gt_orig[valid]) ** 2, axis=1))))
    if pred.get("pnp_fk_orig") is not None:
        row["pnp_fk_orig_rmse_px"] = float(np.sqrt(np.mean(np.sum((pred["pnp_fk_orig"][valid] - gt_orig[valid]) ** 2, axis=1))))
    row["visible_keypoints"] = int(np.count_nonzero(valid))


def predict_one(model, image_tensor, target_like, device, args):
    x = image_tensor.unsqueeze(0).to(device, non_blocking=True)
    K_crop = torch.from_numpy(target_like["K_crop"]).unsqueeze(0).to(device, non_blocking=True)
    with torch.inference_mode(), torch.amp.autocast(device_type="cuda", enabled=device.type == "cuda", dtype=torch.bfloat16):
        out = model(x, K_crop, masks_enc=None, masks_pred=None)
    direct_pose = cmp.pose_from_output(out)
    hm_crop_t, scores_t = heatmap_argmax(out["keypoint_heatmaps"].float().detach().cpu())
    hm_crop = hm_crop_t[0].numpy().astype(np.float32)
    scores = scores_t[0].numpy().astype(np.float32)
    try:
        pnp_pose = pose_from_keypoints_pnp(
            hm_crop,
            pose_action_array(direct_pose),
            target_like["K_crop"],
            scores=scores,
            min_score=float(args.pnp_min_score),
        )
        pnp_status = "ok"
    except Exception as exc:
        pnp_pose = None
        pnp_status = f"{type(exc).__name__}: {exc}"

    pred = {"hm_crop": hm_crop, "hm_scores": scores, "hm_orig": crop_points_to_original(hm_crop, target_like)}
    if pnp_pose is not None:
        kp3d = instrument_keypoints_camera_np(pnp_pose["rot"], pnp_pose["trans"], pose_action_array(pnp_pose))
        pred["pnp_fk_crop"] = project_points_np(kp3d, target_like["K_crop"]).astype(np.float32)
        pred["pnp_fk_orig"] = project_points_np(kp3d, target_like["K_orig"]).astype(np.float32)
        pred["pnp_fk_valid"] = kp3d[:, 2] > 1e-4
    return {"direct": direct_pose, "pnp": pnp_pose}, pred, pnp_status


def suture_target_like(item, args, to_tensor):
    rgb, _ = cmp.load_frame_rgb(args.suture_dataset_root, item["video"], item["frame_id"])
    actual_instance_id = int(item.get("actual_instance_id", item["instance_id"]))
    gt_part = cmp.load_suture_part_mask(args.suture_dataset_root, item["video"], item["frame_id"], actual_instance_id)
    K_orig = rarp_intrinsics(rgb.shape[1], rgb.shape[0])
    target_like = crop_from_part_mask(rgb, K_orig, gt_part, int(args.crop_size), float(args.bbox_padding_frac))
    image_tensor = to_tensor(Image.fromarray(target_like["crop_rgb"]))
    return image_tensor, target_like


def worker_main(rank, items, args):
    device = cmp.configure_device(args.devices[rank])
    print(f"[worker {rank}] device={device} items={len(items)}", flush=True)
    model, _ = load_robopepp_model(Path(args.checkpoint), device)
    renderer = GMSInstrumentTrimeshRenderer(device)
    to_tensor = tv_transforms.Compose(
        [
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    datasets = {}
    if any(item["dataset"] == "needleGrasping" for item in items):
        datasets["needleGrasping"] = build_needle_dataset(args)
    if any(item["dataset"] == "surgripe_lnd" for item in items):
        datasets["surgripe_lnd"] = build_lnd_dataset(args)

    rows = []
    for local_idx, item in enumerate(items):
        row = dict(item)
        try:
            dataset_name = item["dataset"]
            gt_pose = None
            if dataset_name == "needleGrasping":
                image_tensor, target = datasets["needleGrasping"][int(item["dataset_idx"])]
                target_like = tensor_target_like(target)
                gt_pose = pose_from_target_tensor(target)
                row["has_pose_gt"] = 1
            elif dataset_name == "surgripe_lnd":
                image_tensor, target = datasets["surgripe_lnd"][int(item["dataset_idx"])]
                target_like = tensor_target_like(target)
                gt_pose = pose_from_target_tensor(target)
                row["has_pose_gt"] = 1
            elif dataset_name == "suturePulling":
                image_tensor, target_like = suture_target_like(item, args, to_tensor)
                row["has_pose_gt"] = 0
            else:
                raise ValueError(dataset_name)

            poses, pred, pnp_status = predict_one(model, image_tensor, target_like, device, args)
            row["status"] = "ok"
            row["pnp_status"] = pnp_status
            row["direct_status"] = "ok"
            row["hm_score_mean"] = float(np.mean(pred["hm_scores"]))
            row["hm_score_min"] = float(np.min(pred["hm_scores"]))
            add_keypoint_metrics(row, target_like, pred)
            is_lnd = dataset_name == "surgripe_lnd"
            add_pose_metrics(row, "direct", poses.get("direct"), gt_pose, lnd_units=is_lnd)
            add_pose_metrics(row, "pnp", poses.get("pnp"), gt_pose, lnd_units=is_lnd)

            should_vis = (
                (dataset_name == "needleGrasping" and int(item["ordinal"]) < int(args.needle_vis_limit))
                or (dataset_name == "suturePulling" and int(item["ordinal"]) < int(args.suture_vis_limit))
                or (dataset_name == "surgripe_lnd" and int(item["ordinal"]) < int(args.lnd_vis_limit))
            )
            if should_vis:
                canvas = make_visual(target_like, poses, pred, renderer, row, args, gt_pose=gt_pose)
                stem = f"{int(item['ordinal']):05d}_{item['video']}_{norm_frame_id(item['frame_id'])}_inst{item['instance_id']}.jpg"
                vis_path = Path(args.output_dir) / "vis" / dataset_name / stem
                vis_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(canvas).save(vis_path)
                row["vis_path"] = str(vis_path)
            else:
                row["vis_path"] = ""
        except Exception as exc:
            row["status"] = f"{type(exc).__name__}: {exc}"
            row["vis_path"] = ""
            print(f"[worker {rank}] ERROR {item.get('dataset')}/{item.get('video')}/{item.get('frame_id')}: {row['status']}", flush=True)
            if int(args.fail_fast):
                raise
        rows.append(row)
        if local_idx == 0 or (local_idx + 1) % int(args.print_freq) == 0 or local_idx + 1 == len(items):
            print(f"[worker {rank}] {local_idx + 1}/{len(items)}", flush=True)
    out = Path(args.output_dir) / "workers" / f"worker_{rank:02d}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, allow_nan=True), encoding="utf-8")


def split_even(items, n):
    return [items[i::n] for i in range(n)]


def make_contact_sheet(vis_dir, out_path, max_images=24):
    paths = sorted(Path(vis_dir).glob("*.jpg"))[: int(max_images)]
    if not paths:
        return ""
    imgs = [Image.open(p).convert("RGB") for p in paths]
    thumb_w = 900
    thumbs = []
    for img, path in zip(imgs, paths):
        scale = thumb_w / float(img.width)
        thumb = img.resize((thumb_w, max(1, int(img.height * scale))))
        label_h = 26
        canvas = Image.new("RGB", (thumb.width, thumb.height + label_h), "white")
        canvas.paste(thumb, (0, label_h))
        cv2_img = np.asarray(canvas).copy()
        cv2.putText(cv2_img, path.name[:120], (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)
        thumbs.append(Image.fromarray(cv2_img))
    sheet = Image.new("RGB", (thumb_w, sum(t.height for t in thumbs)), "white")
    y = 0
    for thumb in thumbs:
        sheet.paste(thumb, (0, y))
        y += thumb.height
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92)
    return str(out_path)


def summarize(rows):
    by_dataset = {}
    for dataset in sorted({row.get("dataset", "") for row in rows}):
        active = [row for row in rows if row.get("dataset") == dataset]
        keys = [
            "hm_crop_rmse_px",
            "hm_orig_rmse_px",
            "pnp_fk_orig_rmse_px",
            "direct_trans_err_m",
            "direct_trans_err_mm",
            "direct_rot_err_deg",
            "direct_joint_mae_deg",
            "pnp_trans_err_m",
            "pnp_trans_err_mm",
            "pnp_rot_err_deg",
            "pnp_joint_mae_deg",
        ]
        by_dataset[dataset] = {
            "num_rows": len(active),
            "status_counts": status_counts(active, "status"),
            "pnp_status_counts": status_counts(active, "pnp_status"),
            "metrics": {key: stat(active, key) for key in keys if stat(active, key)["count"] > 0},
        }
    return by_dataset


def run(args):
    args.output_dir = str(Path(args.output_dir).resolve())
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    items, meta = build_items(args)
    if not items:
        raise RuntimeError("No eval items")
    manifest = {"items": items, "selection_meta": meta, "args": vars(args)}
    (Path(args.output_dir) / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=True), encoding="utf-8")
    print(f"[manifest] output_dir={args.output_dir}", flush=True)
    print(f"[manifest] items={len(items)} meta={json.dumps(meta, indent=2)}", flush=True)

    chunks = [chunk for chunk in split_even(items, len(args.devices)) if chunk]
    args.devices = list(args.devices)[: len(chunks)]
    if len(chunks) == 1:
        worker_main(0, chunks[0], args)
    else:
        ctx = mp.get_context("spawn")
        procs = []
        for rank, chunk in enumerate(chunks):
            p = ctx.Process(target=worker_main, args=(rank, chunk, args))
            p.start()
            procs.append(p)
        failures = []
        for p in procs:
            p.join()
            if p.exitcode != 0:
                failures.append(p.exitcode)
        if failures:
            raise RuntimeError(f"worker failures: {failures}")

    rows = []
    for path in sorted((Path(args.output_dir) / "workers").glob("worker_*.json")):
        rows.extend(json.loads(path.read_text(encoding="utf-8")))
    rows.sort(key=lambda r: (str(r.get("dataset", "")), int(r.get("ordinal", 10**9)), str(r.get("frame_id", ""))))
    csv_path = Path(args.output_dir) / "per_instance.csv"
    write_csv(csv_path, rows)
    summary = summarize(rows)
    contact_sheets = {}
    for dataset in summary:
        sheet = make_contact_sheet(Path(args.output_dir) / "vis" / dataset, Path(args.output_dir) / "vis" / f"{dataset}_contact_sheet.jpg")
        if sheet:
            contact_sheets[dataset] = sheet
    summary["contact_sheets"] = contact_sheets
    summary["csv"] = str(csv_path)
    (Path(args.output_dir) / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")
    write_summary_md(Path(args.output_dir) / "summary.md", summary)
    print(f"[done] csv={csv_path}", flush=True)
    print(f"[done] summary={Path(args.output_dir) / 'summary.md'}", flush=True)


def fmt_stat(item):
    if not item:
        return "n/a"
    return f"mean={item['mean']:.6g}, median={item['median']:.6g}, rmse={item['rmse']:.6g}, n={item['count']}"


def write_summary_md(path, summary):
    lines = ["# RoboPEPP Keypoint + Trimesh Evaluation", ""]
    for dataset, item in summary.items():
        if dataset in ("contact_sheets", "csv"):
            continue
        lines += [f"## {dataset}", f"- rows: {item['num_rows']}", f"- status: `{item['status_counts']}`", ""]
        metrics = item.get("metrics", {})
        if dataset == "surgripe_lnd":
            lines += [
                "- units for SurgRIPE_LND: translation is mm, rotation is degree.",
                f"- direct wrist trans mm: {fmt_stat(metrics.get('direct_trans_err_mm'))}",
                f"- direct wrist rot deg: {fmt_stat(metrics.get('direct_rot_err_deg'))}",
                f"- keypoint-PnP wrist trans mm: {fmt_stat(metrics.get('pnp_trans_err_mm'))}",
                f"- keypoint-PnP wrist rot deg: {fmt_stat(metrics.get('pnp_rot_err_deg'))}",
            ]
        else:
            lines += [
                f"- direct trans m: {fmt_stat(metrics.get('direct_trans_err_m'))}",
                f"- direct rot deg: {fmt_stat(metrics.get('direct_rot_err_deg'))}",
                f"- direct joint MAE deg: {fmt_stat(metrics.get('direct_joint_mae_deg'))}",
                f"- keypoint-PnP trans m: {fmt_stat(metrics.get('pnp_trans_err_m'))}",
                f"- keypoint-PnP rot deg: {fmt_stat(metrics.get('pnp_rot_err_deg'))}",
                f"- keypoint-PnP joint MAE deg: {fmt_stat(metrics.get('pnp_joint_mae_deg'))}",
                f"- heatmap crop RMSE px: {fmt_stat(metrics.get('hm_crop_rmse_px'))}",
            ]
        lines.append("")
    lines += ["## Outputs", f"- csv: `{summary.get('csv', '')}`"]
    for dataset, sheet in summary.get("contact_sheets", {}).items():
        lines.append(f"- {dataset} contact sheet: `{sheet}`")
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CKPT))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUT))
    parser.add_argument("--datasets", nargs="+", choices=["needleGrasping", "suturePulling", "surgripe_lnd"], default=["needleGrasping", "suturePulling", "surgripe_lnd"])
    parser.add_argument("--devices", nargs="+", default=["cuda:0"])
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--bbox_padding_frac", type=float, default=0.12)
    parser.add_argument("--panel_size", type=int, default=540)
    parser.add_argument("--overlay_alpha", type=float, default=0.85)
    parser.add_argument("--pnp_min_score", type=float, default=0.0)
    parser.add_argument("--needle_dataset_root", type=str, default="/mnt/nas/share/shuojue/data/needleGrasping_videos")
    parser.add_argument("--needle_pose_root", type=str, default="/mnt/nas/share/shuojue/data/needleGrasping_results")
    parser.add_argument("--suture_dataset_root", type=str, default="/mnt/nas/share/shuojue/data/suturePulling_videos")
    parser.add_argument("--suture_ref_vis_dir", type=str, default=str(DEFAULT_SUTURE_REF_VIS))
    parser.add_argument("--suture_video", type=str, default="")
    parser.add_argument("--lnd_root", type=str, default="/mnt/iMVR/daiyun/Dataset/LND")
    parser.add_argument("--dataset_cache_dir", type=str, default=str(DEFAULT_OUT / "dataset_cache"))
    parser.add_argument("--train_ratio", type=float, default=0.95)
    parser.add_argument("--canonicalize_pose_symmetry", type=int, choices=[0, 1], default=1)
    parser.add_argument("--canonical_eps", type=float, default=0.08)
    parser.add_argument("--min_dice_shaft", type=float, default=0.8)
    parser.add_argument("--min_dice_wrist", type=float, default=0.6)
    parser.add_argument("--min_dice_gripper", type=float, default=0.6)
    parser.add_argument("--max_needle_samples", type=int, default=0)
    parser.add_argument("--max_suture_samples", type=int, default=0)
    parser.add_argument("--max_lnd_samples", type=int, default=0)
    parser.add_argument("--needle_vis_limit", type=int, default=80)
    parser.add_argument("--suture_vis_limit", type=int, default=80)
    parser.add_argument("--lnd_vis_limit", type=int, default=80)
    parser.add_argument("--print_freq", type=int, default=50)
    parser.add_argument("--fail_fast", type=int, choices=[0, 1], default=0)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
