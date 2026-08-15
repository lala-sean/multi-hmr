import argparse
import csv
import importlib.util
import json
import math
import os
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
from PIL import Image
import torchvision.transforms as tv_transforms

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

ROBOPEPP_ROOT = Path(__file__).resolve().parent
MULTIHMR_ROOT = ROBOPEPP_ROOT.parents[1]
if str(ROBOPEPP_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBOPEPP_ROOT))
if str(MULTIHMR_ROOT) not in sys.path:
    sys.path.insert(0, str(MULTIHMR_ROOT))

datasets_pkg = types.ModuleType("datasets")
datasets_pkg.__path__ = [str(MULTIHMR_ROOT / "datasets")]
sys.modules["datasets"] = datasets_pkg

from datasets.RarpInstanceDataset import RARPInstanceDataset, _resolve_mask_subfolder  # noqa: E402
from instrument_geometry import (  # noqa: E402
    KEYPOINT_NAMES,
    crop_resize_pad_intrinsics,
    instrument_keypoints_camera_np,
    project_points_np,
    quat_wxyz_to_matrix_np,
    rarp_intrinsics,
)
from instrument_trimesh_renderer import GMSInstrumentTrimeshRenderer  # noqa: E402
from pose_pnp import pose_from_keypoints_pnp  # noqa: E402
from predict_instrument_pose import (  # noqa: E402
    DATASETS,
    add_title,
    concat_panels,
    heatmap_argmax,
    load_model,
    overlay_part_mask,
    pad_mask_to_square,
    pad_rgb_to_square,
    square_image_geometry,
)


PART_LABELS = {"gripper": 1, "wrist": 2, "shaft": 3}
DEFAULT_ROBOPEPP_CKPT = ROBOPEPP_ROOT / "logs/robopepp_instrument_pose_rarp_jepa_bs56_gpu0123/checkpoints/last.pt"
DEFAULT_NEEDLE_HCCE_CSV = MULTIHMR_ROOT / "eval_outputs/best_iter55000_parallel_meshtexturefix/needleGrasping/per_instance.csv"
DEFAULT_SUTURE_HCCE_CSV = (
    MULTIHMR_ROOT
    / "eval_outputs/suturepulling_best55000_ref_current_best_three_exp_full_stride4/densepart_h2/per_instance.csv"
)
DEFAULT_SUTURE_ROOT = Path("/mnt/nas/share/shuojue/data/suturePulling_videos")


class Pose:
    def __setstate__(self, state):
        self.__dict__.update(state if isinstance(state, dict) else {})


def load_robopepp_rarp_dataset_module():
    path = ROBOPEPP_ROOT / "datasets" / "rarp_instrument.py"
    spec = importlib.util.spec_from_file_location("robopepp_quant_rarp_instrument", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RoboPEPPRARPInstrument = load_robopepp_rarp_dataset_module().RoboPEPPRARPInstrument


def resolve_path(path, base=ROBOPEPP_ROOT):
    path = Path(path)
    if path.is_absolute() or path.exists():
        return path
    return base / path


def norm_frame_id(frame_id):
    return f"{int(frame_id):05d}"


def sample_key(video, frame_id, instance_id):
    return str(video), norm_frame_id(frame_id), int(instance_id)


def read_csv_rows(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def float_or_nan(value):
    try:
        if value is None or value == "":
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def int_or_none(value):
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def binary_iou(pred, gt):
    pred = np.asarray(pred).astype(bool)
    gt = np.asarray(gt).astype(bool)
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return float("nan")
    return float(np.logical_and(pred, gt).sum() / union)


def binary_dice(pred, gt):
    pred = np.asarray(pred).astype(bool)
    gt = np.asarray(gt).astype(bool)
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return float("nan")
    return float(2.0 * np.logical_and(pred, gt).sum() / denom)


def segmentation_metrics_from_masks(pred_part, gt_part, prefix):
    pred_part = np.asarray(pred_part, dtype=np.int64)
    gt_part = np.asarray(gt_part, dtype=np.int64)
    pred_inst = pred_part > 0
    gt_inst = gt_part > 0
    metrics = {
        f"{prefix}_inst_iou": binary_iou(pred_inst, gt_inst),
        f"{prefix}_inst_dice": binary_dice(pred_inst, gt_inst),
        f"{prefix}_pred_inst_area": int(pred_inst.sum()),
        f"{prefix}_gt_inst_area": int(gt_inst.sum()),
    }
    part_ious = []
    part_dices = []
    for part_name, label in PART_LABELS.items():
        pred_part_i = pred_inst & (pred_part == label)
        gt_part_i = gt_inst & (gt_part == label)
        iou = binary_iou(pred_part_i, gt_part_i)
        dice = binary_dice(pred_part_i, gt_part_i)
        metrics[f"{prefix}_part_iou_{part_name}"] = iou
        metrics[f"{prefix}_part_dice_{part_name}"] = dice
        if not math.isnan(iou):
            part_ious.append(iou)
        if not math.isnan(dice):
            part_dices.append(dice)
    metrics[f"{prefix}_part_iou_mean"] = float(np.mean(part_ious)) if part_ious else float("nan")
    metrics[f"{prefix}_part_dice_mean"] = float(np.mean(part_dices)) if part_dices else float("nan")
    return metrics


def rotation_error_deg(q_pred, q_gt):
    r_pred = quat_wxyz_to_matrix_np(q_pred)
    r_gt = quat_wxyz_to_matrix_np(q_gt)
    rel = r_pred @ r_gt.T
    trace = float(np.trace(rel))
    cos_angle = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def pose_from_output(out):
    action = out["action_pred"][0].detach().cpu().numpy().astype(np.float64)
    quat = out["wrist_quat_pred"][0].detach().cpu().numpy().astype(np.float64)
    trans = out["wrist_trans_pred"][0].detach().cpu().numpy().astype(np.float64)
    return {
        "rot": quat,
        "trans": trans,
        "alpha": float(action[0]),
        "theta_l": float(action[1]),
        "theta_r": float(action[2]),
    }


def pose_from_target(target):
    return {
        "rot": target["wrist_quat"].numpy().astype(np.float64),
        "trans": target["wrist_trans"].numpy().astype(np.float64),
        "alpha": float(target["action"][0].item()),
        "theta_l": float(target["action"][1].item()),
        "theta_r": float(target["action"][2].item()),
    }


def crop_points_to_original(points_crop, bbox_min, scale_xy, pad_xy):
    points = np.asarray(points_crop, dtype=np.float64).copy()
    scale = np.asarray(scale_xy, dtype=np.float64).reshape(2)
    pad = np.asarray(pad_xy, dtype=np.float64).reshape(2)
    bbox_min = np.asarray(bbox_min, dtype=np.float64).reshape(2)
    points[:, 0] = (points[:, 0] - pad[0]) / scale[0] + bbox_min[0]
    points[:, 1] = (points[:, 1] - pad[1]) / scale[1] + bbox_min[1]
    return points.astype(np.float32)


def pnp_pose_from_model_output(out, K_crop, args):
    pred_hm_crop_t, hm_scores_t = heatmap_argmax(out["keypoint_heatmaps"].float().detach().cpu())
    pred_hm_crop = pred_hm_crop_t[0].numpy().astype(np.float32)
    hm_scores = hm_scores_t[0].numpy().astype(np.float32)
    direct_pose = pose_from_output(out)
    if args.pose_recovery == "pnp":
        pose = pose_from_keypoints_pnp(
            pred_hm_crop,
            [direct_pose["alpha"], direct_pose["theta_l"], direct_pose["theta_r"]],
            np.asarray(K_crop, dtype=np.float32),
            scores=hm_scores,
            min_score=args.pnp_min_score,
        )
    else:
        pose = direct_pose
    return pose, direct_pose, pred_hm_crop, hm_scores


def load_frame_rgb(dataset_root, video, frame_id, v2_force=True):
    video_folder = Path(dataset_root) / f"SARRARP502022_{video}"
    frames_folder = video_folder / "frames_v2" if v2_force and (video_folder / "frames_v2").is_dir() else video_folder / "frames"
    for ext in ("png", "jpg"):
        path = frames_folder / f"{norm_frame_id(frame_id)}.{ext}"
        if path.is_file():
            return np.asarray(Image.open(path).convert("RGB")), path
    raise FileNotFoundError(f"Frame not found: {video}/{frame_id} under {frames_folder}")


def load_suture_instance_masks(dataset_root, video, frame_id, actual_instance_id):
    video_folder = Path(dataset_root) / f"SARRARP502022_{video}"
    instance_folder = video_folder / f"instance{actual_instance_id}"
    mask_frame_id = f"{int(frame_id) - 1:05d}"
    part_mask = None
    for part_name, label in (("shaft", 3), ("wrist", 2), ("gripper", 1)):
        folder = _resolve_mask_subfolder(str(instance_folder), part_name, True)
        if folder is None:
            raise FileNotFoundError(f"{part_name} mask folder not found under {instance_folder}")
        path = Path(folder) / f"{mask_frame_id}.png"
        if not path.is_file():
            raise FileNotFoundError(path)
        mask = np.asarray(Image.open(path).convert("L")) > 0
        if part_mask is None:
            part_mask = np.zeros(mask.shape, dtype=np.uint8)
        part_mask[mask] = int(label)
    if part_mask is None or not np.any(part_mask > 0):
        raise RuntimeError(f"Empty suture mask: {video}/{frame_id}/instance{actual_instance_id}")
    return part_mask


def crop_from_gt_mask(rgb, K_orig, gt_part_mask, crop_size):
    inst_mask = gt_part_mask > 0
    ys, xs = np.where(inst_mask)
    if len(xs) == 0:
        raise RuntimeError("Cannot crop an empty GT mask")
    h, w = rgb.shape[:2]
    bbox_min = np.array([float(xs.min()), float(ys.min())], dtype=np.float32)
    bbox_max = np.array([float(xs.max() + 1), float(ys.max() + 1)], dtype=np.float32)
    bbox_min = np.clip(bbox_min, [0.0, 0.0], [float(w - 1), float(h - 1)])
    bbox_max = np.clip(bbox_max, [1.0, 1.0], [float(w), float(h)])
    x0, y0 = bbox_min.astype(np.int64)
    x1, y1 = np.ceil(bbox_max).astype(np.int64)
    crop = rgb[y0:y1, x0:x1]
    crop_h, crop_w = crop.shape[:2]
    if crop_h <= 0 or crop_w <= 0:
        raise RuntimeError(f"Empty crop from bbox {bbox_min} -> {bbox_max}")
    if crop_w > crop_h:
        new_w = int(crop_size)
        new_h = int(crop_size * crop_h / crop_w)
    else:
        new_h = int(crop_size)
        new_w = int(crop_size * crop_w / crop_h)
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_x = (crop_size - new_w) // 2
    pad_y = (crop_size - new_h) // 2
    crop_square = np.pad(
        resized,
        ((pad_y, crop_size - new_h - pad_y), (pad_x, crop_size - new_w - pad_x), (0, 0)),
        mode="edge",
    )
    scale_x = float(new_w) / float(bbox_max[0] - bbox_min[0])
    scale_y = float(new_h) / float(bbox_max[1] - bbox_min[1])
    K_crop = crop_resize_pad_intrinsics(K_orig, bbox_min=bbox_min, scale_xy=(scale_x, scale_y), pad_xy=(pad_x, pad_y))
    return {
        "crop_rgb": crop_square.astype(np.uint8),
        "K_crop": K_crop.astype(np.float32),
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
        "scale": np.array([scale_x, scale_y], dtype=np.float32),
        "pad": np.array([pad_x, pad_y], dtype=np.float32),
    }


def make_quant_visual(rgb, gt_part_mask, pred_part_mask, pred_overlay, panel_size):
    scale, pad_x, pad_y = square_image_geometry(rgb, panel_size)
    rgb_sq = pad_rgb_to_square(rgb, panel_size, scale, pad_x, pad_y)
    gt_mask_sq = pad_mask_to_square(gt_part_mask, panel_size, scale, pad_x, pad_y)
    pred_mask_sq = pad_mask_to_square(pred_part_mask, panel_size, scale, pad_x, pad_y)
    pred_overlay_sq = pad_rgb_to_square(pred_overlay, panel_size, scale, pad_x, pad_y)
    panels = [
        add_title(rgb_sq, "rgb"),
        add_title(overlay_part_mask(rgb_sq, gt_mask_sq), "gt-part-mask"),
        add_title(pred_overlay_sq, "robopepp-pose-trimesh"),
        add_title(overlay_part_mask(rgb_sq, pred_mask_sq), "robopepp-pose-seg"),
    ]
    return concat_panels(panels)


def tensor_to_image_input(crop_rgb, to_tensor, device):
    return to_tensor(Image.fromarray(crop_rgb.astype(np.uint8))).unsqueeze(0).to(device, non_blocking=True)


def evaluate_needle_item(item, dataset, model, renderer, device, to_tensor, args):
    image, target = dataset[item["dataset_idx"]]
    x = image.unsqueeze(0).to(device, non_blocking=True)
    K_crop = target["K"].unsqueeze(0).to(device, non_blocking=True)
    with torch.inference_mode(), torch.amp.autocast(
        device_type="cuda", enabled=(device.type == "cuda"), dtype=torch.bfloat16
    ):
        out = model(x, K_crop, masks_enc=None, masks_pred=None)
    pred_pose, direct_pose, pred_hm_crop, hm_scores = pnp_pose_from_model_output(out, target["K"].numpy(), args)
    gt_pose = pose_from_target(target)

    action_pred = np.array([pred_pose["alpha"], pred_pose["theta_l"], pred_pose["theta_r"]], dtype=np.float64)
    action_gt = target["action"].numpy().astype(np.float64)
    pred_kp_3d = instrument_keypoints_camera_np(pred_pose["rot"], pred_pose["trans"], action_pred)
    gt_kp_3d = target["keypoints_3d_cam"].numpy().astype(np.float64)
    pred_pose_orig = project_points_np(pred_kp_3d, target["K_orig"].numpy()).astype(np.float32)
    pred_hm_orig = crop_points_to_original(pred_hm_crop, target["bbox_min"].numpy(), target["scale"].numpy(), target["pad"].numpy())
    gt_orig = target["keypoints_orig"].numpy().astype(np.float32)
    gt_crop = target["keypoints_crop"].numpy().astype(np.float32)
    valid = target["keypoints_valid"].numpy().astype(bool)
    valid_orig = target["keypoints_valid_orig"].numpy().astype(bool)

    row = {
        "dataset": "needleGrasping",
        "video": target["video_name"],
        "frame_id": norm_frame_id(target["frame_id"]),
        "instance_id": int(target["instance_id"].item()),
        "sample_idx": int(item["dataset_idx"]),
        "ordinal": int(item["ordinal"]),
        "robopepp_status": "ok",
        "robopepp_pose_recovery": args.pose_recovery,
        "robopepp_hm_score_mean": float(np.mean(hm_scores)),
        "robopepp_hm_score_min": float(np.min(hm_scores)),
        "robopepp_pnp_used_keypoints": int(np.sum(np.isfinite(pred_hm_crop).all(axis=1) & (hm_scores >= args.pnp_min_score))),
        "robopepp_visible_keypoints": int(valid.sum()),
        "robopepp_visible_keypoints_orig": int(valid_orig.sum()),
        "robopepp_trans_err_m": float(np.linalg.norm(np.asarray(pred_pose["trans"]) - np.asarray(gt_pose["trans"]))),
        "robopepp_rot_err_deg": rotation_error_deg(pred_pose["rot"], gt_pose["rot"]),
        "robopepp_action_rmse_rad": float(np.sqrt(np.mean((action_pred - action_gt) ** 2))),
        "robopepp_alpha_err_rad": float(abs(action_pred[0] - action_gt[0])),
        "robopepp_theta_l_err_rad": float(abs(action_pred[1] - action_gt[1])),
        "robopepp_theta_r_err_rad": float(abs(action_pred[2] - action_gt[2])),
        "robopepp_keypoint_3d_rmse_m": float(np.sqrt(np.mean(np.sum((pred_kp_3d - gt_kp_3d) ** 2, axis=1)))),
    }
    if valid.any():
        row["robopepp_hm_crop_rmse_px"] = float(np.sqrt(np.mean(np.sum((pred_hm_crop[valid] - gt_crop[valid]) ** 2, axis=1))))
        row["robopepp_hm_orig_rmse_px"] = float(np.sqrt(np.mean(np.sum((pred_hm_orig[valid] - gt_orig[valid]) ** 2, axis=1))))
        row["robopepp_reproj_rmse_px"] = float(np.sqrt(np.mean(np.sum((pred_pose_orig[valid] - gt_orig[valid]) ** 2, axis=1))))
    else:
        row["robopepp_hm_crop_rmse_px"] = float("nan")
        row["robopepp_hm_orig_rmse_px"] = float("nan")
        row["robopepp_reproj_rmse_px"] = float("nan")

    rgb = target["orig_rgb"].numpy()
    gt_part = target["part_mask_orig"].numpy().astype(np.uint8)
    pred_part = renderer.render_pose_mask(
        pred_pose,
        target["K_orig"].numpy(),
        rgb.shape[:2],
        min_depth=args.render_min_depth,
        draw_margin=args.render_draw_margin,
    )
    row.update(segmentation_metrics_from_masks(pred_part, gt_part, "robopepp_render_seg"))

    if int(args.vis_limit) > 0 and int(item["ordinal"]) < int(args.vis_limit):
        pred_overlay = renderer.render_pose_overlay(rgb, pred_pose, target["K_orig"].numpy(), alpha=args.overlay_alpha)
        canvas = make_quant_visual(rgb, gt_part, pred_part, pred_overlay, int(args.panel_size))
        out_path = Path(args.output_dir) / "needleGrasping" / "vis" / f"{target['video_name']}_{norm_frame_id(target['frame_id'])}_inst{int(target['instance_id'].item())}.jpg"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(canvas).save(out_path)
        row["robopepp_vis_path"] = str(out_path)

    attach_hcce_metrics(row, item.get("hcce_row"))
    return row


def build_suture_dataset_index(args):
    dataset = RARPInstanceDataset(
        split="test",
        training=False,
        img_size=int(args.panel_size),
        dataset_root=str(args.suture_dataset_root),
        pose_root=None,
        min_dice=[args.min_dice_shaft, args.min_dice_wrist, args.min_dice_gripper],
        train_ratio=args.needle_train_ratio,
        subsample=1,
        v2_force=True,
        cse_coord_root=None,
        render_on_the_fly=False,
    )
    return dataset, {(video, norm_frame_id(frame_id)): idx for idx, (video, frame_id, _) in enumerate(dataset.samples)}


def evaluate_suture_item(item, dataset, dataset_index, model, renderer, device, to_tensor, args):
    key = (item["video"], norm_frame_id(item["frame_id"]))
    if key not in dataset_index:
        raise KeyError(f"Suture frame not in RARPInstanceDataset index: {key}")
    _, annot = dataset[dataset_index[key]]
    ref_slot = int(item["instance_id"]) - 1
    instruments = annot["instruments"]
    if ref_slot < 0 or ref_slot >= len(instruments):
        raise RuntimeError(f"instance slot {item['instance_id']} out of range for {key}: {len(instruments)} instruments")
    actual_instance_id = int(instruments[ref_slot]["instance_id"])
    rgb, _ = load_frame_rgb(args.suture_dataset_root, item["video"], item["frame_id"], v2_force=True)
    gt_part = load_suture_instance_masks(args.suture_dataset_root, item["video"], item["frame_id"], actual_instance_id)
    K_orig = rarp_intrinsics(rgb.shape[1], rgb.shape[0])
    crop_info = crop_from_gt_mask(rgb, K_orig, gt_part, int(args.crop_size))
    x = tensor_to_image_input(crop_info["crop_rgb"], to_tensor, device)
    K_crop_t = torch.from_numpy(crop_info["K_crop"]).unsqueeze(0).to(device, non_blocking=True)
    with torch.inference_mode(), torch.amp.autocast(
        device_type="cuda", enabled=(device.type == "cuda"), dtype=torch.bfloat16
    ):
        out = model(x, K_crop_t, masks_enc=None, masks_pred=None)
    pred_pose, _, pred_hm_crop, hm_scores = pnp_pose_from_model_output(out, crop_info["K_crop"], args)
    pred_kp_3d = instrument_keypoints_camera_np(
        pred_pose["rot"],
        pred_pose["trans"],
        [pred_pose["alpha"], pred_pose["theta_l"], pred_pose["theta_r"]],
    )
    pred_pose_orig = project_points_np(pred_kp_3d, K_orig).astype(np.float32)
    pred_hm_orig = crop_points_to_original(pred_hm_crop, crop_info["bbox_min"], crop_info["scale"], crop_info["pad"])
    pred_part = renderer.render_pose_mask(
        pred_pose,
        K_orig,
        rgb.shape[:2],
        min_depth=args.render_min_depth,
        draw_margin=args.render_draw_margin,
    )
    row = {
        "dataset": "suturePulling",
        "video": item["video"],
        "frame_id": norm_frame_id(item["frame_id"]),
        "instance_id": int(item["instance_id"]),
        "actual_instance_id": actual_instance_id,
        "sample_idx": int_or_none(item.get("sample_idx")),
        "ordinal": int(item["ordinal"]),
        "robopepp_status": "ok",
        "robopepp_pose_recovery": args.pose_recovery,
        "robopepp_hm_score_mean": float(np.mean(hm_scores)),
        "robopepp_hm_score_min": float(np.min(hm_scores)),
        "robopepp_pnp_used_keypoints": int(np.sum(np.isfinite(pred_hm_crop).all(axis=1) & (hm_scores >= args.pnp_min_score))),
        "robopepp_pred_kp_orig_json": json.dumps(pred_pose_orig.round(3).tolist()),
        "robopepp_hm_kp_orig_json": json.dumps(pred_hm_orig.round(3).tolist()),
        "robopepp_alpha": float(pred_pose["alpha"]),
        "robopepp_theta_l": float(pred_pose["theta_l"]),
        "robopepp_theta_r": float(pred_pose["theta_r"]),
        "robopepp_trans_x": float(pred_pose["trans"][0]),
        "robopepp_trans_y": float(pred_pose["trans"][1]),
        "robopepp_trans_z": float(pred_pose["trans"][2]),
    }
    row.update(segmentation_metrics_from_masks(pred_part, gt_part, "robopepp_render_seg"))

    if int(args.vis_limit) > 0 and int(item["ordinal"]) < int(args.vis_limit):
        pred_overlay = renderer.render_pose_overlay(rgb, pred_pose, K_orig, alpha=args.overlay_alpha)
        canvas = make_quant_visual(rgb, gt_part, pred_part, pred_overlay, int(args.panel_size))
        out_path = Path(args.output_dir) / "suturePulling" / "vis" / f"{item['video']}_{norm_frame_id(item['frame_id'])}_inst{int(item['instance_id'])}.jpg"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(canvas).save(out_path)
        row["robopepp_vis_path"] = str(out_path)

    attach_hcce_metrics(row, item.get("hcce_row"))
    return row


def attach_hcce_metrics(row, hcce_row):
    if not hcce_row:
        return
    keep = (
        "hcce_status",
        "pose_head_status",
        "hcce_trans_err_m",
        "hcce_rot_err_deg",
        "hcce_action_mse",
        "hcce_alpha_err_rad",
        "hcce_theta_l_err_rad",
        "hcce_theta_r_err_rad",
        "hcce_reproj_rmse_px",
        "hcce_reproj_rmse_shaft_px",
        "hcce_reproj_rmse_wrist_gripper_px",
        "hcce_render_seg_inst_iou",
        "hcce_render_seg_part_iou_mean",
        "hcce_render_seg_part_iou_gripper",
        "hcce_render_seg_part_iou_wrist",
        "hcce_render_seg_part_iou_shaft",
        "model_seg_inst_iou",
        "model_seg_part_iou_mean",
        "model_seg_part_iou_gripper",
        "model_seg_part_iou_wrist",
        "model_seg_part_iou_shaft",
        "posehead_trans_err_m",
        "posehead_rot_err_deg",
        "posehead_render_seg_inst_iou",
        "posehead_render_seg_part_iou_mean",
        "opt_rmse_all",
        "opt_rmse_shaft",
        "opt_rmse_wrist_gripper",
        "match_dist_px",
        "pnp_inliers",
        "total_points",
        "vis_path",
    )
    for key in keep:
        if key not in hcce_row:
            continue
        out_key = key if key.startswith(("hcce_", "model_seg_", "posehead_")) else f"hcce_{key}"
        value = hcce_row.get(key)
        if key.endswith("status") or key == "vis_path":
            row[out_key] = value
        else:
            row[out_key] = float_or_nan(value)


def configure_device(device):
    if str(device).startswith("cuda"):
        idx = int(str(device).split(":", 1)[1]) if ":" in str(device) else 0
        os.environ["EGL_DEVICE_ID"] = str(idx)
        torch.cuda.set_device(idx)
    return torch.device(device if torch.cuda.is_available() else "cpu")


def build_needle_items(args):
    cfg = DATASETS["needleGrasping"]
    dataset = RoboPEPPRARPInstrument(
        cfg["dataset_root"],
        cfg["pose_root"],
        split="test",
        training=False,
        crop_size=int(args.crop_size),
        train_ratio=float(args.needle_train_ratio),
        subsample=1,
        min_dice=(args.min_dice_shaft, args.min_dice_wrist, args.min_dice_gripper),
        canonicalize_pose_symmetry=bool(args.canonicalize_pose_symmetry),
        canonical_eps=float(args.canonical_eps),
        cache_dir=args.dataset_cache_dir,
    )
    index = {sample_key(v, f, inst): i for i, (v, f, inst, _) in enumerate(dataset.samples)}
    hcce_rows = read_csv_rows(args.needle_hcce_csv)
    items = []
    missing_hcce_keys = []
    for row in hcce_rows:
        key = sample_key(row["video"], row["frame_id"], row["instance_id"])
        idx = index.get(key)
        if idx is None:
            missing_hcce_keys.append(key)
            continue
        items.append({"dataset_idx": idx, "hcce_row": row})
    if int(args.max_samples) > 0:
        items = items[: int(args.max_samples)]
    for ordinal, item in enumerate(items):
        item["ordinal"] = ordinal
    meta = {
        "dataset_len": len(dataset),
        "hcce_rows": len(hcce_rows),
        "matched_items": len(items),
        "missing_hcce_keys": len(missing_hcce_keys),
        "missing_hcce_key_examples": ["|".join(map(str, key)) for key in missing_hcce_keys[:10]],
    }
    return dataset, items, meta


def build_suture_items(args):
    hcce_rows = read_csv_rows(args.suture_hcce_csv)
    videos = sorted({row["video"] for row in hcce_rows})
    if int(args.suture_num_videos) > 0:
        videos = videos[: int(args.suture_num_videos)]
    video_set = set(videos)
    items = []
    for row in hcce_rows:
        frame_id = int(row["frame_id"])
        if row["video"] not in video_set:
            continue
        if int(args.suture_frame_stride) > 1 and frame_id % int(args.suture_frame_stride) != 0:
            continue
        items.append(
            {
                "video": row["video"],
                "frame_id": norm_frame_id(frame_id),
                "instance_id": int(row["instance_id"]),
                "sample_idx": row.get("sample_idx", ""),
                "hcce_row": row,
            }
        )
    if int(args.max_samples) > 0:
        items = items[: int(args.max_samples)]
    for ordinal, item in enumerate(items):
        item["ordinal"] = ordinal
    meta = {
        "hcce_rows": len(hcce_rows),
        "selected_videos": videos,
        "selected_video_count": len(videos),
        "matched_items": len(items),
        "frame_stride": int(args.suture_frame_stride),
    }
    return items, meta


def split_evenly(items, n):
    return [items[i::n] for i in range(n)]


def worker_main(rank, dataset_name, items, args):
    device = configure_device(args.devices[rank])
    checkpoint = resolve_path(args.checkpoint)
    model, _ = load_model(checkpoint, device)
    renderer = GMSInstrumentTrimeshRenderer(device)
    to_tensor = tv_transforms.Compose(
        [
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    if dataset_name == "needleGrasping":
        dataset, _, _ = build_needle_items(args)
        dataset_index = None
    elif dataset_name == "suturePulling":
        dataset, dataset_index = build_suture_dataset_index(args)
    else:
        raise ValueError(dataset_name)

    rows = []
    for local_i, item in enumerate(items):
        try:
            if dataset_name == "needleGrasping":
                row = evaluate_needle_item(item, dataset, model, renderer, device, to_tensor, args)
            else:
                row = evaluate_suture_item(item, dataset, dataset_index, model, renderer, device, to_tensor, args)
            rows.append(row)
            if local_i == 0 or (local_i + 1) % int(args.print_freq) == 0 or local_i + 1 == len(items):
                print(f"[{dataset_name} worker {rank}] {local_i + 1}/{len(items)} ok", flush=True)
        except Exception as exc:
            row = {
                "dataset": dataset_name,
                "video": item.get("video", ""),
                "frame_id": norm_frame_id(item.get("frame_id", 0)) if item.get("frame_id", "") != "" else "",
                "instance_id": item.get("instance_id", ""),
                "sample_idx": item.get("dataset_idx", item.get("sample_idx", "")),
                "ordinal": item.get("ordinal", ""),
                "robopepp_status": "error",
                "robopepp_error": f"{type(exc).__name__}: {exc}",
            }
            attach_hcce_metrics(row, item.get("hcce_row"))
            rows.append(row)
            print(f"[{dataset_name} worker {rank}] error: {row['robopepp_error']}", flush=True)
            if int(args.fail_fast):
                raise

    manifest = Path(args.output_dir) / dataset_name / f"worker_{rank:02d}.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(rows, indent=2, allow_nan=True), encoding="utf-8")


def run_workers(dataset_name, items, args):
    args.devices = args.devices or [args.device]
    chunks = [chunk for chunk in split_evenly(items, len(args.devices)) if chunk]
    args.devices = args.devices[: len(chunks)]
    if not chunks:
        raise RuntimeError(f"No {dataset_name} items selected")
    print(f"[{dataset_name}] evaluating {len(items)} item(s) on {args.devices}", flush=True)
    if len(chunks) == 1:
        worker_main(0, dataset_name, chunks[0], args)
    else:
        ctx = mp.get_context("spawn")
        procs = []
        for rank, chunk in enumerate(chunks):
            proc = ctx.Process(target=worker_main, args=(rank, dataset_name, chunk, args))
            proc.start()
            procs.append(proc)
        failures = []
        for proc in procs:
            proc.join()
            if proc.exitcode != 0:
                failures.append(proc.exitcode)
        if failures:
            raise RuntimeError(f"{dataset_name} worker failures: {failures}")
    rows = []
    for path in sorted((Path(args.output_dir) / dataset_name).glob("worker_*.json")):
        rows.extend(json.loads(path.read_text(encoding="utf-8")))
    rows.sort(key=lambda row: int(row["ordinal"]) if str(row.get("ordinal", "")).isdigit() else 10**12)
    write_rows(Path(args.output_dir) / dataset_name / "per_instance.csv", rows)
    summary = summarize_rows(dataset_name, rows)
    (Path(args.output_dir) / dataset_name / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    write_summary_md(Path(args.output_dir) / dataset_name / "summary.md", summary)
    return rows, summary


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}", flush=True)


def numeric_values(rows, key):
    vals = []
    for row in rows:
        if key not in row:
            continue
        value = float_or_nan(row[key])
        if math.isfinite(value):
            vals.append(value)
    return np.asarray(vals, dtype=np.float64)


def stats_for(rows, key):
    vals = numeric_values(rows, key)
    if len(vals) == 0:
        return {"count": 0, "mean": float("nan"), "median": float("nan"), "std": float("nan"), "rmse": float("nan")}
    return {
        "count": int(len(vals)),
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "std": float(np.std(vals)),
        "rmse": float(np.sqrt(np.mean(vals**2))),
    }


def summarize_rows(dataset_name, rows):
    ok_rows = [row for row in rows if row.get("robopepp_status") == "ok"]
    failed = [row for row in rows if row.get("robopepp_status") != "ok"]
    base = {
        "dataset": dataset_name,
        "num_rows": len(rows),
        "num_robopepp_ok": len(ok_rows),
        "num_robopepp_failed": len(failed),
        "videos": sorted({row.get("video", "") for row in rows if row.get("video", "")}),
        "robopepp_failures": [
            {
                "video": row.get("video", ""),
                "frame_id": row.get("frame_id", ""),
                "instance_id": row.get("instance_id", ""),
                "error": row.get("robopepp_error", ""),
            }
            for row in failed[:25]
        ],
        "metrics": {},
    }
    metric_keys = [
        "robopepp_trans_err_m",
        "robopepp_rot_err_deg",
        "robopepp_action_rmse_rad",
        "robopepp_alpha_err_rad",
        "robopepp_theta_l_err_rad",
        "robopepp_theta_r_err_rad",
        "robopepp_keypoint_3d_rmse_m",
        "robopepp_hm_crop_rmse_px",
        "robopepp_hm_orig_rmse_px",
        "robopepp_reproj_rmse_px",
        "robopepp_render_seg_inst_iou",
        "robopepp_render_seg_part_iou_mean",
        "robopepp_render_seg_inst_dice",
        "robopepp_render_seg_part_dice_mean",
        "hcce_trans_err_m",
        "hcce_rot_err_deg",
        "hcce_action_mse",
        "hcce_reproj_rmse_px",
        "hcce_render_seg_inst_iou",
        "hcce_render_seg_part_iou_mean",
        "model_seg_inst_iou",
        "model_seg_part_iou_mean",
        "posehead_trans_err_m",
        "posehead_rot_err_deg",
        "posehead_render_seg_inst_iou",
        "posehead_render_seg_part_iou_mean",
        "hcce_opt_rmse_all",
        "hcce_opt_rmse_shaft",
        "hcce_opt_rmse_wrist_gripper",
        "hcce_match_dist_px",
    ]
    for key in metric_keys:
        stat = stats_for(rows, key)
        if stat["count"] > 0:
            base["metrics"][key] = stat
    return base


def fmt_stat(summary, key):
    stat = summary["metrics"].get(key)
    if not stat:
        return "n/a"
    return f"mean={stat['mean']:.6g}, median={stat['median']:.6g}, rmse={stat['rmse']:.6g}, n={stat['count']}"


def write_summary_md(path, summary):
    lines = [
        f"# {summary['dataset']} RoboPEPP vs densepart_h2 Quant",
        "",
        f"- rows: {summary['num_rows']}",
        f"- RoboPEPP ok/error: {summary['num_robopepp_ok']}/{summary['num_robopepp_failed']}",
        f"- videos: {', '.join(summary['videos'])}",
        "",
        "## Pose",
        f"- RoboPEPP trans err m: {fmt_stat(summary, 'robopepp_trans_err_m')}",
        f"- RoboPEPP rot err deg: {fmt_stat(summary, 'robopepp_rot_err_deg')}",
        f"- RoboPEPP action RMSE rad: {fmt_stat(summary, 'robopepp_action_rmse_rad')}",
        f"- RoboPEPP 3D keypoint RMSE m: {fmt_stat(summary, 'robopepp_keypoint_3d_rmse_m')}",
        f"- RoboPEPP reproj RMSE px: {fmt_stat(summary, 'robopepp_reproj_rmse_px')}",
        f"- HCCE trans err m: {fmt_stat(summary, 'hcce_trans_err_m')}",
        f"- HCCE rot err deg: {fmt_stat(summary, 'hcce_rot_err_deg')}",
        f"- HCCE reproj RMSE px: {fmt_stat(summary, 'hcce_reproj_rmse_px')}",
        f"- HCCE opt RMSE all px: {fmt_stat(summary, 'hcce_opt_rmse_all')}",
        "",
        "## Segmentation",
        f"- RoboPEPP render inst IoU: {fmt_stat(summary, 'robopepp_render_seg_inst_iou')}",
        f"- RoboPEPP render part IoU mean: {fmt_stat(summary, 'robopepp_render_seg_part_iou_mean')}",
        f"- densepart_h2 render inst IoU: {fmt_stat(summary, 'hcce_render_seg_inst_iou')}",
        f"- densepart_h2 render part IoU mean: {fmt_stat(summary, 'hcce_render_seg_part_iou_mean')}",
        f"- densepart_h2 model inst IoU: {fmt_stat(summary, 'model_seg_inst_iou')}",
        f"- densepart_h2 model part IoU mean: {fmt_stat(summary, 'model_seg_part_iou_mean')}",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {path}", flush=True)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["needleGrasping", "suturePulling", "both"], default="both")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_ROBOPEPP_CKPT))
    parser.add_argument("--output_dir", type=str, default=str(ROBOPEPP_ROOT / "logs/robopepp_vs_hcce_quant"))
    parser.add_argument("--needle_hcce_csv", type=str, default=str(DEFAULT_NEEDLE_HCCE_CSV))
    parser.add_argument("--suture_hcce_csv", type=str, default=str(DEFAULT_SUTURE_HCCE_CSV))
    parser.add_argument("--suture_dataset_root", type=str, default=str(DEFAULT_SUTURE_ROOT))
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--panel_size", type=int, default=630)
    parser.add_argument("--pose_recovery", choices=["pnp", "direct"], default="pnp")
    parser.add_argument("--pnp_min_score", type=float, default=0.0)
    parser.add_argument("--overlay_alpha", type=float, default=0.85)
    parser.add_argument("--render_min_depth", type=float, default=1e-4)
    parser.add_argument("--render_draw_margin", type=float, default=20.0)
    parser.add_argument("--needle_train_ratio", type=float, default=0.95)
    parser.add_argument("--canonicalize_pose_symmetry", type=int, choices=[0, 1], default=1)
    parser.add_argument("--canonical_eps", type=float, default=0.08)
    parser.add_argument("--min_dice_shaft", type=float, default=0.8)
    parser.add_argument("--min_dice_wrist", type=float, default=0.6)
    parser.add_argument("--min_dice_gripper", type=float, default=0.6)
    parser.add_argument("--dataset_cache_dir", type=str, default=str(ROBOPEPP_ROOT / "logs/robopepp_quant_cache/dataset_cache"))
    parser.add_argument("--suture_num_videos", type=int, default=10)
    parser.add_argument("--suture_frame_stride", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--vis_limit", type=int, default=12)
    parser.add_argument("--print_freq", type=int, default=25)
    parser.add_argument("--fail_fast", type=int, choices=[0, 1], default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--devices", nargs="*", default=["cuda:0"])
    return parser


def main(args):
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    args.suture_dataset_root = str(resolve_path(args.suture_dataset_root, MULTIHMR_ROOT))
    args.dataset_cache_dir = str(resolve_path(args.dataset_cache_dir))
    all_summaries = {}
    if args.dataset in ("needleGrasping", "both"):
        _, needle_items, needle_meta = build_needle_items(args)
        print(f"[needleGrasping] selection meta: {json.dumps(needle_meta, indent=2)}", flush=True)
        _, summary = run_workers("needleGrasping", needle_items, args)
        summary["selection_meta"] = needle_meta
        (Path(args.output_dir) / "needleGrasping" / "summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=True),
            encoding="utf-8",
        )
        all_summaries["needleGrasping"] = summary
    if args.dataset in ("suturePulling", "both"):
        suture_items, suture_meta = build_suture_items(args)
        print(f"[suturePulling] selection meta: {json.dumps(suture_meta, indent=2)}", flush=True)
        _, summary = run_workers("suturePulling", suture_items, args)
        summary["selection_meta"] = suture_meta
        (Path(args.output_dir) / "suturePulling" / "summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=True),
            encoding="utf-8",
        )
        all_summaries["suturePulling"] = summary
    combined_path = Path(args.output_dir) / "combined_summary.json"
    combined_path.write_text(json.dumps(all_summaries, indent=2, allow_nan=True), encoding="utf-8")
    print(f"wrote {combined_path}", flush=True)


if __name__ == "__main__":
    main(build_parser().parse_args())
