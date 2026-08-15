import argparse
import csv
import importlib.util
import os
import sys
from pathlib import Path

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

from instrument_geometry import (  # noqa: E402
    KEYPOINT_NAMES,
    instrument_keypoints_camera_np,
    project_points_np,
)
from instrument_trimesh_renderer import GMSInstrumentTrimeshRenderer  # noqa: E402
from pose_pnp import pose_from_keypoints_pnp  # noqa: E402


PART_LABELS = {
    "gripper": 1,
    "wrist": 2,
    "shaft": 3,
}


DATASETS = {
    "needlePuncture": {
        "dataset_root": "/mnt/nas/share/shuojue/data/needlePuncture_videos",
        "pose_root": "/mnt/nas/share/shuojue/data/needlePuncture_results",
    },
    "needleGrasping": {
        "dataset_root": "/mnt/nas/share/shuojue/data/needleGrasping_videos",
        "pose_root": "/mnt/nas/share/shuojue/data/needleGrasping_results",
    },
    "knotting": {
        "dataset_root": "/mnt/nas/share/shuojue/data/knotting_videos",
        "pose_root": "/mnt/nas/share/shuojue/data/knotting_results",
    },
}


def _load_local_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _as_args_dict(obj):
    if obj is None:
        return {}
    if hasattr(obj, "__dict__"):
        return vars(obj)
    if isinstance(obj, dict):
        return obj
    raise TypeError(f"Unsupported checkpoint args type: {type(obj)}")


def _model_kwargs_from_checkpoint(ckpt_args):
    src = _as_args_dict(ckpt_args)
    crop_size = int(src.get("crop_size", 224))
    return {
        "backbone": src.get("backbone", "vit_base"),
        "input_shape": (crop_size, crop_size),
        "patch_size": int(src.get("patch_size", 16)),
        "pred_emb_dim": int(src.get("pred_emb_dim", 384)),
        "pred_depth": int(src.get("pred_depth", 12)),
        "num_keypoints": 5,
        "pose_head_iter": int(src.get("pose_head_iter", 4)),
        "pose_head_dropout": float(src.get("pose_head_dropout", 0.3)),
        # The checkpoint already contains JEPA-loaded weights.
        "jepa_path": None,
    }


def load_model(checkpoint_path, device):
    model_module = _load_local_module("robopepp_instrument_model", ROBOPEPP_ROOT / "models" / "instrument_model.py")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" not in ckpt:
        raise KeyError(f"Checkpoint missing model_state_dict: {checkpoint_path}")
    model = model_module.make_robopepp_instrument_posenet(**_model_kwargs_from_checkpoint(ckpt.get("args")))
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, ckpt


def heatmap_argmax(heatmaps):
    if heatmaps.ndim != 4:
        raise ValueError(f"Expected heatmaps [B,K,H,W], got {tuple(heatmaps.shape)}")
    b, k, h, w = heatmaps.shape
    flat = heatmaps.reshape(b, k, -1)
    scores, idx = flat.max(dim=-1)
    xy = torch.stack([idx % w, idx // w], dim=-1).float()
    return xy, scores


def crop_points_to_original(points_crop, target):
    points = np.asarray(points_crop, dtype=np.float64).copy()
    scale = target["scale"].numpy().astype(np.float64)
    pad = target["pad"].numpy().astype(np.float64)
    bbox_min = target["bbox_min"].numpy().astype(np.float64)
    points[:, 0] = (points[:, 0] - pad[0]) / scale[0] + bbox_min[0]
    points[:, 1] = (points[:, 1] - pad[1]) / scale[1] + bbox_min[1]
    return points.astype(np.float32)


def _draw_text(out, text, org, color, scale=0.45):
    cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_keypoints_overlay(rgb, bbox_min, bbox_max, gt_uv, gt_valid, pred_hm_uv, pred_pose_uv, hm_scores):
    out = rgb.copy()
    cv2.rectangle(
        out,
        tuple(np.round(bbox_min).astype(int)),
        tuple(np.round(bbox_max).astype(int)),
        (255, 255, 255),
        2,
        lineType=cv2.LINE_AA,
    )
    legend = [
        ("gt visible", (40, 220, 70)),
        ("pred heatmap", (255, 70, 220)),
        ("pred pose FK", (60, 210, 255)),
    ]
    x0 = 10
    for i, (label, color) in enumerate(legend):
        y = 22 + i * 20
        cv2.circle(out, (x0 + 7, y - 5), 5, color, -1, lineType=cv2.LINE_AA)
        _draw_text(out, label, (x0 + 18, y), color)

    for i, name in enumerate(KEYPOINT_NAMES):
        if bool(gt_valid[i]):
            x, y = np.round(gt_uv[i]).astype(int)
            cv2.circle(out, (x, y), 5, (40, 220, 70), -1, lineType=cv2.LINE_AA)
        xh, yh = np.round(pred_hm_uv[i]).astype(int)
        cv2.drawMarker(out, (xh, yh), (255, 70, 220), cv2.MARKER_CROSS, 13, 2, cv2.LINE_AA)
        xp, yp = np.round(pred_pose_uv[i]).astype(int)
        cv2.circle(out, (xp, yp), 4, (60, 210, 255), 2, lineType=cv2.LINE_AA)
        _draw_text(out, f"{i}:{name} {float(hm_scores[i]):.2f}", (xh + 6, yh - 5), (255, 70, 220), scale=0.36)
    return out


def draw_crop_debug(crop_rgb, gt_crop, valid, pred_hm_crop, pred_pose_crop):
    out = crop_rgb.copy()
    for i, name in enumerate(KEYPOINT_NAMES):
        if bool(valid[i]):
            x, y = np.round(gt_crop[i]).astype(int)
            cv2.circle(out, (x, y), 4, (40, 220, 70), -1, lineType=cv2.LINE_AA)
        xh, yh = np.round(pred_hm_crop[i]).astype(int)
        cv2.drawMarker(out, (xh, yh), (255, 70, 220), cv2.MARKER_CROSS, 11, 2, cv2.LINE_AA)
        xp, yp = np.round(pred_pose_crop[i]).astype(int)
        cv2.circle(out, (xp, yp), 3, (60, 210, 255), 2, lineType=cv2.LINE_AA)
        _draw_text(out, f"{i}:{name}", (xh + 4, yh - 4), (255, 70, 220), scale=0.33)
    return out


def add_title(panel, title):
    out = panel.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(out, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def square_image_geometry(rgb, img_size):
    h, w = rgb.shape[:2]
    scale = float(img_size) / float(max(w, h))
    new_w = int(w * scale)
    new_h = int(h * scale)
    pad_x = (int(img_size) - new_w) // 2
    pad_y = (int(img_size) - new_h) // 2
    return scale, pad_x, pad_y


def pad_rgb_to_square(rgb, img_size, scale, pad_x, pad_y):
    h, w = rgb.shape[:2]
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = np.zeros((int(img_size), int(img_size), 3), dtype=np.uint8)
    padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return padded


def pad_mask_to_square(mask, img_size, scale, pad_x, pad_y):
    h, w = mask.shape[:2]
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(mask.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    padded = np.zeros((int(img_size), int(img_size)), dtype=np.uint8)
    padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return padded


def mask_to_color(mask):
    mask = np.asarray(mask, dtype=np.uint8)
    color = np.zeros(mask.shape + (3,), dtype=np.uint8)
    color[mask == PART_LABELS["gripper"]] = (70, 150, 255)
    color[mask == PART_LABELS["wrist"]] = (40, 210, 120)
    color[mask == PART_LABELS["shaft"]] = (235, 70, 60)
    return color


def overlay_part_mask(rgb, mask, alpha=0.55):
    color = mask_to_color(mask)
    support = (mask > 0)[..., None]
    blended = cv2.addWeighted(rgb, 1.0, color, alpha, 0)
    return np.where(support, blended, rgb).astype(np.uint8)


def concat_panels(panels, separator=6):
    if not panels:
        raise ValueError("No panels to concatenate")
    h = panels[0].shape[0]
    sep = np.full((h, int(separator), 3), 255, dtype=np.uint8)
    out = []
    for i, panel in enumerate(panels):
        if i > 0:
            out.append(sep)
        out.append(panel)
    return np.concatenate(out, axis=1)


def draw_single_keypoint_set(rgb, points, valid, color, title, marker="circle"):
    out = rgb.copy()
    for i, (xy, ok) in enumerate(zip(points, valid)):
        if not bool(ok):
            continue
        x, y = np.round(xy).astype(int)
        if marker == "cross":
            cv2.drawMarker(out, (x, y), color, cv2.MARKER_CROSS, 15, 2, cv2.LINE_AA)
        else:
            cv2.circle(out, (x, y), 5, color, -1, lineType=cv2.LINE_AA)
        _draw_text(out, str(i), (x + 5, y - 5), color, scale=0.42)
    return add_title(out, title)


def make_concat_visual(orig_rgb, gt_mesh_overlay, pred_mesh_overlay, model_part_mask, gt_pose_part_mask, pred_pose_part_mask, panel_size):
    scale, pad_x, pad_y = square_image_geometry(orig_rgb, panel_size)
    rgb_sq = pad_rgb_to_square(orig_rgb, panel_size, scale, pad_x, pad_y)
    gt_mesh_sq = pad_rgb_to_square(gt_mesh_overlay, panel_size, scale, pad_x, pad_y)
    pred_mesh_sq = pad_rgb_to_square(pred_mesh_overlay, panel_size, scale, pad_x, pad_y)
    model_mask_sq = pad_mask_to_square(model_part_mask, panel_size, scale, pad_x, pad_y)
    gt_pose_mask_sq = pad_mask_to_square(gt_pose_part_mask, panel_size, scale, pad_x, pad_y)
    pred_pose_mask_sq = pad_mask_to_square(pred_pose_part_mask, panel_size, scale, pad_x, pad_y)
    panels = [
        add_title(rgb_sq, "rgb"),
        add_title(gt_mesh_sq, "hcce-trimesh"),
        add_title(pred_mesh_sq, "posehead-trimesh"),
        add_title(overlay_part_mask(rgb_sq, model_mask_sq), "model-seg"),
        add_title(overlay_part_mask(rgb_sq, gt_pose_mask_sq), "hcce-proj-seg"),
        add_title(overlay_part_mask(rgb_sq, pred_pose_mask_sq), "posehead-proj-seg"),
    ]
    return concat_panels(panels)


def _save(path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8)).save(path)


def _pose_from_target(target):
    return {
        "rot": target["wrist_quat"].numpy(),
        "trans": target["wrist_trans"].numpy(),
        "alpha": float(target["action"][0].item()),
        "theta_l": float(target["action"][1].item()),
        "theta_r": float(target["action"][2].item()),
    }


def _pose_from_output(out):
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


def _l1(a, b):
    return float(np.mean(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


def select_sample_indices(dataset, args):
    start = int(args.start_index)
    indices = []
    for idx in range(start, len(dataset)):
        video_name, frame_id, instance_id, _ = dataset.samples[idx]
        if args.video_name and video_name != args.video_name:
            continue
        if int(args.instance_id) > 0 and int(instance_id) != int(args.instance_id):
            continue
        if int(args.frame_stride) > 1 and int(frame_id) % int(args.frame_stride) != 0:
            continue
        indices.append(idx)
        if int(args.num_samples) > 0 and len(indices) >= int(args.num_samples):
            break
    num_workers = int(args.num_workers)
    worker_id = int(args.worker_id)
    if num_workers < 1:
        raise ValueError("--num_workers must be >= 1")
    if worker_id < 0 or worker_id >= num_workers:
        raise ValueError("--worker_id must be in [0, num_workers)")
    if num_workers > 1:
        indices = indices[worker_id::num_workers]
    if not indices:
        raise RuntimeError("No samples selected for prediction visualization")
    return indices


def main(args):
    rarp_module = _load_local_module("robopepp_rarp_instrument", ROBOPEPP_ROOT / "datasets" / "rarp_instrument.py")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, _ = load_model(Path(args.checkpoint), device)

    cfg = DATASETS[args.dataset_name]
    dataset = rarp_module.RoboPEPPRARPInstrument(
        cfg["dataset_root"],
        cfg["pose_root"],
        split=args.split,
        training=False,
        crop_size=args.crop_size,
        train_ratio=args.needle_train_ratio,
        subsample=args.subsample,
        min_dice=(args.min_dice_shaft, args.min_dice_wrist, args.min_dice_gripper),
        canonicalize_pose_symmetry=bool(args.canonicalize_pose_symmetry),
        canonical_eps=args.canonical_eps,
        cache_dir=args.dataset_cache_dir,
    )
    mesh_renderer = GMSInstrumentTrimeshRenderer(device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    sample_indices = select_sample_indices(dataset, args)
    concat_dir = out_dir / args.concat_subdir
    if bool(args.save_concat):
        concat_dir.mkdir(parents=True, exist_ok=True)

    for out_i, sample_idx in enumerate(sample_indices):
        image, target = dataset[sample_idx]
        x = image.unsqueeze(0).to(device, non_blocking=True)
        K_crop = target["K"].unsqueeze(0).to(device, non_blocking=True)
        with torch.inference_mode(), torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda"), dtype=torch.bfloat16):
            out = model(x, K_crop, masks_enc=None, masks_pred=None)

        pred_hm_crop_t, hm_scores_t = heatmap_argmax(out["keypoint_heatmaps"].float().detach().cpu())
        pred_hm_crop = pred_hm_crop_t[0].numpy().astype(np.float32)
        hm_scores = hm_scores_t[0].numpy().astype(np.float32)

        direct_pose = _pose_from_output(out)
        if args.pose_recovery == "pnp":
            action = [direct_pose["alpha"], direct_pose["theta_l"], direct_pose["theta_r"]]
            pred_pose = pose_from_keypoints_pnp(
                pred_hm_crop,
                action,
                target["K"].numpy(),
                scores=hm_scores,
                min_score=args.pnp_min_score,
            )
        else:
            pred_pose = direct_pose
        gt_pose = _pose_from_target(target)
        pred_kp_cam = instrument_keypoints_camera_np(pred_pose["rot"], pred_pose["trans"], [pred_pose["alpha"], pred_pose["theta_l"], pred_pose["theta_r"]])
        pred_pose_crop = project_points_np(pred_kp_cam, target["K"].numpy()).astype(np.float32)
        pred_pose_orig = project_points_np(pred_kp_cam, target["K_orig"].numpy()).astype(np.float32)
        pred_hm_orig = crop_points_to_original(pred_hm_crop, target)

        gt_crop = target["keypoints_crop"].numpy().astype(np.float32)
        gt_orig = target["keypoints_orig"].numpy().astype(np.float32)
        valid = target["keypoints_valid"].numpy().astype(bool)
        valid_orig = target["keypoints_valid_orig"].numpy().astype(bool)

        if valid.any():
            hm_crop_err = float(np.linalg.norm(pred_hm_crop[valid] - gt_crop[valid], axis=1).mean())
            hm_orig_err = float(np.linalg.norm(pred_hm_orig[valid] - gt_orig[valid], axis=1).mean())
            pose_orig_err = float(np.linalg.norm(pred_pose_orig[valid] - gt_orig[valid], axis=1).mean())
        else:
            hm_crop_err = float("nan")
            hm_orig_err = float("nan")
            pose_orig_err = float("nan")

        orig_rgb = target["orig_rgb"].numpy()
        crop_rgb = target["crop_rgb"].numpy()
        bbox_min = target["bbox_min"].numpy()
        bbox_max = target["bbox_max"].numpy()
        stem = f"{target['video_name']}_{target['frame_id']}_inst{int(target['instance_id'].item())}"
        detail_stem = f"{args.dataset_name}_{args.split}_{sample_idx:05d}_{stem}"

        overlay = draw_keypoints_overlay(orig_rgb, bbox_min, bbox_max, gt_orig, valid_orig, pred_hm_orig, pred_pose_orig, hm_scores)
        crop_debug = draw_crop_debug(crop_rgb, gt_crop, valid, pred_hm_crop, pred_pose_crop)
        if bool(args.save_detail_panels):
            _save(out_dir / f"{detail_stem}_orig_prediction_overlay.jpg", overlay)
            _save(out_dir / f"{detail_stem}_crop_prediction_debug.jpg", crop_debug)

        pred_mesh_overlay = None
        gt_mesh_overlay = None
        if bool(args.save_detail_panels):
            pred_mesh_overlay = mesh_renderer.render_pose_overlay(
                orig_rgb, pred_pose, target["K_orig"].numpy(), alpha=args.overlay_alpha
            )
            gt_mesh_overlay = mesh_renderer.render_pose_overlay(
                orig_rgb, gt_pose, target["K_orig"].numpy(), alpha=args.overlay_alpha
            )
            _save(out_dir / f"{detail_stem}_orig_pred_instrument_trimesh_overlay.jpg", pred_mesh_overlay)
            _save(out_dir / f"{detail_stem}_orig_gt_instrument_trimesh_overlay.jpg", gt_mesh_overlay)
        if bool(args.save_concat):
            if pred_mesh_overlay is None:
                pred_mesh_overlay = mesh_renderer.render_pose_overlay(
                    orig_rgb, pred_pose, target["K_orig"].numpy(), alpha=args.overlay_alpha
                )
            if gt_mesh_overlay is None:
                gt_mesh_overlay = mesh_renderer.render_pose_overlay(
                    orig_rgb, gt_pose, target["K_orig"].numpy(), alpha=args.overlay_alpha
                )
            model_part_mask = target["part_mask_orig"].numpy().astype(np.uint8)
            gt_pose_part_mask = mesh_renderer.render_pose_mask(
                gt_pose,
                target["K_orig"].numpy(),
                orig_rgb.shape[:2],
                min_depth=args.render_min_depth,
                draw_margin=args.render_draw_margin,
            )
            pred_pose_part_mask = mesh_renderer.render_pose_mask(
                pred_pose,
                target["K_orig"].numpy(),
                orig_rgb.shape[:2],
                min_depth=args.render_min_depth,
                draw_margin=args.render_draw_margin,
            )
            concat_vis = make_concat_visual(
                orig_rgb,
                gt_mesh_overlay,
                pred_mesh_overlay,
                model_part_mask,
                gt_pose_part_mask,
                pred_pose_part_mask,
                int(args.concat_panel_size),
            )
            _save(concat_dir / f"{stem}.jpg", concat_vis)

        row = {
            "out_index": out_i,
            "sample_idx": sample_idx,
            "dataset": args.dataset_name,
            "split": args.split,
            "video_name": target["video_name"],
            "frame_id": target["frame_id"],
            "instance_id": int(target["instance_id"].item()),
            "hm_crop_err_px": hm_crop_err,
            "hm_orig_err_px": hm_orig_err,
            "pose_orig_err_px": pose_orig_err,
            "visible_keypoints": int(valid.sum()),
            "action_l1": _l1([pred_pose["alpha"], pred_pose["theta_l"], pred_pose["theta_r"]], target["action"].numpy()),
            "quat_l1": _l1(pred_pose["rot"], target["wrist_quat"].numpy()),
            "trans_l1": _l1(pred_pose["trans"], target["wrist_trans"].numpy()),
            "pose_recovery": args.pose_recovery,
        }
        rows.append(row)
        if (out_i + 1) % int(args.print_freq) == 0 or out_i == 0 or out_i + 1 == len(sample_indices):
            print(
                f"saved {out_i + 1}/{len(sample_indices)} {stem}: "
                f"hm_orig={hm_orig_err:.2f}px pose_orig={pose_orig_err:.2f}px "
                f"action_l1={row['action_l1']:.4f} -> {out_dir}",
                flush=True,
            )

    if int(args.num_workers) > 1:
        csv_path = out_dir / f"prediction_summary_worker{int(args.worker_id):02d}.csv"
    else:
        csv_path = out_dir / "prediction_summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="logs/robopepp_instrument_pose_rarp_jepa_bs56_gpu0123/checkpoints/last.pt")
    parser.add_argument("--dataset_name", type=str, default="needlePuncture", choices=sorted(DATASETS))
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--output_dir", type=str, default="logs/robopepp_instrument_predictions")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=0)
    parser.add_argument("--subsample", type=int, default=1)
    parser.add_argument("--video_name", type=str, default="")
    parser.add_argument("--instance_id", type=int, default=0)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--worker_id", type=int, default=0)
    parser.add_argument("--save_concat", type=int, default=1, choices=[0, 1])
    parser.add_argument("--concat_subdir", type=str, default="vis")
    parser.add_argument("--concat_panel_size", type=int, default=630)
    parser.add_argument("--save_detail_panels", type=int, default=0, choices=[0, 1])
    parser.add_argument("--print_freq", type=int, default=25)
    parser.add_argument("--dataset_cache_dir", type=str, default="logs/robopepp_instrument_debug_cache/dataset_cache")
    parser.add_argument("--needle_train_ratio", type=float, default=0.95)
    parser.add_argument("--canonicalize_pose_symmetry", type=int, default=1, choices=[0, 1])
    parser.add_argument("--canonical_eps", type=float, default=0.08)
    parser.add_argument("--min_dice_shaft", type=float, default=0.8)
    parser.add_argument("--min_dice_wrist", type=float, default=0.6)
    parser.add_argument("--min_dice_gripper", type=float, default=0.6)
    parser.add_argument("--mesh_render_backend", type=str, default="instrument_trimesh", choices=["instrument_trimesh"])
    parser.add_argument("--overlay_alpha", type=float, default=0.85)
    parser.add_argument("--render_min_depth", type=float, default=1e-4)
    parser.add_argument("--render_draw_margin", type=float, default=20.0)
    parser.add_argument("--pose_recovery", type=str, default="pnp", choices=["pnp", "direct"])
    parser.add_argument("--pnp_min_score", type=float, default=0.0)
    main(parser.parse_args())
