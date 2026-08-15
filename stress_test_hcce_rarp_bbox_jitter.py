import argparse
import csv
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as tv_transforms

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

ROBOPEPP_ROOT = Path(__file__).resolve().parent
MULTIHMR_ROOT = ROBOPEPP_ROOT.parents[1]
if str(ROBOPEPP_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBOPEPP_ROOT))
if str(MULTIHMR_ROOT) not in sys.path:
    sys.path.insert(0, str(MULTIHMR_ROOT))

from compare_crop_hcce_robopepp_rarp import (  # noqa: E402
    DEFAULT_HCCE_CKPT,
    binary_iou,
    crop_points_to_original,
    crop_resize_pad_intrinsics,
    load_hcce_model,
    model_part_mask_crop,
    pose_from_output,
    pose_from_target,
    rotation_error_deg,
    segmentation_metrics_from_masks,
)
from instrument_geometry import KEYPOINT_NAMES, project_points_np  # noqa: E402
from predict_instrument_pose import DATASETS, heatmap_argmax  # noqa: E402


PART_COLORS = np.array(
    [
        [0, 0, 0],
        [230, 80, 80],
        [80, 220, 120],
        [80, 150, 240],
    ],
    dtype=np.uint8,
)


def load_local_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_rarp_module = load_local_module("hcce_bbox_jitter_rarp_dataset", ROBOPEPP_ROOT / "datasets" / "rarp_hcce_crop.py")
RARPCropHCCEDataset = _rarp_module.RARPCropHCCEDataset
_crop_resize_pad_map = _rarp_module._crop_resize_pad_map


def label_image(img, lines, font_scale=0.38):
    out = img.copy()
    if isinstance(lines, str):
        lines = [lines]
    bar_h = 18 + 15 * max(1, len(lines))
    cv2.rectangle(out, (0, 0), (out.shape[1], min(bar_h, out.shape[0])), (0, 0, 0), -1)
    for i, line in enumerate(lines):
        cv2.putText(
            out,
            str(line),
            (5, 15 + 15 * i),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return out


def resize_longer_side(rgb, crop_size):
    h, w = rgb.shape[:2]
    if w > h:
        new_w = int(crop_size)
        new_h = int(round(crop_size * h / w))
    else:
        new_h = int(crop_size)
        new_w = int(round(crop_size * w / h))
    out = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return out, (new_w, new_h)


def pad_to_square(rgb, crop_size):
    h, w = rgb.shape[:2]
    pad_h = (int(crop_size) - h) // 2
    pad_w = (int(crop_size) - w) // 2
    padding = ((pad_h, int(crop_size) - h - pad_h), (pad_w, int(crop_size) - w - pad_w), (0, 0))
    return np.pad(rgb, padding, mode="edge"), (pad_w, pad_h)


def perturb_bbox(bbox_min, bbox_max, image_w, image_h, variant, rng):
    bbox_min = np.asarray(bbox_min, dtype=np.float32).copy()
    bbox_max = np.asarray(bbox_max, dtype=np.float32).copy()
    w = float(bbox_max[0] - bbox_min[0])
    h = float(bbox_max[1] - bbox_min[1])
    side = max(w, h)
    mode = variant["mode"]
    value = float(variant["value"])
    if mode == "base":
        pass
    elif mode == "expand_px":
        # Same shape as training jitter: independently expands four sides outward.
        left, top, right, bottom = rng.random(4).astype(np.float32) * value
        bbox_min -= np.array([left, top], dtype=np.float32)
        bbox_max += np.array([right, bottom], dtype=np.float32)
    elif mode == "shift_frac":
        shift = np.array([variant.get("sx", value) * side, variant.get("sy", 0.0) * side], dtype=np.float32)
        bbox_min += shift
        bbox_max += shift
    elif mode == "scale":
        center = (bbox_min + bbox_max) * 0.5
        half = (bbox_max - bbox_min) * 0.5 * value
        bbox_min = center - half
        bbox_max = center + half
    else:
        raise ValueError(f"unknown bbox jitter mode: {mode}")
    bbox_min = np.clip(bbox_min, [0.0, 0.0], [float(image_w - 1), float(image_h - 1)])
    bbox_max = np.clip(bbox_max, [1.0, 1.0], [float(image_w), float(image_h)])
    if bbox_max[0] <= bbox_min[0] + 1.0 or bbox_max[1] <= bbox_min[1] + 1.0:
        raise RuntimeError(f"invalid perturbed bbox {bbox_min} -> {bbox_max}")
    return bbox_min.astype(np.float32), bbox_max.astype(np.float32)


def build_crop_variant(target, bbox_min, bbox_max, crop_size):
    rgb = target["orig_rgb"].detach().cpu().numpy().astype(np.uint8)
    part_orig = target["part_mask_orig"].detach().cpu().numpy().astype(np.uint8)
    inst_orig = target["inst_mask_orig"].detach().cpu().numpy().astype(bool)
    K_orig = target["K_orig"].detach().cpu().numpy().astype(np.float32)
    x0, y0 = bbox_min.astype(np.int64)
    x1, y1 = np.ceil(bbox_max).astype(np.int64)
    crop = rgb[y0:y1, x0:x1]
    if crop.shape[0] <= 0 or crop.shape[1] <= 0:
        raise RuntimeError(f"empty crop for jitter bbox {bbox_min} -> {bbox_max}")
    resized, (new_w, new_h) = resize_longer_side(crop, crop_size)
    scale_x = float(new_w) / float(bbox_max[0] - bbox_min[0])
    scale_y = float(new_h) / float(bbox_max[1] - bbox_min[1])
    crop_square, (pad_w, pad_h) = pad_to_square(resized, crop_size)
    K_crop = crop_resize_pad_intrinsics(K_orig, bbox_min=bbox_min, scale_xy=(scale_x, scale_y), pad_xy=(pad_w, pad_h))
    gt_part_crop = _crop_resize_pad_map(
        part_orig,
        bbox_min,
        bbox_max,
        crop_size,
        interpolation=cv2.INTER_NEAREST,
        value=0,
    ).astype(np.uint8)
    gt_inst_crop = _crop_resize_pad_map(
        inst_orig.astype(np.uint8),
        bbox_min,
        bbox_max,
        crop_size,
        interpolation=cv2.INTER_NEAREST,
        value=0,
    ).astype(np.float32)
    kp_3d = target["keypoints_3d_cam"].detach().cpu().numpy().astype(np.float32)
    kp_crop = project_points_np(kp_3d, K_crop).astype(np.float32)
    valid_orig = target["keypoints_valid_orig"].detach().cpu().numpy().astype(bool)
    valid_crop = (
        valid_orig
        & np.isfinite(kp_crop).all(axis=1)
        & (kp_crop[:, 0] >= 0.0)
        & (kp_crop[:, 0] < float(crop_size))
        & (kp_crop[:, 1] >= 0.0)
        & (kp_crop[:, 1] < float(crop_size))
    )
    return {
        "crop_rgb": crop_square.astype(np.uint8),
        "K_crop": K_crop.astype(np.float32),
        "gt_part_crop": gt_part_crop,
        "gt_inst_crop": gt_inst_crop,
        "keypoints_crop": kp_crop.astype(np.float32),
        "keypoints_valid": valid_crop,
        "keypoints_orig": target["keypoints_orig"].detach().cpu().numpy().astype(np.float32),
        "keypoints_valid_orig": valid_orig,
        "bbox_min": bbox_min.astype(np.float32),
        "bbox_max": bbox_max.astype(np.float32),
        "scale": np.array([scale_x, scale_y], dtype=np.float32),
        "pad": np.array([pad_w, pad_h], dtype=np.float32),
    }


def part_rgb(part):
    return PART_COLORS[np.clip(part.astype(np.int64), 0, len(PART_COLORS) - 1)]


def overlay_part(rgb, part, alpha=0.38):
    out = rgb.copy()
    mask = part > 0
    colors = part_rgb(part)
    out[mask] = ((1.0 - alpha) * out[mask] + alpha * colors[mask]).astype(np.uint8)
    return out


def seg_diff_rgb(gt_part, pred_part):
    gt = gt_part > 0
    pred = pred_part > 0
    out = np.zeros((*gt.shape, 3), dtype=np.uint8)
    out[gt & pred] = (80, 220, 120)
    out[gt & ~pred] = (240, 80, 80)
    out[~gt & pred] = (80, 150, 240)
    return out


def draw_keypoints(rgb, gt_crop, pred_crop, valid, scores):
    out = rgb.copy()
    for i, _name in enumerate(KEYPOINT_NAMES):
        if bool(valid[i]):
            gx, gy = np.round(gt_crop[i]).astype(int)
            if 0 <= gx < out.shape[1] and 0 <= gy < out.shape[0]:
                cv2.circle(out, (gx, gy), 3, (40, 230, 70), -1, cv2.LINE_AA)
        px, py = np.round(pred_crop[i]).astype(int)
        if 0 <= px < out.shape[1] and 0 <= py < out.shape[0]:
            cv2.drawMarker(out, (px, py), (255, 60, 220), cv2.MARKER_CROSS, 12, 2, cv2.LINE_AA)
            cv2.putText(out, f"{i}:{float(scores[i]):.2f}", (px + 3, py + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 60, 220), 1, cv2.LINE_AA)
        if bool(valid[i]) and 0 <= gx < out.shape[1] and 0 <= gy < out.shape[0] and 0 <= px < out.shape[1] and 0 <= py < out.shape[0]:
            cv2.line(out, (gx, gy), (px, py), (255, 255, 255), 1, cv2.LINE_AA)
    return out


def heatmap_panel(heatmaps):
    heat = np.clip(np.asarray(heatmaps).max(axis=0), 0.0, 1.0)
    bgr = cv2.applyColorMap((heat * 255.0).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def pose_action_array(pose):
    return np.asarray([pose["alpha"], pose["theta_l"], pose["theta_r"]], dtype=np.float64)


def pose_metrics(pred_pose, gt_pose):
    action_err_deg = np.degrees(np.abs(pose_action_array(pred_pose) - pose_action_array(gt_pose)))
    return {
        "direct_trans_err_m": float(np.linalg.norm(np.asarray(pred_pose["trans"]) - np.asarray(gt_pose["trans"]))),
        "direct_rot_err_deg": rotation_error_deg(pred_pose["rot"], gt_pose["rot"]),
        "direct_joint_mae_deg": float(np.mean(action_err_deg)),
    }


def make_variant_panel(crop_like, out, row, variant_name):
    pred_part = model_part_mask_crop(out, inst_thresh=0.5)
    pred_hm_t, score_t = heatmap_argmax(out["keypoint_heatmaps"].float().detach().cpu())
    pred_hm = pred_hm_t[0].numpy().astype(np.float32)
    scores = score_t[0].numpy().astype(np.float32)
    crop_rgb = crop_like["crop_rgb"]
    gt_part = crop_like["gt_part_crop"]
    kp_panel = draw_keypoints(overlay_part(crop_rgb, gt_part), crop_like["keypoints_crop"], pred_hm, crop_like["keypoints_valid"], scores)
    kp_panel = label_image(
        kp_panel,
        [
            variant_name,
            f"hm orig {row['hm_orig_rmse_px']:.1f}px crop {row['hm_crop_rmse_px']:.1f}px",
            f"part mIoU {row['part_iou_mean']:.3f} inst {row['inst_iou']:.3f}",
            f"rot {row['direct_rot_err_deg']:.1f} tr {row['direct_trans_err_m']:.3f}",
        ],
    )
    seg = seg_diff_rgb(gt_part, pred_part)
    seg = label_image(seg, "seg: green both, red GT only, blue pred only")
    hm = label_image(heatmap_panel(out["keypoint_heatmaps"][0].detach().float().cpu().numpy()), "max heatmap")
    bottom = np.concatenate([seg, hm], axis=1)
    bottom = cv2.resize(bottom, (crop_rgb.shape[1], crop_rgb.shape[0]), interpolation=cv2.INTER_AREA)
    return np.concatenate([kp_panel, bottom], axis=0)


def eval_variant(model, device, to_tensor, target, gt_pose, crop_like):
    x = to_tensor(Image.fromarray(crop_like["crop_rgb"])).unsqueeze(0).to(device)
    K_t = torch.from_numpy(crop_like["K_crop"]).unsqueeze(0).to(device)
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=device.type == "cuda", dtype=torch.bfloat16):
        out = model(x, K_t)
    pred_part = model_part_mask_crop(out, inst_thresh=0.5)
    seg = segmentation_metrics_from_masks(pred_part, crop_like["gt_part_crop"], "seg")
    pred_hm_t, score_t = heatmap_argmax(out["keypoint_heatmaps"].float().detach().cpu())
    pred_hm = pred_hm_t[0].numpy().astype(np.float32)
    scores = score_t[0].numpy().astype(np.float32)
    valid = crop_like["keypoints_valid"]
    if np.any(valid):
        crop_diff = pred_hm[valid] - crop_like["keypoints_crop"][valid]
        hm_crop_rmse = float(np.sqrt(np.mean(np.sum(crop_diff * crop_diff, axis=1))))
        pred_orig = crop_points_to_original(pred_hm, crop_like["bbox_min"], crop_like["scale"], crop_like["pad"])
        orig_valid = crop_like["keypoints_valid_orig"] & np.isfinite(pred_orig).all(axis=1)
        orig_diff = pred_orig[orig_valid] - crop_like["keypoints_orig"][orig_valid]
        hm_orig_rmse = float(np.sqrt(np.mean(np.sum(orig_diff * orig_diff, axis=1)))) if np.any(orig_valid) else float("nan")
    else:
        hm_crop_rmse = float("nan")
        hm_orig_rmse = float("nan")
    pred_pose = pose_from_output(out)
    row = {
        "hm_crop_rmse_px": hm_crop_rmse,
        "hm_orig_rmse_px": hm_orig_rmse,
        "hm_score_mean": float(np.mean(scores)),
        "valid_kp_count": int(valid.sum()),
        "inst_iou": float(seg["seg_inst_iou"]),
        "part_iou_mean": float(seg["seg_part_iou_mean"]),
        "part_iou_shaft": float(seg["seg_part_iou_shaft"]),
        "part_iou_wrist": float(seg["seg_part_iou_wrist"]),
        "part_iou_gripper": float(seg["seg_part_iou_gripper"]),
        "bbox_gt_inst_coverage": binary_iou(crop_like["gt_inst_crop"] > 0.5, crop_like["gt_part_crop"] > 0),
    }
    row.update(pose_metrics(pred_pose, gt_pose))
    return out, row


def variant_specs():
    return [
        {"name": "base", "mode": "base", "value": 0.0},
        {"name": "expand30", "mode": "expand_px", "value": 30.0},
        {"name": "expand80", "mode": "expand_px", "value": 80.0},
        {"name": "expand120", "mode": "expand_px", "value": 120.0},
        {"name": "shift+10%x", "mode": "shift_frac", "value": 0.10, "sx": 0.10, "sy": 0.0},
        {"name": "shift-10%x", "mode": "shift_frac", "value": -0.10, "sx": -0.10, "sy": 0.0},
        {"name": "shift+20%xy", "mode": "shift_frac", "value": 0.20, "sx": 0.20, "sy": 0.20},
    ]


def summarize(rows, out_path):
    variants = sorted({r["variant"] for r in rows})
    lines = ["# HCCE RARP bbox jitter stress test", ""]
    lines.append("| variant | n | hm orig px | part mIoU | inst IoU | rot deg | trans m | joint deg |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for variant in variants:
        sub = [r for r in rows if r["variant"] == variant]
        def mean(key):
            vals = [float(r[key]) for r in sub if math.isfinite(float(r[key]))]
            return float(np.mean(vals)) if vals else float("nan")
        lines.append(
            f"| {variant} | {len(sub)} | {mean('hm_orig_rmse_px'):.2f} | {mean('part_iou_mean'):.3f} | "
            f"{mean('inst_iou'):.3f} | {mean('direct_rot_err_deg'):.2f} | {mean('direct_trans_err_m'):.4f} | "
            f"{mean('direct_joint_mae_deg'):.2f} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="needleGrasping", choices=sorted(DATASETS))
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--indices", default="0,20,40,60,80")
    parser.add_argument("--checkpoint", default=str(DEFAULT_HCCE_CKPT))
    parser.add_argument("--output_dir", default=str(ROBOPEPP_ROOT / "logs/hcce_rarp_bbox_jitter_stress"))
    parser.add_argument("--dataset_cache_dir", default=str(ROBOPEPP_ROOT / "logs/hcce_rarp_bbox_jitter_stress/dataset_cache"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260702)
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--bbox_padding_frac", type=float, default=0.12)
    parser.add_argument("--train_ratio", type=float, default=0.95)
    parser.add_argument("--max_rows", type=int, default=0)
    args = parser.parse_args()

    if args.device.startswith("cuda"):
        torch.cuda.set_device(int(args.device.split(":", 1)[1]) if ":" in args.device else 0)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, model_meta = load_hcce_model(args.checkpoint, device)
    cfg = DATASETS[args.dataset]
    dataset = RARPCropHCCEDataset(
        cfg["dataset_root"],
        cfg["pose_root"],
        split=args.split,
        training=False,
        crop_size=args.crop_size,
        train_ratio=args.train_ratio,
        subsample=1,
        canonicalize_pose_symmetry=True,
        bbox_padding_frac=args.bbox_padding_frac,
        cache_dir=args.dataset_cache_dir,
        render_on_the_fly=True,
        coord_render_backend="trimesh",
        require_cse=True,
    )
    to_tensor = tv_transforms.Compose(
        [
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = out_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    indices = [int(v) for v in args.indices.replace(",", " ").split() if v.strip()]
    specs = variant_specs()
    rows = []
    sheets = []
    for ordinal, idx in enumerate(indices):
        image, target = dataset[idx]
        del image
        gt_pose = pose_from_target(target)
        h, w = target["orig_rgb"].shape[:2]
        base_min = target["bbox_min"].detach().cpu().numpy().astype(np.float32)
        base_max = target["bbox_max"].detach().cpu().numpy().astype(np.float32)
        sample_panels = []
        sample_rows = []
        for spec_i, spec in enumerate(specs):
            rng = np.random.default_rng(int(args.seed) + idx * 101 + spec_i)
            bbox_min, bbox_max = perturb_bbox(base_min, base_max, int(w), int(h), spec, rng)
            crop_like = build_crop_variant(target, bbox_min, bbox_max, args.crop_size)
            out, row = eval_variant(model, device, to_tensor, target, gt_pose, crop_like)
            video_name = str(target["video_name"])
            frame_id = str(target["frame_id"])
            instance_id = int(target["instance_id"].item())
            row.update(
                {
                    "dataset": args.dataset,
                    "split": args.split,
                    "dataset_idx": int(idx),
                    "video_name": video_name,
                    "frame_id": frame_id,
                    "instance_id": instance_id,
                    "variant": spec["name"],
                    "bbox_min": json.dumps([float(v) for v in bbox_min.tolist()]),
                    "bbox_max": json.dumps([float(v) for v in bbox_max.tolist()]),
                    "checkpoint": str(args.checkpoint),
                    "model_iter": int(model_meta.get("iter", -1)) if str(model_meta.get("iter", "")).isdigit() else model_meta.get("iter", ""),
                }
            )
            rows.append(row)
            sample_rows.append(row)
            sample_panels.append(make_variant_panel(crop_like, out, row, spec["name"]))
        sheet = np.concatenate(sample_panels, axis=1)
        stem = f"{ordinal:03d}_{args.dataset}_{str(target['video_name'])}_{str(target['frame_id'])}_inst{int(target['instance_id'].item())}"
        stem = stem.replace("/", "_")
        Image.fromarray(sheet).save(vis_dir / f"{stem}_bbox_jitter_stress.jpg", quality=92)
        sheets.append(label_image(sheet, [stem, "green keypoints=GT, magenta=pred heatmap"]))
        print(
            f"[ok] {stem} "
            + " | ".join(
                f"{r['variant']}:hm={r['hm_orig_rmse_px']:.1f},miou={r['part_iou_mean']:.3f},rot={r['direct_rot_err_deg']:.1f}"
                for r in sample_rows
            ),
            flush=True,
        )
    contact = np.concatenate(sheets, axis=0)
    Image.fromarray(contact).save(out_dir / "contact_sheet_bbox_jitter_stress.jpg", quality=92)
    csv_path = out_dir / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summarize(rows, out_dir / "summary.md")
    print(f"[done] wrote {len(rows)} rows to {csv_path}")
    print(f"[done] contact sheet: {out_dir / 'contact_sheet_bbox_jitter_stress.jpg'}")


if __name__ == "__main__":
    main()
