import argparse
import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from datasets.RarpInstanceDataset import RARPInstanceDataset
from datasets.surgpose_instruments import SurgPoseInstruments
from multi_instrument.hcce_codec import normalized_xyz_to_hcce
from utils import denormalize_rgb


PART_COLORS = {
    1: np.array([45, 135, 255], dtype=np.uint8),   # gripper
    2: np.array([80, 220, 120], dtype=np.uint8),   # wrist
    3: np.array([255, 90, 80], dtype=np.uint8),    # shaft
}
KEYPOINT_COLORS = (
    (255, 0, 255),
    (0, 255, 255),
    (255, 128, 0),
    (0, 255, 0),
    (255, 0, 0),
    (0, 128, 255),
    (180, 80, 255),
)


def _resize_nn(arr, size):
    return cv2.resize(arr, (size, size), interpolation=cv2.INTER_NEAREST)


def _to_rgb(img_chw):
    rgb = denormalize_rgb(np.asarray(img_chw), imagenet_normalization=True)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _title(img, text):
    h, w = img.shape[:2]
    canvas = np.zeros((h + 28, w, 3), dtype=np.uint8)
    canvas[28:] = img
    pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil)
    draw.text((8, 6), text, fill=(255, 255, 255))
    return np.asarray(pil)


def _overlay_binary(rgb, mask, color=(255, 255, 0), alpha=0.45):
    mask = _resize_nn(mask.astype(np.uint8), rgb.shape[0]) > 0
    out = rgb.copy()
    c = np.asarray(color, dtype=np.float32)
    out[mask] = (out[mask].astype(np.float32) * (1.0 - alpha) + c * alpha).astype(np.uint8)
    return out


def _overlay_part(rgb, part, alpha=0.5):
    part = _resize_nn(part.astype(np.uint8), rgb.shape[0])
    out = rgb.copy()
    for label, color in PART_COLORS.items():
        m = part == label
        out[m] = (out[m].astype(np.float32) * (1.0 - alpha) + color.astype(np.float32) * alpha).astype(np.uint8)
    return out


def _draw_keypoints(rgb, points, valid):
    out = rgb.copy()
    if points is None or valid is None:
        return out
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    valid = np.asarray(valid, dtype=bool).reshape(-1)
    for i, (xy, ok) in enumerate(zip(points, valid)):
        if not ok or not np.isfinite(xy).all():
            continue
        x, y = int(round(float(xy[0]))), int(round(float(xy[1])))
        if not (0 <= x < out.shape[1] and 0 <= y < out.shape[0]):
            continue
        color = KEYPOINT_COLORS[i % len(KEYPOINT_COLORS)]
        cv2.circle(out, (x, y), 6, color, 2, cv2.LINE_AA)
        cv2.drawMarker(out, (x, y), color, markerType=cv2.MARKER_CROSS, markerSize=16, thickness=2)
    return out


def _depth_panel(depth, valid, size):
    depth = _resize_nn(depth.astype(np.float32), size)
    valid = _resize_nn(valid.astype(np.uint8), size) > 0
    d = np.clip(depth, 0.0, 1.0)
    color = cv2.applyColorMap((d * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    color[~valid] = 0
    return color


def _coord_xyz_panel(coord, size):
    valid = coord[:, :, 3] > 0
    xyz = np.clip((coord[:, :, :3] + 1.0) * 0.5, 0.0, 1.0)
    panel = (xyz * 255).astype(np.uint8)
    panel[~valid] = 0
    return _resize_nn(panel, size)


def _hcce_bit_panel(coord, bits, size):
    valid = coord[:, :, 3] > 0
    xyz = torch.from_numpy(coord[:, :, :3]).float()
    hcce = normalized_xyz_to_hcce(xyz, iteration=bits, coord_min=-1.0, coord_max=1.0).numpy()
    # Show the most significant bit for x/y/z as RGB.
    panel = (hcce[:, :, [0, bits, bits * 2]] * 255).astype(np.uint8)
    panel[~valid] = 0
    return _resize_nn(panel, size)


def _save_grid(path, panels):
    max_h = max(p.shape[0] for p in panels)
    padded = []
    for p in panels:
        if p.shape[0] < max_h:
            pad = np.zeros((max_h - p.shape[0], p.shape[1], 3), dtype=np.uint8)
            p = np.concatenate([p, pad], axis=0)
        padded.append(p)
    Image.fromarray(np.concatenate(padded, axis=1)).save(path)


def visualize_surgpose(args, out_dir):
    random.seed(args.seed)
    np.random.seed(args.seed)
    ds = SurgPoseInstruments(
        split="train",
        training=True,
        img_size=args.img_size,
        n=args.surgpose_n,
        train_episodes=args.surgpose_train_episodes,
        aug_random_crop_rotate=True,
        aug_geom_prob=1.0,
        aug_crop_scale=args.aug_crop_scale,
        aug_max_angle=args.aug_max_angle,
        aug_offset_scale=args.aug_offset_scale,
        aug_color_jitter=bool(args.aug_color_jitter),
        dense_output_stride=2,
        load_depth=True,
        num_keypoints=args.num_keypoints,
        depth_normalize_relative=True,
        depth_norm_low_pct=args.depth_norm_low_pct,
        depth_norm_high_pct=args.depth_norm_high_pct,
    )
    img, annot = ds[0]
    assert annot["instruments"], "SurgPose sample has no instruments"
    rgb = _to_rgb(img)
    kp_rgb = rgb.copy()
    inst_union = np.zeros((args.img_size // 2, args.img_size // 2), dtype=np.uint8)
    part_union = np.zeros((args.img_size // 2, args.img_size // 2), dtype=np.uint8)
    depth_union = np.zeros((args.img_size // 2, args.img_size // 2), dtype=np.float32)
    depth_valid_union = np.zeros((args.img_size // 2, args.img_size // 2), dtype=np.uint8)
    visible_counts = []
    depth_ranges = []
    for inst in annot["instruments"]:
        points = inst.get("keypoints")
        valid = inst.get("keypoints_valid")
        assert points is not None and points.shape == (args.num_keypoints, 2)
        assert valid is not None and valid.shape == (args.num_keypoints,)
        assert valid.any(), "SurgPose keypoints are all invisible"
        kp_rgb = _draw_keypoints(kp_rgb, points, valid)
        inst_union = np.maximum(inst_union, (inst["inst_mask_dense"] > 0).astype(np.uint8))
        part_union[inst["part_mask_dense"] > 0] = inst["part_mask_dense"][inst["part_mask_dense"] > 0]
        assert inst.get("has_depth", False), "SurgPose depth supervision missing"
        dmask = inst["depth_valid_dense"] > 0
        dvals = inst["depth_map_dense"][dmask]
        assert dvals.size > 0 and float(dvals.min()) >= -1e-5 and float(dvals.max()) <= 1.0 + 1e-5
        depth_union[dmask] = inst["depth_map_dense"][dmask]
        depth_valid_union[dmask] = 1
        visible_counts.append(int(valid.sum()))
        depth_ranges.append([float(dvals.min()), float(dvals.max())])

    panels = [
        _title(kp_rgb, "surgpose rgb + native 7 keypoints"),
        _title(_overlay_binary(rgb, inst_union), "instance masks H/2"),
        _title(_overlay_part(rgb, part_union), "part seg H/2"),
        _title(_depth_panel(depth_union, depth_valid_union, args.img_size), "relative depth in inst masks"),
    ]
    _save_grid(out_dir / "surgpose_aug_supervision.jpg", panels)
    return {
        "imagename": annot["imagename"],
        "num_instruments": len(annot["instruments"]),
        "keypoints_shape_per_inst": [args.num_keypoints, 2],
        "keypoints_visible_per_inst": visible_counts,
        "depth_ranges_per_inst": depth_ranges,
    }


def visualize_rarp(args, out_dir):
    random.seed(args.seed + 1)
    np.random.seed(args.seed + 1)
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    ds = RARPInstanceDataset(
        split="train",
        training=True,
        img_size=args.img_size,
        dataset_root=args.rarp_dataset_root,
        pose_root=args.rarp_pose_root,
        min_dice=[args.min_dice_shaft, args.min_dice_wrist, args.min_dice_gripper],
        train_ratio=args.needle_train_ratio,
        subsample=args.rarp_subsample,
        n=args.rarp_n,
        v2_force=True,
        cse_coord_root=None,
        render_on_the_fly=True,
        canonicalize_pose_symmetry=True,
        canonical_eps=args.canonical_eps,
        aug_random_crop_rotate=True,
        aug_geom_prob=1.0,
        aug_crop_scale=args.aug_crop_scale,
        aug_max_angle=args.aug_max_angle,
        aug_offset_scale=args.aug_offset_scale,
        aug_color_jitter=bool(args.aug_color_jitter),
        random_resample=False,
        dense_output_stride=2,
    )
    img, annot = ds[0]
    assert annot["instruments"], "RARP sample has no instruments"
    rgb = _to_rgb(img)
    inst = annot["instruments"][0]
    coord = inst.get("coord_img_dense")
    assert coord is not None, "RARP HCCE coord_img_dense missing"
    assert coord.shape[:2] == (args.img_size // 2, args.img_size // 2)
    valid = coord[:, :, 3] > 0
    assert valid.any(), "RARP HCCE coord map has no valid pixels"

    panels = [
        _title(rgb, "rarp rgb after augmentation"),
        _title(_overlay_binary(rgb, inst["inst_mask_dense"]), "instance mask H/2"),
        _title(_overlay_part(rgb, inst["part_mask_dense"]), "part seg H/2"),
        _title(_coord_xyz_panel(coord, args.img_size), "HCCE coord xyz H/2"),
        _title(_hcce_bit_panel(coord, args.hcce_bits, args.img_size), "HCCE first bits x/y/z"),
    ]
    _save_grid(out_dir / "rarp_aug_hcce_supervision.jpg", panels)
    return {
        "imagename": annot["imagename"],
        "num_instruments": len(annot["instruments"]),
        "coord_shape": list(coord.shape),
        "hcce_valid_pixels": int(valid.sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="debug/multitask_supervision_unit")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--img_size", type=int, default=630)
    parser.add_argument("--num_keypoints", type=int, default=7)
    parser.add_argument("--surgpose_train_episodes", type=str, default="000000,000028")
    parser.add_argument("--surgpose_n", type=int, default=8)
    parser.add_argument("--rarp_dataset_root", type=str, default="/mnt/nas/share/shuojue/data/needlePuncture_videos")
    parser.add_argument("--rarp_pose_root", type=str, default="/mnt/nas/share/shuojue/data/needlePuncture_results")
    parser.add_argument("--rarp_n", type=int, default=16)
    parser.add_argument("--rarp_subsample", type=int, default=64)
    parser.add_argument("--min_dice_shaft", type=float, default=0.8)
    parser.add_argument("--min_dice_wrist", type=float, default=0.6)
    parser.add_argument("--min_dice_gripper", type=float, default=0.6)
    parser.add_argument("--needle_train_ratio", type=float, default=0.95)
    parser.add_argument("--canonical_eps", type=float, default=0.08)
    parser.add_argument("--aug_crop_scale", type=float, default=1.2)
    parser.add_argument("--aug_max_angle", type=float, default=float(np.pi / 6.0))
    parser.add_argument("--aug_offset_scale", type=float, default=1.0)
    parser.add_argument("--aug_color_jitter", type=int, default=1)
    parser.add_argument("--depth_norm_low_pct", type=float, default=2.0)
    parser.add_argument("--depth_norm_high_pct", type=float, default=98.0)
    parser.add_argument("--hcce_bits", type=int, default=8)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "surgpose": visualize_surgpose(args, out_dir),
        "rarp": visualize_rarp(args, out_dir),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved supervision debug images to {out_dir}")


if __name__ == "__main__":
    main()
