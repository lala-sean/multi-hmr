import re
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as tv_transforms

import sys

ROBOPEPP_ROOT = Path(__file__).resolve().parents[1]
if str(ROBOPEPP_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBOPEPP_ROOT))

from instrument_geometry import crop_resize_pad_intrinsics, instrument_keypoints_camera_np, project_points_np  # noqa: E402
from pose_pnp import matrix_to_quat_wxyz_np  # noqa: E402


R_LND_FROM_REPO = np.array(
    [
        [-3.965562107231e-04, -9.999972555239e-01, -2.309044784927e-03],
        [-9.999996414411e-01, 3.948273259691e-04, 7.491522775273e-04],
        [-7.482385475190e-04, 2.309341037986e-03, -9.999970535372e-01],
    ],
    dtype=np.float64,
)
T_LND_FROM_REPO_MM = np.array(
    [3.687278000000e-04, -1.340887690810e-01, 2.029389171000e-03],
    dtype=np.float64,
)


def _resize_longer_side(arr, crop_size, interpolation):
    h, w = arr.shape[:2]
    if w > h:
        new_w = int(crop_size)
        new_h = max(1, int(crop_size * h / w))
    else:
        new_h = int(crop_size)
        new_w = max(1, int(crop_size * w / h))
    out = cv2.resize(arr, (new_w, new_h), interpolation=interpolation)
    return out, (new_w, new_h)


def _pad_to_square(arr, crop_size, value=0):
    h, w = arr.shape[:2]
    pad_h = (int(crop_size) - h) // 2
    pad_w = (int(crop_size) - w) // 2
    if arr.ndim == 2:
        padding = ((pad_h, int(crop_size) - h - pad_h), (pad_w, int(crop_size) - w - pad_w))
    else:
        padding = (
            (pad_h, int(crop_size) - h - pad_h),
            (pad_w, int(crop_size) - w - pad_w),
            (0, 0),
        )
    mode = "edge" if arr.ndim == 3 else "constant"
    kwargs = {} if mode == "edge" else {"constant_values": value}
    return np.pad(arr, padding, mode=mode, **kwargs), (pad_w, pad_h)


def crop_resize_pad_array(arr, bbox_min, bbox_max, crop_size, interpolation, value=0):
    x0, y0 = np.asarray(bbox_min, dtype=np.float32).astype(np.int64)
    x1, y1 = np.ceil(np.asarray(bbox_max, dtype=np.float32)).astype(np.int64)
    crop = arr[y0:y1, x0:x1]
    resized, (new_w, new_h) = _resize_longer_side(crop, crop_size, interpolation)
    square, (pad_w, pad_h) = _pad_to_square(resized, crop_size, value=value)
    scale_x = float(new_w) / float(max(1e-6, bbox_max[0] - bbox_min[0]))
    scale_y = float(new_h) / float(max(1e-6, bbox_max[1] - bbox_min[1]))
    return square, (scale_x, scale_y), (pad_w, pad_h), (new_w, new_h)


def unproject_crop_map_to_orig(crop_map, orig_shape, bbox_min, bbox_max, resized_size, pad_xy, interpolation):
    h, w = int(orig_shape[0]), int(orig_shape[1])
    pad_x, pad_y = [int(v) for v in pad_xy]
    resized_w, resized_h = [int(v) for v in resized_size]
    x0, y0 = np.asarray(bbox_min, dtype=np.float32).astype(np.int64)
    x1, y1 = np.ceil(np.asarray(bbox_max, dtype=np.float32)).astype(np.int64)
    active = crop_map[pad_y : pad_y + resized_h, pad_x : pad_x + resized_w]
    restored = cv2.resize(active, (max(1, x1 - x0), max(1, y1 - y0)), interpolation=interpolation)
    out = np.zeros((h, w), dtype=restored.dtype)
    out[y0:y1, x0:x1] = restored[: y1 - y0, : x1 - x0]
    return out


def load_lnd_intrinsics(split_root):
    config_path = Path(split_root) / "config.yaml"
    text = config_path.read_text(encoding="utf-8")
    match = re.search(r"camera_matrix:.*?data:\s*\[([^\]]+)\]", text, flags=re.S)
    if not match:
        raise RuntimeError(f"Could not parse camera_matrix data from {config_path}")
    vals = [float(v) for v in re.split(r"[,\s]+", match.group(1).strip()) if v]
    if len(vals) != 9:
        raise RuntimeError(f"Expected 9 camera intrinsics values in {config_path}, got {len(vals)}")
    return np.asarray(vals, dtype=np.float32).reshape(3, 3)


def lnd_pose_to_repo_wrist_pose(T_cam_lnd):
    T_cam_lnd = np.asarray(T_cam_lnd, dtype=np.float64).reshape(3, 4)
    R_cam_lnd = T_cam_lnd[:, :3]
    t_cam_lnd_mm = T_cam_lnd[:, 3]
    R_cam_repo = R_cam_lnd @ R_LND_FROM_REPO
    t_cam_repo_m = (R_cam_lnd @ T_LND_FROM_REPO_MM + t_cam_lnd_mm) / 1000.0
    quat = matrix_to_quat_wxyz_np(R_cam_repo)
    return {
        "rot": quat.astype(np.float64),
        "trans": t_cam_repo_m.astype(np.float64),
        "alpha": 0.0,
        "theta_l": 0.0,
        "theta_r": 0.0,
    }


def pose_to_keypoints(pose, K):
    action = np.asarray([pose["alpha"], pose["theta_l"], pose["theta_r"]], dtype=np.float64)
    pts_cam = instrument_keypoints_camera_np(pose["rot"], pose["trans"], action)
    uv = project_points_np(pts_cam, K).astype(np.float32)
    valid = np.isfinite(uv).all(axis=1) & np.isfinite(pts_cam).all(axis=1) & (pts_cam[:, 2] > 1e-4)
    return pts_cam.astype(np.float32), uv.astype(np.float32), valid.astype(bool)


def _bbox_from_mask(mask, image_shape, bbox_scale=2.8, min_crop_size=192, margin_px=16):
    ys, xs = np.where(np.asarray(mask).astype(bool))
    if len(xs) == 0:
        raise RuntimeError("empty LND wrist mask")
    h, w = int(image_shape[0]), int(image_shape[1])
    x0, x1 = float(xs.min()), float(xs.max() + 1)
    y0, y1 = float(ys.min()), float(ys.max() + 1)
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    side = max(x1 - x0, y1 - y0, float(min_crop_size))
    side = side * float(bbox_scale) + 2.0 * float(margin_px)
    x0 = max(0.0, cx - 0.5 * side)
    y0 = max(0.0, cy - 0.5 * side)
    x1 = min(float(w), cx + 0.5 * side)
    y1 = min(float(h), cy + 0.5 * side)
    if x1 <= x0 + 1 or y1 <= y0 + 1:
        raise RuntimeError(f"invalid expanded LND crop bbox {(x0, y0, x1, y1)}")
    return np.asarray([x0, y0], dtype=np.float32), np.asarray([x1, y1], dtype=np.float32)


class SurgripeLNDDataset(Dataset):
    """
    LND TEST/TRAIN reader for crop-based single-instrument transfer.

    The dataset provides only a visible wrist mask.  We use it to build an
    expanded crop, and the model's instance head estimates the full instrument
    mask inside that crop.
    """

    def __init__(
        self,
        root="/mnt/iMVR/daiyun/Dataset/LND",
        split="TEST",
        crop_size=224,
        bbox_scale=1.8,
        min_crop_size=120,
        margin_px=16,
        frame_ids=None,
    ):
        self.root = Path(root)
        self.split = str(split)
        self.split_root = self.root / self.split
        self.crop_size = int(crop_size)
        self.bbox_scale = float(bbox_scale)
        self.min_crop_size = int(min_crop_size)
        self.margin_px = int(margin_px)
        self.K = load_lnd_intrinsics(self.split_root)
        self.to_tensor = tv_transforms.Compose(
            [
                tv_transforms.ToTensor(),
                tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        selected = None if frame_ids is None else {int(v) for v in frame_ids}
        image_paths = sorted((self.split_root / "image").glob("*.png"), key=lambda p: int(p.stem))
        self.samples = []
        for image_path in image_paths:
            frame_id = int(image_path.stem)
            if selected is not None and frame_id not in selected:
                continue
            pose_path = self.split_root / "pose" / f"{frame_id}.npy"
            mask_path = self.split_root / "mask visible" / f"{frame_id - 1:06d}_000000.png"
            if pose_path.is_file() and mask_path.is_file():
                self.samples.append((frame_id, image_path, pose_path, mask_path))
        if not self.samples:
            raise RuntimeError(f"No LND samples found under {self.split_root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_id, image_path, pose_path, mask_path = self.samples[idx]
        rgb = np.asarray(Image.open(image_path).convert("RGB"))
        wrist_mask = np.asarray(Image.open(mask_path).convert("L")) > 0
        h, w = rgb.shape[:2]
        bbox_min, bbox_max = _bbox_from_mask(
            wrist_mask,
            (h, w),
            bbox_scale=self.bbox_scale,
            min_crop_size=self.min_crop_size,
            margin_px=self.margin_px,
        )
        crop_rgb, scale_xy, pad_xy, resized_size = crop_resize_pad_array(
            rgb,
            bbox_min,
            bbox_max,
            self.crop_size,
            cv2.INTER_LINEAR,
            value=0,
        )
        wrist_crop, _, _, _ = crop_resize_pad_array(
            wrist_mask.astype(np.uint8),
            bbox_min,
            bbox_max,
            self.crop_size,
            cv2.INTER_NEAREST,
            value=0,
        )
        K_crop = crop_resize_pad_intrinsics(self.K, bbox_min, scale_xy, pad_xy)
        gt_pose = lnd_pose_to_repo_wrist_pose(np.load(pose_path))
        kp_3d, kp_orig, kp_valid_orig = pose_to_keypoints(gt_pose, self.K)
        kp_crop = kp_orig.copy()
        kp_crop[:, 0] = (kp_crop[:, 0] - bbox_min[0]) * scale_xy[0] + pad_xy[0]
        kp_crop[:, 1] = (kp_crop[:, 1] - bbox_min[1]) * scale_xy[1] + pad_xy[1]
        kp_valid_crop = (
            kp_valid_orig
            & (kp_crop[:, 0] >= 0.0)
            & (kp_crop[:, 0] < float(self.crop_size))
            & (kp_crop[:, 1] >= 0.0)
            & (kp_crop[:, 1] < float(self.crop_size))
        )
        target = {
            "dataset": "surgripe_lnd",
            "split": self.split,
            "frame_id": int(frame_id),
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "pose_path": str(pose_path),
            "orig_rgb": rgb,
            "crop_rgb": crop_rgb.astype(np.uint8),
            "wrist_mask_orig": wrist_mask.astype(np.uint8),
            "wrist_mask_crop": wrist_crop.astype(np.uint8),
            "gt_part_orig": (wrist_mask.astype(np.uint8) * 2),
            "gt_part_crop": (wrist_crop.astype(np.uint8) * 2),
            "K_orig": self.K.astype(np.float32),
            "K_crop": K_crop.astype(np.float32),
            "bbox_min": bbox_min.astype(np.float32),
            "bbox_max": bbox_max.astype(np.float32),
            "scale": np.asarray(scale_xy, dtype=np.float32),
            "pad": np.asarray(pad_xy, dtype=np.float32),
            "resized_size": np.asarray(resized_size, dtype=np.int32),
            "orig_size": np.asarray([h, w], dtype=np.int32),
            "gt_pose": gt_pose,
            "keypoints_3d_cam": kp_3d,
            "keypoints_orig": kp_orig,
            "keypoints_crop": kp_crop.astype(np.float32),
            "keypoints_valid": kp_valid_crop.astype(bool),
            "keypoints_valid_orig": kp_valid_orig.astype(bool),
        }
        return self.to_tensor(Image.fromarray(crop_rgb.astype(np.uint8))), target
