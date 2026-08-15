import json
import math
import random
import re
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset
import torchvision.transforms as tv_transforms

ROBOPEPP_ROOT = Path(__file__).resolve().parents[1]
MULTIHMR_ROOT = Path(__file__).resolve().parents[3]
if str(MULTIHMR_ROOT) not in sys.path:
    sys.path.insert(0, str(MULTIHMR_ROOT))
if str(ROBOPEPP_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBOPEPP_ROOT))

from instrument_geometry import (  # noqa: E402
    KEYPOINT_NAMES,
    crop_resize_pad_intrinsics,
    instrument_keypoints_camera_np,
    project_points_np,
)
from crop_geom_aug import maybe_random_rotate_crop  # noqa: E402
from pose_pnp import matrix_to_quat_wxyz_np  # noqa: E402

datasets_pkg = types.ModuleType("datasets")
datasets_pkg.__path__ = [str(MULTIHMR_ROOT / "datasets")]
sys.modules["datasets"] = datasets_pkg
from datasets.rarp_pose_canonicalization import canonicalize_pose_symmetry  # noqa: E402


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


def _resize_longer_side(rgb, crop_size):
    h, w = rgb.shape[:2]
    if w > h:
        new_w = int(crop_size)
        new_h = int(crop_size * h / w)
    else:
        new_h = int(crop_size)
        new_w = int(crop_size * w / h)
    out = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return out, (new_w, new_h)


def _pad_to_square(rgb, crop_size):
    h, w = rgb.shape[:2]
    pad_h = (crop_size - h) // 2
    pad_w = (crop_size - w) // 2
    padding = (
        (pad_h, crop_size - h - pad_h),
        (pad_w, crop_size - w - pad_w),
        (0, 0),
    )
    return np.pad(rgb, padding, mode="edge"), (pad_w, pad_h)


def _create_belief_maps(image_resolution, points, valid, sigma=2):
    width, height = image_resolution
    out = np.zeros((len(points), height, width), dtype=np.float32)
    radius = int(sigma * 2)
    for i, (point, is_valid) in enumerate(zip(points, valid)):
        if not bool(is_valid):
            continue
        u = int(point[0])
        v = int(point[1])
        if not (0 <= u < width and 0 <= v < height):
            continue
        x0 = max(0, u - radius)
        x1 = min(width, u + radius + 1)
        y0 = max(0, v - radius)
        y1 = min(height, v + radius + 1)
        for x in range(x0, x1):
            for y in range(y0, y1):
                out[i, y, x] = np.exp(-(((x - u) ** 2 + (y - v) ** 2) / (2.0 * sigma * sigma)))
    return out


def _apply_color_jitter(rgb):
    color_factor = 2.0 * random.random()
    c_high = 1.0 + color_factor
    c_low = 1.0 - color_factor
    arr = rgb.copy()
    arr[:, :, 0] = np.clip(arr[:, :, 0] * random.uniform(c_low, c_high), 0, 255)
    arr[:, :, 1] = np.clip(arr[:, :, 1] * random.uniform(c_low, c_high), 0, 255)
    arr[:, :, 2] = np.clip(arr[:, :, 2] * random.uniform(c_low, c_high), 0, 255)
    return np.asarray(Image.fromarray(arr.astype(np.uint8)))


def _occlusion_rect(img_shape, min_area=0.0, max_area=0.3, max_try_times=5):
    h, w = img_shape[:2]
    for _ in range(max_try_times + 1):
        synth_area = (random.random() * (max_area - min_area) + min_area) * w * h
        synth_ratio = random.random() * (2.0 - 0.5) + 0.5
        synth_h = int(max(1, np.sqrt(synth_area * synth_ratio)))
        synth_w = int(max(1, np.sqrt(synth_area / synth_ratio)))
        if synth_w < w and synth_h < h:
            x0 = int(random.random() * (w - synth_w))
            y0 = int(random.random() * (h - synth_h))
            return y0, synth_h, x0, synth_w
    raise RuntimeError("LND occlusion augmentation failed to sample a valid rectangle")


def _apply_occlusion(rgb):
    out = rgb.copy()
    y0, h, x0, w = _occlusion_rect(out.shape)
    out[y0 : y0 + h, x0 : x0 + w] = (np.random.rand(h, w, 3) * 255).astype(np.uint8)
    return out


def _pil_enhance(rgb, enhancer_cls, prob, factor_interval):
    if random.random() > prob:
        return rgb
    pil = Image.fromarray(rgb)
    return np.asarray(enhancer_cls(pil).enhance(random.uniform(*factor_interval)))


def _apply_rgb_aug(rgb):
    out = _pil_enhance(rgb, ImageEnhance.Sharpness, 0.6, (0.0, 50.0))
    out = _pil_enhance(out, ImageEnhance.Contrast, 0.6, (0.7, 1.8))
    out = _pil_enhance(out, ImageEnhance.Brightness, 0.6, (0.7, 1.8))
    out = _pil_enhance(out, ImageEnhance.Color, 0.6, (0.0, 4.0))
    return out.astype(np.uint8)


def _bbox_jitter_for_epoch(epoch):
    if epoch < 5:
        return 5.0
    if epoch < 15:
        return 10.0
    if epoch < 30:
        return 15.0
    if epoch < 50:
        return 20.0
    if epoch < 70:
        return 25.0
    return 30.0


def _bbox_shift_for_epoch(epoch, max_shift_px):
    max_shift_px = float(max_shift_px)
    if max_shift_px <= 0.0:
        return 0.0
    if epoch < 5:
        return min(max_shift_px, 4.0)
    if epoch < 15:
        return min(max_shift_px, 6.0)
    return max_shift_px


def _expand_bbox(bbox_min, bbox_max, width, height, padding_frac):
    if padding_frac <= 0.0:
        return bbox_min, bbox_max
    side = float(max(bbox_max[0] - bbox_min[0], bbox_max[1] - bbox_min[1]))
    pad = np.array([side * float(padding_frac), side * float(padding_frac)], dtype=np.float32)
    bbox_min = bbox_min - pad
    bbox_max = bbox_max + pad
    bbox_min = np.clip(bbox_min, [0.0, 0.0], [float(width - 1), float(height - 1)])
    bbox_max = np.clip(bbox_max, [1.0, 1.0], [float(width), float(height)])
    return bbox_min, bbox_max


def _mask_center(mask):
    if mask is None:
        return None
    ys, xs = np.where(np.asarray(mask, dtype=bool))
    if xs.size == 0:
        return None
    return np.array([float(xs.mean()), float(ys.mean())], dtype=np.float32)


def _shift_bbox_toward_wrist_center(bbox_min, bbox_max, wrist_mask, width, height, max_shift_px):
    max_shift_px = float(max_shift_px)
    wrist_center = _mask_center(wrist_mask)
    if wrist_center is None or max_shift_px <= 0.0:
        return bbox_min, bbox_max
    bbox_center = 0.5 * (bbox_min + bbox_max)
    random_offset = (np.random.rand(2).astype(np.float32) * 2.0 - 1.0) * max_shift_px
    target_center = wrist_center + random_offset
    delta = np.clip(target_center - bbox_center, -max_shift_px, max_shift_px).astype(np.float32)
    bbox_min = bbox_min + delta
    bbox_max = bbox_max + delta
    if bbox_min[0] < 0.0:
        bbox_max[0] -= bbox_min[0]
        bbox_min[0] = 0.0
    if bbox_min[1] < 0.0:
        bbox_max[1] -= bbox_min[1]
        bbox_min[1] = 0.0
    if bbox_max[0] > float(width):
        shift = bbox_max[0] - float(width)
        bbox_min[0] -= shift
        bbox_max[0] = float(width)
    if bbox_max[1] > float(height):
        shift = bbox_max[1] - float(height)
        bbox_min[1] -= shift
        bbox_max[1] = float(height)
    bbox_min = np.clip(bbox_min, [0.0, 0.0], [float(width - 1), float(height - 1)])
    bbox_max = np.clip(bbox_max, [1.0, 1.0], [float(width), float(height)])
    return bbox_min, bbox_max


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
    T_cam_lnd = np.asarray(T_cam_lnd, dtype=np.float64)
    if T_cam_lnd.shape == (4, 4):
        T_cam_lnd = T_cam_lnd[:3, :4]
    T_cam_lnd = T_cam_lnd.reshape(3, 4)
    R_cam_lnd = T_cam_lnd[:, :3]
    t_cam_lnd_mm = T_cam_lnd[:, 3]
    R_cam_repo = R_cam_lnd @ R_LND_FROM_REPO
    t_cam_repo_m = (R_cam_lnd @ T_LND_FROM_REPO_MM + t_cam_lnd_mm) / 1000.0
    quat = matrix_to_quat_wxyz_np(R_cam_repo)
    return {
        "rot": quat.astype(np.float32),
        "trans": t_cam_repo_m.astype(np.float32),
        "alpha": 0.0,
        "theta_l": 0.0,
        "theta_r": 0.0,
        "pose_sym_flipped": False,
    }


def _load_memory(path):
    path = Path(path)
    if path.suffix == ".json":
        with open(path, "r") as f:
            return json.load(f)
    return torch.load(path, map_location="cpu", weights_only=False)


def _resolve_lnd_visible_mask_path(split_root, frame_id):
    original_dir = Path(split_root) / "mask_original"
    visible_dir = Path(split_root) / "mask visible"
    candidates = [
        original_dir / f"{frame_id}.png",
        original_dir / f"{int(frame_id):06d}.png",
        visible_dir / f"{frame_id}.png",
        visible_dir / f"{int(frame_id):06d}.png",
        visible_dir / f"{int(frame_id):06d}_000000.png",
    ]
    for path in candidates:
        if path.is_file():
            return path
    matches = sorted(visible_dir.glob(f"{int(frame_id):06d}_*.png"))
    if matches:
        return matches[0]
    return candidates[-1]


def _memory_record_to_pose(record):
    pose = record["wrist_pose"]
    action = record.get("action", {})
    return {
        "rot": np.asarray(pose["quat_wxyz"], dtype=np.float32),
        "trans": np.asarray(pose["trans_m"], dtype=np.float32),
        "alpha": float(action.get("alpha", 0.0)),
        "theta_l": float(action.get("theta_l", 0.0)),
        "theta_r": float(action.get("theta_r", 0.0)),
        "pose_sym_flipped": False,
    }


class RoboPEPPSurgripeLNDInstrument(Dataset):
    """
    SurgRIPE LND instrument-pose dataset in the same crop/keypoint format as
    RoboPEPPRARPInstrument.

    TRAIN can read refined pseudo pose/action from refine_memory_pool.json.
    TEST defaults to direct GT wrist pose from LND pose files and zero action;
    training code evaluates that split with wrist-only losses.
    """

    def __init__(
        self,
        root="/mnt/iMVR/daiyun/Dataset/LND",
        split="TRAIN",
        training=False,
        crop_size=224,
        memory_path=None,
        use_memory_pose=True,
        canonicalize_pose_symmetry=True,
        canonical_eps=0.08,
        heatmap_sigma=2.0,
        color_jitter=True,
        rgb_augmentation=True,
        occlusion_augmentation=True,
        occlusion_prob=0.5,
        bbox_padding_frac=0.12,
        bbox_jitter=True,
        bbox_shift=True,
        bbox_shift_max_px=8.0,
        aug_random_crop_rotate=False,
        aug_geom_prob=0.3,
        aug_max_angle=math.pi / 6.0,
        frame_ids=None,
        subsample=1,
    ):
        super().__init__()
        self.root = Path(root)
        self.split = str(split)
        self.split_root = self.root / self.split
        self.training = bool(training)
        self.crop_size = int(crop_size)
        self.memory_path = str(memory_path) if memory_path else None
        self.use_memory_pose = bool(use_memory_pose)
        self.canonicalize_pose_symmetry = bool(canonicalize_pose_symmetry)
        self.canonical_eps = float(canonical_eps)
        self.heatmap_sigma = float(heatmap_sigma)
        self.color_jitter = bool(color_jitter)
        self.rgb_augmentation = bool(rgb_augmentation)
        self.occlusion_augmentation = bool(occlusion_augmentation)
        self.occlusion_prob = float(occlusion_prob)
        self.bbox_padding_frac = float(bbox_padding_frac)
        self.bbox_jitter = bool(bbox_jitter)
        self.bbox_shift = bool(bbox_shift)
        self.bbox_shift_max_px = float(bbox_shift_max_px)
        self.aug_random_crop_rotate = bool(aug_random_crop_rotate)
        self.aug_geom_prob = float(aug_geom_prob)
        self.aug_max_angle = float(aug_max_angle)
        self.epoch = 0
        self.K = load_lnd_intrinsics(self.split_root)
        self.memory = _load_memory(memory_path) if memory_path and self.use_memory_pose else {}

        selected = None if frame_ids is None else {int(v) for v in frame_ids}
        image_paths = sorted((self.split_root / "image").glob("*.png"), key=lambda p: int(p.stem))
        self.samples = []
        for image_path in image_paths[:: max(1, int(subsample))]:
            frame_id = int(image_path.stem)
            if selected is not None and frame_id not in selected:
                continue
            pose_path = self.split_root / "pose" / f"{frame_id}.npy"
            inst_path = self.split_root / "sam3_segmentation" / f"{frame_id}.png"
            part_path = self.split_root / "sam3_segmentaion_part" / f"{frame_id}.png"
            visible_path = _resolve_lnd_visible_mask_path(self.split_root, frame_id)
            if not (pose_path.is_file() and inst_path.is_file() and part_path.is_file() and visible_path.is_file()):
                continue
            if self.use_memory_pose and memory_path and str(frame_id) not in self.memory:
                continue
            self.samples.append((frame_id, image_path, pose_path, inst_path, part_path, visible_path))
        if not self.samples:
            raise RuntimeError(f"No SurgRIPE LND samples found under {self.split_root}")

        self.to_tensor = tv_transforms.Compose(
            [
                tv_transforms.ToTensor(),
                tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        src = self.memory_path if self.use_memory_pose and self.memory_path else "direct_lnd_gt_wrist"
        return (
            f"robopepp_surgripe_lnd_instrument: split={self.split} training={self.training} "
            f"N={len(self)} bbox_padding_frac={self.bbox_padding_frac} "
            f"bbox_jitter={self.bbox_jitter} bbox_shift={self.bbox_shift} "
            f"bbox_shift_max_px={self.bbox_shift_max_px} "
            f"random_rotate={self.aug_random_crop_rotate} "
            f"rotate_prob={self.aug_geom_prob} rotate_max_angle={self.aug_max_angle} "
            f"pose_source={src}"
        )

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _load_pose(self, frame_id, pose_path):
        if self.use_memory_pose and self.memory_path:
            pose = _memory_record_to_pose(self.memory[str(frame_id)])
        else:
            pose = lnd_pose_to_repo_wrist_pose(np.load(pose_path))
        pose_t = canonicalize_pose_symmetry(
            pose,
            eps=self.canonical_eps,
            enabled=self.canonicalize_pose_symmetry,
        )
        return {
            "rot": pose_t["rot"].detach().cpu().numpy().astype(np.float32),
            "trans": pose_t["trans"].detach().cpu().numpy().astype(np.float32),
            "alpha": float(pose_t["alpha"].reshape(-1)[0]),
            "theta_l": float(pose_t["theta_l"].reshape(-1)[0]),
            "theta_r": float(pose_t["theta_r"].reshape(-1)[0]),
            "pose_sym_flipped": bool(pose_t["pose_sym_flipped"].item()),
        }

    @staticmethod
    def _part_mask_rarp_style(lnd_part, visible_mask=None):
        out = np.zeros_like(lnd_part, dtype=np.uint8)
        out[lnd_part == 1] = 3
        out[lnd_part == 2] = 2
        out[lnd_part == 3] = 1
        if visible_mask is not None:
            visible = np.asarray(visible_mask, dtype=bool)
            out[(out == 2) & ~visible] = 0
        return out

    def _bbox_from_mask(self, inst_mask, width, height, part_mask=None):
        crop_mask = np.asarray(inst_mask, dtype=bool)
        if part_mask is not None:
            crop_mask = crop_mask | (np.asarray(part_mask) > 0)
        ys, xs = np.where(crop_mask)
        if xs.size == 0:
            raise RuntimeError("cannot crop an empty LND instance mask")
        bbox_min = np.array([float(xs.min()), float(ys.min())], dtype=np.float32)
        bbox_max = np.array([float(xs.max() + 1), float(ys.max() + 1)], dtype=np.float32)
        bbox_min, bbox_max = _expand_bbox(bbox_min, bbox_max, width, height, self.bbox_padding_frac)
        if self.training and self.bbox_jitter:
            jitter = _bbox_jitter_for_epoch(self.epoch)
            if jitter > 0.0:
                bbox_min = bbox_min - np.random.rand(2).astype(np.float32) * jitter
                bbox_max = bbox_max + np.random.rand(2).astype(np.float32) * jitter
        if self.training and self.bbox_shift:
            shift = _bbox_shift_for_epoch(self.epoch, self.bbox_shift_max_px)
            wrist_mask = None if part_mask is None else np.asarray(part_mask) == 2
            bbox_min, bbox_max = _shift_bbox_toward_wrist_center(
                bbox_min,
                bbox_max,
                wrist_mask,
                width,
                height,
                shift,
            )
        bbox_min = np.clip(bbox_min, [0.0, 0.0], [float(width - 1), float(height - 1)])
        bbox_max = np.clip(bbox_max, [1.0, 1.0], [float(width), float(height)])
        if bbox_max[0] <= bbox_min[0] or bbox_max[1] <= bbox_min[1]:
            raise RuntimeError(f"invalid LND crop bbox: min={bbox_min}, max={bbox_max}")
        return bbox_min, bbox_max

    @staticmethod
    def _visibility_for_keypoints(uv_orig, z, part_mask):
        h, w = part_mask.shape
        valid = np.zeros((len(KEYPOINT_NAMES),), dtype=bool)
        hit_labels = np.zeros((len(KEYPOINT_NAMES),), dtype=np.int64)
        allowed = {
            "shaft_axis": (3,),
            "wrist_shaft_joint": (2, 3),
            "wrist_gripper_joint": (1, 2),
            "left_tip": (1,),
            "right_tip": (1,),
        }
        for i, name in enumerate(KEYPOINT_NAMES):
            if not np.isfinite(uv_orig[i]).all() or z[i] <= 1e-4:
                continue
            x = int(round(float(uv_orig[i, 0])))
            y = int(round(float(uv_orig[i, 1])))
            if not (0 <= x < w and 0 <= y < h):
                continue
            y0, y1 = max(0, y - 2), min(h, y + 3)
            x0, x1 = max(0, x - 2), min(w, x + 3)
            patch = part_mask[y0:y1, x0:x1]
            hit = np.isin(patch, allowed[name])
            if hit.any():
                valid[i] = True
                vals, counts = np.unique(patch[hit], return_counts=True)
                hit_labels[i] = int(vals[np.argmax(counts)])
            else:
                hit_labels[i] = int(part_mask[y, x])
        return valid, hit_labels

    def __getitem__(self, idx):
        frame_id, image_path, pose_path, inst_path, part_path, visible_path = self.samples[idx]
        rgb = np.asarray(Image.open(image_path).convert("RGB"))
        inst_mask = np.asarray(Image.open(inst_path).convert("L")) > 0
        lnd_part = np.asarray(Image.open(part_path).convert("L"))
        visible_mask = np.asarray(Image.open(visible_path).convert("L")) > 0
        unoccluded_part_mask = self._part_mask_rarp_style(lnd_part, visible_mask=None)
        part_mask = self._part_mask_rarp_style(lnd_part, visible_mask=visible_mask)
        wrist_pixels_unoccluded = int(np.count_nonzero(unoccluded_part_mask == 2))
        wrist_pixels_visible = int(np.count_nonzero(part_mask == 2))
        wrist_visibility_fraction = float(wrist_pixels_visible / max(1, wrist_pixels_unoccluded))
        h, w = rgb.shape[:2]
        pose = self._load_pose(frame_id, pose_path)
        action = np.array([pose["alpha"], pose["theta_l"], pose["theta_r"]], dtype=np.float32)
        kp_3d_cam = instrument_keypoints_camera_np(pose["rot"], pose["trans"], action)
        kp_uv_orig = project_points_np(kp_3d_cam, self.K).astype(np.float32)
        kp_valid_orig, kp_hit_labels = self._visibility_for_keypoints(kp_uv_orig, kp_3d_cam[:, 2], part_mask)

        bbox_min, bbox_max = self._bbox_from_mask(inst_mask, w, h, part_mask=part_mask)
        x0, y0 = bbox_min.astype(np.int64)
        x1, y1 = np.ceil(bbox_max).astype(np.int64)
        crop = rgb[y0:y1, x0:x1]
        if crop.shape[0] <= 0 or crop.shape[1] <= 0:
            raise RuntimeError(f"empty LND crop for bbox min={bbox_min}, max={bbox_max}")

        crop_resized, (new_w, new_h) = _resize_longer_side(crop, self.crop_size)
        scale_x = float(new_w) / float(bbox_max[0] - bbox_min[0])
        scale_y = float(new_h) / float(bbox_max[1] - bbox_min[1])
        crop_square, (pad_w, pad_h) = _pad_to_square(crop_resized, self.crop_size)
        K_crop = crop_resize_pad_intrinsics(self.K, bbox_min=bbox_min, scale_xy=(scale_x, scale_y), pad_xy=(pad_w, pad_h))
        kp_crop = project_points_np(kp_3d_cam, K_crop).astype(np.float32)

        inside_crop = (
            (kp_crop[:, 0] >= 0.0)
            & (kp_crop[:, 0] < float(self.crop_size))
            & (kp_crop[:, 1] >= 0.0)
            & (kp_crop[:, 1] < float(self.crop_size))
        )
        kp_valid = kp_valid_orig & inside_crop
        crop_square, K_crop, kp_crop, kp_valid, crop_aug_affine = maybe_random_rotate_crop(
            crop_square,
            K_crop,
            kp_crop,
            kp_valid,
            training=self.training,
            enabled=self.aug_random_crop_rotate,
            prob=self.aug_geom_prob,
            max_angle=self.aug_max_angle,
        )
        heatmaps = _create_belief_maps((self.crop_size, self.crop_size), kp_crop, kp_valid, sigma=self.heatmap_sigma)

        aug_rgb = crop_square
        if self.training:
            if self.color_jitter and random.random() < 0.4:
                aug_rgb = _apply_color_jitter(aug_rgb)
            if self.occlusion_augmentation and random.random() < self.occlusion_prob:
                aug_rgb = _apply_occlusion(aug_rgb)
            if self.rgb_augmentation:
                aug_rgb = _apply_rgb_aug(aug_rgb)

        image_tensor = self.to_tensor(Image.fromarray(aug_rgb.astype(np.uint8)))
        target = {
            "action": torch.from_numpy(action),
            "wrist_quat": torch.from_numpy(pose["rot"].astype(np.float32)),
            "wrist_trans": torch.from_numpy(pose["trans"].astype(np.float32)),
            "heatmaps": torch.from_numpy(heatmaps),
            "keypoints_crop": torch.from_numpy(kp_crop.astype(np.float32)),
            "keypoints_orig": torch.from_numpy(kp_uv_orig.astype(np.float32)),
            "keypoints_3d_cam": torch.from_numpy(kp_3d_cam.astype(np.float32)),
            "keypoints_valid": torch.from_numpy(kp_valid.astype(np.bool_)),
            "keypoints_valid_orig": torch.from_numpy(kp_valid_orig.astype(np.bool_)),
            "keypoint_hit_labels": torch.from_numpy(kp_hit_labels.astype(np.int64)),
            "K": torch.from_numpy(K_crop.astype(np.float32)),
            "K_orig": torch.from_numpy(self.K.astype(np.float32)),
            "bbox_min": torch.from_numpy(bbox_min.astype(np.float32)),
            "bbox_max": torch.from_numpy(bbox_max.astype(np.float32)),
            "scale": torch.tensor([scale_x, scale_y], dtype=torch.float32),
            "pad": torch.tensor([pad_w, pad_h], dtype=torch.float32),
            "crop_aug_affine": torch.from_numpy(crop_aug_affine.astype(np.float32)),
            "pose_sym_flipped": torch.tensor(bool(pose["pose_sym_flipped"])),
            "video_name": "surgripe_lnd",
            "frame_id": str(frame_id),
            "instance_id": torch.tensor(0, dtype=torch.int64),
            "img_path": str(image_path),
            "orig_size": torch.tensor([h, w], dtype=torch.int64),
            "crop_rgb": torch.from_numpy(crop_square.astype(np.uint8)),
            "orig_rgb": torch.from_numpy(rgb.astype(np.uint8)),
            "part_mask_orig": torch.from_numpy(part_mask.astype(np.uint8)),
            "inst_mask_orig": torch.from_numpy(inst_mask.astype(np.bool_)),
            "wrist_visibility_fraction": torch.tensor(wrist_visibility_fraction, dtype=torch.float32),
        }
        return image_tensor, target
