import importlib.util
import hashlib
import os
import random
import sys
import types
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

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
    rarp_intrinsics,
)
from crop_geom_aug import maybe_random_rotate_crop  # noqa: E402

datasets_pkg = types.ModuleType("datasets")
datasets_pkg.__path__ = [str(MULTIHMR_ROOT / "datasets")]
sys.modules["datasets"] = datasets_pkg
from datasets.RarpInstanceDataset import RARPInstanceDataset, _resolve_mask_subfolder  # noqa: E402
from datasets.rarp_pose_canonicalization import canonicalize_pose_symmetry  # noqa: E402


class _PoseStub:
    def __setstate__(self, state):
        self.__dict__.update(state if isinstance(state, dict) else {})


_main_mod = sys.modules.get("__main__")
if _main_mod is not None and not hasattr(_main_mod, "Pose"):
    setattr(_main_mod, "Pose", _PoseStub)


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
    raise RuntimeError("RoboPEPP occlusion augmentation failed to sample a valid rectangle")


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


class RoboPEPPRARPInstrument(Dataset):
    """
    Crop-based single-instrument RARP dataset following RoboPEPP's DREAM crop,
    color jitter, occlusion and RGB augmentation sequence.
    """

    def __init__(
        self,
        dataset_root,
        pose_root,
        split="train",
        training=False,
        crop_size=224,
        train_ratio=0.95,
        subsample=1,
        min_dice=(0.8, 0.6, 0.6),
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
        aug_max_angle=np.pi / 6.0,
        cache_dir=None,
    ):
        super().__init__()
        self.dataset_root = dataset_root
        self.pose_root = pose_root
        self.split = split
        self.training = bool(training)
        self.crop_size = int(crop_size)
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
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.epoch = 0
        self.pose_cache = {}
        self.v2_force = False
        self.base = SimpleNamespace(v2_force=False)
        self.skipped_missing_files = 0

        cache_path = self._cache_path(dataset_root, pose_root, split, train_ratio, subsample, min_dice)
        if cache_path is not None and cache_path.is_file():
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
            self.samples = payload["samples"]
            self.v2_force = bool(payload.get("v2_force", False))
            self.skipped_missing_files = int(payload.get("skipped_missing_files", 0))
            self.base = SimpleNamespace(v2_force=self.v2_force)
        else:
            self.base = RARPInstanceDataset(
                split=split,
                training=False,
                img_size=630,
                dataset_root=dataset_root,
                pose_root=pose_root,
                min_dice=list(min_dice),
                train_ratio=train_ratio,
                subsample=subsample,
                cse_coord_root=None,
                render_on_the_fly=False,
                canonicalize_pose_symmetry=False,
                aug_random_crop_rotate=False,
                random_resample=False,
            )
            self.v2_force = bool(self.base.v2_force)
            self.samples = []
            for video_name, frame_id, instances in self.base.samples:
                for instance_id, instance_folder in instances:
                    if self._has_required_sample_files(video_name, frame_id, instance_folder):
                        self.samples.append((video_name, frame_id, int(instance_id), instance_folder))
                    else:
                        self.skipped_missing_files += 1
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "samples": self.samples,
                        "v2_force": self.v2_force,
                        "skipped_missing_files": self.skipped_missing_files,
                    },
                    cache_path,
                )
        if not self.samples:
            raise RuntimeError(f"RoboPEPPRARPInstrument is empty: root={dataset_root}, pose={pose_root}, split={split}")

        self.to_tensor = tv_transforms.Compose(
            [
                tv_transforms.ToTensor(),
                tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        return (
            f"robopepp_rarp_instrument: split={self.split} training={self.training} "
            f"N={len(self)} bbox_padding_frac={self.bbox_padding_frac} "
            f"bbox_jitter={self.bbox_jitter} bbox_shift={self.bbox_shift} "
            f"bbox_shift_max_px={self.bbox_shift_max_px} "
            f"random_rotate={self.aug_random_crop_rotate} "
            f"rotate_prob={self.aug_geom_prob} rotate_max_angle={self.aug_max_angle} "
            f"skipped_missing_files={self.skipped_missing_files} root={self.dataset_root}"
        )

    def _cache_path(self, dataset_root, pose_root, split, train_ratio, subsample, min_dice):
        if self.cache_dir is None:
            return None
        identity = "|".join(
            [
                str(Path(dataset_root).resolve()),
                str(Path(pose_root).resolve()),
                str(split),
                f"{float(train_ratio):.6f}",
                str(int(subsample)),
                ",".join(f"{float(x):.4f}" for x in min_dice),
                "mask_file_filter_v2",
            ]
        )
        digest = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:16]
        name = f"rarp_instrument_{Path(dataset_root).name}_{split}_{digest}.pt"
        return self.cache_dir / name

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _resolve_rgb_path(self, video_name, frame_id):
        video_folder = os.path.join(self.dataset_root, f"SARRARP502022_{video_name}")
        frames_folder = (
            os.path.join(video_folder, "frames_v2")
            if os.path.isdir(os.path.join(video_folder, "frames_v2")) and self.base.v2_force
            else os.path.join(video_folder, "frames")
        )
        for ext in ("png", "jpg"):
            img_path = os.path.join(frames_folder, f"{frame_id}.{ext}")
            if os.path.isfile(img_path):
                return img_path
        return None

    def _resolve_required_mask_dirs(self, instance_folder):
        return {
            "wrist": _resolve_mask_subfolder(instance_folder, "wrist", self.base.v2_force),
            "shaft": _resolve_mask_subfolder(instance_folder, "shaft", self.base.v2_force),
            "gripper": _resolve_mask_subfolder(instance_folder, "gripper", self.base.v2_force),
        }

    def _has_required_sample_files(self, video_name, frame_id, instance_folder):
        if self._resolve_rgb_path(video_name, frame_id) is None:
            return False
        mask_frame_id = f"{int(frame_id) - 1:05d}"
        for folder in self._resolve_required_mask_dirs(instance_folder).values():
            if folder is None:
                return False
            if not os.path.isfile(os.path.join(folder, f"{mask_frame_id}.png")):
                return False
        return True

    def _load_pose(self, video_name, frame_id, instance_id):
        key = (video_name, frame_id, int(instance_id))
        if key in self.pose_cache:
            return self.pose_cache[key]
        case_dir = os.path.join(self.pose_root, f"SARRARP502022_{video_name}_instance{instance_id}")
        mem_path = os.path.join(case_dir, "memory_pool.pth")
        if not os.path.isfile(mem_path):
            raise FileNotFoundError(f"memory_pool.pth not found: {mem_path}")
        memory_pool = torch.load(mem_path, map_location="cpu", weights_only=False)
        if frame_id not in memory_pool:
            raise KeyError(f"frame {frame_id} not in {mem_path}")
        pose = memory_pool[frame_id]["pose_info"]
        pose_dict = {
            "rot": pose.rot.detach().float().reshape(-1),
            "trans": pose.trans.detach().float().reshape(3),
            "alpha": pose.alpha.detach().float().reshape(-1),
            "theta_l": pose.theta_l.detach().float().reshape(-1),
            "theta_r": pose.theta_r.detach().float().reshape(-1),
        }
        pose_t = canonicalize_pose_symmetry(
            pose_dict,
            eps=self.canonical_eps,
            enabled=self.canonicalize_pose_symmetry,
        )
        out = {
            "rot": pose_t["rot"].detach().cpu().numpy().astype(np.float32),
            "trans": pose_t["trans"].detach().cpu().numpy().astype(np.float32),
            "alpha": float(pose_t["alpha"].reshape(-1)[0]),
            "theta_l": float(pose_t["theta_l"].reshape(-1)[0]),
            "theta_r": float(pose_t["theta_r"].reshape(-1)[0]),
            "pose_sym_flipped": bool(pose_t["pose_sym_flipped"].item()),
        }
        self.pose_cache[key] = out
        return out

    def _load_rgb_and_masks(self, video_name, frame_id, instance_id, instance_folder):
        img_path = self._resolve_rgb_path(video_name, frame_id)
        if img_path is None:
            raise FileNotFoundError(f"RGB frame not found for {video_name}/{frame_id}")

        rgb = np.asarray(Image.open(img_path).convert("RGB"))
        mask_frame_id = f"{int(frame_id) - 1:05d}"
        dirs = self._resolve_required_mask_dirs(instance_folder)
        for name, folder in dirs.items():
            if folder is None:
                raise FileNotFoundError(f"{name} mask folder not found under {instance_folder}")
        masks = {}
        for name, folder in dirs.items():
            path = os.path.join(folder, f"{mask_frame_id}.png")
            if not os.path.isfile(path):
                raise FileNotFoundError(f"{name} mask not found: {path}")
            masks[name] = np.asarray(Image.open(path).convert("L")) > 0
            if masks[name].shape != rgb.shape[:2]:
                raise ValueError(f"{name} mask shape {masks[name].shape} does not match RGB {rgb.shape[:2]}: {path}")

        part_mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
        part_mask[masks["shaft"]] = 3
        part_mask[masks["wrist"]] = 2
        part_mask[masks["gripper"]] = 1
        inst_mask = masks["shaft"] | masks["wrist"] | masks["gripper"]
        if not inst_mask.any():
            raise RuntimeError(f"empty instrument mask: {video_name}/{frame_id}/instance{instance_id}")
        return rgb, part_mask, inst_mask, img_path

    def _bbox_from_mask(self, inst_mask, width, height, wrist_mask=None):
        ys, xs = np.where(inst_mask)
        if xs.size == 0:
            raise RuntimeError("cannot crop an empty instrument mask")
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
            raise RuntimeError(f"invalid crop bbox: min={bbox_min}, max={bbox_max}")
        return bbox_min, bbox_max

    @staticmethod
    def _visibility_for_keypoints(uv_orig, z, part_mask):
        h, w = part_mask.shape
        valid = np.zeros((len(KEYPOINT_NAMES),), dtype=bool)
        hit_labels = np.zeros((len(KEYPOINT_NAMES),), dtype=np.int64)
        allowed = OrderedDict(
            [
                ("shaft_axis", (3,)),
                ("wrist_shaft_joint", (2, 3)),
                ("wrist_gripper_joint", (1, 2)),
                ("left_tip", (1,)),
                ("right_tip", (1,)),
            ]
        )
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
            labels = allowed[name]
            hit = np.isin(patch, labels)
            if hit.any():
                valid[i] = True
                vals, counts = np.unique(patch[hit], return_counts=True)
                hit_labels[i] = int(vals[np.argmax(counts)])
            else:
                hit_labels[i] = int(part_mask[y, x])
        return valid, hit_labels

    def __getitem__(self, idx):
        video_name, frame_id, instance_id, instance_folder = self.samples[idx]
        rgb, part_mask, inst_mask, img_path = self._load_rgb_and_masks(video_name, frame_id, instance_id, instance_folder)
        h, w = rgb.shape[:2]
        K_orig = rarp_intrinsics(w, h)
        pose = self._load_pose(video_name, frame_id, instance_id)
        action = np.array([pose["alpha"], pose["theta_l"], pose["theta_r"]], dtype=np.float32)
        kp_3d_cam = instrument_keypoints_camera_np(pose["rot"], pose["trans"], action)
        kp_uv_orig = project_points_np(kp_3d_cam, K_orig).astype(np.float32)
        kp_valid_orig, kp_hit_labels = self._visibility_for_keypoints(kp_uv_orig, kp_3d_cam[:, 2], part_mask)

        bbox_min, bbox_max = self._bbox_from_mask(inst_mask, w, h, wrist_mask=part_mask == 2)
        x0, y0 = bbox_min.astype(np.int64)
        x1, y1 = np.ceil(bbox_max).astype(np.int64)
        crop = rgb[y0:y1, x0:x1]
        crop_h, crop_w = crop.shape[:2]
        if crop_h <= 0 or crop_w <= 0:
            raise RuntimeError(f"empty crop for bbox min={bbox_min}, max={bbox_max}")

        kp_crop = kp_uv_orig.copy()
        kp_crop[:, 0] -= bbox_min[0]
        kp_crop[:, 1] -= bbox_min[1]
        crop_resized, (new_w, new_h) = _resize_longer_side(crop, self.crop_size)
        scale_x = float(new_w) / float(bbox_max[0] - bbox_min[0])
        scale_y = float(new_h) / float(bbox_max[1] - bbox_min[1])
        kp_crop[:, 0] *= scale_x
        kp_crop[:, 1] *= scale_y
        crop_square, (pad_w, pad_h) = _pad_to_square(crop_resized, self.crop_size)
        kp_crop[:, 0] += pad_w
        kp_crop[:, 1] += pad_h

        inside_crop = (
            (kp_crop[:, 0] >= 0.0)
            & (kp_crop[:, 0] < float(self.crop_size))
            & (kp_crop[:, 1] >= 0.0)
            & (kp_crop[:, 1] < float(self.crop_size))
        )
        kp_valid = kp_valid_orig & inside_crop
        K_crop = crop_resize_pad_intrinsics(K_orig, bbox_min=bbox_min, scale_xy=(scale_x, scale_y), pad_xy=(pad_w, pad_h))
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
        heatmaps = _create_belief_maps(
            (self.crop_size, self.crop_size),
            kp_crop,
            kp_valid,
            sigma=self.heatmap_sigma,
        )

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
            "K_orig": torch.from_numpy(K_orig.astype(np.float32)),
            "bbox_min": torch.from_numpy(bbox_min.astype(np.float32)),
            "bbox_max": torch.from_numpy(bbox_max.astype(np.float32)),
            "scale": torch.tensor([scale_x, scale_y], dtype=torch.float32),
            "pad": torch.tensor([pad_w, pad_h], dtype=torch.float32),
            "crop_aug_affine": torch.from_numpy(crop_aug_affine.astype(np.float32)),
            "pose_sym_flipped": torch.tensor(bool(pose["pose_sym_flipped"])),
            "video_name": video_name,
            "frame_id": str(frame_id),
            "instance_id": torch.tensor(instance_id, dtype=torch.int64),
            "img_path": img_path,
            "orig_size": torch.tensor([h, w], dtype=torch.int64),
            "crop_rgb": torch.from_numpy(crop_square.astype(np.uint8)),
            "orig_rgb": torch.from_numpy(rgb.astype(np.uint8)),
            "part_mask_orig": torch.from_numpy(part_mask.astype(np.uint8)),
            "inst_mask_orig": torch.from_numpy(inst_mask.astype(np.bool_)),
        }
        return image_tensor, target
