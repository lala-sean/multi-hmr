import importlib.util
import json
import os
import random
import sys
import time
import hashlib
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.transforms as tv_transforms
import yaml
from PIL import Image, ImageFile
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
ImageFile.LOAD_TRUNCATED_IMAGES = True

ROOT = Path(__file__).resolve().parent
ROBOPEPP_ROOT = ROOT / "submodules" / "RoboPEPP"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROBOPEPP_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBOPEPP_ROOT))


def _load_local_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_rarp_crop_module = _load_local_module(
    "hybrid_robopepp_rarp_hcce_crop",
    ROBOPEPP_ROOT / "datasets" / "rarp_hcce_crop.py",
)
_rarp_aug_module = _load_local_module(
    "hybrid_robopepp_rarp_instrument",
    ROBOPEPP_ROOT / "datasets" / "rarp_instrument.py",
)
_crop_model_module = _load_local_module(
    "hybrid_robopepp_hcce_crop_model",
    ROBOPEPP_ROOT / "models" / "hcce_crop_model.py",
)
_crop_loss_module = _load_local_module(
    "hybrid_robopepp_hcce_crop_loss",
    ROBOPEPP_ROOT / "loss_hcce_crop.py",
)

from datasets.surgpose_instruments import SURGPOSE_ROOT, _episode_ids, _lookup_key  # noqa: E402
from datasets.vos_endovis_instruments import (  # noqa: E402
    VOS_ENDOVIS17_IMAGE_ROOT,
    VOS_ENDOVIS17_ROOT,
    VOS_ENDOVIS18_IMAGE_ROOT,
    VOS_ENDOVIS18_ROOT,
    _PART_REMAP,
)
from loss_instrument import dice_loss  # noqa: E402
from multi_instrument.hcce_codec import normalized_xyz_to_hcce  # noqa: E402


RARPCropHCCEDataset = _rarp_crop_module.RARPCropHCCEDataset
CropHCCEDenseKeypointDPT = _crop_model_module.CropHCCEDenseKeypointDPT
focal_heatmap_loss = _crop_loss_module.focal_heatmap_loss
heatmap_argmax = _crop_loss_module.heatmap_argmax


PUNCTURE_DATASET_ROOT = "/mnt/nas/share/shuojue/data/needlePuncture_videos"
PUNCTURE_POSE_ROOT = "/mnt/nas/share/shuojue/data/needlePuncture_results"
GRASPING_DATASET_ROOT = "/mnt/nas/share/shuojue/data/needleGrasping_videos"
GRASPING_POSE_ROOT = "/mnt/nas/share/shuojue/data/needleGrasping_results"
KNOTTING_DATASET_ROOT = "/mnt/nas/share/shuojue/data/knotting_videos"
KNOTTING_POSE_ROOT = "/mnt/nas/share/shuojue/data/knotting_results"


def _parse_episode_list(value, default):
    if value is None or str(value).strip() == "":
        return list(default)
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _frame_index_from_stem(stem):
    return int(str(stem).split("_")[-1])


def _resize_longer_side_array(arr, crop_size, interpolation):
    h, w = arr.shape[:2]
    if h <= 0 or w <= 0:
        raise RuntimeError(f"empty crop before resize: shape={arr.shape}")
    if w > h:
        new_w = int(crop_size)
        new_h = max(1, int(round(crop_size * h / w)))
    else:
        new_h = int(crop_size)
        new_w = max(1, int(round(crop_size * w / h)))
    return cv2.resize(arr, (new_w, new_h), interpolation=interpolation), (new_w, new_h)


def _pad_array(arr, crop_size, value=0, edge=False):
    h, w = arr.shape[:2]
    pad_h = (crop_size - h) // 2
    pad_w = (crop_size - w) // 2
    if arr.ndim == 2:
        padding = ((pad_h, crop_size - h - pad_h), (pad_w, crop_size - w - pad_w))
    else:
        padding = ((pad_h, crop_size - h - pad_h), (pad_w, crop_size - w - pad_w), (0, 0))
    if edge:
        return np.pad(arr, padding, mode="edge"), (pad_w, pad_h)
    return np.pad(arr, padding, mode="constant", constant_values=value), (pad_w, pad_h)


def _bbox_from_mask(inst_mask, training, epoch, width, height):
    ys, xs = np.where(inst_mask > 0)
    if xs.size == 0:
        raise RuntimeError("cannot crop an empty instrument mask")
    bbox_min = np.array([float(xs.min()), float(ys.min())], dtype=np.float32)
    bbox_max = np.array([float(xs.max() + 1), float(ys.max() + 1)], dtype=np.float32)
    if training:
        jitter = _rarp_aug_module._bbox_jitter_for_epoch(epoch)
        if jitter > 0.0:
            bbox_min -= np.random.rand(2).astype(np.float32) * jitter
            bbox_max += np.random.rand(2).astype(np.float32) * jitter
    bbox_min = np.clip(bbox_min, [0.0, 0.0], [float(width - 1), float(height - 1)])
    bbox_max = np.clip(bbox_max, [1.0, 1.0], [float(width), float(height)])
    if bbox_max[0] <= bbox_min[0] or bbox_max[1] <= bbox_min[1]:
        raise RuntimeError(f"invalid crop bbox: min={bbox_min}, max={bbox_max}")
    return bbox_min, bbox_max


def _crop_resize_pad(arr, bbox_min, bbox_max, crop_size, interpolation, value=0, edge=False):
    x0, y0 = bbox_min.astype(np.int64)
    x1, y1 = np.ceil(bbox_max).astype(np.int64)
    crop = arr[y0:y1, x0:x1]
    resized, (new_w, new_h) = _resize_longer_side_array(crop, crop_size, interpolation)
    padded, (pad_w, pad_h) = _pad_array(resized, crop_size, value=value, edge=edge)
    scale_x = float(new_w) / float(bbox_max[0] - bbox_min[0])
    scale_y = float(new_h) / float(bbox_max[1] - bbox_min[1])
    return padded, (scale_x, scale_y), (pad_w, pad_h)


def _transform_keypoints(points, valid, bbox_min, scale_xy, pad_xy, crop_size):
    out = points.astype(np.float32).copy()
    out[:, 0] = (out[:, 0] - bbox_min[0]) * scale_xy[0] + pad_xy[0]
    out[:, 1] = (out[:, 1] - bbox_min[1]) * scale_xy[1] + pad_xy[1]
    inside = (
        np.isfinite(out).all(axis=1)
        & (out[:, 0] >= 0.0)
        & (out[:, 0] < float(crop_size))
        & (out[:, 1] >= 0.0)
        & (out[:, 1] < float(crop_size))
    )
    return out, (valid.astype(bool) & inside)


def _create_belief_maps(crop_size, points, valid, sigma=2.0):
    out = np.zeros((len(points), crop_size, crop_size), dtype=np.float32)
    radius = int(float(sigma) * 2)
    for i, (point, is_valid) in enumerate(zip(points, valid)):
        if not bool(is_valid):
            continue
        u = int(round(float(point[0])))
        v = int(round(float(point[1])))
        if u - radius < 0 or u + radius + 1 >= crop_size or v - radius < 0 or v + radius + 1 >= crop_size:
            continue
        xs = np.arange(u - radius, u + radius + 1, dtype=np.float32)
        ys = np.arange(v - radius, v + radius + 1, dtype=np.float32)
        xx, yy = np.meshgrid(xs, ys)
        out[i, v - radius : v + radius + 1, u - radius : u + radius + 1] = np.exp(
            -(((xx - u) ** 2 + (yy - v) ** 2) / (2.0 * float(sigma) * float(sigma)))
        )
    return out


def _present_mask_ids(mask_np):
    if mask_np.ndim == 3:
        mask_np = mask_np[..., 0]
    flat = mask_np.reshape(-1).astype(np.int64, copy=False)
    if flat.size == 0:
        return []
    counts = np.bincount(flat)
    return [int(v) for v in np.flatnonzero(counts) if int(v) != 0]


def _imagenet_tensor(rgb):
    transform = tv_transforms.Compose(
        [
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform(Image.fromarray(rgb.astype(np.uint8)))


class RARPHybridCropDataset(Dataset):
    def __init__(self, wrapped, dataset_name):
        self.wrapped = wrapped
        self.dataset_name = dataset_name

    def __len__(self):
        return len(self.wrapped)

    def __repr__(self):
        return f"hybrid_{self.dataset_name}: {self.wrapped}"

    def set_epoch(self, epoch):
        if hasattr(self.wrapped, "set_epoch"):
            self.wrapped.set_epoch(epoch)

    def __getitem__(self, idx):
        image, target = self.wrapped[idx]
        target["dataset_name"] = self.dataset_name
        target["has_pose"] = torch.tensor(True, dtype=torch.bool)
        target["has_keypoints"] = torch.tensor(False, dtype=torch.bool)
        target["keypoint_kind"] = "none"
        return image, target


class SurgPoseCropDataset(Dataset):
    def __init__(
        self,
        root_dir=SURGPOSE_ROOT,
        split="train",
        training=False,
        crop_size=224,
        subsample=1,
        train_episodes=None,
        val_episodes=None,
        num_keypoints=7,
        keypoint_file="keypoints_left_rectified.yaml",
        heatmap_sigma=2.0,
        color_jitter=True,
        rgb_augmentation=True,
        occlusion_augmentation=True,
        occlusion_prob=0.5,
        cache_dir=None,
        enumerate_instances=False,
    ):
        super().__init__()
        self.name = "surgpose"
        self.root_dir = root_dir
        self.split = split
        self.training = bool(training)
        self.crop_size = int(crop_size)
        self.subsample = int(subsample)
        self.num_keypoints = int(num_keypoints)
        self.keypoint_file = str(keypoint_file)
        self.heatmap_sigma = float(heatmap_sigma)
        self.color_jitter = bool(color_jitter)
        self.rgb_augmentation = bool(rgb_augmentation)
        self.occlusion_augmentation = bool(occlusion_augmentation)
        self.occlusion_prob = float(occlusion_prob)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.enumerate_instances = bool(enumerate_instances)
        self.epoch = 0
        self._keypoint_cache = {}

        default_train = _episode_ids(0, 27)
        default_val = _episode_ids(28, 33)
        episodes = (
            _parse_episode_list(train_episodes, default_train)
            if split == "train"
            else _parse_episode_list(val_episodes, default_val)
        )
        cache_path = self._cache_path(episodes)
        if cache_path is not None and cache_path.is_file():
            self.samples = torch.load(cache_path, map_location="cpu", weights_only=False)["samples"]
        else:
            self.samples = []
            for ep in episodes:
                proc = os.path.join(root_dir, ep, "processed_stereo_640")
                left_dir = os.path.join(proc, "left_frames")
                inst_dir = os.path.join(proc, "sam3_segmentation")
                part_dir = os.path.join(proc, "sam3_segmentaion_part")
                if not (os.path.isdir(left_dir) and os.path.isdir(inst_dir) and os.path.isdir(part_dir)):
                    continue
                frames = sorted(fn for fn in os.listdir(left_dir) if fn.endswith(".png"))
                for fn in frames:
                    if not (os.path.isfile(os.path.join(inst_dir, fn)) and os.path.isfile(os.path.join(part_dir, fn))):
                        continue
                    stem = os.path.splitext(fn)[0]
                    if not self.enumerate_instances:
                        self.samples.append((ep, stem, 0))
                    else:
                        inst_mask = np.asarray(Image.open(os.path.join(inst_dir, fn)))
                        for inst_id in sorted(np.unique(inst_mask).tolist()):
                            inst_id = int(inst_id)
                            if inst_id != 0 and np.any(inst_mask == inst_id):
                                self.samples.append((ep, stem, inst_id))
            if self.subsample > 1:
                self.samples = self.samples[:: self.subsample]
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"samples": self.samples}, cache_path)

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        return f"surgpose_crop: split={self.split} training={self.training} N={len(self)} keypoints={self.num_keypoints}"

    def _cache_path(self, episodes):
        if self.cache_dir is None:
            return None
        identity = "|".join(
            [
                str(Path(self.root_dir).resolve()),
                str(self.split),
                str(int(self.subsample)),
                ",".join(episodes),
                f"surgpose_crop_instances_v2_enumerate{int(self.enumerate_instances)}",
            ]
        )
        digest = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:16]
        return self.cache_dir / f"surgpose_crop_{self.split}_{digest}.pt"

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _load_episode_keypoints(self, proc):
        if proc in self._keypoint_cache:
            return self._keypoint_cache[proc]
        path = os.path.join(proc, self.keypoint_file)
        if not os.path.isfile(path):
            self._keypoint_cache[proc] = {}
            return self._keypoint_cache[proc]
        with open(path, "r") as f:
            self._keypoint_cache[proc] = yaml.safe_load(f) or {}
        return self._keypoint_cache[proc]

    def _keypoint_groups_for_frame(self, proc, stem, inst_mask):
        frame_data = _lookup_key(self._load_episode_keypoints(proc), _frame_index_from_stem(stem))
        if not frame_data:
            return {}
        inst_ids = [int(v) for v in sorted(np.unique(inst_mask).tolist()) if int(v) != 0]
        if not inst_ids:
            return {}
        dist_maps = {}
        for inst_id in inst_ids:
            outside = (inst_mask != inst_id).astype(np.uint8)
            dist_maps[inst_id] = cv2.distanceTransform(outside, cv2.DIST_L2, 3)

        grouped = {}
        for base_label in (1, 8):
            pts = np.zeros((self.num_keypoints, 2), dtype=np.float32)
            valid = np.zeros((self.num_keypoints,), dtype=bool)
            for local_idx in range(self.num_keypoints):
                raw = _lookup_key(frame_data, base_label + local_idx)
                if raw is None or len(raw) < 2:
                    continue
                x, y = float(raw[0]), float(raw[1])
                if np.isfinite(x) and np.isfinite(y):
                    pts[local_idx] = (x, y)
                    valid[local_idx] = True
            if not valid.any():
                continue
            h, w = inst_mask.shape[:2]
            scores = {}
            for inst_id in inst_ids:
                dvals = []
                dmap = dist_maps[inst_id]
                for x, y in pts[valid]:
                    xi = int(round(float(x)))
                    yi = int(round(float(y)))
                    dvals.append(float(dmap[yi, xi]) if 0 <= xi < w and 0 <= yi < h else 1e6)
                scores[inst_id] = float(np.mean(dvals)) if dvals else 1e6
            grouped[int(min(scores, key=scores.get))] = {
                "keypoints": pts,
                "keypoints_valid": valid,
            }
        return grouped

    def __getitem__(self, idx):
        ep, stem, inst_id = self.samples[idx]
        fn = f"{stem}.png"
        proc = os.path.join(self.root_dir, ep, "processed_stereo_640")
        rgb = np.asarray(Image.open(os.path.join(proc, "left_frames", fn)).convert("RGB"))
        inst_raw = np.asarray(Image.open(os.path.join(proc, "sam3_segmentation", fn)))
        part_raw = np.asarray(Image.open(os.path.join(proc, "sam3_segmentaion_part", fn)))
        if int(inst_id) == 0:
            inst_ids = [int(v) for v in sorted(np.unique(inst_raw).tolist()) if int(v) != 0]
            if not inst_ids:
                raise RuntimeError(f"no SurgPose instrument ids: {ep}/{stem}")
            inst_id = random.choice(inst_ids) if self.training else inst_ids[0]
        inst_mask = inst_raw == int(inst_id)
        part_mask = np.zeros_like(part_raw, dtype=np.uint8)
        part_mask[inst_mask & (part_raw == 1)] = 3
        part_mask[inst_mask & (part_raw == 2)] = 2
        part_mask[inst_mask & (part_raw == 3)] = 1
        if not inst_mask.any():
            raise RuntimeError(f"empty SurgPose instrument mask: {ep}/{stem}/{inst_id}")

        h, w = rgb.shape[:2]
        bbox_min, bbox_max = _bbox_from_mask(inst_mask, self.training, self.epoch, w, h)
        crop_rgb, scale_xy, pad_xy = _crop_resize_pad(
            rgb,
            bbox_min,
            bbox_max,
            self.crop_size,
            interpolation=cv2.INTER_LINEAR,
            edge=True,
        )
        inst_crop, _, _ = _crop_resize_pad(
            inst_mask.astype(np.uint8),
            bbox_min,
            bbox_max,
            self.crop_size,
            interpolation=cv2.INTER_NEAREST,
            value=0,
        )
        part_crop, _, _ = _crop_resize_pad(
            part_mask,
            bbox_min,
            bbox_max,
            self.crop_size,
            interpolation=cv2.INTER_NEAREST,
            value=0,
        )

        grouped = self._keypoint_groups_for_frame(proc, stem, inst_raw)
        keypoints = np.zeros((self.num_keypoints, 2), dtype=np.float32)
        valid = np.zeros((self.num_keypoints,), dtype=bool)
        if int(inst_id) in grouped:
            keypoints = grouped[int(inst_id)]["keypoints"].astype(np.float32)
            valid = grouped[int(inst_id)]["keypoints_valid"].astype(bool)
        keypoints_crop, valid_crop = _transform_keypoints(
            keypoints,
            valid,
            bbox_min,
            scale_xy,
            pad_xy,
            self.crop_size,
        )
        heatmaps = _create_belief_maps(self.crop_size, keypoints_crop, valid_crop, sigma=self.heatmap_sigma)

        aug_rgb = crop_rgb
        if self.training:
            if self.color_jitter and random.random() < 0.4:
                aug_rgb = _rarp_aug_module._apply_color_jitter(aug_rgb)
            if self.occlusion_augmentation and random.random() < self.occlusion_prob:
                aug_rgb = _rarp_aug_module._apply_occlusion(aug_rgb)
            if self.rgb_augmentation:
                aug_rgb = _rarp_aug_module._apply_rgb_aug(aug_rgb)

        target = {
            "dataset_name": "surgpose",
            "has_pose": torch.tensor(False, dtype=torch.bool),
            "has_keypoints": torch.tensor(bool(valid_crop.any()), dtype=torch.bool),
            "keypoint_kind": "surgpose",
            "inst_mask": torch.from_numpy(inst_crop.astype(np.float32)),
            "part_mask": torch.from_numpy(part_crop.astype(np.int64)),
            "coord_img": torch.zeros((self.crop_size, self.crop_size, 4), dtype=torch.float32),
            "has_cse": torch.tensor(False, dtype=torch.bool),
            "heatmaps": torch.from_numpy(heatmaps),
            "keypoints_crop": torch.from_numpy(keypoints_crop.astype(np.float32)),
            "keypoints_valid": torch.from_numpy(valid_crop.astype(np.bool_)),
            "action": torch.zeros(3, dtype=torch.float32),
            "wrist_quat": torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
            "wrist_trans": torch.zeros(3, dtype=torch.float32),
            "K": torch.eye(3, dtype=torch.float32),
            "bbox_min": torch.from_numpy(bbox_min.astype(np.float32)),
            "bbox_max": torch.from_numpy(bbox_max.astype(np.float32)),
            "scale": torch.tensor(scale_xy, dtype=torch.float32),
            "pad": torch.tensor(pad_xy, dtype=torch.float32),
            "orig_size": torch.tensor([h, w], dtype=torch.int64),
            "instance_id": torch.tensor(int(inst_id), dtype=torch.int64),
            "video_name": ep,
            "frame_id": stem,
            "img_path": os.path.join(proc, "left_frames", fn),
            "crop_rgb": torch.from_numpy(crop_rgb.astype(np.uint8)),
        }
        return _imagenet_tensor(aug_rgb), target


class VOSEndoVisCropDataset(Dataset):
    def __init__(
        self,
        label_root,
        image_root,
        split="train",
        training=False,
        crop_size=224,
        subsample=1,
        heatmap_sigma=2.0,
        color_jitter=True,
        rgb_augmentation=True,
        occlusion_augmentation=True,
        occlusion_prob=0.5,
        cache_dir=None,
        enumerate_instances=False,
    ):
        super().__init__()
        self.name = "vos_endovis17" if "17" in os.path.basename(label_root.rstrip("/")) else "vos_endovis18"
        self.label_root = label_root
        self.image_root = image_root
        self.split = split
        self.training = bool(training)
        self.crop_size = int(crop_size)
        self.subsample = int(subsample)
        self.heatmap_sigma = float(heatmap_sigma)
        self.color_jitter = bool(color_jitter)
        self.rgb_augmentation = bool(rgb_augmentation)
        self.occlusion_augmentation = bool(occlusion_augmentation)
        self.occlusion_prob = float(occlusion_prob)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.enumerate_instances = bool(enumerate_instances)
        self.epoch = 0
        self.seq_meta = {}
        self.samples = []

        cache_path = self._cache_path()
        if cache_path is not None and cache_path.is_file():
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
            self.samples = payload["samples"]
            self.seq_meta = payload.get("seq_meta", {})
        else:
            masks_root = os.path.join(label_root, split, "masks_new")
            for seq in sorted(os.listdir(masks_root)) if os.path.isdir(masks_root) else []:
                meta_path = os.path.join(label_root, split, "Meta_new", f"{seq}.json")
                meta = json.load(open(meta_path, "r")) if os.path.isfile(meta_path) else {"info": {"category": {}}}
                self.seq_meta[seq] = meta
                categories = meta.get("info", {}).get("category", {})
                seq_dir = os.path.join(masks_root, seq)
                frame_counter = 0
                for fn in sorted(os.listdir(seq_dir)):
                    if not fn.endswith(".png"):
                        continue
                    if self.subsample > 1 and frame_counter % self.subsample != 0:
                        frame_counter += 1
                        continue
                    frame_counter += 1
                    stem = os.path.splitext(fn)[0]
                    if not self.enumerate_instances:
                        mask_np = np.asarray(Image.open(os.path.join(seq_dir, fn)))
                        has_instrument = False
                        for inst_id in _present_mask_ids(mask_np):
                            if categories and categories.get(str(inst_id)) != "surgical instrument":
                                continue
                            has_instrument = True
                            break
                        if has_instrument:
                            self.samples.append((seq, stem, 0))
                    else:
                        mask_np = np.asarray(Image.open(os.path.join(seq_dir, fn)))
                        for inst_id in _present_mask_ids(mask_np):
                            if categories and categories.get(str(inst_id)) != "surgical instrument":
                                continue
                            if np.any(mask_np == inst_id):
                                self.samples.append((seq, stem, inst_id))
            if self.subsample > 1 and self.enumerate_instances:
                self.samples = self.samples[:: self.subsample]
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"samples": self.samples, "seq_meta": self.seq_meta}, cache_path)

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        return f"{self.name}_crop: split={self.split} training={self.training} N={len(self)}"

    def _cache_path(self):
        if self.cache_dir is None:
            return None
        identity = "|".join(
            [
                str(Path(self.label_root).resolve()),
                str(Path(self.image_root).resolve()),
                str(self.split),
                str(int(self.subsample)),
                f"vos_crop_instances_v5_enumerate{int(self.enumerate_instances)}",
            ]
        )
        digest = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:16]
        return self.cache_dir / f"{self.name}_crop_{self.split}_{digest}.pt"

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __getitem__(self, idx):
        seq, stem, inst_id = self.samples[idx]
        fn = f"{stem}.png"
        img_path = os.path.join(self.image_root, self.split, "images", seq, fn)
        if not os.path.isfile(img_path) and self.split == "valid":
            img_path = os.path.join(self.image_root, "test", "images", seq, fn)
        rgb = np.asarray(Image.open(img_path).convert("RGB"))
        mask_np = np.asarray(Image.open(os.path.join(self.label_root, self.split, "masks_new", seq, fn)))
        part_raw = np.asarray(Image.open(os.path.join(self.label_root, self.split, "parts_new", seq, fn)))
        if int(inst_id) == 0:
            categories = self.seq_meta.get(seq, {}).get("info", {}).get("category", {})
            inst_ids = []
            for value in _present_mask_ids(mask_np):
                if categories and categories.get(str(value)) != "surgical instrument":
                    continue
                inst_ids.append(value)
            if not inst_ids:
                raise RuntimeError(f"no VOS surgical instrument ids: {seq}/{stem}")
            inst_id = random.choice(inst_ids) if self.training else inst_ids[0]
        inst_mask = mask_np == int(inst_id)
        part_mask = np.zeros_like(part_raw, dtype=np.uint8)
        for raw_value, remapped in _PART_REMAP.items():
            part_mask[inst_mask & (part_raw == raw_value)] = remapped
        if not inst_mask.any():
            raise RuntimeError(f"empty VOS instrument mask: {seq}/{stem}/{inst_id}")

        h, w = rgb.shape[:2]
        bbox_min, bbox_max = _bbox_from_mask(inst_mask, self.training, self.epoch, w, h)
        crop_rgb, scale_xy, pad_xy = _crop_resize_pad(
            rgb,
            bbox_min,
            bbox_max,
            self.crop_size,
            interpolation=cv2.INTER_LINEAR,
            edge=True,
        )
        inst_crop, _, _ = _crop_resize_pad(
            inst_mask.astype(np.uint8),
            bbox_min,
            bbox_max,
            self.crop_size,
            interpolation=cv2.INTER_NEAREST,
            value=0,
        )
        part_crop, _, _ = _crop_resize_pad(
            part_mask,
            bbox_min,
            bbox_max,
            self.crop_size,
            interpolation=cv2.INTER_NEAREST,
            value=0,
        )
        aug_rgb = crop_rgb
        if self.training:
            if self.color_jitter and random.random() < 0.4:
                aug_rgb = _rarp_aug_module._apply_color_jitter(aug_rgb)
            if self.occlusion_augmentation and random.random() < self.occlusion_prob:
                aug_rgb = _rarp_aug_module._apply_occlusion(aug_rgb)
            if self.rgb_augmentation:
                aug_rgb = _rarp_aug_module._apply_rgb_aug(aug_rgb)

        target = {
            "dataset_name": self.name,
            "has_pose": torch.tensor(False, dtype=torch.bool),
            "has_keypoints": torch.tensor(False, dtype=torch.bool),
            "keypoint_kind": "none",
            "inst_mask": torch.from_numpy(inst_crop.astype(np.float32)),
            "part_mask": torch.from_numpy(part_crop.astype(np.int64)),
            "coord_img": torch.zeros((self.crop_size, self.crop_size, 4), dtype=torch.float32),
            "has_cse": torch.tensor(False, dtype=torch.bool),
            "heatmaps": torch.zeros((7, self.crop_size, self.crop_size), dtype=torch.float32),
            "keypoints_crop": torch.zeros((7, 2), dtype=torch.float32),
            "keypoints_valid": torch.zeros((7,), dtype=torch.bool),
            "action": torch.zeros(3, dtype=torch.float32),
            "wrist_quat": torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
            "wrist_trans": torch.zeros(3, dtype=torch.float32),
            "K": torch.eye(3, dtype=torch.float32),
            "bbox_min": torch.from_numpy(bbox_min.astype(np.float32)),
            "bbox_max": torch.from_numpy(bbox_max.astype(np.float32)),
            "scale": torch.tensor(scale_xy, dtype=torch.float32),
            "pad": torch.tensor(pad_xy, dtype=torch.float32),
            "orig_size": torch.tensor([h, w], dtype=torch.int64),
            "instance_id": torch.tensor(int(inst_id), dtype=torch.int64),
            "video_name": seq,
            "frame_id": stem,
            "img_path": img_path,
            "crop_rgb": torch.from_numpy(crop_rgb.astype(np.uint8)),
        }
        return _imagenet_tensor(aug_rgb), target


def collate_fn_hybrid_crop(batch):
    images, targets = zip(*batch)
    x = torch.stack(images, dim=0)
    tensor_keys = [
        "action",
        "wrist_quat",
        "wrist_trans",
        "K",
        "inst_mask",
        "part_mask",
        "coord_img",
        "has_cse",
        "has_pose",
        "has_keypoints",
        "heatmaps",
        "keypoints_crop",
        "keypoints_valid",
        "bbox_min",
        "bbox_max",
        "scale",
        "pad",
        "orig_size",
        "instance_id",
    ]
    y = {}
    for key in tensor_keys:
        vals = []
        for t in targets:
            if key in t:
                val = t[key]
            elif key == "has_pose":
                val = torch.tensor(False, dtype=torch.bool)
            elif key == "has_keypoints":
                val = torch.tensor(False, dtype=torch.bool)
            elif key == "heatmaps":
                val = torch.zeros((7, 224, 224), dtype=torch.float32)
            elif key == "keypoints_crop":
                val = torch.zeros((7, 2), dtype=torch.float32)
            elif key == "keypoints_valid":
                val = torch.zeros((7,), dtype=torch.bool)
            elif key == "coord_img":
                val = torch.zeros((224, 224, 4), dtype=torch.float32)
            elif key == "has_cse":
                val = torch.tensor(False, dtype=torch.bool)
            else:
                raise KeyError(f"missing target key: {key}")

            if key in ("heatmaps", "keypoints_crop", "keypoints_valid") and t.get("keypoint_kind") != "surgpose":
                if key == "heatmaps":
                    val = torch.zeros((7, 224, 224), dtype=torch.float32)
                elif key == "keypoints_crop":
                    val = torch.zeros((7, 2), dtype=torch.float32)
                else:
                    val = torch.zeros((7,), dtype=torch.bool)
            vals.append(val)
        y[key] = torch.stack(vals, dim=0)
    y["dataset_name"] = [t.get("dataset_name", "") for t in targets]
    y["video_name"] = [t.get("video_name", "") for t in targets]
    y["frame_id"] = [t.get("frame_id", "") for t in targets]
    y["img_path"] = [t.get("img_path", "") for t in targets]
    return x, y


def _part_target_to_dense_order(part_mask):
    target = torch.full_like(part_mask.long(), -1)
    target[part_mask == 2] = 0
    target[part_mask == 1] = 1
    target[part_mask == 3] = 2
    return target


def _compute_hcce_loss(out, y, args):
    pred = out["hcce_logits"]
    coord_imgs = y["coord_img"].float()
    inst = y["inst_mask"].float()
    has_cse = y["has_cse"].bool()
    coord_min = float(getattr(args, "hcce_coord_min", -1.0))
    coord_max = float(getattr(args, "hcce_coord_max", 1.0))
    losses = []
    bit_correct = torch.tensor(0.0, device=pred.device)
    bit_count = torch.tensor(0.0, device=pred.device)
    for k in torch.where(has_cse)[0]:
        coord = coord_imgs[k]
        valid = (coord[..., 3].long() > 0) & (inst[k] > 0.5)
        if not valid.any():
            continue
        gt_hcce = normalized_xyz_to_hcce(
            coord[..., :3],
            iteration=args.hcce_bits,
            coord_min=coord_min,
            coord_max=coord_max,
        )
        target = gt_hcce.permute(2, 0, 1)
        losses.append(F.l1_loss(pred[k, :, valid], target[:, valid] * 2.0 - 1.0))
        pred_bits = (torch.sigmoid(pred[k].detach()) > 0.5).float()
        gt_bits = (target.detach() > 0.5).float()
        bit_correct = bit_correct + (pred_bits[:, valid] == gt_bits[:, valid]).float().sum()
        bit_count = bit_count + gt_bits[:, valid].numel()
    loss = torch.stack(losses).mean() if losses else torch.tensor(0.0, device=pred.device)
    return torch.nan_to_num(loss), bit_correct / bit_count.clamp_min(1.0)


def compute_hybrid_crop_losses(out, y, args):
    inst_gt = y["inst_mask"].float()
    inst_loss_dice = dice_loss(torch.sigmoid(out["inst_mask_logits"]), inst_gt)
    inst_loss_bce = F.binary_cross_entropy_with_logits(out["inst_mask_logits"], inst_gt)

    part_target = _part_target_to_dense_order(y["part_mask"])
    part_valid = (part_target >= 0) & (inst_gt > 0.5)
    if part_valid.any():
        logits_flat = out["part_mask_logits"].permute(0, 2, 3, 1)[part_valid]
        target_flat = part_target[part_valid]
        part_ce = F.cross_entropy(logits_flat, target_flat)
        part_acc = (logits_flat.detach().argmax(dim=1) == target_flat).float().mean()
    else:
        part_ce = torch.tensor(0.0, device=inst_gt.device)
        part_acc = torch.tensor(0.0, device=inst_gt.device)

    hcce_loss, hcce_bit_acc = _compute_hcce_loss(out, y, args)

    has_kp = y["has_keypoints"].bool()
    if has_kp.any():
        hm_loss = focal_heatmap_loss(out["keypoint_heatmaps"][has_kp], y["heatmaps"][has_kp].float())
        kp_pred = heatmap_argmax(out["keypoint_heatmaps"][has_kp])
        valid = y["keypoints_valid"][has_kp].bool()
        if valid.any():
            kp_err = torch.linalg.norm(kp_pred[valid] - y["keypoints_crop"][has_kp][valid], dim=-1).mean()
        else:
            kp_err = torch.tensor(0.0, device=inst_gt.device)
    else:
        hm_loss = torch.tensor(0.0, device=inst_gt.device)
        kp_err = torch.tensor(0.0, device=inst_gt.device)

    has_pose = y["has_pose"].bool()
    if has_pose.any():
        pred_quat = F.normalize(out["wrist_quat_pred"][has_pose], p=2, dim=1)
        gt_quat = F.normalize(y["wrist_quat"][has_pose], p=2, dim=1)
        sign = torch.where((pred_quat * gt_quat).sum(dim=1, keepdim=True) < 0.0, -1.0, 1.0)
        quat_l1 = F.l1_loss(pred_quat, gt_quat * sign)
        action_l1 = F.l1_loss(out["action_pred"][has_pose], y["action"][has_pose])
        trans_l1 = F.l1_loss(out["wrist_trans_pred"][has_pose], y["wrist_trans"][has_pose])
    else:
        quat_l1 = torch.tensor(0.0, device=inst_gt.device)
        action_l1 = torch.tensor(0.0, device=inst_gt.device)
        trans_l1 = torch.tensor(0.0, device=inst_gt.device)

    total = (
        args.alpha_dice * inst_loss_dice
        + args.alpha_bce_mask * inst_loss_bce
        + args.alpha_part * part_ce
        + args.alpha_hcce * hcce_loss
        + args.alpha_heatmap * hm_loss
        + args.alpha_action_l1 * action_l1
        + args.alpha_wrist_quat_l1 * quat_l1
        + args.alpha_wrist_trans_l1 * trans_l1
    )
    metrics = {
        "total": total.detach(),
        "dice": inst_loss_dice.detach(),
        "bce_mask": inst_loss_bce.detach(),
        "part_ce": part_ce.detach(),
        "part_acc": part_acc.detach(),
        "hcce": hcce_loss.detach(),
        "hcce_bit_acc": hcce_bit_acc.detach(),
        "heatmap": hm_loss.detach(),
        "kp_err_px": kp_err.detach(),
        "action_l1": action_l1.detach(),
        "wrist_quat_l1": quat_l1.detach(),
        "wrist_trans_l1": trans_l1.detach(),
        "n_pose": has_pose.float().sum().detach(),
        "n_kp": has_kp.float().sum().detach(),
        "n_cse": y["has_cse"].float().sum().detach(),
    }
    return total, metrics


class AverageMeter:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    @property
    def avg(self):
        return self.sum / max(1, self.count)

    def update(self, value, n=1):
        self.sum += float(value) * int(n)
        self.count += int(n)


def setup_dist():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    return device, local_rank, world_size


def is_main_process():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def reduce_float(value, device):
    t = torch.tensor(float(value), device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= dist.get_world_size()
    return float(t.item())


def set_epoch_recursive(dataset, epoch):
    if hasattr(dataset, "set_epoch"):
        dataset.set_epoch(epoch)
    if isinstance(dataset, ConcatDataset):
        for child in dataset.datasets:
            set_epoch_recursive(child, epoch)


def save_checkpoint(path, model, optimizer, scheduler, scaler, args, epoch, iteration):
    raw = model.module if isinstance(model, DDP) else model
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": raw.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "args": vars(args),
            "epoch": epoch,
            "iter": iteration,
        },
        path,
    )


def load_compatible_state(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model_state = model.state_dict()
    compatible = {}
    skipped = []
    for key, value in state.items():
        if key in model_state and tuple(model_state[key].shape) == tuple(value.shape):
            compatible[key] = value
        else:
            skipped.append(key)
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    return ckpt, {"loaded": len(compatible), "skipped": skipped, "missing": missing, "unexpected": unexpected}


def make_rarp_crop_dataset(args, name, split, training, root, pose_root, subsample):
    ds = RARPCropHCCEDataset(
        root,
        pose_root,
        split=split,
        training=training,
        crop_size=args.img_size,
        train_ratio=args.needle_train_ratio,
        subsample=subsample,
        min_dice=(args.min_dice_shaft, args.min_dice_wrist, args.min_dice_gripper),
        canonicalize_pose_symmetry=bool(args.canonicalize_pose_symmetry),
        canonical_eps=args.canonical_eps,
        heatmap_sigma=args.heatmap_sigma,
        color_jitter=bool(args.color_jitter),
        rgb_augmentation=bool(args.rgb_augmentation),
        occlusion_augmentation=bool(args.occlusion_augmentation),
        occlusion_prob=args.occlusion_prob,
        cache_dir=args.dataset_cache_dir,
        cse_coord_root=args.cse_coord_root,
        render_on_the_fly=bool(args.render_on_the_fly),
        coord_render_backend=args.coord_render_backend,
        require_cse=True,
    )
    return RARPHybridCropDataset(ds, name)


def make_vos_datasets(args, split, training, subsample):
    out = []
    if not args.use_vos_endovis:
        return out
    if os.path.isdir(args.vos_endovis17_root):
        out.append(
            VOSEndoVisCropDataset(
                args.vos_endovis17_root,
                args.vos_endovis17_image_root,
                split=split if split != "valid" else "valid",
                training=training,
                crop_size=args.img_size,
                subsample=subsample,
                heatmap_sigma=args.heatmap_sigma,
                color_jitter=bool(args.color_jitter),
                rgb_augmentation=bool(args.rgb_augmentation),
                occlusion_augmentation=bool(args.occlusion_augmentation),
                occlusion_prob=args.occlusion_prob,
                cache_dir=args.dataset_cache_dir,
            )
        )
    if os.path.isdir(args.vos_endovis18_root):
        split18 = split if split in ("train", "test") else "test"
        out.append(
            VOSEndoVisCropDataset(
                args.vos_endovis18_root,
                args.vos_endovis18_image_root,
                split=split18,
                training=training,
                crop_size=args.img_size,
                subsample=subsample,
                heatmap_sigma=args.heatmap_sigma,
                color_jitter=bool(args.color_jitter),
                rgb_augmentation=bool(args.rgb_augmentation),
                occlusion_augmentation=bool(args.occlusion_augmentation),
                occlusion_prob=args.occlusion_prob,
                cache_dir=args.dataset_cache_dir,
            )
        )
    return out


def make_train_datasets(args):
    datasets = []
    specs = {
        "needlePuncture": (args.needle_puncture_data_dir, args.needle_puncture_pose_dir),
        "needleGrasping": (args.needle_grasping_data_dir, args.needle_grasping_pose_dir),
        "knotting": (args.knotting_data_dir, args.knotting_pose_dir),
    }
    for name in [n.strip() for n in args.train_rarp_datasets.split(",") if n.strip()]:
        if name not in specs:
            raise ValueError(f"Unknown RARP dataset {name}; choices={sorted(specs)}")
        datasets.append(make_rarp_crop_dataset(args, name, "train", True, specs[name][0], specs[name][1], args.train_subsample))
    if args.use_surgpose:
        datasets.append(
            SurgPoseCropDataset(
                root_dir=args.surgpose_root,
                split="train",
                training=True,
                crop_size=args.img_size,
                subsample=args.train_subsample,
                train_episodes=args.surgpose_train_episodes,
                val_episodes=args.surgpose_val_episodes,
                num_keypoints=args.num_keypoints,
                heatmap_sigma=args.heatmap_sigma,
                color_jitter=bool(args.color_jitter),
                rgb_augmentation=bool(args.rgb_augmentation),
                occlusion_augmentation=bool(args.occlusion_augmentation),
                occlusion_prob=args.occlusion_prob,
                cache_dir=args.dataset_cache_dir,
            )
        )
    datasets.extend(make_vos_datasets(args, "train", True, args.train_subsample))
    return datasets


def make_val_datasets(args):
    datasets = [
        make_rarp_crop_dataset(
            args,
            "needlePuncture",
            "test",
            False,
            args.needle_puncture_data_dir,
            args.needle_puncture_pose_dir,
            args.val_subsample,
        )
    ]
    if args.use_surgpose:
        datasets.append(
            SurgPoseCropDataset(
                root_dir=args.surgpose_root,
                split="valid",
                training=False,
                crop_size=args.img_size,
                subsample=args.val_subsample,
                train_episodes=args.surgpose_train_episodes,
                val_episodes=args.surgpose_val_episodes,
                num_keypoints=args.num_keypoints,
                heatmap_sigma=args.heatmap_sigma,
                color_jitter=False,
                rgb_augmentation=False,
                occlusion_augmentation=False,
                cache_dir=args.dataset_cache_dir,
            )
        )
    datasets.extend(make_vos_datasets(args, "valid", False, args.val_subsample))
    return datasets


@torch.no_grad()
def evaluate(model, loader, device, args, max_batches=20):
    model.eval()
    meters = {}
    count = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in y.items()}
        with torch.amp.autocast("cuda", enabled=bool(args.amp), dtype=torch.bfloat16):
            out = model(x, y["K"])
            _, metrics = compute_hybrid_crop_losses(out, y, args)
        for key, value in metrics.items():
            meters[key] = meters.get(key, 0.0) + float(value.item())
        count += 1
        if count >= max_batches:
            break
    if count == 0:
        return {}
    return {f"val_{k}": reduce_float(v / count, device) for k, v in meters.items()}


def main(args):
    if args.img_size != 224:
        raise ValueError("This crop-hybrid script is fixed to --img_size 224.")
    if args.num_keypoints != 7:
        raise ValueError("SurgPose keypoint version expects --num_keypoints 7.")

    device, local_rank, world_size = setup_dist()
    use_ddp = world_size > 1
    torch.backends.cudnn.benchmark = True

    log_dir = Path(args.save_dir) / args.name
    args.log_dir = str(log_dir)
    if args.dataset_cache_dir is None:
        args.dataset_cache_dir = str(log_dir / "dataset_cache")
    if is_main_process():
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    if use_ddp:
        if is_main_process():
            train_datasets = make_train_datasets(args)
            val_datasets = make_val_datasets(args)
            dist.barrier()
        else:
            dist.barrier()
            train_datasets = make_train_datasets(args)
            val_datasets = make_val_datasets(args)
    else:
        train_datasets = make_train_datasets(args)
        val_datasets = make_val_datasets(args)

    train_dataset = ConcatDataset(train_datasets)
    val_dataset = ConcatDataset(val_datasets)
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if use_ddp else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if use_ddp else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn_hybrid_crop,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn_hybrid_crop,
        persistent_workers=args.num_workers > 0,
    )

    model = CropHCCEDenseKeypointDPT(
        img_size=args.img_size,
        backbone=args.backbone,
        pretrained_backbone=bool(args.pretrained_backbone),
        hcce_feat_dim=args.hcce_feat_dim,
        hcce_bits=args.hcce_bits,
        num_keypoints=args.num_keypoints,
        pose_head_iter=args.pose_head_iter,
        pose_head_dropout=args.pose_head_dropout,
    ).to(device)

    resume_ckpt = None
    resume_epoch = 0
    resume_iter = 0
    if args.init_from is not None:
        _, report = load_compatible_state(model, args.init_from, device)
        if is_main_process():
            print(f"INIT_FROM: {args.init_from}", flush=True)
            print(f"loaded compatible tensors: {report['loaded']}, skipped: {len(report['skipped'])}", flush=True)
            if report["skipped"]:
                print("skipped keys head:", report["skipped"][:12], flush=True)
    if args.resume is not None:
        resume_ckpt, report = load_compatible_state(model, args.resume, device)
        resume_epoch = int(resume_ckpt.get("epoch", 0))
        resume_iter = int(resume_ckpt.get("iter", 0))
        if bool(args.reset_iter_on_resume):
            resume_epoch = 0
            resume_iter = 0
        if is_main_process():
            print(f"RESUME: {args.resume}", flush=True)
            print(f"loaded compatible tensors: {report['loaded']}, skipped: {len(report['skipped'])}", flush=True)

    if use_ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    raw = model.module if isinstance(model, DDP) else model
    optimizer = torch.optim.AdamW(
        [
            {"params": raw.encoder.parameters(), "lr": args.lr_backbone},
            {"params": raw.dense_head.parameters(), "lr": args.lr_dense},
            {"params": raw.keypoint_net.parameters(), "lr": args.lr_keypoint},
            {
                "params": list(raw.action_head.parameters()) + list(raw.wrist_pose_head.parameters()),
                "lr": args.lr_pose,
            },
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[args.lr_backbone, args.lr_dense, args.lr_keypoint, args.lr_pose],
        total_steps=args.max_iter,
        pct_start=0.0,
        final_div_factor=args.final_div_factor,
        cycle_momentum=False,
    )
    scaler = None
    if resume_ckpt is not None and bool(args.resume_optimizer):
        if "optimizer_state_dict" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in resume_ckpt and resume_ckpt["scheduler_state_dict"] is not None and bool(args.resume_scheduler):
            scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])
        if scaler is not None and "scaler_state_dict" in resume_ckpt and resume_ckpt["scaler_state_dict"] is not None:
            scaler.load_state_dict(resume_ckpt["scaler_state_dict"])

    if is_main_process():
        print(f"LOG_DIR: {log_dir}", flush=True)
        print(f"WORLD_SIZE: {world_size}", flush=True)
        print("MODEL: CropHCCEDenseKeypointDPT, shared DINOv2 encoder, 224 dense outputs", flush=True)
        print("KEYPOINT SUPERVISION: SurgPose 7-point heatmaps only", flush=True)
        print("HYBRID LOSSES: segmentation all, HCCE/pose RARP only, keypoint SurgPose only", flush=True)
        print("PART LABELS: dataset 1=gripper, 2=wrist, 3=shaft; logits [wrist, gripper, shaft]", flush=True)
        for ds in train_datasets:
            print(ds, flush=True)
        for ds in val_datasets:
            print(f"VAL {ds}", flush=True)

    iteration = resume_iter
    epoch = resume_epoch
    last_log = time.time()
    while iteration < args.max_iter:
        set_epoch_recursive(train_dataset, epoch)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        meters = {}
        for x, y in train_loader:
            iteration += 1
            x = x.to(device, non_blocking=True)
            y = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in y.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=bool(args.amp), dtype=torch.bfloat16):
                out = model(x, y["K"])
                loss, metrics = compute_hybrid_crop_losses(out, y, args)
            if not torch.isfinite(loss):
                if is_main_process():
                    bad = {k: float(v.detach().float().cpu().item()) for k, v in metrics.items() if torch.is_tensor(v) and v.numel() == 1}
                    print(f"NONFINITE LOSS at iter {iteration}: {bad}", flush=True)
                raise RuntimeError(f"non-finite training loss at iter {iteration}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            for key, value in metrics.items():
                meters.setdefault(key, AverageMeter()).update(float(value.item()))

            if iteration % args.log_freq == 0 and is_main_process():
                elapsed = max(time.time() - last_log, 1e-6)
                last_log = time.time()
                msg = [
                    f"iter {iteration:07d}",
                    f"epoch {epoch}",
                    f"loss {meters['total'].avg:.4f}",
                    f"dice {meters['dice'].avg:.4f}",
                    f"part {meters['part_ce'].avg:.4f}",
                    f"part_acc {meters['part_acc'].avg:.3f}",
                    f"hcce {meters['hcce'].avg:.4f}",
                    f"bit {meters['hcce_bit_acc'].avg:.3f}",
                    f"hm {meters['heatmap'].avg:.4f}",
                    f"kp {meters['kp_err_px'].avg:.2f}px",
                    f"poseN {meters['n_pose'].avg:.1f}",
                    f"kpN {meters['n_kp'].avg:.1f}",
                    f"cseN {meters['n_cse'].avg:.1f}",
                    f"{args.log_freq * args.batch_size * max(world_size, 1) / elapsed:.1f} img/s",
                ]
                print(" | ".join(msg), flush=True)

            if iteration % args.val_freq == 0:
                val = evaluate(model, val_loader, device, args, max_batches=args.val_batches)
                if is_main_process():
                    print("VAL " + " | ".join(f"{k} {v:.4f}" for k, v in val.items()), flush=True)
                    save_checkpoint(
                        log_dir / "checkpoints" / f"iter{iteration:07d}.pt",
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        args,
                        epoch,
                        iteration,
                    )
                model.train()

            if iteration % args.ckpt_freq == 0 and is_main_process():
                save_checkpoint(log_dir / "checkpoints" / "last.pt", model, optimizer, scheduler, scaler, args, epoch, iteration)

            if iteration >= args.max_iter:
                break
        epoch += 1

    if is_main_process():
        save_checkpoint(log_dir / "checkpoints" / "last.pt", model, optimizer, scheduler, scaler, args, epoch, iteration)
        print(f"Finished training at iter {iteration}", flush=True)
    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="logs")
    parser.add_argument("--name", type=str, default="instrument_hcce_crop_surgpose_keypoint_hybrid")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--init_from", type=str, default=None)
    parser.add_argument("--resume_optimizer", type=int, default=1, choices=[0, 1])
    parser.add_argument("--resume_scheduler", type=int, default=1, choices=[0, 1])
    parser.add_argument("--reset_iter_on_resume", type=int, default=0, choices=[0, 1])

    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--needle_puncture_data_dir", type=str, default=PUNCTURE_DATASET_ROOT)
    parser.add_argument("--needle_puncture_pose_dir", type=str, default=PUNCTURE_POSE_ROOT)
    parser.add_argument("--needle_grasping_data_dir", type=str, default=GRASPING_DATASET_ROOT)
    parser.add_argument("--needle_grasping_pose_dir", type=str, default=GRASPING_POSE_ROOT)
    parser.add_argument("--knotting_data_dir", type=str, default=KNOTTING_DATASET_ROOT)
    parser.add_argument("--knotting_pose_dir", type=str, default=KNOTTING_POSE_ROOT)
    parser.add_argument("--train_rarp_datasets", type=str, default="needlePuncture,needleGrasping,knotting")
    parser.add_argument("--surgpose_root", type=str, default=SURGPOSE_ROOT)
    parser.add_argument("--surgpose_train_episodes", type=str, default="")
    parser.add_argument("--surgpose_val_episodes", type=str, default="")
    parser.add_argument("--use_surgpose", type=int, default=1, choices=[0, 1])
    parser.add_argument("--use_vos_endovis", type=int, default=1, choices=[0, 1])
    parser.add_argument("--vos_endovis17_root", type=str, default=VOS_ENDOVIS17_ROOT)
    parser.add_argument("--vos_endovis18_root", type=str, default=VOS_ENDOVIS18_ROOT)
    parser.add_argument("--vos_endovis17_image_root", type=str, default=VOS_ENDOVIS17_IMAGE_ROOT)
    parser.add_argument("--vos_endovis18_image_root", type=str, default=VOS_ENDOVIS18_IMAGE_ROOT)
    parser.add_argument("--num_keypoints", type=int, default=7)

    parser.add_argument("--min_dice_shaft", type=float, default=0.8)
    parser.add_argument("--min_dice_wrist", type=float, default=0.6)
    parser.add_argument("--min_dice_gripper", type=float, default=0.6)
    parser.add_argument("--needle_train_ratio", type=float, default=0.95)
    parser.add_argument("--canonicalize_pose_symmetry", type=int, default=1, choices=[0, 1])
    parser.add_argument("--canonical_eps", type=float, default=0.08)
    parser.add_argument("--cse_coord_root", type=str, default=None)
    parser.add_argument("--render_on_the_fly", type=int, default=1, choices=[0, 1])
    parser.add_argument("--coord_render_backend", type=str, default="trimesh", choices=["trimesh", "gaussian"])
    parser.add_argument("--dataset_cache_dir", type=str, default=None)

    parser.add_argument("--batch_size", type=int, default=56)
    parser.add_argument("--val_batch_size", type=int, default=56)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--train_subsample", type=int, default=1)
    parser.add_argument("--val_subsample", type=int, default=20)
    parser.add_argument("--max_iter", type=int, default=60000)
    parser.add_argument("--log_freq", type=int, default=10)
    parser.add_argument("--val_freq", type=int, default=1000)
    parser.add_argument("--val_batches", type=int, default=20)
    parser.add_argument("--ckpt_freq", type=int, default=1000)
    parser.add_argument("--amp", type=int, default=1, choices=[0, 1])
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--backbone", type=str, default="dinov2_vits14", choices=["dinov2_vits14", "dinov2_vitb14"])
    parser.add_argument("--pretrained_backbone", type=int, default=1, choices=[0, 1])
    parser.add_argument("--hcce_feat_dim", type=int, default=256)
    parser.add_argument("--hcce_bits", type=int, default=8)
    parser.add_argument("--pose_head_iter", type=int, default=4)
    parser.add_argument("--pose_head_dropout", type=float, default=0.3)

    parser.add_argument("--lr_backbone", type=float, default=5e-5)
    parser.add_argument("--lr_dense", type=float, default=1e-4)
    parser.add_argument("--lr_keypoint", type=float, default=1e-4)
    parser.add_argument("--lr_pose", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-7)
    parser.add_argument("--final_div_factor", type=float, default=1e4)

    parser.add_argument("--alpha_dice", type=float, default=5.0)
    parser.add_argument("--alpha_bce_mask", type=float, default=2.0)
    parser.add_argument("--alpha_part", type=float, default=2.0)
    parser.add_argument("--alpha_hcce", type=float, default=1.0)
    parser.add_argument("--alpha_heatmap", type=float, default=1.0)
    parser.add_argument("--alpha_action_l1", type=float, default=1.0)
    parser.add_argument("--alpha_wrist_quat_l1", type=float, default=1.0)
    parser.add_argument("--alpha_wrist_trans_l1", type=float, default=10.0)
    parser.add_argument("--hcce_coord_min", type=float, default=-1.0)
    parser.add_argument("--hcce_coord_max", type=float, default=1.0)
    parser.add_argument("--heatmap_sigma", type=float, default=2.0)

    parser.add_argument("--color_jitter", type=int, default=1, choices=[0, 1])
    parser.add_argument("--rgb_augmentation", type=int, default=1, choices=[0, 1])
    parser.add_argument("--occlusion_augmentation", type=int, default=1, choices=[0, 1])
    parser.add_argument("--occlusion_prob", type=float, default=0.5)
    main(parser.parse_args())
