import json
import os
import random

import cv2
import numpy as np
import yaml
from PIL import Image, ImageFile
from torch.utils.data import Dataset

from datasets.rarp_augmentation import transform_rarp_sample
from utils import normalize_rgb

ImageFile.LOAD_TRUNCATED_IMAGES = True


SURGPOSE_ROOT = "/mnt/nas/share/shuojue/data/surgpose"


def _episode_ids(start, end):
    return [f"{i:06d}" for i in range(start, end + 1)]


def _parse_episode_list(value, default):
    if value is None or str(value).strip() == "":
        return list(default)
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _resize_nn(arr, size):
    return cv2.resize(arr, (size, size), interpolation=cv2.INTER_NEAREST)


def _normalize_depth_in_mask(depth, valid, low_pct=2.0, high_pct=98.0, eps=1e-6):
    vals = depth[valid]
    if vals.size == 0:
        return np.zeros_like(depth, dtype=np.float32)
    lo, hi = np.percentile(vals.astype(np.float32), [float(low_pct), float(high_pct)])
    if not np.isfinite(lo) or not np.isfinite(hi) or float(hi - lo) <= eps:
        return np.zeros_like(depth, dtype=np.float32)
    out = (depth.astype(np.float32) - float(lo)) / float(hi - lo)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _warp_depth_for_instance(
    depth,
    inst_mask,
    affine,
    img_size,
    output_size,
    normalize_relative=True,
    norm_low_pct=2.0,
    norm_high_pct=98.0,
):
    finite = np.isfinite(depth)
    valid = finite & (inst_mask > 0)
    if normalize_relative:
        depth_src = _normalize_depth_in_mask(
            depth,
            valid,
            low_pct=norm_low_pct,
            high_pct=norm_high_pct,
        )
    else:
        depth_src = depth.astype(np.float32)
    depth_clean = np.where(valid, depth_src, 0.0).astype(np.float32)
    warped_depth = cv2.warpAffine(
        depth_clean,
        affine,
        (img_size, img_size),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(np.float32)
    warped_valid = cv2.warpAffine(
        valid.astype(np.uint8),
        affine,
        (img_size, img_size),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(np.uint8)
    depth_out = _resize_nn(warped_depth, output_size).astype(np.float32)
    valid_out = _resize_nn(warped_valid, output_size).astype(np.float32)
    depth_out[valid_out <= 0] = 0.0
    return depth_out, valid_out


def _frame_index_from_stem(stem):
    return int(str(stem).split("_")[-1])


def _lookup_key(data, key):
    if key in data:
        return data[key]
    skey = str(key)
    if skey in data:
        return data[skey]
    return None


class SurgPoseInstruments(Dataset):
    """
    SurgPose processed stereo data as instrument detection/segmentation/depth data.

    Expected layout:
        {episode}/processed_stereo_640/left_frames/frame_*.png
        {episode}/processed_stereo_640/sam3_segmentation/frame_*.png
        {episode}/processed_stereo_640/sam3_segmentaion_part/frame_*.png
        {episode}/processed_stereo_640/depth_npy/frame_*.npy

    Part labels are remapped to the project convention:
        SurgPose: 1=shaft, 2=wrist, 3=gripper
        Project: 1=gripper, 2=wrist, 3=shaft
    """

    def __init__(
        self,
        root_dir=SURGPOSE_ROOT,
        split="train",
        training=False,
        img_size=630,
        subsample=1,
        n=-1,
        train_episodes=None,
        val_episodes=None,
        aug_random_crop_rotate=False,
        aug_geom_prob=1.0,
        aug_crop_scale=1.2,
        aug_max_angle=np.pi,
        aug_offset_scale=1.0,
        aug_color_jitter=False,
        dense_output_stride=2,
        load_depth=True,
        num_keypoints=7,
        keypoint_file="keypoints_left_rectified.yaml",
        depth_normalize_relative=True,
        depth_norm_low_pct=2.0,
        depth_norm_high_pct=98.0,
    ):
        super().__init__()
        self.name = "surgpose"
        self.split = split
        self.training = bool(training)
        self.root_dir = root_dir
        self.img_size = int(img_size)
        self.subsample = int(subsample)
        self.aug_random_crop_rotate = bool(aug_random_crop_rotate)
        self.aug_geom_prob = float(aug_geom_prob)
        self.aug_crop_scale = float(aug_crop_scale)
        self.aug_max_angle = float(aug_max_angle)
        self.aug_offset_scale = float(aug_offset_scale)
        self.aug_color_jitter = bool(aug_color_jitter)
        self.dense_output_stride = int(dense_output_stride)
        self.load_depth = bool(load_depth)
        self.num_keypoints = int(num_keypoints)
        self.keypoint_file = str(keypoint_file)
        self.depth_normalize_relative = bool(depth_normalize_relative)
        self.depth_norm_low_pct = float(depth_norm_low_pct)
        self.depth_norm_high_pct = float(depth_norm_high_pct)
        self._keypoint_cache = {}

        default_train = _episode_ids(0, 27)
        default_val = _episode_ids(28, 33)
        episodes = (
            _parse_episode_list(train_episodes, default_train)
            if split == "train"
            else _parse_episode_list(val_episodes, default_val)
        )

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
                stem = os.path.splitext(fn)[0]
                if not os.path.isfile(os.path.join(inst_dir, fn)):
                    continue
                if not os.path.isfile(os.path.join(part_dir, fn)):
                    continue
                self.samples.append((ep, stem))

        if self.subsample > 1:
            self.samples = self.samples[:: self.subsample]
        if n >= 0:
            self.samples = self.samples[:n]

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        return f"{self.name}: split={self.split} - N={len(self.samples)}"

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
        frame_key = _frame_index_from_stem(stem)
        frame_data = _lookup_key(self._load_episode_keypoints(proc), frame_key)
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
                if not (np.isfinite(x) and np.isfinite(y)):
                    continue
                pts[local_idx] = (x, y)
                valid[local_idx] = True
            if not valid.any():
                continue

            scores = {}
            h, w = inst_mask.shape[:2]
            for inst_id in inst_ids:
                dvals = []
                dmap = dist_maps[inst_id]
                for x, y in pts[valid]:
                    xi = int(round(float(x)))
                    yi = int(round(float(y)))
                    if 0 <= xi < w and 0 <= yi < h:
                        dvals.append(float(dmap[yi, xi]))
                    else:
                        dvals.append(1e6)
                scores[inst_id] = float(np.mean(dvals)) if dvals else 1e6
            assigned_inst = min(scores, key=scores.get)
            grouped[int(assigned_inst)] = {
                "keypoints": pts,
                "keypoints_valid": valid,
            }
        return grouped

    def __getitem__(self, idx):
        if self.training:
            idx = random.randrange(len(self.samples))
        ep, stem = self.samples[idx]
        proc = os.path.join(self.root_dir, ep, "processed_stereo_640")
        frame = f"{stem}.png"

        rgb = np.asarray(Image.open(os.path.join(proc, "left_frames", frame)).convert("RGB"))
        inst_mask = np.asarray(Image.open(os.path.join(proc, "sam3_segmentation", frame)))
        part_raw = np.asarray(Image.open(os.path.join(proc, "sam3_segmentaion_part", frame)))
        depth = None
        if self.load_depth:
            depth_path = os.path.join(proc, "depth_npy", f"{stem}.npy")
            if os.path.isfile(depth_path):
                depth = np.load(depth_path).astype(np.float32)

        keypoints_by_inst = self._keypoint_groups_for_frame(proc, stem, inst_mask)
        instruments = []
        for inst_id in sorted(np.unique(inst_mask).tolist()):
            if inst_id == 0:
                continue
            inst_binary = inst_mask == inst_id
            part_mask = np.zeros_like(part_raw, dtype=np.int64)
            part_mask[inst_binary & (part_raw == 3)] = 1  # gripper
            part_mask[inst_binary & (part_raw == 2)] = 2  # wrist
            part_mask[inst_binary & (part_raw == 1)] = 3  # shaft
            wrist_region = part_mask == 2
            if not np.any(wrist_region):
                continue
            ys, xs = np.where(wrist_region)
            wrist_center = np.array([xs.mean(), ys.mean()], dtype=np.float32)
            inst = {
                "instance_id": int(inst_id),
                "inst_mask": inst_binary.astype(np.float32),
                "part_mask": part_mask,
                "wrist_center": wrist_center,
            }
            kp = keypoints_by_inst.get(int(inst_id))
            if kp is not None:
                inst["keypoints"] = kp["keypoints"]
                inst["keypoints_valid"] = kp["keypoints_valid"]
            instruments.append(inst)

        dense_output_size = self.img_size // self.dense_output_stride
        rgb_sq, out_instruments, affine = transform_rarp_sample(
            rgb,
            instruments,
            img_size=self.img_size,
            output_size=self.img_size // 4,
            dense_output_size=dense_output_size,
            training=self.training,
            aug_random_crop_rotate=self.aug_random_crop_rotate,
            aug_geom_prob=self.aug_geom_prob,
            aug_crop_scale=self.aug_crop_scale,
            aug_max_angle=self.aug_max_angle,
            aug_offset_scale=self.aug_offset_scale,
            aug_color_jitter=self.aug_color_jitter,
        )

        if depth is not None:
            full_inst_lookup = {int(inst["instance_id"]): inst["inst_mask"] for inst in instruments}
            for inst in out_instruments:
                full_mask = full_inst_lookup.get(int(inst["instance_id"]))
                if full_mask is None:
                    continue
                depth_map, depth_valid = _warp_depth_for_instance(
                    depth,
                    full_mask,
                    affine,
                    self.img_size,
                    dense_output_size,
                    normalize_relative=self.depth_normalize_relative,
                    norm_low_pct=self.depth_norm_low_pct,
                    norm_high_pct=self.depth_norm_high_pct,
                )
                inst["depth_map_dense"] = depth_map
                inst["depth_valid_dense"] = depth_valid
                inst["has_depth"] = bool(np.any(depth_valid > 0))

        img_array = normalize_rgb(rgb_sq, imagenet_normalization=1)
        return img_array, {
            "imagename": f"{ep}/{stem}",
            "instruments": out_instruments,
            "dataset_source": 0,
        }
