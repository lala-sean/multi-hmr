import json
import os
import random

import numpy as np
from PIL import Image, ImageFile
from torch.utils.data import Dataset

from datasets.rarp_augmentation import transform_rarp_sample
from utils import normalize_rgb

ImageFile.LOAD_TRUNCATED_IMAGES = True


VOS_ENDOVIS17_ROOT = "/mnt/nas/share/shuojue/VOS-Endovis17"
VOS_ENDOVIS18_ROOT = "/mnt/nas/share/shuojue/VOS-Endovis18"
VOS_ENDOVIS17_IMAGE_ROOT = "/mnt/nas/haofeng/data/VOS-Endovis17"
VOS_ENDOVIS18_IMAGE_ROOT = "/mnt/nas/haofeng/data/VOS-Endovis18"

_PART_REMAP = {1: 3, 2: 1, 3: 2, 4: 1}


class VOSEndoVisInstruments(Dataset):
    """
    VOS-EndoVis labels in the same instrument sample format as RARP50.

    RGB is read from the original VOS-Endovis roots, while masks/parts are read
    from the *_new processed label roots. Only ids whose Meta_new category is
    "surgical instrument" are used.
    """

    def __init__(
        self,
        label_root,
        image_root,
        split="train",
        training=False,
        img_size=630,
        subsample=1,
        n=-1,
        aug_random_crop_rotate=False,
        aug_geom_prob=1.0,
        aug_crop_scale=1.2,
        aug_max_angle=np.pi,
        aug_offset_scale=1.0,
        aug_color_jitter=False,
        dense_output_stride=2,
    ):
        super().__init__()
        self.name = "vos_endovis17" if "17" in os.path.basename(label_root.rstrip("/")) else "vos_endovis18"
        self.split = split
        self.training = bool(training)
        self.label_root = label_root
        self.image_root = image_root
        self.img_size = int(img_size)
        self.subsample = int(subsample)
        self.aug_random_crop_rotate = bool(aug_random_crop_rotate)
        self.aug_geom_prob = float(aug_geom_prob)
        self.aug_crop_scale = float(aug_crop_scale)
        self.aug_max_angle = float(aug_max_angle)
        self.aug_offset_scale = float(aug_offset_scale)
        self.aug_color_jitter = bool(aug_color_jitter)
        self.dense_output_stride = int(dense_output_stride)

        self.samples = []
        self.seq_meta = {}
        masks_root = os.path.join(label_root, split, "masks_new")
        for seq in sorted(os.listdir(masks_root)) if os.path.isdir(masks_root) else []:
            meta_path = os.path.join(label_root, split, "Meta_new", f"{seq}.json")
            meta = json.load(open(meta_path, "r")) if os.path.isfile(meta_path) else {"info": {"category": {}}}
            self.seq_meta[seq] = meta
            seq_dir = os.path.join(masks_root, seq)
            for fn in sorted(os.listdir(seq_dir)):
                if fn.endswith(".png"):
                    self.samples.append((seq, os.path.splitext(fn)[0]))

        if self.subsample > 1:
            self.samples = self.samples[:: self.subsample]
        if n >= 0:
            self.samples = self.samples[:n]

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        return f"{self.name}: split={self.split} - N={len(self.samples)}"

    def __getitem__(self, idx):
        if self.training:
            idx = random.randrange(len(self.samples))
        seq, stem = self.samples[idx]
        fn = f"{stem}.png"

        img_path = os.path.join(self.image_root, self.split, "images", seq, fn)
        if not os.path.isfile(img_path) and self.split == "valid":
            img_path = os.path.join(self.image_root, "test", "images", seq, fn)
        rgb = np.asarray(Image.open(img_path).convert("RGB"))
        mask_np = np.asarray(Image.open(os.path.join(self.label_root, self.split, "masks_new", seq, fn)))
        part_raw = np.asarray(Image.open(os.path.join(self.label_root, self.split, "parts_new", seq, fn)))

        categories = self.seq_meta.get(seq, {}).get("info", {}).get("category", {})
        instruments = []
        for inst_id in sorted(np.unique(mask_np).tolist()):
            if inst_id == 0:
                continue
            if categories and categories.get(str(int(inst_id))) != "surgical instrument":
                continue
            inst_binary = mask_np == inst_id
            part_mask = np.zeros_like(part_raw, dtype=np.int64)
            for raw_value, remapped in _PART_REMAP.items():
                part_mask[inst_binary & (part_raw == raw_value)] = remapped
            wrist_region = part_mask == 2
            if not np.any(wrist_region):
                continue
            ys, xs = np.where(wrist_region)
            wrist_center = np.array([xs.mean(), ys.mean()], dtype=np.float32)
            instruments.append(
                {
                    "instance_id": int(inst_id),
                    "inst_mask": inst_binary.astype(np.float32),
                    "part_mask": part_mask,
                    "wrist_center": wrist_center,
                    "keypoint": wrist_center.copy(),
                }
            )

        rgb_sq, instruments, _ = transform_rarp_sample(
            rgb,
            instruments,
            img_size=self.img_size,
            output_size=self.img_size // 4,
            dense_output_size=self.img_size // self.dense_output_stride,
            training=self.training,
            aug_random_crop_rotate=self.aug_random_crop_rotate,
            aug_geom_prob=self.aug_geom_prob,
            aug_crop_scale=self.aug_crop_scale,
            aug_max_angle=self.aug_max_angle,
            aug_offset_scale=self.aug_offset_scale,
            aug_color_jitter=self.aug_color_jitter,
        )
        for inst in instruments:
            inst["keypoint"] = inst["wrist_center"].copy()
        img_array = normalize_rgb(rgb_sq, imagenet_normalization=1)
        return img_array, {
            "imagename": f"{self.name}/{seq}/{stem}",
            "instruments": instruments,
            "dataset_source": 0,
        }
