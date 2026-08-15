import json
import os
import random
import time
import hashlib
from argparse import ArgumentParser
from datetime import timedelta
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image, ImageFile
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

import train_instrument_hcce_crop_surgpose_keypoint_dpt as base
from datasets.surgical_instruments import RARP50_DIR

ImageFile.LOAD_TRUNCATED_IMAGES = True


PUNCTURE_DATASET_ROOT = base.PUNCTURE_DATASET_ROOT
PUNCTURE_POSE_ROOT = base.PUNCTURE_POSE_ROOT
GRASPING_DATASET_ROOT = base.GRASPING_DATASET_ROOT
GRASPING_POSE_ROOT = base.GRASPING_POSE_ROOT
KNOTTING_DATASET_ROOT = base.KNOTTING_DATASET_ROOT
KNOTTING_POSE_ROOT = base.KNOTTING_POSE_ROOT

CropHCCEDenseKeypointDPT = base.CropHCCEDenseKeypointDPT
RARPCropHCCEDataset = base.RARPCropHCCEDataset
VOSEndoVisCropDataset = base.VOSEndoVisCropDataset


def setup_dist(timeout_sec=7200):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(seconds=int(timeout_sec)),
        )
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    return device, local_rank, world_size


def dist_barrier(local_rank):
    if dist.is_available() and dist.is_initialized():
        dist.barrier(device_ids=[int(local_rank)] if torch.cuda.is_available() else None)


class RARPJointKeypointCropDataset(Dataset):
    """RARP crop wrapper: pose/HCCE/part/instance plus 5 joint keypoint heatmaps."""

    def __init__(self, wrapped, dataset_name):
        self.wrapped = wrapped
        self.dataset_name = dataset_name

    def __len__(self):
        return len(self.wrapped)

    def __repr__(self):
        return f"rarp_joint_{self.dataset_name}: {self.wrapped}"

    def set_epoch(self, epoch):
        if hasattr(self.wrapped, "set_epoch"):
            self.wrapped.set_epoch(epoch)

    def __getitem__(self, idx):
        image, target = self.wrapped[idx]
        valid = target.get("keypoints_valid")
        has_keypoints = bool(valid is not None and torch.as_tensor(valid).bool().any().item())
        target["dataset_name"] = self.dataset_name
        target["sample_group"] = "rarp"
        target["has_pose"] = torch.tensor(True, dtype=torch.bool)
        target["has_keypoints"] = torch.tensor(has_keypoints, dtype=torch.bool)
        target["keypoint_kind"] = "rarp_joint"
        return image, target


class SegOnlyKeypointShapeAdapter(Dataset):
    """Force segmentation-only datasets to expose the same keypoint tensor shape."""

    def __init__(self, wrapped, num_keypoints=5, crop_size=224, sample_group=None):
        self.wrapped = wrapped
        self.num_keypoints = int(num_keypoints)
        self.crop_size = int(crop_size)
        self.sample_group = sample_group

    def __len__(self):
        return len(self.wrapped)

    def __repr__(self):
        return repr(self.wrapped)

    def set_epoch(self, epoch):
        if hasattr(self.wrapped, "set_epoch"):
            self.wrapped.set_epoch(epoch)

    def __getitem__(self, idx):
        image, target = self.wrapped[idx]
        target["has_keypoints"] = torch.tensor(False, dtype=torch.bool)
        target["keypoint_kind"] = "none"
        target["heatmaps"] = torch.zeros((self.num_keypoints, self.crop_size, self.crop_size), dtype=torch.float32)
        target["keypoints_crop"] = torch.zeros((self.num_keypoints, 2), dtype=torch.float32)
        target["keypoints_valid"] = torch.zeros((self.num_keypoints,), dtype=torch.bool)
        if self.sample_group is not None:
            target["sample_group"] = self.sample_group
        return image, target


class SurgicalInstrumentCropDataset(Dataset):
    """
    Crop version of the SurgicalInstrument/RARP50 segmentation dataset.

    It reads the same raw RARP50 RGB/instance/part files as datasets.surgical_instruments,
    but returns one 224x224 single-instrument crop.  Only instance and part
    segmentation are supervised for this dataset.
    """

    def __init__(
        self,
        root_dir=RARP50_DIR,
        split="train",
        training=False,
        crop_size=224,
        subsample=1,
        num_keypoints=5,
        color_jitter=True,
        rgb_augmentation=True,
        occlusion_augmentation=True,
        occlusion_prob=0.5,
        cache_dir=None,
        enumerate_instances=False,
    ):
        super().__init__()
        self.name = "surgical_instrument"
        self.root_dir = root_dir
        self.split = split
        self.training = bool(training)
        self.crop_size = int(crop_size)
        self.subsample = int(subsample)
        self.num_keypoints = int(num_keypoints)
        self.color_jitter = bool(color_jitter)
        self.rgb_augmentation = bool(rgb_augmentation)
        self.occlusion_augmentation = bool(occlusion_augmentation)
        self.occlusion_prob = float(occlusion_prob)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.enumerate_instances = bool(enumerate_instances)
        self.epoch = 0

        self.image_dir = os.path.join(root_dir, split, "images")
        self.mask_dir = os.path.join(root_dir, split, "masks")
        self.parts_dir = os.path.join(root_dir, split, "parts")
        self.meta_dir = os.path.join(root_dir, split, "Meta")
        cache_path = self._cache_path()
        if cache_path is not None and cache_path.is_file():
            self.samples = torch.load(cache_path, map_location="cpu", weights_only=False)["samples"]
        else:
            self.samples = self._build_samples()
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"samples": self.samples}, cache_path)
        if not self.samples:
            raise RuntimeError(f"SurgicalInstrumentCropDataset is empty: root={root_dir} split={split}")

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        return (
            f"surgical_instrument_crop: split={self.split} training={self.training} "
            f"N={len(self)} enumerate_instances={int(self.enumerate_instances)}"
        )

    def _cache_path(self):
        if self.cache_dir is None:
            return None
        identity = "|".join(
            [
                str(Path(self.root_dir).resolve()),
                str(self.split),
                str(int(self.subsample)),
                f"surgical_crop_v3_pilpart_enumerate{int(self.enumerate_instances)}",
            ]
        )
        digest = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:16]
        return self.cache_dir / f"surgical_instrument_crop_{self.split}_{digest}.pt"

    @staticmethod
    def _valid_instances(mask_np, parts_np):
        decoded_parts = np.where(parts_np > 0, parts_np // 20, 0)
        inst_ids = []
        for inst_id in sorted(np.unique(mask_np).tolist()):
            inst_id = int(inst_id)
            if inst_id == 0:
                continue
            inst_binary = mask_np == inst_id
            if (inst_binary & (decoded_parts == 1)).any():
                inst_ids.append(inst_id)
        return inst_ids

    def _build_samples(self):
        samples = []
        if not os.path.isdir(self.image_dir):
            return samples
        frame_counter = 0
        for video_id in sorted(os.listdir(self.image_dir)):
            frames_dir = os.path.join(self.image_dir, video_id)
            if not os.path.isdir(frames_dir):
                continue
            for frame_fn in sorted(os.listdir(frames_dir)):
                if not frame_fn.endswith(".jpg"):
                    continue
                if self.subsample > 1 and frame_counter % self.subsample != 0:
                    frame_counter += 1
                    continue
                frame_counter += 1
                frame_id = frame_fn[:-4]
                mask_path = os.path.join(self.mask_dir, video_id, f"{frame_id}.png")
                parts_path = os.path.join(self.parts_dir, video_id, f"{frame_id}.png")
                if not (os.path.isfile(mask_path) and os.path.isfile(parts_path)):
                    continue
                if self.enumerate_instances:
                    mask_np = np.asarray(Image.open(mask_path))
                    parts_np = np.asarray(Image.open(parts_path))
                    inst_ids = self._valid_instances(mask_np, parts_np)
                    if not inst_ids:
                        continue
                    for inst_id in inst_ids:
                        samples.append((video_id, frame_id, int(inst_id)))
                else:
                    parts_np = np.asarray(Image.open(parts_path))
                    decoded_parts = np.where(parts_np > 0, parts_np // 20, 0)
                    if np.any(decoded_parts == 1):
                        samples.append((video_id, frame_id, 0))
        return samples

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _load_raw(self, video_id, frame_id):
        img_path = os.path.join(self.image_dir, video_id, f"{frame_id}.jpg")
        rgb = np.asarray(Image.open(img_path).convert("RGB"))
        mask_np = np.asarray(Image.open(os.path.join(self.mask_dir, video_id, f"{frame_id}.png")))
        parts_np = np.asarray(Image.open(os.path.join(self.parts_dir, video_id, f"{frame_id}.png")))
        return rgb, mask_np, parts_np, img_path

    def __getitem__(self, idx):
        video_id, frame_id, inst_id = self.samples[idx]
        rgb, mask_np, parts_np, img_path = self._load_raw(video_id, frame_id)
        inst_ids = self._valid_instances(mask_np, parts_np)
        if int(inst_id) == 0:
            if not inst_ids:
                raise RuntimeError(f"no valid SurgicalInstrument instance: {video_id}/{frame_id}")
            inst_id = random.choice(inst_ids) if self.training else inst_ids[0]

        inst_mask = mask_np == int(inst_id)
        decoded_parts = np.where(parts_np > 0, parts_np // 20, 0)
        part_mask = np.zeros_like(mask_np, dtype=np.uint8)
        part_mask[inst_mask & (decoded_parts == 0)] = 1
        part_mask[inst_mask & (decoded_parts == 1)] = 2
        part_mask[inst_mask & (decoded_parts == 2)] = 3
        if not inst_mask.any():
            raise RuntimeError(f"empty SurgicalInstrument mask: {video_id}/{frame_id}/{inst_id}")

        h, w = rgb.shape[:2]
        bbox_min, bbox_max = base._bbox_from_mask(inst_mask, self.training, self.epoch, w, h)
        crop_rgb, scale_xy, pad_xy = base._crop_resize_pad(
            rgb,
            bbox_min,
            bbox_max,
            self.crop_size,
            interpolation=cv2.INTER_LINEAR,
            edge=True,
        )
        inst_crop, _, _ = base._crop_resize_pad(
            inst_mask.astype(np.uint8),
            bbox_min,
            bbox_max,
            self.crop_size,
            interpolation=cv2.INTER_NEAREST,
            value=0,
        )
        part_crop, _, _ = base._crop_resize_pad(
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
                aug_rgb = base._rarp_aug_module._apply_color_jitter(aug_rgb)
            if self.occlusion_augmentation and random.random() < self.occlusion_prob:
                aug_rgb = base._rarp_aug_module._apply_occlusion(aug_rgb)
            if self.rgb_augmentation:
                aug_rgb = base._rarp_aug_module._apply_rgb_aug(aug_rgb)

        target = {
            "dataset_name": "surgical_instrument",
            "sample_group": "surgical",
            "has_pose": torch.tensor(False, dtype=torch.bool),
            "has_keypoints": torch.tensor(False, dtype=torch.bool),
            "keypoint_kind": "none",
            "inst_mask": torch.from_numpy(inst_crop.astype(np.float32)),
            "part_mask": torch.from_numpy(part_crop.astype(np.int64)),
            "coord_img": torch.zeros((self.crop_size, self.crop_size, 4), dtype=torch.float32),
            "has_cse": torch.tensor(False, dtype=torch.bool),
            "heatmaps": torch.zeros((self.num_keypoints, self.crop_size, self.crop_size), dtype=torch.float32),
            "keypoints_crop": torch.zeros((self.num_keypoints, 2), dtype=torch.float32),
            "keypoints_valid": torch.zeros((self.num_keypoints,), dtype=torch.bool),
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
            "video_name": video_id,
            "frame_id": frame_id,
            "img_path": img_path,
            "crop_rgb": torch.from_numpy(crop_rgb.astype(np.uint8)),
        }
        return base._imagenet_tensor(aug_rgb), target


class FixedRateHybridDataset(Dataset):
    """Epoch-level fixed-rate mixture over dataset groups."""

    def __init__(self, groups, rates, samples_per_epoch=0, seed=12345):
        super().__init__()
        if len(groups) != len(rates):
            raise ValueError("groups and rates must have the same length")
        self.groups = [(str(name), ds) for name, ds in groups]
        self.rates = np.asarray(rates, dtype=np.float64)
        if np.any(self.rates < 0) or float(self.rates.sum()) <= 0.0:
            raise ValueError(f"invalid sampling rates: {rates}")
        self.rates = self.rates / self.rates.sum()
        for rate, (name, ds) in zip(self.rates, self.groups):
            if rate > 0.0 and len(ds) <= 0:
                raise RuntimeError(f"group {name} has rate {rate} but is empty")
        self.samples_per_epoch = int(samples_per_epoch) if int(samples_per_epoch) > 0 else int(sum(len(ds) for _, ds in self.groups))
        if self.samples_per_epoch <= 0:
            raise RuntimeError("samples_per_epoch must be positive")
        self.seed = int(seed)
        self.epoch = -1
        self.plan = []
        self.set_epoch(0)

    def __len__(self):
        return self.samples_per_epoch

    def __repr__(self):
        pieces = [
            f"{name}:rate={rate:.3f},N={len(ds)}"
            for rate, (name, ds) in zip(self.rates, self.groups)
        ]
        return f"fixed_rate_hybrid_crop(samples_per_epoch={self.samples_per_epoch}; " + "; ".join(pieces) + ")"

    def _counts_for_epoch(self):
        raw = self.rates * float(self.samples_per_epoch)
        counts = np.floor(raw).astype(np.int64)
        residual = self.samples_per_epoch - int(counts.sum())
        if residual > 0:
            order = np.argsort(-(raw - counts))
            for idx in order[:residual]:
                counts[idx] += 1
        return counts

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        rng = np.random.default_rng(self.seed + self.epoch)
        plan = []
        counts = self._counts_for_epoch()
        for group_idx, count in enumerate(counts.tolist()):
            if count <= 0:
                continue
            name, ds = self.groups[group_idx]
            idxs = rng.integers(0, len(ds), size=count, endpoint=False)
            plan.extend((group_idx, int(sample_idx)) for sample_idx in idxs.tolist())
            base.set_epoch_recursive(ds, self.epoch)
        rng.shuffle(plan)
        self.plan = plan

    def __getitem__(self, idx):
        if len(self.plan) != self.samples_per_epoch:
            self.set_epoch(self.epoch if self.epoch >= 0 else 0)
        group_idx, sample_idx = self.plan[int(idx) % self.samples_per_epoch]
        group_name, ds = self.groups[group_idx]
        image, target = ds[sample_idx]
        target["sample_group"] = group_name
        return image, target


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
                vals.append(t[key])
                continue
            if key == "has_pose" or key == "has_keypoints" or key == "has_cse":
                vals.append(torch.tensor(False, dtype=torch.bool))
            elif key == "heatmaps":
                vals.append(torch.zeros((5, 224, 224), dtype=torch.float32))
            elif key == "keypoints_crop":
                vals.append(torch.zeros((5, 2), dtype=torch.float32))
            elif key == "keypoints_valid":
                vals.append(torch.zeros((5,), dtype=torch.bool))
            elif key == "coord_img":
                vals.append(torch.zeros((224, 224, 4), dtype=torch.float32))
            else:
                raise KeyError(f"missing target key: {key}")
        y[key] = torch.stack(vals, dim=0)
    y["dataset_name"] = [t.get("dataset_name", "") for t in targets]
    y["sample_group"] = [t.get("sample_group", "") for t in targets]
    y["video_name"] = [t.get("video_name", "") for t in targets]
    y["frame_id"] = [t.get("frame_id", "") for t in targets]
    y["img_path"] = [t.get("img_path", "") for t in targets]
    return x, y


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
    return RARPJointKeypointCropDataset(ds, name)


def _concat_or_single(datasets):
    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)


def make_vos_datasets(args, split, training, subsample):
    datasets = []
    if not args.use_vos_endovis:
        return datasets
    if os.path.isdir(args.vos_endovis17_root):
        datasets.append(
            SegOnlyKeypointShapeAdapter(
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
                ),
                num_keypoints=args.num_keypoints,
                crop_size=args.img_size,
                sample_group="vos",
            )
        )
    if os.path.isdir(args.vos_endovis18_root):
        split18 = split if split in ("train", "test") else "test"
        datasets.append(
            SegOnlyKeypointShapeAdapter(
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
                ),
                num_keypoints=args.num_keypoints,
                crop_size=args.img_size,
                sample_group="vos",
            )
        )
    return datasets


def make_train_dataset(args):
    specs = {
        "needlePuncture": (args.needle_puncture_data_dir, args.needle_puncture_pose_dir),
        "needleGrasping": (args.needle_grasping_data_dir, args.needle_grasping_pose_dir),
        "knotting": (args.knotting_data_dir, args.knotting_pose_dir),
    }
    rarp_children = []
    for name in [n.strip() for n in args.train_rarp_datasets.split(",") if n.strip()]:
        if name not in specs:
            raise ValueError(f"Unknown RARP dataset {name}; choices={sorted(specs)}")
        rarp_children.append(make_rarp_crop_dataset(args, name, "train", True, specs[name][0], specs[name][1], args.train_subsample))
    if not rarp_children:
        raise RuntimeError("at least one RARP training dataset is required")

    groups = [("rarp", _concat_or_single(rarp_children))]
    rates = [args.rarp_sampling_rate]

    if args.use_surgical_instrument:
        groups.append(
            (
                "surgical",
                SurgicalInstrumentCropDataset(
                    root_dir=args.surgical_instrument_root,
                    split="train",
                    training=True,
                    crop_size=args.img_size,
                    subsample=args.train_subsample,
                    num_keypoints=args.num_keypoints,
                    color_jitter=bool(args.color_jitter),
                    rgb_augmentation=bool(args.rgb_augmentation),
                    occlusion_augmentation=bool(args.occlusion_augmentation),
                    occlusion_prob=args.occlusion_prob,
                    cache_dir=args.dataset_cache_dir,
                    enumerate_instances=bool(args.enumerate_surgical_instances),
                ),
            )
        )
        rates.append(args.surgical_sampling_rate)

    vos = make_vos_datasets(args, "train", True, args.train_subsample)
    if vos:
        groups.append(("vos", _concat_or_single(vos)))
        rates.append(args.vos_sampling_rate)

    return FixedRateHybridDataset(
        groups,
        rates,
        samples_per_epoch=args.samples_per_epoch,
        seed=args.sampling_seed,
    )


def make_val_datasets(args):
    specs = {
        "needlePuncture": (args.needle_puncture_data_dir, args.needle_puncture_pose_dir),
        "needleGrasping": (args.needle_grasping_data_dir, args.needle_grasping_pose_dir),
        "knotting": (args.knotting_data_dir, args.knotting_pose_dir),
    }
    datasets = []
    for name in [n.strip() for n in args.val_rarp_datasets.split(",") if n.strip()]:
        if name not in specs:
            raise ValueError(f"Unknown RARP val dataset {name}; choices={sorted(specs)}")
        datasets.append(make_rarp_crop_dataset(args, name, "test", False, specs[name][0], specs[name][1], args.val_subsample))
    if args.use_surgical_instrument and os.path.isdir(os.path.join(args.surgical_instrument_root, "test", "images")):
        datasets.append(
            SurgicalInstrumentCropDataset(
                root_dir=args.surgical_instrument_root,
                split="test",
                training=False,
                crop_size=args.img_size,
                subsample=args.val_subsample,
                num_keypoints=args.num_keypoints,
                color_jitter=False,
                rgb_augmentation=False,
                occlusion_augmentation=False,
                cache_dir=args.dataset_cache_dir,
                enumerate_instances=bool(args.enumerate_surgical_instances),
            )
        )
    datasets.extend(make_vos_datasets(args, "valid", False, args.val_subsample))
    if not datasets:
        raise RuntimeError("validation dataset list is empty")
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
            _, metrics = base.compute_hybrid_crop_losses(out, y, args)
        for key, value in metrics.items():
            meters[key] = meters.get(key, 0.0) + float(value.item())
        count += 1
        if count >= max_batches:
            break
    if count == 0:
        return {}
    return {f"val_{k}": base.reduce_float(v / count, device) for k, v in meters.items()}


def _group_counts_from_loader(loader, max_batches):
    counts = {}
    batches = 0
    for _, y in loader:
        for name in y["sample_group"]:
            counts[name] = counts.get(name, 0) + 1
        batches += 1
        if batches >= max_batches:
            break
    return counts


def main(args):
    if args.img_size != 224:
        raise ValueError("This crop-HCCE script is fixed to --img_size 224.")
    if args.num_keypoints != 5:
        raise ValueError("RARP joint-keypoint crop training expects --num_keypoints 5.")

    device, local_rank, world_size = setup_dist(args.dist_timeout_sec)
    use_ddp = world_size > 1
    torch.backends.cudnn.benchmark = True

    log_dir = Path(args.save_dir) / args.name
    args.log_dir = str(log_dir)
    if args.dataset_cache_dir is None:
        args.dataset_cache_dir = str(log_dir / "dataset_cache")
    if base.is_main_process():
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    if use_ddp:
        if base.is_main_process():
            train_dataset = make_train_dataset(args)
            val_datasets = make_val_datasets(args)
            dist_barrier(local_rank)
        else:
            dist_barrier(local_rank)
            train_dataset = make_train_dataset(args)
            val_datasets = make_val_datasets(args)
    else:
        train_dataset = make_train_dataset(args)
        val_datasets = make_val_datasets(args)

    val_dataset = ConcatDataset(val_datasets)
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if use_ddp else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if use_ddp else None
    persistent_workers = bool(args.persistent_workers) and args.num_workers > 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn_hybrid_crop,
        persistent_workers=persistent_workers,
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
        persistent_workers=persistent_workers,
    )

    if base.is_main_process():
        print(f"LOG_DIR: {log_dir}", flush=True)
        print(f"WORLD_SIZE: {world_size}", flush=True)
        print("MODEL: CropHCCEDenseKeypointDPT, shared DINOv2 encoder, 224 dense outputs", flush=True)
        print("KEYPOINT SUPERVISION: RARP 5 joint-keypoint heatmaps only", flush=True)
        print("HYBRID LOSSES: segmentation all; HCCE/pose/action/keypoint RARP only", flush=True)
        print("PART LABELS: dataset 1=gripper, 2=wrist, 3=shaft; logits [wrist, gripper, shaft]", flush=True)
        print(
            f"SAMPLING RATES: rarp={args.rarp_sampling_rate:.3f}, "
            f"surgical={args.surgical_sampling_rate:.3f}, vos={args.vos_sampling_rate:.3f}",
            flush=True,
        )
        print(f"TRAIN {train_dataset}", flush=True)
        for ds in val_datasets:
            print(f"VAL {ds}", flush=True)
        if not persistent_workers:
            print("DATALOADER: persistent_workers=0 so fixed-rate plans and bbox jitter refresh each epoch", flush=True)

    if args.inspect_datasets_only:
        if base.is_main_process():
            if args.inspect_batches <= 0:
                counts = {}
                for group_idx, _ in train_dataset.plan:
                    name = train_dataset.groups[group_idx][0]
                    counts[name] = counts.get(name, 0) + 1
                print(f"INSPECT_EPOCH_PLAN_COUNTS: {counts}", flush=True)
            else:
                counts = _group_counts_from_loader(train_loader, args.inspect_batches)
                print(f"INSPECT_BATCH_COUNTS({args.inspect_batches} batches): {counts}", flush=True)
        if use_ddp:
            dist.destroy_process_group()
        return

    model = CropHCCEDenseKeypointDPT(
        img_size=args.img_size,
        backbone=args.backbone,
        pretrained_backbone=bool(args.pretrained_backbone),
        hcce_feat_dim=args.hcce_feat_dim,
        hcce_bits=args.hcce_bits,
        num_keypoints=args.num_keypoints,
        pose_head_iter=args.pose_head_iter,
        pose_head_dropout=args.pose_head_dropout,
        keypoint_feat_size=args.keypoint_feat_size,
    ).to(device)

    resume_ckpt = None
    resume_epoch = 0
    resume_iter = 0
    if args.init_from is not None:
        _, report = base.load_compatible_state(model, args.init_from, device)
        if base.is_main_process():
            print(f"INIT_FROM: {args.init_from}", flush=True)
            print(f"loaded compatible tensors: {report['loaded']}, skipped: {len(report['skipped'])}", flush=True)
            if report["skipped"]:
                print("skipped keys head:", report["skipped"][:12], flush=True)
    if args.resume is not None:
        resume_ckpt, report = base.load_compatible_state(model, args.resume, device)
        resume_epoch = int(resume_ckpt.get("epoch", 0))
        resume_iter = int(resume_ckpt.get("iter", 0))
        if bool(args.reset_iter_on_resume):
            resume_epoch = 0
            resume_iter = 0
        if base.is_main_process():
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

    iteration = resume_iter
    epoch = resume_epoch
    last_log = time.time()
    while iteration < args.max_iter:
        base.set_epoch_recursive(train_dataset, epoch)
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
                loss, metrics = base.compute_hybrid_crop_losses(out, y, args)
            if not torch.isfinite(loss):
                if base.is_main_process():
                    bad = {k: float(v.detach().float().cpu().item()) for k, v in metrics.items() if torch.is_tensor(v) and v.numel() == 1}
                    print(f"NONFINITE LOSS at iter {iteration}: {bad}", flush=True)
                raise RuntimeError(f"non-finite training loss at iter {iteration}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            for key, value in metrics.items():
                meters.setdefault(key, base.AverageMeter()).update(float(value.item()))

            if iteration % args.log_freq == 0 and base.is_main_process():
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
                if base.is_main_process():
                    print("VAL " + " | ".join(f"{k} {v:.4f}" for k, v in val.items()), flush=True)
                    base.save_checkpoint(
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

            if iteration % args.ckpt_freq == 0 and base.is_main_process():
                base.save_checkpoint(log_dir / "checkpoints" / "last.pt", model, optimizer, scheduler, scaler, args, epoch, iteration)

            if iteration >= args.max_iter:
                break
        epoch += 1

    if base.is_main_process():
        base.save_checkpoint(log_dir / "checkpoints" / "last.pt", model, optimizer, scheduler, scaler, args, epoch, iteration)
        print(f"Finished training at iter {iteration}", flush=True)
    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="logs")
    parser.add_argument("--name", type=str, default="instrument_hcce_crop_rarp_surgical_vos_fixedmix")
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
    parser.add_argument("--val_rarp_datasets", type=str, default="needlePuncture")
    parser.add_argument("--surgical_instrument_root", type=str, default=RARP50_DIR)
    parser.add_argument("--use_surgical_instrument", type=int, default=1, choices=[0, 1])
    parser.add_argument("--enumerate_surgical_instances", type=int, default=0, choices=[0, 1])
    parser.add_argument("--use_vos_endovis", type=int, default=1, choices=[0, 1])
    parser.add_argument("--vos_endovis17_root", type=str, default=base.VOS_ENDOVIS17_ROOT)
    parser.add_argument("--vos_endovis18_root", type=str, default=base.VOS_ENDOVIS18_ROOT)
    parser.add_argument("--vos_endovis17_image_root", type=str, default=base.VOS_ENDOVIS17_IMAGE_ROOT)
    parser.add_argument("--vos_endovis18_image_root", type=str, default=base.VOS_ENDOVIS18_IMAGE_ROOT)
    parser.add_argument("--num_keypoints", type=int, default=5)

    parser.add_argument("--rarp_sampling_rate", type=float, default=0.75)
    parser.add_argument("--surgical_sampling_rate", type=float, default=0.15)
    parser.add_argument("--vos_sampling_rate", type=float, default=0.10)
    parser.add_argument("--samples_per_epoch", type=int, default=0)
    parser.add_argument("--sampling_seed", type=int, default=12345)

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
    parser.add_argument("--persistent_workers", type=int, default=0, choices=[0, 1])
    parser.add_argument("--dist_timeout_sec", type=int, default=7200)
    parser.add_argument("--train_subsample", type=int, default=1)
    parser.add_argument("--val_subsample", type=int, default=20)
    parser.add_argument("--max_iter", type=int, default=60000)
    parser.add_argument("--log_freq", type=int, default=10)
    parser.add_argument("--val_freq", type=int, default=1000)
    parser.add_argument("--val_batches", type=int, default=20)
    parser.add_argument("--ckpt_freq", type=int, default=1000)
    parser.add_argument("--amp", type=int, default=1, choices=[0, 1])
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--inspect_datasets_only", type=int, default=0, choices=[0, 1])
    parser.add_argument("--inspect_batches", type=int, default=4)

    parser.add_argument("--backbone", type=str, default="dinov2_vits14", choices=["dinov2_vits14", "dinov2_vitb14"])
    parser.add_argument("--pretrained_backbone", type=int, default=1, choices=[0, 1])
    parser.add_argument("--hcce_feat_dim", type=int, default=256)
    parser.add_argument("--hcce_bits", type=int, default=8)
    parser.add_argument("--pose_head_iter", type=int, default=4)
    parser.add_argument("--pose_head_dropout", type=float, default=0.3)
    parser.add_argument("--keypoint_feat_size", type=int, default=14)

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
