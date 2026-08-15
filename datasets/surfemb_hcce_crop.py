import importlib.util
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

ROBOPEPP_ROOT = Path(__file__).resolve().parents[1]
MULTIHMR_ROOT = Path(__file__).resolve().parents[3]
if str(ROBOPEPP_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBOPEPP_ROOT))
if str(MULTIHMR_ROOT) not in sys.path:
    sys.path.insert(0, str(MULTIHMR_ROOT))

from instrument_geometry import (  # noqa: E402
    SURFEMB_SHAFT_NORM_X_MIN,
    project_points_np,
    surfemb_surface_sampling_mask,
)


def _load_local_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_rarp_hcce_crop = _load_local_module(
    "robopepp_rarp_hcce_crop_for_surfemb",
    ROBOPEPP_ROOT / "datasets" / "rarp_hcce_crop.py",
)
_lnd_hcce_crop = _load_local_module(
    "robopepp_lnd_hcce_crop_for_surfemb",
    ROBOPEPP_ROOT / "datasets" / "lnd_hcce_crop.py",
)
_surf_aug = _load_local_module(
    "robopepp_surfemb_augment",
    ROBOPEPP_ROOT / "datasets" / "surfemb_augment.py",
)

RARPCropHCCEDataset = _rarp_hcce_crop.RARPCropHCCEDataset
SurgripeLNDHCCECropDataset = _lnd_hcce_crop.SurgripeLNDHCCECropDataset


def _as_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _create_belief_maps(image_resolution, points, valid, sigma=2.0):
    width, height = image_resolution
    out = np.zeros((len(points), height, width), dtype=np.float32)
    radius = int(float(sigma) * 2.0)
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
                out[i, y, x] = np.exp(-(((x - u) ** 2 + (y - v) ** 2) / (2.0 * float(sigma) ** 2)))
    return out


def _default_surface_points_path():
    refined = (
        ROBOPEPP_ROOT
        / "assets"
        / "instrument_surface_samples_surfemb_x2.13mm_wg1over3_shafttop30mm"
        / "instrument_surface_points_all.npy"
    )
    if refined.is_file():
        return refined
    return ROBOPEPP_ROOT / "assets" / "instrument_surface_samples" / "instrument_surface_points_all.npy"


class SurfEmbCropAugmentedDataset(Dataset):
    """
    SurfEmb-style crop/augmentation wrapper for the existing HCCE crop datasets.

    The wrapped dataset still owns all dataset-specific pose, mask, symmetry and
    keypoint logic.  This wrapper replaces the final image/label transform with
    SurfEmb's RandomRotatedMaskCrop and photometric augmentations, and applies
    the same affine crop to RGB, masks, canonical coordinate image, keypoints
    and K.
    """

    def __init__(
        self,
        base_dataset,
        surface_points_path=None,
        crop_size=224,
        n_pos=1024,
        n_neg=1024,
        crop_scale=1.2,
        max_angle=math.pi,
        offset_scale=1.0,
        training=True,
        heatmap_sigma=2.0,
        ensure_full_mask=True,
        shaft_norm_x_min=SURFEMB_SHAFT_NORM_X_MIN,
    ):
        super().__init__()
        self.base_dataset = base_dataset
        self.surface_points_path = Path(surface_points_path) if surface_points_path else _default_surface_points_path()
        self.crop_size = int(crop_size)
        self.n_pos = int(n_pos)
        self.n_neg = int(n_neg)
        self.crop_scale = float(crop_scale)
        self.max_angle = float(max_angle)
        self.offset_scale = float(offset_scale)
        self.training = bool(training)
        self.heatmap_sigma = float(heatmap_sigma)
        self.ensure_full_mask = bool(ensure_full_mask)
        self.surface_points, self.surface_part_ids = _surf_aug.load_surface_points(self.surface_points_path)
        self.shaft_norm_x_min = float(shaft_norm_x_min)
        self.surface_sample_indices = np.flatnonzero(
            surfemb_surface_sampling_mask(
                self.surface_points,
                self.surface_part_ids,
                self.shaft_norm_x_min,
            )
        )

    def __len__(self):
        return len(self.base_dataset)

    def __repr__(self):
        return (
            f"surfemb_crop_wrapper(training={self.training} N={len(self)} "
            f"crop_size={self.crop_size} n_pos={self.n_pos} n_neg={self.n_neg} "
            f"crop_scale={self.crop_scale} max_angle={self.max_angle:.3f} "
            f"offset_scale={self.offset_scale} shaft_norm_x_min={self.shaft_norm_x_min:g} "
            f"base={self.base_dataset})"
        )

    def set_epoch(self, epoch):
        if hasattr(self.base_dataset, "set_epoch"):
            self.base_dataset.set_epoch(epoch)

    def _make_crop_matrix(self, union_mask):
        max_angle = self.max_angle if self.training else 0.0
        offset_scale = self.offset_scale if self.training else 0.0
        return _surf_aug.random_rotated_mask_crop_matrix(
            union_mask,
            self.crop_size,
            crop_scale=self.crop_scale,
            max_angle=max_angle,
            offset_scale=offset_scale,
            ensure_full_mask=self.ensure_full_mask,
        )

    def _warp_rgb(self, rgb, M):
        rgb = np.asarray(rgb, dtype=np.uint8)
        if self.training:
            aug_rgb = _surf_aug.surfemb_precrop_photometric(rgb)
            crop_rgb = _surf_aug.warp_rgb(aug_rgb, M, self.crop_size)
            return _surf_aug.surfemb_postcrop_photometric(crop_rgb)
        return cv2.warpAffine(
            rgb,
            M,
            (self.crop_size, self.crop_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

    def __getitem__(self, idx):
        _, target = self.base_dataset[idx]
        rgb = _as_numpy(target["crop_rgb"]).astype(np.uint8)
        inst = (_as_numpy(target["inst_mask"]) > 0).astype(np.uint8)
        part = _as_numpy(target["part_mask"]).astype(np.uint8)
        coord = _as_numpy(target["coord_img"]).astype(np.float32)
        coord_mask = coord[..., 3] > 0
        union_mask = (inst > 0) | (part > 0) | coord_mask
        M = self._make_crop_matrix(union_mask)
        M3 = _surf_aug.matrix3_from_affine(M)

        crop_rgb = self._warp_rgb(rgb, M)
        inst_crop = _surf_aug.warp_map(inst, M, self.crop_size, interpolation=cv2.INTER_NEAREST, value=0)
        part_crop = _surf_aug.warp_map(part, M, self.crop_size, interpolation=cv2.INTER_NEAREST, value=0)
        coord_crop = _surf_aug.warp_map(coord, M, self.crop_size, interpolation=cv2.INTER_NEAREST, value=0).astype(np.float32)
        coord_crop[..., 3] = (coord_crop[..., 3] > 0).astype(np.float32)

        keypoints_crop = _surf_aug.transform_points(_as_numpy(target["keypoints_crop"]), M)
        keypoints_valid = _as_numpy(target["keypoints_valid"]).astype(bool)
        inside = (
            (keypoints_crop[:, 0] >= 0.0)
            & (keypoints_crop[:, 0] < float(self.crop_size))
            & (keypoints_crop[:, 1] >= 0.0)
            & (keypoints_crop[:, 1] < float(self.crop_size))
        )
        keypoints_valid = keypoints_valid & inside
        heatmaps = _create_belief_maps(
            (self.crop_size, self.crop_size),
            keypoints_crop,
            keypoints_valid,
            sigma=self.heatmap_sigma,
        )
        K = M3 @ _as_numpy(target["K"]).astype(np.float32)

        target_out = dict(target)
        target_out.update(
            {
                "heatmaps": torch.from_numpy(heatmaps),
                "keypoints_crop": torch.from_numpy(keypoints_crop.astype(np.float32)),
                "keypoints_valid": torch.from_numpy(keypoints_valid.astype(np.bool_)),
                "K": torch.from_numpy(K.astype(np.float32)),
                "crop_rgb": torch.from_numpy(crop_rgb.astype(np.uint8)),
                "inst_mask": torch.from_numpy(inst_crop.astype(np.float32)),
                "part_mask": torch.from_numpy(part_crop.astype(np.int64)),
                "coord_img": torch.from_numpy(coord_crop.astype(np.float32)),
                "has_cse": torch.tensor(bool((coord_crop[..., 3] > 0).any()), dtype=torch.bool),
                "surfemb_M_crop": torch.from_numpy(M.astype(np.float32)),
                "surfemb_obj_idx": torch.tensor(0, dtype=torch.long),
            }
        )

        sample_eligible = surfemb_surface_sampling_mask(
            coord_crop[..., :3].reshape(-1, 3),
            coord_crop[..., 3].astype(np.int64).reshape(-1),
            self.shaft_norm_x_min,
        ).reshape(coord_crop.shape[:2])
        pos_mask = (coord_crop[..., 3] > 0) & (inst_crop > 0) & sample_eligible
        target_out["surfemb_mask_samples"] = torch.from_numpy(_surf_aug.sample_mask_yx(pos_mask, self.n_pos))
        if len(self.surface_sample_indices) == 0:
            raise RuntimeError("surface point pool is empty")
        surf_idx = np.random.choice(
            self.surface_sample_indices,
            int(self.n_neg),
            replace=int(self.n_neg) > len(self.surface_sample_indices),
        )
        target_out["surfemb_surface_samples"] = torch.from_numpy(self.surface_points[surf_idx].astype(np.float32))
        target_out["surfemb_surface_part_ids"] = torch.from_numpy(self.surface_part_ids[surf_idx].astype(np.int64))

        kp3d = _as_numpy(target["keypoints_3d_cam"]).astype(np.float32)
        uv = project_points_np(kp3d, K).astype(np.float32)
        valid = keypoints_valid & np.isfinite(uv).all(axis=-1) & (kp3d[:, 2] > 1e-4)
        if valid.any():
            resid = float(np.max(np.linalg.norm(uv[valid] - keypoints_crop[valid], axis=-1)))
        else:
            resid = float("nan")
        target_out["surfemb_kpt_proj_resid_px"] = torch.tensor(resid, dtype=torch.float32)

        image_tensor = _surf_aug.imagenet_tensor(crop_rgb)
        return image_tensor, target_out


def collate_fn_surfemb_crop(batch):
    images, targets = zip(*batch)
    x = torch.stack(images, dim=0)
    tensor_keys = [
        "action",
        "wrist_quat",
        "wrist_trans",
        "heatmaps",
        "keypoints_crop",
        "keypoints_orig",
        "keypoints_3d_cam",
        "keypoints_valid",
        "keypoints_valid_orig",
        "keypoint_hit_labels",
        "K",
        "K_orig",
        "bbox_min",
        "bbox_max",
        "scale",
        "pad",
        "pose_sym_flipped",
        "instance_id",
        "orig_size",
        "inst_mask",
        "part_mask",
        "coord_img",
        "has_cse",
        "surfemb_M_crop",
        "surfemb_obj_idx",
        "surfemb_mask_samples",
        "surfemb_surface_samples",
        "surfemb_surface_part_ids",
        "surfemb_kpt_proj_resid_px",
    ]
    y = {}
    for key in tensor_keys:
        y[key] = torch.stack([t[key] for t in targets], dim=0)
    y["video_name"] = [t["video_name"] for t in targets]
    y["frame_id"] = [t["frame_id"] for t in targets]
    y["img_path"] = [t["img_path"] for t in targets]
    return x, y


def save_surfemb_crop_debug_panel(path, target):
    from PIL import Image

    rgb = _as_numpy(target["crop_rgb"]).astype(np.uint8)
    inst = (_as_numpy(target["inst_mask"]) > 0).astype(np.uint8)
    part = _as_numpy(target["part_mask"]).astype(np.uint8)
    coord = _as_numpy(target["coord_img"]).astype(np.float32)
    heat = _as_numpy(target["heatmaps"]).max(axis=0)
    colors = np.array(
        [
            [0, 0, 0],
            [230, 80, 80],
            [80, 220, 120],
            [80, 150, 240],
        ],
        dtype=np.uint8,
    )
    part_rgb = colors[np.clip(part, 0, 3)]
    overlay = rgb.copy()
    overlay[inst > 0] = (0.55 * overlay[inst > 0] + 0.45 * part_rgb[inst > 0]).astype(np.uint8)
    kp = _as_numpy(target["keypoints_crop"]).astype(np.float32)
    valid = _as_numpy(target["keypoints_valid"]).astype(bool)
    for u, v in kp[valid]:
        cv2.circle(overlay, (int(round(float(u))), int(round(float(v)))), 3, (255, 255, 0), -1)
    coord_rgb = ((coord[..., :3] + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)
    coord_rgb[coord[..., 3] <= 0] = 0
    heat_rgb = cv2.applyColorMap((heat * 255.0).clip(0, 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
    heat_rgb = cv2.cvtColor(heat_rgb, cv2.COLOR_BGR2RGB)
    panel = np.concatenate([rgb, overlay, part_rgb, coord_rgb, heat_rgb], axis=1)
    Image.fromarray(panel).save(path)
