import importlib.util
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROBOPEPP_ROOT = Path(__file__).resolve().parents[1]
MULTIHMR_ROOT = Path(__file__).resolve().parents[3]
if str(ROBOPEPP_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBOPEPP_ROOT))
if str(MULTIHMR_ROOT) not in sys.path:
    sys.path.insert(0, str(MULTIHMR_ROOT))


def _load_local_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_rarp_instrument = _load_local_module(
    "robopepp_rarp_instrument_for_hcce",
    ROBOPEPP_ROOT / "datasets" / "rarp_instrument.py",
)
RoboPEPPRARPInstrument = _rarp_instrument.RoboPEPPRARPInstrument

# rarp_instrument.py installs the multi-hmr datasets package into sys.modules.
from datasets.RarpInstanceDataset import RARPInstanceDataset  # noqa: E402


def _resize_longer_side_array(arr, crop_size, interpolation):
    h, w = arr.shape[:2]
    if w > h:
        new_w = int(crop_size)
        new_h = int(crop_size * h / w)
    else:
        new_h = int(crop_size)
        new_w = int(crop_size * w / h)
    return cv2.resize(arr, (new_w, new_h), interpolation=interpolation), (new_w, new_h)


def _pad_2d(arr, crop_size, value=0):
    h, w = arr.shape[:2]
    pad_h = (crop_size - h) // 2
    pad_w = (crop_size - w) // 2
    if arr.ndim == 2:
        padding = ((pad_h, crop_size - h - pad_h), (pad_w, crop_size - w - pad_w))
    else:
        padding = ((pad_h, crop_size - h - pad_h), (pad_w, crop_size - w - pad_w), (0, 0))
    return np.pad(arr, padding, mode="constant", constant_values=value)


def _crop_resize_pad_map(arr, bbox_min, bbox_max, crop_size, interpolation, value=0):
    x0, y0 = bbox_min.astype(np.int64)
    x1, y1 = np.ceil(bbox_max).astype(np.int64)
    crop = arr[y0:y1, x0:x1]
    resized, _ = _resize_longer_side_array(crop, crop_size, interpolation)
    return _pad_2d(resized, crop_size, value=value)


def _current_egl_device_id():
    if not torch.cuda.is_available():
        return 0
    current = int(torch.cuda.current_device())
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible.strip():
        ids = [v.strip() for v in visible.split(",") if v.strip()]
        if current < len(ids):
            try:
                return int(ids[current])
            except ValueError:
                pass
    return current


class RARPCropHCCEDataset(RoboPEPPRARPInstrument):
    """
    RoboPEPP-aligned 224x224 crop dataset for single-instrument HCCE training.

    RGB crop, bbox jitter, color jitter, synthetic occlusion and RGB augmentation
    are inherited from RoboPEPPRARPInstrument.  This wrapper adds dense crop
    targets for HCCE/part/instance/keypoint heads at the same 224 resolution.
    """

    def __init__(
        self,
        *args,
        cse_coord_root=None,
        render_on_the_fly=True,
        coord_render_backend="trimesh",
        require_cse=True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.cse_coord_root = cse_coord_root
        self.render_on_the_fly = bool(render_on_the_fly)
        self.coord_render_backend = str(coord_render_backend)
        self.require_cse = bool(require_cse)
        self._cse_renderer = None

    def _load_cse_coord(self, video_name, frame_id, instance_id, pose, height, width):
        coord_img = None
        if self.cse_coord_root is not None:
            coord_file = os.path.join(
                self.cse_coord_root,
                f"SARRARP502022_{video_name}_instance{instance_id}",
                f"{frame_id}.npy",
            )
            if not os.path.isfile(coord_file):
                raise FileNotFoundError(f"CSE coord file not found: {coord_file}")
            coord_img = np.load(coord_file).astype(np.float32)

        if coord_img is None and self.render_on_the_fly:
            if self.coord_render_backend == "trimesh":
                if self._cse_renderer is None:
                    from instrument_trimesh_renderer import GMSInstrumentTrimeshRenderer

                    if not torch.cuda.is_available():
                        raise RuntimeError("trimesh canonical coordinate rendering requires CUDA for GMS Instrument.")
                    self._cse_renderer = GMSInstrumentTrimeshRenderer(
                        torch.device(f"cuda:{torch.cuda.current_device()}")
                    )
                K = _rarp_instrument.rarp_intrinsics(width, height)
                egl_idx = int(os.environ.get("EGL_DEVICE_ID", _current_egl_device_id()))
                coord_img = self._cse_renderer.render_canonical_coords(
                    pose,
                    K,
                    (height, width),
                    device_idx=egl_idx,
                )
            elif self.coord_render_backend == "gaussian":
                if self._cse_renderer is None:
                    gms_root = MULTIHMR_ROOT / "submodules" / "gaussian-mesh-splatting"
                    for p in (
                        gms_root / "submodules" / "depth-diff-gaussian-rasterization",
                        gms_root / "submodules" / "simple-knn",
                    ):
                        p_str = str(p)
                        if p_str not in sys.path:
                            sys.path.insert(0, p_str)
                    self._cse_renderer = RARPInstanceDataset(
                        split="train",
                        training=False,
                        img_size=630,
                        dataset_root=self.dataset_root,
                        pose_root=self.pose_root,
                        min_dice=[0.8, 0.6, 0.6],
                        train_ratio=0.95,
                        subsample=1,
                        n=0,
                        cse_coord_root=None,
                        render_on_the_fly=True,
                        canonicalize_pose_symmetry=False,
                        random_resample=False,
                    )
                pose_params = {
                    "rot": torch.from_numpy(pose["rot"]).float(),
                    "trans": torch.from_numpy(pose["trans"]).float(),
                    "alpha": torch.tensor([pose["alpha"]], dtype=torch.float32),
                    "theta_l": torch.tensor([pose["theta_l"]], dtype=torch.float32),
                    "theta_r": torch.tensor([pose["theta_r"]], dtype=torch.float32),
                }
                coord_img = self._cse_renderer._render_cse_coord(pose_params, height, width)
            else:
                raise ValueError(f"Unknown coord_render_backend: {self.coord_render_backend}")

        if coord_img is None and self.require_cse:
            raise RuntimeError(
                f"Missing HCCE coord target for {video_name}/{frame_id}/instance{instance_id}"
            )
        if coord_img is not None and coord_img.shape[:2] != (height, width):
            raise ValueError(
                f"CSE coord shape {coord_img.shape[:2]} does not match image {(height, width)}"
            )
        return coord_img

    def __getitem__(self, idx):
        image_tensor, target = super().__getitem__(idx)
        video_name, frame_id, instance_id, _ = self.samples[idx]
        height, width = [int(v) for v in target["orig_size"].tolist()]
        bbox_min = target["bbox_min"].numpy().astype(np.float32)
        bbox_max = target["bbox_max"].numpy().astype(np.float32)

        inst_mask_orig = target["inst_mask_orig"].numpy().astype(np.uint8)
        part_mask_orig = target["part_mask_orig"].numpy().astype(np.uint8)
        inst_mask_crop = _crop_resize_pad_map(
            inst_mask_orig,
            bbox_min,
            bbox_max,
            self.crop_size,
            interpolation=cv2.INTER_NEAREST,
            value=0,
        ).astype(np.float32)
        part_mask_crop = _crop_resize_pad_map(
            part_mask_orig,
            bbox_min,
            bbox_max,
            self.crop_size,
            interpolation=cv2.INTER_NEAREST,
            value=0,
        ).astype(np.int64)

        pose = self._load_pose(video_name, frame_id, int(instance_id))
        coord_img = self._load_cse_coord(video_name, frame_id, int(instance_id), pose, height, width)
        if coord_img is None:
            coord_crop = np.zeros((self.crop_size, self.crop_size, 4), dtype=np.float32)
            has_cse = False
        else:
            coord_crop = _crop_resize_pad_map(
                coord_img.astype(np.float32),
                bbox_min,
                bbox_max,
                self.crop_size,
                interpolation=cv2.INTER_NEAREST,
                value=0,
            ).astype(np.float32)
            has_cse = bool((coord_crop[..., 3] > 0).any())

        target.update(
            {
                "inst_mask": torch.from_numpy(inst_mask_crop),
                "part_mask": torch.from_numpy(part_mask_crop),
                "coord_img": torch.from_numpy(coord_crop),
                "has_cse": torch.tensor(has_cse, dtype=torch.bool),
            }
        )
        return image_tensor, target


def collate_fn_rarp_crop_hcce(batch):
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
    ]
    y = {}
    for key in tensor_keys:
        y[key] = torch.stack([t[key] for t in targets], dim=0)
    y["video_name"] = [t["video_name"] for t in targets]
    y["frame_id"] = [t["frame_id"] for t in targets]
    y["img_path"] = [t["img_path"] for t in targets]
    return x, y


def save_crop_debug_panel(path, target):
    rgb = target["crop_rgb"].numpy().astype(np.uint8)
    inst = (target["inst_mask"].numpy() > 0.5).astype(np.uint8)
    part = target["part_mask"].numpy().astype(np.uint8)
    coord = target["coord_img"].numpy()
    heat = target["heatmaps"].numpy().max(axis=0)

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

    kp = target["keypoints_crop"].numpy()
    valid = target["keypoints_valid"].numpy().astype(bool)
    for i, (u, v) in enumerate(kp):
        if valid[i]:
            cv2.circle(overlay, (int(round(u)), int(round(v))), 3, (255, 255, 0), -1)

    coord_rgb = ((coord[..., :3] + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)
    coord_rgb[coord[..., 3] <= 0] = 0
    heat_rgb = cv2.applyColorMap((heat * 255.0).clip(0, 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
    heat_rgb = cv2.cvtColor(heat_rgb, cv2.COLOR_BGR2RGB)
    panel = np.concatenate([rgb, overlay, part_rgb, coord_rgb, heat_rgb], axis=1)
    Image.fromarray(panel).save(path)
