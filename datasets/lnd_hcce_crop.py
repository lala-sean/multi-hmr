import importlib.util
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

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


_rarp_hcce_crop = _load_local_module("robopepp_rarp_hcce_crop_for_lnd", ROBOPEPP_ROOT / "datasets" / "rarp_hcce_crop.py")
_lnd_instrument = _load_local_module(
    "robopepp_surgripe_lnd_instrument_for_hcce",
    ROBOPEPP_ROOT / "datasets" / "surgripe_lnd_instrument.py",
)
_crop_resize_pad_map = _rarp_hcce_crop._crop_resize_pad_map
_current_egl_device_id = _rarp_hcce_crop._current_egl_device_id
RoboPEPPSurgripeLNDInstrument = _lnd_instrument.RoboPEPPSurgripeLNDInstrument


class SurgripeLNDHCCECropDataset(RoboPEPPSurgripeLNDInstrument):
    """
    LND crop dataset with the same RoboPEPP crop/keypoint targets plus dense
    HCCE supervision rendered from the repo-format wrist pose.
    """

    def __init__(
        self,
        *args,
        render_on_the_fly=True,
        coord_render_backend="trimesh",
        require_cse=True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.render_on_the_fly = bool(render_on_the_fly)
        self.coord_render_backend = str(coord_render_backend)
        self.require_cse = bool(require_cse)
        self._cse_renderer = None

    def __repr__(self):
        return (
            f"surgripe_lnd_hcce_crop: split={self.split} training={self.training} "
            f"N={len(self)} bbox_padding_frac={self.bbox_padding_frac} "
            f"render_on_the_fly={self.render_on_the_fly} backend={self.coord_render_backend}"
        )

    @staticmethod
    def _pose_from_target(target):
        action = target["action"].detach().cpu().numpy().astype(np.float32).reshape(3)
        return {
            "rot": target["wrist_quat"].detach().cpu().numpy().astype(np.float32),
            "trans": target["wrist_trans"].detach().cpu().numpy().astype(np.float32),
            "alpha": float(action[0]),
            "theta_l": float(action[1]),
            "theta_r": float(action[2]),
        }

    def _load_cse_coord(self, pose, K, height, width):
        if not self.render_on_the_fly:
            if self.require_cse:
                raise RuntimeError("LND HCCE requires render_on_the_fly=1; no cached coord map path is defined.")
            return None
        if self.coord_render_backend != "trimesh":
            raise ValueError(f"Unknown LND coord_render_backend: {self.coord_render_backend}")
        if self._cse_renderer is None:
            from instrument_trimesh_renderer import GMSInstrumentTrimeshRenderer

            if not torch.cuda.is_available():
                raise RuntimeError("LND HCCE coordinate rendering requires CUDA.")
            self._cse_renderer = GMSInstrumentTrimeshRenderer(torch.device(f"cuda:{torch.cuda.current_device()}"))
        egl_idx = int(os.environ.get("EGL_DEVICE_ID", _current_egl_device_id()))
        return self._cse_renderer.render_canonical_coords(
            pose,
            np.asarray(K, dtype=np.float32),
            (int(height), int(width)),
            device_idx=egl_idx,
        ).astype(np.float32)

    def __getitem__(self, idx):
        image_tensor, target = super().__getitem__(idx)
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

        pose = self._pose_from_target(target)
        K_orig = target["K_orig"].numpy().astype(np.float32)
        coord_img = self._load_cse_coord(pose, K_orig, height, width)
        coord_crop = self._crop_coord(coord_img, bbox_min, bbox_max)
        has_cse = bool((coord_crop[..., 3] > 0).any())
        if self.require_cse and not has_cse:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            time.sleep(0.05)
            self._cse_renderer = None
            coord_img = self._load_cse_coord(pose, K_orig, height, width)
            coord_crop = self._crop_coord(coord_img, bbox_min, bbox_max)
            has_cse = bool((coord_crop[..., 3] > 0).any())
        if self.require_cse and not has_cse:
            raise RuntimeError(f"Empty LND HCCE coord crop at sample index {idx}, frame {target['frame_id']}")

        target.update(
            {
                "inst_mask": torch.from_numpy(inst_mask_crop),
                "part_mask": torch.from_numpy(part_mask_crop),
                "coord_img": torch.from_numpy(coord_crop),
                "has_cse": torch.tensor(has_cse, dtype=torch.bool),
            }
        )
        return image_tensor, target

    def _crop_coord(self, coord_img, bbox_min, bbox_max):
        return _crop_resize_pad_map(
            coord_img,
            bbox_min,
            bbox_max,
            self.crop_size,
            interpolation=cv2.INTER_NEAREST,
            value=0,
        ).astype(np.float32)
