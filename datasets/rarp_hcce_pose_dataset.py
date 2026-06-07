import os
import random
import sys

import numpy as np
import torch

from datasets.RarpInstanceDataset import RARPInstanceDataset
from datasets.collate_cse import collate_fn_instrument_cse
from datasets.rarp_pose_canonicalization import canonicalize_pose_symmetry as _canonicalize_pose_symmetry


class _PoseStub:
    def __setstate__(self, state):
        self.__dict__.update(state if isinstance(state, dict) else {})


_main_mod = sys.modules.get("__main__")
if _main_mod is not None and not hasattr(_main_mod, "Pose"):
    setattr(_main_mod, "Pose", _PoseStub)


class RARPHCCEPoseDataset(torch.utils.data.Dataset):
    """
    New wrapper that adds action/wrist-pose supervision without changing
    RARPInstanceDataset.
    """

    def __init__(
        self,
        *args,
        training=False,
        canonicalize_pose_symmetry=False,
        canonical_eps=0.08,
        **kwargs,
    ):
        super().__init__()
        self.training = training
        self.canonicalize_pose_symmetry = bool(canonicalize_pose_symmetry)
        self.canonical_eps = float(canonical_eps)
        self.base = RARPInstanceDataset(
            *args,
            training=training,
            canonicalize_pose_symmetry=self.canonicalize_pose_symmetry,
            canonical_eps=self.canonical_eps,
            random_resample=False,
            **kwargs,
        )
        self.pose_cache = {}

    @property
    def name(self):
        return self.base.name

    @property
    def split(self):
        return self.base.split

    @property
    def samples(self):
        return self.base.samples

    def __len__(self):
        return len(self.base)

    def __repr__(self):
        return f"rarp_hcce_pose: split={self.split} - N={len(self)}"

    def __getitem__(self, idx):
        if self.training:
            idx = random.randrange(len(self.base))
        img, annot = self.base[idx]
        video_name, frame_id, instances_list = self.base.samples[idx]
        instance_ids = [instance_id for instance_id, _ in instances_list]
        instruments = annot["instruments"]
        for inst, fallback_instance_id in zip(instruments, instance_ids):
            instance_id = inst.get("instance_id", fallback_instance_id)
            pose = self._load_pose(video_name, frame_id, instance_id)
            inst["action_gt"] = np.array(
                [pose["alpha"], pose["theta_l"], pose["theta_r"]],
                dtype=np.float32,
            )
            inst["wrist_quat_gt"] = pose["rot"].astype(np.float32)
            inst["wrist_trans_gt"] = pose["trans"].astype(np.float32)
            inst["pose_sym_flipped"] = bool(pose["pose_sym_flipped"])
            inst["has_pose"] = True
        return img, annot

    def _load_pose(self, video_name, frame_id, instance_id):
        key = (video_name, frame_id, instance_id)
        if key in self.pose_cache:
            return self.pose_cache[key]
        case_dir = os.path.join(
            self.base.pose_root,
            f"SARRARP502022_{video_name}_instance{instance_id}",
        )
        mem_path = os.path.join(case_dir, "memory_pool.pth")
        if not os.path.isfile(mem_path):
            raise FileNotFoundError(f"memory_pool.pth not found: {mem_path}")
        memory_pool = torch.load(mem_path, map_location="cpu", weights_only=False)
        if frame_id not in memory_pool:
            raise KeyError(f"frame {frame_id} not in {mem_path}")
        pose = memory_pool[frame_id]["pose_info"]
        pose_dict = {
            "rot": pose.rot.detach().float().cpu().reshape(-1).numpy(),
            "trans": pose.trans.detach().float().cpu().reshape(3).numpy(),
            "alpha": float(pose.alpha.detach().float().reshape(-1)[0]),
            "theta_l": float(pose.theta_l.detach().float().reshape(-1)[0]),
            "theta_r": float(pose.theta_r.detach().float().reshape(-1)[0]),
        }
        out_t = _canonicalize_pose_symmetry(
            pose_dict,
            eps=self.canonical_eps,
            enabled=self.canonicalize_pose_symmetry,
        )
        out = {
            "rot": out_t["rot"].detach().cpu().numpy(),
            "trans": out_t["trans"].detach().cpu().numpy(),
            "alpha": float(out_t["alpha"].reshape(-1)[0]),
            "theta_l": float(out_t["theta_l"].reshape(-1)[0]),
            "theta_r": float(out_t["theta_r"].reshape(-1)[0]),
            "pose_sym_flipped": bool(out_t["pose_sym_flipped"].item()),
        }
        if out["rot"].shape[0] != 4:
            raise ValueError(f"Expected quaternion with 4 values, got {out['rot'].shape}")
        self.pose_cache[key] = out
        return out


def collate_fn_instrument_hcce_pose(batch):
    img_array, y = collate_fn_instrument_cse(batch)
    bs = len(batch)
    max_inst = y["valid_instruments"].shape[1]
    action_gt = np.zeros((bs, max_inst, 3), dtype=np.float32)
    wrist_quat_gt = np.zeros((bs, max_inst, 4), dtype=np.float32)
    wrist_trans_gt = np.zeros((bs, max_inst, 3), dtype=np.float32)
    has_pose = np.zeros((bs, max_inst), dtype=bool)
    pose_sym_flipped = np.zeros((bs, max_inst), dtype=bool)
    for i in range(bs):
        for j, inst in enumerate(batch[i][1]["instruments"]):
            if inst.get("has_pose", False):
                action_gt[i, j] = inst["action_gt"]
                wrist_quat_gt[i, j] = inst["wrist_quat_gt"]
                wrist_trans_gt[i, j] = inst["wrist_trans_gt"]
                has_pose[i, j] = True
                pose_sym_flipped[i, j] = bool(inst.get("pose_sym_flipped", False))
    y["action_gt"] = torch.from_numpy(action_gt)
    y["wrist_quat_gt"] = torch.from_numpy(wrist_quat_gt)
    y["wrist_trans_gt"] = torch.from_numpy(wrist_trans_gt)
    y["has_pose"] = torch.from_numpy(has_pose)
    y["pose_sym_flipped"] = torch.from_numpy(pose_sym_flipped)
    return img_array, y
