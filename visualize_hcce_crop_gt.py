import importlib.util
import json
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

ROBOPEPP_ROOT = Path(__file__).resolve().parent
MULTIHMR_ROOT = ROBOPEPP_ROOT.parents[1]
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


_dataset_module = _load_local_module("robopepp_hcce_crop_dataset_vis", ROBOPEPP_ROOT / "datasets" / "rarp_hcce_crop.py")
RARPCropHCCEDataset = _dataset_module.RARPCropHCCEDataset

from multi_instrument.hcce_codec import normalized_xyz_to_hcce  # noqa: E402


PUNCTURE_DATASET_ROOT = "/mnt/nas/share/shuojue/data/needlePuncture_videos"
PUNCTURE_POSE_ROOT = "/mnt/nas/share/shuojue/data/needlePuncture_results"


PART_COLORS = np.array(
    [
        [0, 0, 0],
        [230, 80, 80],    # gripper
        [80, 220, 120],   # wrist
        [80, 150, 240],   # shaft
    ],
    dtype=np.uint8,
)

COORD_PART_COLORS = np.array(
    [
        [0, 0, 0],
        [80, 150, 240],   # shaft
        [80, 220, 120],   # wrist
        [245, 120, 60],   # left gripper
        [190, 80, 230],   # right gripper
    ],
    dtype=np.uint8,
)


def _label(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 20), (0, 0, 0), -1)
    cv2.putText(out, text, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _overlay_part(rgb, part, keypoints, valid):
    part_rgb = PART_COLORS[np.clip(part, 0, 3)]
    mask = part > 0
    out = rgb.copy()
    out[mask] = (0.55 * out[mask] + 0.45 * part_rgb[mask]).astype(np.uint8)
    for i, (u, v) in enumerate(keypoints):
        color = (255, 255, 0) if bool(valid[i]) else (255, 0, 255)
        cv2.circle(out, (int(round(u)), int(round(v))), 3, color, -1)
        cv2.putText(out, str(i), (int(round(u)) + 3, int(round(v)) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    return out


def _coord_rgb(coord):
    valid = coord[..., 3] > 0
    out = ((coord[..., :3] + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)
    out[~valid] = 0
    return out


def _hcce_first_rgb(coord, bits=8):
    xyz = torch.from_numpy(coord[..., :3].astype(np.float32))
    hcce = normalized_xyz_to_hcce(xyz, iteration=bits).numpy()
    valid = coord[..., 3] > 0
    first = np.stack([hcce[..., 0], hcce[..., bits], hcce[..., bits * 2]], axis=-1)
    out = (first * 255.0).clip(0, 255).astype(np.uint8)
    out[~valid] = 0
    return out, hcce


def _bit_grid(hcce, valid, bits=8, thumb=56):
    rows = []
    for axis, axis_name in enumerate(["x", "y", "z"]):
        tiles = []
        for b in range(bits):
            ch = hcce[..., axis * bits + b]
            tile = (ch * 255.0).clip(0, 255).astype(np.uint8)
            tile[~valid] = 0
            tile = cv2.resize(tile, (thumb, thumb), interpolation=cv2.INTER_NEAREST)
            tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2RGB)
            tile = _label(tile, f"{axis_name}{b}")
            tiles.append(tile)
        rows.append(np.concatenate(tiles, axis=1))
    return np.concatenate(rows, axis=0)


def _make_panel(target, bits=8):
    rgb = target["crop_rgb"].numpy().astype(np.uint8)
    inst = target["inst_mask"].numpy() > 0.5
    part = target["part_mask"].numpy().astype(np.uint8)
    coord = target["coord_img"].numpy().astype(np.float32)
    keypoints = target["keypoints_crop"].numpy()
    valid_kp = target["keypoints_valid"].numpy().astype(bool)

    overlay = _overlay_part(rgb, part, keypoints, valid_kp)
    inst_rgb = np.zeros_like(rgb)
    inst_rgb[inst] = [255, 255, 255]
    part_rgb = PART_COLORS[np.clip(part, 0, 3)]
    coord_rgb = _coord_rgb(coord)
    coord_part = COORD_PART_COLORS[np.clip(coord[..., 3].astype(np.int64), 0, 4)]
    hcce_first, hcce = _hcce_first_rgb(coord, bits=bits)
    bit_grid = _bit_grid(hcce, coord[..., 3] > 0, bits=bits)

    top = np.concatenate(
        [
            _label(rgb, "crop rgb"),
            _label(overlay, "part + kpt"),
            _label(inst_rgb, "inst gt"),
            _label(part_rgb, "part gt"),
            _label(coord_rgb, "canon xyz"),
            _label(coord_part, "coord part"),
            _label(hcce_first, "hcce bit0 xyz"),
        ],
        axis=1,
    )
    bit_grid = cv2.resize(bit_grid, (top.shape[1], 168), interpolation=cv2.INTER_NEAREST)
    bit_grid = _label(bit_grid, "hcce bits: rows x/y/z, columns bit0..bit7")
    return np.concatenate([top, bit_grid], axis=0)


def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = RARPCropHCCEDataset(
        args.dataset_root,
        args.pose_root,
        split=args.split,
        training=bool(args.training),
        crop_size=224,
        train_ratio=args.train_ratio,
        subsample=args.subsample,
        canonicalize_pose_symmetry=True,
        cse_coord_root=args.cse_coord_root,
        render_on_the_fly=True,
        coord_render_backend=args.coord_render_backend,
        cache_dir=args.dataset_cache_dir,
    )
    if bool(args.training):
        ds.set_epoch(args.epoch)

    records = []
    for out_i in range(args.num_samples):
        idx = out_i * args.stride
        _, target = ds[idx]
        panel = _make_panel(target, bits=args.hcce_bits)
        stem = f"{out_i:03d}_{target['video_name']}_{target['frame_id']}_inst{int(target['instance_id'].item())}"
        stem = stem.replace("/", "_")
        Image.fromarray(panel).save(out_dir / f"{stem}_crop_hcce_gt.jpg")
        records.append(
            {
                "file": f"{stem}_crop_hcce_gt.jpg",
                "dataset_idx": int(idx),
                "video_name": target["video_name"],
                "frame_id": target["frame_id"],
                "instance_id": int(target["instance_id"].item()),
                "bbox_min": [float(v) for v in target["bbox_min"].tolist()],
                "bbox_max": [float(v) for v in target["bbox_max"].tolist()],
                "K_crop": target["K"].numpy().tolist(),
                "part_labels": [int(v) for v in sorted(torch.unique(target["part_mask"]).tolist())],
                "coord_part_labels": [int(v) for v in sorted(torch.unique(target["coord_img"][..., 3].long()).tolist())],
                "valid_keypoints": [bool(v) for v in target["keypoints_valid"].tolist()],
                "pose_sym_flipped": bool(target["pose_sym_flipped"].item()),
            }
        )
    (out_dir / "manifest.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"wrote {len(records)} panels to {out_dir}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default=PUNCTURE_DATASET_ROOT)
    parser.add_argument("--pose_root", type=str, default=PUNCTURE_POSE_ROOT)
    parser.add_argument("--out_dir", type=str, default=str(ROBOPEPP_ROOT / "logs" / "hcce_crop_gt_vis"))
    parser.add_argument("--dataset_cache_dir", type=str, default=str(ROBOPEPP_ROOT / "logs" / "hcce_crop224_keypointnet_rarp_gpu0123_bs8" / "dataset_cache"))
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--training", type=int, default=0, choices=[0, 1])
    parser.add_argument("--epoch", type=int, default=80)
    parser.add_argument("--subsample", type=int, default=100)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=12)
    parser.add_argument("--train_ratio", type=float, default=0.95)
    parser.add_argument("--cse_coord_root", type=str, default=None)
    parser.add_argument("--coord_render_backend", type=str, default="trimesh", choices=["trimesh", "gaussian"])
    parser.add_argument("--hcce_bits", type=int, default=8)
    main(parser.parse_args())
