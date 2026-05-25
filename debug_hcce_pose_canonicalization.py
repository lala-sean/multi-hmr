import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from datasets.RarpInstanceDataset import RARPInstanceDataset
from multi_instrument.hcce_codec import hcce_decode_torch, normalized_xyz_to_hcce


PUNCTURE_DATASET_ROOT = "/mnt/nas/share/shuojue/data/needlePuncture_videos"
PUNCTURE_POSE_ROOT = "/mnt/nas/share/shuojue/data/needlePuncture_results"
GRASPING_DATASET_ROOT = "/mnt/nas/share/shuojue/data/needleGrasping_videos"
GRASPING_POSE_ROOT = "/mnt/nas/share/shuojue/data/needleGrasping_results"
KNOTTING_DATASET_ROOT = "/mnt/nas/share/shuojue/data/knotting_videos"
KNOTTING_POSE_ROOT = "/mnt/nas/share/shuojue/data/knotting_results"


def _dataset_roots(name):
    if name == "needlePuncture":
        return PUNCTURE_DATASET_ROOT, PUNCTURE_POSE_ROOT
    if name == "needleGrasping":
        return GRASPING_DATASET_ROOT, GRASPING_POSE_ROOT
    if name == "knotting":
        return KNOTTING_DATASET_ROOT, KNOTTING_POSE_ROOT
    raise ValueError(f"Unsupported dataset_name: {name}")


def _rgb_path(dataset_root, video_name, frame_id):
    video_folder = Path(dataset_root) / f"SARRARP502022_{video_name}"
    frames = video_folder / "frames_v2"
    if not frames.is_dir():
        frames = video_folder / "frames"
    png = frames / f"{frame_id}.png"
    if png.is_file():
        return png
    jpg = frames / f"{frame_id}.jpg"
    if jpg.is_file():
        return jpg
    raise FileNotFoundError(f"RGB frame not found for {video_name}/{frame_id}")


def _panel_title(img, title):
    out = Image.new("RGB", (img.width, img.height + 28), (0, 0, 0))
    out.paste(img, (0, 28))
    draw = ImageDraw.Draw(out)
    draw.text((8, 7), title, fill=(255, 255, 255))
    return out


def _xyz_panel(coord_img, bits=8):
    valid = coord_img[..., 3] > 0
    xyz = torch.from_numpy(coord_img[..., :3]).float()
    hcce = normalized_xyz_to_hcce(xyz, iteration=bits)
    rgb = hcce[..., [0, bits, bits * 2]].numpy()
    panel = np.zeros_like(rgb, dtype=np.float32)
    panel[valid] = rgb[valid]
    return (panel * 255.0).clip(0, 255).astype(np.uint8), valid, hcce


def _concat_h(images):
    heights = [im.height for im in images]
    widths = [im.width for im in images]
    out = Image.new("RGB", (sum(widths), max(heights)), (0, 0, 0))
    x = 0
    for im in images:
        out.paste(im, (x, 0))
        x += im.width
    return out


def _binary_grid(hcce, valid, axis_name, axis_offset, bits=8):
    panels = []
    valid_np = valid.astype(bool)
    for level in range(bits):
        bit = (hcce[..., axis_offset + level].numpy() > 0.5) & valid_np
        img = np.zeros((*bit.shape, 3), dtype=np.uint8)
        img[bit] = 255
        panels.append(_panel_title(Image.fromarray(img), f"{axis_name} level {level}"))
    return _concat_h(panels)


def _decoded_maps(hcce, valid, bits=8):
    bits_tensor = (hcce > 0.5).float()[None]
    decoded = hcce_decode_torch(bits_tensor).cpu().numpy()[0] / 255.0
    decoded[~valid] = 0
    maps = []
    for axis, name in enumerate(["x", "y", "z"]):
        gray = (decoded[..., axis] * 255.0).clip(0, 255).astype(np.uint8)
        rgb = np.repeat(gray[..., None], 3, axis=2)
        maps.append(_panel_title(Image.fromarray(rgb), f"decoded {name}"))
    xyz = (decoded * 255.0).clip(0, 255).astype(np.uint8)
    maps.append(_panel_title(Image.fromarray(xyz), "decoded xyz"))
    return _concat_h(maps)


def _find_flipped_sample(ds, max_scan):
    scanned = 0
    for video_name, frame_id, insts in ds.samples:
        for instance_id, _ in insts:
            key = (video_name, frame_id, instance_id)
            pose = ds.pose_data.get(key)
            if pose is not None:
                scanned += 1
                if bool(pose.get("pose_sym_flipped", torch.tensor(False)).item()):
                    return key
                if max_scan > 0 and scanned >= max_scan:
                    break
        if max_scan > 0 and scanned >= max_scan:
            break
    raise RuntimeError(f"No flipped sample found after scanning {scanned} poses")


def main(args):
    dataset_root, pose_root = _dataset_roots(args.dataset_name)
    common = dict(
        split=args.split,
        training=False,
        img_size=args.img_size,
        dataset_root=dataset_root,
        pose_root=pose_root,
        min_dice=[args.min_dice_shaft, args.min_dice_wrist, args.min_dice_gripper],
        train_ratio=args.needle_train_ratio,
        subsample=args.subsample,
        render_on_the_fly=True,
        canonical_eps=args.canonical_eps,
    )
    ds_orig = RARPInstanceDataset(**common, canonicalize_pose_symmetry=False)
    ds_can = RARPInstanceDataset(**common, canonicalize_pose_symmetry=True)

    key = _find_flipped_sample(ds_can, args.max_scan)
    video_name, frame_id, instance_id = key
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rgb = Image.open(_rgb_path(dataset_root, video_name, frame_id)).convert("RGB")
    img_w, img_h = rgb.size
    coord_orig = ds_orig._render_cse_coord(ds_orig.pose_data[key], img_h, img_w)
    coord_can = ds_can._render_cse_coord(ds_can.pose_data[key], img_h, img_w)
    if coord_orig is None or coord_can is None:
        raise RuntimeError("CSE render returned None")

    orig_level0, orig_valid, _ = _xyz_panel(coord_orig, bits=args.hcce_bits)
    can_level0, can_valid, can_hcce = _xyz_panel(coord_can, bits=args.hcce_bits)
    diff = np.abs(can_level0.astype(np.float32) - orig_level0.astype(np.float32)).mean(axis=2)
    diff_rgb = np.zeros((*diff.shape, 3), dtype=np.uint8)
    both_valid = orig_valid | can_valid
    if both_valid.any():
        val = (diff / (np.percentile(diff[both_valid], 95) + 1e-6)).clip(0, 1)
        diff_rgb[..., 0] = (val * 255).astype(np.uint8)
        diff_rgb[..., 1] = ((1.0 - val) * 255).astype(np.uint8)
        diff_rgb[~both_valid] = 0

    stem = f"{args.dataset_name}_{video_name}_{frame_id}_inst{instance_id}_flip1"
    level0_vis = _concat_h([
        _panel_title(rgb, "rgb"),
        _panel_title(Image.fromarray(orig_level0), "original level0 xyz"),
        _panel_title(Image.fromarray(can_level0), "canonical level0 xyz"),
        _panel_title(Image.fromarray(diff_rgb), "level0 diff"),
    ])
    level0_vis.save(out_dir / f"{stem}_level0_compare.png")

    offsets = {"x": 0, "y": args.hcce_bits, "z": args.hcce_bits * 2}
    for axis_name, offset in offsets.items():
        _binary_grid(can_hcce, can_valid, axis_name, offset, bits=args.hcce_bits).save(
            out_dir / f"{stem}_{axis_name}_binary_levels.png"
        )
    _decoded_maps(can_hcce, can_valid, bits=args.hcce_bits).save(
        out_dir / f"{stem}_decoded_xyz_maps.png"
    )
    with open(out_dir / f"{stem}_selected.txt", "w", encoding="utf-8") as f:
        f.write(f"dataset={args.dataset_name}\n")
        f.write(f"split={args.split}\n")
        f.write(f"video={video_name}\n")
        f.write(f"frame={frame_id}\n")
        f.write(f"instance={instance_id}\n")
        f.write(f"canonical_eps={args.canonical_eps}\n")
    print(f"Saved debug visualizations to {out_dir}")
    print(f"Selected {key}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="needlePuncture",
                        choices=["needlePuncture", "needleGrasping", "knotting"])
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--img_size", type=int, default=630)
    parser.add_argument("--subsample", type=int, default=1)
    parser.add_argument("--max_scan", type=int, default=2000)
    parser.add_argument("--min_dice_shaft", type=float, default=0.8)
    parser.add_argument("--min_dice_wrist", type=float, default=0.6)
    parser.add_argument("--min_dice_gripper", type=float, default=0.6)
    parser.add_argument("--needle_train_ratio", type=float, default=0.95)
    parser.add_argument("--canonical_eps", type=float, default=0.08)
    parser.add_argument("--hcce_bits", type=int, default=8)
    parser.add_argument("--out_dir", type=str, default="debug_vis/hcce_pose_canonicalization")
    main(parser.parse_args())
