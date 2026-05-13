import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from torch.utils.data import ConcatDataset, DataLoader

from datasets.RarpInstanceDataset import RARPInstanceDataset
from datasets.collate_cse import collate_fn_instrument_cse
from utils import denormalize_rgb


DATASETS = {
    "needle_puncture": (
        "/mnt/nas/share/shuojue/data/needlePuncture_videos",
        "/mnt/nas/share/shuojue/data/needlePuncture_results",
        False,
    ),
    "needle_grasping": (
        "/mnt/nas/share/shuojue/data/needleGrasping_videos",
        "/mnt/nas/share/shuojue/data/needleGrasping_results",
        False,
    ),
    "knotting": (
        "/mnt/nas/share/shuojue/data/knotting_videos",
        "/mnt/nas/share/shuojue/data/knotting_results",
        False,
    ),
    "suture_pulling": (
        "/mnt/nas/share/shuojue/data/suturePulling_videos",
        None,
        True,
    ),
}

PART_COLORS = {
    1: np.array([255, 0, 0], dtype=np.uint8),      # shaft
    2: np.array([0, 255, 0], dtype=np.uint8),      # wrist
    3: np.array([0, 0, 255], dtype=np.uint8),      # left gripper
    4: np.array([0, 0, 255], dtype=np.uint8),      # right gripper
}


def resize_nn(arr, new_h, new_w):
    h, w = arr.shape[:2]
    ri = np.floor(np.arange(new_h) * h / new_h).astype(np.int64)
    ci = np.floor(np.arange(new_w) * w / new_w).astype(np.int64)
    return arr[ri[:, None], ci[None, :]]


def coord_xyz_to_rgb(coord_img):
    """CNOS/test_cse_pseudo-style per-part xyz normalization."""
    h, w = coord_img.shape[:2]
    vis = np.zeros((h, w, 3), dtype=np.float32)
    part = coord_img[:, :, 3].astype(np.int64)
    for pid in [1, 2, 3, 4]:
        mask = part == pid
        if not mask.any():
            continue
        xyz = coord_img[mask, :3]
        center = (xyz.max(axis=0) + xyz.min(axis=0)) * 0.5
        radius = float(np.linalg.norm(xyz - center, axis=1).max()) + 1e-8
        xyz_norm = (xyz - center) / radius
        vis[mask] = np.clip(xyz_norm * 0.5 + 0.5, 0.0, 1.0)
    return (vis * 255.0).astype(np.uint8)


def coord_part_to_rgb(coord_img):
    part = coord_img[:, :, 3].astype(np.int64)
    out = np.zeros((*part.shape, 3), dtype=np.uint8)
    for pid, color in PART_COLORS.items():
        out[part == pid] = color
    return out


def mask_to_rgb(mask, color):
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    out[mask > 0] = np.asarray(color, dtype=np.uint8)
    return out


def label_panel(panel, text):
    panel = panel.copy()
    draw = ImageDraw.Draw(Image.fromarray(panel))
    # Recreate after drawing because ImageDraw mutates the PIL image object.
    pil = Image.fromarray(panel)
    draw = ImageDraw.Draw(pil)
    draw.rectangle([0, 0, min(panel.shape[1], 360), 22], fill=(0, 0, 0))
    draw.text((6, 4), text, fill=(255, 255, 255))
    return np.asarray(pil)


def make_canvas(rgb, coord_img, inst_mask, part_mask, title):
    h_img, w_img = rgb.shape[:2]
    coord_up = resize_nn(coord_img, h_img, w_img)
    inst_up = resize_nn(inst_mask, h_img, w_img)
    part_up = resize_nn(part_mask, h_img, w_img)

    coord_rgb = coord_xyz_to_rgb(coord_up)
    coord_part = coord_part_to_rgb(coord_up)
    valid = (coord_up[:, :, 3] > 0)[..., None].astype(np.float32)
    overlay = ((1.0 - 0.55 * valid) * rgb + 0.55 * valid * coord_part).astype(np.uint8)

    inst_rgb = mask_to_rgb(inst_up > 0.5, (255, 255, 255))
    train_part_rgb = np.zeros_like(rgb)
    train_part_rgb[part_up == 1] = [0, 0, 255]
    train_part_rgb[part_up == 2] = [0, 255, 0]
    train_part_rgb[part_up == 3] = [255, 0, 0]

    panels = [
        label_panel(rgb, "rgb"),
        label_panel(coord_rgb, "coord xyz"),
        label_panel(coord_part, "coord part"),
        label_panel(overlay, "overlay"),
        label_panel(inst_rgb, "train inst mask"),
        label_panel(train_part_rgb, "train part mask"),
    ]

    sep = np.full((h_img, 4, 3), 255, dtype=np.uint8)
    row1 = np.concatenate([panels[0], sep, panels[1], sep, panels[2]], axis=1)
    row2 = np.concatenate([panels[3], sep, panels[4], sep, panels[5]], axis=1)
    hsep = np.full((4, row1.shape[1], 3), 255, dtype=np.uint8)
    canvas = np.concatenate([row1, hsep, row2], axis=0)
    canvas = label_panel(canvas, title)
    return canvas


def build_dataset(args):
    names = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    datasets = []
    for name in names:
        root, pose_root, v2_force = DATASETS[name]
        cse_root = None if name == "suture_pulling" else args.cse_coord_root
        render_otf = False if name == "suture_pulling" else bool(args.render_on_the_fly)
        datasets.append(
            RARPInstanceDataset(
                split="train",
                training=False,
                img_size=args.img_size,
                dataset_root=root,
                pose_root=pose_root,
                min_dice=[args.min_dice_shaft, args.min_dice_wrist, args.min_dice_gripper],
                train_ratio=args.needle_train_ratio,
                subsample=args.subsample,
                n=args.n,
                v2_force=v2_force,
                cse_coord_root=cse_root,
                render_on_the_fly=render_otf,
            )
        )
    return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=[*DATASETS.keys(), "all"], default="needle_puncture")
    parser.add_argument("--out_dir", default="debug_vis/cse_batch")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--subsample", type=int, default=200)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--max_batches", type=int, default=10)
    parser.add_argument("--cse_coord_root", default=None)
    parser.add_argument("--render_on_the_fly", type=int, default=1, choices=[0, 1])
    parser.add_argument("--needle_train_ratio", type=float, default=0.95)
    parser.add_argument("--min_dice_shaft", type=float, default=0.8)
    parser.add_argument("--min_dice_wrist", type=float, default=0.6)
    parser.add_argument("--min_dice_gripper", type=float, default=0.6)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(args)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn_instrument_cse,
    )

    saved = 0
    inspected = 0
    for batch_idx, (x, y) in enumerate(loader):
        inspected += 1
        bs = x.shape[0]
        for bi in range(bs):
            rgb = denormalize_rgb(x[bi].numpy())
            n_inst = int(y["valid_instruments"][bi].sum().item())
            for inst_idx in range(n_inst):
                has_cse = bool(y["has_cse"][bi, inst_idx].item())
                if not has_cse:
                    print(f"[skip] batch={batch_idx} item={bi} inst={inst_idx}: has_cse=False")
                    continue
                coord_img = y["coord_imgs"][bi, inst_idx].numpy()
                inst_mask = y["inst_masks"][bi, inst_idx].numpy()
                part_mask = y["part_masks"][bi, inst_idx].numpy()
                title = (
                    f"batch {batch_idx} item {bi} inst {inst_idx} "
                    f"{y['imagename'][bi]} coord={coord_img.shape}"
                )
                canvas = make_canvas(rgb, coord_img, inst_mask, part_mask, title)
                safe_name = y["imagename"][bi].replace("/", "_")
                out_path = out_dir / f"batch{batch_idx:03d}_item{bi:02d}_inst{inst_idx:02d}_{safe_name}.png"
                Image.fromarray(canvas).save(out_path)
                saved += 1
                print(f"[ok] wrote {out_path}")
        if saved > 0 or inspected >= args.max_batches:
            break

    if saved == 0:
        print(
            "[warn] no coord_img was found in inspected batches. "
            "Try --dataset needle_puncture --render_on_the_fly 1 --num_workers 0."
        )


if __name__ == "__main__":
    main()
