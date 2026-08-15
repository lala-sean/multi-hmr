import argparse
import csv
import importlib.util
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

ROBOPEPP_ROOT = Path(__file__).resolve().parent
MULTIHMR_ROOT = ROBOPEPP_ROOT.parents[1]
if str(ROBOPEPP_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBOPEPP_ROOT))
if str(MULTIHMR_ROOT) not in sys.path:
    sys.path.insert(0, str(MULTIHMR_ROOT))

from multi_instrument.hcce_codec import normalized_xyz_to_hcce  # noqa: E402
from compare_crop_hcce_robopepp_rarp import (  # noqa: E402
    DEFAULT_HCCE_CKPT,
    NEEDLE_DATASET_ROOT,
    NEEDLE_POSE_ROOT,
    load_hcce_model,
)


def load_local_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_dataset_module = load_local_module("robopepp_rarp_hcce_crop_bits_vis", ROBOPEPP_ROOT / "datasets" / "rarp_hcce_crop.py")
RARPCropHCCEDataset = _dataset_module.RARPCropHCCEDataset


PART_COLORS = np.array(
    [
        [0, 0, 0],
        [230, 80, 80],
        [80, 220, 120],
        [80, 150, 240],
    ],
    dtype=np.uint8,
)
PRED_PART_TO_GT_LABEL = {0: 2, 1: 1, 2: 3}


def norm_frame_id(value):
    return f"{int(value):05d}"


def read_manifest(path):
    rows = []
    with Path(path).open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "video": str(row["video"]),
                    "frame_id": norm_frame_id(row["frame_id"]),
                    "instance_id": int(row["instance_id"]),
                }
            )
    return rows


def label_image(img, text, font_scale=0.42):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(out, text, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def part_rgb(part):
    return PART_COLORS[np.clip(part.astype(np.int64), 0, len(PART_COLORS) - 1)]


def pred_part_mask(out, inst_thresh):
    inst = torch.sigmoid(out["inst_mask_logits"][0].detach().float().cpu()).numpy() >= float(inst_thresh)
    dense = torch.argmax(out["part_mask_logits"][0].detach().float().cpu(), dim=0).numpy().astype(np.int64)
    part = np.zeros(dense.shape, dtype=np.uint8)
    for dense_label, gt_label in PRED_PART_TO_GT_LABEL.items():
        part[inst & (dense == dense_label)] = int(gt_label)
    return part


def gt_hcce_bits(coord, inst_mask, bits, coord_min, coord_max):
    xyz = torch.from_numpy(coord[..., :3].astype(np.float32))
    hcce = normalized_xyz_to_hcce(
        xyz,
        iteration=int(bits),
        coord_min=float(coord_min),
        coord_max=float(coord_max),
    ).numpy().astype(np.float32)
    valid = (coord[..., 3] > 0) & (inst_mask > 0.5)
    return (hcce >= 0.5).astype(np.uint8), valid


def pred_hcce_bits(out):
    pred = out["hcce_logits"][0].detach().float().cpu().numpy()
    return (pred > 0.0).astype(np.uint8).transpose(1, 2, 0)


def bit_tile(bit_img, mask, tile, title, color=(255, 255, 255)):
    arr = np.zeros(bit_img.shape, dtype=np.uint8)
    arr[mask] = (bit_img[mask].astype(np.uint8) * 255)
    rgb = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    rgb[~mask] = (18, 18, 18)
    rgb = cv2.resize(rgb, (tile, tile), interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(rgb, (0, 0), (tile, 21), (0, 0, 0), -1)
    cv2.putText(rgb, title, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
    return rgb


def diff_tile(gt, pred, mask, tile, title):
    rgb = np.zeros((*gt.shape, 3), dtype=np.uint8)
    rgb[~mask] = (18, 18, 18)
    same = mask & (gt == pred)
    miss = mask & (gt == 1) & (pred == 0)
    extra = mask & (gt == 0) & (pred == 1)
    rgb[same] = (180, 180, 180)
    rgb[miss] = (240, 70, 70)
    rgb[extra] = (70, 190, 240)
    rgb = cv2.resize(rgb, (tile, tile), interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(rgb, (0, 0), (tile, 21), (0, 0, 0), -1)
    cv2.putText(rgb, title, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    return rgb


def mask_overlap_rgb(gt_mask, pred_mask):
    out = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
    both = gt_mask & pred_mask
    gt_only = gt_mask & ~pred_mask
    pred_only = pred_mask & ~gt_mask
    out[both] = (80, 220, 120)
    out[gt_only] = (240, 80, 80)
    out[pred_only] = (80, 150, 240)
    return out


def make_bits_panel(target, out, model_meta, args, row_name):
    rgb = target["crop_rgb"].numpy().astype(np.uint8)
    gt_part = target["part_mask"].numpy().astype(np.uint8)
    coord = target["coord_img"].numpy().astype(np.float32)
    inst = target["inst_mask"].numpy().astype(np.float32)
    pred_part = pred_part_mask(out, args.inst_thresh)
    pred_inst = pred_part > 0

    bits = int(model_meta.get("hcce_bits", args.hcce_bits))
    coord_min = float(model_meta.get("hcce_coord_min", args.hcce_coord_min))
    coord_max = float(model_meta.get("hcce_coord_max", args.hcce_coord_max))
    gt_bits, gt_valid = gt_hcce_bits(coord, inst, bits, coord_min, coord_max)
    pred_bits = pred_hcce_bits(out)
    pred_valid = pred_inst
    gt_part_valid = gt_part > 0

    tile = int(args.tile)
    axes = ["x", "y", "z"]
    bit_acc = {}
    grid_rows = []
    for axis_i, axis_name in enumerate(axes):
        start = axis_i * bits
        end = start + bits
        axis_gt = gt_bits[..., start:end]
        axis_pred = pred_bits[..., start:end]
        if gt_valid.any():
            bit_acc[axis_name] = float((axis_gt[gt_valid] == axis_pred[gt_valid]).mean())
        else:
            bit_acc[axis_name] = float("nan")

        rows = [
            ("GT", gt_valid, "gt"),
            ("Pred@GT", gt_valid, "pred_gt"),
            ("Diff@GT", gt_valid, "diff"),
            ("Pred@Pred", pred_valid, "pred_pred"),
        ]
        for row_label, mask, kind in rows:
            tiles = []
            for b in range(bits):
                ch = start + b
                title = f"{axis_name}{b} {row_label}"
                if kind == "gt":
                    tiles.append(bit_tile(gt_bits[..., ch], mask, tile, title))
                elif kind == "diff":
                    tiles.append(diff_tile(gt_bits[..., ch], pred_bits[..., ch], mask, tile, title))
                else:
                    tiles.append(bit_tile(pred_bits[..., ch], mask, tile, title))
            grid_rows.append(np.concatenate(tiles, axis=1))

    grid = np.concatenate(grid_rows, axis=0)
    overview_panels = [
        label_image(rgb, "crop rgb"),
        label_image(part_rgb(gt_part), "GT part"),
        label_image(part_rgb(pred_part), "Pred part"),
        label_image(mask_overlap_rgb(gt_part_valid, pred_valid), "valid: both green, GT red, pred blue"),
    ]
    overview = np.concatenate(overview_panels, axis=1)
    if overview.shape[1] != grid.shape[1]:
        overview = cv2.resize(overview, (grid.shape[1], overview.shape[0]), interpolation=cv2.INTER_AREA)

    header_h = 42
    header = np.zeros((header_h, grid.shape[1], 3), dtype=np.uint8)
    title = (
        f"{row_name} | bit acc on GT valid: "
        f"x={bit_acc['x']:.3f} y={bit_acc['y']:.3f} z={bit_acc['z']:.3f} | "
        "diff colors: gray same, red gt1/pred0, blue gt0/pred1"
    )
    cv2.putText(header, title, (8, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    panel = np.concatenate([header, overview, grid], axis=0)
    return panel, bit_acc


def make_contact_sheet(image_paths, out_path, max_width=2400):
    rows = []
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    except Exception:
        font = ImageFont.load_default()
    for idx, path in enumerate(image_paths):
        im = Image.open(path).convert("RGB")
        if im.width > max_width:
            new_h = int(round(im.height * max_width / im.width))
            im = im.resize((max_width, new_h), Image.Resampling.LANCZOS)
        label_h = 38
        canvas = Image.new("RGB", (im.width, im.height + label_h), (255, 255, 255))
        canvas.paste(im, (0, label_h))
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 7), f"{idx:02d}  {path.name}", fill=(0, 0, 0), font=font)
        rows.append(canvas)
    if not rows:
        return
    sheet = Image.new("RGB", (max(r.width for r in rows), sum(r.height for r in rows)), (255, 255, 255))
    y = 0
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height
    sheet.save(out_path, quality=92, optimize=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--hcce_checkpoint", type=str, default=str(DEFAULT_HCCE_CKPT))
    parser.add_argument("--dataset_root", type=str, default=NEEDLE_DATASET_ROOT)
    parser.add_argument("--pose_root", type=str, default=NEEDLE_POSE_ROOT)
    parser.add_argument("--dataset_cache_dir", type=str, default=str(ROBOPEPP_ROOT / "logs/hcce_bits_gt_pred/dataset_cache"))
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--train_ratio", type=float, default=0.95)
    parser.add_argument("--subsample", type=int, default=1)
    parser.add_argument("--tile", type=int, default=112)
    parser.add_argument("--inst_thresh", type=float, default=0.5)
    parser.add_argument("--hcce_bits", type=int, default=8)
    parser.add_argument("--hcce_coord_min", type=float, default=-1.0)
    parser.add_argument("--hcce_coord_max", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_items", type=int, default=0)
    parser.add_argument("--coord_render_backend", type=str, default="trimesh", choices=["trimesh", "gaussian"])
    args = parser.parse_args()

    if args.device.startswith("cuda"):
        idx = int(args.device.split(":", 1)[1]) if ":" in args.device else 0
        os.environ["EGL_DEVICE_ID"] = str(idx)
        torch.cuda.set_device(idx)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(args.manifest_csv)
    if args.max_items > 0:
        rows = rows[: args.max_items]

    dataset = RARPCropHCCEDataset(
        args.dataset_root,
        args.pose_root,
        split=args.split,
        training=False,
        crop_size=args.crop_size,
        train_ratio=args.train_ratio,
        subsample=args.subsample,
        canonicalize_pose_symmetry=True,
        render_on_the_fly=True,
        coord_render_backend=args.coord_render_backend,
        cache_dir=args.dataset_cache_dir,
    )
    index = {(str(v), norm_frame_id(f), int(inst)): i for i, (v, f, inst, _) in enumerate(dataset.samples)}
    model, model_meta = load_hcce_model(args.hcce_checkpoint, device)

    outputs = []
    records = []
    for ordinal, row in enumerate(rows):
        key = (row["video"], row["frame_id"], row["instance_id"])
        idx = index.get(key)
        if idx is None:
            print(f"[missing] {key}")
            records.append({"ordinal": ordinal, **row, "status": "missing"})
            continue
        image, target = dataset[idx]
        with torch.no_grad():
            pred = model(image.unsqueeze(0).to(device))
        stem = f"{ordinal:03d}_{key[0]}_{key[1]}_inst{key[2]}"
        panel, acc = make_bits_panel(target, pred, model_meta, args, stem)
        path = out_dir / f"{stem}_hcce_bits_gt_pred.jpg"
        Image.fromarray(panel).save(path, quality=94)
        outputs.append(path)
        rec = {
            "ordinal": ordinal,
            **row,
            "dataset_idx": int(idx),
            "file": str(path),
            "bit_acc_x": acc["x"],
            "bit_acc_y": acc["y"],
            "bit_acc_z": acc["z"],
            "bbox_min": [float(v) for v in target["bbox_min"].tolist()],
            "bbox_max": [float(v) for v in target["bbox_max"].tolist()],
            "status": "ok",
        }
        records.append(rec)
        print(f"[ok] {ordinal + 1}/{len(rows)} {path.name} acc x/y/z={acc['x']:.3f}/{acc['y']:.3f}/{acc['z']:.3f}")

    make_contact_sheet(outputs, out_dir / "contact_sheet_hcce_bits_gt_pred.jpg")
    (out_dir / "manifest.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"[done] wrote {len(outputs)} images to {out_dir}")


if __name__ == "__main__":
    main()
