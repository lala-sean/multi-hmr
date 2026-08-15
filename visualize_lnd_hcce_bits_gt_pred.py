import argparse
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

from compare_crop_hcce_robopepp_rarp import load_hcce_model  # noqa: E402
from multi_instrument.hcce_codec import normalized_xyz_to_hcce  # noqa: E402


DEFAULT_CKPT = ROBOPEPP_ROOT / "logs/hcce_crop224_rarp_lnd_refinemem_bs56_gpu0234/checkpoints/last.pt"
DEFAULT_OUT = ROBOPEPP_ROOT / "logs/surgripe_lnd_hcce_bits_iter1000"
DEFAULT_LND_ROOT = "/mnt/iMVR/daiyun/Dataset/LND"


def load_local_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_lnd_module = load_local_module("robopepp_lnd_hcce_bits_vis_dataset", ROBOPEPP_ROOT / "datasets" / "lnd_hcce_crop.py")
SurgripeLNDHCCECropDataset = _lnd_module.SurgripeLNDHCCECropDataset


PART_COLORS = np.array(
    [
        [0, 0, 0],
        [230, 80, 80],
        [80, 220, 120],
        [80, 150, 240],
    ],
    dtype=np.uint8,
)
COORD_PART_COLORS = np.array(
    [
        [0, 0, 0],
        [80, 150, 240],
        [80, 220, 120],
        [245, 120, 60],
        [190, 80, 230],
    ],
    dtype=np.uint8,
)
PRED_PART_TO_GT_LABEL = {0: 2, 1: 1, 2: 3}


def label_image(img, text, font_scale=0.42):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(out, text, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def part_rgb(part):
    return PART_COLORS[np.clip(part.astype(np.int64), 0, len(PART_COLORS) - 1)]


def coord_rgb(coord):
    valid = coord[..., 3] > 0
    out = ((coord[..., :3] + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)
    out[~valid] = 0
    return out


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


def bit_tile(bit_img, mask, tile, title):
    arr = np.zeros(bit_img.shape, dtype=np.uint8)
    arr[mask] = bit_img[mask].astype(np.uint8) * 255
    rgb = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    rgb[~mask] = (18, 18, 18)
    rgb = cv2.resize(rgb, (tile, tile), interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(rgb, (0, 0), (tile, 21), (0, 0, 0), -1)
    cv2.putText(rgb, title, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)
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
    cv2.putText(rgb, title, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)
    return rgb


def mask_overlap_rgb(gt_mask, pred_mask):
    out = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
    out[gt_mask & pred_mask] = (80, 220, 120)
    out[gt_mask & ~pred_mask] = (240, 80, 80)
    out[~gt_mask & pred_mask] = (80, 150, 240)
    return out


def make_bits_panel(target, out, model_meta, args, row_name):
    rgb = target["crop_rgb"].numpy().astype(np.uint8)
    gt_part = target["part_mask"].numpy().astype(np.uint8)
    coord = target["coord_img"].numpy().astype(np.float32)
    inst = target["inst_mask"].numpy().astype(np.float32)
    pred_part = pred_part_mask(out, args.inst_thresh)
    pred_inst = pred_part > 0
    gt_part_valid = gt_part > 0

    bits = int(model_meta.get("hcce_bits", args.hcce_bits))
    coord_min = float(model_meta.get("hcce_coord_min", args.hcce_coord_min))
    coord_max = float(model_meta.get("hcce_coord_max", args.hcce_coord_max))
    gt_bits, gt_valid = gt_hcce_bits(coord, inst, bits, coord_min, coord_max)
    pred_bits = pred_hcce_bits(out)

    tile = int(args.tile)
    bit_acc = {}
    grid_rows = []
    for axis_i, axis_name in enumerate(["x", "y", "z"]):
        start = axis_i * bits
        end = start + bits
        axis_gt = gt_bits[..., start:end]
        axis_pred = pred_bits[..., start:end]
        bit_acc[axis_name] = float((axis_gt[gt_valid] == axis_pred[gt_valid]).mean()) if gt_valid.any() else float("nan")
        for row_label, mask, kind in [
            ("GT", gt_valid, "gt"),
            ("Pred@GT", gt_valid, "pred"),
            ("Diff@GT", gt_valid, "diff"),
            ("Pred@Pred", pred_inst, "pred"),
        ]:
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
    coord_part = COORD_PART_COLORS[np.clip(coord[..., 3].astype(np.int64), 0, len(COORD_PART_COLORS) - 1)]
    overview = np.concatenate(
        [
            label_image(rgb, "crop rgb"),
            label_image(part_rgb(gt_part), "GT part"),
            label_image(part_rgb(pred_part), "Pred part"),
            label_image(mask_overlap_rgb(gt_part_valid, pred_inst), "valid overlap"),
            label_image(coord_rgb(coord), "GT canon xyz"),
            label_image(coord_part, "GT coord part"),
        ],
        axis=1,
    )
    if overview.shape[1] != grid.shape[1]:
        overview = cv2.resize(overview, (grid.shape[1], overview.shape[0]), interpolation=cv2.INTER_AREA)

    header = np.zeros((42, grid.shape[1], 3), dtype=np.uint8)
    title = (
        f"{row_name} | bit acc on GT valid: "
        f"x={bit_acc['x']:.3f} y={bit_acc['y']:.3f} z={bit_acc['z']:.3f} | "
        "diff: gray same, red gt1/pred0, blue gt0/pred1"
    )
    cv2.putText(header, title, (8, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    return np.concatenate([header, overview, grid], axis=0), bit_acc


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


def parse_frame_ids(text):
    if not text:
        return None
    return [int(v) for v in text.replace(",", " ").split()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUT))
    parser.add_argument("--hcce_checkpoint", type=str, default=str(DEFAULT_CKPT))
    parser.add_argument("--lnd_root", type=str, default=DEFAULT_LND_ROOT)
    parser.add_argument("--split", type=str, default="TEST", choices=["TRAIN", "TEST"])
    parser.add_argument("--frame_ids", type=str, default="1,10,50,100,150,200,260,320,373")
    parser.add_argument("--num_samples", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--dataset_cache_dir", type=str, default=str(ROBOPEPP_ROOT / "logs/surgripe_lnd_hcce_bits_iter1000/dataset_cache"))
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--bbox_padding_frac", type=float, default=0.12)
    parser.add_argument("--tile", type=int, default=88)
    parser.add_argument("--inst_thresh", type=float, default=0.5)
    parser.add_argument("--hcce_bits", type=int, default=8)
    parser.add_argument("--hcce_coord_min", type=float, default=-1.0)
    parser.add_argument("--hcce_coord_max", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--coord_render_backend", type=str, default="trimesh", choices=["trimesh"])
    args = parser.parse_args()

    if args.device.startswith("cuda"):
        idx = int(args.device.split(":", 1)[1]) if ":" in args.device else 0
        torch.cuda.set_device(idx)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_ids = parse_frame_ids(args.frame_ids)
    dataset = SurgripeLNDHCCECropDataset(
        root=args.lnd_root,
        split=args.split,
        training=False,
        crop_size=args.crop_size,
        memory_path=None,
        use_memory_pose=False,
        bbox_padding_frac=args.bbox_padding_frac,
        frame_ids=frame_ids,
        subsample=args.stride,
        render_on_the_fly=True,
        coord_render_backend=args.coord_render_backend,
        require_cse=True,
    )
    model, model_meta = load_hcce_model(args.hcce_checkpoint, device)
    outputs = []
    records = []
    count = len(dataset) if args.num_samples <= 0 else min(len(dataset), args.num_samples)
    for idx in range(count):
        image, target = dataset[idx]
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda", dtype=torch.bfloat16):
            pred = model(image.unsqueeze(0).to(device))
        frame_id = str(target["frame_id"])
        stem = f"{idx:03d}_surgripe_lnd_{args.split}_{int(frame_id):05d}_inst0"
        panel, acc = make_bits_panel(target, pred, model_meta, args, stem)
        path = out_dir / f"{stem}_hcce_bits_gt_pred.jpg"
        Image.fromarray(panel).save(path, quality=94)
        outputs.append(path)
        rec = {
            "dataset_idx": int(idx),
            "frame_id": int(frame_id),
            "file": str(path),
            "bit_acc_x": acc["x"],
            "bit_acc_y": acc["y"],
            "bit_acc_z": acc["z"],
            "bbox_min": [float(v) for v in target["bbox_min"].tolist()],
            "bbox_max": [float(v) for v in target["bbox_max"].tolist()],
            "K_crop": target["K"].numpy().tolist(),
        }
        records.append(rec)
        print(f"[ok] {path.name} bit_acc x/y/z={acc['x']:.3f}/{acc['y']:.3f}/{acc['z']:.3f}", flush=True)
    make_contact_sheet(outputs, out_dir / "contact_sheet_hcce_bits_gt_pred.jpg")
    (out_dir / "manifest.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"[done] wrote {len(outputs)} images to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
