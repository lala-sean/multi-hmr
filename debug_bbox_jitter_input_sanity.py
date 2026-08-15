import argparse
import csv
import hashlib
import importlib.util
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROBOPEPP_ROOT = Path(__file__).resolve().parent
if str(ROBOPEPP_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBOPEPP_ROOT))

from instrument_geometry import KEYPOINT_NAMES  # noqa: E402
from predict_instrument_pose import DATASETS, heatmap_argmax, load_model  # noqa: E402


DEFAULT_CKPT = ROBOPEPP_ROOT / "logs/robopepp_instrument_pose_rarp_lnd_refinemem_bs56_gpu1567/checkpoints/last.pt"
DEFAULT_MEMORY = (
    ROBOPEPP_ROOT.parent
    / "gaussian-mesh-splatting/Results2/surgripe_lnd_refine_memory_train/TRAIN/refine_memory_pool.json"
)


def load_local_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def unnormalize_image_tensor(image_tensor):
    arr = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    arr = np.clip((arr * std + mean) * 255.0, 0, 255)
    return arr.astype(np.uint8)


def image_hash(arr):
    return hashlib.sha1(np.ascontiguousarray(arr).tobytes()).hexdigest()[:10]


def add_title(img, lines):
    out = img.copy()
    if isinstance(lines, str):
        lines = [lines]
    h = 18 + 16 * max(1, len(lines))
    cv2.rectangle(out, (0, 0), (out.shape[1], h), (0, 0, 0), -1)
    for i, line in enumerate(lines):
        cv2.putText(out, str(line), (6, 16 + 16 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def draw_keypoints(img, gt, pred, valid, scores=None):
    out = img.copy()
    for i, name in enumerate(KEYPOINT_NAMES):
        if not bool(valid[i]):
            continue
        gx, gy = np.round(gt[i]).astype(int)
        px, py = np.round(pred[i]).astype(int)
        if 0 <= gx < out.shape[1] and 0 <= gy < out.shape[0]:
            cv2.circle(out, (gx, gy), 4, (40, 220, 70), -1, lineType=cv2.LINE_AA)
            cv2.putText(out, f"g{i}", (gx + 4, gy - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (40, 220, 70), 1, cv2.LINE_AA)
        if 0 <= px < out.shape[1] and 0 <= py < out.shape[0]:
            cv2.drawMarker(out, (px, py), (255, 60, 220), cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)
            label = f"h{i}" if scores is None else f"h{i}:{float(scores[i]):.2f}"
            cv2.putText(out, label, (px + 4, py + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 60, 220), 1, cv2.LINE_AA)
        if 0 <= gx < out.shape[1] and 0 <= gy < out.shape[0] and 0 <= px < out.shape[1] and 0 <= py < out.shape[0]:
            cv2.line(out, (gx, gy), (px, py), (255, 255, 255), 1, cv2.LINE_AA)
    return out


def heat_panel(heatmaps):
    heat = np.clip(np.asarray(heatmaps).max(axis=0), 0.0, 1.0)
    heat = cv2.applyColorMap((heat * 255.0).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)


def crop_points_to_orig(points_crop, target):
    points = np.asarray(points_crop, dtype=np.float64).copy()
    scale = target["scale"].detach().cpu().numpy().astype(np.float64)
    pad = target["pad"].detach().cpu().numpy().astype(np.float64)
    bbox_min = target["bbox_min"].detach().cpu().numpy().astype(np.float64)
    points[:, 0] = (points[:, 0] - pad[0]) / scale[0] + bbox_min[0]
    points[:, 1] = (points[:, 1] - pad[1]) / scale[1] + bbox_min[1]
    return points.astype(np.float32)


def orig_panel(target, pred_crop, valid):
    rgb = target["orig_rgb"].detach().cpu().numpy().astype(np.uint8).copy()
    bbox_min = target["bbox_min"].detach().cpu().numpy()
    bbox_max = target["bbox_max"].detach().cpu().numpy()
    cv2.rectangle(rgb, tuple(np.round(bbox_min).astype(int)), tuple(np.round(bbox_max).astype(int)), (255, 255, 255), 2)
    gt_orig = target["keypoints_orig"].detach().cpu().numpy().astype(np.float32)
    pred_orig = crop_points_to_orig(pred_crop, target)
    rgb = draw_keypoints(rgb, gt_orig, pred_orig, valid)
    h, w = rgb.shape[:2]
    scale = 224.0 / float(max(h, w))
    resized = cv2.resize(rgb, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((224, 224, 3), dtype=np.uint8)
    y0 = (224 - resized.shape[0]) // 2
    x0 = (224 - resized.shape[1]) // 2
    canvas[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
    return canvas


def build_dataset(args):
    if args.dataset == "lnd":
        mod = load_local_module("debug_lnd_instrument", ROBOPEPP_ROOT / "datasets" / "surgripe_lnd_instrument.py")
        return mod.RoboPEPPSurgripeLNDInstrument(
            root=args.lnd_root,
            split="TRAIN",
            training=True,
            crop_size=args.crop_size,
            memory_path=args.memory_path,
            use_memory_pose=True,
            canonicalize_pose_symmetry=True,
            canonical_eps=args.canonical_eps,
            heatmap_sigma=args.heatmap_sigma,
            bbox_padding_frac=args.bbox_padding_frac,
            bbox_jitter=True,
            color_jitter=False,
            rgb_augmentation=False,
            occlusion_augmentation=False,
            occlusion_prob=0.0,
        )
    mod = load_local_module("debug_rarp_instrument", ROBOPEPP_ROOT / "datasets" / "rarp_instrument.py")
    cfg = DATASETS[args.rarp_dataset_name]
    return mod.RoboPEPPRARPInstrument(
        cfg["dataset_root"],
        cfg["pose_root"],
        split=args.rarp_split,
        training=True,
        crop_size=args.crop_size,
        train_ratio=args.train_ratio,
        subsample=1,
        min_dice=(args.min_dice_shaft, args.min_dice_wrist, args.min_dice_gripper),
        canonicalize_pose_symmetry=True,
        canonical_eps=args.canonical_eps,
        heatmap_sigma=args.heatmap_sigma,
        bbox_padding_frac=args.bbox_padding_frac,
        color_jitter=False,
        rgb_augmentation=False,
        occlusion_augmentation=False,
        occlusion_prob=0.0,
        cache_dir=args.dataset_cache_dir,
    )


def parse_ints(text):
    return [int(v) for v in str(text).split(",") if v.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["lnd", "rarp"], default="lnd")
    parser.add_argument("--rarp_dataset_name", default="needleGrasping", choices=sorted(DATASETS))
    parser.add_argument("--rarp_split", default="train", choices=["train", "test"])
    parser.add_argument("--indices", default="777")
    parser.add_argument("--epochs", default="0,30,50,70", help="0/30/50/70 correspond to jitter 0/30/50/80 px")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    parser.add_argument("--output_dir", default=str(ROBOPEPP_ROOT / "logs/lnd_train_refine_keypoint_debug_after_fix"))
    parser.add_argument("--lnd_root", default="/mnt/iMVR/daiyun/Dataset/LND")
    parser.add_argument("--memory_path", default=str(DEFAULT_MEMORY))
    parser.add_argument("--dataset_cache_dir", default=str(ROBOPEPP_ROOT / "logs/dataset_cache"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=123456)
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--bbox_padding_frac", type=float, default=0.12)
    parser.add_argument("--train_ratio", type=float, default=0.95)
    parser.add_argument("--min_dice_shaft", type=float, default=0.8)
    parser.add_argument("--min_dice_wrist", type=float, default=0.6)
    parser.add_argument("--min_dice_gripper", type=float, default=0.6)
    parser.add_argument("--heatmap_sigma", type=float, default=2.0)
    parser.add_argument("--canonical_eps", type=float, default=0.08)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, _ = load_model(Path(args.checkpoint), device)
    dataset = build_dataset(args)

    rows = []
    sheet_rows = []
    indices = parse_ints(args.indices)
    epochs = parse_ints(args.epochs)
    for idx in indices:
        panels = []
        base_image_np = None
        base_crop_np = None
        base_pred = None
        base_gt = None
        for epoch in epochs:
            dataset.set_epoch(epoch)
            seed = int(args.seed) + int(idx)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            image, target = dataset[idx]
            x = image.unsqueeze(0).to(device)
            K = target["K"].unsqueeze(0).to(device)
            with torch.inference_mode(), torch.amp.autocast(device_type="cuda", enabled=device.type == "cuda", dtype=torch.bfloat16):
                out = model(x, K, masks_enc=None, masks_pred=None)
            pred_t, score_t = heatmap_argmax(out["keypoint_heatmaps"].float().detach().cpu())
            pred = pred_t[0].numpy().astype(np.float32)
            scores = score_t[0].numpy().astype(np.float32)
            gt = target["keypoints_crop"].detach().cpu().numpy().astype(np.float32)
            valid = target["keypoints_valid"].detach().cpu().numpy().astype(bool)
            input_rgb = unnormalize_image_tensor(image)
            crop_rgb = target["crop_rgb"].detach().cpu().numpy().astype(np.uint8)

            if base_image_np is None:
                base_image_np = image.detach().cpu().numpy().copy()
                base_crop_np = crop_rgb.copy()
                base_pred = pred.copy()
                base_gt = gt.copy()
            input_max_abs = float(np.max(np.abs(image.detach().cpu().numpy() - base_image_np)))
            crop_mean_abs = float(np.mean(np.abs(crop_rgb.astype(np.float32) - base_crop_np.astype(np.float32))))
            pred_mean_shift = float(np.linalg.norm(pred - base_pred, axis=1).mean())
            gt_mean_shift = float(np.linalg.norm(gt - base_gt, axis=1).mean())
            err = np.linalg.norm(pred - gt, axis=1)
            valid_err = err[valid]

            bbox_min = target["bbox_min"].detach().cpu().numpy()
            bbox_max = target["bbox_max"].detach().cpu().numpy()
            title = [
                f"epoch {epoch} input hash {image_hash(input_rgb)}",
                f"input dmax {input_max_abs:.3f} crop dmae {crop_mean_abs:.1f}",
                f"pred shift {pred_mean_shift:.1f} gt shift {gt_mean_shift:.1f} err {float(valid_err.mean()) if valid_err.size else float('nan'):.1f}",
            ]
            panels.append(add_title(draw_keypoints(input_rgb, gt, pred, valid, scores), title))
            panels.append(add_title(heat_panel(out["keypoint_heatmaps"][0].detach().float().cpu().numpy()), "pred heatmap"))
            panels.append(add_title(orig_panel(target, pred, valid), f"orig bbox [{bbox_min[0]:.1f},{bbox_min[1]:.1f}]-[{bbox_max[0]:.1f},{bbox_max[1]:.1f}]"))

            rows.append(
                {
                    "dataset": args.dataset,
                    "rarp_dataset_name": args.rarp_dataset_name if args.dataset == "rarp" else "",
                    "dataset_idx": idx,
                    "epoch": epoch,
                    "input_hash": image_hash(input_rgb),
                    "input_max_abs_vs_epoch0": input_max_abs,
                    "crop_mean_abs_vs_epoch0": crop_mean_abs,
                    "pred_mean_shift_vs_epoch0_px": pred_mean_shift,
                    "gt_mean_shift_vs_epoch0_px": gt_mean_shift,
                    "valid_count": int(valid.sum()),
                    "valid_mean_err_px": float(valid_err.mean()) if valid_err.size else float("nan"),
                    "bbox_min": json.dumps([float(v) for v in bbox_min.tolist()]),
                    "bbox_max": json.dumps([float(v) for v in bbox_max.tolist()]),
                    "gt_keypoints_crop": json.dumps(gt.round(3).tolist()),
                    "pred_keypoints_crop": json.dumps(pred.round(3).tolist()),
                    "scores": json.dumps(scores.round(5).tolist()),
                }
            )
        sheet_rows.append(np.concatenate(panels, axis=1))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.dataset}"
    if args.dataset == "rarp":
        stem += f"_{args.rarp_dataset_name}"
    stem += "_bbox_jitter_input_sanity"
    sheet = np.concatenate(sheet_rows, axis=0)
    sheet_path = out_dir / f"{stem}_sheet.jpg"
    csv_path = out_dir / f"{stem}_metrics.csv"
    Image.fromarray(sheet).save(sheet_path, quality=92)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"sheet={sheet_path}")
    print(f"csv={csv_path}")


if __name__ == "__main__":
    main()
