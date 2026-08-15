import argparse
import csv
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

from instrument_geometry import KEYPOINT_NAMES, instrument_keypoints_camera_np, project_points_np  # noqa: E402
from predict_instrument_pose import heatmap_argmax, load_model  # noqa: E402


def load_local_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


lnd_module = load_local_module("debug_lnd_instrument", ROBOPEPP_ROOT / "datasets" / "surgripe_lnd_instrument.py")
RoboPEPPSurgripeLNDInstrument = lnd_module.RoboPEPPSurgripeLNDInstrument


DEFAULT_CKPT = ROBOPEPP_ROOT / "logs/robopepp_instrument_pose_rarp_lnd_refinemem_bs56_gpu1567/checkpoints/last.pt"
DEFAULT_MEMORY = (
    ROBOPEPP_ROOT.parent
    / "gaussian-mesh-splatting/Results2/surgripe_lnd_refine_memory_train/TRAIN/refine_memory_pool.json"
)


def unnormalize_image_tensor(image_tensor):
    arr = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    arr = np.clip((arr * std + mean) * 255.0, 0, 255)
    return arr.astype(np.uint8)


def crop_to_orig(points_crop, target):
    points = np.asarray(points_crop, dtype=np.float64).copy()
    scale = target["scale"].detach().cpu().numpy().astype(np.float64)
    pad = target["pad"].detach().cpu().numpy().astype(np.float64)
    bbox_min = target["bbox_min"].detach().cpu().numpy().astype(np.float64)
    points[:, 0] = (points[:, 0] - pad[0]) / scale[0] + bbox_min[0]
    points[:, 1] = (points[:, 1] - pad[1]) / scale[1] + bbox_min[1]
    return points.astype(np.float32)


def draw_points(img, gt, pred, valid, scores=None, draw_invalid_pred=False, title="", err=None):
    out = img.copy()
    if title:
        cv2.rectangle(out, (0, 0), (out.shape[1], 26), (0, 0, 0), -1)
        cv2.putText(out, title, (7, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    for i, name in enumerate(KEYPOINT_NAMES):
        if not bool(valid[i]) and not draw_invalid_pred:
            continue
        color_gt = (30, 220, 70) if bool(valid[i]) else (120, 120, 120)
        color_pred = (255, 60, 220) if bool(valid[i]) else (255, 80, 40)
        if np.isfinite(gt[i]).all():
            x, y = np.round(gt[i]).astype(int)
            if 0 <= x < out.shape[1] and 0 <= y < out.shape[0]:
                cv2.circle(out, (x, y), 5, color_gt, -1, lineType=cv2.LINE_AA)
                cv2.putText(out, f"g{i}", (x + 5, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color_gt, 1, cv2.LINE_AA)
        if np.isfinite(pred[i]).all():
            x, y = np.round(pred[i]).astype(int)
            if 0 <= x < out.shape[1] and 0 <= y < out.shape[0]:
                cv2.drawMarker(out, (x, y), color_pred, cv2.MARKER_CROSS, 15, 2, cv2.LINE_AA)
                suffix = "" if scores is None else f":{float(scores[i]):.2f}"
                cv2.putText(out, f"h{i}{suffix}", (x + 5, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.34, color_pred, 1, cv2.LINE_AA)
        if bool(valid[i]) and np.isfinite(gt[i]).all() and np.isfinite(pred[i]).all():
            p0 = tuple(np.round(gt[i]).astype(int))
            p1 = tuple(np.round(pred[i]).astype(int))
            cv2.line(out, p0, p1, (255, 255, 255), 1, cv2.LINE_AA)
    if err is not None:
        cv2.putText(out, err, (7, out.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, err, (7, out.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def add_heat_panel(heatmaps, title):
    heat = np.clip(np.asarray(heatmaps).max(axis=0), 0.0, 1.0)
    heat = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    cv2.rectangle(heat, (0, 0), (heat.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(heat, title, (7, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return heat


def draw_orig_panel(target, gt_crop, pred_crop, valid, scores):
    rgb = target["orig_rgb"].detach().cpu().numpy().astype(np.uint8).copy()
    gt_orig = target["keypoints_orig"].detach().cpu().numpy().astype(np.float32)
    pred_orig = crop_to_orig(pred_crop, target)
    h, w = rgb.shape[:2]
    scale = 640.0 / max(h, w)
    canvas = cv2.resize(rgb, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_LINEAR)
    gt_s = gt_orig * scale
    pred_s = pred_orig * scale
    return draw_points(canvas, gt_s, pred_s, valid, scores=scores, title="original: GT valid + pred valid")


def action_from_target(target):
    return target["action"].detach().cpu().numpy().astype(np.float32)


def pose_from_target(target):
    return {
        "rot": target["wrist_quat"].detach().cpu().numpy().astype(np.float32),
        "trans": target["wrist_trans"].detach().cpu().numpy().astype(np.float32),
        "alpha": float(target["action"][0]),
        "theta_l": float(target["action"][1]),
        "theta_r": float(target["action"][2]),
    }


def choose_indices(dataset, args):
    if args.indices:
        return [int(v) for v in args.indices.split(",") if v.strip()]
    rng = random.Random(args.seed)
    n = min(int(args.num_samples), len(dataset))
    return sorted(rng.sample(range(len(dataset)), n))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CKPT))
    parser.add_argument("--output_dir", type=str, default=str(ROBOPEPP_ROOT / "logs/lnd_train_refine_keypoint_debug"))
    parser.add_argument("--lnd_root", type=str, default="/mnt/iMVR/daiyun/Dataset/LND")
    parser.add_argument("--memory_path", type=str, default=str(DEFAULT_MEMORY))
    parser.add_argument("--num_samples", type=int, default=12)
    parser.add_argument("--indices", type=str, default="")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epoch", type=int, default=83)
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--bbox_padding_frac", type=float, default=0.12)
    parser.add_argument("--bbox_jitter", type=int, default=1, choices=[0, 1])
    parser.add_argument("--heatmap_sigma", type=float, default=2.0)
    parser.add_argument("--canonical_eps", type=float, default=0.08)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, ckpt = load_model(Path(args.checkpoint), device)

    dataset = RoboPEPPSurgripeLNDInstrument(
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
        bbox_jitter=bool(args.bbox_jitter),
        color_jitter=True,
        rgb_augmentation=True,
        occlusion_augmentation=True,
        occlusion_prob=0.5,
    )
    dataset.set_epoch(args.epoch)

    out_dir = Path(args.output_dir)
    vis_dir = out_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for rank, idx in enumerate(choose_indices(dataset, args)):
        # Reset seed per sample so training augmentations are reproducible.
        random.seed(args.seed * 100000 + idx)
        np.random.seed(args.seed * 100000 + idx)
        torch.manual_seed(args.seed * 100000 + idx)
        image, target = dataset[idx]
        x = image.unsqueeze(0).to(device)
        K = target["K"].unsqueeze(0).to(device)
        with torch.inference_mode(), torch.amp.autocast(device_type="cuda", enabled=device.type == "cuda", dtype=torch.bfloat16):
            out = model(x, K, masks_enc=None, masks_pred=None)
        pred_crop_t, score_t = heatmap_argmax(out["keypoint_heatmaps"].float().detach().cpu())
        pred_crop = pred_crop_t[0].numpy().astype(np.float32)
        scores = score_t[0].numpy().astype(np.float32)

        gt_crop = target["keypoints_crop"].detach().cpu().numpy().astype(np.float32)
        gt_valid = target["keypoints_valid"].detach().cpu().numpy().astype(bool)
        gt_heat = target["heatmaps"].detach().cpu().numpy().astype(np.float32)
        gt_heat_peak = np.stack(np.unravel_index(gt_heat.reshape(len(KEYPOINT_NAMES), -1).argmax(axis=1), gt_heat.shape[1:]), axis=1)[:, ::-1].astype(np.float32)

        pose = pose_from_target(target)
        kp3d = instrument_keypoints_camera_np(pose["rot"], pose["trans"], action_from_target(target))
        uv_from_kcrop = project_points_np(kp3d, target["K"].detach().cpu().numpy()).astype(np.float32)
        proj_diff = np.linalg.norm(uv_from_kcrop - gt_crop, axis=1)
        gt_heat_diff = np.linalg.norm(gt_heat_peak - np.floor(gt_crop), axis=1)
        pred_err = np.linalg.norm(pred_crop - gt_crop, axis=1)
        valid_err = pred_err[gt_valid]

        clean_crop = target["crop_rgb"].detach().cpu().numpy().astype(np.uint8)
        input_crop = unnormalize_image_tensor(image)
        mean_err = float(valid_err.mean()) if valid_err.size else float("nan")
        err_text = f"valid={int(gt_valid.sum())}/5 mean_hm_err={mean_err:.2f}px"
        panels = [
            draw_points(input_crop, gt_crop, pred_crop, gt_valid, scores=scores, title="TRAIN input aug: GT green / pred magenta", err=err_text),
            draw_points(clean_crop, gt_crop, pred_crop, gt_valid, scores=scores, title="clean crop: valid GT/pred only", err=err_text),
            draw_points(clean_crop, gt_crop, pred_crop, gt_valid, scores=scores, draw_invalid_pred=True, title="clean crop: all pred channels; invalid red"),
            add_heat_panel(gt_heat, "GT heatmap max"),
            add_heat_panel(out["keypoint_heatmaps"][0].detach().float().cpu().numpy(), "pred heatmap max"),
            draw_orig_panel(target, gt_crop, pred_crop, gt_valid, scores),
        ]
        row_img = np.concatenate(panels[:5], axis=1)
        orig_panel = panels[5]
        if orig_panel.shape[0] != row_img.shape[0]:
            new_w = int(round(orig_panel.shape[1] * row_img.shape[0] / orig_panel.shape[0]))
            orig_panel = cv2.resize(orig_panel, (new_w, row_img.shape[0]), interpolation=cv2.INTER_LINEAR)
        canvas = np.concatenate([row_img, orig_panel], axis=1)
        stem = f"{rank:02d}_idx{idx:04d}_frame{target['frame_id']}"
        vis_path = vis_dir / f"{stem}.jpg"
        Image.fromarray(canvas).save(vis_path, quality=92)

        rows.append(
            {
                "rank": rank,
                "dataset_idx": idx,
                "frame_id": str(target["frame_id"]),
                "img_path": str(target["img_path"]),
                "vis_path": str(vis_path),
                "valid_keypoints": int(gt_valid.sum()),
                "pred_valid_mean_err_px": mean_err,
                "pred_valid_max_err_px": float(valid_err.max()) if valid_err.size else float("nan"),
                "gt_projection_max_diff_px": float(proj_diff[gt_valid].max()) if gt_valid.any() else float("nan"),
                "gt_projection_mean_diff_px": float(proj_diff[gt_valid].mean()) if gt_valid.any() else float("nan"),
                "gt_heatmap_peak_max_diff_px": float(gt_heat_diff[gt_valid].max()) if gt_valid.any() else float("nan"),
                "gt_heatmap_peak_mean_diff_px": float(gt_heat_diff[gt_valid].mean()) if gt_valid.any() else float("nan"),
                "hm_score_mean": float(scores.mean()),
                "hm_score_min": float(scores.min()),
                "valid_mask": "".join("1" if v else "0" for v in gt_valid.tolist()),
                "pred_per_kp_err_px": json.dumps([float(v) for v in pred_err.tolist()]),
                "pred_per_kp_score": json.dumps([float(v) for v in scores.tolist()]),
                "bbox_min": json.dumps([float(v) for v in target["bbox_min"].tolist()]),
                "bbox_max": json.dumps([float(v) for v in target["bbox_max"].tolist()]),
            }
        )

    csv_path = out_dir / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "memory_path": str(Path(args.memory_path).resolve()),
        "dataset": repr(dataset),
        "epoch": args.epoch,
        "num_rows": len(rows),
        "mean_pred_valid_err_px": float(np.nanmean([r["pred_valid_mean_err_px"] for r in rows])),
        "max_gt_projection_diff_px": float(np.nanmax([r["gt_projection_max_diff_px"] for r in rows])),
        "max_gt_heatmap_peak_diff_px": float(np.nanmax([r["gt_heatmap_peak_max_diff_px"] for r in rows])),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"csv={csv_path}")
    print(f"vis_dir={vis_dir}")


if __name__ == "__main__":
    main()
