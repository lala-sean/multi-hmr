import argparse
import csv
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

ROBOPEPP_ROOT = Path(__file__).resolve().parent
MULTIHMR_ROOT = ROBOPEPP_ROOT.parents[1]
if str(ROBOPEPP_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBOPEPP_ROOT))
if str(MULTIHMR_ROOT) not in sys.path:
    sys.path.insert(0, str(MULTIHMR_ROOT))

import compare_crop_hcce_robopepp_rarp as cmp  # noqa: E402


DEFAULT_LND_REFINE_MEMORY = (
    MULTIHMR_ROOT
    / "submodules/gaussian-mesh-splatting/Results2/surgripe_lnd_refine_memory_train/TRAIN/refine_memory_pool.json"
)


def _load_local_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


surgripe_lnd_instrument = _load_local_module(
    "robopepp_surgripe_lnd_instrument_dataset",
    ROBOPEPP_ROOT / "datasets" / "surgripe_lnd_instrument.py",
)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = []
    seen = set()
    preferred = [
        "dataset",
        "split",
        "frame_id",
        "ordinal",
        "robopepp_pnp_status",
        "robopepp_direct_status",
        "hcce_direct_status",
        "hcce_fit_status",
        "hcce_kp_pnp_status",
    ]
    for key in preferred:
        if any(key in row for row in rows):
            seen.add(key)
            fieldnames.append(key)
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def batch_item(batch, key, i):
    value = batch[key]
    if torch.is_tensor(value):
        return value[i]
    if isinstance(value, (list, tuple)):
        return value[i]
    return value


def resized_size_from_bbox(bbox_min, bbox_max, crop_size):
    width = float(bbox_max[0] - bbox_min[0])
    height = float(bbox_max[1] - bbox_min[1])
    if width > height:
        new_w = int(crop_size)
        new_h = int(crop_size * height / max(width, 1e-6))
    else:
        new_h = int(crop_size)
        new_w = int(crop_size * width / max(height, 1e-6))
    return np.asarray([max(1, new_w), max(1, new_h)], dtype=np.int32)


def pose_from_batch_target(target, i):
    action = _to_numpy(batch_item(target, "action", i)).astype(np.float64).reshape(3)
    quat = _to_numpy(batch_item(target, "wrist_quat", i)).astype(np.float64).reshape(4)
    quat = quat / max(np.linalg.norm(quat), 1e-12)
    if quat[0] < 0.0:
        quat = -quat
    return {
        "rot": quat,
        "trans": _to_numpy(batch_item(target, "wrist_trans", i)).astype(np.float64).reshape(3),
        "alpha": float(action[0]),
        "theta_l": float(action[1]),
        "theta_r": float(action[2]),
    }


def target_like_from_batch(target, i, crop_size):
    bbox_min = _to_numpy(batch_item(target, "bbox_min", i)).astype(np.float32)
    bbox_max = _to_numpy(batch_item(target, "bbox_max", i)).astype(np.float32)
    gt_part_orig = _to_numpy(batch_item(target, "part_mask_orig", i)).astype(np.uint8)
    gt_part_crop = cmp.crop_resize_pad_map(
        gt_part_orig,
        bbox_min,
        bbox_max,
        int(crop_size),
        cv2.INTER_NEAREST,
        value=0,
    ).astype(np.uint8)
    frame_id = int(str(batch_item(target, "frame_id", i)))
    return {
        "dataset": "surgripe_lnd",
        "split": "TRAIN",
        "frame_id": frame_id,
        "image_path": str(batch_item(target, "img_path", i)),
        "crop_source": "sam3_inst_part",
        "orig_rgb": _to_numpy(batch_item(target, "orig_rgb", i)).astype(np.uint8),
        "crop_rgb": _to_numpy(batch_item(target, "crop_rgb", i)).astype(np.uint8),
        "gt_part_orig": gt_part_orig,
        "gt_part_crop": gt_part_crop,
        "K_orig": _to_numpy(batch_item(target, "K_orig", i)).astype(np.float32),
        "K_crop": _to_numpy(batch_item(target, "K", i)).astype(np.float32),
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
        "scale": _to_numpy(batch_item(target, "scale", i)).astype(np.float32),
        "pad": _to_numpy(batch_item(target, "pad", i)).astype(np.float32),
        "resized_size": resized_size_from_bbox(bbox_min, bbox_max, int(crop_size)),
        "gt_pose": pose_from_batch_target(target, i),
        "keypoints_orig": _to_numpy(batch_item(target, "keypoints_orig", i)).astype(np.float32),
        "keypoints_crop": _to_numpy(batch_item(target, "keypoints_crop", i)).astype(np.float32),
        "keypoints_valid": _to_numpy(batch_item(target, "keypoints_valid", i)).astype(bool),
        "keypoints_valid_orig": _to_numpy(batch_item(target, "keypoints_valid_orig", i)).astype(bool),
    }


def slice_output(out, i):
    return {
        key: value[i : i + 1] if torch.is_tensor(value) and value.ndim > 0 else value
        for key, value in out.items()
    }


def make_fit_args(args, hcce_meta):
    return SimpleNamespace(
        inst_thresh=float(args.inst_thresh),
        hcce_bits=int(hcce_meta.get("hcce_bits", args.hcce_bits)),
        hcce_coord_min=float(hcce_meta.get("hcce_coord_min", args.hcce_coord_min)),
        hcce_coord_max=float(hcce_meta.get("hcce_coord_max", args.hcce_coord_max)),
        hcce_bit_thresh=float(args.hcce_bit_thresh),
        max_points_per_part=int(args.max_points_per_part),
        point_select="random",
        fit_seg_source="pred",
        surface_snap_method=str(args.surface_snap_method),
        surface_k_faces=int(args.surface_k_faces),
        hcce_axis_scale=args.hcce_axis_scale,
        shaft_raw_x_min=args.shaft_raw_x_min,
        min_wrist_points=int(args.min_wrist_points),
        min_total_points=int(args.min_total_points),
        min_shaft_points=int(args.min_shaft_points),
        min_pnp_inliers=int(args.min_pnp_inliers),
        pnp_iters=int(args.pnp_iters),
        pnp_reproj_error=float(args.pnp_reproj_error),
        pnp_confidence=float(args.pnp_confidence),
        pnp_min_score=float(args.pnp_min_score),
        freeze_wrist_after_pnp=0,
        hcce_fit_mode=str(args.hcce_fit_mode),
        optim_strategy="decoupled",
        optim_parts="wrist_gripper",
        optim_loss=str(args.optim_loss),
        optim_f_scale=float(args.optim_f_scale),
        optim_max_nfev=int(args.optim_max_nfev),
        gripper_no_wrist_pose=0,
        min_depth=1e-4,
        behind_camera_penalty=1e4,
        render_seg_metrics=0,
        render_seg_metrics_limit=-1,
        render_min_depth=1e-4,
        render_draw_margin=20.0,
        overlay_alpha=0.85,
    )


def add_heatmap_rmse(row, prefix, hm, target_like):
    valid = target_like["keypoints_valid"]
    if valid is None or not np.any(valid):
        row[f"{prefix}_hm_crop_rmse_px"] = float("nan")
        return
    diff = hm[valid] - target_like["keypoints_crop"][valid]
    row[f"{prefix}_hm_crop_rmse_px"] = float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def evaluate_one(local_idx, global_idx, target_like, robo_i, hcce_i, models, cad, fit_args, rng, args):
    row = {
        "dataset": "surgripe_lnd",
        "split": str(args.split),
        "frame_id": int(target_like["frame_id"]),
        "ordinal": int(global_idx),
        "batch_local_idx": int(local_idx),
        "image_path": target_like["image_path"],
        "crop_source": target_like["crop_source"],
        "gt_source": "refine_memory" if args.use_memory_pose else "direct_lnd_gt_wrist",
    }
    poses = {}
    hm_cache = {}

    try:
        poses["robopepp_direct"] = cmp.pose_from_output(robo_i)
        row["robopepp_direct_status"] = "ok"
    except Exception as exc:
        poses["robopepp_direct"] = None
        row["robopepp_direct_status"] = f"{type(exc).__name__}: {exc}"

    try:
        poses["robopepp_pnp"], robo_hm, robo_scores, _ = cmp.pnp_from_heatmap_output(
            robo_i,
            target_like["K_crop"],
            float(args.pnp_min_score),
        )
        row["robopepp_pnp_status"] = "ok"
        row["robopepp_hm_score_mean"] = float(np.mean(robo_scores))
        row["robopepp_hm_score_min"] = float(np.min(robo_scores))
        hm_cache["robopepp_pnp"] = robo_hm
    except Exception as exc:
        poses["robopepp_pnp"] = None
        row["robopepp_pnp_status"] = f"{type(exc).__name__}: {exc}"

    try:
        poses["hcce_direct"] = cmp.pose_from_output(hcce_i)
        row["hcce_direct_status"] = "ok"
    except Exception as exc:
        poses["hcce_direct"] = None
        row["hcce_direct_status"] = f"{type(exc).__name__}: {exc}"

    try:
        poses["hcce_fit"], fit_extra = cmp.fit_pose_from_hcce(
            hcce_i,
            cad,
            models["hcce_meta"],
            target_like,
            fit_args,
            rng,
        )
        row["hcce_fit_status"] = "ok"
        row.update(fit_extra)
    except Exception as exc:
        poses["hcce_fit"] = None
        row["hcce_fit_status"] = f"{type(exc).__name__}: {exc}"
        if int(args.fail_fast):
            raise

    try:
        poses["hcce_kp_pnp"], hcce_hm, hcce_scores, _ = cmp.pnp_from_heatmap_output(
            hcce_i,
            target_like["K_crop"],
            float(args.pnp_min_score),
        )
        row["hcce_kp_pnp_status"] = "ok"
        row["hcce_kp_hm_score_mean"] = float(np.mean(hcce_scores))
        row["hcce_kp_hm_score_min"] = float(np.min(hcce_scores))
        hm_cache["hcce_kp_pnp"] = hcce_hm
    except Exception as exc:
        poses["hcce_kp_pnp"] = None
        row["hcce_kp_pnp_status"] = f"{type(exc).__name__}: {exc}"

    cmp.add_model_crop_segmentation(row, hcce_i, target_like, fit_args)
    for prefix, pose in poses.items():
        cmp.add_pose_errors(row, pose, target_like["gt_pose"], prefix)
        cmp.add_keypoint_reprojection(row, pose, target_like, prefix)
    for prefix, hm in hm_cache.items():
        add_heatmap_rmse(row, prefix, hm, target_like)
    return row


def finite_values(rows, key, scale=1.0):
    vals = []
    for row in rows:
        try:
            value = float(row.get(key, float("nan"))) * float(scale)
        except (TypeError, ValueError):
            value = float("nan")
        if math.isfinite(value):
            vals.append(value)
    return np.asarray(vals, dtype=np.float64)


def stat(rows, key, scale=1.0):
    vals = finite_values(rows, key, scale=scale)
    if vals.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
            "max": float("nan"),
            "rmse": float("nan"),
        }
    return {
        "count": int(vals.size),
        "mean": float(vals.mean()),
        "median": float(np.median(vals)),
        "p90": float(np.quantile(vals, 0.90)),
        "p95": float(np.quantile(vals, 0.95)),
        "max": float(vals.max()),
        "rmse": float(np.sqrt(np.mean(vals * vals))),
    }


def summarize(rows):
    methods = ("robopepp_direct", "robopepp_pnp", "hcce_direct", "hcce_fit", "hcce_kp_pnp")
    metric_scales = {
        "trans_err_m": 1000.0,
        "rot_err_deg": 1.0,
        "joint_mae_deg": 1.0,
        "alpha_err_deg": 1.0,
        "theta_l_err_deg": 1.0,
        "theta_r_err_deg": 1.0,
        "kp_reproj_rmse_px": 1.0,
        "hm_crop_rmse_px": 1.0,
    }
    out = {
        "num_rows": len(rows),
        "status_counts": {},
        "metrics": {},
    }
    for method in methods:
        status_key = f"{method}_status"
        counts = {}
        for row in rows:
            value = str(row.get(status_key, "missing"))
            counts[value] = counts.get(value, 0) + 1
        out["status_counts"][status_key] = counts
        for suffix, scale in metric_scales.items():
            key = f"{method}_{suffix}"
            if any(key in row for row in rows):
                summary_key = key + ("_mm" if suffix == "trans_err_m" else "")
                out["metrics"][summary_key] = stat(rows, key, scale=scale)
    for key in (
        "hcce_fit_reproj_rmse_wrist_px",
        "hcce_fit_reproj_rmse_all_px",
        "hcce_model_crop_seg_part_iou_mean",
    ):
        if any(key in row for row in rows):
            out["metrics"][key] = stat(rows, key, scale=1.0)
    return out


def fmt_stat(summary, key):
    item = summary["metrics"].get(key)
    if not item or int(item["count"]) == 0:
        return "n/a"
    return (
        f"mean={item['mean']:.4g}, median={item['median']:.4g}, "
        f"p90={item['p90']:.4g}, p95={item['p95']:.4g}, max={item['max']:.4g}, n={item['count']}"
    )


def write_summary_md(path, summary, args, csv_path):
    lines = [
        "# SurgRIPE-LND Batched Eval",
        "",
        f"- split: `{args.split}`",
        f"- rows: {summary['num_rows']}",
        f"- batch_size: {args.batch_size}",
        f"- chunk: {args.chunk_rank}/{args.num_chunks}",
        f"- gt_source: `{'refine_memory' if args.use_memory_pose else 'direct_lnd_gt_wrist'}`",
        f"- hcce_fit_mode: `{args.hcce_fit_mode}`",
        f"- csv: `{csv_path}`",
        "",
        "## Status",
        json.dumps(summary["status_counts"], indent=2, sort_keys=True),
        "",
        "## Metrics",
    ]
    for method in ("robopepp_direct", "robopepp_pnp", "hcce_direct", "hcce_fit", "hcce_kp_pnp"):
        lines.append(
            f"- {method}: trans_mm {fmt_stat(summary, method + '_trans_err_m_mm')}; "
            f"rot_deg {fmt_stat(summary, method + '_rot_err_deg')}; "
            f"joint_mae_deg {fmt_stat(summary, method + '_joint_mae_deg')}; "
            f"kp_reproj_px {fmt_stat(summary, method + '_kp_reproj_rmse_px')}"
        )
        hm_key = method + "_hm_crop_rmse_px"
        if hm_key in summary["metrics"]:
            lines.append(f"  - hm_crop_rmse_px {fmt_stat(summary, hm_key)}")
    lines.append(f"- hcce_fit_reproj_rmse_wrist_px {fmt_stat(summary, 'hcce_fit_reproj_rmse_wrist_px')}")
    lines.append(f"- hcce_model_crop_seg_part_iou_mean {fmt_stat(summary, 'hcce_model_crop_seg_part_iou_mean')}")
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lnd_root", type=str, default="/mnt/iMVR/daiyun/Dataset/LND")
    parser.add_argument("--split", type=str, default="TRAIN")
    parser.add_argument("--memory_path", type=str, default=str(DEFAULT_LND_REFINE_MEMORY))
    parser.add_argument("--use_memory_pose", type=int, choices=[0, 1], default=1)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--robopepp_checkpoint", type=str, required=True)
    parser.add_argument("--hcce_checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--chunk_rank", type=int, default=0)
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--bbox_padding_frac", type=float, default=0.12)
    parser.add_argument("--inst_thresh", type=float, default=0.5)
    parser.add_argument("--hcce_bits", type=int, default=8)
    parser.add_argument("--hcce_coord_min", type=float, default=-1.0)
    parser.add_argument("--hcce_coord_max", type=float, default=1.0)
    parser.add_argument("--hcce_bit_thresh", type=float, default=0.5)
    parser.add_argument("--max_points_per_part", type=int, default=1200)
    parser.add_argument("--surface_snap_method", choices=["surface", "vertex"], default="surface")
    parser.add_argument("--surface_k_faces", type=int, default=32)
    parser.add_argument("--hcce_axis_scale", type=str, default=None)
    parser.add_argument("--shaft_raw_x_min", type=float, default=-0.5)
    parser.add_argument("--min_wrist_points", type=int, default=12)
    parser.add_argument("--min_total_points", type=int, default=24)
    parser.add_argument("--min_shaft_points", type=int, default=24)
    parser.add_argument("--min_pnp_inliers", type=int, default=8)
    parser.add_argument("--pnp_iters", type=int, default=300)
    parser.add_argument("--pnp_reproj_error", type=float, default=8.0)
    parser.add_argument("--pnp_confidence", type=float, default=0.99)
    parser.add_argument("--pnp_min_score", type=float, default=0.0)
    parser.add_argument("--optim_loss", choices=["linear", "soft_l1", "huber", "cauchy", "arctan"], default="soft_l1")
    parser.add_argument("--optim_f_scale", type=float, default=8.0)
    parser.add_argument("--optim_max_nfev", type=int, default=200)
    parser.add_argument("--hcce_fit_mode", choices=["full", "wrist_only"], default="wrist_only")
    parser.add_argument("--print_freq", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fail_fast", type=int, choices=[0, 1], default=0)
    return parser


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = cmp.configure_device(args.device)
    dataset = surgripe_lnd_instrument.RoboPEPPSurgripeLNDInstrument(
        root=args.lnd_root,
        split=args.split,
        training=False,
        crop_size=args.crop_size,
        memory_path=args.memory_path if args.use_memory_pose else None,
        use_memory_pose=bool(args.use_memory_pose),
        canonicalize_pose_symmetry=True,
        bbox_padding_frac=float(args.bbox_padding_frac),
        bbox_jitter=False,
        bbox_shift=False,
    )
    indices = list(range(len(dataset)))
    if int(args.max_samples) > 0:
        indices = indices[: int(args.max_samples)]
    if int(args.num_chunks) > 1:
        if not (0 <= int(args.chunk_rank) < int(args.num_chunks)):
            raise ValueError(f"chunk_rank must be in [0,num_chunks), got {args.chunk_rank}/{args.num_chunks}")
        indices = indices[int(args.chunk_rank) :: int(args.num_chunks)]
    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        persistent_workers=False,
    )
    print(f"[config] output_dir={output_dir}", flush=True)
    print(f"[config] dataset={dataset}", flush=True)
    print(f"[config] selected={len(indices)} batch_size={args.batch_size} chunk={args.chunk_rank}/{args.num_chunks}", flush=True)
    print(f"[config] robopepp_checkpoint={args.robopepp_checkpoint}", flush=True)
    print(f"[config] hcce_checkpoint={args.hcce_checkpoint}", flush=True)
    print(f"[config] memory_path={args.memory_path if args.use_memory_pose else ''}", flush=True)

    robo_model, _ = cmp.load_robopepp_model(Path(args.robopepp_checkpoint), device)
    hcce_model, hcce_meta = cmp.load_hcce_model(Path(args.hcce_checkpoint), device)
    cad = cmp.InstrumentCAD(cmp.CAD_ROOT)
    models = {"robopepp": robo_model, "hcce": hcce_model, "hcce_meta": hcce_meta}
    fit_args = make_fit_args(args, hcce_meta)
    rng = np.random.default_rng(int(args.seed) + int(args.chunk_rank) * 100003)

    rows = []
    processed = 0
    for batch_idx, (images, target) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        K_crop = target["K"].to(device, non_blocking=True)
        with torch.inference_mode(), torch.amp.autocast(
            device_type="cuda",
            enabled=(device.type == "cuda"),
            dtype=torch.bfloat16,
        ):
            robo_out = robo_model(images, K_crop, masks_enc=None, masks_pred=None)
            hcce_out = hcce_model(images, K_crop)
        batch_size = int(images.shape[0])
        for i in range(batch_size):
            global_idx = indices[processed + i]
            target_like = target_like_from_batch(target, i, int(args.crop_size))
            row = evaluate_one(
                i,
                global_idx,
                target_like,
                slice_output(robo_out, i),
                slice_output(hcce_out, i),
                models,
                cad,
                fit_args,
                rng,
                args,
            )
            rows.append(row)
        processed += batch_size
        if batch_idx == 0 or (batch_idx + 1) % int(args.print_freq) == 0 or processed >= len(indices):
            print(f"[progress] {processed}/{len(indices)} rows, batch {batch_idx + 1}/{len(loader)}", flush=True)

    csv_path = output_dir / "per_frame.csv"
    write_csv(csv_path, rows)
    summary = summarize(rows)
    summary.update(
        {
            "csv": str(csv_path),
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        }
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")
    write_summary_md(output_dir / "summary.md", summary, args, csv_path)
    print(f"[done] csv={csv_path}", flush=True)
    print(f"[done] summary={output_dir / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main(build_parser().parse_args())
