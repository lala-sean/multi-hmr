import argparse
import csv
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

ROBOPEPP_ROOT = Path(__file__).resolve().parent
MULTIHMR_ROOT = ROBOPEPP_ROOT.parents[1]
if str(ROBOPEPP_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBOPEPP_ROOT))
if str(MULTIHMR_ROOT) not in sys.path:
    sys.path.insert(0, str(MULTIHMR_ROOT))

from datasets.rarp_pose_canonicalization import canonicalize_pose_symmetry, pose_to_numpy  # noqa: E402
from instrument_geometry import fk_matrices_np, quat_wxyz_to_matrix_np  # noqa: E402
from models.surfemb_keypoint_crop_model import SurfEmbKeypointCropDPT  # noqa: E402
from models.surfemb_resnet_crop_model import SurfEmbResNetCropModel  # noqa: E402
from surfemb_articulated_pose import (  # noqa: E402
    PART_NAMES,
    articulated_add_mm,
    encode_surface_keys,
    estimate_articulated_pose,
    load_part_surfaces,
    part_pose_errors,
)


DEFAULT_SURFACE_ROOT = (
    ROBOPEPP_ROOT
    / "assets"
    / "instrument_surface_samples_surfemb_x2.13mm_wg1over3_shafttop30mm"
)
DEFAULT_MODELS = [
    "resnet="
    + str(
        ROBOPEPP_ROOT
        / "logs/surfemb_resnet_crop224_rarp_lnd_refinemem_b56_gpu4567/checkpoints/best_val_total.pt"
    ),
    "dino_multihead="
    + str(
        ROBOPEPP_ROOT
        / "logs/surfemb_keypoint_crop_part206020_fbo4_p2048_b32_gpu0123/checkpoints/best_val_total.pt"
    ),
]


class Pose:
    """Compatibility shim for legacy RARP memory_pool.pth pickles."""

    def __setstate__(self, state):
        self.__dict__.update(state if isinstance(state, dict) else {})


def load_local_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_rarp_module = load_local_module("surfemb_eval_rarp_dataset", ROBOPEPP_ROOT / "datasets/rarp_instrument.py")
_surf_aug = load_local_module("surfemb_eval_augment", ROBOPEPP_ROOT / "datasets/surfemb_augment.py")
RoboPEPPRARPInstrument = _rarp_module.RoboPEPPRARPInstrument


def parse_model_specs(values):
    specs = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"--model must be NAME=CHECKPOINT, got {value!r}")
        name, path = value.split("=", 1)
        path = Path(path).resolve()
        if not name or not path.is_file():
            raise FileNotFoundError(f"Invalid model spec {value!r}")
        kind = "resnet" if "resnet" in name.lower() else "dino"
        specs.append({"name": name, "path": str(path), "kind": kind})
    return specs


def build_dataset(args):
    return RoboPEPPRARPInstrument(
        args.needle_dataset_root,
        args.needle_pose_root,
        split="test",
        training=False,
        crop_size=int(args.crop_size),
        train_ratio=float(args.train_ratio),
        subsample=1,
        min_dice=(args.min_dice_shaft, args.min_dice_wrist, args.min_dice_gripper),
        canonicalize_pose_symmetry=bool(args.canonicalize_pose_symmetry),
        canonical_eps=float(args.canonical_eps),
        bbox_padding_frac=float(args.bbox_padding_frac),
        cache_dir=args.dataset_cache_dir,
    )


def target_pose(target):
    action = target["action"].detach().cpu().numpy().astype(np.float64)
    quat = target["wrist_quat"].detach().cpu().numpy().astype(np.float64)
    quat /= max(np.linalg.norm(quat), 1e-12)
    return {
        "rot": quat,
        "trans": target["wrist_trans"].detach().cpu().numpy().astype(np.float64),
        "alpha": float(action[0]),
        "theta_l": float(action[1]),
        "theta_r": float(action[2]),
    }


def canonicalize_prediction(pose, eps):
    canonical = canonicalize_pose_symmetry(
        {
            "rot": pose["rot"],
            "trans": pose["trans"],
            "alpha": [pose["alpha"]],
            "theta_l": [pose["theta_l"]],
            "theta_r": [pose["theta_r"]],
        },
        eps=float(eps),
        enabled=True,
    )
    canonical = pose_to_numpy(canonical)
    return {
        "rot": np.asarray(canonical["rot"], dtype=np.float64).reshape(4),
        "trans": np.asarray(canonical["trans"], dtype=np.float64).reshape(3),
        "alpha": float(np.asarray(canonical["alpha"]).reshape(-1)[0]),
        "theta_l": float(np.asarray(canonical["theta_l"]).reshape(-1)[0]),
        "theta_r": float(np.asarray(canonical["theta_r"]).reshape(-1)[0]),
        "pose_sym_flipped": bool(np.asarray(canonical["pose_sym_flipped"]).item()),
        "chain_cost": float(pose.get("chain_cost", float("nan"))),
        "chain_nfev": int(pose.get("chain_nfev", 0)),
    }


def make_surfemb_crop(target, args):
    rgb = target["orig_rgb"].detach().cpu().numpy().astype(np.uint8)
    mask = target["inst_mask_orig"].detach().cpu().numpy().astype(bool)
    K_orig = target["K_orig"].detach().cpu().numpy().astype(np.float32)
    M = _surf_aug.random_rotated_mask_crop_matrix(
        mask,
        int(args.crop_size),
        crop_scale=float(args.surfemb_crop_scale),
        max_angle=0.0,
        offset_scale=0.0,
        ensure_full_mask=True,
    )
    crop = cv2.warpAffine(
        rgb,
        M,
        (int(args.crop_size), int(args.crop_size)),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    K_crop = _surf_aug.matrix3_from_affine(M) @ K_orig
    return _surf_aug.imagenet_tensor(crop), K_crop.astype(np.float32), crop, M


def load_model(spec, device):
    checkpoint = torch.load(spec["path"], map_location="cpu", weights_only=False)
    saved_args = checkpoint.get("args", {})
    if not isinstance(saved_args, dict):
        saved_args = vars(saved_args)
    common = {
        "img_size": int(saved_args.get("crop_size", 224)),
        "surfemb_emb_dim": int(saved_args.get("surfemb_emb_dim", 12)),
        "surfemb_mlp_hidden_features": int(saved_args.get("surfemb_mlp_hidden_features", 256)),
        "surfemb_mlp_hidden_layers": int(saved_args.get("surfemb_mlp_hidden_layers", 2)),
    }
    if spec["kind"] == "resnet":
        model = SurfEmbResNetCropModel(**common)
    else:
        model = SurfEmbKeypointCropDPT(
            **common,
            backbone=saved_args.get("backbone", "dinov2_vits14"),
            pretrained_backbone=False,
            dense_feat_dim=int(saved_args.get("dense_feat_dim", 256)),
            keypoint_feat_size=int(saved_args.get("keypoint_feat_size", 14)),
            pose_head_iter=int(saved_args.get("pose_head_iter", 4)),
            pose_head_dropout=float(saved_args.get("pose_head_dropout", 0.3)),
        )
    state = checkpoint["model_state_dict"]
    load_result = model.load_state_dict(state, strict=True)
    model = model.eval().to(device)
    return model, int(checkpoint.get("iter", -1)), str(load_result)


def rotation_error_deg(q_pred, q_gt):
    rel = quat_wxyz_to_matrix_np(q_pred) @ quat_wxyz_to_matrix_np(q_gt).T
    cosine = np.clip((float(np.trace(rel)) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def angle_error_deg(pred, gt):
    delta = math.atan2(math.sin(float(pred) - float(gt)), math.cos(float(pred) - float(gt)))
    return abs(math.degrees(delta))


def pose_metrics(pred, gt, surfaces):
    row = {
        "wrist_trans_err_mm": float(np.linalg.norm(np.asarray(pred["trans"]) - np.asarray(gt["trans"])) * 1000.0),
        "wrist_rot_err_deg": rotation_error_deg(pred["rot"], gt["rot"]),
        "alpha_err_deg": angle_error_deg(pred["alpha"], gt["alpha"]),
        "theta_l_err_deg": angle_error_deg(pred["theta_l"], gt["theta_l"]),
        "theta_r_err_deg": angle_error_deg(pred["theta_r"], gt["theta_r"]),
    }
    row["joint_mae_deg"] = float(np.mean([row["alpha_err_deg"], row["theta_l_err_deg"], row["theta_r_err_deg"]]))
    row.update(part_pose_errors(pred, gt))
    row.update(articulated_add_mm(pred, gt, surfaces))
    return row


@torch.inference_mode()
def estimate_from_output(model, surfaces, out, K_crop, args, seed):
    pose, diagnostics, _ = estimate_articulated_pose(
        out["inst_mask_logits"].float(),
        out["surfemb_queries"].float(),
        model,
        surfaces,
        K_crop,
        max_poses=int(args.max_poses),
        max_pose_evaluations=int(args.max_pose_evaluations),
        pose_batch_size=int(args.pose_batch_size),
        top_k=int(args.top_k),
        down_sample_scale=int(args.down_sample_scale),
        seed=int(seed),
        require_all_parts=bool(args.require_all_parts),
        pose_estimator=args.pose_estimator,
        topk_max_correspondences=int(args.topk_max_correspondences),
        topk_min_correspondences=int(args.topk_min_correspondences),
        topk_min_part_probability=float(args.topk_min_part_probability),
        topk_min_object_probability=float(args.topk_min_object_probability),
        topk_margin_power=float(args.topk_margin_power),
        topk_ransac_iterations=int(args.topk_ransac_iterations),
        topk_ransac_reprojection_error=float(args.topk_ransac_reprojection_error),
        topk_ransac_confidence=float(args.topk_ransac_confidence),
        topk_min_inliers=int(args.topk_min_inliers),
        topk_min_inlier_fraction=float(args.topk_min_inlier_fraction),
    )
    return canonicalize_prediction(pose, args.canonical_eps), diagnostics


@torch.inference_mode()
def predict_one(model, surfaces, image, K_crop, device, args, seed):
    x = image[None].to(device=device, non_blocking=True)
    K_tensor = torch.from_numpy(K_crop)[None].to(device=device, non_blocking=True)
    with torch.amp.autocast(device_type="cuda", enabled=device.type == "cuda" and bool(args.amp), dtype=torch.bfloat16):
        out = model(x, K_tensor)
    return estimate_from_output(
        model,
        surfaces,
        {"inst_mask_logits": out["inst_mask_logits"][0], "surfemb_queries": out["surfemb_queries"][0]},
        K_crop,
        args,
        seed,
    )


def stat(values):
    values = np.asarray([value for value in values if math.isfinite(float(value))], dtype=np.float64)
    if len(values) == 0:
        return {"count": 0, "mean": float("nan"), "median": float("nan"), "std": float("nan"), "rmse": float("nan")}
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std()),
        "rmse": float(np.sqrt(np.mean(values * values))),
    }


def summarize(rows, model_specs):
    metrics = [
        "wrist_trans_err_mm",
        "wrist_rot_err_deg",
        "joint_mae_deg",
        "alpha_err_deg",
        "theta_l_err_deg",
        "theta_r_err_deg",
        "articulated_add_mm",
    ]
    for part in PART_NAMES:
        metrics.extend([f"{part}_trans_err_mm", f"{part}_rot_err_deg", f"{part}_add_mm"])
    summary = {}
    for spec in model_specs:
        selected = [row for row in rows if row["model"] == spec["name"]]
        ok = [row for row in selected if row.get("status") == "ok"]
        summary[spec["name"]] = {
            "checkpoint": spec["path"],
            "checkpoint_iter": next((row.get("checkpoint_iter") for row in selected), None),
            "total": len(selected),
            "success": len(ok),
            "success_rate": float(len(ok) / max(1, len(selected))),
            "status_counts": {},
            "metrics": {metric: stat([row.get(metric, float("nan")) for row in ok]) for metric in metrics},
        }
        for row in selected:
            status = str(row.get("status", "missing"))
            summary[spec["name"]]["status_counts"][status] = summary[spec["name"]]["status_counts"].get(status, 0) + 1
    return summary


def summarize_paired(rows, model_specs):
    metrics = ["wrist_trans_err_mm", "wrist_rot_err_deg", "joint_mae_deg", "articulated_add_mm"]
    if len(model_specs) != 2:
        return None
    name_a, name_b = (spec["name"] for spec in model_specs)
    by_model_index = {(row["model"], int(row["dataset_idx"])): row for row in rows}
    paired = []
    dataset_indices = sorted({int(row["dataset_idx"]) for row in rows})
    for dataset_index in dataset_indices:
        row_a = by_model_index.get((name_a, dataset_index))
        row_b = by_model_index.get((name_b, dataset_index))
        if row_a is not None and row_b is not None and row_a.get("status") == row_b.get("status") == "ok":
            paired.append((row_a, row_b))

    result = {"model_a": name_a, "model_b": name_b, "paired_success": len(paired), "metrics": {}}
    for metric in metrics:
        values_a = np.asarray([float(row_a[metric]) for row_a, _ in paired], dtype=np.float64)
        values_b = np.asarray([float(row_b[metric]) for _, row_b in paired], dtype=np.float64)
        result["metrics"][metric] = {
            name_a: stat(values_a),
            name_b: stat(values_b),
            f"{name_a}_wins": int(np.sum(values_a < values_b)),
            "ties": int(np.sum(values_a == values_b)),
            f"{name_b}_wins": int(np.sum(values_b < values_a)),
            f"{name_a}_win_rate": float(np.mean(values_a < values_b)) if len(paired) else float("nan"),
            f"{name_b}_win_rate": float(np.mean(values_b < values_a)) if len(paired) else float("nan"),
        }
    return result


def write_csv(path, rows):
    keys = []
    seen = set()
    preferred = ["model", "dataset_idx", "video", "frame_id", "instance_id", "status", "checkpoint_iter"]
    for key in preferred:
        if any(key in row for row in rows):
            keys.append(key)
            seen.add(key)
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def worker_main(rank, chunks, model_specs, args):
    indices = chunks[rank]
    cv2.setNumThreads(0)
    torch.set_num_threads(max(1, int(args.cpu_threads_per_worker)))
    device = torch.device(args.devices[rank])
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dataset = build_dataset(args)
    rows = []
    batch_size = max(1, int(args.inference_batch_size))
    for model_index, spec in enumerate(model_specs):
        model, checkpoint_iter, load_result = load_model(spec, device)
        surfaces = load_part_surfaces(args.surface_root, keys_per_part=args.surface_keys_per_part, seed=args.surface_seed)
        encode_surface_keys(model, surfaces, device, mask_keys_per_part=args.mask_keys_per_part)
        print(
            f"[worker {rank}] loaded {spec['name']} iter={checkpoint_iter} "
            f"inference_batch={batch_size}: {load_result}",
            flush=True,
        )

        for batch_start in range(0, len(indices), batch_size):
            batch_indices = indices[batch_start : batch_start + batch_size]
            samples = []
            for dataset_index in batch_indices:
                _, target = dataset[int(dataset_index)]
                image, K_crop, _, _ = make_surfemb_crop(target, args)
                samples.append(
                    {
                        "dataset_idx": int(dataset_index),
                        "image": image,
                        "K_crop": K_crop,
                        "gt": target_pose(target),
                        "video": str(target["video_name"]),
                        "frame_id": str(target["frame_id"]),
                        "instance_id": int(target["instance_id"].item()),
                    }
                )

            x = torch.stack([sample["image"] for sample in samples]).to(device=device, non_blocking=True)
            K_tensor = torch.from_numpy(np.stack([sample["K_crop"] for sample in samples])).to(
                device=device, non_blocking=True
            )
            forward_start = time.perf_counter()
            with torch.inference_mode(), torch.amp.autocast(
                device_type="cuda",
                enabled=device.type == "cuda" and bool(args.amp),
                dtype=torch.bfloat16,
            ):
                outputs = model(x, K_tensor)
            forward_sec_per_sample = float(time.perf_counter() - forward_start) / max(1, len(samples))

            for sample_index, sample in enumerate(samples):
                row = {
                    "model": spec["name"],
                    "checkpoint_iter": checkpoint_iter,
                    "dataset_idx": sample["dataset_idx"],
                    "video": sample["video"],
                    "frame_id": sample["frame_id"],
                    "instance_id": sample["instance_id"],
                    "forward_sec": forward_sec_per_sample,
                }
                solve_start = time.perf_counter()
                try:
                    pred, diagnostics = estimate_from_output(
                        model,
                        surfaces,
                        {
                            "inst_mask_logits": outputs["inst_mask_logits"][sample_index],
                            "surfemb_queries": outputs["surfemb_queries"][sample_index],
                        },
                        sample["K_crop"],
                        args,
                        seed=(
                            int(args.pose_seed)
                            + sample["dataset_idx"] * 1009
                            + model_index * 10000019
                        ),
                    )
                    row["status"] = "ok"
                    row.update(pose_metrics(pred, sample["gt"], surfaces))
                    row["pred_alpha_deg"] = math.degrees(pred["alpha"])
                    row["pred_theta_l_deg"] = math.degrees(pred["theta_l"])
                    row["pred_theta_r_deg"] = math.degrees(pred["theta_r"])
                    row["chain_cost"] = float(pred["chain_cost"])
                    row["chain_nfev"] = int(pred["chain_nfev"])
                    for key, value in diagnostics.items():
                        if isinstance(value, (int, float, str)):
                            row[f"diag_{key}"] = value
                except Exception as exc:
                    row["status"] = f"{type(exc).__name__}: {exc}"
                    if int(args.fail_fast):
                        raise
                row["solve_sec"] = float(time.perf_counter() - solve_start)
                row["elapsed_sec"] = row["forward_sec"] + row["solve_sec"]
                rows.append(row)

            completed = min(batch_start + len(samples), len(indices))
            if batch_start == 0 or completed % int(args.print_freq) < len(samples) or completed == len(indices):
                print(f"[worker {rank}] {spec['name']} {completed}/{len(indices)}", flush=True)

        del model, surfaces, outputs, x, K_tensor
        if device.type == "cuda":
            torch.cuda.empty_cache()
    worker_path = Path(args.output_dir) / "workers" / f"worker_{rank:02d}.json"
    worker_path.parent.mkdir(parents=True, exist_ok=True)
    worker_path.write_text(json.dumps(rows, indent=2, allow_nan=True), encoding="utf-8")


def write_summary_markdown(path, summary, paired, args):
    lines = [
        "# SurfEmb articulated RARP evaluation",
        "",
        f"- test split: needleGrasping/test",
        f"- surface keys per part: {args.surface_keys_per_part}",
        f"- wrist pose estimator: {args.pose_estimator}",
        f"- AP3P hypotheses per part (legacy mode): {args.max_poses}",
        f"- scored hypotheses per part: {args.max_pose_evaluations}",
        f"- require all four parts: {bool(args.require_all_parts)}",
        "",
    ]
    for name, item in summary.items():
        lines.extend([f"## {name}", "", f"- success: {item['success']}/{item['total']} ({item['success_rate']:.2%})"])
        for metric in ("wrist_trans_err_mm", "wrist_rot_err_deg", "joint_mae_deg", "articulated_add_mm"):
            value = item["metrics"][metric]
            lines.append(f"- {metric}: mean={value['mean']:.4f}, median={value['median']:.4f}, rmse={value['rmse']:.4f}")
        lines.append("")
    if paired is not None:
        name_a, name_b = paired["model_a"], paired["model_b"]
        lines.extend(["## Paired comparison", "", f"- common successful samples: {paired['paired_success']}"])
        for metric, item in paired["metrics"].items():
            lines.append(
                f"- {metric}: {name_a} mean={item[name_a]['mean']:.4f}, "
                f"{name_b} mean={item[name_b]['mean']:.4f}; wins "
                f"{item[f'{name_a}_wins']}/{item[f'{name_b}_wins']}"
            )
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main(args):
    args.output_dir = str(Path(args.output_dir).resolve())
    args.dataset_cache_dir = str(Path(args.dataset_cache_dir).resolve())
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model_specs = parse_model_specs(args.model)
    dataset = build_dataset(args)
    total = len(dataset) if int(args.max_samples) <= 0 else min(len(dataset), int(args.max_samples))
    indices = list(range(total))
    chunks = [indices[index:: len(args.devices)] for index in range(len(args.devices))]
    chunks = [chunk for chunk in chunks if chunk]
    print(f"dataset={dataset} selected={total} models={[item['name'] for item in model_specs]}", flush=True)
    if len(chunks) == 1:
        worker_main(0, chunks, model_specs, args)
    else:
        mp.spawn(worker_main, args=(chunks, model_specs, args), nprocs=len(chunks), join=True)

    rows = []
    for rank in range(len(chunks)):
        rows.extend(json.loads((Path(args.output_dir) / "workers" / f"worker_{rank:02d}.json").read_text(encoding="utf-8")))
    rows.sort(key=lambda row: (int(row["dataset_idx"]), str(row["model"])))
    write_csv(Path(args.output_dir) / "per_instance.csv", rows)
    summary = summarize(rows, model_specs)
    paired = summarize_paired(rows, model_specs)
    if paired is not None:
        summary["paired_comparison"] = paired
    summary["evaluation_config"] = {
        "dataset": "needleGrasping/test",
        "selected_samples": total,
        "surface_keys_per_part": int(args.surface_keys_per_part),
        "mask_keys_per_part": int(args.mask_keys_per_part),
        "max_poses": int(args.max_poses),
        "max_pose_evaluations": int(args.max_pose_evaluations),
        "down_sample_scale": int(args.down_sample_scale),
        "require_all_parts": bool(args.require_all_parts),
        "pose_estimator": args.pose_estimator,
    }
    (Path(args.output_dir) / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")
    model_summary = {
        key: value for key, value in summary.items() if key not in ("evaluation_config", "paired_comparison")
    }
    write_summary_markdown(Path(args.output_dir) / "summary.md", model_summary, paired, args)
    print(json.dumps(model_summary, indent=2, allow_nan=True), flush=True)
    print(f"summary={Path(args.output_dir) / 'summary.md'}", flush=True)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", default=None, help="NAME=CHECKPOINT; repeat for multiple models")
    parser.add_argument("--output_dir", default=str(ROBOPEPP_ROOT / "logs/surfemb_articulated_needlegrasping_test"))
    parser.add_argument("--devices", nargs="+", default=["cuda:0"])
    parser.add_argument("--needle_dataset_root", default="/mnt/nas/share/shuojue/data/needleGrasping_videos")
    parser.add_argument("--needle_pose_root", default="/mnt/nas/share/shuojue/data/needleGrasping_results")
    parser.add_argument("--dataset_cache_dir", default=str(ROBOPEPP_ROOT / "logs/robopepp_rarp_lnd_refinemem_eval_keypoint_trimesh/dataset_cache"))
    parser.add_argument("--surface_root", default=str(DEFAULT_SURFACE_ROOT))
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--surfemb_crop_scale", type=float, default=1.2)
    parser.add_argument("--bbox_padding_frac", type=float, default=0.12)
    parser.add_argument("--train_ratio", type=float, default=0.95)
    parser.add_argument("--min_dice_shaft", type=float, default=0.8)
    parser.add_argument("--min_dice_wrist", type=float, default=0.6)
    parser.add_argument("--min_dice_gripper", type=float, default=0.6)
    parser.add_argument("--canonicalize_pose_symmetry", type=int, choices=[0, 1], default=1)
    parser.add_argument("--canonical_eps", type=float, default=0.08)
    parser.add_argument("--surface_keys_per_part", type=int, default=4096)
    parser.add_argument("--mask_keys_per_part", type=int, default=512)
    parser.add_argument("--surface_seed", type=int, default=2026)
    parser.add_argument("--pose_seed", type=int, default=12345)
    parser.add_argument("--down_sample_scale", type=int, default=3)
    parser.add_argument(
        "--pose_estimator",
        choices=("topk_ransac", "probability_ap3p"),
        default="topk_ransac",
    )
    parser.add_argument("--topk_max_correspondences", type=int, default=512)
    parser.add_argument("--topk_min_correspondences", type=int, default=12)
    parser.add_argument("--topk_min_part_probability", type=float, default=0.05)
    parser.add_argument("--topk_min_object_probability", type=float, default=0.5)
    parser.add_argument("--topk_margin_power", type=float, default=0.0)
    parser.add_argument("--topk_ransac_iterations", type=int, default=2000)
    parser.add_argument("--topk_ransac_reprojection_error", type=float, default=3.0)
    parser.add_argument("--topk_ransac_confidence", type=float, default=0.999)
    parser.add_argument("--topk_min_inliers", type=int, default=8)
    parser.add_argument("--topk_min_inlier_fraction", type=float, default=0.6)
    parser.add_argument("--max_poses", type=int, default=1024)
    parser.add_argument("--max_pose_evaluations", type=int, default=256)
    parser.add_argument("--pose_batch_size", type=int, default=64)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--require_all_parts", type=int, choices=[0, 1], default=1)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--amp", type=int, choices=[0, 1], default=1)
    parser.add_argument("--inference_batch_size", type=int, default=32)
    parser.add_argument("--cpu_threads_per_worker", type=int, default=2)
    parser.add_argument("--print_freq", type=int, default=10)
    parser.add_argument("--fail_fast", type=int, choices=[0, 1], default=0)
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    if args.model is None:
        args.model = list(DEFAULT_MODELS)
    main(args)
