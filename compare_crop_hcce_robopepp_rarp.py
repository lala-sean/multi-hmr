import argparse
import csv
import importlib.util
import json
import math
import os
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
from PIL import Image
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from trimesh.triangles import closest_point as closest_points_on_triangles
import torchvision.transforms as tv_transforms

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

ROBOPEPP_ROOT = Path(__file__).resolve().parent
MULTIHMR_ROOT = ROBOPEPP_ROOT.parents[1]
if str(ROBOPEPP_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBOPEPP_ROOT))
if str(MULTIHMR_ROOT) not in sys.path:
    sys.path.insert(0, str(MULTIHMR_ROOT))

datasets_pkg = types.ModuleType("datasets")
datasets_pkg.__path__ = [str(MULTIHMR_ROOT / "datasets")]
sys.modules["datasets"] = datasets_pkg

from datasets.RarpInstanceDataset import RARPInstanceDataset, _resolve_mask_subfolder  # noqa: E402
from estimate_rarp_cse_articulate_pose import (  # noqa: E402
    CAD_ROOT,
    Correspondences,
    InstrumentCAD,
    _homogeneous_transform,
    optimize_pose,
    sample_mask_pixels,
    solve_wrist_pnp,
)
from estimate_rarp_hcce_articulate_pose import decode_hcce_logits  # noqa: E402
from instrument_geometry import (  # noqa: E402
    KEYPOINT_NAMES,
    crop_resize_pad_intrinsics,
    instrument_keypoints_camera_np,
    project_points_np,
    quat_wxyz_to_matrix_np,
    rarp_intrinsics,
)
from instrument_trimesh_renderer import GMSInstrumentTrimeshRenderer  # noqa: E402
from pose_pnp import matrix_to_quat_wxyz_np, pose_from_keypoints_pnp  # noqa: E402
from predict_instrument_pose import (  # noqa: E402
    add_title,
    concat_panels,
    heatmap_argmax,
    load_model as load_robopepp_model,
    overlay_part_mask,
    pad_mask_to_square,
    pad_rgb_to_square,
    square_image_geometry,
)


DEFAULT_ROBOPEPP_CKPT = ROBOPEPP_ROOT / "logs/robopepp_instrument_pose_rarp_jepa_bs56_gpu0123/checkpoints/last.pt"
DEFAULT_HCCE_CKPT = ROBOPEPP_ROOT / "logs/hcce_crop224_fixbf16_rarp_gpu0123_bs56_fromscratch/checkpoints/last.pt"
DEFAULT_NEEDLE_MANIFEST = MULTIHMR_ROOT / "eval_outputs/best_iter55000_parallel_meshtexturefix/needleGrasping/per_instance.csv"
DEFAULT_SUTURE_MANIFEST = (
    MULTIHMR_ROOT
    / "eval_outputs/suturepulling_best55000_ref_current_best_three_exp_full_stride4/densepart_h2/per_instance.csv"
)

NEEDLE_DATASET_ROOT = "/mnt/nas/share/shuojue/data/needleGrasping_videos"
NEEDLE_POSE_ROOT = "/mnt/nas/share/shuojue/data/needleGrasping_results"
SUTURE_DATASET_ROOT = "/mnt/nas/share/shuojue/data/suturePulling_videos"

PART_LABELS = {"gripper": 1, "wrist": 2, "shaft": 3}
HCCE_DENSE_TO_PART = {0: PART_LABELS["wrist"], 1: PART_LABELS["gripper"], 2: PART_LABELS["shaft"]}
JOINT_NAMES = ("alpha", "theta_l", "theta_r")


def parse_axis_scale(value):
    if value is None or value == "":
        return None
    if isinstance(value, np.ndarray):
        vals = value.astype(np.float64).reshape(-1).tolist()
    elif isinstance(value, (list, tuple)):
        vals = [float(v) for v in value]
    else:
        vals = [float(v) for v in str(value).replace(",", " ").split()]
    if len(vals) != 3:
        raise ValueError(f"hcce_axis_scale must have exactly 3 values, got {value!r}")
    arr = np.asarray(vals, dtype=np.float64)
    if not np.all(np.isfinite(arr)) or np.any(np.abs(arr) < 1e-12):
        raise ValueError(f"Invalid hcce_axis_scale={value!r}")
    return arr


def hcce_axis_scale_from_args(model_meta, args):
    axis_scale = parse_axis_scale(args.hcce_axis_scale)
    if axis_scale is None:
        axis_scale = parse_axis_scale(model_meta.get("hcce_axis_scale", (1.0, 1.0, 1.0)))
    return axis_scale


def select_mask_pixels(mask, max_points, rng, scores=None, mode="random"):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return xs, ys
    max_points = int(max_points)
    if max_points > 0 and len(xs) > max_points:
        if mode == "random":
            keep = rng.choice(len(xs), size=max_points, replace=False)
        else:
            raise ValueError(f"Unsupported point_select={mode!r}")
        xs = xs[keep]
        ys = ys[keep]
    return xs, ys


def canon_norm_to_part(cad, part_name, xyz_norm):
    xyz_world = np.asarray(xyz_norm, dtype=np.float64) * float(cad.canon_scale)
    return _homogeneous_transform(xyz_world, cad.canon_to_part[part_name])


def surface_cache_for_part(cad, part_name):
    cache = getattr(cad, "_crop_hcce_surface_closest_cache", None)
    if cache is None:
        cache = {}
        cad._crop_hcce_surface_closest_cache = cache
    if part_name in cache:
        return cache[part_name]

    vertices_part = np.asarray(cad.vertices_part[part_name], dtype=np.float64)
    vertices_norm = np.asarray(cad.canon_norm[part_name], dtype=np.float64)
    faces = np.asarray(cad.faces[part_name], dtype=np.int64)
    triangles_part = vertices_part[faces]
    area = 0.5 * np.linalg.norm(
        np.cross(
            triangles_part[:, 1] - triangles_part[:, 0],
            triangles_part[:, 2] - triangles_part[:, 0],
        ),
        axis=1,
    )
    keep = area > 1e-14
    if not np.any(keep):
        raise RuntimeError(f"No valid mesh faces for {part_name}")
    clean_faces = faces[keep]
    triangles_norm = vertices_norm[clean_faces]
    item = {
        "triangles_norm": triangles_norm,
        "tree": cKDTree(triangles_norm.mean(axis=1)),
    }
    cache[part_name] = item
    return item


def closest_part_surface_points(cad, part_name, xyz_norm, k_faces=32):
    xyz_norm = np.asarray(xyz_norm, dtype=np.float64)
    if len(xyz_norm) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    surface = surface_cache_for_part(cad, part_name)
    triangles_norm = surface["triangles_norm"]
    k_faces = int(k_faces)
    exhaustive = k_faces <= 0 or k_faces >= len(triangles_norm)
    if exhaustive:
        best = np.zeros((len(xyz_norm), 3), dtype=np.float64)
        chunk = 16
        for start in range(0, len(xyz_norm), chunk):
            pts = xyz_norm[start:start + chunk]
            flat_tri = np.broadcast_to(
                triangles_norm[None],
                (len(pts), len(triangles_norm), 3, 3),
            ).reshape(-1, 3, 3)
            flat_pts = np.repeat(pts, len(triangles_norm), axis=0)
            close = closest_points_on_triangles(flat_tri, flat_pts)
            dist2 = np.sum((close - flat_pts) ** 2, axis=1).reshape(len(pts), len(triangles_norm))
            best_idx = np.argmin(dist2, axis=1)
            best[start:start + len(pts)] = close.reshape(len(pts), len(triangles_norm), 3)[
                np.arange(len(pts)), best_idx
            ]
    else:
        _, cand_idx = surface["tree"].query(
            xyz_norm, k=min(max(1, k_faces), len(triangles_norm))
        )
        cand_idx = np.asarray(cand_idx, dtype=np.int64)
        if cand_idx.ndim == 1:
            cand_idx = cand_idx[:, None]
        flat_tri = triangles_norm[cand_idx.reshape(-1)]
        flat_pts = np.repeat(xyz_norm, cand_idx.shape[1], axis=0)
        close = closest_points_on_triangles(flat_tri, flat_pts)
        dist2 = np.sum((close - flat_pts) ** 2, axis=1).reshape(len(xyz_norm), cand_idx.shape[1])
        best_local = np.argmin(dist2, axis=1)
        best = close.reshape(len(xyz_norm), cand_idx.shape[1], 3)[np.arange(len(xyz_norm)), best_local]
    return canon_norm_to_part(cad, part_name, best).astype(np.float64)


def closest_gripper_surface_points(cad, xyz_norm, k_faces=32):
    left = closest_part_surface_points(cad, "l_gripper", xyz_norm, k_faces=k_faces)
    right = closest_part_surface_points(cad, "r_gripper", xyz_norm, k_faces=k_faces)
    left_norm = _homogeneous_transform(left, np.linalg.inv(cad.canon_to_part["l_gripper"]))
    right_norm = _homogeneous_transform(right, np.linalg.inv(cad.canon_to_part["r_gripper"]))
    left_norm = left_norm / float(cad.canon_scale)
    right_norm = right_norm / float(cad.canon_scale)
    dist_l = np.linalg.norm(left_norm - xyz_norm, axis=1)
    dist_r = np.linalg.norm(right_norm - xyz_norm, axis=1)
    use_l = dist_l <= dist_r
    points = np.empty_like(left)
    names = np.empty((len(xyz_norm),), dtype=object)
    points[use_l] = left[use_l]
    points[~use_l] = right[~use_l]
    names[use_l] = "l_gripper"
    names[~use_l] = "r_gripper"
    return points.astype(np.float64), names


class Pose:
    def __setstate__(self, state):
        self.__dict__.update(state if isinstance(state, dict) else {})


def _load_local_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_rarp_module = _load_local_module(
    "robopepp_compare_rarp_instrument",
    ROBOPEPP_ROOT / "datasets" / "rarp_instrument.py",
)
RoboPEPPRARPInstrument = _rarp_module.RoboPEPPRARPInstrument
_hcce_model_module = _load_local_module(
    "robopepp_compare_hcce_crop_model",
    ROBOPEPP_ROOT / "models" / "hcce_crop_model.py",
)
CropHCCEDenseKeypointDPT = _hcce_model_module.CropHCCEDenseKeypointDPT


@dataclass
class EvalItem:
    dataset: str
    video: str
    frame_id: str
    instance_id: int
    ordinal: int
    dataset_idx: int = -1
    actual_instance_id: int = -1
    source_sample_idx: str = ""

    def to_json(self):
        return {
            "dataset": self.dataset,
            "video": self.video,
            "frame_id": self.frame_id,
            "instance_id": self.instance_id,
            "ordinal": self.ordinal,
            "dataset_idx": self.dataset_idx,
            "actual_instance_id": self.actual_instance_id,
            "source_sample_idx": self.source_sample_idx,
        }

    @staticmethod
    def from_json(payload):
        return EvalItem(
            dataset=payload["dataset"],
            video=payload["video"],
            frame_id=norm_frame_id(payload["frame_id"]),
            instance_id=int(payload["instance_id"]),
            ordinal=int(payload["ordinal"]),
            dataset_idx=int(payload.get("dataset_idx", -1)),
            actual_instance_id=int(payload.get("actual_instance_id", -1)),
            source_sample_idx=str(payload.get("source_sample_idx", "")),
        )


def norm_frame_id(frame_id):
    return f"{int(frame_id):05d}"


def sample_key(video, frame_id, instance_id):
    return str(video), norm_frame_id(frame_id), int(instance_id)


def resolve_path(path, base=ROBOPEPP_ROOT):
    path = Path(path)
    if path.is_absolute() or path.exists():
        return path
    return base / path


def read_csv_rows(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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
        "video",
        "frame_id",
        "instance_id",
        "ordinal",
        "dataset_idx",
        "actual_instance_id",
        "has_pose_gt",
        "robopepp_pnp_status",
        "hcce_direct_status",
        "hcce_fit_status",
        "hcce_kp_pnp_status",
        "vis_path",
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


def float_or_nan(value):
    try:
        if value is None or value == "":
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def finite_values(rows, key):
    vals = []
    for row in rows:
        value = float_or_nan(row.get(key))
        if math.isfinite(value):
            vals.append(value)
    return np.asarray(vals, dtype=np.float64)


def stats_for(rows, key):
    vals = finite_values(rows, key)
    if vals.size == 0:
        return {"count": 0, "mean": float("nan"), "median": float("nan"), "std": float("nan"), "rmse": float("nan")}
    return {
        "count": int(vals.size),
        "mean": float(vals.mean()),
        "median": float(np.median(vals)),
        "std": float(vals.std()),
        "rmse": float(np.sqrt(np.mean(vals * vals))),
    }


def summarize_rows(rows):
    methods = ("robopepp_direct", "robopepp_pnp", "hcce_direct", "hcce_fit", "hcce_kp_pnp")
    metric_suffixes = (
        "trans_err_m",
        "rot_err_deg",
        "joint_mae_deg",
        "alpha_err_deg",
        "theta_l_err_deg",
        "theta_r_err_deg",
        "render_seg_inst_iou",
        "render_seg_part_iou_mean",
        "kp_reproj_rmse_px",
        "hm_crop_rmse_px",
    )
    out = {
        "num_rows": len(rows),
        "num_pose_gt": int(sum(int(row.get("has_pose_gt", 0)) for row in rows)),
        "datasets": sorted({row.get("dataset", "") for row in rows if row.get("dataset", "")}),
        "metrics": {},
        "status_counts": {},
    }
    for method in methods:
        status_key = f"{method}_status"
        counts = {}
        for row in rows:
            status = str(row.get(status_key, "missing"))
            counts[status] = counts.get(status, 0) + 1
        out["status_counts"][method] = counts
        for suffix in metric_suffixes:
            key = f"{method}_{suffix}"
            stat = stats_for(rows, key)
            if stat["count"] > 0:
                out["metrics"][key] = stat
    return out


def fmt_stat(summary, key):
    stat = summary["metrics"].get(key)
    if not stat:
        return "n/a"
    return f"mean={stat['mean']:.6g}, median={stat['median']:.6g}, rmse={stat['rmse']:.6g}, n={stat['count']}"


def write_summary_md(path, summary_by_name):
    lines = [
        "# Crop HCCE vs RoboPEPP RARP Quant",
        "",
        "Metric note: `robopepp_pnp` and `hcce_kp_pnp` solve only wrist SE3 from heatmap keypoints. Their joint angles are copied from the same model's direct/action head, so their joint MAE is not an independent keypoint-PnP joint estimate.",
        "",
    ]
    for name, summary in summary_by_name.items():
        lines += [
            f"## {name}",
            f"- rows: {summary['num_rows']}",
            f"- rows with pose GT: {summary['num_pose_gt']}",
            "",
            "| method | trans L2 m | rot deg | joint MAE deg | render part IoU |",
            "| --- | --- | --- | --- | --- |",
        ]
        for method in ("robopepp_pnp", "robopepp_direct", "hcce_direct", "hcce_fit", "hcce_kp_pnp"):
            lines.append(
                "| "
                + " | ".join(
                    [
                        method,
                        fmt_stat(summary, f"{method}_trans_err_m"),
                        fmt_stat(summary, f"{method}_rot_err_deg"),
                        fmt_stat(summary, f"{method}_joint_mae_deg"),
                        fmt_stat(summary, f"{method}_render_seg_part_iou_mean"),
                    ]
                )
                + " |"
            )
        lines += ["", "Status counts:", json.dumps(summary["status_counts"], indent=2, sort_keys=True), ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def binary_iou(pred, gt):
    pred = np.asarray(pred).astype(bool)
    gt = np.asarray(gt).astype(bool)
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return float("nan")
    return float(np.logical_and(pred, gt).sum() / union)


def segmentation_metrics_from_masks(pred_part, gt_part, prefix):
    pred_part = np.asarray(pred_part, dtype=np.int64)
    gt_part = np.asarray(gt_part, dtype=np.int64)
    metrics = {
        f"{prefix}_inst_iou": binary_iou(pred_part > 0, gt_part > 0),
        f"{prefix}_pred_inst_area": int((pred_part > 0).sum()),
        f"{prefix}_gt_inst_area": int((gt_part > 0).sum()),
    }
    part_ious = []
    for part_name, label in PART_LABELS.items():
        iou = binary_iou(pred_part == label, gt_part == label)
        metrics[f"{prefix}_part_iou_{part_name}"] = iou
        if math.isfinite(iou):
            part_ious.append(iou)
    metrics[f"{prefix}_part_iou_mean"] = float(np.mean(part_ious)) if part_ious else float("nan")
    return metrics


def rotation_error_deg(q_pred, q_gt):
    r_pred = quat_wxyz_to_matrix_np(q_pred)
    r_gt = quat_wxyz_to_matrix_np(q_gt)
    rel = r_pred @ r_gt.T
    cos_angle = np.clip((float(np.trace(rel)) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def pose_from_output(out):
    action = out["action_pred"][0].detach().float().cpu().numpy().astype(np.float64)
    quat = out["wrist_quat_pred"][0].detach().float().cpu().numpy().astype(np.float64)
    quat = quat / max(np.linalg.norm(quat), 1e-12)
    if quat[0] < 0.0:
        quat = -quat
    trans = out["wrist_trans_pred"][0].detach().float().cpu().numpy().astype(np.float64)
    return {
        "rot": quat,
        "trans": trans,
        "alpha": float(action[0]),
        "theta_l": float(action[1]),
        "theta_r": float(action[2]),
    }


def pose_from_target(target):
    quat = target["wrist_quat"].detach().cpu().numpy().astype(np.float64)
    quat = quat / max(np.linalg.norm(quat), 1e-12)
    if quat[0] < 0.0:
        quat = -quat
    action = target["action"].detach().cpu().numpy().astype(np.float64)
    return {
        "rot": quat,
        "trans": target["wrist_trans"].detach().cpu().numpy().astype(np.float64),
        "alpha": float(action[0]),
        "theta_l": float(action[1]),
        "theta_r": float(action[2]),
    }


def pose_from_opt_params(params):
    params = np.asarray(params, dtype=np.float64).reshape(9)
    rot_mat, _ = cv2.Rodrigues(params[:3].reshape(3, 1))
    return {
        "rot": matrix_to_quat_wxyz_np(rot_mat),
        "trans": params[3:6].astype(np.float64),
        "alpha": float(params[6]),
        "theta_l": float(params[7]),
        "theta_r": float(params[8]),
    }


def pose_action_array(pose):
    return np.asarray([pose["alpha"], pose["theta_l"], pose["theta_r"]], dtype=np.float64)


def add_pose_errors(row, pred_pose, gt_pose, prefix):
    if gt_pose is None or pred_pose is None:
        for key in ("trans_err_m", "rot_err_deg", "joint_mae_deg", "alpha_err_deg", "theta_l_err_deg", "theta_r_err_deg"):
            row[f"{prefix}_{key}"] = float("nan")
        return
    pred_action = pose_action_array(pred_pose)
    gt_action = pose_action_array(gt_pose)
    action_err_deg = np.degrees(np.abs(pred_action - gt_action))
    row[f"{prefix}_trans_err_m"] = float(np.linalg.norm(np.asarray(pred_pose["trans"]) - np.asarray(gt_pose["trans"])))
    row[f"{prefix}_rot_err_deg"] = rotation_error_deg(pred_pose["rot"], gt_pose["rot"])
    row[f"{prefix}_joint_mae_deg"] = float(np.mean(action_err_deg))
    for i, name in enumerate(JOINT_NAMES):
        row[f"{prefix}_{name}_err_deg"] = float(action_err_deg[i])


def add_keypoint_reprojection(row, pose, target_like, prefix):
    if pose is None or target_like.get("keypoints_crop") is None:
        row[f"{prefix}_kp_reproj_rmse_px"] = float("nan")
        return
    action = pose_action_array(pose)
    kp_3d = instrument_keypoints_camera_np(pose["rot"], pose["trans"], action)
    pred_uv = project_points_np(kp_3d, target_like["K_crop"])
    gt_uv = target_like["keypoints_crop"]
    valid = target_like["keypoints_valid"]
    if valid is None or not np.any(valid):
        row[f"{prefix}_kp_reproj_rmse_px"] = float("nan")
        return
    err = pred_uv[valid] - gt_uv[valid]
    row[f"{prefix}_kp_reproj_rmse_px"] = float(np.sqrt(np.mean(np.sum(err * err, axis=1))))


def crop_points_to_original(points_crop, bbox_min, scale_xy, pad_xy):
    points = np.asarray(points_crop, dtype=np.float64).copy()
    scale = np.asarray(scale_xy, dtype=np.float64).reshape(2)
    pad = np.asarray(pad_xy, dtype=np.float64).reshape(2)
    bbox_min = np.asarray(bbox_min, dtype=np.float64).reshape(2)
    points[:, 0] = (points[:, 0] - pad[0]) / scale[0] + bbox_min[0]
    points[:, 1] = (points[:, 1] - pad[1]) / scale[1] + bbox_min[1]
    return points.astype(np.float32)


def project_camera_points(points_cam, K):
    points_cam = np.asarray(points_cam, dtype=np.float64)
    uvw = points_cam @ np.asarray(K, dtype=np.float64).reshape(3, 3).T
    return uvw[:, :2] / points_cam[:, 2:3]


def subset_correspondences(corr, mask):
    return Correspondences(
        uv=corr.uv[mask],
        points_part=corr.points_part[mask],
        part_names=corr.part_names[mask],
    )


def correspondence_residuals(cad, corr, K, args, rvec, trans, alpha, theta_l, theta_r):
    transforms = cad.fk(rvec, trans, alpha, theta_l, theta_r)
    residuals = []
    for part_name in sorted(set(corr.part_names.tolist())):
        idx = np.where(corr.part_names == part_name)[0]
        points_cam = cad.transform_points(part_name, corr.points_part[idx], transforms)
        z_ok = points_cam[:, 2] > float(args.min_depth)
        part_res = np.full(
            (len(idx), 2),
            float(getattr(args, "behind_camera_penalty", 1e4)),
            dtype=np.float64,
        )
        if np.any(z_ok):
            part_res[z_ok] = project_camera_points(points_cam[z_ok], K) - corr.uv[idx][z_ok]
        residuals.append(part_res.reshape(-1))
    return np.concatenate(residuals, axis=0)


def optimize_pose_gripper_joint_only(cad, corr, rvec0, trans0, K, args):
    wrist_mask = corr.part_names == "wrist"
    shaft_mask = corr.part_names == "shaft"
    gripper_mask = np.isin(corr.part_names, ["l_gripper", "r_gripper"])
    wrist_shaft_mask = wrist_mask | shaft_mask
    if int(wrist_shaft_mask.sum()) < int(args.min_total_points):
        raise RuntimeError(
            f"Not enough wrist/shaft correspondences: {int(wrist_shaft_mask.sum())} < {args.min_total_points}"
        )
    if int(wrist_mask.sum()) < int(args.min_wrist_points):
        raise RuntimeError(f"Not enough wrist correspondences: {int(wrist_mask.sum())} < {args.min_wrist_points}")
    if int(shaft_mask.sum()) < int(args.min_shaft_points):
        raise RuntimeError(f"Not enough shaft correspondences: {int(shaft_mask.sum())} < {args.min_shaft_points}")
    if int(gripper_mask.sum()) < int(args.min_total_points):
        raise RuntimeError(
            f"Not enough gripper correspondences for gripper-joint fit: "
            f"{int(gripper_mask.sum())} < {args.min_total_points}"
        )

    wrist_shaft_corr = subset_correspondences(corr, wrist_shaft_mask)
    gripper_corr = subset_correspondences(corr, gripper_mask)
    joint_lower = np.array(
        [-math.pi / 2.0, -80.0 / 180.0 * math.pi, -80.0 / 180.0 * math.pi],
        dtype=np.float64,
    )
    joint_upper = np.array(
        [math.pi / 2.0, 80.0 / 180.0 * math.pi, 80.0 / 180.0 * math.pi],
        dtype=np.float64,
    )

    def residual_wrist_shaft(params):
        return correspondence_residuals(
            cad,
            wrist_shaft_corr,
            K,
            args,
            params[:3],
            params[3:6],
            float(params[6]),
            0.0,
            0.0,
        )

    x0_ws = np.array(
        [rvec0[0], rvec0[1], rvec0[2], trans0[0], trans0[1], trans0[2], 0.0],
        dtype=np.float64,
    )
    lower_ws = np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, joint_lower[0]], dtype=np.float64)
    upper_ws = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, joint_upper[0]], dtype=np.float64)
    res_ws = least_squares(
        residual_wrist_shaft,
        x0_ws,
        bounds=(lower_ws, upper_ws),
        loss=args.optim_loss,
        f_scale=args.optim_f_scale,
        max_nfev=args.optim_max_nfev,
        verbose=0,
    )
    if not res_ws.success:
        raise RuntimeError(f"Wrist/shaft optimization failed: {res_ws.message}")
    rvec_opt = res_ws.x[:3]
    trans_opt = res_ws.x[3:6]
    alpha_opt = float(res_ws.x[6])

    def residual_gripper_joints(theta):
        return correspondence_residuals(
            cad,
            gripper_corr,
            K,
            args,
            rvec_opt,
            trans_opt,
            alpha_opt,
            float(theta[0]),
            float(theta[1]),
        )

    res_gripper = least_squares(
        residual_gripper_joints,
        np.zeros(2, dtype=np.float64),
        bounds=(joint_lower[1:], joint_upper[1:]),
        loss=args.optim_loss,
        f_scale=args.optim_f_scale,
        max_nfev=args.optim_max_nfev,
        verbose=0,
    )
    if not res_gripper.success:
        raise RuntimeError(f"Gripper joint optimization failed: {res_gripper.message}")

    opt_params = np.array(
        [
            rvec_opt[0],
            rvec_opt[1],
            rvec_opt[2],
            trans_opt[0],
            trans_opt[1],
            trans_opt[2],
            alpha_opt,
            float(res_gripper.x[0]),
            float(res_gripper.x[1]),
        ],
        dtype=np.float64,
    )

    def rmse(mask):
        if not bool(mask.any()):
            return float("nan")
        active = subset_correspondences(corr, mask)
        res = correspondence_residuals(
            cad,
            active,
            K,
            args,
            opt_params[:3],
            opt_params[3:6],
            opt_params[6],
            opt_params[7],
            opt_params[8],
        )
        return float(np.sqrt(np.mean(res.reshape(-1, 2) ** 2)))

    extra = {
        "hcce_fit_gripper_no_wrist_pose": 1,
        "hcce_fit_wrist_shaft_corr_count": int(wrist_shaft_mask.sum()),
        "hcce_fit_gripper_corr_count": int(gripper_mask.sum()),
        "hcce_fit_reproj_rmse_wrist_px": rmse(wrist_mask),
        "hcce_fit_reproj_rmse_gripper_px": rmse(gripper_mask),
    }
    return (
        opt_params,
        rmse(np.ones(len(corr), dtype=bool)),
        int(res_ws.nfev + res_gripper.nfev),
        rmse(wrist_mask | gripper_mask),
        rmse(shaft_mask),
        extra,
    )


def optimize_pose_wrist_only(cad, corr, rvec0, trans0, K, args):
    wrist_mask = corr.part_names == "wrist"
    if int(wrist_mask.sum()) < int(args.min_wrist_points):
        raise RuntimeError(f"Not enough wrist correspondences: {int(wrist_mask.sum())} < {args.min_wrist_points}")
    wrist_corr = subset_correspondences(corr, wrist_mask)

    def residual_wrist(params):
        return correspondence_residuals(
            cad,
            wrist_corr,
            K,
            args,
            params[:3],
            params[3:6],
            0.0,
            0.0,
            0.0,
        )

    x0 = np.array(
        [rvec0[0], rvec0[1], rvec0[2], trans0[0], trans0[1], trans0[2]],
        dtype=np.float64,
    )
    res = least_squares(
        residual_wrist,
        x0,
        loss=args.optim_loss,
        f_scale=args.optim_f_scale,
        max_nfev=args.optim_max_nfev,
        verbose=0,
    )
    if not res.success:
        raise RuntimeError(f"Wrist-only optimization failed: {res.message}")
    opt_params = np.array(
        [
            res.x[0],
            res.x[1],
            res.x[2],
            res.x[3],
            res.x[4],
            res.x[5],
            0.0,
            0.0,
            0.0,
        ],
        dtype=np.float64,
    )
    reproj = residual_wrist(res.x).reshape(-1, 2)
    rmse_wrist = float(np.sqrt(np.mean(reproj * reproj)))
    extra = {
        "hcce_fit_mode": "wrist_only",
        "hcce_fit_wrist_only_corr_count": int(wrist_mask.sum()),
        "hcce_fit_reproj_rmse_wrist_px": rmse_wrist,
    }
    return opt_params, rmse_wrist, int(res.nfev), rmse_wrist, float("nan"), extra


def pnp_from_heatmap_output(out, K_crop, min_score):
    pred_hm_crop_t, hm_scores_t = heatmap_argmax(out["keypoint_heatmaps"].float().detach().cpu())
    pred_hm_crop = pred_hm_crop_t[0].numpy().astype(np.float32)
    hm_scores = hm_scores_t[0].numpy().astype(np.float32)
    direct_pose = pose_from_output(out)
    pose = pose_from_keypoints_pnp(
        pred_hm_crop,
        pose_action_array(direct_pose),
        np.asarray(K_crop, dtype=np.float32),
        scores=hm_scores,
        min_score=float(min_score),
    )
    return pose, pred_hm_crop, hm_scores, direct_pose


def model_part_mask_crop(out, inst_thresh):
    inst = torch.sigmoid(out["inst_mask_logits"][0].detach().float().cpu()).numpy() >= float(inst_thresh)
    dense_part = torch.argmax(out["part_mask_logits"][0].detach().float().cpu(), dim=0).numpy().astype(np.int64)
    part = np.zeros(dense_part.shape, dtype=np.uint8)
    for dense_label, gt_label in HCCE_DENSE_TO_PART.items():
        part[inst & (dense_part == dense_label)] = int(gt_label)
    return part


def resize_longer_side_array(arr, crop_size, interpolation):
    h, w = arr.shape[:2]
    if w > h:
        new_w = int(crop_size)
        new_h = int(crop_size * h / w)
    else:
        new_h = int(crop_size)
        new_w = int(crop_size * w / h)
    return cv2.resize(arr, (new_w, new_h), interpolation=interpolation), (new_w, new_h)


def pad_2d(arr, crop_size, value=0):
    h, w = arr.shape[:2]
    pad_h = (crop_size - h) // 2
    pad_w = (crop_size - w) // 2
    if arr.ndim == 2:
        padding = ((pad_h, crop_size - h - pad_h), (pad_w, crop_size - w - pad_w))
    else:
        padding = ((pad_h, crop_size - h - pad_h), (pad_w, crop_size - w - pad_w), (0, 0))
    return np.pad(arr, padding, mode="constant", constant_values=value)


def crop_resize_pad_map(arr, bbox_min, bbox_max, crop_size, interpolation, value=0):
    x0, y0 = np.asarray(bbox_min, dtype=np.float32).astype(np.int64)
    x1, y1 = np.ceil(np.asarray(bbox_max, dtype=np.float32)).astype(np.int64)
    crop = arr[y0:y1, x0:x1]
    resized, _ = resize_longer_side_array(crop, crop_size, interpolation)
    return pad_2d(resized, crop_size, value=value)


def add_model_crop_segmentation(row, out, target_like, args):
    if target_like.get("gt_part_crop") is None:
        return
    pred = model_part_mask_crop(out, args.inst_thresh)
    row.update(segmentation_metrics_from_masks(pred, target_like["gt_part_crop"], "hcce_model_crop_seg"))


def build_crop_hcce_correspondences(out, cad, model_meta, target_like, args, rng):
    inst_prob = torch.sigmoid(out["inst_mask_logits"][0].detach().float().cpu()).numpy()
    part_logits = out["part_mask_logits"][0].detach().float().cpu()
    dense_part = torch.argmax(part_logits, dim=0).numpy().astype(np.int64)
    part_conf = torch.softmax(part_logits, dim=0).max(dim=0).values.numpy().astype(np.float32)
    xyz = decode_hcce_logits(
        out["hcce_logits"][0].detach().float().cpu(),
        bits=int(args.hcce_bits),
        coord_min=float(args.hcce_coord_min),
        coord_max=float(args.hcce_coord_max),
        threshold=float(args.hcce_bit_thresh),
    ).detach().cpu().numpy()
    axis_scale = hcce_axis_scale_from_args(model_meta, args)

    inst_mask = inst_prob >= float(args.inst_thresh)
    all_uv = []
    all_points = []
    all_part_names = []
    wrist_uv = []
    wrist_points = []
    counts = {}
    fit_seg_source = str(getattr(args, "fit_seg_source", "pred"))
    gt_part_crop = target_like.get("gt_part_crop")
    if fit_seg_source == "gt":
        if gt_part_crop is None:
            raise RuntimeError("--fit_seg_source gt requires gt_part_crop")
        gt_part_crop = np.asarray(gt_part_crop)
        if gt_part_crop.shape[:2] != inst_prob.shape[:2]:
            raise RuntimeError(
                f"GT part crop shape {gt_part_crop.shape[:2]} does not match HCCE output {inst_prob.shape[:2]}"
            )

    fit_mode = str(getattr(args, "hcce_fit_mode", "full"))
    if fit_mode == "wrist_only":
        part_specs = [("wrist", 0, PART_LABELS["wrist"], int(args.max_points_per_part))]
    else:
        part_specs = [
            ("shaft", 2, PART_LABELS["shaft"], int(args.max_points_per_part)),
            ("wrist", 0, PART_LABELS["wrist"], int(args.max_points_per_part)),
            ("gripper", 1, PART_LABELS["gripper"], int(args.max_points_per_part)),
        ]
    for part_name, dense_label, gt_label, max_points in part_specs:
        if fit_seg_source == "gt":
            part_mask = gt_part_crop == int(gt_label)
        else:
            part_mask = inst_mask & (dense_part == dense_label)
        if part_name == "shaft" and args.shaft_raw_x_min is not None:
            part_mask = (
                part_mask
                & np.isfinite(xyz[:, :, 0])
                & (xyz[:, :, 0] > float(args.shaft_raw_x_min))
            )
        counts[f"{part_name}_candidate_points"] = int(np.count_nonzero(part_mask))
        xs, ys = select_mask_pixels(
            part_mask,
            max_points,
            rng,
            scores=inst_prob * part_conf,
            mode=str(args.point_select),
        )
        if len(xs) == 0:
            counts[f"{part_name}_raw_points"] = 0
            continue
        uv = np.stack([xs.astype(np.float64) + 0.5, ys.astype(np.float64) + 0.5], axis=1)
        xyz_norm = xyz[ys, xs].astype(np.float64)
        finite = np.isfinite(xyz_norm).all(axis=1)
        uv = uv[finite]
        xyz_norm = xyz_norm[finite]
        if len(xyz_norm) == 0:
            counts[f"{part_name}_raw_points"] = 0
            continue
        counts[f"{part_name}_raw_points"] = int(len(uv))
        xyz_cad_norm = xyz_norm / axis_scale.reshape(1, 3)
        if str(args.surface_snap_method) == "surface":
            if part_name == "gripper":
                points_part, names = closest_gripper_surface_points(
                    cad, xyz_cad_norm, k_faces=int(args.surface_k_faces)
                )
            else:
                points_part = closest_part_surface_points(
                    cad, part_name, xyz_cad_norm, k_faces=int(args.surface_k_faces)
                )
                names = np.full((len(points_part),), part_name, dtype=object)
        elif part_name == "gripper":
            points_part, names = cad.nearest_gripper_points(xyz_cad_norm)
        else:
            points_part = cad.nearest_part_points(part_name, xyz_cad_norm)
            names = np.full((len(points_part),), part_name, dtype=object)
        all_uv.append(uv)
        all_points.append(points_part)
        all_part_names.append(names)
        if part_name == "wrist":
            wrist_uv.append(uv)
            wrist_points.append(points_part)

    if not wrist_uv:
        raise RuntimeError("No predicted wrist HCCE pixels")
    wrist_uv = np.concatenate(wrist_uv, axis=0)
    wrist_points = np.concatenate(wrist_points, axis=0)
    if len(wrist_uv) < int(args.min_wrist_points):
        raise RuntimeError(f"Not enough wrist correspondences: {len(wrist_uv)} < {args.min_wrist_points}")
    if not all_uv:
        raise RuntimeError("No HCCE correspondences")
    corr = Correspondences(
        uv=np.concatenate(all_uv, axis=0).astype(np.float64),
        points_part=np.concatenate(all_points, axis=0).astype(np.float64),
        part_names=np.concatenate(all_part_names, axis=0),
    )
    if len(corr) < int(args.min_total_points):
        raise RuntimeError(f"Not enough total correspondences: {len(corr)} < {args.min_total_points}")
    return corr, wrist_uv.astype(np.float64), wrist_points.astype(np.float64), counts


def fit_pose_from_hcce(out, cad, model_meta, target_like, args, rng):
    K_crop = target_like["K_crop"]
    corr, wrist_uv, wrist_points, counts = build_crop_hcce_correspondences(out, cad, model_meta, target_like, args, rng)
    axis_scale = hcce_axis_scale_from_args(model_meta, args)
    rvec0, trans0, pnp_inliers = solve_wrist_pnp(wrist_uv, wrist_points, np.asarray(K_crop, dtype=np.float64), args)
    fit_mode = str(getattr(args, "hcce_fit_mode", "full"))
    if fit_mode == "wrist_only":
        opt_params, rmse_all, nfev, rmse_wg, rmse_shaft, fit_mode_extra = optimize_pose_wrist_only(
            cad, corr, rvec0, trans0, np.asarray(K_crop, dtype=np.float64), args
        )
    elif int(getattr(args, "gripper_no_wrist_pose", 0)):
        opt_params, rmse_all, nfev, rmse_wg, rmse_shaft, fit_mode_extra = optimize_pose_gripper_joint_only(
            cad, corr, rvec0, trans0, np.asarray(K_crop, dtype=np.float64), args
        )
    else:
        opt_params, rmse_all, nfev, rmse_wg, rmse_shaft = optimize_pose(
            cad, corr, rvec0, trans0, np.asarray(K_crop, dtype=np.float64), args
        )
        fit_mode_extra = {"hcce_fit_gripper_no_wrist_pose": 0, "hcce_fit_mode": "full"}
    extra = {
        "hcce_fit_corr_count": int(len(corr)),
        "hcce_fit_wrist_corr_count": int(len(wrist_uv)),
        "hcce_fit_pnp_inliers": int(pnp_inliers),
        "hcce_fit_optim_nfev": int(nfev),
        "hcce_fit_reproj_rmse_all_px": float(rmse_all),
        "hcce_fit_reproj_rmse_wrist_gripper_px": float(rmse_wg),
        "hcce_fit_reproj_rmse_shaft_px": float(rmse_shaft),
        "hcce_fit_surface_snap_method": str(args.surface_snap_method),
        "hcce_fit_surface_k_faces": int(args.surface_k_faces),
        "hcce_fit_point_select": str(args.point_select),
        "hcce_fit_seg_source": str(getattr(args, "fit_seg_source", "pred")),
        "hcce_fit_axis_scale": ",".join(f"{v:g}" for v in axis_scale),
    }
    extra.update(fit_mode_extra)
    extra.update({f"hcce_fit_{k}": v for k, v in counts.items()})
    return pose_from_opt_params(opt_params), extra


def render_pose_segmentation(row, renderer, pose, target_like, prefix, args):
    if not bool(args.render_seg_metrics):
        for key in (
            "inst_iou",
            "pred_inst_area",
            "gt_inst_area",
            "part_iou_gripper",
            "part_iou_wrist",
            "part_iou_shaft",
            "part_iou_mean",
        ):
            row[f"{prefix}_render_seg_{key}"] = float("nan")
        return None
    if int(args.render_seg_metrics_limit) >= 0 and int(row.get("ordinal", -1)) >= int(args.render_seg_metrics_limit):
        for key in (
            "inst_iou",
            "pred_inst_area",
            "gt_inst_area",
            "part_iou_gripper",
            "part_iou_wrist",
            "part_iou_shaft",
            "part_iou_mean",
        ):
            row[f"{prefix}_render_seg_{key}"] = float("nan")
        return None
    if pose is None:
        row.update(segmentation_metrics_from_masks(np.zeros_like(target_like["gt_part_orig"]), target_like["gt_part_orig"], f"{prefix}_render_seg"))
        return None
    part = renderer.render_pose_mask(
        pose,
        target_like["K_orig"],
        target_like["orig_rgb"].shape[:2],
        min_depth=float(args.render_min_depth),
        draw_margin=float(args.render_draw_margin),
    )
    row.update(segmentation_metrics_from_masks(part, target_like["gt_part_orig"], f"{prefix}_render_seg"))
    return part


def make_concat_visual(target_like, poses, renderer, row, args, gt_pose=None):
    rgb = target_like["orig_rgb"]
    gt_part = target_like["gt_part_orig"]
    scale, pad_x, pad_y = square_image_geometry(rgb, int(args.panel_size))
    rgb_sq = pad_rgb_to_square(rgb, int(args.panel_size), scale, pad_x, pad_y)
    gt_sq = pad_mask_to_square(gt_part, int(args.panel_size), scale, pad_x, pad_y)
    panels = [
        add_title(rgb_sq, "rgb"),
        add_title(overlay_part_mask(rgb_sq, gt_sq), "gt-seg"),
    ]
    if gt_pose is not None:
        gt_panel = renderer.render_pose_overlay(
            rgb,
            gt_pose,
            target_like["K_orig"],
            alpha=float(args.overlay_alpha),
        )
        panels.append(add_title(pad_rgb_to_square(gt_panel, int(args.panel_size), scale, pad_x, pad_y), "gt-pose"))
    for key, title in (
        ("robopepp_pnp", "robopepp-pnp"),
        ("robopepp_direct", "robopepp-direct"),
        ("hcce_direct", "hcce-direct"),
        ("hcce_fit", "hcce-fit"),
        ("hcce_kp_pnp", "hcce-kp-pnp"),
    ):
        pose = poses.get(key)
        if pose is None:
            panel = rgb.copy()
            cv2.putText(panel, "failed", (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 40, 40), 2, cv2.LINE_AA)
        else:
            panel = renderer.render_pose_overlay(
                rgb,
                pose,
                target_like["K_orig"],
                alpha=float(args.overlay_alpha),
            )
        panel_sq = pad_rgb_to_square(panel, int(args.panel_size), scale, pad_x, pad_y)
        rot = float_or_nan(row.get(f"{key}_rot_err_deg"))
        trans = float_or_nan(row.get(f"{key}_trans_err_m"))
        if math.isfinite(rot) and math.isfinite(trans):
            title_i = f"{title} t={trans:.3f} r={rot:.1f}"
        else:
            title_i = title
        panels.append(add_title(panel_sq, title_i))
    return concat_panels(panels)


def make_crop_gt_visual(target_like, renderer, gt_pose, args):
    crop_rgb = target_like["crop_rgb"].copy()
    gt_part_crop = target_like.get("gt_part_crop")
    if gt_part_crop is None:
        gt_part_crop = np.zeros(crop_rgb.shape[:2], dtype=np.uint8)
    crop_overlay = overlay_part_mask(crop_rgb, gt_part_crop)
    panels = [add_title(crop_rgb, "crop-rgb"), add_title(crop_overlay, "crop-gt-seg")]
    if gt_pose is not None:
        gt_mesh_crop = renderer.render_pose_overlay(
            crop_rgb,
            gt_pose,
            target_like["K_crop"],
            alpha=float(args.overlay_alpha),
        )
        panels.append(add_title(gt_mesh_crop, "crop-gt-mesh-Kcrop"))
    if target_like.get("keypoints_crop") is not None:
        kp_panel = crop_rgb.copy()
        valid = target_like["keypoints_valid"]
        for i, xy in enumerate(target_like["keypoints_crop"]):
            if valid is not None and not bool(valid[i]):
                continue
            x, y = np.round(xy).astype(int)
            cv2.circle(kp_panel, (x, y), 4, (50, 240, 80), -1, lineType=cv2.LINE_AA)
            cv2.putText(kp_panel, str(i), (x + 5, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 240, 80), 1, cv2.LINE_AA)
        panels.append(add_title(kp_panel, "crop-gt-keypoints"))
    return concat_panels(panels)


def load_hcce_model(checkpoint_path, device):
    checkpoint_path = resolve_path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_args = ckpt.get("args") or {}
    if hasattr(ckpt_args, "__dict__"):
        ckpt_args = vars(ckpt_args)
    state = ckpt["model_state_dict"]
    num_keypoints = ckpt_args.get("num_keypoints")
    if num_keypoints is None:
        weight = state.get("keypoint_net.out_layer1.weight")
        num_keypoints = int(weight.shape[0]) if weight is not None else 5
    model = CropHCCEDenseKeypointDPT(
        img_size=int(ckpt_args.get("img_size", 224)),
        backbone=ckpt_args.get("backbone", "dinov2_vits14"),
        pretrained_backbone=False,
        hcce_feat_dim=int(ckpt_args.get("hcce_feat_dim", 256)),
        hcce_bits=int(ckpt_args.get("hcce_bits", 8)),
        num_keypoints=int(num_keypoints),
        action_dim=3,
        pose_head_iter=int(ckpt_args.get("pose_head_iter", 4)),
        pose_head_dropout=float(ckpt_args.get("pose_head_dropout", 0.3)),
    )
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    model_meta = dict(ckpt_args)
    model_meta.setdefault("hcce_bits", int(ckpt_args.get("hcce_bits", 8)))
    model_meta.setdefault("hcce_coord_min", float(ckpt_args.get("hcce_coord_min", -1.0)))
    model_meta.setdefault("hcce_coord_max", float(ckpt_args.get("hcce_coord_max", 1.0)))
    model_meta.setdefault("hcce_axis_scale", ckpt_args.get("hcce_axis_scale", (1.0, 1.0, 1.0)))
    model_meta.setdefault("num_keypoints", int(num_keypoints))
    return model, model_meta


def configure_device(device_name):
    if str(device_name).startswith("cuda"):
        idx = int(str(device_name).split(":", 1)[1]) if ":" in str(device_name) else 0
        os.environ["EGL_DEVICE_ID"] = str(idx)
        torch.cuda.set_device(idx)
    return torch.device(device_name if torch.cuda.is_available() else "cpu")


def build_needle_dataset(args):
    return RoboPEPPRARPInstrument(
        str(args.needle_dataset_root),
        str(args.needle_pose_root),
        split="test",
        training=False,
        crop_size=int(args.crop_size),
        train_ratio=float(args.train_ratio),
        subsample=1,
        min_dice=(args.min_dice_shaft, args.min_dice_wrist, args.min_dice_gripper),
        canonicalize_pose_symmetry=bool(args.canonicalize_pose_symmetry),
        canonical_eps=float(args.canonical_eps),
        cache_dir=str(args.dataset_cache_dir),
    )


def build_needle_items(args):
    dataset = build_needle_dataset(args)
    index = {sample_key(v, f, inst): i for i, (v, f, inst, _) in enumerate(dataset.samples)}
    if args.needle_manifest_csv and Path(args.needle_manifest_csv).is_file():
        rows = read_csv_rows(args.needle_manifest_csv)
        items = []
        missing = []
        for row in rows:
            key = sample_key(row["video"], row["frame_id"], row["instance_id"])
            idx = index.get(key)
            if idx is None:
                missing.append("|".join(map(str, key)))
                continue
            items.append(
                EvalItem(
                    dataset="needleGrasping",
                    video=key[0],
                    frame_id=key[1],
                    instance_id=key[2],
                    ordinal=len(items),
                    dataset_idx=int(idx),
                    source_sample_idx=str(row.get("sample_idx", "")),
                )
            )
    else:
        missing = []
        items = [
            EvalItem("needleGrasping", v, norm_frame_id(f), int(inst), ordinal=i, dataset_idx=i)
            for i, (v, f, inst, _) in enumerate(dataset.samples)
        ]
    if int(args.max_needle_samples) > 0:
        items = items[: int(args.max_needle_samples)]
        for i, item in enumerate(items):
            item.ordinal = i
    meta = {
        "dataset_len": len(dataset),
        "manifest_csv": str(args.needle_manifest_csv),
        "selected_items": len(items),
        "missing_manifest_items": len(missing),
        "missing_examples": missing[:10],
    }
    return items, meta


def build_suture_dataset_index(args):
    dataset = RARPInstanceDataset(
        split="test",
        training=False,
        img_size=int(args.panel_size),
        dataset_root=str(args.suture_dataset_root),
        pose_root=None,
        min_dice=[args.min_dice_shaft, args.min_dice_wrist, args.min_dice_gripper],
        train_ratio=float(args.train_ratio),
        subsample=1,
        v2_force=True,
        cse_coord_root=None,
        render_on_the_fly=False,
    )
    return dataset, {(video, norm_frame_id(frame_id)): idx for idx, (video, frame_id, _) in enumerate(dataset.samples)}


def build_suture_items(args):
    if not args.suture_manifest_csv or not Path(args.suture_manifest_csv).is_file():
        return [], {"selected_items": 0, "reason": "no suture manifest csv"}
    rows = read_csv_rows(args.suture_manifest_csv)
    videos = sorted({row["video"] for row in rows})
    if int(args.suture_num_videos) > 0:
        videos = videos[: int(args.suture_num_videos)]
    video_set = set(videos)
    frame_filter = None
    if getattr(args, "suture_frame_ids", None):
        frame_filter = {norm_frame_id(frame_id) for frame_id in args.suture_frame_ids}
    dataset, index = build_suture_dataset_index(args)
    items = []
    failures = []
    for row in rows:
        video = row["video"]
        frame_id = norm_frame_id(row["frame_id"])
        if video not in video_set:
            continue
        if int(args.suture_frame_stride) > 1 and int(frame_id) % int(args.suture_frame_stride) != 0:
            continue
        if frame_filter is not None and frame_id not in frame_filter:
            continue
        key = (video, frame_id)
        idx = index.get(key)
        if idx is None:
            failures.append({"video": video, "frame_id": frame_id, "reason": "frame not in RARPInstanceDataset"})
            continue
        try:
            _, annot = dataset[idx]
            ref_slot = int(row["instance_id"]) - 1
            instruments = annot["instruments"]
            if ref_slot < 0 or ref_slot >= len(instruments):
                raise RuntimeError(f"slot {ref_slot} out of {len(instruments)}")
            actual_instance_id = int(instruments[ref_slot]["instance_id"])
        except Exception as exc:
            failures.append({"video": video, "frame_id": frame_id, "reason": f"{type(exc).__name__}: {exc}"})
            continue
        items.append(
            EvalItem(
                dataset="suturePulling",
                video=video,
                frame_id=frame_id,
                instance_id=int(row["instance_id"]),
                ordinal=len(items),
                actual_instance_id=actual_instance_id,
                source_sample_idx=str(row.get("sample_idx", "")),
            )
        )
    if int(args.max_suture_samples) > 0:
        items = items[: int(args.max_suture_samples)]
        for i, item in enumerate(items):
            item.ordinal = i
    meta = {
        "manifest_csv": str(args.suture_manifest_csv),
        "manifest_rows": len(rows),
        "selected_videos": videos,
        "selected_items": len(items),
        "frame_stride": int(args.suture_frame_stride),
        "frame_filter": sorted(frame_filter) if frame_filter is not None else [],
        "mapping_failures": failures[:25],
        "mapping_failure_count": len(failures),
    }
    return items, meta


def load_frame_rgb(dataset_root, video, frame_id):
    video_folder = Path(dataset_root) / f"SARRARP502022_{video}"
    frames_folder = video_folder / "frames_v2" if (video_folder / "frames_v2").is_dir() else video_folder / "frames"
    for ext in ("png", "jpg"):
        path = frames_folder / f"{norm_frame_id(frame_id)}.{ext}"
        if path.is_file():
            return np.asarray(Image.open(path).convert("RGB")), path
    raise FileNotFoundError(f"Frame not found: {video}/{frame_id}")


def load_suture_part_mask(dataset_root, video, frame_id, actual_instance_id):
    video_folder = Path(dataset_root) / f"SARRARP502022_{video}"
    instance_folder = video_folder / f"instance{actual_instance_id}"
    mask_frame_id = f"{int(frame_id) - 1:05d}"
    part_mask = None
    for part_name, label in (("shaft", 3), ("wrist", 2), ("gripper", 1)):
        folder = _resolve_mask_subfolder(str(instance_folder), part_name, True)
        if folder is None:
            raise FileNotFoundError(f"{part_name} mask folder not found under {instance_folder}")
        path = Path(folder) / f"{mask_frame_id}.png"
        if not path.is_file():
            raise FileNotFoundError(path)
        mask = np.asarray(Image.open(path).convert("L")) > 0
        if part_mask is None:
            part_mask = np.zeros(mask.shape, dtype=np.uint8)
        part_mask[mask] = int(label)
    if part_mask is None or not np.any(part_mask > 0):
        raise RuntimeError(f"empty suture mask: {video}/{frame_id}/instance{actual_instance_id}")
    return part_mask


def crop_from_gt_mask(rgb, K_orig, gt_part_mask, crop_size):
    inst_mask = gt_part_mask > 0
    ys, xs = np.where(inst_mask)
    if len(xs) == 0:
        raise RuntimeError("Cannot crop an empty GT mask")
    h, w = rgb.shape[:2]
    bbox_min = np.array([float(xs.min()), float(ys.min())], dtype=np.float32)
    bbox_max = np.array([float(xs.max() + 1), float(ys.max() + 1)], dtype=np.float32)
    bbox_min = np.clip(bbox_min, [0.0, 0.0], [float(w - 1), float(h - 1)])
    bbox_max = np.clip(bbox_max, [1.0, 1.0], [float(w), float(h)])
    x0, y0 = bbox_min.astype(np.int64)
    x1, y1 = np.ceil(bbox_max).astype(np.int64)
    crop = rgb[y0:y1, x0:x1]
    if crop.shape[0] <= 0 or crop.shape[1] <= 0:
        raise RuntimeError(f"Empty crop from bbox {bbox_min} -> {bbox_max}")
    resized, (new_w, new_h) = resize_longer_side_array(crop, crop_size, cv2.INTER_LINEAR)
    pad_x = (int(crop_size) - new_w) // 2
    pad_y = (int(crop_size) - new_h) // 2
    crop_square = np.pad(
        resized,
        ((pad_y, int(crop_size) - new_h - pad_y), (pad_x, int(crop_size) - new_w - pad_x), (0, 0)),
        mode="edge",
    )
    scale_x = float(new_w) / float(bbox_max[0] - bbox_min[0])
    scale_y = float(new_h) / float(bbox_max[1] - bbox_min[1])
    K_crop = crop_resize_pad_intrinsics(K_orig, bbox_min, (scale_x, scale_y), (pad_x, pad_y))
    gt_part_crop = crop_resize_pad_map(gt_part_mask, bbox_min, bbox_max, crop_size, cv2.INTER_NEAREST, value=0).astype(np.uint8)
    return {
        "crop_rgb": crop_square.astype(np.uint8),
        "gt_part_crop": gt_part_crop,
        "K_crop": K_crop.astype(np.float32),
        "bbox_min": bbox_min.astype(np.float32),
        "bbox_max": bbox_max.astype(np.float32),
        "scale": np.array([scale_x, scale_y], dtype=np.float32),
        "pad": np.array([pad_x, pad_y], dtype=np.float32),
    }


def target_like_from_needle(target):
    bbox_min = target["bbox_min"].detach().cpu().numpy().astype(np.float32)
    bbox_max = target["bbox_max"].detach().cpu().numpy().astype(np.float32)
    gt_part_orig = target["part_mask_orig"].detach().cpu().numpy().astype(np.uint8)
    gt_part_crop = crop_resize_pad_map(gt_part_orig, bbox_min, bbox_max, 224, cv2.INTER_NEAREST, value=0).astype(np.uint8)
    return {
        "orig_rgb": target["orig_rgb"].detach().cpu().numpy().astype(np.uint8),
        "crop_rgb": target["crop_rgb"].detach().cpu().numpy().astype(np.uint8),
        "gt_part_orig": gt_part_orig,
        "gt_part_crop": gt_part_crop,
        "K_orig": target["K_orig"].detach().cpu().numpy().astype(np.float32),
        "K_crop": target["K"].detach().cpu().numpy().astype(np.float32),
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
        "scale": target["scale"].detach().cpu().numpy().astype(np.float32),
        "pad": target["pad"].detach().cpu().numpy().astype(np.float32),
        "keypoints_crop": target["keypoints_crop"].detach().cpu().numpy().astype(np.float32),
        "keypoints_orig": target["keypoints_orig"].detach().cpu().numpy().astype(np.float32),
        "keypoints_valid": target["keypoints_valid"].detach().cpu().numpy().astype(bool),
        "keypoints_valid_orig": target["keypoints_valid_orig"].detach().cpu().numpy().astype(bool),
    }


def target_like_from_suture(item, args):
    rgb, _ = load_frame_rgb(args.suture_dataset_root, item.video, item.frame_id)
    gt_part = load_suture_part_mask(args.suture_dataset_root, item.video, item.frame_id, item.actual_instance_id)
    K_orig = rarp_intrinsics(rgb.shape[1], rgb.shape[0])
    crop = crop_from_gt_mask(rgb, K_orig, gt_part, int(args.crop_size))
    return {
        "orig_rgb": rgb.astype(np.uint8),
        "crop_rgb": crop["crop_rgb"],
        "gt_part_orig": gt_part.astype(np.uint8),
        "gt_part_crop": crop["gt_part_crop"],
        "K_orig": K_orig.astype(np.float32),
        "K_crop": crop["K_crop"],
        "bbox_min": crop["bbox_min"],
        "bbox_max": crop["bbox_max"],
        "scale": crop["scale"],
        "pad": crop["pad"],
        "keypoints_crop": None,
        "keypoints_orig": None,
        "keypoints_valid": None,
        "keypoints_valid_orig": None,
    }


def tensor_to_input(crop_rgb, to_tensor, device):
    return to_tensor(Image.fromarray(crop_rgb.astype(np.uint8))).unsqueeze(0).to(device, non_blocking=True)


def evaluate_item(item, datasets, models, renderer, cad, to_tensor, device, args, rng):
    row = {
        "dataset": item.dataset,
        "video": item.video,
        "frame_id": norm_frame_id(item.frame_id),
        "instance_id": int(item.instance_id),
        "actual_instance_id": int(item.actual_instance_id),
        "ordinal": int(item.ordinal),
        "dataset_idx": int(item.dataset_idx),
        "source_sample_idx": item.source_sample_idx,
    }
    if item.dataset == "needleGrasping":
        image, target = datasets["needle"][item.dataset_idx]
        x = image.unsqueeze(0).to(device, non_blocking=True)
        target_like = target_like_from_needle(target)
        gt_pose = pose_from_target(target)
        row["has_pose_gt"] = 1
    elif item.dataset == "suturePulling":
        target_like = target_like_from_suture(item, args)
        x = tensor_to_input(target_like["crop_rgb"], to_tensor, device)
        gt_pose = None
        row["has_pose_gt"] = 0
    else:
        raise ValueError(item.dataset)

    K_crop_t = torch.from_numpy(target_like["K_crop"]).unsqueeze(0).to(device, non_blocking=True)
    with torch.inference_mode(), torch.amp.autocast(
        device_type="cuda",
        enabled=(device.type == "cuda"),
        dtype=torch.bfloat16,
    ):
        robo_out = None
        if not int(args.hcce_fit_only):
            robo_out = models["robopepp"](x, K_crop_t, masks_enc=None, masks_pred=None)
        hcce_out = models["hcce"](x, K_crop_t)

    poses = {}
    hm_cache = {}
    if not int(args.hcce_fit_only):
        try:
            robo_pnp, robo_hm, robo_scores, robo_direct = pnp_from_heatmap_output(
                robo_out, target_like["K_crop"], args.pnp_min_score
            )
            poses["robopepp_pnp"] = robo_pnp
            poses["robopepp_direct"] = robo_direct
            row["robopepp_pnp_status"] = "ok"
            row["robopepp_direct_status"] = "ok"
            row["robopepp_hm_score_mean"] = float(np.mean(robo_scores))
            row["robopepp_hm_score_min"] = float(np.min(robo_scores))
            hm_cache["robopepp"] = (robo_hm, robo_scores)
        except Exception as exc:
            poses["robopepp_pnp"] = None
            try:
                poses["robopepp_direct"] = pose_from_output(robo_out)
                row["robopepp_direct_status"] = "ok"
            except Exception:
                poses["robopepp_direct"] = None
                row["robopepp_direct_status"] = "error"
            row["robopepp_pnp_status"] = f"{type(exc).__name__}: {exc}"

        try:
            poses["hcce_direct"] = pose_from_output(hcce_out)
            row["hcce_direct_status"] = "ok"
        except Exception as exc:
            poses["hcce_direct"] = None
            row["hcce_direct_status"] = f"{type(exc).__name__}: {exc}"

    try:
        hcce_fit_pose, fit_extra = fit_pose_from_hcce(
            hcce_out,
            cad,
            models.get("hcce_meta", {}),
            target_like,
            args,
            rng,
        )
        poses["hcce_fit"] = hcce_fit_pose
        row["hcce_fit_status"] = "ok"
        row.update(fit_extra)
    except Exception as exc:
        poses["hcce_fit"] = None
        row["hcce_fit_status"] = f"{type(exc).__name__}: {exc}"
        if int(args.fail_fast):
            raise

    skip_hcce_kp_pnp = int(args.skip_hcce_kp_pnp) or int(models.get("hcce_meta", {}).get("num_keypoints", 5)) != len(KEYPOINT_NAMES)
    if not int(args.hcce_fit_only) and not skip_hcce_kp_pnp:
        try:
            hcce_pnp, hcce_hm, hcce_scores, _ = pnp_from_heatmap_output(
                hcce_out, target_like["K_crop"], args.pnp_min_score
            )
            poses["hcce_kp_pnp"] = hcce_pnp
            row["hcce_kp_pnp_status"] = "ok"
            row["hcce_kp_hm_score_mean"] = float(np.mean(hcce_scores))
            row["hcce_kp_hm_score_min"] = float(np.min(hcce_scores))
            hm_cache["hcce"] = (hcce_hm, hcce_scores)
        except Exception as exc:
            poses["hcce_kp_pnp"] = None
            row["hcce_kp_pnp_status"] = f"{type(exc).__name__}: {exc}"
    elif skip_hcce_kp_pnp:
        poses["hcce_kp_pnp"] = None
        row["hcce_kp_pnp_status"] = (
            f"skipped: hcce num_keypoints={int(models.get('hcce_meta', {}).get('num_keypoints', -1))} "
            f"does not match RARP geometry keypoints={len(KEYPOINT_NAMES)}"
        )

    if target_like.get("keypoints_crop") is not None:
        for prefix, cache_key in (("robopepp_pnp", "robopepp"), ("hcce_kp_pnp", "hcce")):
            if cache_key not in hm_cache:
                row[f"{prefix}_hm_crop_rmse_px"] = float("nan")
                continue
            hm, _ = hm_cache[cache_key]
            valid = target_like["keypoints_valid"]
            if valid is not None and np.any(valid):
                diff = hm[valid] - target_like["keypoints_crop"][valid]
                row[f"{prefix}_hm_crop_rmse_px"] = float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))
            else:
                row[f"{prefix}_hm_crop_rmse_px"] = float("nan")

    add_model_crop_segmentation(row, hcce_out, target_like, args)
    for prefix, pose in poses.items():
        add_pose_errors(row, pose, gt_pose, prefix)
        add_keypoint_reprojection(row, pose, target_like, prefix)
        render_pose_segmentation(row, renderer, pose, target_like, prefix, args)

    if int(args.vis_limit) != 0 and item.ordinal < int(args.vis_limit):
        canvas = make_concat_visual(target_like, poses, renderer, row, args, gt_pose=gt_pose)
        vis_path = Path(args.output_dir) / "vis" / item.dataset / f"{item.ordinal:05d}_{item.video}_{item.frame_id}_inst{item.instance_id}.jpg"
        vis_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(canvas).save(vis_path)
        row["vis_path"] = str(vis_path)
        if item.dataset == "needleGrasping" and int(args.crop_vis_limit) != 0 and item.ordinal < int(args.crop_vis_limit):
            crop_canvas = make_crop_gt_visual(target_like, renderer, gt_pose, args)
            crop_path = Path(args.output_dir) / "vis_crop_gt" / item.dataset / f"{item.ordinal:05d}_{item.video}_{item.frame_id}_inst{item.instance_id}.jpg"
            crop_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(crop_canvas).save(crop_path)
            row["crop_gt_vis_path"] = str(crop_path)
    else:
        row["vis_path"] = ""
    return row


def split_evenly(items, n):
    return [items[i::n] for i in range(n)]


def worker_main(rank, items_payload, args):
    items = [EvalItem.from_json(item) for item in items_payload]
    device = configure_device(args.devices[rank])
    print(f"[worker {rank}] device={device} items={len(items)}", flush=True)
    robo_model = None
    if not int(args.hcce_fit_only):
        robo_model, _ = load_robopepp_model(resolve_path(args.robopepp_checkpoint), device)
    hcce_model, hcce_meta = load_hcce_model(resolve_path(args.hcce_checkpoint), device)
    needs_renderer = bool(args.render_seg_metrics) or int(args.vis_limit) != 0 or int(args.crop_vis_limit) != 0
    renderer = GMSInstrumentTrimeshRenderer(device) if needs_renderer else None
    cad = InstrumentCAD(CAD_ROOT)
    to_tensor = tv_transforms.Compose(
        [
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    datasets = {}
    if any(item.dataset == "needleGrasping" for item in items):
        datasets["needle"] = build_needle_dataset(args)
    models = {"robopepp": robo_model, "hcce": hcce_model, "hcce_meta": hcce_meta}
    rng = np.random.default_rng(int(args.seed) + 1009 * int(rank))
    rows = []
    for local_i, item in enumerate(items):
        try:
            row = evaluate_item(item, datasets, models, renderer, cad, to_tensor, device, args, rng)
        except Exception as exc:
            row = item.to_json()
            row["status"] = f"{type(exc).__name__}: {exc}"
            print(f"[worker {rank}] ERROR {item.dataset}/{item.video}/{item.frame_id}/inst{item.instance_id}: {row['status']}", flush=True)
            if int(args.fail_fast):
                raise
        rows.append(row)
        if local_i == 0 or (local_i + 1) % int(args.print_freq) == 0 or local_i + 1 == len(items):
            print(f"[worker {rank}] {local_i + 1}/{len(items)}", flush=True)
    out = Path(args.output_dir) / "workers" / f"worker_{rank:02d}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, allow_nan=True), encoding="utf-8")


def run_workers(items, args):
    devices = list(args.devices)
    chunks = [chunk for chunk in split_evenly(items, len(devices)) if chunk]
    args.devices = devices[: len(chunks)]
    if len(chunks) == 1:
        worker_main(0, [item.to_json() for item in chunks[0]], args)
    else:
        ctx = mp.get_context("spawn")
        procs = []
        for rank, chunk in enumerate(chunks):
            proc = ctx.Process(target=worker_main, args=(rank, [item.to_json() for item in chunk], args))
            proc.start()
            procs.append(proc)
        failures = []
        for proc in procs:
            proc.join()
            if proc.exitcode != 0:
                failures.append(proc.exitcode)
        if failures:
            raise RuntimeError(f"worker failures: {failures}")
    rows = []
    for path in sorted((Path(args.output_dir) / "workers").glob("worker_*.json")):
        rows.extend(json.loads(path.read_text(encoding="utf-8")))
    rows.sort(key=lambda row: (str(row.get("dataset", "")), int(row.get("ordinal", 10**12))))
    return rows


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=str(ROBOPEPP_ROOT / "logs/crop_hcce_vs_robopepp_rarp_eval"))
    parser.add_argument("--robopepp_checkpoint", type=str, default=str(DEFAULT_ROBOPEPP_CKPT))
    parser.add_argument("--hcce_checkpoint", type=str, default=str(DEFAULT_HCCE_CKPT))
    parser.add_argument("--needle_manifest_csv", type=str, default=str(DEFAULT_NEEDLE_MANIFEST))
    parser.add_argument("--suture_manifest_csv", type=str, default=str(DEFAULT_SUTURE_MANIFEST))
    parser.add_argument("--needle_dataset_root", type=str, default=NEEDLE_DATASET_ROOT)
    parser.add_argument("--needle_pose_root", type=str, default=NEEDLE_POSE_ROOT)
    parser.add_argument("--suture_dataset_root", type=str, default=SUTURE_DATASET_ROOT)
    parser.add_argument("--dataset_cache_dir", type=str, default=str(ROBOPEPP_ROOT / "logs/crop_hcce_vs_robopepp_rarp_eval/dataset_cache"))
    parser.add_argument("--datasets", nargs="+", choices=["needleGrasping", "suturePulling"], default=["needleGrasping", "suturePulling"])
    parser.add_argument("--max_needle_samples", type=int, default=0)
    parser.add_argument("--max_suture_samples", type=int, default=0)
    parser.add_argument("--suture_num_videos", type=int, default=10)
    parser.add_argument("--suture_frame_stride", type=int, default=4)
    parser.add_argument("--suture_frame_ids", nargs="*", default=None)
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--panel_size", type=int, default=630)
    parser.add_argument("--train_ratio", type=float, default=0.95)
    parser.add_argument("--canonicalize_pose_symmetry", type=int, choices=[0, 1], default=1)
    parser.add_argument("--canonical_eps", type=float, default=0.08)
    parser.add_argument("--min_dice_shaft", type=float, default=0.8)
    parser.add_argument("--min_dice_wrist", type=float, default=0.6)
    parser.add_argument("--min_dice_gripper", type=float, default=0.6)
    parser.add_argument("--inst_thresh", type=float, default=0.5)
    parser.add_argument("--hcce_bits", type=int, default=8)
    parser.add_argument("--hcce_coord_min", type=float, default=-1.0)
    parser.add_argument("--hcce_coord_max", type=float, default=1.0)
    parser.add_argument("--hcce_bit_thresh", type=float, default=0.5)
    parser.add_argument("--max_points_per_part", type=int, default=1200)
    parser.add_argument("--point_select", choices=["random"], default="random")
    parser.add_argument("--fit_seg_source", choices=["pred", "gt"], default="pred")
    parser.add_argument("--surface_snap_method", choices=["surface", "vertex"], default="surface")
    parser.add_argument("--surface_k_faces", type=int, default=0)
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
    parser.add_argument("--freeze_wrist_after_pnp", type=int, choices=[0, 1], default=0)
    parser.add_argument("--optim_strategy", choices=["decoupled", "single"], default="decoupled")
    parser.add_argument("--optim_parts", choices=["wrist_gripper", "all"], default="wrist_gripper")
    parser.add_argument("--optim_loss", choices=["linear", "soft_l1", "huber", "cauchy", "arctan"], default="soft_l1")
    parser.add_argument("--optim_f_scale", type=float, default=8.0)
    parser.add_argument("--optim_max_nfev", type=int, default=200)
    parser.add_argument("--gripper_no_wrist_pose", type=int, choices=[0, 1], default=0)
    parser.add_argument("--min_depth", type=float, default=1e-4)
    parser.add_argument("--behind_camera_penalty", type=float, default=1e4)
    parser.add_argument("--render_min_depth", type=float, default=1e-4)
    parser.add_argument("--render_draw_margin", type=float, default=20.0)
    parser.add_argument("--render_seg_metrics", type=int, choices=[0, 1], default=1)
    parser.add_argument("--render_seg_metrics_limit", type=int, default=-1)
    parser.add_argument("--overlay_alpha", type=float, default=0.85)
    parser.add_argument("--vis_limit", type=int, default=160)
    parser.add_argument("--crop_vis_limit", type=int, default=40)
    parser.add_argument("--hcce_fit_only", type=int, choices=[0, 1], default=0)
    parser.add_argument("--skip_hcce_kp_pnp", type=int, choices=[0, 1], default=0)
    parser.add_argument("--print_freq", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fail_fast", type=int, choices=[0, 1], default=0)
    parser.add_argument("--devices", nargs="+", default=["cuda:0"])
    return parser


def main(args):
    args.output_dir = str(resolve_path(args.output_dir))
    args.robopepp_checkpoint = str(resolve_path(args.robopepp_checkpoint))
    args.hcce_checkpoint = str(resolve_path(args.hcce_checkpoint))
    args.needle_manifest_csv = str(resolve_path(args.needle_manifest_csv, MULTIHMR_ROOT))
    args.suture_manifest_csv = str(resolve_path(args.suture_manifest_csv, MULTIHMR_ROOT))
    args.dataset_cache_dir = str(resolve_path(args.dataset_cache_dir))
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print(f"[config] output_dir={args.output_dir}", flush=True)
    print(f"[config] robopepp_checkpoint={args.robopepp_checkpoint}", flush=True)
    print(f"[config] hcce_checkpoint={args.hcce_checkpoint}", flush=True)

    all_items = []
    selection_meta = {}
    if "needleGrasping" in args.datasets:
        needle_items, meta = build_needle_items(args)
        all_items.extend(needle_items)
        selection_meta["needleGrasping"] = meta
        print(f"[manifest] needleGrasping {json.dumps(meta, indent=2)}", flush=True)
    if "suturePulling" in args.datasets:
        suture_items, meta = build_suture_items(args)
        all_items.extend(suture_items)
        selection_meta["suturePulling"] = meta
        print(f"[manifest] suturePulling {json.dumps(meta, indent=2)}", flush=True)
    if not all_items:
        raise RuntimeError("No evaluation items selected")

    manifest_path = Path(args.output_dir) / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "items": [item.to_json() for item in all_items],
                "selection_meta": selection_meta,
                "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[manifest] wrote {manifest_path}", flush=True)

    rows = run_workers(all_items, args)
    all_csv = Path(args.output_dir) / "per_instance.csv"
    write_csv(all_csv, rows)
    by_dataset = {
        name: summarize_rows([row for row in rows if row.get("dataset") == name])
        for name in sorted({row.get("dataset", "") for row in rows if row.get("dataset", "")})
    }
    by_dataset["all"] = summarize_rows(rows)
    summary_path = Path(args.output_dir) / "summary.json"
    summary_path.write_text(json.dumps(by_dataset, indent=2, allow_nan=True), encoding="utf-8")
    write_summary_md(Path(args.output_dir) / "summary.md", by_dataset)
    print(f"[ok] wrote {all_csv}", flush=True)
    print(f"[ok] wrote {summary_path}", flush=True)
    for name, summary in by_dataset.items():
        print(
            f"[{name}] robopepp_pnp trans={fmt_stat(summary, 'robopepp_pnp_trans_err_m')} "
            f"rot={fmt_stat(summary, 'robopepp_pnp_rot_err_deg')} "
            f"joint={fmt_stat(summary, 'robopepp_pnp_joint_mae_deg')}",
            flush=True,
        )
        print(
            f"[{name}] hcce_direct trans={fmt_stat(summary, 'hcce_direct_trans_err_m')} "
            f"rot={fmt_stat(summary, 'hcce_direct_rot_err_deg')} "
            f"joint={fmt_stat(summary, 'hcce_direct_joint_mae_deg')}",
            flush=True,
        )
        print(
            f"[{name}] hcce_fit trans={fmt_stat(summary, 'hcce_fit_trans_err_m')} "
            f"rot={fmt_stat(summary, 'hcce_fit_rot_err_deg')} "
            f"joint={fmt_stat(summary, 'hcce_fit_joint_mae_deg')}",
            flush=True,
        )
        print(
            f"[{name}] hcce_kp_pnp trans={fmt_stat(summary, 'hcce_kp_pnp_trans_err_m')} "
            f"rot={fmt_stat(summary, 'hcce_kp_pnp_rot_err_deg')} "
            f"joint={fmt_stat(summary, 'hcce_kp_pnp_joint_mae_deg')}",
            flush=True,
        )


if __name__ == "__main__":
    main(build_parser().parse_args())
