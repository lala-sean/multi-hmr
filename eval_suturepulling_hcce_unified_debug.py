import argparse
import csv
import json
import math
import os
import random
import re
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.spatial import cKDTree
from trimesh.triangles import closest_point as closest_points_on_triangles

from datasets.RarpInstanceDataset import RARPInstanceDataset
from datasets.rarp_hcce_pose_dataset import collate_fn_instrument_hcce_pose
from datasets.rarp_pose_canonicalization import quat_to_matrix
from estimate_rarp_cse_articulate_pose import (
    CAD_ROOT,
    Correspondences,
    InstrumentCAD,
    PART_LABELS,
    REPO_ROOT,
    _homogeneous_transform,
    _paint_projected_triangles,
    _project,
    _render_trimesh_panel,
    image_geometry,
    match_gt_to_predictions,
    optimize_pose,
    quarter_pixels_to_original,
    sample_mask_pixels,
    solve_wrist_pnp,
)
from estimate_rarp_hcce_articulate_pose import decode_hcce_logits
from multi_instrument.multi_instrument_hcce_densepart_dpt import (
    MultiInstrumentHCCEDensePartDPT,
)
from multi_instrument.multi_instrument_hcce_densepart_keypoint_dpt import (
    MultiInstrumentHCCEDensePartKeypointDPT,
)
from multi_instrument.multi_instrument_hcce_pose_dpt import MultiInstrumentHCCEPoseDPT


SUTURE_PULLING_ROOT = "/mnt/nas/share/shuojue/data/suturePulling_videos"
PART_COLORS = {
    PART_LABELS["gripper"]: (70, 150, 255),
    PART_LABELS["wrist"]: (40, 210, 120),
    PART_LABELS["shaft"]: (235, 70, 60),
}
DENSE_PART_TO_DATASET = np.array(
    [PART_LABELS["wrist"], PART_LABELS["gripper"], PART_LABELS["shaft"]],
    dtype=np.uint8,
)


def _namespace_to_dict(value):
    if value is None:
        return {}
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    if isinstance(value, dict):
        return dict(value)
    raise TypeError(f"Unsupported checkpoint args type: {type(value)}")


def _parse_axis_scale(value):
    if value is None:
        return np.array([1.0, 1.0, 1.0], dtype=np.float64)
    if isinstance(value, str):
        parts = [float(x.strip()) for x in value.split(",") if x.strip()]
    elif isinstance(value, (list, tuple, np.ndarray)):
        parts = [float(x) for x in value]
    else:
        parts = [float(value), float(value), float(value)]
    if len(parts) != 3:
        raise ValueError(f"hcce_axis_scale must have 3 values, got {value!r}")
    return np.asarray(parts, dtype=np.float64)


def _model_args_from_ckpt(ckpt_args):
    src = _namespace_to_dict(ckpt_args)
    defaults = dict(
        img_size=630,
        backbone="dinov2_vits14",
        pretrained_backbone=False,
        xat_dim=512,
        xat_depth=4,
        xat_heads=16,
        xat_dim_head=32,
        xat_mlp_dim=2048,
        xat_dropout=0.0,
        mask_dim=256,
        num_parts=4,
        hcce_feat_dim=256,
        hcce_bits=8,
        action_dim=3,
        use_pose_heads=0,
        use_depth_head=0,
        num_keypoints=7,
        pose_head_iter=4,
        pose_head_dropout=0.3,
    )
    for key in list(defaults):
        if key in src:
            defaults[key] = src[key]
    defaults["pretrained_backbone"] = False
    coord_args = {
        "hcce_coord_min": float(src.get("hcce_coord_min", -1.0)),
        "hcce_coord_max": float(src.get("hcce_coord_max", 1.0)),
        "hcce_axis_scale": _parse_axis_scale(src.get("hcce_axis_scale", (1.0, 1.0, 1.0))),
    }
    return defaults, coord_args, src


def load_model_unified(checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"]
    model_args, coord_args, raw_args = _model_args_from_ckpt(ckpt.get("args"))
    is_densepart = any(k.startswith("dense_dpt_head.") for k in state)
    has_keypoint = any(
        k.startswith("keypoint_head.") or k.startswith("keypoint_heatmap_head.")
        for k in state
    )
    has_depth = any(k.startswith("depth_dpt_head.") for k in state)
    if has_keypoint:
        model_args["use_depth_head"] = int(has_depth)
        model_cls = MultiInstrumentHCCEDensePartKeypointDPT
        variant = "densepart_keypoint_depth_h2" if has_depth else "densepart_keypoint_h2"
    elif is_densepart:
        model_cls = MultiInstrumentHCCEDensePartDPT
        variant = "densepart_h2"
    else:
        model_cls = MultiInstrumentHCCEPoseDPT
        variant = "pose_dpt_h4"
    model = model_cls(**model_args).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, {
        **model_args,
        **coord_args,
        "variant": variant,
        "checkpoint_iter": int(ckpt.get("iter", -1)),
        "checkpoint_epoch": int(ckpt.get("epoch", -1)),
        "raw_args": raw_args,
    }


def build_suture_dataset(args):
    return RARPInstanceDataset(
        split="test",
        training=False,
        img_size=args.img_size,
        dataset_root=SUTURE_PULLING_ROOT,
        pose_root=None,
        min_dice=[args.min_dice_shaft, args.min_dice_wrist, args.min_dice_gripper],
        train_ratio=args.train_ratio,
        subsample=1,
        v2_force=True,
        cse_coord_root=None,
        render_on_the_fly=False,
        canonicalize_pose_symmetry=bool(args.canonicalize_pose_symmetry),
        canonical_eps=args.canonical_eps,
    )


def select_video_names(dataset, args):
    if getattr(args, "reference_samples", None):
        videos = sorted({key[0] for key in args.reference_samples})
        missing = sorted(set(videos) - {sample[0] for sample in dataset.samples})
        if missing:
            raise ValueError(f"Reference videos are not in test split: {missing}")
        return videos
    videos = sorted({sample[0] for sample in dataset.samples})
    if args.video_names:
        missing = [v for v in args.video_names if v not in videos]
        if missing:
            raise ValueError(f"Requested videos are not in test split: {missing}")
        return list(args.video_names)
    rng = random.Random(args.seed)
    return rng.sample(videos, min(args.num_videos, len(videos)))


def select_sample_indices(dataset, video_name, stride, max_frames):
    idxs = [i for i, sample in enumerate(dataset.samples) if sample[0] == video_name]
    idxs.sort(key=lambda i: int(dataset.samples[i][1]))
    if stride > 1:
        idxs = [
            i for i in idxs
            if int(dataset.samples[i][1]) % int(stride) == 0
        ]
    if max_frames > 0:
        idxs = idxs[: int(max_frames)]
    return idxs


def load_reference_samples(vis_dir):
    if not vis_dir:
        return None
    vis_dir = Path(vis_dir)
    if not vis_dir.is_dir():
        raise FileNotFoundError(f"Reference vis dir not found: {vis_dir}")
    samples = set()
    pattern = re.compile(r"^(suturePulling_\d+_video\d+)_(\d+)_inst(\d+)\.jpg$")
    for path in vis_dir.glob("*.jpg"):
        match = pattern.match(path.name)
        if not match:
            continue
        video, frame_id, inst_id = match.groups()
        samples.add((video, frame_id, int(inst_id)))
    if not samples:
        raise ValueError(f"No reference sample jpgs parsed from {vis_dir}")
    return samples


def reference_frames_for_video(reference_samples, video_name):
    if not reference_samples:
        return None
    return {frame_id for video, frame_id, _ in reference_samples if video == video_name}


def reference_instances_for_frame(reference_samples, video_name, frame_id):
    if not reference_samples:
        return None
    return {
        inst_id
        for video, ref_frame_id, inst_id in reference_samples
        if video == video_name and ref_frame_id == frame_id
    }


def pad_rgb_to_square(rgb, img_size, scale, pad_x, pad_y):
    h, w = rgb.shape[:2]
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = Image.fromarray(rgb).resize((new_w, new_h), Image.BILINEAR)
    padded = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = np.asarray(resized, dtype=np.uint8)
    return padded


def colorize_part_mask(mask):
    out = np.zeros(mask.shape + (3,), dtype=np.uint8)
    for label, color in PART_COLORS.items():
        out[mask == label] = color
    return out


def overlay_part_mask(rgb, mask, alpha=0.55):
    color = colorize_part_mask(mask)
    blended = cv2.addWeighted(rgb, 1.0, color, alpha, 0.0)
    return np.where((mask > 0)[..., None], blended, rgb).astype(np.uint8)


def add_title(panel, title):
    out = panel.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(
        out,
        title,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return out


def tensor_to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def quat_to_rvec(quat):
    quat_t = torch.as_tensor(tensor_to_numpy(quat), dtype=torch.float32).reshape(4)
    rot_mat = quat_to_matrix(quat_t).detach().cpu().numpy().astype(np.float64)
    rvec, _ = cv2.Rodrigues(rot_mat)
    return rvec.reshape(3), rot_mat


def pose_head_params_or_none(pred):
    needed = ("action_pred", "wrist_quat_pred", "wrist_trans_pred")
    if any(key not in pred for key in needed):
        return None
    rvec, _ = quat_to_rvec(pred["wrist_quat_pred"])
    trans = tensor_to_numpy(pred["wrist_trans_pred"]).astype(np.float64).reshape(3)
    action = tensor_to_numpy(pred["action_pred"]).astype(np.float64).reshape(3)
    return np.concatenate([rvec, trans, action], axis=0).astype(np.float64)


def tensor_sigmoid_np(x):
    return torch.sigmoid(x.detach()).cpu().numpy()


def select_mask_pixels(mask, max_points, rng, scores=None, mode="random"):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return xs, ys
    max_points = int(max_points)
    if max_points > 0 and len(xs) > max_points:
        if mode == "top" and scores is not None:
            vals = np.asarray(scores)[ys, xs]
            keep = np.argpartition(-vals, max_points - 1)[:max_points]
            keep = keep[np.argsort(-vals[keep])]
        elif mode == "random":
            keep = rng.choice(len(xs), size=max_points, replace=False)
        else:
            raise ValueError(f"Unsupported point_select={mode!r}")
        xs = xs[keep]
        ys = ys[keep]
    return xs, ys


def resize_float_mask(mask, size_hw):
    tensor = torch.from_numpy(mask).float()[None, None]
    out = F.interpolate(tensor, size=size_hw, mode="bilinear", align_corners=False)
    return out[0, 0].numpy()


def pred_part_label(pred, args):
    inst_prob = tensor_sigmoid_np(pred["inst_mask_logits"])
    logits = pred["part_mask_logits"].detach().cpu()
    part_idx = torch.argmax(logits, dim=0).numpy().astype(np.int64)

    if logits.shape[0] == 3:
        part_label = DENSE_PART_TO_DATASET[np.clip(part_idx, 0, 2)]
    else:
        part_label = part_idx.astype(np.uint8)

    target_hw = part_label.shape
    if inst_prob.shape != target_hw:
        inst_prob = resize_float_mask(inst_prob, target_hw)
    inst = inst_prob >= args.inst_thresh
    return np.where(inst, part_label, 0).astype(np.uint8), inst_prob


def pred_part_label_conf(pred, args):
    part_label, inst_prob = pred_part_label(pred, args)
    logits = pred["part_mask_logits"].detach().cpu()
    part_prob = torch.softmax(logits.float(), dim=0).max(dim=0).values.numpy()
    if part_prob.shape != part_label.shape:
        part_prob = resize_float_mask(part_prob, part_label.shape)
    return part_label, inst_prob, part_prob


def _canon_norm_to_part(cad, part_name, xyz_norm):
    xyz_world = np.asarray(xyz_norm, dtype=np.float64) * float(cad.canon_scale)
    return _homogeneous_transform(xyz_world, cad.canon_to_part[part_name])


def _surface_cache_for_part(cad, part_name):
    cache = getattr(cad, "_surface_closest_cache", None)
    if cache is None:
        cache = {}
        cad._surface_closest_cache = cache
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
    surface = _surface_cache_for_part(cad, part_name)
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
            dist2 = np.sum((close - flat_pts) ** 2, axis=1).reshape(
                len(pts), len(triangles_norm)
            )
            best_idx = np.argmin(dist2, axis=1)
            best[start:start + len(pts)] = close.reshape(
                len(pts), len(triangles_norm), 3
            )[np.arange(len(pts)), best_idx]
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
        dist2 = np.sum((close - flat_pts) ** 2, axis=1).reshape(
            len(xyz_norm), cand_idx.shape[1]
        )
        best_local = np.argmin(dist2, axis=1)
        best = close.reshape(len(xyz_norm), cand_idx.shape[1], 3)[
            np.arange(len(xyz_norm)), best_local
        ]
    return _canon_norm_to_part(cad, part_name, best).astype(np.float64)


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


def decoded_xyz_np(pred, model_meta, args):
    return decode_hcce_logits(
        pred["hcce_logits"],
        bits=int(model_meta["hcce_bits"] if args.hcce_bits is None else args.hcce_bits),
        coord_min=float(model_meta["hcce_coord_min"]),
        coord_max=float(model_meta["hcce_coord_max"]),
        threshold=float(args.hcce_bit_thresh),
    ).detach().cpu().numpy()


def build_correspondences_unified(pred, cad, geom, model_meta, args, rng):
    rgb, _, scale, pad_x, pad_y = geom
    part_label, inst_prob, part_conf = pred_part_label_conf(pred, args)
    xyz = decoded_xyz_np(pred, model_meta, args)
    if part_label.shape != xyz.shape[:2]:
        part_label = np.asarray(
            Image.fromarray(part_label).resize((xyz.shape[1], xyz.shape[0]), Image.NEAREST),
            dtype=np.uint8,
        )
        inst_prob = resize_float_mask(inst_prob, xyz.shape[:2])
        part_conf = resize_float_mask(part_conf, xyz.shape[:2])

    axis_scale = np.asarray(model_meta.get("hcce_axis_scale", (1.0, 1.0, 1.0)), dtype=np.float64)
    all_uv, all_points, all_names = [], [], []
    wrist_uv, wrist_points = [], []
    debug = {}

    for part_name, label in (
        ("shaft", PART_LABELS["shaft"]),
        ("wrist", PART_LABELS["wrist"]),
        ("gripper", PART_LABELS["gripper"]),
    ):
        mask = (inst_prob >= args.inst_thresh) & (part_label == label)
        if part_name == "shaft" and args.shaft_raw_x_min is not None:
            mask = (
                mask
                & np.isfinite(xyz[:, :, 0])
                & (xyz[:, :, 0] > float(args.shaft_raw_x_min))
            )
        xs, ys = select_mask_pixels(
            mask,
            int(args.max_points_per_part),
            rng,
            scores=inst_prob * part_conf,
            mode=getattr(args, "point_select", "random"),
        )
        if len(xs) == 0:
            debug[f"{part_name}_raw_points"] = 0
            continue
        uv = quarter_pixels_to_original(
            xs,
            ys,
            xyz.shape[0],
            args.img_size,
            scale,
            pad_x,
            pad_y,
        )
        valid_uv = (
            (uv[:, 0] >= 0.0)
            & (uv[:, 0] < rgb.shape[1])
            & (uv[:, 1] >= 0.0)
            & (uv[:, 1] < rgb.shape[0])
        )
        xs, ys, uv = xs[valid_uv], ys[valid_uv], uv[valid_uv]
        xyz_norm = xyz[ys, xs].astype(np.float64)
        finite = np.isfinite(xyz_norm).all(axis=1)
        xs, ys, uv, xyz_norm = xs[finite], ys[finite], uv[finite], xyz_norm[finite]
        if len(xyz_norm) == 0:
            debug[f"{part_name}_raw_points"] = 0
            continue

        xyz_cad_norm = xyz_norm / axis_scale.reshape(1, 3)
        if part_name == "gripper":
            points, names = closest_gripper_surface_points(
                cad, xyz_cad_norm, k_faces=args.surface_k_faces
            )
        else:
            points = closest_part_surface_points(
                cad, part_name, xyz_cad_norm, k_faces=args.surface_k_faces
            )
            names = np.full((len(points),), part_name, dtype=object)

        all_uv.append(uv)
        all_points.append(points)
        all_names.append(names)
        debug[f"{part_name}_raw_points"] = int(len(uv))
        if part_name == "wrist":
            wrist_uv.append(uv)
            wrist_points.append(points)

    if not wrist_uv:
        raise RuntimeError("No wrist HCCE correspondences")
    wrist_uv = np.concatenate(wrist_uv, axis=0).astype(np.float64)
    wrist_points = np.concatenate(wrist_points, axis=0).astype(np.float64)
    if len(wrist_uv) < args.min_wrist_points:
        raise RuntimeError(f"Not enough wrist points: {len(wrist_uv)} < {args.min_wrist_points}")
    if not all_uv:
        raise RuntimeError("No HCCE correspondences")

    corr = Correspondences(
        uv=np.concatenate(all_uv, axis=0).astype(np.float64),
        points_part=np.concatenate(all_points, axis=0).astype(np.float64),
        part_names=np.concatenate(all_names, axis=0),
    )
    if len(corr) < args.min_total_points:
        raise RuntimeError(f"Not enough correspondences: {len(corr)} < {args.min_total_points}")
    debug["total_points"] = int(len(corr))
    return corr, wrist_uv, wrist_points, part_label, inst_prob, xyz, debug


def estimate_pose(pred, cad, geom, model_meta, args, rng):
    corr, wrist_uv, wrist_points, part_label, inst_prob, xyz, debug = build_correspondences_unified(
        pred, cad, geom, model_meta, args, rng
    )
    rvec0, trans0, inliers = solve_wrist_pnp(wrist_uv, wrist_points, geom[1], args)
    opt_result = optimize_pose(cad, corr, rvec0, trans0, geom[1], args)
    pose = opt_result[0]
    rmse_all = opt_result[1] if len(opt_result) > 1 else float("nan")
    nfev = opt_result[2] if len(opt_result) > 2 else -1
    rmse_wg = opt_result[3] if len(opt_result) > 3 else float("nan")
    rmse_shaft = opt_result[4] if len(opt_result) > 4 else float("nan")
    debug.update(
        {
            "hcce_status": "ok",
            "pnp_inliers": int(inliers),
            "opt_nfev": int(nfev),
            "opt_rmse_all": float(rmse_all),
            "opt_rmse_wrist_gripper": float(rmse_wg),
            "opt_rmse_shaft": float(rmse_shaft),
        }
    )
    return pose, (part_label, inst_prob, xyz), debug


def render_mesh_overlay_direct(rgb, cad, pose, K, args):
    if pose is None:
        return rgb.copy()

    rvec = pose[:3]
    trans = pose[3:6]
    alpha, theta_l, theta_r = pose[6], pose[7], pose[8]
    posed_vertices = cad.posed_mesh_vertices(rvec, trans, alpha, theta_l, theta_r)

    h, w = rgb.shape[:2]
    if args.mesh_render_backend == "pyrender":
        mesh_panel, support = _render_trimesh_panel(
            cad, posed_vertices, K, h, w, args, return_support=True
        )
    else:
        args.render_h, args.render_w = h, w
        mesh_panel = _paint_projected_triangles(
            cad, posed_vertices, K, args, shade_mesh=True
        )
        support = np.any(mesh_panel > 0, axis=2)

    out = rgb.copy()
    if np.any(support):
        out[support] = mesh_panel[support]
    edge_px = int(getattr(args, "mesh_overlay_edge_px", 0))
    if edge_px > 0 and np.any(support):
        kernel = np.ones((3, 3), dtype=np.uint8)
        edge = cv2.morphologyEx(support.astype(np.uint8), cv2.MORPH_GRADIENT, kernel)
        edge = cv2.dilate(edge, kernel, iterations=edge_px).astype(bool)
        out[edge] = (255, 255, 255)
    return out


def part_label_from_name(part_name):
    if part_name == "shaft":
        return PART_LABELS["shaft"]
    if part_name == "wrist":
        return PART_LABELS["wrist"]
    return PART_LABELS["gripper"]


def get_surface_render_points(cad, args):
    key = (
        int(args.render_surface_points_shaft),
        int(args.render_surface_points_wrist),
        int(args.render_surface_points_gripper),
    )
    if getattr(cad, "_unified_surface_render_key", None) == key:
        return cad._unified_surface_render_points

    counts = {
        "shaft": int(args.render_surface_points_shaft),
        "wrist": int(args.render_surface_points_wrist),
        "l_gripper": int(args.render_surface_points_gripper),
        "r_gripper": int(args.render_surface_points_gripper),
    }
    points = {}
    for part_name, count in counts.items():
        if count <= 0:
            points[part_name] = cad.vertices_part[part_name]
        else:
            points[part_name] = cad.meshes[part_name].sample(count).astype(np.float64)
    cad._unified_surface_render_key = key
    cad._unified_surface_render_points = points
    return points


def splat_radius_for_part(part_name, args):
    if part_name == "shaft":
        return int(args.render_shaft_splat_radius_px)
    return int(args.render_splat_radius_px)


def rasterize_pose_surface_splat(cad, pose, K, height, width, args):
    if pose is None or not np.isfinite(pose).all():
        return np.zeros((height, width), dtype=np.uint8)
    transforms = cad.fk(pose[:3], pose[3:6], pose[6], pose[7], pose[8])
    label_mask = np.zeros((height, width), dtype=np.uint8)
    projected = []
    for part_name, points_part in get_surface_render_points(cad, args).items():
        points_cam = cad.transform_points(part_name, points_part, transforms)
        valid_z = points_cam[:, 2] > args.min_depth
        if not np.any(valid_z):
            continue
        uv = _project(points_cam[valid_z], K)
        depth = points_cam[valid_z, 2]
        in_frame = (
            (uv[:, 0] >= -args.draw_margin)
            & (uv[:, 0] < width + args.draw_margin)
            & (uv[:, 1] >= -args.draw_margin)
            & (uv[:, 1] < height + args.draw_margin)
        )
        if not np.any(in_frame):
            continue
        projected.append(
            (
                depth[in_frame],
                uv[in_frame],
                part_label_from_name(part_name),
                splat_radius_for_part(part_name, args),
            )
        )
    if not projected:
        return label_mask

    depths = np.concatenate([item[0] for item in projected], axis=0)
    uvs = np.concatenate([item[1] for item in projected], axis=0)
    labels = np.concatenate(
        [np.full((len(item[0]),), item[2], dtype=np.uint8) for item in projected],
        axis=0,
    )
    radii = np.concatenate(
        [np.full((len(item[0]),), item[3], dtype=np.int32) for item in projected],
        axis=0,
    )
    order = np.argsort(depths)[::-1]
    for idx in order:
        x = int(round(float(uvs[idx, 0])))
        y = int(round(float(uvs[idx, 1])))
        radius = int(radii[idx])
        if radius <= 0:
            if 0 <= x < width and 0 <= y < height:
                label_mask[y, x] = labels[idx]
        else:
            cv2.circle(label_mask, (x, y), radius, int(labels[idx]), -1, cv2.LINE_8)
    return label_mask


def rasterize_pose_triangles(cad, pose, K, height, width, args):
    if pose is None or not np.isfinite(pose).all():
        return np.zeros((height, width), dtype=np.uint8)
    posed_vertices = cad.posed_mesh_vertices(pose[:3], pose[3:6], pose[6], pose[7], pose[8])
    label_mask = np.zeros((height, width), dtype=np.uint8)
    triangles = []
    for part_name, vertices_cam in posed_vertices.items():
        label = part_label_from_name(part_name)
        faces = cad.faces[part_name]
        uv = np.zeros((vertices_cam.shape[0], 2), dtype=np.float64)
        valid_z = vertices_cam[:, 2] > args.min_depth
        uv[valid_z] = _project(vertices_cam[valid_z], K)
        for face in faces:
            if not np.all(valid_z[face]):
                continue
            pts = uv[face]
            if (
                pts[:, 0].max() < -args.draw_margin
                or pts[:, 1].max() < -args.draw_margin
                or pts[:, 0].min() >= width + args.draw_margin
                or pts[:, 1].min() >= height + args.draw_margin
            ):
                continue
            triangles.append((float(vertices_cam[face, 2].mean()), pts, label))
    for _, pts, label in sorted(triangles, key=lambda item: item[0], reverse=True):
        cv2.fillConvexPoly(label_mask, np.round(pts).astype(np.int32), int(label), cv2.LINE_8)
    return label_mask


def rasterize_pose_part_mask(cad, pose, K, height, width, args):
    if args.render_seg_mode == "surface_splat":
        label_mask = rasterize_pose_surface_splat(cad, pose, K, height, width, args)
    elif args.render_seg_mode == "hybrid":
        label_mask = rasterize_pose_triangles(cad, pose, K, height, width, args)
        splat_mask = rasterize_pose_surface_splat(cad, pose, K, height, width, args)
        fill = (label_mask == 0) & (splat_mask > 0)
        label_mask[fill] = splat_mask[fill]
    elif args.render_seg_mode == "triangles":
        label_mask = rasterize_pose_triangles(cad, pose, K, height, width, args)
    else:
        raise ValueError(f"Unsupported render_seg_mode={args.render_seg_mode!r}")

    if args.render_shaft_close_px > 0:
        k = int(args.render_shaft_close_px) * 2 + 1
        kernel = np.ones((k, k), dtype=np.uint8)
        shaft = (label_mask == PART_LABELS["shaft"]).astype(np.uint8)
        shaft = cv2.morphologyEx(shaft, cv2.MORPH_CLOSE, kernel, iterations=1).astype(bool)
        label_mask[(label_mask == 0) & shaft] = PART_LABELS["shaft"]

    if args.render_shaft_dilate_px > 0:
        k = int(args.render_shaft_dilate_px) * 2 + 1
        kernel = np.ones((k, k), dtype=np.uint8)
        shaft = cv2.dilate(
            (label_mask == PART_LABELS["shaft"]).astype(np.uint8),
            kernel,
            iterations=1,
        ).astype(bool)
        label_mask[(label_mask == 0) & shaft] = PART_LABELS["shaft"]
    return label_mask


def original_mask_to_quarter(part_mask, img_size, scale, pad_x, pad_y):
    h, w = part_mask.shape[:2]
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = Image.fromarray(part_mask.astype(np.uint8)).resize((new_w, new_h), Image.NEAREST)
    padded = np.zeros((img_size, img_size), dtype=np.uint8)
    padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = np.asarray(resized, dtype=np.uint8)
    quarter_size = img_size // 4
    return np.asarray(
        Image.fromarray(padded).resize((quarter_size, quarter_size), Image.NEAREST),
        dtype=np.uint8,
    )


def render_pose_quarter_mask(cad, pose, geom, args):
    if pose is None or not np.isfinite(pose).all():
        return None
    rgb, K, scale, pad_x, pad_y = geom
    part_original = rasterize_pose_part_mask(cad, pose, K, rgb.shape[0], rgb.shape[1], args)
    return original_mask_to_quarter(part_original, args.img_size, scale, pad_x, pad_y)


def upsample_part_mask(mask, img_size):
    if mask is None:
        return np.zeros((img_size, img_size), dtype=np.uint8)
    return np.asarray(
        Image.fromarray(mask.astype(np.uint8)).resize((img_size, img_size), Image.NEAREST),
        dtype=np.uint8,
    )


KEYPOINT_COLORS = (
    (255, 0, 255),
    (0, 255, 255),
    (255, 128, 0),
    (0, 255, 0),
    (255, 0, 0),
    (0, 128, 255),
    (180, 80, 255),
)


def draw_keypoint_marker(panel, pred, color=None):
    out = panel.copy()
    if pred is None:
        return out
    keypoint = pred.get("keypoint_xy")
    if keypoint is None:
        return out
    if torch.is_tensor(keypoint):
        keypoint = keypoint.detach().cpu().numpy()
    h, w = out.shape[:2]
    keypoint = np.asarray(keypoint, dtype=np.float64).reshape(-1, 2)
    for j, xy in enumerate(keypoint):
        if not np.isfinite(xy).all():
            continue
        x, y = int(round(float(xy[0]))), int(round(float(xy[1])))
        if not (0 <= x < w and 0 <= y < h):
            continue
        c = KEYPOINT_COLORS[j % len(KEYPOINT_COLORS)] if color is None else color
        cv2.circle(out, (x, y), 5, c, 2, cv2.LINE_AA)
        cv2.drawMarker(
            out,
            (x, y),
            c,
            markerType=cv2.MARKER_CROSS,
            markerSize=14,
            thickness=2,
            line_type=cv2.LINE_AA,
        )
    return out


def save_visualization(path, geom, cad, hcce_pose, posehead_pose, pred_part, args, pred=None):
    rgb, K, scale, pad_x, pad_y = geom
    rgb_sq = pad_rgb_to_square(rgb, args.img_size, scale, pad_x, pad_y)
    pred_part_sq = np.asarray(
        Image.fromarray(pred_part).resize((args.img_size, args.img_size), Image.NEAREST),
        dtype=np.uint8,
    )
    hcce_mesh_sq = pad_rgb_to_square(
        render_mesh_overlay_direct(rgb, cad, hcce_pose, K, args),
        args.img_size,
        scale,
        pad_x,
        pad_y,
    )
    posehead_mesh_sq = pad_rgb_to_square(
        render_mesh_overlay_direct(rgb, cad, posehead_pose, K, args),
        args.img_size,
        scale,
        pad_x,
        pad_y,
    )
    hcce_render_mask = upsample_part_mask(
        render_pose_quarter_mask(cad, hcce_pose, geom, args), args.img_size
    )
    posehead_render_mask = upsample_part_mask(
        render_pose_quarter_mask(cad, posehead_pose, geom, args), args.img_size
    )
    rgb_sq = draw_keypoint_marker(rgb_sq, pred)
    hcce_mesh_sq = draw_keypoint_marker(hcce_mesh_sq, pred)
    posehead_mesh_sq = draw_keypoint_marker(posehead_mesh_sq, pred)
    model_seg_sq = draw_keypoint_marker(overlay_part_mask(rgb_sq, pred_part_sq), pred)
    hcce_proj_sq = draw_keypoint_marker(overlay_part_mask(rgb_sq, hcce_render_mask), pred)
    posehead_proj_sq = draw_keypoint_marker(overlay_part_mask(rgb_sq, posehead_render_mask), pred)
    panels = [
        add_title(rgb_sq, "rgb"),
        add_title(hcce_mesh_sq, "hcce-trimesh"),
        add_title(posehead_mesh_sq, "posehead-trimesh"),
        add_title(model_seg_sq, "model-seg"),
        add_title(hcce_proj_sq, "hcce-proj-seg"),
        add_title(posehead_proj_sq, "posehead-proj-seg"),
    ]
    sep = np.full((args.img_size, 6, 3), 255, dtype=np.uint8)
    canvas = panels[0]
    for panel in panels[1:]:
        canvas = np.concatenate([canvas, sep, panel], axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(path)


def solve_part_pnp(points, uv, K, args):
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        points.astype(np.float32),
        uv.astype(np.float32),
        K.astype(np.float32),
        None,
        iterationsCount=args.pnp_iters,
        reprojectionError=args.pnp_reproj_error,
        confidence=args.pnp_confidence,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not ok or inliers is None:
        raise RuntimeError("solvePnPRansac failed")
    if len(inliers) < args.min_pnp_inliers:
        raise RuntimeError(f"not enough inliers: {len(inliers)} < {args.min_pnp_inliers}")
    return rvec.reshape(3).astype(np.float64), tvec.reshape(3).astype(np.float64), inliers.reshape(-1)


def project_part_mesh_debug(rgb, cad, part_name, rvec, trans, K):
    R, _ = cv2.Rodrigues(rvec.reshape(3, 1))
    pts = (R @ cad.vertices_part[part_name].T).T + trans.reshape(1, 3)
    uv = _project(pts, K)
    out = rgb.copy()
    faces = cad.faces[part_name]
    color = (40, 220, 120) if part_name == "wrist" else (235, 70, 60)
    if part_name in ("l_gripper", "r_gripper"):
        color = (70, 150, 255)
    for tri in faces[:: max(1, len(faces) // 300)]:
        poly = np.round(uv[tri]).astype(np.int32)
        if np.all((poly[:, 0] >= -100) & (poly[:, 0] < rgb.shape[1] + 100) & (poly[:, 1] >= -100) & (poly[:, 1] < rgb.shape[0] + 100)):
            cv2.polylines(out, [poly], True, color, 1, cv2.LINE_AA)
    return out


def render_part_trimesh_panel(cad, part_name, rvec, trans, K, h, w, args, return_support=False):
    try:
        import pyrender
    except ImportError as exc:
        raise ImportError("pyrender is required for trimesh PnP debug projection") from exc

    R, _ = cv2.Rodrigues(rvec.reshape(3, 1))
    vertices_cam = (R @ cad.vertices_part[part_name].T).T + trans.reshape(1, 3)
    mesh = cad.meshes[part_name].copy()
    mesh.vertices = np.asarray(vertices_cam, dtype=np.float64)

    extrinsic = np.eye(4, dtype=np.float64)
    extrinsic[:, 1:3] *= -1

    ambient = float(getattr(args, "pyrender_ambient_light", 0.35))
    scene = pyrender.Scene(
        bg_color=[0, 0, 0, 0],
        ambient_light=[ambient, ambient, ambient],
    )
    scene.add(
        pyrender.Mesh.from_trimesh(
            mesh,
            smooth=bool(getattr(args, "mesh_render_smooth", 1)),
        )
    )
    camera = pyrender.IntrinsicsCamera(
        fx=float(K[0, 0]),
        fy=float(K[1, 1]),
        cx=float(K[0, 2]),
        cy=float(K[1, 2]),
        znear=float(args.pyrender_znear),
        zfar=float(args.pyrender_zfar),
    )
    scene.add(camera, pose=extrinsic)

    light = pyrender.PointLight(intensity=float(args.pyrender_light_intensity))
    light_pose = extrinsic.copy()
    light_pose[:, 1:3] *= -1
    light_pose[2, 3] += 0.1
    light_pose[0, 3] += 0.1
    scene.add(light, pose=light_pose)

    renderer = pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h)
    try:
        color, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    finally:
        renderer.delete()
    color = color[:, :, :3].astype(np.uint8)
    support = depth > 0
    if return_support:
        return color, support
    return color


def render_part_mesh_debug(rgb, cad, part_name, rvec, trans, K, args):
    try:
        mesh_panel, support = render_part_trimesh_panel(
            cad, part_name, rvec, trans, K, rgb.shape[0], rgb.shape[1], args, return_support=True
        )
    except Exception:
        return project_part_mesh_debug(rgb, cad, part_name, rvec, trans, K)
    out = rgb.copy()
    if np.any(support):
        out[support] = mesh_panel[support]
    return out


def part_candidates(pred, geom, model_meta, args, part_name):
    part_label, inst_prob, part_conf = pred_part_label_conf(pred, args)
    xyz = decoded_xyz_np(pred, model_meta, args)
    if part_label.shape != xyz.shape[:2]:
        part_label = np.asarray(
            Image.fromarray(part_label).resize((xyz.shape[1], xyz.shape[0]), Image.NEAREST),
            dtype=np.uint8,
        )
        inst_prob = resize_float_mask(inst_prob, xyz.shape[:2])
        part_conf = resize_float_mask(part_conf, xyz.shape[:2])
    label = PART_LABELS["gripper"] if part_name == "gripper" else PART_LABELS[part_name]
    mask = (inst_prob >= args.inst_thresh) & (part_label == label)
    if part_name == "shaft" and args.shaft_raw_x_min is not None:
        mask = (
            mask
            & np.isfinite(xyz[:, :, 0])
            & (xyz[:, :, 0] > float(args.shaft_raw_x_min))
        )
    ys, xs = np.where(mask)
    if len(xs) > int(args.debug_max_points):
        if getattr(args, "point_select", "random") == "top":
            vals = inst_prob[ys, xs] * part_conf[ys, xs]
            keep = np.argpartition(-vals, int(args.debug_max_points) - 1)[
                : int(args.debug_max_points)
            ]
            keep = keep[np.argsort(-vals[keep])]
        else:
            rng = np.random.default_rng(args.seed + len(xs))
            keep = rng.choice(len(xs), int(args.debug_max_points), replace=False)
        xs, ys = xs[keep], ys[keep]
    uv = quarter_pixels_to_original(
        xs,
        ys,
        xyz.shape[0],
        args.img_size,
        geom[2],
        geom[3],
        geom[4],
    )
    valid = (
        (uv[:, 0] >= 0.0)
        & (uv[:, 0] < geom[0].shape[1])
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] < geom[0].shape[0])
    )
    xyz_norm = xyz[ys[valid], xs[valid]].astype(np.float64)
    uv = uv[valid]
    finite = np.isfinite(xyz_norm).all(axis=1)
    uv = uv[finite]
    xyz_norm = xyz_norm[finite]
    return uv, xyz_norm


def reproj_rmse(points, uv, rvec, trans, K):
    R, _ = cv2.Rodrigues(rvec.reshape(3, 1))
    pts = (R @ points.T).T + trans.reshape(1, 3)
    proj = _project(pts, K)
    err = np.linalg.norm(proj - uv, axis=1)
    return float(np.sqrt(np.mean(err ** 2))), err, proj


def project_part_points(points, rvec, trans, K):
    R, _ = cv2.Rodrigues(rvec.reshape(3, 1))
    pts = (R @ points.T).T + trans.reshape(1, 3)
    return _project(pts, K)


def top_point_color(rank):
    palette = [
        (55, 220, 95),
        (255, 210, 60),
        (70, 170, 255),
        (255, 90, 210),
        (40, 230, 230),
        (255, 130, 65),
        (170, 120, 255),
        (120, 255, 120),
        (255, 255, 255),
        (80, 120, 255),
    ]
    return palette[int(rank) % len(palette)]


def pnp_match_crop(points, height, width, margin=48):
    pts = np.concatenate([p for p in points if p is not None and len(p) > 0], axis=0)
    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]
    if len(pts) == 0:
        return (0, 0, width, height)
    x0 = int(np.floor(np.min(pts[:, 0]) - margin))
    y0 = int(np.floor(np.min(pts[:, 1]) - margin))
    x1 = int(np.ceil(np.max(pts[:, 0]) + margin))
    y1 = int(np.ceil(np.max(pts[:, 1]) + margin))
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(width, max(x0 + 8, x1))
    y1 = min(height, max(y0 + 8, y1))
    return (x0, y0, x1, y1)


def crop_zoom_panel(img, crop, zoom, title):
    x0, y0, x1, y1 = crop
    panel = img[y0:y1, x0:x1].copy()
    panel = cv2.resize(
        panel,
        ((x1 - x0) * int(zoom), (y1 - y0) * int(zoom)),
        interpolation=cv2.INTER_LINEAR,
    )
    return add_title(panel, title)


def draw_pnp_match_zoom(
    path,
    rgb,
    cad,
    physical,
    rvec,
    trans,
    K,
    uv,
    raw_points,
    snapped_points,
    raw_proj,
    snapped_proj,
    err,
    args,
):
    if len(uv) == 0:
        return None
    top_n = min(int(args.top_pnp_points), len(err))
    order = np.argsort(err)[:top_n]
    h, w = rgb.shape[:2]
    zoom = int(getattr(args, "pnp_match_zoom", 3))
    crop = pnp_match_crop(
        [uv[order], raw_proj[order], snapped_proj[order]],
        h,
        w,
        margin=int(getattr(args, "pnp_match_margin", 48)),
    )

    dark = np.zeros_like(rgb)
    raw_mesh = render_part_mesh_debug(dark, cad, physical, rvec, trans, K, args)
    snap_mesh = render_part_mesh_debug(dark, cad, physical, rvec, trans, K, args)
    img_panel = rgb.copy()

    raw_panel = crop_zoom_panel(
        raw_mesh, crop, zoom, f"raw {physical} HCCE coord on mesh"
    )
    snap_panel = crop_zoom_panel(
        snap_mesh, crop, zoom, f"{physical} surface closest projection"
    )
    image_panel = crop_zoom_panel(img_panel, crop, zoom, "image correspondences")

    sep_w = 8
    sep = np.full((raw_panel.shape[0], sep_w, 3), 255, dtype=np.uint8)
    canvas = np.concatenate([raw_panel, sep, snap_panel, sep, image_panel], axis=1)

    x0, y0, _, _ = crop
    panel_w = raw_panel.shape[1]
    offset_raw = np.array([0, 0], dtype=np.float64)
    offset_snap = np.array([panel_w + sep_w, 0], dtype=np.float64)
    offset_img = np.array([(panel_w + sep_w) * 2, 0], dtype=np.float64)

    def to_canvas(pt, offset):
        return tuple(
            np.round((np.asarray(pt, dtype=np.float64) - np.array([x0, y0])) * zoom + offset)
            .astype(int)
            .tolist()
        )

    for rank, idx in enumerate(order):
        color = top_point_color(rank)
        raw_xy = to_canvas(raw_proj[idx], offset_raw)
        snap_xy = to_canvas(snapped_proj[idx], offset_snap)
        img_xy = to_canvas(uv[idx], offset_img)
        cv2.circle(canvas, raw_xy, 5, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, snap_xy, 5, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, img_xy, 5, color, -1, cv2.LINE_AA)
        cv2.line(canvas, snap_xy, img_xy, color, 1, cv2.LINE_AA)

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(path)
    return {
        "path": str(path),
        "crop_xyxy": [int(v) for v in crop],
        "zoom": int(zoom),
        "top_indices": [int(i) for i in order],
        "top_errors_px": [float(err[i]) for i in order],
    }


def write_part_pnp_debug(out_dir, pred, geom, cad, model_meta, args):
    rows = []
    axis_scale = np.asarray(model_meta.get("hcce_axis_scale", (1.0, 1.0, 1.0)), dtype=np.float64)
    for logical_part in ("wrist", "shaft", "gripper"):
        uv, xyz_norm = part_candidates(pred, geom, model_meta, args, logical_part)
        part_dir = out_dir / f"{logical_part}_pnp"
        part_dir.mkdir(parents=True, exist_ok=True)
        meta = {"part": logical_part, "n_raw": int(len(uv))}
        try:
            if len(uv) < args.min_pnp_inliers:
                raise RuntimeError(f"not enough points: {len(uv)}")
            xyz_cad_norm = xyz_norm / axis_scale.reshape(1, 3)
            part_jobs = []
            if logical_part == "gripper":
                points, names = closest_gripper_surface_points(
                    cad, xyz_cad_norm, k_faces=args.surface_k_faces
                )
                for physical in ("l_gripper", "r_gripper"):
                    idx = np.where(names == physical)[0]
                    if len(idx) >= args.min_pnp_inliers:
                        raw_points = _canon_norm_to_part(cad, physical, xyz_cad_norm[idx])
                        part_jobs.append((physical, uv[idx], points[idx], raw_points))
            else:
                points = closest_part_surface_points(
                    cad, logical_part, xyz_cad_norm, k_faces=args.surface_k_faces
                )
                raw_points = _canon_norm_to_part(cad, logical_part, xyz_cad_norm)
                part_jobs.append((logical_part, uv, points, raw_points))
            if not part_jobs:
                raise RuntimeError("no physical part has enough points")

            for physical, uv_i, pts_i, raw_pts_i in part_jobs:
                sub_dir = part_dir if logical_part != "gripper" else part_dir / physical
                sub_dir.mkdir(parents=True, exist_ok=True)
                rvec, trans, inliers = solve_part_pnp(pts_i, uv_i, geom[1], args)
                rmse, err, proj = reproj_rmse(pts_i, uv_i, rvec, trans, geom[1])
                raw_proj = project_part_points(raw_pts_i, rvec, trans, geom[1])
                raw_err = np.linalg.norm(raw_proj - uv_i, axis=1)
                order = np.argsort(err)[: min(args.top_pnp_points, len(err))]
                rgb = geom[0].copy()
                mesh = render_part_mesh_debug(rgb, cad, physical, rvec, trans, geom[1], args)
                pts_panel = mesh.copy()
                for rank, idx in enumerate(order):
                    color = (
                        int(80 + 150 * ((rank * 37) % 255) / 255),
                        int(255 - 120 * ((rank * 73) % 255) / 255),
                        int(80 + 150 * ((rank * 19) % 255) / 255),
                    )
                    cv2.circle(pts_panel, tuple(np.round(uv_i[idx]).astype(int)), 4, color, -1, cv2.LINE_AA)
                    cv2.circle(pts_panel, tuple(np.round(proj[idx]).astype(int)), 4, color, 1, cv2.LINE_AA)
                    cv2.line(
                        pts_panel,
                        tuple(np.round(uv_i[idx]).astype(int)),
                        tuple(np.round(proj[idx]).astype(int)),
                        color,
                        1,
                        cv2.LINE_AA,
                    )
                canvas = np.concatenate(
                    [
                        add_title(rgb, "rgb"),
                        add_title(mesh, f"{physical} pnp mesh"),
                        add_title(pts_panel, "top reproj points"),
                    ],
                    axis=1,
                )
                Image.fromarray(canvas).save(sub_dir / f"{physical}_pnp_summary.jpg")
                match_zoom = draw_pnp_match_zoom(
                    sub_dir / f"{physical}_pnp_top_points_match_zoom.jpg",
                    geom[0],
                    cad,
                    physical,
                    rvec,
                    trans,
                    geom[1],
                    uv_i,
                    raw_pts_i,
                    pts_i,
                    raw_proj,
                    proj,
                    err,
                    args,
                )
                np.savez(
                    sub_dir / f"{physical}_pnp_correspondences.npz",
                    uv=uv_i,
                    points_part=pts_i,
                    raw_points_part=raw_pts_i,
                    proj=proj,
                    raw_proj=raw_proj,
                    err=err,
                    raw_err=raw_err,
                    inliers=inliers,
                    rvec=rvec,
                    trans=trans,
                )
                rows.append(
                    {
                        "logical_part": logical_part,
                        "physical_part": physical,
                        "status": "ok",
                        "n_points": int(len(uv_i)),
                        "inliers": int(len(inliers)),
                        "rmse_px": rmse,
                        "summary": str(sub_dir / f"{physical}_pnp_summary.jpg"),
                        "match_zoom": "" if match_zoom is None else str(
                            sub_dir / f"{physical}_pnp_top_points_match_zoom.jpg"
                        ),
                    }
                )
                meta.setdefault("match_zoom", {})[physical] = match_zoom
            meta["status"] = "ok"
        except Exception as exc:
            meta["status"] = f"{type(exc).__name__}: {exc}"
            (part_dir / "error.txt").write_text(meta["status"], encoding="utf-8")
            rows.append(
                {
                    "logical_part": logical_part,
                    "physical_part": "",
                    "status": meta["status"],
                    "n_points": int(len(uv)),
                    "inliers": 0,
                    "rmse_px": float("nan"),
                    "summary": "",
                }
            )
        (part_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    with (out_dir / "part_pnp_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def process_checkpoint(args, exp_name, checkpoint, dataset, videos):
    device = torch.device(args.device)
    model, model_meta = load_model_unified(checkpoint, device)
    if int(model_meta["img_size"]) != int(args.img_size):
        raise ValueError(
            f"{checkpoint}: checkpoint img_size={model_meta['img_size']} != --img_size={args.img_size}"
        )
    cad = InstrumentCAD(CAD_ROOT)
    rng = np.random.default_rng(args.seed)
    out_root = args.output_dir / exp_name
    rows, failures, debug_candidates = [], [], []
    for video_name in videos:
        sample_indices = select_sample_indices(dataset, video_name, args.frame_stride, args.frames_per_video)
        ref_frames = reference_frames_for_video(args.reference_samples, video_name)
        if ref_frames is not None:
            sample_indices = [
                idx for idx in sample_indices
                if dataset.samples[idx][1] in ref_frames
            ]
        print(f"[{exp_name}] {video_name}: {len(sample_indices)} samples", flush=True)
        for local_i, sample_idx in enumerate(sample_indices):
            video, frame_id, instances_list = dataset.samples[sample_idx]
            try:
                img_array, annot = dataset[sample_idx]
                batch_img, y = collate_fn_instrument_hcce_pose([(img_array, annot)])
                with torch.no_grad():
                    preds = model(
                        batch_img.to(device),
                        is_training=False,
                        det_thresh=args.det_thresh,
                        nms_kernel_size=args.nms_kernel_size,
                    )
                if isinstance(preds, tuple):
                    preds = preds[-1]
                if not preds:
                    raise RuntimeError("no predictions")
                valid_gt = y["valid_instruments"][0].bool().numpy()
                centers = y["wrist_centers"][0][valid_gt].numpy().astype(np.float64)
                matches = match_gt_to_predictions(preds, centers, args.match_max_dist)
                geom = image_geometry(dataset, video, frame_id)
                ref_instances = reference_instances_for_frame(
                    args.reference_samples, video, frame_id
                )
                for gt_idx, pred_idx, dist in matches:
                    pred = preds[pred_idx]
                    instance_id = int(instances_list[gt_idx][0])
                    if ref_instances is not None and instance_id not in ref_instances:
                        continue
                    posehead_pose = pose_head_params_or_none(pred)
                    try:
                        pose, aux, extra = estimate_pose(pred, cad, geom, model_meta, args, rng)
                        status = "ok"
                    except Exception as exc:
                        pose, aux, extra = None, None, {}
                        status = f"{type(exc).__name__}: {exc}"
                    pred_part = pred_part_label(pred, args)[0]
                    vis_path = (
                        out_root
                        / "vis"
                        / "suturePulling"
                        / f"{video}_{frame_id}_inst{instance_id}.jpg"
                    )
                    save_visualization(
                        vis_path,
                        geom,
                        cad,
                        pose,
                        posehead_pose,
                        pred_part,
                        args,
                        pred=pred,
                    )
                    row = {
                        "experiment": exp_name,
                        "checkpoint": str(checkpoint),
                        "checkpoint_iter": int(model_meta["checkpoint_iter"]),
                        "variant": model_meta["variant"],
                        "video": video,
                        "frame_id": frame_id,
                        "sample_idx": int(sample_idx),
                        "instance_id": int(instance_id),
                        "pred_idx": int(pred_idx),
                        "match_dist_px": float(dist),
                        "status": status,
                        "pose_head_status": "ok" if posehead_pose is not None else "disabled",
                        "vis_path": str(vis_path),
                    }
                    row.update(extra)
                    rows.append(row)
                    if len(debug_candidates) < args.debug_samples_per_exp:
                        debug_candidates.append((video, frame_id, instance_id, pred, geom))
            except Exception as exc:
                failures.append(
                    {
                        "experiment": exp_name,
                        "video": video,
                        "frame_id": frame_id,
                        "sample_idx": int(sample_idx),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            if (local_i + 1) % 10 == 0 or local_i + 1 == len(sample_indices):
                print(f"[{exp_name}] {video}: {local_i + 1}/{len(sample_indices)}", flush=True)

    debug_rows = []
    for video, frame_id, inst_id, pred, geom in debug_candidates:
        debug_dir = (
            out_root
            / "part_pnp_debug"
            / f"{video}_{frame_id}_inst{inst_id}"
        )
        debug_dir.mkdir(parents=True, exist_ok=True)
        debug_rows.extend(write_part_pnp_debug(debug_dir, pred, geom, cad, model_meta, args))

    write_rows(out_root / "per_instance.csv", rows)
    if debug_rows:
        write_rows(out_root / "part_pnp_debug_summary.csv", debug_rows)
    (out_root / "summary.json").write_text(
        json.dumps(
            {
                "experiment": exp_name,
                "checkpoint": str(checkpoint),
                "model_meta": {
                    k: (v.tolist() if isinstance(v, np.ndarray) else v)
                    for k, v in model_meta.items()
                    if k != "raw_args"
                },
                "videos": videos,
                "num_rows": len(rows),
                "num_failures": len(failures),
                "failures": failures,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return rows, failures


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def parse_exp(value):
    if "=" not in value:
        path = Path(value)
        return path.parent.parent.name, path
    name, path = value.split("=", 1)
    return name, Path(path)


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(json_safe(v) for v in value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified suturePulling HCCE inference/debug for H/4 query and H/2 densepart models."
    )
    parser.add_argument(
        "--exp",
        action="append",
        required=True,
        help="Experiment spec NAME=/path/to/ckpt.pt. May be repeated.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=REPO_ROOT / "eval_outputs/suturepulling_unified_hcce_debug",
    )
    parser.add_argument("--video_names", nargs="*", default=None)
    parser.add_argument(
        "--reference_vis_dir",
        type=Path,
        default=None,
        help="Optional directory of existing *_inst*.jpg visualizations; only those video/frame/instance samples are rerun.",
    )
    parser.add_argument("--num_videos", type=int, default=2)
    parser.add_argument("--frame_stride", type=int, default=3)
    parser.add_argument("--frames_per_video", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--img_size", type=int, default=630)
    parser.add_argument("--train_ratio", type=float, default=0.95)
    parser.add_argument("--canonicalize_pose_symmetry", type=int, default=1, choices=[0, 1])
    parser.add_argument("--canonical_eps", type=float, default=0.08)
    parser.add_argument("--min_dice_shaft", type=float, default=0.8)
    parser.add_argument("--min_dice_wrist", type=float, default=0.6)
    parser.add_argument("--min_dice_gripper", type=float, default=0.6)
    parser.add_argument("--det_thresh", type=float, default=0.3)
    parser.add_argument("--nms_kernel_size", type=int, default=3)
    parser.add_argument("--match_max_dist", type=float, default=80.0)
    parser.add_argument("--inst_thresh", type=float, default=0.5)
    parser.add_argument("--hcce_bits", type=int, default=None)
    parser.add_argument("--hcce_bit_thresh", type=float, default=0.5)
    parser.add_argument("--max_points_per_part", type=int, default=1200)
    parser.add_argument("--point_select", choices=["random", "top"], default="random")
    parser.add_argument("--debug_max_points", type=int, default=1200)
    parser.add_argument("--debug_samples_per_exp", type=int, default=2)
    parser.add_argument("--top_pnp_points", type=int, default=10)
    parser.add_argument("--surface_snap_method", choices=["surface", "vertex"], default="surface")
    parser.add_argument("--surface_k_faces", type=int, default=0)
    parser.add_argument("--shaft_raw_x_min", type=float, default=-0.5)
    parser.add_argument("--min_wrist_points", type=int, default=12)
    parser.add_argument("--min_total_points", type=int, default=24)
    parser.add_argument("--min_shaft_points", type=int, default=24)
    parser.add_argument("--min_pnp_inliers", type=int, default=8)
    parser.add_argument("--pnp_iters", type=int, default=300)
    parser.add_argument("--pnp_reproj_error", type=float, default=8.0)
    parser.add_argument("--pnp_confidence", type=float, default=0.99)
    parser.add_argument("--freeze_wrist_after_pnp", type=int, default=0, choices=[0, 1])
    parser.add_argument("--optim_strategy", choices=["decoupled", "single"], default="single")
    parser.add_argument("--optim_parts", choices=["wrist_gripper", "all"], default="all")
    parser.add_argument(
        "--optim_loss",
        choices=["linear", "soft_l1", "huber", "cauchy", "arctan"],
        default="soft_l1",
    )
    parser.add_argument("--optim_f_scale", type=float, default=8.0)
    parser.add_argument("--optim_max_nfev", type=int, default=200)
    parser.add_argument("--min_depth", type=float, default=1e-4)
    parser.add_argument("--behind_camera_penalty", type=float, default=1e4)
    parser.add_argument("--draw_margin", type=int, default=64)
    parser.add_argument("--overlay_alpha", type=float, default=0.85)
    parser.add_argument("--mesh_overlay_alpha", type=float, default=0.85)
    parser.add_argument("--line_width", type=int, default=1)
    parser.add_argument("--mesh_overlay_edge_px", type=int, default=0)
    parser.add_argument("--mesh_render_backend", choices=["pyrender", "software"], default="pyrender")
    parser.add_argument("--pyrender_znear", type=float, default=0.001)
    parser.add_argument("--pyrender_zfar", type=float, default=0.6)
    parser.add_argument("--pyrender_light_intensity", type=float, default=3.0)
    parser.add_argument("--pyrender_ambient_light", type=float, default=0.35)
    parser.add_argument("--mesh_render_smooth", type=int, default=1)
    parser.add_argument("--render_seg_mode", choices=["surface_splat", "hybrid", "triangles"], default="surface_splat")
    parser.add_argument("--render_surface_points_shaft", type=int, default=20000)
    parser.add_argument("--render_surface_points_wrist", type=int, default=6000)
    parser.add_argument("--render_surface_points_gripper", type=int, default=8000)
    parser.add_argument("--render_splat_radius_px", type=int, default=3)
    parser.add_argument("--render_shaft_splat_radius_px", type=int, default=5)
    parser.add_argument("--render_shaft_close_px", type=int, default=0)
    parser.add_argument("--render_shaft_dilate_px", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.reference_samples = load_reference_samples(args.reference_vis_dir)
    dataset = build_suture_dataset(args)
    videos = select_video_names(dataset, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "selected_videos.txt").write_text(
        "\n".join(videos) + "\n", encoding="utf-8"
    )
    all_rows, all_failures = [], []
    for exp in args.exp:
        name, checkpoint = parse_exp(exp)
        rows, failures = process_checkpoint(args, name, checkpoint, dataset, videos)
        all_rows.extend(rows)
        all_failures.extend(failures)
    write_rows(args.output_dir / "all_per_instance.csv", all_rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "videos": videos,
                "experiments": [parse_exp(exp)[0] for exp in args.exp],
                "num_rows": len(all_rows),
                "num_failures": len(all_failures),
                "failures": all_failures,
                "args": {
                    k: json_safe(v)
                    for k, v in vars(args).items()
                    if k != "exp"
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[ok] wrote outputs to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
