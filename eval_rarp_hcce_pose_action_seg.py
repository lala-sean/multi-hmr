import argparse
import csv
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from datasets.rarp_hcce_pose_dataset import (
    RARPHCCEPoseDataset,
    collate_fn_instrument_hcce_pose,
)
from datasets.RarpInstanceDataset import RARPInstanceDataset
from datasets.rarp_pose_canonicalization import quat_to_matrix
from estimate_rarp_cse_articulate_pose import (
    CAD_ROOT,
    PART_LABELS,
    REPO_ROOT,
    InstrumentCAD,
    _binary_iou,
    _paint_projected_triangles,
    _project,
    _rotation_error_deg,
    _render_trimesh_panel,
    draw_projected_mesh,
    image_geometry,
    indices_for_video,
    match_gt_to_predictions,
    optimize_pose,
    select_videos,
    solve_wrist_pnp,
)
from estimate_rarp_hcce_articulate_pose import (
    build_correspondences,
    load_model,
)


DATASETS = {
    "needlePuncture": {
        "dataset_root": "/mnt/nas/share/shuojue/data/needlePuncture_videos",
        "pose_root": "/mnt/nas/share/shuojue/data/needlePuncture_results",
        "v2_force": False,
    },
    "needleGrasping": {
        "dataset_root": "/mnt/nas/share/shuojue/data/needleGrasping_videos",
        "pose_root": "/mnt/nas/share/shuojue/data/needleGrasping_results",
        "v2_force": False,
    },
    "knotting": {
        "dataset_root": "/mnt/nas/share/shuojue/data/knotting_videos",
        "pose_root": "/mnt/nas/share/shuojue/data/knotting_results",
        "v2_force": False,
    },
    "suturePulling": {
        "dataset_root": "/mnt/nas/share/shuojue/data/suturePulling_videos",
        "pose_root": None,
        "v2_force": True,
    },
}


def build_dataset(args, dataset_name):
    cfg = DATASETS[dataset_name]
    if cfg["pose_root"] is None:
        dataset = RARPInstanceDataset(
            split="test",
            training=False,
            img_size=args.img_size,
            dataset_root=cfg["dataset_root"],
            pose_root=cfg["pose_root"],
            min_dice=[args.min_dice_shaft, args.min_dice_wrist, args.min_dice_gripper],
            train_ratio=args.train_ratio,
            subsample=1,
            v2_force=bool(cfg.get("v2_force", False)),
            cse_coord_root=None,
            render_on_the_fly=False,
        )
    else:
        dataset = RARPHCCEPoseDataset(
            split="test",
            training=False,
            img_size=args.img_size,
            dataset_root=cfg["dataset_root"],
            pose_root=cfg["pose_root"],
            min_dice=[args.min_dice_shaft, args.min_dice_wrist, args.min_dice_gripper],
            train_ratio=args.train_ratio,
            subsample=1,
            v2_force=bool(cfg.get("v2_force", False)),
            cse_coord_root=None,
            render_on_the_fly=False,
            canonicalize_pose_symmetry=bool(args.canonicalize_pose_symmetry),
            canonical_eps=args.canonical_eps,
        )
    if len(dataset) == 0:
        raise RuntimeError(f"Dataset is empty: {dataset_name} test split")
    return dataset


def tensor_to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def quat_to_rvec(quat):
    quat_t = torch.as_tensor(tensor_to_numpy(quat), dtype=torch.float32).reshape(4)
    rot_mat = quat_to_matrix(quat_t).detach().cpu().numpy().astype(np.float64)
    rvec, _ = cv2.Rodrigues(rot_mat)
    return rvec.reshape(3), rot_mat


def get_gt_pose(y, gt_idx):
    quat = y["wrist_quat_gt"][0, gt_idx]
    action = y["action_gt"][0, gt_idx].detach().cpu().numpy().astype(np.float64)
    trans = y["wrist_trans_gt"][0, gt_idx].detach().cpu().numpy().astype(np.float64)
    _, rot_mat = quat_to_rvec(quat)
    return {
        "rot_mat": rot_mat,
        "trans": trans.reshape(3),
        "action": action.reshape(3),
        "alpha": float(action[0]),
        "theta_l": float(action[1]),
        "theta_r": float(action[2]),
    }


def pose_head_params(pred):
    needed = ("action_pred", "wrist_quat_pred", "wrist_trans_pred")
    if any(key not in pred for key in needed):
        raise RuntimeError("Prediction does not contain pose-head outputs")
    rvec, _ = quat_to_rvec(pred["wrist_quat_pred"])
    trans = tensor_to_numpy(pred["wrist_trans_pred"]).astype(np.float64).reshape(3)
    action = tensor_to_numpy(pred["action_pred"]).astype(np.float64).reshape(3)
    return np.concatenate([rvec, trans, action], axis=0).astype(np.float64)


def resolve_checkpoint_path(path):
    path = Path(path)
    if path.is_dir():
        candidates = [
            path / "checkpoints" / "best.pt",
            path / "best.pt",
            path / "checkpoints" / "last.pt",
            path / "last.pt",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(
            f"No checkpoint file found under {path}; tried checkpoints/best.pt, "
            "best.pt, checkpoints/last.pt, last.pt"
        )
    return path


def blank_metrics(prefix):
    out = {}
    for name in (
        "inst_iou",
        "part_iou_mean",
        "part_iou_gripper",
        "part_iou_wrist",
        "part_iou_shaft",
        "pred_inst_area",
        "gt_inst_area",
    ):
        out[f"{prefix}_{name}"] = float("nan")
    return out


def segmentation_metrics_from_masks(pred_part, gt_inst, gt_part, prefix):
    pred_part = np.asarray(pred_part, dtype=np.int64)
    pred_inst = pred_part > 0
    gt_inst = np.asarray(gt_inst).astype(bool)
    gt_part = np.asarray(gt_part, dtype=np.int64)
    metrics = {
        f"{prefix}_inst_iou": _binary_iou(pred_inst, gt_inst),
        f"{prefix}_pred_inst_area": int(pred_inst.sum()),
        f"{prefix}_gt_inst_area": int(gt_inst.sum()),
    }
    part_ious = []
    for label, name in [(1, "gripper"), (2, "wrist"), (3, "shaft")]:
        iou = _binary_iou(pred_inst & (pred_part == label), gt_inst & (gt_part == label))
        metrics[f"{prefix}_part_iou_{name}"] = iou
        if not np.isnan(iou):
            part_ious.append(iou)
    metrics[f"{prefix}_part_iou_mean"] = (
        float(np.mean(part_ious)) if part_ious else float("nan")
    )
    return metrics


def model_segmentation_metrics(pred, y, gt_idx, args, prefix="model_seg"):
    pred_inst = (
        torch.sigmoid(pred["inst_mask_logits"]).detach().cpu().numpy()
        >= args.inst_thresh
    )
    pred_part_logits = pred["part_mask_logits"].detach().cpu()
    pred_part = torch.argmax(pred_part_logits, dim=0).numpy().astype(np.int64)
    pred_part = np.where(pred_inst, pred_part, 0)
    gt_inst = y["inst_masks"][0, gt_idx].numpy() > 0.5
    gt_part = y["part_masks"][0, gt_idx].numpy().astype(np.int64)
    return segmentation_metrics_from_masks(pred_part, gt_inst, gt_part, prefix)


def model_part_mask(pred, args):
    pred_inst = (
        torch.sigmoid(pred["inst_mask_logits"]).detach().cpu().numpy()
        >= args.inst_thresh
    )
    pred_part = torch.argmax(pred["part_mask_logits"].detach().cpu(), dim=0)
    pred_part = pred_part.numpy().astype(np.uint8)
    return np.where(pred_inst, pred_part, 0).astype(np.uint8)


def part_label(part_name):
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
    if getattr(cad, "_eval_surface_render_key", None) == key:
        return cad._eval_surface_render_points

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
    cad._eval_surface_render_key = key
    cad._eval_surface_render_points = points
    return points


def splat_radius_for_part(part_name, args):
    if part_name == "shaft":
        return int(args.render_shaft_splat_radius_px)
    return int(args.render_splat_radius_px)


def rasterize_pose_surface_splat(cad, pose_params, K, height, width, args):
    transforms = cad.fk(
        pose_params[:3],
        pose_params[3:6],
        pose_params[6],
        pose_params[7],
        pose_params[8],
    )
    surface_points = get_surface_render_points(cad, args)
    label_mask = np.zeros((height, width), dtype=np.uint8)
    projected = []
    for part_name, points_part in surface_points.items():
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
                part_label(part_name),
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
            cv2.circle(label_mask, (x, y), radius, int(labels[idx]), thickness=-1, lineType=cv2.LINE_8)
    return label_mask


def rasterize_pose_triangles(cad, pose_params, K, height, width, args):
    posed_vertices = cad.posed_mesh_vertices(
        pose_params[:3],
        pose_params[3:6],
        pose_params[6],
        pose_params[7],
        pose_params[8],
    )
    label_mask = np.zeros((height, width), dtype=np.uint8)
    triangles = []
    for part_name, vertices_cam in posed_vertices.items():
        label = part_label(part_name)

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
            depth = float(vertices_cam[face, 2].mean())
            triangles.append((depth, pts, label))

    for _, pts, label in sorted(triangles, key=lambda item: item[0], reverse=True):
        poly = np.round(pts).astype(np.int32)
        cv2.fillConvexPoly(label_mask, poly, int(label), lineType=cv2.LINE_8)

    return label_mask


def rasterize_pose_part_mask(cad, pose_params, K, height, width, args):
    if args.render_seg_mode == "surface_splat":
        label_mask = rasterize_pose_surface_splat(cad, pose_params, K, height, width, args)
    elif args.render_seg_mode == "hybrid":
        label_mask = rasterize_pose_triangles(cad, pose_params, K, height, width, args)
        splat_mask = rasterize_pose_surface_splat(cad, pose_params, K, height, width, args)
        label_mask[(label_mask == 0) & (splat_mask > 0)] = splat_mask[(label_mask == 0) & (splat_mask > 0)]
    elif args.render_seg_mode == "triangles":
        label_mask = rasterize_pose_triangles(cad, pose_params, K, height, width, args)
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
        shaft = (label_mask == PART_LABELS["shaft"]).astype(np.uint8)
        shaft = cv2.dilate(shaft, kernel, iterations=1).astype(bool)
        label_mask[(label_mask == 0) & shaft] = PART_LABELS["shaft"]
    return label_mask


def original_mask_to_quarter(part_mask, img_size, scale, pad_x, pad_y):
    h, w = part_mask.shape[:2]
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = Image.fromarray(part_mask.astype(np.uint8)).resize(
        (new_w, new_h), Image.NEAREST
    )
    padded = np.zeros((img_size, img_size), dtype=np.uint8)
    padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = np.asarray(resized, dtype=np.uint8)
    quarter_size = img_size // 4
    return np.asarray(
        Image.fromarray(padded).resize((quarter_size, quarter_size), Image.NEAREST),
        dtype=np.uint8,
    )


def original_mask_to_padded(part_mask, img_size, scale, pad_x, pad_y):
    h, w = part_mask.shape[:2]
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = Image.fromarray(part_mask.astype(np.uint8)).resize(
        (new_w, new_h), Image.NEAREST
    )
    padded = np.zeros((img_size, img_size), dtype=np.uint8)
    padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = np.asarray(resized, dtype=np.uint8)
    return padded


def render_pose_quarter_mask(cad, pose_params, geom, args):
    if pose_params is None or not np.isfinite(pose_params).all():
        return None
    rgb, K, scale, pad_x, pad_y = geom
    part_original = rasterize_pose_part_mask(
        cad, pose_params, K, rgb.shape[0], rgb.shape[1], args
    )
    return original_mask_to_quarter(
        part_original, args.img_size, scale, pad_x, pad_y
    )


def render_pose_mesh_padded_mask(cad, pose_params, geom, args):
    if pose_params is None or not np.isfinite(pose_params).all():
        return None
    rgb, K, scale, pad_x, pad_y = geom
    part_original = rasterize_pose_triangles(
        cad, pose_params, K, rgb.shape[0], rgb.shape[1], args
    )
    return original_mask_to_padded(part_original, args.img_size, scale, pad_x, pad_y)


def render_segmentation_metrics(cad, pose_params, geom, y, gt_idx, args, prefix):
    part_quarter = render_pose_quarter_mask(cad, pose_params, geom, args)
    if part_quarter is None:
        return blank_metrics(prefix)
    gt_inst = y["inst_masks"][0, gt_idx].numpy() > 0.5
    gt_part = y["part_masks"][0, gt_idx].numpy().astype(np.int64)
    return segmentation_metrics_from_masks(part_quarter, gt_inst, gt_part, prefix)


def padded_rgb_from_geom(geom, img_size):
    rgb, _, scale, pad_x, pad_y = geom
    return pad_rgb_to_square(rgb, img_size, scale, pad_x, pad_y)


def pad_rgb_to_square(rgb, img_size, scale, pad_x, pad_y):
    h, w = rgb.shape[:2]
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = Image.fromarray(rgb).resize((new_w, new_h), Image.BILINEAR)
    padded = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = np.asarray(resized, dtype=np.uint8)
    return padded


def mask_to_color(mask):
    mask = np.asarray(mask, dtype=np.uint8)
    color = np.zeros(mask.shape + (3,), dtype=np.uint8)
    color[mask == PART_LABELS["gripper"]] = (70, 150, 255)
    color[mask == PART_LABELS["wrist"]] = (40, 210, 120)
    color[mask == PART_LABELS["shaft"]] = (235, 70, 60)
    return color


def upsample_part_mask(mask, img_size):
    if mask is None:
        return np.zeros((img_size, img_size), dtype=np.uint8)
    return np.asarray(
        Image.fromarray(mask.astype(np.uint8)).resize((img_size, img_size), Image.NEAREST),
        dtype=np.uint8,
    )


def overlay_part_mask(rgb, mask, alpha=0.55):
    color = mask_to_color(mask)
    support = (mask > 0)[..., None]
    blended = cv2.addWeighted(rgb, 1.0, color, alpha, 0)
    return np.where(support, blended, rgb).astype(np.uint8)


def overlay_mesh_mask(rgb, mask, alpha=0.45):
    if mask is None:
        return rgb.copy()
    mask = np.asarray(mask, dtype=np.uint8)
    panel = overlay_part_mask(rgb, mask, alpha=alpha)
    edges = np.zeros(mask.shape, dtype=np.uint8)
    for label in (PART_LABELS["gripper"], PART_LABELS["wrist"], PART_LABELS["shaft"]):
        part = (mask == label).astype(np.uint8)
        if part.any():
            contours, _ = cv2.findContours(part, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(edges, contours, -1, 255, 2, lineType=cv2.LINE_AA)
    panel[edges > 0] = (255, 255, 255)
    return panel


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


def blank_like_mesh_panel(rgb):
    h, w = rgb.shape[:2]
    sep = np.full((h, 6, 3), 255, dtype=np.uint8)
    blank = np.zeros_like(rgb)
    return np.concatenate([rgb, sep, blank, sep, blank, sep, rgb], axis=1)


def draw_projected_mesh_or_blank(rgb, cad, pose_params, K, args):
    if pose_params is None or not np.isfinite(pose_params).all():
        return blank_like_mesh_panel(rgb)
    return draw_projected_mesh(rgb, cad, pose_params, K, args)


def render_mesh_overlay(rgb, cad, pose_params, K, args):
    if pose_params is None or not np.isfinite(pose_params).all():
        return rgb.copy()
    posed_vertices = cad.posed_mesh_vertices(
        pose_params[:3],
        pose_params[3:6],
        pose_params[6],
        pose_params[7],
        pose_params[8],
    )
    if args.mesh_render_backend == "pyrender":
        mesh_panel, support = _render_trimesh_panel(
            cad,
            posed_vertices,
            K,
            rgb.shape[0],
            rgb.shape[1],
            args,
            return_support=True,
        )
    else:
        args.render_h, args.render_w = rgb.shape[:2]
        mesh_panel = _paint_projected_triangles(
            cad, posed_vertices, K, args, shade_mesh=True
        )
        support = np.any(mesh_panel > 0, axis=2)
    opacity = float(np.clip(args.mesh_overlay_alpha, 0.0, 1.0))
    blended = (
        (1.0 - opacity) * rgb.astype(np.float32)
        + opacity * mesh_panel.astype(np.float32)
    )
    out = np.where(support[:, :, None], blended, rgb).astype(np.uint8)
    edge_px = int(getattr(args, "mesh_overlay_edge_px", 0))
    if edge_px > 0 and support.any():
        kernel = np.ones((3, 3), dtype=np.uint8)
        edge = cv2.morphologyEx(support.astype(np.uint8), cv2.MORPH_GRADIENT, kernel)
        edge = cv2.dilate(edge, kernel, iterations=edge_px).astype(bool)
        out[edge] = (255, 255, 255)
    return out


def save_projected_mesh_visualization(path, geom, cad, hcce_pose, posehead_pose, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb, K = geom[0], geom[1]
    hcce_panel = draw_projected_mesh_or_blank(rgb, cad, hcce_pose, K, args)
    posehead_panel = draw_projected_mesh_or_blank(rgb, cad, posehead_pose, K, args)
    hcce_panel = add_title(hcce_panel, "hcce projected mesh")
    posehead_panel = add_title(posehead_panel, "posehead projected mesh")
    sep = np.full((8, hcce_panel.shape[1], 3), 255, dtype=np.uint8)
    canvas = np.concatenate([hcce_panel, sep, posehead_panel], axis=0)
    Image.fromarray(canvas).save(path)


def save_individual_visualizations(path, geom, pred, y, gt_idx, cad, hcce_pose, posehead_pose, args):
    stem_dir = path.with_suffix("")
    stem_dir.mkdir(parents=True, exist_ok=True)
    rgb_padded = padded_rgb_from_geom(geom, args.img_size)
    rgb_original, K = geom[0], geom[1]

    gt_part_q = y["part_masks"][0, gt_idx].numpy().astype(np.uint8)
    gt_inst_q = y["inst_masks"][0, gt_idx].numpy() > 0.5
    gt_mask = upsample_part_mask(np.where(gt_inst_q, gt_part_q, 0).astype(np.uint8), args.img_size)
    pred_mask = upsample_part_mask(model_part_mask(pred, args), args.img_size)
    hcce_render_mask = upsample_part_mask(
        render_pose_quarter_mask(cad, hcce_pose, geom, args), args.img_size
    )
    posehead_render_mask = upsample_part_mask(
        render_pose_quarter_mask(cad, posehead_pose, geom, args), args.img_size
    )

    Image.fromarray(rgb_padded).save(stem_dir / "rgb_padded.jpg")
    Image.fromarray(mask_to_color(gt_mask)).save(stem_dir / "gt_part_mask.png")
    Image.fromarray(mask_to_color(pred_mask)).save(stem_dir / "model_seg_mask.png")
    Image.fromarray(overlay_part_mask(rgb_padded, pred_mask)).save(
        stem_dir / "model_seg_overlay.jpg"
    )
    Image.fromarray(mask_to_color(hcce_render_mask)).save(
        stem_dir / "hcce_render_seg_mask.png"
    )
    Image.fromarray(overlay_part_mask(rgb_padded, hcce_render_mask)).save(
        stem_dir / "hcce_render_seg_overlay.jpg"
    )
    Image.fromarray(mask_to_color(posehead_render_mask)).save(
        stem_dir / "posehead_render_seg_mask.png"
    )
    Image.fromarray(overlay_part_mask(rgb_padded, posehead_render_mask)).save(
        stem_dir / "posehead_render_seg_overlay.jpg"
    )
    Image.fromarray(rgb_original).save(stem_dir / "rgb_original.jpg")
    Image.fromarray(render_mesh_overlay(rgb_original, cad, hcce_pose, K, args)).save(
        stem_dir / "hcce_mesh_overlay_original.jpg"
    )
    Image.fromarray(render_mesh_overlay(rgb_original, cad, posehead_pose, K, args)).save(
        stem_dir / "posehead_mesh_overlay_original.jpg"
    )
    return stem_dir


def save_instance_visualization(
    path,
    geom,
    pred,
    y,
    gt_idx,
    cad,
    hcce_pose,
    posehead_pose,
    args,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = padded_rgb_from_geom(geom, args.img_size)
    gt_part_q = y["part_masks"][0, gt_idx].numpy().astype(np.uint8)
    gt_inst_q = y["inst_masks"][0, gt_idx].numpy() > 0.5
    gt_part_q = np.where(gt_inst_q, gt_part_q, 0).astype(np.uint8)

    rgb_original, K, scale, pad_x, pad_y = geom
    model_mask = upsample_part_mask(model_part_mask(pred, args), args.img_size)
    hcce_render_mask = upsample_part_mask(
        render_pose_quarter_mask(cad, hcce_pose, geom, args), args.img_size
    )
    posehead_render_mask = upsample_part_mask(
        render_pose_quarter_mask(cad, posehead_pose, geom, args), args.img_size
    )
    hcce_mesh_overlay = pad_rgb_to_square(
        render_mesh_overlay(rgb_original, cad, hcce_pose, K, args),
        args.img_size,
        scale,
        pad_x,
        pad_y,
    )
    posehead_mesh_overlay = pad_rgb_to_square(
        render_mesh_overlay(rgb_original, cad, posehead_pose, K, args),
        args.img_size,
        scale,
        pad_x,
        pad_y,
    )

    masks = [
        ("rgb", None),
        ("hcce-trimesh", hcce_mesh_overlay),
        ("posehead-trimesh", posehead_mesh_overlay),
        ("model-seg", model_mask),
        ("hcce-proj-seg", hcce_render_mask),
        ("posehead-proj-seg", posehead_render_mask),
    ]
    panels = []
    for title, item in masks:
        if item is None:
            panel = rgb
        elif item.ndim == 3:
            panel = item
        else:
            panel = overlay_part_mask(rgb, item)
        panels.append(add_title(panel, title))
    sep = np.full((args.img_size, 6, 3), 255, dtype=np.uint8)
    canvas = panels[0]
    for panel in panels[1:]:
        canvas = np.concatenate([canvas, sep, panel], axis=1)
    Image.fromarray(canvas).save(path)

    mesh_path = None
    asset_dir = None
    if bool(args.save_projected_mesh_panel):
        mesh_path = path.with_name(path.stem + "_projected_mesh.jpg")
        save_projected_mesh_visualization(mesh_path, geom, cad, hcce_pose, posehead_pose, args)
    if bool(args.save_vis_assets):
        asset_dir = save_individual_visualizations(
            path, geom, pred, y, gt_idx, cad, hcce_pose, posehead_pose, args
        )
    return mesh_path, asset_dir


def pose_error_metrics(pose_params, gt_pose, prefix):
    if gt_pose is None:
        names = [
            "rot_err_deg",
            "trans_err_m",
            "trans_mse",
            "action_mse",
            "alpha_sqerr",
            "theta_l_sqerr",
            "theta_r_sqerr",
            "alpha_err_rad",
            "theta_l_err_rad",
            "theta_r_err_rad",
        ]
        return {f"{prefix}_{name}": float("nan") for name in names}
    if pose_params is None or not np.isfinite(pose_params).all():
        names = [
            "rot_err_deg",
            "trans_err_m",
            "trans_mse",
            "action_mse",
            "alpha_sqerr",
            "theta_l_sqerr",
            "theta_r_sqerr",
            "alpha_err_rad",
            "theta_l_err_rad",
            "theta_r_err_rad",
        ]
        return {f"{prefix}_{name}": float("nan") for name in names}

    rot_mat, _ = cv2.Rodrigues(pose_params[:3].reshape(3, 1))
    trans_diff = pose_params[3:6] - gt_pose["trans"]
    action_pred = pose_params[6:9]
    action_diff = action_pred - gt_pose["action"]
    return {
        f"{prefix}_rot_err_deg": _rotation_error_deg(rot_mat, gt_pose["rot_mat"]),
        f"{prefix}_trans_err_m": float(np.linalg.norm(trans_diff)),
        f"{prefix}_trans_mse": float(np.mean(trans_diff ** 2)),
        f"{prefix}_action_mse": float(np.mean(action_diff ** 2)),
        f"{prefix}_alpha_sqerr": float(action_diff[0] ** 2),
        f"{prefix}_theta_l_sqerr": float(action_diff[1] ** 2),
        f"{prefix}_theta_r_sqerr": float(action_diff[2] ** 2),
        f"{prefix}_alpha_err_rad": abs(float(action_diff[0])),
        f"{prefix}_theta_l_err_rad": abs(float(action_diff[1])),
        f"{prefix}_theta_r_err_rad": abs(float(action_diff[2])),
    }


def pose_value_columns(pose_params, prefix):
    names = ["rvec_x", "rvec_y", "rvec_z", "tx", "ty", "tz", "alpha", "theta_l", "theta_r"]
    if pose_params is None:
        return {f"{prefix}_pred_{name}": float("nan") for name in names}
    return {
        f"{prefix}_pred_{name}": float(value)
        for name, value in zip(names, pose_params.tolist())
    }


def should_save_visualization(args, dataset_name, video_name, frame_id, vis_state):
    if args.vis_every <= 0:
        return False
    if args.max_vis >= 0 and vis_state["saved"] >= args.max_vis:
        return False
    vis_videos_by_dataset = vis_state.get("vis_videos_by_dataset", {})
    if dataset_name in vis_videos_by_dataset and video_name not in vis_videos_by_dataset[dataset_name]:
        return False
    if vis_state["seen"] % args.vis_every != 0:
        return False
    if args.vis_frame_stride > 1:
        try:
            frame_int = int(frame_id)
        except (TypeError, ValueError):
            key = (dataset_name, video_name)
            frame_int = vis_state.setdefault("video_seen_frames", {}).get(key, 0)
            vis_state["video_seen_frames"][key] = frame_int + 1
        if frame_int % args.vis_frame_stride != 0:
            return False
    return True


def estimate_hcce_pose(pred, cad, geom, K, args, rng):
    corr, wrist_uv, wrist_points, _, _ = build_correspondences(pred, cad, geom, args, rng)
    rvec0, trans0, pnp_inliers = solve_wrist_pnp(wrist_uv, wrist_points, K, args)
    opt_params, reproj_rmse, opt_nfev, reproj_rmse_wg, reproj_rmse_shaft = optimize_pose(
        cad, corr, rvec0, trans0, K, args
    )
    return opt_params.astype(np.float64), {
        "hcce_corr_count": len(corr),
        "hcce_wrist_corr_count": len(wrist_uv),
        "hcce_pnp_inliers": pnp_inliers,
        "hcce_optim_nfev": opt_nfev,
        "hcce_reproj_rmse_px": reproj_rmse,
        "hcce_reproj_rmse_wrist_gripper_px": reproj_rmse_wg,
        "hcce_reproj_rmse_shaft_px": reproj_rmse_shaft,
        "hcce_status": "ok",
    }


def failure_hcce_extra(exc):
    return {
        "hcce_corr_count": float("nan"),
        "hcce_wrist_corr_count": float("nan"),
        "hcce_pnp_inliers": float("nan"),
        "hcce_optim_nfev": float("nan"),
        "hcce_reproj_rmse_px": float("nan"),
        "hcce_reproj_rmse_wrist_gripper_px": float("nan"),
        "hcce_reproj_rmse_shaft_px": float("nan"),
        "hcce_status": f"{type(exc).__name__}: {exc}",
    }


def process_sample(dataset_name, dataset, sample_idx, model, cad, device, args, rng, vis_state):
    video_name, frame_id, instances_list = dataset.samples[sample_idx]
    img_array, annot = dataset[sample_idx]
    batch_img, y = collate_fn_instrument_hcce_pose([(img_array, annot)])
    with torch.no_grad():
        preds = model(
            batch_img.to(device),
            is_training=False,
            det_thresh=args.det_thresh,
            nms_kernel_size=args.nms_kernel_size,
        )
    if not isinstance(preds, list) or len(preds) == 0:
        raise RuntimeError(f"No predicted instances for {video_name}/{frame_id}")

    valid_gt = y["valid_instruments"][0].bool().numpy()
    gt_wrist_centers = y["wrist_centers"][0][valid_gt].numpy().astype(np.float64)
    matches = match_gt_to_predictions(preds, gt_wrist_centers, args.match_max_dist)
    base_dataset = getattr(dataset, "base", dataset)
    geom = image_geometry(base_dataset, video_name, frame_id)
    K = geom[1]

    rows = []
    for gt_idx, pred_idx, match_dist in matches:
        pred = preds[pred_idx]
        instance_id = int(instances_list[gt_idx][0])
        has_pose_gt = bool(y.get("has_pose", torch.zeros_like(y["valid_instruments"]))[0, gt_idx])
        gt_pose = get_gt_pose(y, gt_idx) if has_pose_gt else None

        try:
            hcce_pose, hcce_extra = estimate_hcce_pose(pred, cad, geom, K, args, rng)
        except Exception as exc:
            if not bool(args.keep_failed_instances):
                raise
            hcce_pose = None
            hcce_extra = failure_hcce_extra(exc)

        if bool(args.eval_pose_head):
            try:
                ph_pose = pose_head_params(pred)
                ph_status = "ok"
            except Exception as exc:
                if not bool(args.keep_failed_instances):
                    raise
                ph_pose = None
                ph_status = f"{type(exc).__name__}: {exc}"
        else:
            ph_pose = None
            ph_status = "disabled"

        row = {
            "dataset": dataset_name,
            "video": video_name,
            "frame_id": frame_id,
            "instance_id": instance_id,
            "sample_idx": sample_idx,
            "pred_idx": pred_idx,
            "gt_idx": gt_idx,
            "match_dist_px": match_dist,
            "pose_head_status": ph_status,
            "has_pose_gt": int(has_pose_gt),
            "gt_alpha": float("nan") if gt_pose is None else gt_pose["alpha"],
            "gt_theta_l": float("nan") if gt_pose is None else gt_pose["theta_l"],
            "gt_theta_r": float("nan") if gt_pose is None else gt_pose["theta_r"],
            "gt_tx": float("nan") if gt_pose is None else gt_pose["trans"][0],
            "gt_ty": float("nan") if gt_pose is None else gt_pose["trans"][1],
            "gt_tz": float("nan") if gt_pose is None else gt_pose["trans"][2],
        }
        row.update(model_segmentation_metrics(pred, y, gt_idx, args))
        row.update(hcce_extra)
        row.update(pose_error_metrics(hcce_pose, gt_pose, "hcce"))
        row.update(pose_error_metrics(ph_pose, gt_pose, "posehead"))
        row.update(render_segmentation_metrics(cad, hcce_pose, geom, y, gt_idx, args, "hcce_render_seg"))
        row.update(render_segmentation_metrics(cad, ph_pose, geom, y, gt_idx, args, "posehead_render_seg"))
        row.update(pose_value_columns(hcce_pose, "hcce"))
        row.update(pose_value_columns(ph_pose, "posehead"))

        if should_save_visualization(args, dataset_name, video_name, frame_id, vis_state):
            vis_path = (
                args.output_dir
                / "vis"
                / dataset_name
                / f"{video_name}_{frame_id}_inst{instance_id}.jpg"
            )
            mesh_vis_path, vis_asset_dir = save_instance_visualization(
                vis_path,
                geom,
                pred,
                y,
                gt_idx,
                cad,
                hcce_pose,
                ph_pose,
                args,
            )
            row["vis_path"] = str(vis_path)
            row["mesh_vis_path"] = "" if mesh_vis_path is None else str(mesh_vis_path)
            row["vis_asset_dir"] = "" if vis_asset_dir is None else str(vis_asset_dir)
            vis_state["saved"] += 1
        else:
            row["vis_path"] = ""
            row["mesh_vis_path"] = ""
            row["vis_asset_dir"] = ""
        vis_state["seen"] += 1
        rows.append(row)
    return rows


def numeric_summary(rows):
    if not rows:
        return {"count": 0}
    keys = sorted({key for row in rows for key, value in row.items() if isinstance(value, (int, float, np.number))})
    summary = {"count": len(rows)}
    for key in keys:
        values = np.array([float(row.get(key, np.nan)) for row in rows], dtype=np.float64)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        summary[f"{key}_mean"] = float(np.mean(finite))
        summary[f"{key}_median"] = float(np.median(finite))
        summary[f"{key}_count"] = int(finite.size)

    for prefix in ("hcce", "posehead"):
        key = f"{prefix}_action_mse"
        vals = np.array([float(row.get(key, np.nan)) for row in rows], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            summary[f"{prefix}_action_rmse"] = float(math.sqrt(float(np.mean(vals))))
    return summary


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    preferred = [
        "dataset", "video", "frame_id", "instance_id", "sample_idx",
        "gt_idx", "pred_idx", "match_dist_px", "hcce_status", "pose_head_status",
    ]
    fieldnames = preferred + [key for key in fieldnames if key not in preferred]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Batch-evaluate current HCCE+pose-head training checkpoints on RARP test "
            "sets for action pose MSE and segmentation metrics."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT
        / "logs/instrument_rarp_hcce_pose_dpt_sym_poseheads_2gpu_bs6/checkpoints/best.pt",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=REPO_ROOT / "eval_outputs/rarp_hcce_pose_action_seg",
    )
    parser.add_argument(
        "--dataset_names",
        nargs="+",
        choices=sorted(DATASETS.keys()),
        default=["needlePuncture", "needleGrasping", "knotting"],
    )
    parser.add_argument("--video_select", choices=["random"], default="random")
    parser.add_argument("--video_name", type=str, default=None)
    parser.add_argument("--num_videos", type=int, default=999999)
    parser.add_argument("--frame_select", choices=["first", "random", "even"], default="first")
    parser.add_argument("--frames_per_video", type=int, default=-1)
    parser.add_argument("--max_samples_per_dataset", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--img_size", type=int, default=630)
    parser.add_argument("--train_ratio", type=float, default=0.95)
    parser.add_argument("--min_dice_shaft", type=float, default=0.8)
    parser.add_argument("--min_dice_wrist", type=float, default=0.6)
    parser.add_argument("--min_dice_gripper", type=float, default=0.6)
    parser.add_argument("--canonicalize_pose_symmetry", type=int, default=1, choices=[0, 1])
    parser.add_argument("--canonical_eps", type=float, default=0.08)

    parser.add_argument("--det_thresh", type=float, default=0.3)
    parser.add_argument("--nms_kernel_size", type=int, default=3)
    parser.add_argument("--match_max_dist", type=float, default=80.0)
    parser.add_argument("--inst_thresh", type=float, default=0.5)

    parser.add_argument("--hcce_bits", type=int, default=None)
    parser.add_argument("--hcce_coord_min", type=float, default=None)
    parser.add_argument("--hcce_coord_max", type=float, default=None)
    parser.add_argument("--hcce_bit_thresh", type=float, default=0.5)
    parser.add_argument("--max_points_per_part", type=int, default=1200)
    parser.add_argument("--min_wrist_points", type=int, default=12)
    parser.add_argument("--min_total_points", type=int, default=24)
    parser.add_argument("--min_shaft_points", type=int, default=24)
    parser.add_argument("--min_pnp_inliers", type=int, default=8)
    parser.add_argument("--pnp_iters", type=int, default=300)
    parser.add_argument("--pnp_reproj_error", type=float, default=8.0)
    parser.add_argument("--pnp_confidence", type=float, default=0.99)
    parser.add_argument("--freeze_wrist_after_pnp", type=int, default=0, choices=[0, 1])
    parser.add_argument("--optim_strategy", choices=["decoupled", "single"], default="decoupled")
    parser.add_argument("--optim_parts", choices=["wrist_gripper", "all"], default="wrist_gripper")
    parser.add_argument(
        "--optim_loss",
        choices=["linear", "soft_l1", "huber", "cauchy", "arctan"],
        default="soft_l1",
    )
    parser.add_argument("--optim_f_scale", type=float, default=8.0)
    parser.add_argument("--optim_max_nfev", type=int, default=200)

    parser.add_argument("--min_depth", type=float, default=1e-4)
    parser.add_argument("--draw_margin", type=int, default=64)
    parser.add_argument("--overlay_alpha", type=float, default=0.85)
    parser.add_argument("--mesh_overlay_alpha", type=float, default=0.85)
    parser.add_argument(
        "--mesh_render_backend",
        choices=["pyrender", "software"],
        default="pyrender",
    )
    parser.add_argument("--pyrender_znear", type=float, default=0.001)
    parser.add_argument("--pyrender_zfar", type=float, default=0.6)
    parser.add_argument("--pyrender_light_intensity", type=float, default=1.0)
    parser.add_argument("--pyrender_ambient_light", type=float, default=0.3)
    parser.add_argument("--mesh_render_smooth", type=int, default=1)
    parser.add_argument("--mesh_overlay_edge_px", type=int, default=0)
    parser.add_argument(
        "--render_seg_mode",
        choices=["surface_splat", "hybrid", "triangles"],
        default="surface_splat",
        help="How to render pose-derived segmentation masks.",
    )
    parser.add_argument("--render_surface_points_shaft", type=int, default=20000)
    parser.add_argument("--render_surface_points_wrist", type=int, default=6000)
    parser.add_argument("--render_surface_points_gripper", type=int, default=8000)
    parser.add_argument("--render_splat_radius_px", type=int, default=3)
    parser.add_argument("--render_shaft_splat_radius_px", type=int, default=5)
    parser.add_argument(
        "--render_shaft_close_px",
        type=int,
        default=0,
        help="Close holes inside the rendered shaft mask before metric resizing.",
    )
    parser.add_argument(
        "--render_shaft_dilate_px",
        type=int,
        default=0,
        help="Dilate only the rendered shaft region into background pixels before quarter resize.",
    )
    parser.add_argument("--keep_failed_instances", type=int, default=1, choices=[0, 1])
    parser.add_argument("--skip_failed_samples", type=int, default=1, choices=[0, 1])
    parser.add_argument(
        "--vis_every",
        type=int,
        default=25,
        help="Save one visualization every N evaluated instances; <=0 disables.",
    )
    parser.add_argument(
        "--vis_frame_stride",
        type=int,
        default=1,
        help="For visualizations, save only frames whose frame_id is divisible by this stride.",
    )
    parser.add_argument(
        "--vis_video_count",
        type=int,
        default=-1,
        help="Randomly choose this many videos per dataset for visualization only; <0 means no video filter.",
    )
    parser.add_argument(
        "--max_vis",
        type=int,
        default=80,
        help="Maximum number of instance visualizations to save; <0 means unlimited.",
    )
    parser.add_argument("--save_projected_mesh_panel", type=int, default=0, choices=[0, 1])
    parser.add_argument("--save_vis_assets", type=int, default=0, choices=[0, 1])
    parser.add_argument(
        "--eval_pose_head",
        type=int,
        default=-1,
        choices=[-1, 0, 1],
        help="1=force pose-head metrics, 0=disable, -1=auto from checkpoint args.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    device = torch.device(args.device)
    args.checkpoint = resolve_checkpoint_path(args.checkpoint)
    model, model_args, coord_args = load_model(args.checkpoint, device)
    if int(model_args["img_size"]) != int(args.img_size):
        raise ValueError(
            f"--img_size={args.img_size} does not match checkpoint img_size={model_args['img_size']}"
        )
    checkpoint_has_pose_head = bool(model_args.get("use_pose_heads", 0))
    if args.eval_pose_head < 0:
        args.eval_pose_head = int(checkpoint_has_pose_head)
    elif bool(args.eval_pose_head) and not checkpoint_has_pose_head:
        raise ValueError(
            "--eval_pose_head=1 was requested, but this checkpoint was built with "
            "use_pose_heads=0."
        )
    if not bool(args.eval_pose_head):
        print(
            "[info] pose-head metrics disabled; evaluating HCCE-estimated pose only.",
            flush=True,
        )
    args.hcce_bits = int(model_args["hcce_bits"] if args.hcce_bits is None else args.hcce_bits)
    args.hcce_coord_min = float(
        coord_args["hcce_coord_min"] if args.hcce_coord_min is None else args.hcce_coord_min
    )
    args.hcce_coord_max = float(
        coord_args["hcce_coord_max"] if args.hcce_coord_max is None else args.hcce_coord_max
    )

    cad = InstrumentCAD(CAD_ROOT)
    all_rows = []
    failures = []
    vis_state = {"seen": 0, "saved": 0, "vis_videos_by_dataset": {}}
    for dataset_name in args.dataset_names:
        dataset = build_dataset(args, dataset_name)
        video_names = select_videos(dataset, args)
        if args.vis_video_count > 0:
            vis_rng = random.Random(f"{args.seed}:vis:{dataset_name}")
            n_vis_videos = min(int(args.vis_video_count), len(video_names))
            vis_videos = set(vis_rng.sample(video_names, n_vis_videos))
            vis_state["vis_videos_by_dataset"][dataset_name] = vis_videos
            print(
                f"[vis] {dataset_name}: {len(vis_videos)} videos -> "
                f"{', '.join(sorted(vis_videos))}",
                flush=True,
            )
        sample_indices = []
        for video_name in video_names:
            sample_indices.extend(
                indices_for_video(
                    dataset,
                    video_name,
                    args.frames_per_video,
                    frame_select=args.frame_select,
                    seed=args.seed,
                )
            )
        if args.max_samples_per_dataset > 0:
            sample_indices = sample_indices[:args.max_samples_per_dataset]

        print(
            f"[dataset] {dataset_name}: {len(sample_indices)} samples from "
            f"{len(video_names)} videos",
            flush=True,
        )
        dataset_rows = []
        for count, sample_idx in enumerate(sample_indices, start=1):
            try:
                rows = process_sample(
                    dataset_name,
                    dataset,
                    sample_idx,
                    model,
                    cad,
                    device,
                    args,
                    rng,
                    vis_state,
                )
                dataset_rows.extend(rows)
                all_rows.extend(rows)
            except Exception as exc:
                video_name, frame_id, _ = dataset.samples[sample_idx]
                failures.append(
                    {
                        "dataset": dataset_name,
                        "video": video_name,
                        "frame_id": frame_id,
                        "sample_idx": int(sample_idx),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                if not bool(args.skip_failed_samples):
                    raise
                print(f"[skip] {dataset_name}/{video_name}/{frame_id}: {exc}", flush=True)
            if count % 25 == 0 or count == len(sample_indices):
                print(f"[progress] {dataset_name}: {count}/{len(sample_indices)}", flush=True)

        write_csv(args.output_dir / f"{dataset_name}_per_instance.csv", dataset_rows)

    write_csv(args.output_dir / "per_instance.csv", all_rows)
    summary = {
        "checkpoint": str(args.checkpoint),
        "datasets": args.dataset_names,
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "all": numeric_summary(all_rows),
        "by_dataset": {
            name: numeric_summary([row for row in all_rows if row["dataset"] == name])
            for name in args.dataset_names
        },
        "failures": failures,
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(f"[ok] wrote {args.output_dir / 'per_instance.csv'}")
    print(f"[ok] wrote {args.output_dir / 'summary.json'}")
    print(
        "[key] action MSE: "
        f"hcce={summary['all'].get('hcce_action_mse_mean', float('nan')):.6g}, "
        f"posehead={summary['all'].get('posehead_action_mse_mean', float('nan')):.6g}",
        flush=True,
    )
    print(
        "[key] segmentation part IoU mean: "
        f"model={summary['all'].get('model_seg_part_iou_mean_mean', float('nan')):.6g}, "
        f"hcce-render={summary['all'].get('hcce_render_seg_part_iou_mean_mean', float('nan')):.6g}, "
        f"posehead-render={summary['all'].get('posehead_render_seg_part_iou_mean_mean', float('nan')):.6g}",
        flush=True,
    )


if __name__ == "__main__":
    main()
