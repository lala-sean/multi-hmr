import json
import os
from argparse import ArgumentParser
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("EGL_DEVICE_ID", "7")

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
import yaml
from PIL import Image

from demo_seg import (
    KEYPOINT_COLORS,
    SURGPOSE_LEFT_FRAMES,
    SURGPOSE_METADATA,
    SURGPOSE_SAM3_SEGMENTATION,
    add_pose_recovery_args,
    apply_external_instances,
    configure_device,
    draw_keypoints,
    draw_wrist_centers,
    external_idx_from_instances,
    image_paths_from_input,
    load_camera_K,
    load_demo_model,
    model_seg_panel,
    normalize_predictions,
    pad_original_mask,
    preprocess,
    recover_hcce_poses,
    split_evenly,
)
from estimate_rarp_cse_articulate_pose import CAD_ROOT, InstrumentCAD, RARP_FC
from eval_suturepulling_hcce_unified_debug import (
    add_title,
    decoded_xyz_np,
    pad_rgb_to_square,
    pose_head_params_or_none,
    render_mesh_overlay_direct,
)


DEPTH_CKPT = (
    "logs/instrument_rarp_hcce_densepart_keypoint_heatmap7_depthrel_dpt_h2_vos_surgpose30_gpu4567_bs6_noeval"
    "/checkpoints/last.pt"
)
SURGPOSE_PART_SEG = (
    "/mnt/nas/share/shuojue/data/surgpose/000000/processed_stereo_640/sam3_segmentaion_part"
)


def draw_keypoints_small(panel, preds, color=None):
    out = panel.copy()
    h, w = out.shape[:2]
    for pred in preds:
        keypoint = pred.get("keypoint_xy")
        if keypoint is None:
            continue
        if torch.is_tensor(keypoint):
            keypoint = keypoint.detach().cpu().numpy()
        keypoint = np.asarray(keypoint, dtype=np.float64).reshape(-1, 2)
        for j, xy in enumerate(keypoint):
            if not np.isfinite(xy).all():
                continue
            x, y = int(round(float(xy[0]))), int(round(float(xy[1])))
            if not (0 <= x < w and 0 <= y < h):
                continue
            c = KEYPOINT_COLORS[j % len(KEYPOINT_COLORS)] if color is None else color
            cv2.circle(out, (x, y), 3, c, 1, cv2.LINE_AA)
            cv2.drawMarker(
                out,
                (x, y),
                c,
                markerType=cv2.MARKER_CROSS,
                markerSize=8,
                thickness=1,
                line_type=cv2.LINE_AA,
            )
    return out


def _to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().float().cpu().numpy()
    return np.asarray(value)


def _resize_nearest(mask, shape_hw):
    return np.asarray(
        Image.fromarray(mask.astype(np.uint8)).resize(
            (int(shape_hw[1]), int(shape_hw[0])),
            Image.NEAREST,
        ),
        dtype=np.uint8,
    )


def _resize_float(img, shape_hw):
    return cv2.resize(
        img.astype(np.float32),
        (int(shape_hw[1]), int(shape_hw[0])),
        interpolation=cv2.INTER_LINEAR,
    )


def _pad_original_float(arr, img_size, scale, pad_x, pad_y):
    h, w = arr.shape[:2]
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(
        arr.astype(np.float32),
        (new_w, new_h),
        interpolation=cv2.INTER_NEAREST,
    )
    padded = np.zeros((img_size, img_size), dtype=np.float32)
    padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return padded


def _normalize_depth_in_mask(depth, valid, low_pct=2.0, high_pct=98.0, eps=1e-6):
    vals = depth[valid]
    if vals.size == 0:
        return np.zeros_like(depth, dtype=np.float32)
    lo, hi = np.percentile(vals.astype(np.float32), [float(low_pct), float(high_pct)])
    if not np.isfinite(lo) or not np.isfinite(hi) or float(hi - lo) <= eps:
        return np.zeros_like(depth, dtype=np.float32)
    out = (depth.astype(np.float32) - float(lo)) / float(hi - lo)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _colorize_valid_scalar(values, valid, cmap=cv2.COLORMAP_TURBO):
    color = cv2.applyColorMap(
        np.clip(values * 255.0, 0, 255).astype(np.uint8),
        cmap,
    )
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    color[~valid] = 0
    return color


def external_mask_path_for_image(img_path, args):
    if not args.external_inst_dir:
        return None
    root = Path(args.external_inst_dir)
    candidates = [
        root / Path(img_path).name,
        root / f"{Path(img_path).stem}.png",
        root / f"{Path(img_path).stem}.jpg",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return candidates[0]


def load_external_instances_for_demo(img_path, rgb_shape, geom, args, patch_size):
    mask_path = external_mask_path_for_image(img_path, args)
    if mask_path is None:
        return None
    if not Path(mask_path).is_file():
        if args.require_external_inst:
            raise FileNotFoundError(f"External instance mask not found: {mask_path}")
        return []

    mask = np.asarray(Image.open(mask_path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.shape[:2] != tuple(rgb_shape[:2]):
        mask = _resize_nearest(mask, rgb_shape[:2])

    if args.external_union_as_instance:
        label_masks = [(1, mask > 0)]
    else:
        label_masks = [(int(x), mask == int(x)) for x in sorted(np.unique(mask)) if int(x) > 0]

    part_mask = None
    if args.external_part_dir:
        part_path = external_mask_path_for_image(img_path, type("Args", (), {"external_inst_dir": args.external_part_dir})())
        if part_path is not None and Path(part_path).is_file():
            part_mask = np.asarray(Image.open(part_path))
            if part_mask.ndim == 3:
                part_mask = part_mask[..., 0]
            if part_mask.shape[:2] != tuple(rgb_shape[:2]):
                part_mask = _resize_nearest(part_mask, rgb_shape[:2])

    _, _, scale, pad_x, pad_y = geom
    grid_size = args.img_size // int(patch_size)
    instances = []
    for label, inst_orig in label_masks:
        if int(inst_orig.sum()) < int(args.external_min_area):
            continue
        inst_pad = pad_original_mask(inst_orig.astype(np.uint8), args.img_size, scale, pad_x, pad_y)
        query_pad = inst_pad
        query_source = "instance_center"
        if part_mask is not None:
            wrist_orig = inst_orig & (part_mask == 2)
            if int(wrist_orig.sum()) >= int(args.external_min_area):
                query_pad = pad_original_mask(wrist_orig.astype(np.uint8), args.img_size, scale, pad_x, pad_y)
                query_source = "wrist_part_center"
        ys, xs = np.where(query_pad > 0)
        if len(xs) == 0:
            continue
        cx = float(xs.mean())
        cy = float(ys.mean())
        patch_x = int(np.clip(cx / float(patch_size), 0, grid_size - 1))
        patch_y = int(np.clip(cy / float(patch_size), 0, grid_size - 1))
        instances.append(
            {
                "label": int(label),
                "mask_orig": inst_orig.astype(np.uint8),
                "mask_padded": inst_pad.astype(np.uint8),
                "center_xy": (cx, cy),
                "idx_yx": (patch_y, patch_x),
                "mask_path": str(mask_path),
                "area": int(inst_orig.sum()),
                "query_source": query_source,
            }
        )
    return instances


def gt_relative_depth_panel(img_path, img_np, geom, external_instances, args):
    if not args.gt_depth_dir:
        return add_title(img_np.copy(), "gt-rel-depth unavailable")
    depth_path = Path(args.gt_depth_dir) / f"{Path(img_path).stem}.npy"
    if not depth_path.is_file():
        return add_title(img_np.copy(), "gt-rel-depth unavailable")
    if not external_instances:
        return add_title(img_np.copy(), "gt-rel-depth no mask")

    depth = np.load(depth_path).astype(np.float32)
    rgb_orig, _, scale, pad_x, pad_y = geom
    if depth.shape[:2] != rgb_orig.shape[:2]:
        depth = cv2.resize(
            depth,
            (rgb_orig.shape[1], rgb_orig.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    depth_acc = np.zeros((args.img_size, args.img_size), dtype=np.float32)
    valid_acc = np.zeros((args.img_size, args.img_size), dtype=bool)
    for inst in external_instances:
        mask = np.asarray(inst["mask_orig"], dtype=bool)
        if mask.shape != depth.shape:
            mask = _resize_nearest(mask.astype(np.uint8), depth.shape).astype(bool)
        valid = np.isfinite(depth) & (depth > 0) & mask
        rel = _normalize_depth_in_mask(
            depth,
            valid,
            low_pct=args.gt_depth_norm_low_pct,
            high_pct=args.gt_depth_norm_high_pct,
        )
        rel_pad = _pad_original_float(rel, args.img_size, scale, pad_x, pad_y)
        valid_pad = pad_original_mask(valid.astype(np.uint8), args.img_size, scale, pad_x, pad_y) > 0
        depth_acc[valid_pad] = rel_pad[valid_pad]
        valid_acc |= valid_pad

    if not valid_acc.any():
        return add_title(img_np.copy(), "gt-rel-depth empty")
    color = _colorize_valid_scalar(depth_acc, valid_acc, cmap=cv2.COLORMAP_TURBO)
    out = img_np.copy()
    out[valid_acc] = color[valid_acc]
    return add_title(out, "gt-rel-depth")


def _lookup_yaml_key(data, key):
    if key in data:
        return data[key]
    skey = str(key)
    if skey in data:
        return data[skey]
    return None


def _frame_index_from_stem(stem):
    return int(str(stem).split("_")[-1])


def load_gt_keypoints(img_path, geom, args):
    if not args.gt_keypoint_file:
        return []
    path = Path(args.gt_keypoint_file)
    if not path.is_file():
        return []
    with path.open("r") as f:
        data = yaml.safe_load(f) or {}
    frame_data = _lookup_yaml_key(data, _frame_index_from_stem(Path(img_path).stem))
    if not frame_data:
        return []

    _, _, scale, pad_x, pad_y = geom
    groups = []
    for base_label in (1, 8):
        pts = []
        valid = []
        for local_idx in range(7):
            raw = _lookup_yaml_key(frame_data, base_label + local_idx)
            if raw is None or len(raw) < 2:
                pts.append((0.0, 0.0))
                valid.append(False)
                continue
            x = float(raw[0]) * scale + pad_x
            y = float(raw[1]) * scale + pad_y
            ok = np.isfinite(x) and np.isfinite(y) and 0 <= x < args.img_size and 0 <= y < args.img_size
            pts.append((x, y))
            valid.append(bool(ok))
        if any(valid):
            groups.append((np.asarray(pts, dtype=np.float32), np.asarray(valid, dtype=bool)))
    return groups


def gt_keypoint_panel(img_path, img_np, geom, args):
    out = img_np.copy()
    groups = load_gt_keypoints(img_path, geom, args)
    if not groups:
        return add_title(out, "gt-kpt unavailable")
    for pts, valid in groups:
        for j, xy in enumerate(pts):
            if not valid[j]:
                continue
            x, y = int(round(float(xy[0]))), int(round(float(xy[1])))
            color = KEYPOINT_COLORS[j % len(KEYPOINT_COLORS)]
            cv2.circle(out, (x, y), 4, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(out, (x, y), 2, color, -1, cv2.LINE_AA)
    return add_title(out, "gt-kpt")


def gt_hcce_panel(img_np, source_name="surgpose"):
    out = img_np.copy()
    out = (out.astype(np.float32) * 0.35).astype(np.uint8)
    cv2.putText(
        out,
        f"{source_name}: no HCCE GT",
        (24, max(44, out.shape[0] // 2 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        "no pose/CAD coord_img labels",
        (24, max(82, out.shape[0] // 2 + 30)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return add_title(out, "gt-hcce")


def depth_panel(img_np, preds, args):
    depth_acc = np.full(img_np.shape[:2], np.nan, dtype=np.float32)
    count = np.zeros(img_np.shape[:2], dtype=np.float32)
    for pred in preds:
        if "depth_pred" not in pred:
            continue
        depth = _to_numpy(pred["depth_pred"]).squeeze()
        if depth.ndim != 2:
            continue
        inst = torch.sigmoid(pred["inst_mask_logits"].detach()).cpu().numpy()
        inst = _resize_float(inst, depth.shape) >= float(args.inst_thresh)
        depth = np.where(inst, depth, np.nan).astype(np.float32)
        depth = _resize_float(np.nan_to_num(depth, nan=0.0), img_np.shape[:2])
        inst_full = _resize_nearest(inst.astype(np.uint8), img_np.shape[:2]).astype(bool)
        valid = inst_full & np.isfinite(depth)
        depth_acc[valid] = np.nan_to_num(depth_acc[valid], nan=0.0) + depth[valid]
        count[valid] += 1.0
    valid = count > 0
    if not valid.any():
        return add_title(img_np.copy(), "depth unavailable")
    depth_acc[valid] /= count[valid]
    vals = depth_acc[valid]
    lo, hi = np.percentile(vals, [2, 98])
    if hi <= lo:
        hi = lo + 1e-6
    norm = np.clip((depth_acc - lo) / (hi - lo), 0.0, 1.0)
    color = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    out = img_np.copy()
    out[valid] = color[valid]
    return add_title(out, "depth")


def hcce_x_panel(img_np, preds, model_meta, args):
    x_acc = np.full(img_np.shape[:2], np.nan, dtype=np.float32)
    count = np.zeros(img_np.shape[:2], dtype=np.float32)
    for pred in preds:
        if "hcce_logits" not in pred:
            continue
        try:
            xyz = decoded_xyz_np(pred, model_meta, args)
        except Exception:
            continue
        x = xyz[..., 0].astype(np.float32)
        inst = torch.sigmoid(pred["inst_mask_logits"].detach()).cpu().numpy()
        inst = _resize_float(inst, x.shape) >= float(args.inst_thresh)
        x = np.where(inst, x, np.nan)
        x_full = _resize_float(np.nan_to_num(x, nan=0.0), img_np.shape[:2])
        inst_full = _resize_nearest(inst.astype(np.uint8), img_np.shape[:2]).astype(bool)
        x_acc[inst_full] = np.nan_to_num(x_acc[inst_full], nan=0.0) + x_full[inst_full]
        count[inst_full] += 1.0
    valid = count > 0
    if not valid.any():
        return add_title(img_np.copy(), "hcce-x unavailable")
    x_acc[valid] /= count[valid]
    norm = np.clip((x_acc + 1.0) * 0.5, 0.0, 1.0)
    color = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    out = img_np.copy()
    out[valid] = color[valid]
    return add_title(out, "hcce-x")


def hcce_mesh_panel(geom, cad, poses, preds, args):
    rgb_orig, K, scale, pad_x, pad_y = geom
    mesh_rgb = rgb_orig.copy()
    for pose in poses:
        mesh_rgb = render_mesh_overlay_direct(mesh_rgb, cad, pose, K, args)
    mesh_sq = pad_rgb_to_square(mesh_rgb, args.img_size, scale, pad_x, pad_y)
    return add_title(draw_keypoints_small(mesh_sq, preds), "hcce-trimesh")


def pose_head_mesh_panel(geom, cad, preds, args):
    rgb_orig, K, scale, pad_x, pad_y = geom
    mesh_rgb = rgb_orig.copy()
    used = 0
    for pred in preds:
        pose = pose_head_params_or_none(pred)
        if pose is None:
            continue
        mesh_rgb = render_mesh_overlay_direct(mesh_rgb, cad, pose, K, args)
        used += 1
    mesh_sq = pad_rgb_to_square(mesh_rgb, args.img_size, scale, pad_x, pad_y)
    return add_title(draw_keypoints_small(mesh_sq, preds), "posehead-trimesh" if used else "posehead unavailable")


def make_visualization(img_path, img_np, preds, geom, cad, poses, model_meta, external_instances, args):
    rgb = add_title(draw_keypoints_small(draw_wrist_centers(img_np, preds), preds), "rgb+kpt")
    gt_kpt = gt_keypoint_panel(img_path, img_np, geom, args)
    seg = add_title(model_seg_panel(img_np, preds, args), "model-seg+kpt")
    gt_depth = gt_relative_depth_panel(img_path, img_np, geom, external_instances, args)
    depth = depth_panel(img_np, preds, args)
    gt_hcce = gt_hcce_panel(img_np, source_name=args.dataset_name)
    hcce_x = hcce_x_panel(img_np, preds, model_meta, args)
    if cad is not None:
        hcce_mesh = hcce_mesh_panel(geom, cad, poses, preds, args)
        pose_mesh = pose_head_mesh_panel(geom, cad, preds, args)
    else:
        hcce_mesh = add_title(img_np.copy(), "hcce unavailable")
        pose_mesh = add_title(img_np.copy(), "posehead unavailable")
    sep = np.full((args.img_size, 6, 3), 255, dtype=np.uint8)
    panels = [rgb, gt_kpt, seg, gt_depth, depth, gt_hcce, hcce_x, hcce_mesh, pose_mesh]
    row = panels[0]
    for panel in panels[1:]:
        row = np.concatenate([row, sep, panel], axis=1)
    return row


def worker_main(rank, args, img_paths):
    device_name = args.devices[rank]
    device = configure_device(device_name)
    args.device = device_name
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    model, model_meta = load_demo_model(args, device)
    cad = InstrumentCAD(CAD_ROOT) if model_meta.get("has_hcce", False) else None
    camera_K = load_camera_K(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[worker {rank}] device={device_name} images={len(img_paths)} "
        f"variant={model_meta.get('variant')} iter={model_meta.get('checkpoint_iter')}",
        flush=True,
    )
    rows = []
    for local_idx, img_path in enumerate(img_paths):
        img_tensor, img_np, geom = preprocess(
            img_path, args.img_size, args.focal, args.cx, args.cy, K_override=camera_K
        )
        external_instances = load_external_instances_for_demo(
            img_path,
            geom[0].shape,
            geom,
            args,
            patch_size=getattr(model, "patch_size", 14),
        )
        external_idx = external_idx_from_instances(external_instances, device)
        img_tensor = img_tensor.to(device)
        if external_instances is not None and not external_instances:
            output = []
        else:
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                    output = model(
                        img_tensor,
                        idx=external_idx,
                        is_training=False,
                        det_thresh=args.det_thresh,
                        nms_kernel_size=args.nms_kernel_size,
                    )
        preds = apply_external_instances(normalize_predictions(output), external_instances, args)
        image_seed = args.seed + rank * 100000 + local_idx
        if model_meta.get("has_hcce", False):
            poses, pose_rows = recover_hcce_poses(preds, cad, geom, model_meta, args, image_seed)
        else:
            poses, pose_rows = [], []

        vis = make_visualization(img_path, img_np, preds, geom, cad, poses, model_meta, external_instances, args)
        stem = Path(img_path).stem
        out_path = output_dir / f"{stem}_depth_kpt_hcce_posehead.jpg"
        Image.fromarray(vis).save(out_path)
        row = {
            "image": str(img_path),
            "output": str(out_path),
            "device": device_name,
            "checkpoint": str(args.pretrained),
            "model_variant": model_meta.get("variant"),
            "checkpoint_iter": model_meta.get("checkpoint_iter"),
            "num_predictions": len(preds),
            "external_instance_mode": external_instances is not None,
            "external_union_as_instance": bool(args.external_union_as_instance),
                "external_instance_count": 0 if external_instances is None else len(external_instances),
            "num_hcce_poses": len(poses),
            "pose_recovery": pose_rows,
            "max_points_per_part": int(args.max_points_per_part),
            "point_select": args.point_select,
            "external_part_dir": str(args.external_part_dir),
        }
        (output_dir / f"{stem}_depth_kpt_hcce_posehead.json").write_text(
            json.dumps(row, indent=2), encoding="utf-8"
        )
        rows.append(row)
        print(
            f"[worker {rank}] {Path(img_path).name}: "
            f"{len(preds)} det, {len(poses)} hcce pose -> {out_path.name}",
            flush=True,
        )
    (output_dir / f"worker_{rank}_manifest.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )


def run(args):
    img_paths = image_paths_from_input(args.input, args.recursive, args.max_images)
    if not img_paths:
        raise RuntimeError(f"No images found: {args.input}")
    args.devices = args.devices or [args.device]
    args.devices = [d for d in args.devices if d]
    if not args.devices:
        args.devices = ["cuda:7" if torch.cuda.is_available() else "cpu"]
    chunks = [chunk for chunk in split_evenly(img_paths, len(args.devices)) if chunk]
    print(
        f"Processing {len(img_paths)} image(s) -> {args.output_dir}; "
        f"devices={args.devices}; checkpoint={args.pretrained}; "
        f"max_points_per_part={args.max_points_per_part}",
        flush=True,
    )
    if len(chunks) == 1:
        args.devices = [args.devices[0]]
        worker_main(0, args, chunks[0])
        return

    ctx = mp.get_context("spawn")
    procs = []
    for rank, chunk in enumerate(chunks):
        proc = ctx.Process(target=worker_main, args=(rank, args, chunk))
        proc.start()
        procs.append(proc)
    failed = []
    for proc in procs:
        proc.join()
        if proc.exitcode != 0:
            failed.append(proc.exitcode)
    if failed:
        raise RuntimeError(f"Worker failure exit codes: {failed}")


def build_parser():
    parser = ArgumentParser()
    parser.add_argument("--input", type=str, default=str(SURGPOSE_LEFT_FRAMES))
    parser.add_argument(
        "--output_dir",
        type=str,
        default="demo_outputs/depth_keypoint_pose_demo",
    )
    parser.add_argument("--pretrained", type=str, default=DEPTH_CKPT)
    parser.add_argument("--external_inst_dir", type=str, default=str(SURGPOSE_SAM3_SEGMENTATION))
    parser.add_argument("--external_part_dir", type=str, default=str(SURGPOSE_PART_SEG))
    parser.add_argument("--require_external_inst", type=int, default=0, choices=[0, 1])
    parser.add_argument("--external_union_as_instance", type=int, default=0, choices=[0, 1])
    parser.add_argument("--external_min_area", type=int, default=64)
    parser.add_argument("--camera_metadata", type=str, default=str(SURGPOSE_METADATA))
    parser.add_argument("--camera_key", type=str, default="K_left_rectified_scaled")
    parser.add_argument("--dataset_name", type=str, default="surgpose")
    parser.add_argument(
        "--gt_depth_dir",
        type=str,
        default="/mnt/nas/share/shuojue/data/surgpose/000000/processed_stereo_640/depth_npy",
    )
    parser.add_argument("--gt_depth_norm_low_pct", type=float, default=2.0)
    parser.add_argument("--gt_depth_norm_high_pct", type=float, default=98.0)
    parser.add_argument(
        "--gt_keypoint_file",
        type=str,
        default="/mnt/nas/share/shuojue/data/surgpose/000000/processed_stereo_640/keypoints_left_rectified.yaml",
    )
    parser.add_argument("--img_size", type=int, default=630)
    parser.add_argument("--det_thresh", type=float, default=0.5)
    parser.add_argument("--nms_kernel_size", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--amp", type=int, default=1, choices=[0, 1])
    parser.add_argument("--device", type=str, default="cuda:7")
    parser.add_argument("--devices", nargs="*", default=["cuda:7"])
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--max_images", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser = add_pose_recovery_args(parser)
    parser.set_defaults(max_points_per_part=2048, point_select="top", optim_strategy="single", optim_parts="all")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
