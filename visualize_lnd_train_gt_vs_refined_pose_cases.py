#!/usr/bin/env python3
"""Visualize excluded LND TRAIN poses before and after refine-memory fitting."""

import argparse
import csv
import importlib.util
import json
import os
import re
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import numpy as np
from PIL import Image

from instrument_opengl_renderer import InstrumentOpenGLDepthRenderer


ROOT = Path(__file__).resolve().parent
MULTIHMR_ROOT = ROOT.parents[1]
DEFAULT_LND_ROOT = Path("/mnt/iMVR/daiyun/Dataset/LND")
DEFAULT_ORIGINAL_MEMORY = (
    MULTIHMR_ROOT
    / "submodules/gaussian-mesh-splatting/Results2/surgripe_lnd_action_gt_full/TRAIN/memory_pool.json"
)
DEFAULT_REFINED_MEMORY = (
    MULTIHMR_ROOT
    / "submodules/gaussian-mesh-splatting/Results2/surgripe_lnd_refine_memory_train/TRAIN/refine_memory_pool.json"
)
DEFAULT_OUTPUT = ROOT / "logs/lnd_train_gt_vs_refined_excluded_cases"
DEFAULT_FRAME_IDS = (176, 340, 408, 713, 778, 779, 1024, 1125)
NO_WRIST_PIXEL_IDS = {340, 408, 779, 1125}
LOW_IOU_IDS = {176, 713, 778, 1024}

# RGB colors. Rendered part IDs are shaft=1, effective wrist=2, gripper=3.
PART_COLORS = {
    1: np.array([59, 130, 246], dtype=np.uint8),
    2: np.array([34, 197, 94], dtype=np.uint8),
    3: np.array([249, 115, 22], dtype=np.uint8),
}
WHITE = (245, 245, 245)
BLACK = (12, 12, 12)


def load_canonicalizer():
    path = MULTIHMR_ROOT / "datasets/rarp_pose_canonicalization.py"
    spec = importlib.util.spec_from_file_location("lnd_pose_vis_canonicalization", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load pose canonicalization from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.canonicalize_pose_symmetry


CANONICALIZE_POSE = load_canonicalizer()


def load_intrinsics(split_root):
    text = (Path(split_root) / "config.yaml").read_text(encoding="utf-8")
    match = re.search(r"camera_matrix:.*?data:\s*\[([^\]]+)\]", text, flags=re.S)
    if not match:
        raise RuntimeError(f"Could not parse camera intrinsics under {split_root}")
    values = [float(v) for v in re.split(r"[,\s]+", match.group(1).strip()) if v]
    if len(values) != 9:
        raise RuntimeError(f"Expected 9 intrinsic values, got {len(values)}")
    return np.asarray(values, dtype=np.float32).reshape(3, 3)


def load_memory(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def pose_from_record(record):
    wrist = record["wrist_pose"]
    action = record.get("action", {})
    pose = {
        "rot": np.asarray(wrist["quat_wxyz"], dtype=np.float32),
        "trans": np.asarray(wrist["trans_m"], dtype=np.float32),
        "alpha": float(action.get("alpha", 0.0)),
        "theta_l": float(action.get("theta_l", 0.0)),
        "theta_r": float(action.get("theta_r", 0.0)),
    }
    canonical = CANONICALIZE_POSE(pose, eps=0.08, enabled=True)
    return {
        "rot": canonical["rot"].detach().cpu().numpy().astype(np.float32),
        "trans": canonical["trans"].detach().cpu().numpy().astype(np.float32),
        "alpha": float(canonical["alpha"].reshape(-1)[0]),
        "theta_l": float(canonical["theta_l"].reshape(-1)[0]),
        "theta_r": float(canonical["theta_r"].reshape(-1)[0]),
        "pose_sym_flipped": bool(canonical["pose_sym_flipped"].item()),
    }


def resolve_visible_mask(split_root, frame_id):
    candidates = [
        split_root / "mask_original" / f"{frame_id}.png",
        split_root / "mask_original" / f"{frame_id:06d}.png",
        split_root / "mask visible" / f"{frame_id}.png",
        split_root / "mask visible" / f"{frame_id:06d}.png",
        split_root / "mask visible" / f"{frame_id:06d}_000000.png",
    ]
    for path in candidates:
        if path.is_file():
            return path
    matches = sorted((split_root / "mask visible").glob(f"{frame_id:06d}_*.png"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No visible mask found for LND TRAIN frame {frame_id}")


def load_case(split_root, frame_id):
    rgb = np.asarray(Image.open(split_root / "image" / f"{frame_id}.png").convert("RGB"))
    lnd_part = np.asarray(
        Image.open(split_root / "sam3_segmentaion_part" / f"{frame_id}.png").convert("L")
    )
    visible = np.asarray(Image.open(resolve_visible_mask(split_root, frame_id)).convert("L")) > 0
    instance = np.asarray(
        Image.open(split_root / "sam3_segmentation" / f"{frame_id}.png").convert("L")
    ) > 0

    # Match RoboPEPPSurgripeLNDInstrument._part_mask_rarp_style exactly.
    part = np.zeros_like(lnd_part, dtype=np.uint8)
    part[lnd_part == 1] = 3
    part[lnd_part == 2] = 2
    part[lnd_part == 3] = 1
    part[(part == 2) & ~visible] = 0
    return rgb, part, instance


def mask_iou(a, b):
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    union = np.count_nonzero(a | b)
    return float(np.count_nonzero(a & b) / union) if union else float("nan")


def rotation_error_deg(pose_a, pose_b):
    qa = np.asarray(pose_a["rot"], dtype=np.float64)
    qb = np.asarray(pose_b["rot"], dtype=np.float64)
    qa /= np.linalg.norm(qa).clip(1e-12)
    qb /= np.linalg.norm(qb).clip(1e-12)
    dot = np.clip(abs(float(np.dot(qa, qb))), 0.0, 1.0)
    return float(np.degrees(2.0 * np.arccos(dot)))


def blend_parts(rgb, part_mask, alpha=0.46):
    out = np.asarray(rgb, dtype=np.uint8).copy()
    for part_id, color in PART_COLORS.items():
        support = np.asarray(part_mask) == part_id
        out[support] = np.clip(
            (1.0 - alpha) * out[support].astype(np.float32) + alpha * color,
            0,
            255,
        ).astype(np.uint8)
        contours, _ = cv2.findContours(
            support.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(out, contours, -1, tuple(int(v) for v in color), 2, cv2.LINE_AA)
    return out


def wrist_overlap_panel(rgb, gt_wrist, rendered_wrist):
    out = np.asarray(rgb, dtype=np.uint8).copy()
    gt = np.asarray(gt_wrist, dtype=bool)
    rendered = np.asarray(rendered_wrist, dtype=bool)
    overlap = gt & rendered
    gt_only = gt & ~rendered
    render_only = rendered & ~gt
    colors = (
        (overlap, np.array([40, 220, 90], dtype=np.uint8)),
        (gt_only, np.array([250, 210, 35], dtype=np.uint8)),
        (render_only, np.array([235, 55, 70], dtype=np.uint8)),
    )
    for support, color in colors:
        out[support] = np.clip(
            0.42 * out[support].astype(np.float32) + 0.58 * color,
            0,
            255,
        ).astype(np.uint8)
    for support, color in colors:
        contours, _ = cv2.findContours(
            support.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(out, contours, -1, tuple(int(v) for v in color), 1, cv2.LINE_AA)
    return out


def square_roi(mask, padding=28):
    ys, xs = np.where(np.asarray(mask, dtype=bool))
    h, w = mask.shape[:2]
    if xs.size == 0:
        return 0, 0, w, h
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    side = max(x1 - x0, y1 - y0) + 2 * int(padding)
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    side = min(side, max(h, w))
    x0 = int(round(cx - side / 2.0))
    y0 = int(round(cy - side / 2.0))
    x0 = min(max(0, x0), max(0, w - side))
    y0 = min(max(0, y0), max(0, h - side))
    x1 = min(w, x0 + side)
    y1 = min(h, y0 + side)
    return x0, y0, x1, y1


def crop_resize(image, roi, panel_size):
    x0, y0, x1, y1 = roi
    crop = np.asarray(image)[y0:y1, x0:x1]
    return cv2.resize(crop, (panel_size, panel_size), interpolation=cv2.INTER_CUBIC)


def label_panel(image, title, subtitle):
    image = np.asarray(image, dtype=np.uint8)
    header = np.full((66, image.shape[1], 3), 248, dtype=np.uint8)
    def draw_fitted(text, y, initial_scale):
        scale = float(initial_scale)
        while scale > 0.25:
            width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
            if width <= header.shape[1] - 16:
                break
            scale -= 0.02
        cv2.putText(header, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, scale, BLACK, 1, cv2.LINE_AA)

    draw_fitted(title, 23, 0.55)
    draw_fitted(subtitle, 49, 0.39)
    return np.concatenate((header, image), axis=0)


def render_training_parts(renderer, pose, K, image_shape):
    _, part, _, valid = renderer.render_canonical_coordinates(pose, K, image_shape)
    return np.where(valid, part, 0).astype(np.uint8)


def join_row(panels, gap=8):
    separator = np.full((panels[0].shape[0], gap, 3), 255, dtype=np.uint8)
    out = panels[0]
    for panel in panels[1:]:
        out = np.concatenate((out, separator, panel), axis=1)
    return out


def make_frame_panel(
    frame_id,
    rgb,
    gt_part,
    instance,
    original_mask,
    refined_mask,
    original_pose,
    refined_pose,
    panel_size,
):
    gt_wrist = gt_part == 2
    original_wrist = original_mask == 2
    refined_wrist = refined_mask == 2
    original_iou = mask_iou(gt_wrist, original_wrist)
    refined_iou = mask_iou(gt_wrist, refined_wrist)
    delta_r = rotation_error_deg(original_pose, refined_pose)
    delta_t = float(np.linalg.norm(refined_pose["trans"] - original_pose["trans"]) * 1000.0)
    delta_alpha = float(np.degrees(refined_pose["alpha"] - original_pose["alpha"]))

    support = gt_wrist | original_wrist | refined_wrist
    roi = square_roi(support, padding=52)
    if frame_id in NO_WRIST_PIXEL_IDS:
        group = "no GT wrist pixels"
    elif frame_id in LOW_IOU_IDS:
        group = "reported low-IoU case"
    else:
        group = "representative TRAIN case"

    top = join_row(
        [
            label_panel(
                crop_resize(blend_parts(rgb, gt_part), roi, panel_size),
                "SAM-visible GT parts",
                f"frame {frame_id} | wrist_px={int(gt_wrist.sum())} | {group}",
            ),
            label_panel(
                crop_resize(blend_parts(rgb, original_mask), roi, panel_size),
                "Original TRAIN GT pose",
                f"effective wrist render_px={int(original_wrist.sum())}",
            ),
            label_panel(
                crop_resize(blend_parts(rgb, refined_mask), roi, panel_size),
                "Refined pseudo pose",
                f"dR={delta_r:.2f}deg dT={delta_t:.2f}mm dAlpha={delta_alpha:.2f}deg",
            ),
        ]
    )
    bottom = join_row(
        [
            label_panel(
                crop_resize(blend_parts(rgb, np.where(gt_wrist, 2, 0)), roi, panel_size),
                "SAM-visible wrist supervision",
                "green = SAM-visible wrist",
            ),
            label_panel(
                crop_resize(wrist_overlap_panel(rgb, gt_wrist, original_wrist), roi, panel_size),
                "Original wrist overlap",
                f"IoU={original_iou:.3f} | green both | yellow GT-only | red pose-only",
            ),
            label_panel(
                crop_resize(wrist_overlap_panel(rgb, gt_wrist, refined_wrist), roi, panel_size),
                "Refined wrist overlap",
                f"IoU={refined_iou:.3f} | green both | yellow GT-only | red pose-only",
            ),
        ]
    )
    separator = np.full((8, top.shape[1], 3), 255, dtype=np.uint8)
    panel = np.concatenate((top, separator, bottom), axis=0)
    stats = {
        "frame_id": frame_id,
        "group": group,
        "gt_wrist_pixels": int(gt_wrist.sum()),
        "original_render_wrist_pixels": int(original_wrist.sum()),
        "refined_render_wrist_pixels": int(refined_wrist.sum()),
        "original_wrist_iou": original_iou,
        "refined_wrist_iou": refined_iou,
        "rot_delta_deg": delta_r,
        "trans_delta_mm": delta_t,
        "alpha_delta_deg": delta_alpha,
        "theta_l_delta_deg": float(np.degrees(refined_pose["theta_l"] - original_pose["theta_l"])),
        "theta_r_delta_deg": float(np.degrees(refined_pose["theta_r"] - original_pose["theta_r"])),
    }
    return panel, stats


def make_contact_sheet(paths, output_path, max_width=1800):
    images = [np.asarray(Image.open(path).convert("RGB")) for path in paths]
    resized = []
    for image in images:
        scale = min(1.0, float(max_width) / float(image.shape[1]))
        resized.append(
            cv2.resize(
                image,
                (int(round(image.shape[1] * scale)), int(round(image.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
        )
    gap = np.full((10, resized[0].shape[1], 3), 255, dtype=np.uint8)
    sheet = resized[0]
    for image in resized[1:]:
        sheet = np.concatenate((sheet, gap, image), axis=0)
    Image.fromarray(sheet).save(output_path, quality=94)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lnd_root", type=Path, default=DEFAULT_LND_ROOT)
    parser.add_argument("--original_memory", type=Path, default=DEFAULT_ORIGINAL_MEMORY)
    parser.add_argument("--refined_memory", type=Path, default=DEFAULT_REFINED_MEMORY)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frame_ids", type=int, nargs="+", default=DEFAULT_FRAME_IDS)
    parser.add_argument("--panel_size", type=int, default=360)
    parser.add_argument("--egl_device", type=int, default=0)
    args = parser.parse_args()

    split_root = args.lnd_root / "TRAIN"
    K = load_intrinsics(split_root)
    original_memory = load_memory(args.original_memory)
    refined_memory = load_memory(args.refined_memory)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    renderer = InstrumentOpenGLDepthRenderer(w=640, h=480, device_idx=args.egl_device)

    rows = []
    all_paths = []
    group_paths = {"no_wrist_pixels": [], "reported_low_iou": [], "representative": []}
    try:
        for frame_id in args.frame_ids:
            key = str(int(frame_id))
            if key not in original_memory or key not in refined_memory:
                raise KeyError(f"Frame {frame_id} missing from one of the memory pools")
            rgb, gt_part, instance = load_case(split_root, int(frame_id))
            original_pose = pose_from_record(original_memory[key])
            refined_pose = pose_from_record(refined_memory[key])
            original_mask = render_training_parts(renderer, original_pose, K, rgb.shape[:2])
            refined_mask = render_training_parts(renderer, refined_pose, K, rgb.shape[:2])
            panel, stats = make_frame_panel(
                int(frame_id),
                rgb,
                gt_part,
                instance,
                original_mask,
                refined_mask,
                original_pose,
                refined_pose,
                args.panel_size,
            )
            path = args.output_dir / f"frame_{int(frame_id):04d}_gt_vs_refined.jpg"
            Image.fromarray(panel).save(path, quality=96)
            stats["visualization"] = str(path.resolve())
            rows.append(stats)
            all_paths.append(path)
            if int(frame_id) in NO_WRIST_PIXEL_IDS:
                group = "no_wrist_pixels"
            elif int(frame_id) in LOW_IOU_IDS:
                group = "reported_low_iou"
            else:
                group = "representative"
            group_paths[group].append(path)
            print(
                f"frame={frame_id} gt_px={stats['gt_wrist_pixels']} "
                f"original_iou={stats['original_wrist_iou']:.4f} "
                f"refined_iou={stats['refined_wrist_iou']:.4f} "
                f"dR={stats['rot_delta_deg']:.3f}deg dT={stats['trans_delta_mm']:.3f}mm",
                flush=True,
            )
    finally:
        renderer.release()

    csv_path = args.output_dir / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, allow_nan=True)
    if all_paths:
        make_contact_sheet(all_paths, args.output_dir / "contact_sheet_all.jpg")
    for group, paths in group_paths.items():
        if paths:
            make_contact_sheet(paths, args.output_dir / f"contact_sheet_{group}.jpg")
    print(f"Saved {len(rows)} cases to {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
