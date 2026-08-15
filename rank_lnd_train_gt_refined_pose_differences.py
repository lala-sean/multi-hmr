#!/usr/bin/env python3
"""Rank LND TRAIN frames by GT-to-refined render and pose differences."""

import argparse
import csv
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np

import visualize_lnd_train_gt_vs_refined_pose_cases as vis


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "logs/lnd_train_gt_vs_refined_large_differences"


def rank_worker(payload):
    frame_ids, device_idx, original_path, refined_path, K, image_shape = payload
    original_memory = vis.load_memory(original_path)
    refined_memory = vis.load_memory(refined_path)
    renderer = vis.InstrumentOpenGLDepthRenderer(
        w=int(image_shape[1]),
        h=int(image_shape[0]),
        device_idx=int(device_idx),
    )
    rows = []
    try:
        for frame_id in frame_ids:
            key = str(int(frame_id))
            original_pose = vis.pose_from_record(original_memory[key])
            refined_pose = vis.pose_from_record(refined_memory[key])
            original_part = vis.render_training_parts(renderer, original_pose, K, image_shape)
            refined_part = vis.render_training_parts(renderer, refined_pose, K, image_shape)

            original_fg = original_part > 0
            refined_fg = refined_part > 0
            fg_union = original_fg | refined_fg
            fg_intersection = original_fg & refined_fg
            semantic_changed = fg_union & (original_part != refined_part)
            wrist_union = (original_part == 2) | (refined_part == 2)
            wrist_changed = wrist_union & ((original_part == 2) != (refined_part == 2))
            union_px = int(fg_union.sum())
            wrist_union_px = int(wrist_union.sum())
            delta = refined_memory[key].get("refine_delta", {})
            rows.append(
                {
                    "frame_id": int(frame_id),
                    "render_union_pixels": union_px,
                    "original_render_pixels": int(original_fg.sum()),
                    "refined_render_pixels": int(refined_fg.sum()),
                    "foreground_iou": float(fg_intersection.sum() / max(1, union_px)),
                    "semantic_changed_pixels": int(semantic_changed.sum()),
                    "semantic_changed_fraction": float(semantic_changed.sum() / max(1, union_px)),
                    "wrist_union_pixels": wrist_union_px,
                    "wrist_changed_pixels": int(wrist_changed.sum()),
                    "wrist_changed_fraction": float(wrist_changed.sum() / max(1, wrist_union_px)),
                    "rot_delta_deg": float(delta.get("rot_delta_deg", np.nan)),
                    "trans_delta_mm": float(delta.get("trans_delta_norm_m", np.nan)) * 1000.0,
                    "alpha_delta_deg": float(delta.get("alpha_delta_deg", np.nan)),
                }
            )
    finally:
        renderer.release()
    return rows


def top_unique(rows, key, count, minimum_union_pixels=0):
    eligible = [
        row
        for row in rows
        if int(row["render_union_pixels"]) >= int(minimum_union_pixels)
        and np.isfinite(float(row[key]))
    ]
    return sorted(eligible, key=lambda row: float(row[key]), reverse=True)[: int(count)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lnd_root", type=Path, default=vis.DEFAULT_LND_ROOT)
    parser.add_argument("--original_memory", type=Path, default=vis.DEFAULT_ORIGINAL_MEMORY)
    parser.add_argument("--refined_memory", type=Path, default=vis.DEFAULT_REFINED_MEMORY)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--devices", type=int, nargs="+", default=(0, 1, 2, 3))
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--min_render_union_pixels", type=int, default=500)
    args = parser.parse_args()

    original_memory = vis.load_memory(args.original_memory)
    refined_memory = vis.load_memory(args.refined_memory)
    frame_ids = sorted(set(map(int, original_memory)) & set(map(int, refined_memory)))
    if not frame_ids:
        raise RuntimeError("No shared frame IDs in original and refined memory pools")
    split_root = args.lnd_root / "TRAIN"
    K = vis.load_intrinsics(split_root)
    first_rgb, _, _ = vis.load_case(split_root, frame_ids[0])
    image_shape = tuple(first_rgb.shape[:2])

    chunks = [list(map(int, chunk)) for chunk in np.array_split(frame_ids, len(args.devices)) if len(chunk)]
    payloads = [
        (
            chunk,
            args.devices[index % len(args.devices)],
            str(args.original_memory),
            str(args.refined_memory),
            K,
            image_shape,
        )
        for index, chunk in enumerate(chunks)
    ]
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=len(payloads), mp_context=context) as executor:
        worker_rows = list(executor.map(rank_worker, payloads))
    rows = sorted((row for batch in worker_rows for row in batch), key=lambda row: row["frame_id"])

    selections = {
        "large_seg_difference": top_unique(
            rows,
            "semantic_changed_fraction",
            args.top_k,
            minimum_union_pixels=args.min_render_union_pixels,
        ),
        "large_rotation_difference": top_unique(rows, "rot_delta_deg", args.top_k),
        "large_translation_difference": top_unique(rows, "trans_delta_mm", args.top_k),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "all_frame_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "selected_cases.json").open("w", encoding="utf-8") as handle:
        json.dump(selections, handle, indent=2)

    for group, selected in selections.items():
        print(group, flush=True)
        for row in selected:
            print(
                f"  frame={row['frame_id']} seg_changed={row['semantic_changed_fraction']:.4f} "
                f"wrist_changed={row['wrist_changed_fraction']:.4f} "
                f"dR={row['rot_delta_deg']:.3f}deg dT={row['trans_delta_mm']:.3f}mm",
                flush=True,
            )
    print(f"Saved ranking to {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
