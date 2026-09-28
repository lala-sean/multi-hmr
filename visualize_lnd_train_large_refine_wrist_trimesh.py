#!/usr/bin/env python3
"""Render large LND TRAIN wrist-pose changes with the actual wrist CAD mesh."""

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import numpy as np
from PIL import Image
import pyrender
import torch

import visualize_lnd_train_gt_vs_refined_pose_cases as base
from instrument_geometry import fk_matrices_np
from instrument_trimesh_renderer import GMSInstrumentTrimeshRenderer


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "logs/lnd_train_refine_large_wrist_trimesh"
ORIGINAL_COLOR = (36, 218, 112, 255)
REFINED_COLOR = (246, 70, 174, 255)


class WristTrimeshRenderer:
    """Rasterize the Instrument-Splatting wrist Trimesh with camera intrinsics."""

    def __init__(self):
        source = GMSInstrumentTrimeshRenderer(torch.device("cpu"))
        self.wrist = source.part_meshes["wrist"].copy()

    @staticmethod
    def _camera_pose():
        pose = np.eye(4, dtype=np.float64)
        pose[:, 1:3] *= -1.0
        return pose

    def render(self, pose, K, image_shape, rgba):
        transforms = fk_matrices_np(
            pose["rot"],
            pose["trans"],
            pose["alpha"],
            pose["theta_l"],
            pose["theta_r"],
        )
        mesh = self.wrist.copy()
        mesh.apply_transform(np.asarray(transforms["wrist"], dtype=np.float64))

        color = tuple(float(value) / 255.0 for value in rgba)
        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=color,
            metallicFactor=0.08,
            roughnessFactor=0.58,
            doubleSided=False,
            smooth=False,
        )
        scene = pyrender.Scene(
            bg_color=(0.0, 0.0, 0.0, 0.0),
            ambient_light=(0.34, 0.34, 0.34),
        )
        scene.add(pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False))

        K = np.asarray(K, dtype=np.float64)
        camera_pose = self._camera_pose()
        scene.add(
            pyrender.IntrinsicsCamera(
                fx=float(K[0, 0]),
                fy=float(K[1, 1]),
                cx=float(K[0, 2]),
                cy=float(K[1, 2]),
                znear=0.001,
                zfar=1.0,
            ),
            pose=camera_pose,
        )
        scene.add(
            pyrender.DirectionalLight(color=np.ones(3), intensity=1.25),
            pose=camera_pose,
        )

        height, width = int(image_shape[0]), int(image_shape[1])
        renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
        try:
            rendered_rgba, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        finally:
            renderer.delete()
        return {
            "rgb": np.asarray(rendered_rgba[..., :3], dtype=np.uint8),
            "valid": np.asarray(depth) > 0.0,
        }


def overlay_mesh(rgb, rendered, alpha=0.88):
    out = np.asarray(rgb, dtype=np.uint8).copy()
    valid = np.asarray(rendered["valid"], dtype=bool)
    out[valid] = np.clip(
        (1.0 - float(alpha)) * out[valid].astype(np.float32)
        + float(alpha) * rendered["rgb"][valid].astype(np.float32),
        0,
        255,
    ).astype(np.uint8)
    return out


def pose_rows(original_memory, refined_memory):
    rows = []
    frame_ids = sorted(set(map(int, original_memory)) & set(map(int, refined_memory)))
    for frame_id in frame_ids:
        key = str(frame_id)
        original = base.pose_from_record(original_memory[key])
        refined = base.pose_from_record(refined_memory[key])
        rows.append(
            {
                "frame_id": frame_id,
                "trans_delta_mm": float(
                    np.linalg.norm(refined["trans"] - original["trans"]) * 1000.0
                ),
                "rot_delta_deg": base.rotation_error_deg(original, refined),
                "original_pose": original,
                "refined_pose": refined,
            }
        )
    trans_scale = max(float(np.percentile([r["trans_delta_mm"] for r in rows], 95)), 1e-6)
    rot_scale = max(float(np.percentile([r["rot_delta_deg"] for r in rows], 95)), 1e-6)
    for row in rows:
        row["joint_score"] = float(
            np.hypot(row["trans_delta_mm"] / trans_scale, row["rot_delta_deg"] / rot_scale)
        )
    return rows


def select_rows(rows, top_translation, top_rotation, top_joint):
    selected = {}
    groups = (
        ("T", "trans_delta_mm", int(top_translation)),
        ("R", "rot_delta_deg", int(top_rotation)),
        ("B", "joint_score", int(top_joint)),
    )
    for tag, key, count in groups:
        for row in sorted(rows, key=lambda item: item[key], reverse=True)[:count]:
            frame_id = int(row["frame_id"])
            if frame_id not in selected:
                selected[frame_id] = {**row, "selected_by": []}
            selected[frame_id]["selected_by"].append(tag)
    return sorted(selected.values(), key=lambda item: item["joint_score"], reverse=True)


def make_case_panel(row, rgb, gt_wrist, K, renderer, panel_size):
    original = renderer.render(row["original_pose"], K, rgb.shape[:2], ORIGINAL_COLOR)
    refined = renderer.render(row["refined_pose"], K, rgb.shape[:2], REFINED_COLOR)
    roi = base.square_roi(original["valid"] | refined["valid"] | gt_wrist, padding=42)
    frame_id = int(row["frame_id"])
    tags = "/".join(row["selected_by"])
    panels = [
        base.label_panel(
            base.crop_resize(rgb, roi, panel_size),
            f"Frame {frame_id} | selected {tags}",
            "LND TRAIN RGB | shared ROI",
        ),
        base.label_panel(
            base.crop_resize(overlay_mesh(rgb, original), roi, panel_size),
            "Original converted GT wrist pose",
            "green = shaded wrist CAD triangles",
        ),
        base.label_panel(
            base.crop_resize(overlay_mesh(rgb, refined), roi, panel_size),
            "Refined wrist pose",
            f"dT={row['trans_delta_mm']:.2f}mm | dR={row['rot_delta_deg']:.2f}deg",
        ),
    ]
    panel = base.join_row(panels, gap=8)
    return panel, [int(value) for value in roi]


def make_grid(paths, output_path, columns=2, gap=12):
    images = [np.asarray(Image.open(path).convert("RGB")) for path in paths]
    if not images:
        raise ValueError("Cannot build a contact sheet without images")
    height = max(image.shape[0] for image in images)
    width = max(image.shape[1] for image in images)
    rows = (len(images) + int(columns) - 1) // int(columns)
    canvas = np.full(
        (rows * height + (rows - 1) * gap, columns * width + (columns - 1) * gap, 3),
        255,
        dtype=np.uint8,
    )
    for index, image in enumerate(images):
        row, column = divmod(index, int(columns))
        y = row * (height + gap)
        x = column * (width + gap)
        canvas[y : y + image.shape[0], x : x + image.shape[1]] = image
    Image.fromarray(canvas).save(output_path, quality=96)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lnd_root", type=Path, default=base.DEFAULT_LND_ROOT)
    parser.add_argument("--original_memory", type=Path, default=base.DEFAULT_ORIGINAL_MEMORY)
    parser.add_argument("--refined_memory", type=Path, default=base.DEFAULT_REFINED_MEMORY)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top_translation", type=int, default=12)
    parser.add_argument("--top_rotation", type=int, default=12)
    parser.add_argument("--top_joint", type=int, default=12)
    parser.add_argument("--panel_size", type=int, default=300)
    parser.add_argument("--columns", type=int, default=2)
    args = parser.parse_args()

    split_root = args.lnd_root / "TRAIN"
    K = base.load_intrinsics(split_root)
    original_memory = base.load_memory(args.original_memory)
    refined_memory = base.load_memory(args.refined_memory)
    rows = pose_rows(original_memory, refined_memory)
    selected = select_rows(rows, args.top_translation, args.top_rotation, args.top_joint)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    renderer = WristTrimeshRenderer()
    image_paths = []
    summary_rows = []
    for rank, row in enumerate(selected, start=1):
        frame_id = int(row["frame_id"])
        rgb, gt_part, _ = base.load_case(split_root, frame_id)
        panel, roi = make_case_panel(row, rgb, gt_part == 2, K, renderer, args.panel_size)
        path = args.output_dir / f"rank_{rank:02d}_frame_{frame_id:04d}_wrist_trimesh.jpg"
        Image.fromarray(panel).save(path, quality=97)
        image_paths.append(path)
        summary_rows.append(
            {
                "rank": rank,
                "frame_id": frame_id,
                "selected_by": "/".join(row["selected_by"]),
                "trans_delta_mm": row["trans_delta_mm"],
                "rot_delta_deg": row["rot_delta_deg"],
                "joint_score": row["joint_score"],
                "roi_xyxy": roi,
                "image": str(path.resolve()),
            }
        )
        print(
            f"rank={rank:02d} frame={frame_id:04d} selected={summary_rows[-1]['selected_by']} "
            f"dT={row['trans_delta_mm']:.3f}mm dR={row['rot_delta_deg']:.3f}deg",
            flush=True,
        )

    sheet_path = args.output_dir / "contact_sheet_large_wrist_pose_changes.jpg"
    make_grid(image_paths, sheet_path, columns=args.columns)
    with (args.output_dir / "selected_cases.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    with (args.output_dir / "selected_cases.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_rows, handle, indent=2)
    print(f"Saved {len(summary_rows)} cases and contact sheet to {sheet_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
