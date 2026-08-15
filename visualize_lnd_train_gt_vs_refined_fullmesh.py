#!/usr/bin/env python3
"""Render true trimesh geometry before and after LND TRAIN pose refinement."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import numpy as np
from PIL import Image
import pyrender
import torch
import trimesh

import visualize_lnd_train_gt_vs_refined_pose_cases as base
from instrument_geometry import (
    GRIPPER_JOINT_OFFSET_M,
    SURFEMB_GRIPPER_STATIC_THRESHOLD_M,
    fk_matrices_np,
    make_transform_np,
)
from instrument_opengl_renderer import _split_gripper_mesh_vertices
from instrument_trimesh_renderer import GMSInstrumentTrimeshRenderer


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = (
    ROOT
    / "logs/lnd_train_gt_vs_refined_large_differences/large_translation_difference_trimesh"
)
DEFAULT_FRAME_IDS = (341, 402, 438, 894, 387, 806, 430, 425)

# RGBA part materials. Static gripper rear triangles use the wrist material.
PART_COLORS = {
    "shaft": (52, 128, 235, 255),
    "wrist": (42, 196, 92, 255),
    "l_gripper": (245, 123, 34, 255),
    "r_gripper": (224, 64, 142, 255),
}


def mesh_from_unindexed_triangles(vertices):
    vertices = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    if len(vertices) == 0 or len(vertices) % 3:
        raise ValueError(f"Invalid unindexed triangle array: {vertices.shape}")
    faces = np.arange(len(vertices), dtype=np.int64).reshape(-1, 3)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


class PartTrimeshRenderer:
    """Pyrender renderer over actual part Trimesh geometry and triangle faces."""

    def __init__(self):
        base_renderer = GMSInstrumentTrimeshRenderer(torch.device("cpu"))
        self.source_meshes = {
            "shaft": base_renderer.part_meshes["shaft"].copy(),
            "wrist": base_renderer.part_meshes["wrist"].copy(),
        }
        self.gripper_meshes = {}
        for name in ("l_gripper", "r_gripper"):
            static_vertices, moving_vertices = _split_gripper_mesh_vertices(
                base_renderer.part_meshes[name],
                threshold=SURFEMB_GRIPPER_STATIC_THRESHOLD_M,
            )
            self.gripper_meshes[name] = {
                "static": mesh_from_unindexed_triangles(static_vertices),
                "moving": mesh_from_unindexed_triangles(moving_vertices),
            }

    @staticmethod
    def transformed(mesh, transform):
        result = mesh.copy()
        result.apply_transform(np.asarray(transform, dtype=np.float64))
        return result

    def world_meshes(self, pose):
        transforms = fk_matrices_np(
            pose["rot"], pose["trans"], pose["alpha"], pose["theta_l"], pose["theta_r"]
        )
        wrist_to_gripper = make_transform_np(
            np.eye(3, dtype=np.float64),
            [GRIPPER_JOINT_OFFSET_M, 0.0, 0.0],
        )
        meshes = [
            ("shaft", self.transformed(self.source_meshes["shaft"], transforms["shaft"])),
            ("wrist", self.transformed(self.source_meshes["wrist"], transforms["wrist"])),
        ]
        for name in ("l_gripper", "r_gripper"):
            meshes.append(
                (
                    "wrist",
                    self.transformed(
                        self.gripper_meshes[name]["static"],
                        transforms["wrist"] @ wrist_to_gripper,
                    ),
                )
            )
            meshes.append(
                (name, self.transformed(self.gripper_meshes[name]["moving"], transforms[name]))
            )
        return meshes

    @staticmethod
    def camera_pose():
        pose = np.eye(4, dtype=np.float64)
        pose[:, 1:3] *= -1.0
        return pose

    def build_scene(self, meshes, K, wireframe=False):
        scene = pyrender.Scene(
            bg_color=(0.0, 0.0, 0.0, 0.0),
            ambient_light=(0.28, 0.28, 0.28),
        )
        for part_name, mesh in meshes:
            if wireframe:
                material = pyrender.MetallicRoughnessMaterial(
                    baseColorFactor=(0.025, 0.03, 0.035, 1.0),
                    emissiveFactor=(0.025, 0.03, 0.035),
                    metallicFactor=0.0,
                    roughnessFactor=1.0,
                    doubleSided=False,
                    wireframe=True,
                )
            else:
                rgba = PART_COLORS[part_name]
                color = tuple(float(value) / 255.0 for value in rgba)
                material = pyrender.MetallicRoughnessMaterial(
                    baseColorFactor=color,
                    metallicFactor=0.05,
                    roughnessFactor=0.62,
                    doubleSided=False,
                    smooth=False,
                )
            scene.add(
                pyrender.Mesh.from_trimesh(
                    mesh,
                    material=material,
                    wireframe=bool(wireframe),
                    smooth=False,
                )
            )

        K = np.asarray(K, dtype=np.float64)
        camera_pose = self.camera_pose()
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
        if not wireframe:
            side_light_pose = camera_pose.copy()
            side_light_pose[0, 3] += 0.10
            side_light_pose[2, 3] += 0.10
            scene.add(
                pyrender.PointLight(color=np.ones(3), intensity=2.2),
                pose=side_light_pose,
            )
        return scene

    def render(self, pose, K, image_shape):
        height, width = int(image_shape[0]), int(image_shape[1])
        meshes = self.world_meshes(pose)
        renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
        try:
            solid, depth = renderer.render(
                self.build_scene(meshes, K, wireframe=False),
                flags=pyrender.RenderFlags.RGBA,
            )
        finally:
            renderer.delete()
        support = np.asarray(depth) > 0.0
        color = np.asarray(solid[..., :3], dtype=np.uint8).copy()
        return {"rgb": color, "depth": np.asarray(depth), "valid": support}


def mesh_overlay(rgb, rendered, alpha=0.95):
    out = np.asarray(rgb, dtype=np.uint8).copy()
    support = rendered["valid"]
    out[support] = np.clip(
        (1.0 - float(alpha)) * out[support].astype(np.float32)
        + float(alpha) * rendered["rgb"][support].astype(np.float32),
        0,
        255,
    ).astype(np.uint8)
    return out


def silhouette_edges(mask, thickness=2):
    edge = cv2.morphologyEx(
        np.asarray(mask, dtype=np.uint8),
        cv2.MORPH_GRADIENT,
        np.ones((3, 3), np.uint8),
    )
    if thickness > 1:
        edge = cv2.dilate(edge, np.ones((thickness, thickness), np.uint8))
    return edge.astype(bool)


def direct_difference_overlay(rgb, original, refined):
    out = np.asarray(rgb, dtype=np.uint8).copy()
    original_only = original["valid"] & ~refined["valid"]
    refined_only = refined["valid"] & ~original["valid"]
    out[original_only] = np.clip(
        0.55 * out[original_only].astype(np.float32) + 0.45 * np.array([255, 205, 30]),
        0,
        255,
    ).astype(np.uint8)
    out[refined_only] = np.clip(
        0.55 * out[refined_only].astype(np.float32) + 0.45 * np.array([20, 225, 245]),
        0,
        255,
    ).astype(np.uint8)
    original_edge = silhouette_edges(original["valid"])
    refined_edge = silhouette_edges(refined["valid"])
    both = original_edge & refined_edge
    out[original_edge & ~both] = np.array([255, 205, 30], dtype=np.uint8)
    out[refined_edge & ~both] = np.array([20, 225, 245], dtype=np.uint8)
    out[both] = np.array([250, 250, 250], dtype=np.uint8)
    return out


def make_frame_panel(frame_id, rgb, original_pose, refined_pose, K, renderer, panel_size):
    original = renderer.render(original_pose, K, rgb.shape[:2])
    refined = renderer.render(refined_pose, K, rgb.shape[:2])
    roi = base.square_roi(original["valid"] | refined["valid"], padding=24)
    delta_r = base.rotation_error_deg(original_pose, refined_pose)
    delta_t = float(np.linalg.norm(refined_pose["trans"] - original_pose["trans"]) * 1000.0)
    delta_alpha = float(np.degrees(refined_pose["alpha"] - original_pose["alpha"]))

    panels = [
        base.label_panel(
            base.crop_resize(rgb, roi, panel_size),
            "Original RGB",
            f"LND TRAIN frame {frame_id} | same zoom and K",
        ),
        base.label_panel(
            base.crop_resize(mesh_overlay(rgb, original), roi, panel_size),
            "Before refine: rendered Trimesh",
            "actual triangle faces + part materials + lighting",
        ),
        base.label_panel(
            base.crop_resize(mesh_overlay(rgb, refined), roi, panel_size),
            "After refine: rendered Trimesh",
            f"dT={delta_t:.2f}mm | dR={delta_r:.2f}deg | dAlpha={delta_alpha:.2f}deg",
        ),
        base.label_panel(
            base.crop_resize(direct_difference_overlay(rgb, original, refined), roi, panel_size),
            "Direct before/after silhouette difference",
            "yellow before | cyan after | white overlap",
        ),
    ]
    return base.join_row(panels), {
        "frame_id": int(frame_id),
        "trans_delta_mm": delta_t,
        "rot_delta_deg": delta_r,
        "alpha_delta_deg": delta_alpha,
        "roi_xyxy": [int(v) for v in roi],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lnd_root", type=Path, default=base.DEFAULT_LND_ROOT)
    parser.add_argument("--original_memory", type=Path, default=base.DEFAULT_ORIGINAL_MEMORY)
    parser.add_argument("--refined_memory", type=Path, default=base.DEFAULT_REFINED_MEMORY)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frame_ids", type=int, nargs="+", default=DEFAULT_FRAME_IDS)
    parser.add_argument("--panel_size", type=int, default=430)
    args = parser.parse_args()

    split_root = args.lnd_root / "TRAIN"
    K = base.load_intrinsics(split_root)
    original_memory = base.load_memory(args.original_memory)
    refined_memory = base.load_memory(args.refined_memory)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    renderer = PartTrimeshRenderer()
    paths = []
    rows = []
    for frame_id in args.frame_ids:
        key = str(int(frame_id))
        if key not in original_memory or key not in refined_memory:
            raise KeyError(f"Frame {frame_id} missing from one of the memory pools")
        rgb, _, _ = base.load_case(split_root, int(frame_id))
        original_pose = base.pose_from_record(original_memory[key])
        refined_pose = base.pose_from_record(refined_memory[key])
        panel, row = make_frame_panel(
            int(frame_id), rgb, original_pose, refined_pose, K, renderer, int(args.panel_size)
        )
        path = args.output_dir / f"frame_{int(frame_id):04d}_before_after_trimesh.jpg"
        Image.fromarray(panel).save(path, quality=97)
        row["visualization"] = str(path.resolve())
        paths.append(path)
        rows.append(row)
        print(
            f"frame={frame_id} dT={row['trans_delta_mm']:.3f}mm "
            f"dR={row['rot_delta_deg']:.3f}deg dAlpha={row['alpha_delta_deg']:.3f}deg",
            flush=True,
        )

    if paths:
        base.make_contact_sheet(paths, args.output_dir / "contact_sheet_trimesh.jpg", max_width=2200)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    print(f"Saved {len(paths)} Trimesh comparisons to {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
