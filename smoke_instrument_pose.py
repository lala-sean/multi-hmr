import argparse
import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

ROBOPEPP_ROOT = Path(__file__).resolve().parent
MULTIHMR_ROOT = ROBOPEPP_ROOT.parents[1]
if str(ROBOPEPP_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBOPEPP_ROOT))
if str(MULTIHMR_ROOT) not in sys.path:
    sys.path.insert(0, str(MULTIHMR_ROOT))

from instrument_geometry import (  # noqa: E402
    KEYPOINT_NAMES,
    pose_dict_to_params,
    project_points_np,
)


DATASETS = {
    "needlePuncture": {
        "dataset_root": "/mnt/nas/share/shuojue/data/needlePuncture_videos",
        "pose_root": "/mnt/nas/share/shuojue/data/needlePuncture_results",
    },
    "needleGrasping": {
        "dataset_root": "/mnt/nas/share/shuojue/data/needleGrasping_videos",
        "pose_root": "/mnt/nas/share/shuojue/data/needleGrasping_results",
    },
    "knotting": {
        "dataset_root": "/mnt/nas/share/shuojue/data/knotting_videos",
        "pose_root": "/mnt/nas/share/shuojue/data/knotting_results",
    },
}


def _load_local_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _draw_keypoints(rgb, points, valid, names, radius=4):
    out = rgb.copy()
    colors = {
        True: (30, 220, 60),
        False: (230, 50, 50),
    }
    for i, (xy, ok, name) in enumerate(zip(points, valid, names)):
        x, y = int(round(float(xy[0]))), int(round(float(xy[1])))
        color = colors[bool(ok)]
        cv2.circle(out, (x, y), radius, color, -1, lineType=cv2.LINE_AA)
        cv2.putText(
            out,
            f"{i}:{name}",
            (x + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA,
        )
    return out


def _add_title(panel, title):
    out = panel.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(out, title, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _heatmap_panel(heatmaps):
    panels = []
    for i, hmap in enumerate(heatmaps):
        hm = np.clip(hmap, 0.0, 1.0)
        img = (hm * 255).astype(np.uint8)
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        panels.append(_add_title(img, f"{i}:{KEYPOINT_NAMES[i]}"))
    top = np.concatenate(panels[:3], axis=1)
    bottom = np.concatenate(
        panels[3:] + [np.zeros_like(panels[0])],
        axis=1,
    )
    return np.concatenate([top, bottom], axis=0)


def _save(path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8)).save(path)


def _pose_params_from_target(target):
    pose = {
        "rot": target["wrist_quat"].numpy(),
        "trans": target["wrist_trans"].numpy(),
        "alpha": float(target["action"][0].item()),
        "theta_l": float(target["action"][1].item()),
        "theta_r": float(target["action"][2].item()),
    }
    return pose_dict_to_params(pose)


def main(args):
    rarp_module = _load_local_module("robopepp_rarp_instrument", ROBOPEPP_ROOT / "datasets" / "rarp_instrument.py")
    from estimate_rarp_cse_articulate_pose import CAD_ROOT, InstrumentCAD, draw_projected_mesh

    cfg = DATASETS[args.dataset_name]
    dataset = rarp_module.RoboPEPPRARPInstrument(
        cfg["dataset_root"],
        cfg["pose_root"],
        split=args.split,
        training=False,
        crop_size=args.crop_size,
        train_ratio=args.needle_train_ratio,
        subsample=args.subsample,
        min_dice=(args.min_dice_shaft, args.min_dice_wrist, args.min_dice_gripper),
        canonicalize_pose_symmetry=bool(args.canonicalize_pose_symmetry),
        canonical_eps=args.canonical_eps,
    )
    cad = InstrumentCAD(CAD_ROOT)
    render_args = SimpleNamespace(
        overlay_alpha=args.overlay_alpha,
        mesh_render_backend=args.mesh_render_backend,
        mesh_render_smooth=1,
        pyrender_znear=args.pyrender_znear,
        pyrender_zfar=args.pyrender_zfar,
        pyrender_light_intensity=args.pyrender_light_intensity,
        pyrender_ambient_light=args.pyrender_ambient_light,
        min_depth=args.min_depth,
        draw_margin=args.draw_margin,
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    count = min(args.num_samples, len(dataset))
    for local_idx in range(count):
        sample_idx = args.start_index + local_idx
        image, target = dataset[sample_idx]
        crop_rgb = target["crop_rgb"].numpy()
        orig_rgb = target["orig_rgb"].numpy()
        K_crop = target["K"].numpy()
        K_orig = target["K_orig"].numpy()
        pose_params = _pose_params_from_target(target)

        keypoints_crop = target["keypoints_crop"].numpy()
        keypoints_orig = target["keypoints_orig"].numpy()
        valid = target["keypoints_valid"].numpy().astype(bool)
        reproj_crop = project_points_np(target["keypoints_3d_cam"].numpy(), K_crop).astype(np.float32)
        crop_err = np.linalg.norm(reproj_crop - keypoints_crop, axis=1)
        max_crop_err = float(np.max(crop_err))
        if max_crop_err > args.max_reprojection_error:
            raise RuntimeError(
                f"crop reprojection check failed for sample {sample_idx}: max err {max_crop_err:.4f}px"
            )

        bbox_min = target["bbox_min"].numpy()
        bbox_max = target["bbox_max"].numpy()
        orig_panel = orig_rgb.copy()
        cv2.rectangle(
            orig_panel,
            tuple(np.round(bbox_min).astype(int)),
            tuple(np.round(bbox_max).astype(int)),
            (255, 255, 255),
            2,
        )
        orig_panel = _draw_keypoints(orig_panel, keypoints_orig, target["keypoints_valid_orig"].numpy().astype(bool), KEYPOINT_NAMES)
        crop_panel = _draw_keypoints(crop_rgb, keypoints_crop, valid, KEYPOINT_NAMES)
        reproj_panel = _draw_keypoints(crop_panel, reproj_crop, valid, [f"reproj_{n}" for n in KEYPOINT_NAMES], radius=2)
        heatmap_panel = _heatmap_panel(target["heatmaps"].numpy())

        stem = f"{args.dataset_name}_{args.split}_{sample_idx:05d}_inst{int(target['instance_id'].item())}"
        _save(out_dir / f"{stem}_orig_bbox_keypoints.jpg", _add_title(orig_panel, "original bbox + gt keypoints"))
        _save(out_dir / f"{stem}_crop_keypoints_reprojection.jpg", _add_title(reproj_panel, f"crop keypoints, max reproj err {max_crop_err:.4f}px"))
        _save(out_dir / f"{stem}_heatmaps.jpg", heatmap_panel)

        crop_mesh = draw_projected_mesh(crop_rgb, cad, pose_params, K_crop, render_args)
        orig_mesh = draw_projected_mesh(orig_rgb, cad, pose_params, K_orig, render_args)
        _save(out_dir / f"{stem}_crop_trimesh_projection.jpg", crop_mesh)
        _save(out_dir / f"{stem}_orig_trimesh_projection.jpg", orig_mesh)

        print(
            f"saved sample {sample_idx}: visible={valid.astype(int).tolist()} "
            f"max_crop_reproj_err={max_crop_err:.4f}px -> {out_dir / stem}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="needlePuncture", choices=sorted(DATASETS))
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--output_dir", type=str, default="logs/robopepp_instrument_smoke")
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--subsample", type=int, default=50)
    parser.add_argument("--needle_train_ratio", type=float, default=0.95)
    parser.add_argument("--canonicalize_pose_symmetry", type=int, default=1, choices=[0, 1])
    parser.add_argument("--canonical_eps", type=float, default=0.08)
    parser.add_argument("--min_dice_shaft", type=float, default=0.8)
    parser.add_argument("--min_dice_wrist", type=float, default=0.6)
    parser.add_argument("--min_dice_gripper", type=float, default=0.6)
    parser.add_argument("--max_reprojection_error", type=float, default=1e-3)
    parser.add_argument("--mesh_render_backend", type=str, default="pyrender", choices=["pyrender", "software"])
    parser.add_argument("--overlay_alpha", type=float, default=0.85)
    parser.add_argument("--pyrender_znear", type=float, default=0.001)
    parser.add_argument("--pyrender_zfar", type=float, default=0.6)
    parser.add_argument("--pyrender_light_intensity", type=float, default=1.0)
    parser.add_argument("--pyrender_ambient_light", type=float, default=0.3)
    parser.add_argument("--min_depth", type=float, default=1e-4)
    parser.add_argument("--draw_margin", type=float, default=20.0)
    main(parser.parse_args())

