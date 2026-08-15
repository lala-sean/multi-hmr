import json
import os
from argparse import ArgumentParser
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image

from train_instrument_hcce_crop_surgpose_keypoint_dpt import (
    GRASPING_DATASET_ROOT,
    GRASPING_POSE_ROOT,
    KNOTTING_DATASET_ROOT,
    KNOTTING_POSE_ROOT,
    PUNCTURE_DATASET_ROOT,
    PUNCTURE_POSE_ROOT,
    SURGPOSE_ROOT,
    VOS_ENDOVIS17_IMAGE_ROOT,
    VOS_ENDOVIS17_ROOT,
    VOS_ENDOVIS18_IMAGE_ROOT,
    VOS_ENDOVIS18_ROOT,
    SurgPoseCropDataset,
    VOSEndoVisCropDataset,
    make_rarp_crop_dataset,
)
from submodules.RoboPEPP.instrument_geometry import (
    KEYPOINT_NAMES,
    instrument_keypoints_camera_np,
    project_points_np,
)


PART_COLORS = np.array(
    [
        [0, 0, 0],
        [230, 80, 80],
        [80, 220, 120],
        [80, 150, 240],
    ],
    dtype=np.uint8,
)


def _to_np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def _overlay_part(rgb, inst_mask, part_mask):
    part_rgb = PART_COLORS[np.clip(part_mask.astype(np.int64), 0, 3)]
    out = rgb.copy()
    m = inst_mask > 0.5
    out[m] = (0.55 * out[m] + 0.45 * part_rgb[m]).astype(np.uint8)
    return out, part_rgb


def _draw_points(img, points, valid, color, labels=None):
    out = img.copy()
    for i, (uv, ok) in enumerate(zip(points, valid)):
        if not bool(ok) or not np.isfinite(uv).all():
            continue
        x, y = int(round(float(uv[0]))), int(round(float(uv[1])))
        if 0 <= x < out.shape[1] and 0 <= y < out.shape[0]:
            cv2.circle(out, (x, y), 4, color, -1, lineType=cv2.LINE_AA)
            if labels is not None:
                cv2.putText(
                    out,
                    str(labels[i]),
                    (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    color,
                    1,
                    cv2.LINE_AA,
                )
    return out


def _coord_panel(coord_img):
    if coord_img is None:
        return np.zeros((224, 224, 3), dtype=np.uint8)
    coord = _to_np(coord_img)
    if coord.shape[-1] < 4 or not (coord[..., 3] > 0).any():
        return np.zeros((coord.shape[0], coord.shape[1], 3), dtype=np.uint8)
    out = ((coord[..., :3] + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)
    out[coord[..., 3] <= 0] = 0
    return out


def save_rarp_panel(dataset, idx, out_dir):
    image, target = dataset[idx]
    rgb = _to_np(target["crop_rgb"]).astype(np.uint8)
    inst = _to_np(target["inst_mask"]).astype(np.float32)
    part = _to_np(target["part_mask"]).astype(np.uint8)
    overlay, part_rgb = _overlay_part(rgb, inst, part)

    action = _to_np(target["action"]).astype(np.float32)
    quat = _to_np(target["wrist_quat"]).astype(np.float32)
    trans = _to_np(target["wrist_trans"]).astype(np.float32)
    K_crop = _to_np(target["K"]).astype(np.float32)
    kp_cam = instrument_keypoints_camera_np(quat, trans, action)
    kp_proj_crop = project_points_np(kp_cam, K_crop).astype(np.float32)
    kp_dataset = _to_np(target["keypoints_crop"]).astype(np.float32)
    valid = _to_np(target["keypoints_valid"]).astype(bool)
    err = np.linalg.norm(kp_proj_crop[valid] - kp_dataset[valid], axis=-1) if valid.any() else np.zeros((0,))

    panel = overlay
    panel = _draw_points(panel, kp_dataset, valid, (20, 255, 20), labels=list(range(len(KEYPOINT_NAMES))))
    panel = _draw_points(panel, kp_proj_crop, valid, (255, 30, 30), labels=None)
    coord_rgb = _coord_panel(target.get("coord_img"))
    concat = np.concatenate([rgb, panel, part_rgb, coord_rgb], axis=1)

    out_path = out_dir / f"rarp_{idx:04d}_{target['video_name']}_{target['frame_id']}_inst{int(target['instance_id'])}.png"
    Image.fromarray(concat).save(out_path)
    return {
        "kind": "rarp",
        "idx": int(idx),
        "path": str(out_path),
        "video_name": target["video_name"],
        "frame_id": target["frame_id"],
        "instance_id": int(target["instance_id"]),
        "has_cse": bool(target["has_cse"]),
        "crop_keypoint_reproj_rmse_px": float(np.sqrt(np.mean(err ** 2))) if err.size else None,
        "crop_keypoint_reproj_max_px": float(err.max()) if err.size else None,
        "K_crop": K_crop.tolist(),
        "bbox_min": _to_np(target["bbox_min"]).astype(float).tolist(),
        "bbox_max": _to_np(target["bbox_max"]).astype(float).tolist(),
        "scale": _to_np(target["scale"]).astype(float).tolist(),
        "pad": _to_np(target["pad"]).astype(float).tolist(),
    }


def save_surgpose_panel(dataset, idx, out_dir):
    image, target = dataset[idx]
    rgb = _to_np(target["crop_rgb"]).astype(np.uint8)
    inst = _to_np(target["inst_mask"]).astype(np.float32)
    part = _to_np(target["part_mask"]).astype(np.uint8)
    overlay, part_rgb = _overlay_part(rgb, inst, part)
    kp = _to_np(target["keypoints_crop"]).astype(np.float32)
    valid = _to_np(target["keypoints_valid"]).astype(bool)
    overlay = _draw_points(overlay, kp, valid, (255, 255, 30), labels=list(range(kp.shape[0])))
    heat = _to_np(target["heatmaps"]).max(axis=0)
    heat_rgb = cv2.applyColorMap((heat * 255.0).clip(0, 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
    heat_rgb = cv2.cvtColor(heat_rgb, cv2.COLOR_BGR2RGB)
    concat = np.concatenate([rgb, overlay, part_rgb, heat_rgb], axis=1)
    out_path = out_dir / f"surgpose_{idx:04d}_{target['video_name']}_{target['frame_id']}_inst{int(target['instance_id'])}.png"
    Image.fromarray(concat).save(out_path)
    return {
        "kind": "surgpose",
        "idx": int(idx),
        "path": str(out_path),
        "episode": target["video_name"],
        "frame_id": target["frame_id"],
        "instance_id": int(target["instance_id"]),
        "n_valid_keypoints": int(valid.sum()),
        "bbox_min": _to_np(target["bbox_min"]).astype(float).tolist(),
        "bbox_max": _to_np(target["bbox_max"]).astype(float).tolist(),
        "scale": _to_np(target["scale"]).astype(float).tolist(),
        "pad": _to_np(target["pad"]).astype(float).tolist(),
    }


def save_vos_panel(dataset, idx, out_dir):
    image, target = dataset[idx]
    rgb = _to_np(target["crop_rgb"]).astype(np.uint8)
    inst = _to_np(target["inst_mask"]).astype(np.float32)
    part = _to_np(target["part_mask"]).astype(np.uint8)
    overlay, part_rgb = _overlay_part(rgb, inst, part)
    concat = np.concatenate([rgb, overlay, part_rgb], axis=1)
    out_path = out_dir / f"{target['dataset_name']}_{idx:04d}_{target['video_name']}_{target['frame_id']}_inst{int(target['instance_id'])}.png"
    Image.fromarray(concat).save(out_path)
    return {
        "kind": target["dataset_name"],
        "idx": int(idx),
        "path": str(out_path),
        "seq": target["video_name"],
        "frame_id": target["frame_id"],
        "instance_id": int(target["instance_id"]),
        "part_labels": sorted(np.unique(part).astype(int).tolist()),
        "bbox_min": _to_np(target["bbox_min"]).astype(float).tolist(),
        "bbox_max": _to_np(target["bbox_max"]).astype(float).tolist(),
        "scale": _to_np(target["scale"]).astype(float).tolist(),
        "pad": _to_np(target["pad"]).astype(float).tolist(),
    }


def make_args(cli):
    return SimpleNamespace(
        img_size=224,
        needle_train_ratio=0.95,
        min_dice_shaft=0.8,
        min_dice_wrist=0.6,
        min_dice_gripper=0.6,
        canonicalize_pose_symmetry=1,
        canonical_eps=0.08,
        heatmap_sigma=2.0,
        color_jitter=0,
        rgb_augmentation=0,
        occlusion_augmentation=0,
        occlusion_prob=0.5,
        dataset_cache_dir=cli.cache_dir,
        cse_coord_root=cli.cse_coord_root,
        render_on_the_fly=cli.render_hcce,
        coord_render_backend=cli.coord_render_backend,
    )


def main():
    parser = ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="debug_outputs/hcce_crop_surgpose_dataset_vis")
    parser.add_argument("--cache_dir", type=str, default="logs/instrument_hcce_crop_surgpose_keypoint_hybrid_4567_bs56/dataset_cache")
    parser.add_argument("--render_hcce", type=int, default=0, choices=[0, 1])
    parser.add_argument("--cse_coord_root", type=str, default=None)
    parser.add_argument("--coord_render_backend", type=str, default="trimesh", choices=["trimesh", "gaussian"])
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--rarp_dataset", type=str, default="needlePuncture", choices=["needlePuncture", "needleGrasping", "knotting"])
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    common = make_args(args)
    roots = {
        "needlePuncture": (PUNCTURE_DATASET_ROOT, PUNCTURE_POSE_ROOT),
        "needleGrasping": (GRASPING_DATASET_ROOT, GRASPING_POSE_ROOT),
        "knotting": (KNOTTING_DATASET_ROOT, KNOTTING_POSE_ROOT),
    }

    rarp = make_rarp_crop_dataset(
        common,
        args.rarp_dataset,
        "train",
        False,
        roots[args.rarp_dataset][0],
        roots[args.rarp_dataset][1],
        subsample=200,
    )
    if not args.render_hcce:
        rarp.wrapped.require_cse = False

    surgpose = SurgPoseCropDataset(
        root_dir=SURGPOSE_ROOT,
        split="train",
        training=False,
        crop_size=224,
        subsample=500,
        train_episodes="000000,000001",
        num_keypoints=7,
        color_jitter=False,
        rgb_augmentation=False,
        occlusion_augmentation=False,
        cache_dir=args.cache_dir,
    )
    vos17 = VOSEndoVisCropDataset(
        VOS_ENDOVIS17_ROOT,
        VOS_ENDOVIS17_IMAGE_ROOT,
        split="train",
        training=False,
        crop_size=224,
        subsample=500,
        color_jitter=False,
        rgb_augmentation=False,
        occlusion_augmentation=False,
        cache_dir=args.cache_dir,
    )
    vos18 = VOSEndoVisCropDataset(
        VOS_ENDOVIS18_ROOT,
        VOS_ENDOVIS18_IMAGE_ROOT,
        split="train",
        training=False,
        crop_size=224,
        subsample=500,
        color_jitter=False,
        rgb_augmentation=False,
        occlusion_augmentation=False,
        cache_dir=args.cache_dir,
    )

    summary = []
    for i in range(min(args.num_samples, len(rarp))):
        summary.append(save_rarp_panel(rarp, i, out_dir))
    for i in range(min(args.num_samples, len(surgpose))):
        summary.append(save_surgpose_panel(surgpose, i, out_dir))
    for dataset in (vos17, vos18):
        for i in range(min(args.num_samples, len(dataset))):
            summary.append(save_vos_panel(dataset, i, out_dir))

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved {len(summary)} panels to {out_dir}")
    print(f"summary: {summary_path}")
    for item in summary:
        if item["kind"] == "rarp":
            print(
                f"RARP idx={item['idx']} reproj_rmse={item['crop_keypoint_reproj_rmse_px']} "
                f"max={item['crop_keypoint_reproj_max_px']} has_cse={item['has_cse']}"
            )


if __name__ == "__main__":
    main()
