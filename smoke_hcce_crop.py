import importlib.util
import json
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

ROBOPEPP_ROOT = Path(__file__).resolve().parent
MULTIHMR_ROOT = ROBOPEPP_ROOT.parents[1]
if str(ROBOPEPP_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBOPEPP_ROOT))
if str(MULTIHMR_ROOT) not in sys.path:
    sys.path.insert(0, str(MULTIHMR_ROOT))


def _load_local_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_dataset_module = _load_local_module("robopepp_hcce_crop_dataset_smoke", ROBOPEPP_ROOT / "datasets" / "rarp_hcce_crop.py")
_loss_module = _load_local_module("robopepp_hcce_crop_loss_smoke", ROBOPEPP_ROOT / "loss_hcce_crop.py")
_model_module = _load_local_module("robopepp_hcce_crop_model_smoke", ROBOPEPP_ROOT / "models" / "hcce_crop_model.py")
RARPCropHCCEDataset = _dataset_module.RARPCropHCCEDataset
collate_fn_rarp_crop_hcce = _dataset_module.collate_fn_rarp_crop_hcce
save_crop_debug_panel = _dataset_module.save_crop_debug_panel
compute_crop_hcce_losses = _loss_module.compute_crop_hcce_losses
CropHCCEDenseKeypointDPT = _model_module.CropHCCEDenseKeypointDPT


PUNCTURE_DATASET_ROOT = "/mnt/nas/share/shuojue/data/needlePuncture_videos"
PUNCTURE_POSE_ROOT = "/mnt/nas/share/shuojue/data/needlePuncture_results"


def _project(points, K):
    z = np.clip(points[:, 2:3], 1e-8, None)
    uvw = points @ K.T
    return uvw[:, :2] / z


def _draw_orig_projection(path, target):
    rgb = target["orig_rgb"].numpy().astype(np.uint8).copy()
    part = target["part_mask_orig"].numpy().astype(np.uint8)
    colors = np.array([[0, 0, 0], [230, 80, 80], [80, 220, 120], [80, 150, 240]], dtype=np.uint8)
    part_rgb = colors[np.clip(part, 0, 3)]
    inst = target["inst_mask_orig"].numpy().astype(bool)
    rgb[inst] = (0.6 * rgb[inst] + 0.4 * part_rgb[inst]).astype(np.uint8)
    kp = target["keypoints_orig"].numpy()
    valid = target["keypoints_valid_orig"].numpy().astype(bool)
    for i, (u, v) in enumerate(kp):
        color = (255, 255, 0) if valid[i] else (255, 0, 255)
        cv2.circle(rgb, (int(round(u)), int(round(v))), 5, color, -1)
    Image.fromarray(rgb).save(path)


def _projection_closed_loop_error(target):
    kp3 = target["keypoints_3d_cam"].numpy()
    K_crop = target["K"].numpy()
    uv_from_k = _project(kp3, K_crop)
    uv_target = target["keypoints_crop"].numpy()
    diff_k = np.linalg.norm(uv_from_k - uv_target, axis=1)

    uv_orig = target["keypoints_orig"].numpy().copy()
    bbox_min = target["bbox_min"].numpy()
    scale = target["scale"].numpy()
    pad = target["pad"].numpy()
    uv_xform = uv_orig.copy()
    uv_xform[:, 0] = (uv_xform[:, 0] - bbox_min[0]) * scale[0] + pad[0]
    uv_xform[:, 1] = (uv_xform[:, 1] - bbox_min[1]) * scale[1] + pad[1]
    diff_xform = np.linalg.norm(uv_xform - uv_target, axis=1)
    return {
        "max_project_with_K_crop_px": float(diff_k.max()),
        "mean_project_with_K_crop_px": float(diff_k.mean()),
        "max_orig_to_crop_transform_px": float(diff_xform.max()),
        "mean_orig_to_crop_transform_px": float(diff_xform.mean()),
    }


def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds_eval = RARPCropHCCEDataset(
        args.dataset_root,
        args.pose_root,
        split="train",
        training=False,
        crop_size=224,
        train_ratio=args.train_ratio,
        subsample=args.subsample,
        canonicalize_pose_symmetry=True,
        cse_coord_root=args.cse_coord_root,
        render_on_the_fly=bool(args.render_on_the_fly),
        coord_render_backend=args.coord_render_backend,
        cache_dir=str(out_dir / "dataset_cache"),
    )
    ds_aug = RARPCropHCCEDataset(
        args.dataset_root,
        args.pose_root,
        split="train",
        training=True,
        crop_size=224,
        train_ratio=args.train_ratio,
        subsample=args.subsample,
        canonicalize_pose_symmetry=True,
        cse_coord_root=args.cse_coord_root,
        render_on_the_fly=bool(args.render_on_the_fly),
        coord_render_backend=args.coord_render_backend,
        cache_dir=str(out_dir / "dataset_cache"),
    )
    ds_aug.set_epoch(80)

    records = []
    batch = []
    for i in range(args.num_samples):
        _, target = ds_eval[i]
        save_crop_debug_panel(out_dir / f"eval_{i:02d}_crop_concat.jpg", target)
        _draw_orig_projection(out_dir / f"eval_{i:02d}_orig_projection.jpg", target)
        records.append(
            {
                "sample": i,
                "video_name": target["video_name"],
                "frame_id": target["frame_id"],
                "instance_id": int(target["instance_id"].item()),
                "bbox_min": [float(v) for v in target["bbox_min"].tolist()],
                "bbox_max": [float(v) for v in target["bbox_max"].tolist()],
                "scale": [float(v) for v in target["scale"].tolist()],
                "pad": [float(v) for v in target["pad"].tolist()],
                "K_crop": target["K"].numpy().round(6).tolist(),
                "pose_sym_flipped": bool(target["pose_sym_flipped"].item()),
                "part_labels_crop": [int(v) for v in sorted(torch.unique(target["part_mask"]).tolist())],
                "has_cse": bool(target["has_cse"].item()),
                "projection_check": _projection_closed_loop_error(target),
            }
        )
        batch.append((ds_eval[i][0], target))

    for i in range(min(args.num_samples, 3)):
        _, target = ds_aug[i]
        save_crop_debug_panel(out_dir / f"aug_epoch80_{i:02d}_crop_concat.jpg", target)
        _draw_orig_projection(out_dir / f"aug_epoch80_{i:02d}_orig_projection.jpg", target)

    x, y = collate_fn_rarp_crop_hcce(batch[: max(1, min(2, len(batch)))])
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = CropHCCEDenseKeypointDPT(
        img_size=224,
        backbone=args.backbone,
        pretrained_backbone=bool(args.pretrained_backbone),
        hcce_bits=args.hcce_bits,
    ).to(device)
    model.train()
    x = x.to(device)
    y = {k: v.to(device) if torch.is_tensor(v) else v for k, v in y.items()}
    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available() and not args.cpu):
        out = model(x, y["K"])
        loss, metrics = compute_crop_hcce_losses(out, y, args)

    summary = {
        "dataset": repr(ds_eval),
        "augmentation": {
            "crop_size": 224,
            "bbox_jitter_epoch80_px": 80.0,
            "color_jitter": True,
            "occlusion_augmentation": True,
            "rgb_augmentation": True,
            "coord_render_backend": args.coord_render_backend,
        },
        "part_label_alignment": {
            "rarp": "0=background, 1=gripper, 2=wrist, 3=shaft",
            "surgical_instruments_rarp50": "decoded to 1=gripper, 2=wrist, 3=shaft",
            "vos_endovis": "_PART_REMAP {1:shaft, 2:gripper, 3:wrist, 4:gripper} -> 1=gripper,2=wrist,3=shaft",
            "surgpose": "raw 3=gripper,2=wrist,1=shaft -> 1=gripper,2=wrist,3=shaft",
            "dense_logits": "[wrist, gripper, shaft]",
        },
        "model_outputs": {k: list(v.shape) for k, v in out.items() if torch.is_tensor(v)},
        "loss": float(loss.detach().cpu().item()),
        "metrics": {k: float(v.detach().cpu().item()) for k, v in metrics.items()},
        "samples": records,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default=PUNCTURE_DATASET_ROOT)
    parser.add_argument("--pose_root", type=str, default=PUNCTURE_POSE_ROOT)
    parser.add_argument("--out_dir", type=str, default=str(ROBOPEPP_ROOT / "logs" / "hcce_crop224_smoke"))
    parser.add_argument("--subsample", type=int, default=50)
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--train_ratio", type=float, default=0.95)
    parser.add_argument("--cse_coord_root", type=str, default=None)
    parser.add_argument("--render_on_the_fly", type=int, default=1, choices=[0, 1])
    parser.add_argument("--coord_render_backend", type=str, default="trimesh", choices=["trimesh", "gaussian"])
    parser.add_argument("--backbone", type=str, default="dinov2_vits14")
    parser.add_argument("--pretrained_backbone", type=int, default=0, choices=[0, 1])
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--hcce_bits", type=int, default=8)
    parser.add_argument("--hcce_coord_min", type=float, default=-1.0)
    parser.add_argument("--hcce_coord_max", type=float, default=1.0)
    parser.add_argument("--alpha_dice", type=float, default=5.0)
    parser.add_argument("--alpha_bce_mask", type=float, default=2.0)
    parser.add_argument("--alpha_part", type=float, default=2.0)
    parser.add_argument("--alpha_hcce", type=float, default=1.0)
    parser.add_argument("--alpha_heatmap", type=float, default=1.0)
    parser.add_argument("--alpha_action_l1", type=float, default=1.0)
    parser.add_argument("--alpha_wrist_quat_l1", type=float, default=1.0)
    parser.add_argument("--alpha_wrist_trans_l1", type=float, default=10.0)
    parser.add_argument("--alpha_keypoint_2d", type=float, default=1.0)
    parser.add_argument("--alpha_keypoint_3d", type=float, default=0.0)
    main(parser.parse_args())
