import argparse
import importlib.util
import json
import os
import sys
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

from instrument_trimesh_renderer import GMSInstrumentTrimeshRenderer  # noqa: E402
from predict_instrument_pose import (  # noqa: E402
    add_title,
    concat_panels,
    overlay_part_mask,
    pad_mask_to_square,
    pad_rgb_to_square,
    square_image_geometry,
)
from predict_surgripe_lnd import (  # noqa: E402
    crop_map_to_orig,
    decode_hcce_seg_full,
    draw_crop_box,
    overlay_binary,
    recrop_with_gt_mesh,
)


def _load_local_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


surgripe_lnd = _load_local_module(
    "robopepp_surgripe_lnd_dataset_seg",
    ROBOPEPP_ROOT / "datasets" / "surgripe_lnd.py",
)
hcce_model_module = _load_local_module(
    "robopepp_hcce_crop_model_seg",
    ROBOPEPP_ROOT / "models" / "hcce_crop_model.py",
)


def configure_device(name):
    if str(name).startswith("cuda"):
        idx = int(str(name).split(":", 1)[1]) if ":" in str(name) else 0
        os.environ["EGL_DEVICE_ID"] = str(idx)
        torch.cuda.set_device(idx)
    return torch.device(name if torch.cuda.is_available() else "cpu")


def load_hcce_model_any_keypoints(checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_args = ckpt.get("args") or {}
    if hasattr(ckpt_args, "__dict__"):
        ckpt_args = vars(ckpt_args)
    state = ckpt["model_state_dict"]
    num_keypoints = ckpt_args.get("num_keypoints")
    if num_keypoints is None:
        weight = state.get("keypoint_net.out_layer1.weight")
        num_keypoints = int(weight.shape[0]) if weight is not None else 5
    model = hcce_model_module.CropHCCEDenseKeypointDPT(
        img_size=int(ckpt_args.get("img_size", 224)),
        backbone=ckpt_args.get("backbone", "dinov2_vits14"),
        pretrained_backbone=False,
        hcce_feat_dim=int(ckpt_args.get("hcce_feat_dim", 256)),
        hcce_bits=int(ckpt_args.get("hcce_bits", 8)),
        num_keypoints=int(num_keypoints),
        action_dim=3,
        pose_head_iter=int(ckpt_args.get("pose_head_iter", 4)),
        pose_head_dropout=float(ckpt_args.get("pose_head_dropout", 0.3)),
    )
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, dict(ckpt_args)


def prob_to_rgb(prob):
    prob = np.asarray(prob, dtype=np.float32)
    prob = np.clip(prob, 0.0, 1.0)
    heat = cv2.applyColorMap((prob * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
    return cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)


def make_seg_visual(target, renderer, hcce_out, inst_thresh, panel_size):
    rgb = target["orig_rgb"]
    scale, pad_x, pad_y = square_image_geometry(rgb, int(panel_size))
    inst_prob_full, inst_full, part_full = decode_hcce_seg_full(hcce_out, target, float(inst_thresh))
    inst_prob_crop = torch.sigmoid(hcce_out["inst_mask_logits"][0].detach().float().cpu()).numpy()
    part_crop = torch.argmax(hcce_out["part_mask_logits"][0].detach().float().cpu(), dim=0).numpy().astype(np.uint8)
    part_crop_vis = np.zeros_like(part_crop, dtype=np.uint8)
    part_crop_vis[part_crop == 0] = 2
    part_crop_vis[part_crop == 1] = 1
    part_crop_vis[part_crop == 2] = 3
    part_crop_vis[inst_prob_crop < float(inst_thresh)] = 0

    gt_mesh = renderer.render_pose_overlay(
        rgb,
        target["gt_pose"],
        target["K_orig"],
        alpha=0.85,
    )
    crop_rgb_big = cv2.resize(target["crop_rgb"], (int(panel_size), int(panel_size)), interpolation=cv2.INTER_LINEAR)
    crop_inst_big = cv2.resize(prob_to_rgb(inst_prob_crop), (int(panel_size), int(panel_size)), interpolation=cv2.INTER_NEAREST)
    crop_part_big = cv2.resize(
        overlay_part_mask(target["crop_rgb"], part_crop_vis),
        (int(panel_size), int(panel_size)),
        interpolation=cv2.INTER_NEAREST,
    )

    panels = [
        add_title(pad_rgb_to_square(draw_crop_box(rgb, target), int(panel_size), scale, pad_x, pad_y), "rgb + GT-mesh bbox"),
        add_title(
            pad_rgb_to_square(overlay_binary(rgb, target["wrist_mask_orig"], color=(0, 255, 0)), int(panel_size), scale, pad_x, pad_y),
            "GT wrist mask",
        ),
        add_title(pad_rgb_to_square(gt_mesh, int(panel_size), scale, pad_x, pad_y), "GT repo mesh"),
        add_title(
            pad_rgb_to_square(overlay_binary(rgb, inst_full, color=(0, 220, 255)), int(panel_size), scale, pad_x, pad_y),
            f"inst seg > {float(inst_thresh):.2f}",
        ),
        add_title(
            pad_rgb_to_square(overlay_part_mask(rgb, part_full), int(panel_size), scale, pad_x, pad_y),
            "part seg",
        ),
        add_title(crop_rgb_big, "crop rgb"),
        add_title(crop_inst_big, "crop inst prob"),
        add_title(crop_part_big, "crop part seg"),
    ]
    metrics = {
        "frame_id": int(target["frame_id"]),
        "inst_area_full": int(inst_full.sum()),
        "wrist_area_gt": int(target["wrist_mask_orig"].sum()),
        "crop_inst_prob_mean": float(inst_prob_crop.mean()),
        "crop_inst_prob_max": float(inst_prob_crop.max()),
        "bbox_min": [float(v) for v in target["bbox_min"]],
        "bbox_max": [float(v) for v in target["bbox_max"]],
    }
    return concat_panels(panels), metrics


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lnd_root", type=str, default="/mnt/iMVR/daiyun/Dataset/LND")
    parser.add_argument("--split", type=str, default="TEST")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(
            MULTIHMR_ROOT
            / "logs/instrument_hcce_crop_surgpose_keypoint_hybrid_gpu4567_bs56_bf16_fromscratch/checkpoints/last.pt"
        ),
    )
    parser.add_argument("--output_dir", type=str, default=str(ROBOPEPP_ROOT / "logs/surgripe_lnd_seg_gpu4567_hybrid"))
    parser.add_argument("--device", type=str, default="cuda:4")
    parser.add_argument("--frame_ids", nargs="*", type=int, default=[1, 10, 50, 100, 150, 200, 260, 320, 373])
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--crop_source", choices=["gt_mesh", "wrist_mask"], default="gt_mesh")
    parser.add_argument("--mesh_bbox_scale", type=float, default=1.0)
    parser.add_argument("--mesh_min_crop_size", type=int, default=80)
    parser.add_argument("--mesh_bbox_margin_px", type=int, default=2)
    parser.add_argument("--bbox_scale", type=float, default=1.8)
    parser.add_argument("--min_crop_size", type=int, default=120)
    parser.add_argument("--bbox_margin_px", type=int, default=16)
    parser.add_argument("--inst_thresh", type=float, default=0.5)
    parser.add_argument("--panel_size", type=int, default=540)
    return parser


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = configure_device(args.device)
    dataset = surgripe_lnd.SurgripeLNDDataset(
        root=args.lnd_root,
        split=args.split,
        crop_size=args.crop_size,
        bbox_scale=args.bbox_scale,
        min_crop_size=args.min_crop_size,
        margin_px=args.bbox_margin_px,
        frame_ids=args.frame_ids,
    )
    renderer = GMSInstrumentTrimeshRenderer(device)
    model, meta = load_hcce_model_any_keypoints(args.checkpoint, device)
    rows = []
    with torch.no_grad():
        for i in range(len(dataset)):
            image_tensor, target = dataset[i]
            if args.crop_source == "gt_mesh":
                image_tensor, target = recrop_with_gt_mesh(renderer, target, dataset.to_tensor, args)
            x = image_tensor.unsqueeze(0).to(device, non_blocking=True)
            K = torch.from_numpy(target["K_crop"]).unsqueeze(0).to(device)
            out = model(x, K)
            canvas, row = make_seg_visual(target, renderer, out, args.inst_thresh, args.panel_size)
            row["checkpoint"] = str(args.checkpoint)
            row["crop_source"] = str(target.get("crop_source", args.crop_source))
            out_path = output_dir / "vis" / f"frame_{int(target['frame_id']):06d}_seg.jpg"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(canvas).save(out_path)
            row["vis_path"] = str(out_path)
            rows.append(row)
            print(f"[{i + 1}/{len(dataset)}] frame={target['frame_id']} -> {out_path}", flush=True)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "ckpt_name": meta.get("name", ""),
                "num_keypoints": meta.get("num_keypoints", None),
                "rows": rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"[done] {output_dir}", flush=True)


if __name__ == "__main__":
    main(build_parser().parse_args())
