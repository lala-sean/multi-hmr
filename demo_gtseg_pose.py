import argparse
import csv
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
from PIL import Image
import torchvision.transforms as tv_transforms

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

ROBOPEPP_ROOT = Path(__file__).resolve().parent
MULTIHMR_ROOT = ROBOPEPP_ROOT.parents[1]
if str(ROBOPEPP_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBOPEPP_ROOT))
if str(MULTIHMR_ROOT) not in sys.path:
    sys.path.insert(0, str(MULTIHMR_ROOT))

from instrument_geometry import (  # noqa: E402
    crop_resize_pad_intrinsics,
    instrument_keypoints_camera_np,
    project_points_np,
    rarp_intrinsics,
)
from predict_instrument_pose import (  # noqa: E402
    add_title,
    concat_panels,
    crop_points_to_original,
    draw_crop_debug,
    draw_keypoints_overlay,
    heatmap_argmax,
    load_model,
    overlay_part_mask,
    pad_mask_to_square,
    pad_rgb_to_square,
    square_image_geometry,
)
from instrument_trimesh_renderer import GMSInstrumentTrimeshRenderer  # noqa: E402
from pose_pnp import pose_from_keypoints_pnp  # noqa: E402


def _cuda_index(device):
    if not str(device).startswith("cuda"):
        return None
    parts = str(device).split(":", 1)
    return int(parts[1]) if len(parts) == 2 and parts[1] else 0


def configure_device(device):
    cuda_idx = _cuda_index(device)
    if cuda_idx is not None:
        os.environ["EGL_DEVICE_ID"] = str(cuda_idx)
        torch.cuda.set_device(cuda_idx)
    return torch.device(device if torch.cuda.is_available() else "cpu")


def list_images(input_dir, max_images=-1):
    paths = sorted(
        p for p in Path(input_dir).iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    if int(max_images) > 0:
        paths = paths[: int(max_images)]
    if not paths:
        raise RuntimeError(f"No images found under {input_dir}")
    return paths


def load_camera_K(args, width, height):
    if args.camera_metadata:
        meta_path = Path(args.camera_metadata)
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if args.camera_key in meta:
                return np.asarray(meta[args.camera_key], dtype=np.float32)
    if args.focal > 0:
        return np.array(
            [[float(args.focal), 0.0, width / 2.0], [0.0, float(args.focal), height / 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
    return rarp_intrinsics(width, height)


def resolve_mask_path(mask_dir, image_path):
    if not mask_dir:
        return None
    root = Path(mask_dir)
    candidates = [
        root / f"{image_path.stem}.png",
        root / f"{image_path.stem}.jpg",
        root / image_path.name,
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def load_mask(mask_dir, image_path, shape_hw, required=False):
    path = resolve_mask_path(mask_dir, image_path)
    if path is None:
        if required:
            raise FileNotFoundError(f"GT mask not found for {image_path.name} under {mask_dir}")
        return None, None
    mask = np.asarray(Image.open(path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.shape[:2] != tuple(shape_hw):
        mask = cv2.resize(mask.astype(np.uint8), (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)
    return mask.astype(np.uint8), path


def remap_part_mask(part_mask, mode):
    if part_mask is None:
        return None
    part_mask = part_mask.astype(np.uint8)
    if mode == "surgpose":
        out = np.zeros_like(part_mask, dtype=np.uint8)
        out[part_mask == 1] = 3
        out[part_mask == 2] = 2
        out[part_mask == 3] = 1
        return out
    return part_mask


def resize_longer_side(rgb, crop_size):
    h, w = rgb.shape[:2]
    if w > h:
        new_w = int(crop_size)
        new_h = int(crop_size * h / w)
    else:
        new_h = int(crop_size)
        new_w = int(crop_size * w / h)
    return cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR), (new_w, new_h)


def pad_to_square(rgb, crop_size):
    h, w = rgb.shape[:2]
    pad_h = (crop_size - h) // 2
    pad_w = (crop_size - w) // 2
    padding = ((pad_h, crop_size - h - pad_h), (pad_w, crop_size - w - pad_w), (0, 0))
    return np.pad(rgb, padding, mode="edge"), (pad_w, pad_h)


def bbox_from_mask(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise RuntimeError("Empty GT instance mask")
    return (
        np.array([float(xs.min()), float(ys.min())], dtype=np.float32),
        np.array([float(xs.max() + 1), float(ys.max() + 1)], dtype=np.float32),
    )


def crop_instance(rgb, K_orig, inst_mask, crop_size):
    h, w = rgb.shape[:2]
    bbox_min, bbox_max = bbox_from_mask(inst_mask)
    bbox_min = np.clip(bbox_min, [0.0, 0.0], [float(w - 1), float(h - 1)])
    bbox_max = np.clip(bbox_max, [1.0, 1.0], [float(w), float(h)])
    x0, y0 = bbox_min.astype(np.int64)
    x1, y1 = np.ceil(bbox_max).astype(np.int64)
    crop = rgb[y0:y1, x0:x1]
    crop_resized, (new_w, new_h) = resize_longer_side(crop, crop_size)
    scale_x = float(new_w) / float(bbox_max[0] - bbox_min[0])
    scale_y = float(new_h) / float(bbox_max[1] - bbox_min[1])
    crop_square, (pad_w, pad_h) = pad_to_square(crop_resized, crop_size)
    K_crop = crop_resize_pad_intrinsics(K_orig, bbox_min=bbox_min, scale_xy=(scale_x, scale_y), pad_xy=(pad_w, pad_h))
    return {
        "crop_rgb": crop_square.astype(np.uint8),
        "K_crop": K_crop.astype(np.float32),
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
        "scale": np.array([scale_x, scale_y], dtype=np.float32),
        "pad": np.array([pad_w, pad_h], dtype=np.float32),
    }


def pose_from_output(out):
    action = out["action_pred"][0].detach().cpu().numpy().astype(np.float64)
    quat = out["wrist_quat_pred"][0].detach().cpu().numpy().astype(np.float64)
    trans = out["wrist_trans_pred"][0].detach().cpu().numpy().astype(np.float64)
    return {
        "rot": quat,
        "trans": trans,
        "alpha": float(action[0]),
        "theta_l": float(action[1]),
        "theta_r": float(action[2]),
    }


def crop_points_to_orig_np(points_crop, crop_info):
    target = {
        "scale": torch.from_numpy(crop_info["scale"]),
        "pad": torch.from_numpy(crop_info["pad"]),
        "bbox_min": torch.from_numpy(crop_info["bbox_min"]),
    }
    return crop_points_to_original(points_crop, target)


def draw_demo_keypoints(orig_rgb, instance_results):
    out = orig_rgb.copy()
    for item in instance_results:
        bbox_min = item["bbox_min"]
        bbox_max = item["bbox_max"]
        cv2.rectangle(out, tuple(np.round(bbox_min).astype(int)), tuple(np.round(bbox_max).astype(int)), (255, 255, 255), 2, cv2.LINE_AA)
        for xy in item["pred_hm_orig"]:
            x, y = np.round(xy).astype(int)
            cv2.drawMarker(out, (x, y), (255, 70, 220), cv2.MARKER_CROSS, 13, 2, cv2.LINE_AA)
        for xy in item["pred_pose_orig"]:
            x, y = np.round(xy).astype(int)
            cv2.circle(out, (x, y), 4, (60, 210, 255), 2, lineType=cv2.LINE_AA)
    return out


def make_crop_panel(instance_results, panel_size):
    panel = np.zeros((panel_size, panel_size, 3), dtype=np.uint8)
    if not instance_results:
        return add_title(panel, "crop-debug")
    cell = 224
    cols = max(1, panel_size // cell)
    for i, item in enumerate(instance_results[: max(1, cols * cols)]):
        crop_vis = draw_crop_debug(
            item["crop_rgb"],
            np.zeros_like(item["pred_hm_crop"], dtype=np.float32),
            np.zeros((item["pred_hm_crop"].shape[0],), dtype=bool),
            item["pred_hm_crop"],
            item["pred_pose_crop"],
        )
        row = i // cols
        col = i % cols
        y0 = row * cell
        x0 = col * cell
        if y0 + cell <= panel_size and x0 + cell <= panel_size:
            panel[y0:y0 + cell, x0:x0 + cell] = crop_vis[:cell, :cell]
    return add_title(panel, "crop-debug")


def save_demo_visual(path, orig_rgb, gt_part_mask, pred_pose_mask, pose_mesh_rgb, instance_results, panel_size):
    scale, pad_x, pad_y = square_image_geometry(orig_rgb, panel_size)
    rgb_sq = pad_rgb_to_square(orig_rgb, panel_size, scale, pad_x, pad_y)
    kpt_sq = pad_rgb_to_square(draw_demo_keypoints(orig_rgb, instance_results), panel_size, scale, pad_x, pad_y)
    gt_mask_sq = pad_mask_to_square(gt_part_mask, panel_size, scale, pad_x, pad_y)
    pred_mask_sq = pad_mask_to_square(pred_pose_mask, panel_size, scale, pad_x, pad_y)
    mesh_sq = pad_rgb_to_square(pose_mesh_rgb, panel_size, scale, pad_x, pad_y)
    panels = [
        add_title(rgb_sq, "rgb"),
        add_title(overlay_part_mask(rgb_sq, gt_mask_sq), "gt-seg"),
        make_crop_panel(instance_results, panel_size),
        add_title(kpt_sq, "robopepp-kpt"),
        add_title(overlay_part_mask(rgb_sq, pred_mask_sq), "posehead-proj-seg"),
        add_title(mesh_sq, "posehead-trimesh"),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(concat_panels(panels)).save(path)


def process_image(image_path, model, mesh_renderer, device, args, to_tensor):
    rgb = np.asarray(Image.open(image_path).convert("RGB"))
    h, w = rgb.shape[:2]
    K_orig = load_camera_K(args, w, h)
    inst_mask, inst_path = load_mask(args.inst_mask_dir, image_path, (h, w), required=True)
    raw_part, part_path = load_mask(args.part_mask_dir, image_path, (h, w), required=False)
    part_mask = remap_part_mask(raw_part, args.part_label_mode)
    labels = [int(x) for x in sorted(np.unique(inst_mask).tolist()) if int(x) > 0]
    if args.max_instances > 0:
        labels = labels[: args.max_instances]

    if part_mask is None:
        gt_part_mask = np.where(inst_mask > 0, 1, 0).astype(np.uint8)
    else:
        gt_part_mask = np.where(inst_mask > 0, part_mask, 0).astype(np.uint8)

    pose_mesh_rgb = rgb.copy()
    pred_pose_mask = np.zeros((h, w), dtype=np.uint8)
    rows = []
    instance_results = []
    for label in labels:
        inst_bool = inst_mask == label
        if int(inst_bool.sum()) < int(args.min_area):
            continue
        crop_info = crop_instance(rgb, K_orig, inst_bool, args.crop_size)
        x = to_tensor(Image.fromarray(crop_info["crop_rgb"])).unsqueeze(0).to(device)
        K_crop = torch.from_numpy(crop_info["K_crop"]).unsqueeze(0).to(device)
        with torch.inference_mode(), torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda"), dtype=torch.bfloat16):
            out = model(x, K_crop, masks_enc=None, masks_pred=None)
        pred_hm_crop_t, hm_scores_t = heatmap_argmax(out["keypoint_heatmaps"].float().detach().cpu())
        pred_hm_crop = pred_hm_crop_t[0].numpy().astype(np.float32)
        hm_scores = hm_scores_t[0].numpy().astype(np.float32)
        direct_pose = pose_from_output(out)
        if args.pose_recovery == "pnp":
            pred_pose = pose_from_keypoints_pnp(
                pred_hm_crop,
                [direct_pose["alpha"], direct_pose["theta_l"], direct_pose["theta_r"]],
                crop_info["K_crop"],
                scores=hm_scores,
                min_score=args.pnp_min_score,
            )
        else:
            pred_pose = direct_pose
        pred_kp_cam = instrument_keypoints_camera_np(
            pred_pose["rot"],
            pred_pose["trans"],
            [pred_pose["alpha"], pred_pose["theta_l"], pred_pose["theta_r"]],
        )
        pred_pose_crop = project_points_np(pred_kp_cam, crop_info["K_crop"]).astype(np.float32)
        pred_pose_orig = project_points_np(pred_kp_cam, K_orig).astype(np.float32)
        pred_hm_orig = crop_points_to_orig_np(pred_hm_crop, crop_info)
        pose_mesh_rgb = mesh_renderer.render_pose_overlay(pose_mesh_rgb, pred_pose, K_orig, alpha=args.overlay_alpha)
        mask_i = mesh_renderer.render_pose_mask(pred_pose, K_orig, rgb.shape[:2])
        pred_pose_mask[mask_i > 0] = mask_i[mask_i > 0]

        instance_results.append(
            {
                **crop_info,
                "pred_hm_crop": pred_hm_crop,
                "pred_pose_crop": pred_pose_crop,
                "pred_hm_orig": pred_hm_orig,
                "pred_pose_orig": pred_pose_orig,
                "crop_rgb": crop_info["crop_rgb"],
            }
        )
        rows.append(
            {
                "image": str(image_path),
                "instance_label": label,
                "area": int(inst_bool.sum()),
                "mask_path": str(inst_path),
                "part_mask_path": "" if part_path is None else str(part_path),
                "action_alpha": pred_pose["alpha"],
                "action_theta_l": pred_pose["theta_l"],
                "action_theta_r": pred_pose["theta_r"],
                "trans_x": float(pred_pose["trans"][0]),
                "trans_y": float(pred_pose["trans"][1]),
                "trans_z": float(pred_pose["trans"][2]),
                "hm_score_mean": float(np.mean(hm_scores)),
                "pose_recovery": args.pose_recovery,
            }
        )

    out_name = f"{image_path.stem}_robopepp_gtseg_pose.jpg"
    out_path = Path(args.output_dir) / out_name
    save_demo_visual(out_path, rgb, gt_part_mask, pred_pose_mask, pose_mesh_rgb, instance_results, args.panel_size)
    return rows, out_path


def worker_main(rank, args, image_paths):
    device = configure_device(args.devices[rank])
    model, _ = load_model(args.checkpoint, device)
    mesh_renderer = GMSInstrumentTrimeshRenderer(device)
    to_tensor = tv_transforms.Compose(
        [
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    all_rows = []
    for i, image_path in enumerate(image_paths):
        rows, out_path = process_image(image_path, model, mesh_renderer, device, args, to_tensor)
        all_rows.extend(rows)
        print(f"[worker {rank}] {image_path.name}: {len(rows)} GT instances -> {out_path.name}", flush=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    manifest = Path(args.output_dir) / f"worker_{rank}_manifest.json"
    manifest.write_text(json.dumps(all_rows, indent=2), encoding="utf-8")


def split_evenly(items, n):
    return [items[i::n] for i in range(n)]


def run(args):
    image_paths = list_images(args.input_dir, args.max_images)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    args.devices = args.devices or [args.device]
    chunks = [chunk for chunk in split_evenly(image_paths, len(args.devices)) if chunk]
    args.devices = args.devices[: len(chunks)]
    print(f"Processing {len(image_paths)} image(s) -> {args.output_dir}; devices={args.devices}", flush=True)
    if len(chunks) == 1:
        worker_main(0, args, chunks[0])
    else:
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

    rows = []
    for p in sorted(Path(args.output_dir).glob("worker_*_manifest.json")):
        rows.extend(json.loads(p.read_text(encoding="utf-8")))
    if rows:
        csv_path = Path(args.output_dir) / "prediction_summary.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {csv_path}", flush=True)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="logs/robopepp_instrument_pose_rarp_jepa_bs56_gpu0123/checkpoints/last.pt")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--inst_mask_dir", type=str, required=True)
    parser.add_argument("--part_mask_dir", type=str, default="")
    parser.add_argument("--part_label_mode", type=str, default="project", choices=["project", "surgpose"])
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--camera_metadata", type=str, default="")
    parser.add_argument("--camera_key", type=str, default="K_left_rectified_scaled")
    parser.add_argument("--focal", type=float, default=-1.0)
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--panel_size", type=int, default=630)
    parser.add_argument("--overlay_alpha", type=float, default=0.85)
    parser.add_argument("--pose_recovery", type=str, default="pnp", choices=["pnp", "direct"])
    parser.add_argument("--pnp_min_score", type=float, default=0.0)
    parser.add_argument("--min_area", type=int, default=64)
    parser.add_argument("--max_instances", type=int, default=-1)
    parser.add_argument("--max_images", type=int, default=-1)
    parser.add_argument("--device", type=str, default="cuda:4")
    parser.add_argument("--devices", nargs="*", default=["cuda:4"])
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
