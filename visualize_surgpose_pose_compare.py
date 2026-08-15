import argparse
import csv
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as tv_transforms
from PIL import Image

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

ROBOPEPP_ROOT = Path(__file__).resolve().parent
MULTIHMR_ROOT = ROBOPEPP_ROOT.parents[1]
if str(ROBOPEPP_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBOPEPP_ROOT))
if str(MULTIHMR_ROOT) not in sys.path:
    sys.path.insert(0, str(MULTIHMR_ROOT))

import compare_crop_hcce_robopepp_rarp as cmp  # noqa: E402
from demo_gtseg_pose import crop_instance, load_camera_K  # noqa: E402
from instrument_geometry import instrument_keypoints_camera_np, project_points_np  # noqa: E402
from instrument_trimesh_renderer import GMSInstrumentTrimeshRenderer  # noqa: E402
from predict_instrument_pose import (  # noqa: E402
    add_title,
    concat_panels,
    heatmap_argmax,
    load_model as load_robopepp_model,
    overlay_part_mask,
    pad_mask_to_square,
    pad_rgb_to_square,
    square_image_geometry,
)


def configure_device(device_name):
    if str(device_name).startswith("cuda"):
        idx = int(str(device_name).split(":", 1)[1]) if ":" in str(device_name) else 0
        os.environ["EGL_DEVICE_ID"] = str(idx)
        torch.cuda.set_device(idx)
    return torch.device(device_name if torch.cuda.is_available() else "cpu")


def list_images(input_dir, max_images):
    paths = sorted(
        p for p in Path(input_dir).iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg")
    )
    if int(max_images) > 0:
        paths = paths[: int(max_images)]
    if not paths:
        raise RuntimeError(f"No images under {input_dir}")
    return paths


def frame_number_from_path(path):
    stem = Path(path).stem
    digits = "".join(ch if ch.isdigit() else " " for ch in stem).split()
    if not digits:
        return None
    return int(digits[-1])


def filter_images(paths, args):
    out = []
    for path in paths:
        frame_num = frame_number_from_path(path)
        if frame_num is None:
            continue
        if args.frame_parity == "even" and frame_num % 2 != 0:
            continue
        if args.frame_parity == "odd" and frame_num % 2 != 1:
            continue
        if int(args.frame_stride) > 1 and frame_num % int(args.frame_stride) != 0:
            continue
        if int(args.frame_start) >= 0 and frame_num < int(args.frame_start):
            continue
        if int(args.frame_end) >= 0 and frame_num > int(args.frame_end):
            continue
        out.append(path)
    if int(args.max_images) > 0:
        out = out[: int(args.max_images)]
    if not out:
        raise RuntimeError(f"No images left after frame filters under {args.input_dir}")
    return out


def load_part_mask(part_mask_dir, image_path, shape_hw):
    path = Path(part_mask_dir) / f"{image_path.stem}.png"
    if not path.is_file():
        raise FileNotFoundError(path)
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.shape[:2] != tuple(shape_hw):
        arr = cv2.resize(arr.astype(np.uint8), (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)
    return arr.astype(np.uint8), path


def remap_surgpose_part(raw_part):
    # SurgPose raw labels: 1 shaft, 2 wrist, 3 gripper. Our convention: 1 gripper, 2 wrist, 3 shaft.
    out = np.zeros_like(raw_part, dtype=np.uint8)
    out[raw_part == 1] = 3
    out[raw_part == 2] = 2
    out[raw_part == 3] = 1
    return out


def connected_instances(part_mask, min_area):
    n, labels, stats, _ = cv2.connectedComponentsWithStats((part_mask > 0).astype(np.uint8), 8)
    items = []
    for label in range(1, n):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < int(min_area):
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        items.append(
            {
                "instance_id": len(items) + 1,
                "component_label": int(label),
                "area": area,
                "bbox": [x, y, w, h],
                "mask": labels == label,
            }
        )
    return items


def crop_mask_like(mask_orig, crop_info, crop_size):
    return cmp.crop_resize_pad_map(
        mask_orig,
        crop_info["bbox_min"],
        crop_info["bbox_max"],
        int(crop_size),
        cv2.INTER_NEAREST,
        value=0,
    ).astype(np.uint8)


def target_like_from_surgpose(rgb, K_orig, part_mask_std, inst_mask, crop_info, crop_size):
    inst_part = np.where(inst_mask, part_mask_std, 0).astype(np.uint8)
    return {
        "orig_rgb": rgb.astype(np.uint8),
        "crop_rgb": crop_info["crop_rgb"].astype(np.uint8),
        "K_orig": K_orig.astype(np.float32),
        "K_crop": crop_info["K_crop"].astype(np.float32),
        "bbox_min": crop_info["bbox_min"].astype(np.float32),
        "bbox_max": crop_info["bbox_max"].astype(np.float32),
        "scale": crop_info["scale"].astype(np.float32),
        "pad": crop_info["pad"].astype(np.float32),
        "gt_part_orig": inst_part,
        "gt_part_crop": crop_mask_like(inst_part, crop_info, crop_size),
    }


def tensor_input(crop_rgb, to_tensor, device):
    return to_tensor(Image.fromarray(crop_rgb.astype(np.uint8))).unsqueeze(0).to(device)


def run_hcce_model(model, crop_rgb, K_crop, to_tensor, device):
    x = tensor_input(crop_rgb, to_tensor, device)
    K = torch.from_numpy(np.asarray(K_crop, dtype=np.float32)).unsqueeze(0).to(device)
    with torch.inference_mode(), torch.amp.autocast(
        device_type="cuda",
        enabled=(device.type == "cuda"),
        dtype=torch.bfloat16,
    ):
        return model(x, K)


def run_robopepp_model(model, crop_rgb, K_crop, to_tensor, device):
    x = tensor_input(crop_rgb, to_tensor, device)
    K = torch.from_numpy(np.asarray(K_crop, dtype=np.float32)).unsqueeze(0).to(device)
    with torch.inference_mode(), torch.amp.autocast(
        device_type="cuda",
        enabled=(device.type == "cuda"),
        dtype=torch.bfloat16,
    ):
        return model(x, K, masks_enc=None, masks_pred=None)


def make_fit_args(args):
    fit_args = cmp.build_parser().parse_args([])
    fit_args.crop_size = int(args.crop_size)
    fit_args.inst_thresh = float(args.inst_thresh)
    fit_args.fit_seg_source = "gt"
    fit_args.surface_snap_method = "surface"
    fit_args.surface_k_faces = int(args.surface_k_faces)
    fit_args.point_select = "random"
    fit_args.shaft_raw_x_min = float(args.shaft_raw_x_min)
    fit_args.max_points_per_part = int(args.max_points_per_part)
    fit_args.min_wrist_points = int(args.min_wrist_points)
    fit_args.min_total_points = int(args.min_total_points)
    fit_args.min_shaft_points = int(args.min_shaft_points)
    fit_args.pnp_min_score = float(args.pnp_min_score)
    fit_args.optim_strategy = str(args.optim_strategy)
    fit_args.optim_parts = str(args.optim_parts)
    fit_args.optim_loss = str(args.optim_loss)
    fit_args.optim_f_scale = float(args.optim_f_scale)
    fit_args.optim_max_nfev = int(args.optim_max_nfev)
    fit_args.min_depth = float(args.min_depth)
    fit_args.behind_camera_penalty = float(args.behind_camera_penalty)
    return fit_args


def pose_keypoint_overlay(rgb, pose, K_orig, color=(255, 70, 220)):
    out = rgb.copy()
    if pose is None:
        return out
    action = [pose["alpha"], pose["theta_l"], pose["theta_r"]]
    points_cam = instrument_keypoints_camera_np(pose["rot"], pose["trans"], action)
    uv = project_points_np(points_cam, K_orig)
    for i, xy in enumerate(uv):
        if not np.isfinite(xy).all():
            continue
        x, y = np.round(xy).astype(int)
        cv2.drawMarker(out, (x, y), color, cv2.MARKER_CROSS, 13, 2, cv2.LINE_AA)
        cv2.putText(out, str(i), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
    return out


def render_many(renderer, rgb, poses, K_orig, alpha):
    panel = rgb.copy()
    for pose in poses:
        if pose is not None:
            panel = renderer.render_pose_overlay(panel, pose, K_orig, alpha=alpha)
    return panel


def add_instance_boxes(panel, instances):
    out = panel.copy()
    for item in instances:
        x, y, w, h = item["bbox"]
        cv2.rectangle(out, (x, y), (x + w, y + h), (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(
            out,
            f"inst{item['instance_id']}",
            (x + 4, max(18, y + 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return out


def save_frame_visual(out_path, rgb, gt_part, instances, method_poses, renderer, K_orig, args):
    scale, pad_x, pad_y = square_image_geometry(rgb, int(args.panel_size))
    rgb_box = add_instance_boxes(rgb, instances)
    rgb_sq = pad_rgb_to_square(rgb_box, int(args.panel_size), scale, pad_x, pad_y)
    gt_sq = pad_mask_to_square(gt_part, int(args.panel_size), scale, pad_x, pad_y)
    panels = [
        add_title(rgb_sq, "rgb + gt boxes"),
        add_title(overlay_part_mask(rgb_sq, gt_sq), "gt part segmentation"),
    ]
    for title, poses in method_poses:
        overlay = render_many(renderer, rgb, poses, K_orig, float(args.overlay_alpha))
        if "keypoint" in title.lower() or "robopepp" in title.lower():
            for pose in poses:
                overlay = pose_keypoint_overlay(overlay, pose, K_orig)
        panel_sq = pad_rgb_to_square(overlay, int(args.panel_size), scale, pad_x, pad_y)
        panels.append(add_title(panel_sq, title))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(concat_panels(panels)).save(out_path)


def process_image(image_path, models, renderer, cad, to_tensor, device, args, rng):
    rgb = np.asarray(Image.open(image_path).convert("RGB"))
    h, w = rgb.shape[:2]
    K_orig = load_camera_K(args, w, h)
    raw_part, part_path = load_part_mask(args.part_mask_dir, image_path, (h, w))
    part_std = remap_surgpose_part(raw_part)
    instances = connected_instances(part_std, args.min_area)
    if int(args.max_instances) > 0:
        instances = instances[: int(args.max_instances)]

    hcce_fit_poses = []
    hcce_direct_poses = []
    hcce_kp_poses = []
    robo_poses = []
    rows = []
    fit_args = make_fit_args(args)
    for item in instances:
        crop_info = crop_instance(rgb, K_orig, item["mask"], int(args.crop_size))
        target_like = target_like_from_surgpose(
            rgb,
            K_orig,
            part_std,
            item["mask"],
            crop_info,
            int(args.crop_size),
        )
        hcce_out = run_hcce_model(models["hcce"], crop_info["crop_rgb"], crop_info["K_crop"], to_tensor, device)
        if bool(args.include_robopepp):
            robo_out = run_robopepp_model(models["robopepp"], crop_info["crop_rgb"], crop_info["K_crop"], to_tensor, device)
        else:
            robo_out = None

        try:
            hcce_direct_pose = cmp.pose_from_output(hcce_out)
            hcce_direct_status = "ok"
        except Exception as exc:
            hcce_direct_pose = None
            hcce_direct_status = f"{type(exc).__name__}: {exc}"

        try:
            hcce_fit_pose, fit_extra = cmp.fit_pose_from_hcce(
                hcce_out,
                cad,
                models["hcce_meta"],
                target_like,
                fit_args,
                rng,
            )
            hcce_fit_status = "ok"
        except Exception as exc:
            hcce_fit_pose = None
            fit_extra = {}
            hcce_fit_status = f"{type(exc).__name__}: {exc}"
        skip_hcce_kp_pnp = int(args.skip_hcce_kp_pnp) or int(models["hcce_meta"].get("num_keypoints", 5)) != len(cmp.KEYPOINT_NAMES)
        if skip_hcce_kp_pnp:
            hcce_kp_pose = None
            hcce_hm_scores = np.asarray([], dtype=np.float32)
            hcce_kp_status = (
                f"skipped: hcce num_keypoints={int(models['hcce_meta'].get('num_keypoints', -1))} "
                f"does not match RARP geometry keypoints={len(cmp.KEYPOINT_NAMES)}"
            )
        else:
            try:
                hcce_kp_pose, hcce_hm_crop, hcce_hm_scores, _ = cmp.pnp_from_heatmap_output(
                    hcce_out,
                    crop_info["K_crop"],
                    float(args.pnp_min_score),
                )
                hcce_kp_status = "ok"
            except Exception as exc:
                hcce_kp_pose = None
                hcce_hm_scores = np.asarray([], dtype=np.float32)
                hcce_kp_status = f"{type(exc).__name__}: {exc}"
        if robo_out is not None:
            try:
                robo_pose, robo_hm_crop, robo_hm_scores, _ = cmp.pnp_from_heatmap_output(
                    robo_out,
                    crop_info["K_crop"],
                    float(args.pnp_min_score),
                )
                robo_status = "ok"
            except Exception as exc:
                robo_pose = None
                robo_hm_scores = np.asarray([], dtype=np.float32)
                robo_status = f"{type(exc).__name__}: {exc}"
        else:
            robo_pose = None
            robo_hm_scores = np.asarray([], dtype=np.float32)
            robo_status = "skipped"

        hcce_fit_poses.append(hcce_fit_pose)
        hcce_direct_poses.append(hcce_direct_pose)
        hcce_kp_poses.append(hcce_kp_pose)
        robo_poses.append(robo_pose)
        rows.append(
            {
                "image": str(image_path),
                "part_mask_path": str(part_path),
                "instance_id": int(item["instance_id"]),
                "component_label": int(item["component_label"]),
                "area": int(item["area"]),
                "bbox_x": int(item["bbox"][0]),
                "bbox_y": int(item["bbox"][1]),
                "bbox_w": int(item["bbox"][2]),
                "bbox_h": int(item["bbox"][3]),
                "hcce_fit_status": hcce_fit_status,
                "hcce_fit_rmse_all_px": fit_extra.get("hcce_fit_reproj_rmse_all_px", float("nan")),
                "hcce_fit_corr_count": fit_extra.get("hcce_fit_corr_count", 0),
                "hcce_direct_status": hcce_direct_status,
                "hcce_keypoint_pnp_status": hcce_kp_status,
                "hcce_keypoint_score_mean": float(np.mean(hcce_hm_scores)) if len(hcce_hm_scores) else float("nan"),
                "robopepp_pnp_status": robo_status,
                "robopepp_keypoint_score_mean": float(np.mean(robo_hm_scores)) if len(robo_hm_scores) else float("nan"),
            }
        )

    method_poses = [
        ("HCCE direct", hcce_direct_poses),
        ("HCCE fit from dense correspondence", hcce_fit_poses),
    ]
    if not (int(args.skip_hcce_kp_pnp) or int(models["hcce_meta"].get("num_keypoints", 5)) != len(cmp.KEYPOINT_NAMES)):
        method_poses.append(("HCCE keypoint PnP", hcce_kp_poses))
    if bool(args.include_robopepp):
        method_poses.append(("RoboPEPP keypoint PnP", robo_poses))
    out_path = Path(args.output_dir) / f"{image_path.stem}_pose_compare.jpg"
    save_frame_visual(out_path, rgb, part_std, instances, method_poses, renderer, K_orig, args)
    return rows, out_path


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_contact_sheets(image_paths, output_dir, chunk_size=40, max_width=2400):
    if int(chunk_size) <= 0 or not image_paths:
        return []
    out_dir = Path(output_dir) / "contact_sheets"
    out_dir.mkdir(parents=True, exist_ok=True)
    sheets = []
    try:
        from PIL import ImageDraw, ImageFont

        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    except Exception:
        ImageDraw = None
        font = None
    for chunk_idx, start in enumerate(range(0, len(image_paths), int(chunk_size))):
        chunk = image_paths[start : start + int(chunk_size)]
        rows = []
        for path in chunk:
            im = Image.open(path).convert("RGB")
            if im.width > int(max_width):
                new_h = int(round(im.height * int(max_width) / im.width))
                im = im.resize((int(max_width), new_h), Image.Resampling.LANCZOS)
            label_h = 34
            canvas = Image.new("RGB", (im.width, im.height + label_h), (255, 255, 255))
            canvas.paste(im, (0, label_h))
            if ImageDraw is not None:
                draw = ImageDraw.Draw(canvas)
                draw.text((10, 6), Path(path).name, fill=(0, 0, 0), font=font)
            rows.append(canvas)
        sheet = Image.new("RGB", (max(row.width for row in rows), sum(row.height for row in rows)), (255, 255, 255))
        y = 0
        for row in rows:
            sheet.paste(row, (0, y))
            y += row.height
        out_path = out_dir / f"contact_sheet_{chunk_idx:03d}_{start:05d}_{start + len(chunk) - 1:05d}.jpg"
        sheet.save(out_path, quality=92, optimize=True)
        sheets.append(str(out_path))
    (out_dir / "contact_sheets.json").write_text(json.dumps(sheets, indent=2), encoding="utf-8")
    return sheets


def run(args):
    device = configure_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models = {}
    models["hcce"], models["hcce_meta"] = cmp.load_hcce_model(args.hcce_checkpoint, device)
    if bool(args.include_robopepp):
        models["robopepp"], _ = load_robopepp_model(args.robopepp_checkpoint, device)
    renderer = GMSInstrumentTrimeshRenderer(device)
    cad = cmp.InstrumentCAD(cmp.CAD_ROOT)
    to_tensor = tv_transforms.Compose(
        [
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    rng = np.random.default_rng(int(args.seed))
    rows = []
    image_paths = filter_images(list_images(args.input_dir, 0), args)
    vis_paths = []
    print(f"[config] selected_images={len(image_paths)} input_dir={args.input_dir}", flush=True)
    for idx, image_path in enumerate(image_paths):
        image_rows, out_path = process_image(image_path, models, renderer, cad, to_tensor, device, args, rng)
        rows.extend(image_rows)
        vis_paths.append(str(out_path))
        print(f"[{idx + 1}] {image_path.name}: {len(image_rows)} instances -> {out_path}", flush=True)
    write_csv(output_dir / "prediction_summary.csv", rows)
    (output_dir / "vis_paths.txt").write_text("\n".join(vis_paths) + "\n", encoding="utf-8")
    sheets = make_contact_sheets(vis_paths, output_dir, chunk_size=args.contact_sheet_chunk, max_width=args.contact_sheet_width)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config["selected_images"] = len(image_paths)
    config["contact_sheets"] = sheets
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"[ok] wrote {output_dir}", flush=True)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default=str(MULTIHMR_ROOT / "demo_inputs/surgpose000_noncontig"))
    parser.add_argument(
        "--part_mask_dir",
        type=str,
        default="/mnt/nas/share/shuojue/data/surgpose/000000/processed_stereo_640/sam3_segmentaion_part",
    )
    parser.add_argument(
        "--camera_metadata",
        type=str,
        default="/mnt/nas/share/shuojue/data/surgpose/000000/processed_stereo_640/metadata.json",
    )
    parser.add_argument("--camera_key", type=str, default="K_left_rectified_scaled")
    parser.add_argument("--focal", type=float, default=-1.0)
    parser.add_argument("--output_dir", type=str, default=str(ROBOPEPP_ROOT / "logs/surgpose000_pose_compare_lastpt"))
    parser.add_argument("--hcce_checkpoint", type=str, default=str(cmp.DEFAULT_HCCE_CKPT))
    parser.add_argument("--robopepp_checkpoint", type=str, default=str(cmp.DEFAULT_ROBOPEPP_CKPT))
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--panel_size", type=int, default=640)
    parser.add_argument("--overlay_alpha", type=float, default=0.85)
    parser.add_argument("--min_area", type=int, default=64)
    parser.add_argument("--max_images", type=int, default=8)
    parser.add_argument("--frame_parity", type=str, default="all", choices=["all", "even", "odd"])
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--frame_start", type=int, default=-1)
    parser.add_argument("--frame_end", type=int, default=-1)
    parser.add_argument("--contact_sheet_chunk", type=int, default=40)
    parser.add_argument("--contact_sheet_width", type=int, default=2400)
    parser.add_argument("--max_instances", type=int, default=-1)
    parser.add_argument("--include_robopepp", type=int, choices=[0, 1], default=0)
    parser.add_argument("--inst_thresh", type=float, default=0.5)
    parser.add_argument("--surface_k_faces", type=int, default=0)
    parser.add_argument("--shaft_raw_x_min", type=float, default=-0.5)
    parser.add_argument("--max_points_per_part", type=int, default=1200)
    parser.add_argument("--min_wrist_points", type=int, default=12)
    parser.add_argument("--min_total_points", type=int, default=24)
    parser.add_argument("--min_shaft_points", type=int, default=24)
    parser.add_argument("--pnp_min_score", type=float, default=0.0)
    parser.add_argument("--skip_hcce_kp_pnp", type=int, choices=[0, 1], default=0)
    parser.add_argument("--optim_strategy", type=str, default="decoupled", choices=["decoupled", "single"])
    parser.add_argument("--optim_parts", type=str, default="wrist_gripper", choices=["wrist_gripper", "all"])
    parser.add_argument("--optim_loss", type=str, default="soft_l1")
    parser.add_argument("--optim_f_scale", type=float, default=8.0)
    parser.add_argument("--optim_max_nfev", type=int, default=200)
    parser.add_argument("--min_depth", type=float, default=1e-4)
    parser.add_argument("--behind_camera_penalty", type=float, default=1e4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
