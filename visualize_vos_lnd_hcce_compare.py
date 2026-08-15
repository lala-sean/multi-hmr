import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as tv_transforms
from PIL import Image, ImageDraw, ImageFont

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

ROBOPEPP_ROOT = Path(__file__).resolve().parent
MULTIHMR_ROOT = ROBOPEPP_ROOT.parents[1]
if str(ROBOPEPP_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBOPEPP_ROOT))
if str(MULTIHMR_ROOT) not in sys.path:
    sys.path.insert(0, str(MULTIHMR_ROOT))

import compare_crop_hcce_robopepp_rarp as cmp  # noqa: E402
from instrument_geometry import rarp_intrinsics  # noqa: E402
from instrument_trimesh_renderer import GMSInstrumentTrimeshRenderer  # noqa: E402
from predict_instrument_pose import (  # noqa: E402
    add_title,
    concat_panels,
    overlay_part_mask,
    pad_mask_to_square,
    pad_rgb_to_square,
    square_image_geometry,
)


VOS17_LABEL_ROOT = Path("/mnt/nas/share/shuojue/VOS-Endovis17")
VOS18_LABEL_ROOT = Path("/mnt/nas/share/shuojue/VOS-Endovis18")
VOS17_IMAGE_ROOT = Path("/mnt/nas/haofeng/data/VOS-Endovis17")
VOS18_IMAGE_ROOT = Path("/mnt/nas/haofeng/data/VOS-Endovis18")

OLD_HCCE_CKPT = ROBOPEPP_ROOT / "logs/hcce_crop224_fixbf16_rarp_gpu0123_bs56_fromscratch/checkpoints/last.pt"
HYBRID_HCCE_CKPT = (
    MULTIHMR_ROOT
    / "logs/instrument_hcce_crop_surgpose_keypoint_hybrid_gpu4567_bs56_bf16_fromscratch/checkpoints/last.pt"
)

PART_REMAP = {1: 3, 2: 1, 3: 2, 4: 1}


def configure_device(device_name):
    if str(device_name).startswith("cuda"):
        idx = int(str(device_name).split(":", 1)[1]) if ":" in str(device_name) else 0
        os.environ["EGL_DEVICE_ID"] = str(idx)
        torch.cuda.set_device(idx)
    return torch.device(device_name if torch.cuda.is_available() else "cpu")


def remap_part(part_raw, inst_mask):
    out = np.zeros_like(part_raw, dtype=np.uint8)
    for raw_value, label in PART_REMAP.items():
        out[inst_mask & (part_raw == raw_value)] = int(label)
    return out


def load_vos_roots(dataset):
    if dataset == "endovis17":
        return VOS17_LABEL_ROOT, VOS17_IMAGE_ROOT
    if dataset == "endovis18":
        return VOS18_LABEL_ROOT, VOS18_IMAGE_ROOT
    raise ValueError(dataset)


def image_path_for(image_root, split, seq, stem):
    path = image_root / split / "images" / seq / f"{stem}.png"
    if path.is_file():
        return path
    if split == "valid":
        path = image_root / "test" / "images" / seq / f"{stem}.png"
        if path.is_file():
            return path
    raise FileNotFoundError(path)


def camera_K(width, height, focal):
    if float(focal) > 0.0:
        return np.array(
            [[float(focal), 0.0, float(width) / 2.0], [0.0, float(focal), float(height) / 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
    return rarp_intrinsics(width, height)


def enumerate_lnd_frames(args):
    frames = []
    for dataset in args.datasets:
        label_root, image_root = load_vos_roots(dataset)
        for split in args.splits:
            meta_dir = image_root / split / "Meta"
            mask_root = label_root / split / "masks_new"
            part_root = label_root / split / "parts_new"
            if not meta_dir.is_dir() or not mask_root.is_dir() or not part_root.is_dir():
                continue
            for meta_path in sorted(meta_dir.glob("*.json")):
                meta = json.load(open(meta_path, "r"))
                categories = meta.get("info", {}).get("category", {})
                lnd_ids = sorted(int(k) for k, v in categories.items() if str(v).lower() == "large needle driver")
                if not lnd_ids:
                    continue
                seq = meta_path.stem
                frame_idx = -1
                for mask_path in sorted((mask_root / seq).glob("*.png")):
                    frame_idx += 1
                    stem = mask_path.stem
                    if int(args.frame_stride) > 1 and frame_idx % int(args.frame_stride) != 0:
                        continue
                    if int(args.frame_start) >= 0 and int(stem) < int(args.frame_start):
                        continue
                    if int(args.frame_end) >= 0 and int(stem) > int(args.frame_end):
                        continue
                    mask_np = np.asarray(Image.open(mask_path))
                    visible = []
                    for inst_id in lnd_ids:
                        area = int(np.count_nonzero(mask_np == inst_id))
                        if area >= int(args.min_area):
                            visible.append({"inst_id": int(inst_id), "area": area})
                    if visible:
                        frames.append(
                            {
                                "dataset": dataset,
                                "split": split,
                                "seq": seq,
                                "stem": stem,
                                "visible": visible,
                                "image_path": str(image_path_for(image_root, split, seq, stem)),
                                "mask_path": str(mask_path),
                                "part_path": str(part_root / seq / f"{stem}.png"),
                            }
                        )
    if int(args.max_frames) > 0:
        frames = frames[: int(args.max_frames)]
    return frames


def tensor_input(crop_rgb, to_tensor, device):
    return to_tensor(Image.fromarray(crop_rgb.astype(np.uint8))).unsqueeze(0).to(device, non_blocking=True)


def run_hcce(model, crop_rgb, K_crop, to_tensor, device):
    x = tensor_input(crop_rgb, to_tensor, device)
    K = torch.from_numpy(np.asarray(K_crop, dtype=np.float32)).unsqueeze(0).to(device, non_blocking=True)
    with torch.inference_mode(), torch.amp.autocast(
        device_type="cuda",
        enabled=(device.type == "cuda"),
        dtype=torch.bfloat16,
    ):
        return model(x, K)


def make_fit_args(args, meta):
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
    fit_args.gripper_no_wrist_pose = int(args.gripper_no_wrist_pose)
    fit_args.min_depth = float(args.min_depth)
    fit_args.behind_camera_penalty = float(args.behind_camera_penalty)
    fit_args.hcce_bits = int(meta.get("hcce_bits", 8))
    fit_args.hcce_coord_min = float(meta.get("hcce_coord_min", -1.0))
    fit_args.hcce_coord_max = float(meta.get("hcce_coord_max", 1.0))
    fit_args.hcce_bit_thresh = float(args.hcce_bit_thresh)
    fit_args.hcce_axis_scale = args.hcce_axis_scale
    return fit_args


def render_many(renderer, rgb, poses, K_orig, alpha):
    out = rgb.copy()
    for pose in poses:
        if pose is not None:
            out = renderer.render_pose_overlay(out, pose, K_orig, alpha=alpha)
    return out


def add_lnd_boxes(rgb, items):
    out = rgb.copy()
    for item in items:
        x0, y0, x1, y1 = item["bbox"]
        cv2.rectangle(out, (x0, y0), (x1, y1), (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(
            out,
            f"LND id{item['inst_id']}",
            (x0 + 5, max(18, y0 + 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return out


def blank_like(rgb, text):
    out = rgb.copy()
    cv2.putText(out, text[:120], (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 80, 80), 2, cv2.LINE_AA)
    return out


def save_visual(out_path, rgb, gt_part, instances, method_poses, method_status, renderer, K_orig, args):
    scale, pad_x, pad_y = square_image_geometry(rgb, int(args.panel_size))
    rgb_sq = pad_rgb_to_square(add_lnd_boxes(rgb, instances), int(args.panel_size), scale, pad_x, pad_y)
    gt_sq = pad_mask_to_square(gt_part, int(args.panel_size), scale, pad_x, pad_y)
    panels = [
        add_title(rgb_sq, "rgb + LND GT bbox"),
        add_title(overlay_part_mask(rgb_sq, gt_sq), "GT part seg"),
    ]
    for title, poses in method_poses:
        ok_count = sum(p is not None for p in poses)
        if ok_count:
            overlay = render_many(renderer, rgb, poses, K_orig, float(args.overlay_alpha))
        else:
            overlay = blank_like(rgb, method_status.get(title, "no pose"))
        panel = pad_rgb_to_square(overlay, int(args.panel_size), scale, pad_x, pad_y)
        panels.append(add_title(panel, f"{title} ({ok_count}/{len(poses)} ok)"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(concat_panels(panels)).save(out_path, quality=92)


def process_frame(frame, models, renderer, cad, to_tensor, device, args, rng):
    rgb = np.asarray(Image.open(frame["image_path"]).convert("RGB"))
    mask_np = np.asarray(Image.open(frame["mask_path"]))
    part_raw = np.asarray(Image.open(frame["part_path"]))
    h, w = rgb.shape[:2]
    K_orig = camera_K(w, h, args.focal)

    instances = []
    gt_part_all = np.zeros((h, w), dtype=np.uint8)
    old_fit_poses = []
    new_fit_poses = []
    old_kp_poses = []
    rows = []

    old_fit_args = make_fit_args(args, models["old_meta"])
    new_fit_args = make_fit_args(args, models["new_meta"])

    for visible in frame["visible"]:
        inst_id = int(visible["inst_id"])
        inst_mask = mask_np == inst_id
        inst_part = remap_part(part_raw, inst_mask)
        gt_part_all[inst_part > 0] = inst_part[inst_part > 0]
        ys, xs = np.where(inst_mask)
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        crop = cmp.crop_from_gt_mask(rgb, K_orig, inst_part, int(args.crop_size))
        target_like = {
            "orig_rgb": rgb.astype(np.uint8),
            "crop_rgb": crop["crop_rgb"],
            "gt_part_orig": inst_part.astype(np.uint8),
            "gt_part_crop": crop["gt_part_crop"],
            "K_orig": K_orig.astype(np.float32),
            "K_crop": crop["K_crop"],
            "bbox_min": crop["bbox_min"],
            "bbox_max": crop["bbox_max"],
            "scale": crop["scale"],
            "pad": crop["pad"],
            "keypoints_crop": None,
            "keypoints_orig": None,
            "keypoints_valid": None,
            "keypoints_valid_orig": None,
        }

        old_out = run_hcce(models["old"], crop["crop_rgb"], crop["K_crop"], to_tensor, device)
        new_out = run_hcce(models["new"], crop["crop_rgb"], crop["K_crop"], to_tensor, device)

        row = {
            "dataset": frame["dataset"],
            "split": frame["split"],
            "seq": frame["seq"],
            "frame_id": frame["stem"],
            "inst_id": inst_id,
            "area": int(visible["area"]),
            "bbox_x0": bbox[0],
            "bbox_y0": bbox[1],
            "bbox_x1": bbox[2],
            "bbox_y1": bbox[3],
            "image_path": frame["image_path"],
            "mask_path": frame["mask_path"],
            "part_path": frame["part_path"],
        }

        try:
            pose, extra = cmp.fit_pose_from_hcce(old_out, cad, models["old_meta"], target_like, old_fit_args, rng)
            row["old_hcce_fit_status"] = "ok"
            row.update({f"old_{k}": v for k, v in extra.items()})
        except Exception as exc:
            pose = None
            row["old_hcce_fit_status"] = f"{type(exc).__name__}: {exc}"
        old_fit_poses.append(pose)

        try:
            pose, extra = cmp.fit_pose_from_hcce(new_out, cad, models["new_meta"], target_like, new_fit_args, rng)
            row["hybrid_hcce_fit_status"] = "ok"
            row.update({f"hybrid_{k}": v for k, v in extra.items()})
        except Exception as exc:
            pose = None
            row["hybrid_hcce_fit_status"] = f"{type(exc).__name__}: {exc}"
        new_fit_poses.append(pose)

        if int(models["old_meta"].get("num_keypoints", 5)) == len(cmp.KEYPOINT_NAMES):
            try:
                pose, _, scores, _ = cmp.pnp_from_heatmap_output(old_out, crop["K_crop"], float(args.pnp_min_score))
                row["old_keypoint_pnp_status"] = "ok"
                row["old_keypoint_score_mean"] = float(np.mean(scores))
                row["old_keypoint_score_min"] = float(np.min(scores))
            except Exception as exc:
                pose = None
                row["old_keypoint_pnp_status"] = f"{type(exc).__name__}: {exc}"
        else:
            pose = None
            row["old_keypoint_pnp_status"] = (
                f"skipped: old num_keypoints={int(models['old_meta'].get('num_keypoints', -1))} "
                f"does not match RARP geometry keypoints={len(cmp.KEYPOINT_NAMES)}"
            )
        old_kp_poses.append(pose)

        instances.append({"inst_id": inst_id, "bbox": bbox, "area": int(visible["area"])})
        rows.append(row)

    method_poses = [
        ("old HCCE fit", old_fit_poses),
        ("hybrid HCCE fit", new_fit_poses),
        ("old keypoint PnP", old_kp_poses),
    ]
    method_status = {
        "old HCCE fit": "; ".join(r["old_hcce_fit_status"] for r in rows),
        "hybrid HCCE fit": "; ".join(r["hybrid_hcce_fit_status"] for r in rows),
        "old keypoint PnP": "; ".join(r["old_keypoint_pnp_status"] for r in rows),
    }
    out_path = (
        Path(args.output_dir)
        / "vis"
        / frame["dataset"]
        / frame["split"]
        / frame["seq"]
        / f"{frame['dataset']}_{frame['split']}_{frame['seq']}_{frame['stem']}_lnd.jpg"
    )
    save_visual(out_path, rgb, gt_part_all, instances, method_poses, method_status, renderer, K_orig, args)
    for row in rows:
        row["vis_path"] = str(out_path)
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


def make_contact_sheets(vis_paths, output_dir, chunk_size, max_width):
    if int(chunk_size) <= 0:
        return []
    out_dir = Path(output_dir) / "contact_sheets"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    except Exception:
        font = None
    sheets = []
    for sheet_idx, start in enumerate(range(0, len(vis_paths), int(chunk_size))):
        chunk = vis_paths[start : start + int(chunk_size)]
        items = []
        for path in chunk:
            im = Image.open(path).convert("RGB")
            if im.width > int(max_width):
                new_h = int(round(im.height * int(max_width) / im.width))
                im = im.resize((int(max_width), new_h), Image.Resampling.LANCZOS)
            label_h = 36
            canvas = Image.new("RGB", (im.width, im.height + label_h), "white")
            canvas.paste(im, (0, label_h))
            draw = ImageDraw.Draw(canvas)
            draw.text((10, 6), Path(path).name, fill=(0, 0, 0), font=font)
            items.append(canvas)
        if not items:
            continue
        sheet = Image.new("RGB", (max(im.width for im in items), sum(im.height for im in items)), "white")
        y = 0
        for im in items:
            sheet.paste(im, (0, y))
            y += im.height
        out_path = out_dir / f"vos_lnd_sheet_{sheet_idx:03d}_{start:05d}_{start + len(chunk) - 1:05d}.jpg"
        sheet.save(out_path, quality=92)
        sheets.append(str(out_path))
    return sheets


def summarize(rows, frames, vis_paths, sheets, args):
    status = {}
    for key in ("old_hcce_fit_status", "hybrid_hcce_fit_status", "old_keypoint_pnp_status"):
        status[key] = dict(Counter(row.get(key, "") for row in rows))
    by_dataset = Counter(frame["dataset"] for frame in frames)
    by_split = Counter(f"{frame['dataset']}/{frame['split']}" for frame in frames)
    payload = {
        "frames": len(frames),
        "instances": len(rows),
        "visuals": len(vis_paths),
        "by_dataset": dict(by_dataset),
        "by_split": dict(by_split),
        "status": status,
        "old_hcce_checkpoint": str(args.old_hcce_checkpoint),
        "hybrid_hcce_checkpoint": str(args.hybrid_hcce_checkpoint),
        "fit_seg_source": "gt",
        "intrinsics": "rarp_intrinsics" if float(args.focal) <= 0.0 else f"focal={args.focal}",
        "frame_stride": int(args.frame_stride),
    }
    Path(args.output_dir, "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# VOS LND HCCE Compare",
        "",
        f"- frames: {len(frames)}",
        f"- instances: {len(rows)}",
        f"- visuals: {len(vis_paths)}",
        f"- vis dir: `{Path(args.output_dir) / 'vis'}`",
        f"- contact sheets: `{Path(args.output_dir) / 'contact_sheets'}`",
        f"- old HCCE ckpt: `{args.old_hcce_checkpoint}`",
        f"- hybrid HCCE ckpt: `{args.hybrid_hcce_checkpoint}`",
        f"- fit segmentation source: GT part mask",
        f"- intrinsics: {payload['intrinsics']}",
        "",
        "Status counts:",
        "```json",
        json.dumps(status, indent=2),
        "```",
    ]
    Path(args.output_dir, "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    Path(args.output_dir, "vis_paths.txt").write_text("\n".join(map(str, vis_paths)) + "\n", encoding="utf-8")
    Path(args.output_dir, "contact_sheets.json").write_text(json.dumps(sheets, indent=2), encoding="utf-8")


def run(args):
    args.output_dir = str(Path(args.output_dir).resolve())
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    device = configure_device(args.device)
    frames = enumerate_lnd_frames(args)
    if not frames:
        raise RuntimeError("No visible large needle driver frames found")
    print(f"[config] selected frames={len(frames)}", flush=True)
    print(f"[config] output_dir={args.output_dir}", flush=True)
    print(f"[config] old_hcce={args.old_hcce_checkpoint}", flush=True)
    print(f"[config] hybrid_hcce={args.hybrid_hcce_checkpoint}", flush=True)

    old_model, old_meta = cmp.load_hcce_model(args.old_hcce_checkpoint, device)
    new_model, new_meta = cmp.load_hcce_model(args.hybrid_hcce_checkpoint, device)
    models = {"old": old_model, "old_meta": old_meta, "new": new_model, "new_meta": new_meta}
    renderer = GMSInstrumentTrimeshRenderer(device)
    cad = cmp.InstrumentCAD(cmp.CAD_ROOT)
    to_tensor = tv_transforms.Compose(
        [
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    rng = np.random.default_rng(int(args.seed))

    all_rows = []
    vis_paths = []
    for idx, frame in enumerate(frames):
        rows, vis_path = process_frame(frame, models, renderer, cad, to_tensor, device, args, rng)
        all_rows.extend(rows)
        vis_paths.append(str(vis_path))
        if idx == 0 or (idx + 1) % int(args.print_freq) == 0 or idx + 1 == len(frames):
            print(f"[{idx + 1}/{len(frames)}] wrote {vis_path}", flush=True)
    write_csv(Path(args.output_dir) / "prediction_summary.csv", all_rows)
    sheets = make_contact_sheets(vis_paths, args.output_dir, args.contact_sheet_chunk, args.contact_sheet_width)
    summarize(all_rows, frames, vis_paths, sheets, args)
    print(f"[ok] wrote {args.output_dir}", flush=True)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["endovis17", "endovis18"], choices=["endovis17", "endovis18"])
    parser.add_argument("--splits", nargs="+", default=["test"], choices=["train", "valid", "test"])
    parser.add_argument("--output_dir", type=str, default=str(ROBOPEPP_ROOT / "logs/vos_lnd_hcce_compare"))
    parser.add_argument("--old_hcce_checkpoint", type=str, default=str(OLD_HCCE_CKPT))
    parser.add_argument("--hybrid_hcce_checkpoint", type=str, default=str(HYBRID_HCCE_CKPT))
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--panel_size", type=int, default=640)
    parser.add_argument("--overlay_alpha", type=float, default=0.85)
    parser.add_argument("--focal", type=float, default=-1.0)
    parser.add_argument("--min_area", type=int, default=64)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--frame_start", type=int, default=-1)
    parser.add_argument("--frame_end", type=int, default=-1)
    parser.add_argument("--max_frames", type=int, default=-1)
    parser.add_argument("--inst_thresh", type=float, default=0.5)
    parser.add_argument("--surface_k_faces", type=int, default=0)
    parser.add_argument("--shaft_raw_x_min", type=float, default=-0.5)
    parser.add_argument("--max_points_per_part", type=int, default=300)
    parser.add_argument("--min_wrist_points", type=int, default=12)
    parser.add_argument("--min_total_points", type=int, default=24)
    parser.add_argument("--min_shaft_points", type=int, default=24)
    parser.add_argument("--pnp_min_score", type=float, default=0.0)
    parser.add_argument("--optim_strategy", type=str, default="decoupled", choices=["decoupled", "single"])
    parser.add_argument("--optim_parts", type=str, default="wrist_gripper", choices=["wrist_gripper", "all"])
    parser.add_argument("--optim_loss", type=str, default="soft_l1")
    parser.add_argument("--optim_f_scale", type=float, default=8.0)
    parser.add_argument("--optim_max_nfev", type=int, default=80)
    parser.add_argument("--gripper_no_wrist_pose", type=int, choices=[0, 1], default=0)
    parser.add_argument("--hcce_bit_thresh", type=float, default=0.5)
    parser.add_argument("--hcce_axis_scale", type=str, default=None)
    parser.add_argument("--min_depth", type=float, default=1e-4)
    parser.add_argument("--behind_camera_penalty", type=float, default=1e4)
    parser.add_argument("--contact_sheet_chunk", type=int, default=24)
    parser.add_argument("--contact_sheet_width", type=int, default=2400)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--print_freq", type=int, default=20)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
