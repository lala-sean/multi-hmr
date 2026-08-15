#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from eval_surfemb_articulated_rarp import (
    DEFAULT_MODELS,
    build_dataset,
    load_model,
    make_surfemb_crop,
    parse_model_specs,
    target_pose,
)
from instrument_geometry import fk_matrices_np, project_points_np
from instrument_opengl_renderer import InstrumentOpenGLDepthRenderer
from surfemb_articulated_pose import (
    encode_surface_keys,
    load_part_surfaces,
)


ROBOPEPP_ROOT = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = ROBOPEPP_ROOT / "logs" / "surfemb_wrist_gripper_correspondence_vis"
DEFAULT_INDICES = "0,433,438,874"

PART_COLORS = {
    "wrist": np.array([30, 220, 90], dtype=np.uint8),
    "l_gripper": np.array([40, 210, 255], dtype=np.uint8),
    "r_gripper": np.array([245, 80, 190], dtype=np.uint8),
}


def _font(size):
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for path in candidates:
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _label_tile(rgb, title, subtitle="", scale=2):
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8)).resize(
        (rgb.shape[1] * scale, rgb.shape[0] * scale), Image.Resampling.BILINEAR
    )
    header_h = 62
    out = Image.new("RGB", (image.width, image.height + header_h), (248, 248, 248))
    out.paste(image, (0, header_h))
    draw = ImageDraw.Draw(out)
    draw.text((10, 6), title, fill=(15, 15, 15), font=_font(20))
    if subtitle:
        draw.text((10, 34), subtitle, fill=(65, 65, 65), font=_font(15))
    return np.asarray(out)


def _spatial_sample(mask, count, seed):
    points = np.argwhere(np.asarray(mask, dtype=bool))
    if len(points) <= int(count):
        return points
    rng = np.random.default_rng(int(seed))
    selected = [int(rng.integers(0, len(points)))]
    min_dist = np.full((len(points),), np.inf, dtype=np.float64)
    for _ in range(1, int(count)):
        delta = points - points[selected[-1]]
        min_dist = np.minimum(min_dist, np.sum(delta * delta, axis=1))
        selected.append(int(np.argmax(min_dist)))
    return points[np.asarray(selected)]


def _crop_point_from_ds(yx, scale):
    yx = np.asarray(yx, dtype=np.float32)
    return np.stack(
        ((yx[:, 1] + 0.5) * scale - 0.5, (yx[:, 0] + 0.5) * scale - 0.5),
        axis=1,
    )


def _project_surface_matches(surface, key_indices, transform, K_crop):
    points = np.asarray(surface.points_m, dtype=np.float64)[np.asarray(key_indices)]
    points_cam = points @ transform[:3, :3].T + transform[:3, 3]
    return project_points_np(points_cam, K_crop), points_cam[:, 2]


def _error_color(error):
    if error <= 3.0:
        return (35, 220, 75)
    if error <= 8.0:
        return (255, 205, 35)
    return (245, 55, 55)


def _draw_matches(crop, query_uv, projected_uv, errors, part_names):
    out = np.asarray(crop, dtype=np.uint8).copy()
    for query, projected, error, part_name in zip(query_uv, projected_uv, errors, part_names):
        if not np.isfinite(projected).all():
            continue
        q = tuple(np.rint(query).astype(int))
        p = tuple(np.rint(projected).astype(int))
        color = _error_color(float(error))
        cv2.line(out, q, p, color, 1, cv2.LINE_AA)
        cv2.circle(out, q, 3, tuple(int(v) for v in PART_COLORS[part_name]), -1, cv2.LINE_AA)
        cv2.circle(out, q, 3, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.drawMarker(out, p, color, cv2.MARKER_TILTED_CROSS, 7, 1, cv2.LINE_AA)
    return out


def _overlay_effective_parts(crop, effective_part_crop):
    out = np.asarray(crop, dtype=np.uint8).copy()
    colors = {2: PART_COLORS["wrist"], 1: np.array([245, 80, 190], dtype=np.uint8)}
    for label, color in colors.items():
        mask = effective_part_crop == label
        out[mask] = np.clip(out[mask].astype(np.float32) * 0.42 + color * 0.58, 0, 255).astype(np.uint8)
    return out


def _overlay_predicted_parts(crop, effective_part_crop, pred_part_ds, scale):
    h, w = pred_part_ds.shape
    pred_crop = cv2.resize(pred_part_ds.astype(np.uint8), (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros(effective_part_crop.shape, dtype=np.uint8)
    canvas[: pred_crop.shape[0], : pred_crop.shape[1]] = pred_crop
    roi = (effective_part_crop == 1) | (effective_part_crop == 2)
    out = np.asarray(crop, dtype=np.uint8).copy()
    for index, name in enumerate(("wrist", "l_gripper", "r_gripper"), start=1):
        mask = roi & (canvas == index)
        color = PART_COLORS[name]
        out[mask] = np.clip(out[mask].astype(np.float32) * 0.35 + color * 0.65, 0, 255).astype(np.uint8)
    out[~roi] = (out[~roi].astype(np.float32) * 0.35).astype(np.uint8)
    return out


@torch.inference_mode()
def _model_visuals(model, surfaces, crop_tensor, crop_rgb, K_crop, effective_part_crop, pose, args, seed):
    device = next(model.parameters()).device
    x = crop_tensor[None].to(device=device, non_blocking=True)
    K_tensor = torch.from_numpy(K_crop)[None].to(device=device, non_blocking=True)
    with torch.amp.autocast(
        device_type=device.type,
        enabled=device.type == "cuda" and bool(args.amp),
        dtype=torch.bfloat16,
    ):
        out = model(x, K_tensor)
    query = F.avg_pool2d(out["surfemb_queries"][0].float()[None], int(args.down_sample_scale))[0]
    _, h, w = query.shape
    query_flat = query.permute(1, 2, 0).reshape(h * w, -1)

    # This visualization deliberately excludes shaft from both classification
    # and correspondence retrieval.
    part_names = ("wrist", "l_gripper", "r_gripper")
    part_scores = []
    for name in part_names:
        keys = surfaces[name].mask_keys
        part_scores.append(torch.logsumexp(query_flat @ keys.T, dim=1) - math.log(float(len(keys))))
    pred_part_ds = torch.stack(part_scores, dim=1).argmax(dim=1).reshape(h, w).cpu().numpy() + 1

    sample_y = np.arange(h, dtype=np.int64) * int(args.down_sample_scale) + (int(args.down_sample_scale) // 2)
    sample_x = np.arange(w, dtype=np.int64) * int(args.down_sample_scale) + (int(args.down_sample_scale) // 2)
    effective_ds = effective_part_crop[np.ix_(sample_y, sample_x)]
    binary_pred = np.where(pred_part_ds == 1, 2, 1)
    eval_roi = (effective_ds == 1) | (effective_ds == 2)
    part_accuracy = float((binary_pred[eval_roi] == effective_ds[eval_roi]).mean()) if eval_roi.any() else float("nan")

    transforms = fk_matrices_np(pose["rot"], pose["trans"], pose["alpha"], pose["theta_l"], pose["theta_r"])
    outputs = {}
    for group_name, label in (("wrist", 2), ("gripper", 1)):
        yx = _spatial_sample(effective_ds == label, args.points_per_part, seed + label * 1009)
        if len(yx) == 0:
            outputs[group_name] = {
                "image": crop_rgb.copy(),
                "n": 0,
                "median": float("nan"),
                "p90": float("nan"),
                "confidence": float("nan"),
            }
            continue
        flat_idx = yx[:, 0] * w + yx[:, 1]
        selected_queries = query_flat[torch.from_numpy(flat_idx).to(device=device)]
        if group_name == "wrist":
            candidate_names = ("wrist",)
        else:
            candidate_names = ("l_gripper", "r_gripper")
        candidate_keys = torch.cat([surfaces[name].keys for name in candidate_names], dim=0)
        logits = selected_queries @ candidate_keys.T
        probability = torch.softmax(logits, dim=1)
        confidence, match_idx = probability.max(dim=1)
        match_idx = match_idx.cpu().numpy()

        matched_names = []
        local_indices = []
        offset = 0
        counts = [len(surfaces[name].keys) for name in candidate_names]
        for value in match_idx:
            for name, count in zip(candidate_names, counts):
                if value < offset + count:
                    matched_names.append(name)
                    local_indices.append(int(value - offset))
                    break
                offset += count
            offset = 0

        projected = np.empty((len(yx), 2), dtype=np.float64)
        depth = np.empty((len(yx),), dtype=np.float64)
        for name in candidate_names:
            select = np.asarray([n == name for n in matched_names])
            if not select.any():
                continue
            uv, z = _project_surface_matches(
                surfaces[name],
                np.asarray(local_indices)[select],
                transforms[name],
                K_crop,
            )
            projected[select] = uv
            depth[select] = z
        query_uv = _crop_point_from_ds(yx, int(args.down_sample_scale))
        errors = np.linalg.norm(projected - query_uv, axis=1)
        errors[depth <= 0.0] = np.inf
        outputs[group_name] = {
            "image": _draw_matches(crop_rgb, query_uv, projected, errors, matched_names),
            "n": int(len(yx)),
            "median": float(np.median(errors)),
            "p90": float(np.percentile(errors, 90)),
            "confidence": float(confidence.median().item()),
        }

    outputs["part_map"] = _overlay_predicted_parts(
        crop_rgb, effective_part_crop, pred_part_ds, int(args.down_sample_scale)
    )
    outputs["part_accuracy"] = part_accuracy
    return outputs


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _make_contact_sheet(paths, output_path):
    images = [Image.open(path).convert("RGB") for path in paths]
    width = max(image.width for image in images)
    thumb_h = 470
    resized = [image.resize((width * thumb_h // image.height, thumb_h), Image.Resampling.LANCZOS) for image in images]
    sheet = Image.new("RGB", (max(image.width for image in resized), sum(image.height for image in resized)), "white")
    top = 0
    for image in resized:
        sheet.paste(image, (0, top))
        top += image.height
    sheet.save(output_path, quality=94)


def _add_case_banner(panel, text):
    image = Image.fromarray(panel)
    banner_h = 48
    out = Image.new("RGB", (image.width, image.height + banner_h), (28, 28, 28))
    out.paste(image, (0, banner_h))
    ImageDraw.Draw(out).text((14, 11), text, fill=(255, 255, 255), font=_font(21))
    return np.asarray(out)


def main(args):
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = parse_model_specs(args.model or list(DEFAULT_MODELS))
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    dataset = build_dataset(args)
    renderer = InstrumentOpenGLDepthRenderer(args.crop_size, args.crop_size, device_idx=args.egl_device)
    loaded = []
    for spec in specs:
        model, checkpoint_iter, _ = load_model(spec, device)
        surfaces = load_part_surfaces(args.surface_root, keys_per_part=args.surface_keys_per_part, seed=args.surface_seed)
        encode_surface_keys(model, surfaces, device, mask_keys_per_part=args.mask_keys_per_part)
        loaded.append((spec, model, surfaces, checkpoint_iter))

    rows = []
    case_paths = []
    for dataset_idx in args.dataset_indices:
        _, target = dataset[int(dataset_idx)]
        crop_tensor, K_crop, crop_rgb, M_crop = make_surfemb_crop(target, args)
        pose = target_pose(target)
        K_orig = target["K_orig"].cpu().numpy()
        orig_shape = tuple(int(v) for v in target["orig_size"].cpu().numpy())
        effective_orig = renderer.render_pose_mask(pose, K_orig, orig_shape)
        inst_orig = target["inst_mask_orig"].cpu().numpy() > 0
        effective_orig[~inst_orig] = 0
        effective_crop = cv2.warpAffine(
            effective_orig,
            M_crop,
            (args.crop_size, args.crop_size),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        gt_overlay = _overlay_effective_parts(crop_rgb, effective_crop)

        panel_rows = []
        for model_index, (spec, model, surfaces, checkpoint_iter) in enumerate(loaded):
            result = _model_visuals(
                model,
                surfaces,
                crop_tensor,
                crop_rgb,
                K_crop,
                effective_crop,
                pose,
                args,
                seed=int(args.seed) + int(dataset_idx) * 1009 + model_index * 1000003,
            )
            wrist = result["wrist"]
            gripper = result["gripper"]
            row = np.concatenate(
                [
                    _label_tile(gt_overlay, f"{spec['name']}: GT visible WG", "green=wrist, pink=moving gripper; shaft hidden"),
                    _label_tile(
                        result["part_map"],
                        "Predicted WG part from embeddings",
                        f"binary wrist/gripper accuracy={result['part_accuracy']:.3f}",
                    ),
                    _label_tile(
                        wrist["image"],
                        "Wrist top-1 correspondence",
                        f"N={wrist['n']} median={wrist['median']:.1f}px p90={wrist['p90']:.1f}px",
                    ),
                    _label_tile(
                        gripper["image"],
                        "Gripper top-1 correspondence",
                        f"N={gripper['n']} median={gripper['median']:.1f}px p90={gripper['p90']:.1f}px",
                    ),
                ],
                axis=1,
            )
            panel_rows.append(row)
            for group, values in (("wrist", wrist), ("gripper", gripper)):
                rows.append(
                    {
                        "model": spec["name"],
                        "checkpoint_iter": checkpoint_iter,
                        "dataset_idx": int(dataset_idx),
                        "video": str(target["video_name"]),
                        "frame_id": str(target["frame_id"]),
                        "instance_id": int(target["instance_id"].item()),
                        "group": group,
                        "n_matches": values["n"],
                        "median_reprojection_px": values["median"],
                        "p90_reprojection_px": values["p90"],
                        "median_top1_probability": values["confidence"],
                        "wrist_gripper_part_accuracy": result["part_accuracy"],
                    }
                )

        panel = np.concatenate(panel_rows, axis=0)
        panel = _add_case_banner(
            panel,
            f"dataset_idx={int(dataset_idx)}  {target['video_name']}  "
            f"frame={target['frame_id']}  instance={int(target['instance_id'].item())}",
        )
        case_name = (
            f"idx{int(dataset_idx):04d}_{target['video_name']}_"
            f"frame{target['frame_id']}_inst{int(target['instance_id'].item())}.jpg"
        )
        case_path = out_dir / case_name
        Image.fromarray(panel).save(case_path, quality=94)
        case_paths.append(case_path)
        print(f"saved {case_path}", flush=True)

    if rows:
        _write_csv(out_dir / "correspondence_summary.csv", rows)
    if case_paths:
        _make_contact_sheet(case_paths, out_dir / "all_cases_contact_sheet.jpg")
    print(f"summary={out_dir / 'correspondence_summary.csv'}", flush=True)
    print(f"contact_sheet={out_dir / 'all_cases_contact_sheet.jpg'}", flush=True)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", default=None, help="NAME=CHECKPOINT; repeat for multiple models")
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--egl_device", type=int, default=0)
    parser.add_argument("--dataset_indices", default=DEFAULT_INDICES)
    parser.add_argument("--points_per_part", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--amp", type=int, choices=[0, 1], default=1)
    parser.add_argument("--down_sample_scale", type=int, default=3)
    parser.add_argument("--surface_keys_per_part", type=int, default=4096)
    parser.add_argument("--mask_keys_per_part", type=int, default=512)
    parser.add_argument("--surface_seed", type=int, default=2026)
    parser.add_argument("--needle_dataset_root", default="/mnt/nas/share/shuojue/data/needleGrasping_videos")
    parser.add_argument("--needle_pose_root", default="/mnt/nas/share/shuojue/data/needleGrasping_results")
    parser.add_argument(
        "--dataset_cache_dir",
        default=str(ROBOPEPP_ROOT / "logs/robopepp_rarp_lnd_refinemem_eval_keypoint_trimesh/dataset_cache"),
    )
    parser.add_argument(
        "--surface_root",
        default=str(
            ROBOPEPP_ROOT
            / "assets"
            / "instrument_surface_samples_surfemb_x2.13mm_wg1over3_shafttop30mm"
        ),
    )
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--surfemb_crop_scale", type=float, default=1.2)
    parser.add_argument("--bbox_padding_frac", type=float, default=0.12)
    parser.add_argument("--train_ratio", type=float, default=0.95)
    parser.add_argument("--min_dice_shaft", type=float, default=0.8)
    parser.add_argument("--min_dice_wrist", type=float, default=0.6)
    parser.add_argument("--min_dice_gripper", type=float, default=0.6)
    parser.add_argument("--canonicalize_pose_symmetry", type=int, choices=[0, 1], default=1)
    parser.add_argument("--canonical_eps", type=float, default=0.08)
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    parsed.dataset_indices = [int(value) for value in parsed.dataset_indices.split(",") if value.strip()]
    main(parsed)
