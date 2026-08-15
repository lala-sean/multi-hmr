#!/usr/bin/env python3
import argparse
import gc
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

import visualize_surfemb_wrist_gripper_pose_fit as surfvis
import compare_crop_hcce_robopepp_rarp as hcce_eval
from surfemb_articulated_pose import encode_surface_keys, load_part_surfaces


ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "logs" / "surfemb_vs_hcce_wrist_gripper_mesh"
DEFAULT_HCCE = (
    ROOT
    / "logs"
    / "hcce_crop224_rarp_lnd_refinemem_bs56_gpu0234"
    / "checkpoints"
    / "last.pt"
)
DEFAULT_INDICES = "249,874,749,433,422"
METHODS = (
    ("hcce_pred_mask", "HCCE W/G fit (pred mask)"),
    ("hcce_sam_roi", "HCCE W/G fit (SAM ROI)"),
    ("resnet", "ResNet SurfEmb W/G fit"),
    ("dino_multihead", "DINO SurfEmb W/G fit"),
)
CATEGORIES = {
    249: "success",
    874: "success",
    749: "model-contrast",
    433: "failure",
    422: "failure",
}


def pose_row(dataset_idx, target, method, checkpoint_iter, pred=None, gt=None, status="ok", extra=None):
    row = {
        "dataset_idx": int(dataset_idx),
        "video": str(target["video_name"]),
        "frame_id": str(target["frame_id"]),
        "instance_id": int(target["instance_id"].item()),
        "method": method,
        "checkpoint_iter": int(checkpoint_iter),
        "status": status,
    }
    if pred is not None and gt is not None:
        row.update(surfvis._pose_metrics(pred, gt))
    if extra:
        for key, value in extra.items():
            if isinstance(value, (bool, int, float, str)):
                row[key] = value
    return row


def run_surfemb(dataset, args, device, predictions, rows):
    specs = surfvis.parse_model_specs(args.model or list(surfvis.DEFAULT_MODELS))
    for model_index, spec in enumerate(specs):
        model, checkpoint_iter, _ = surfvis.load_model(spec, device)
        surfaces = load_part_surfaces(
            args.surface_root,
            keys_per_part=int(args.surface_keys_per_part),
            seed=int(args.surface_seed),
        )
        encode_surface_keys(model, surfaces, device, mask_keys_per_part=int(args.mask_keys_per_part))
        for dataset_idx in args.candidate_indices:
            _, target = dataset[int(dataset_idx)]
            image, K_crop, _, M_crop = surfvis.make_surfemb_crop(target, args)
            part_orig = target["part_mask_orig"].cpu().numpy().astype(np.uint8)
            part_crop = cv2.warpAffine(
                part_orig,
                M_crop,
                (int(args.crop_size), int(args.crop_size)),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            gt = surfvis.target_pose(target)
            try:
                pred, diagnostics = surfvis.fit_wrist_gripper_pose(
                    model,
                    surfaces,
                    image,
                    K_crop,
                    part_crop,
                    args,
                    seed=int(args.seed) + int(dataset_idx) * 1009 + model_index * 1000003,
                )
                predictions[(int(dataset_idx), spec["name"])] = pred
                row = pose_row(dataset_idx, target, spec["name"], checkpoint_iter, pred, gt, extra=diagnostics)
            except Exception as exc:
                row = pose_row(
                    dataset_idx,
                    target,
                    spec["name"],
                    checkpoint_iter,
                    status=f"{type(exc).__name__}: {exc}",
                )
            rows.append(row)
            print(
                f"idx={dataset_idx} method={spec['name']} status={row['status']} "
                f"t={row.get('wrist_trans_err_mm', float('nan')):.2f} "
                f"r={row.get('wrist_rot_err_deg', float('nan')):.2f}",
                flush=True,
            )
        del model, surfaces
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def run_hcce(dataset, args, device, predictions, rows):
    model, model_meta = hcce_eval.load_hcce_model(args.hcce_checkpoint, device)
    checkpoint = torch.load(args.hcce_checkpoint, map_location="cpu", weights_only=False)
    checkpoint_iter = int(checkpoint.get("iter", -1))
    cad = hcce_eval.InstrumentCAD(hcce_eval.CAD_ROOT)

    for dataset_idx in args.candidate_indices:
        image, target = dataset[int(dataset_idx)]
        target_like = hcce_eval.target_like_from_needle(target)
        K_crop_t = torch.from_numpy(target_like["K_crop"])[None].to(device, non_blocking=True)
        x = image[None].to(device, non_blocking=True)
        with torch.inference_mode(), torch.amp.autocast(
            device_type="cuda",
            enabled=device.type == "cuda" and bool(args.amp),
            dtype=torch.bfloat16,
        ):
            output = model(x, K_crop_t)
        gt = surfvis.target_pose(target)
        for source, method in (("pred", "hcce_pred_mask"), ("gt", "hcce_sam_roi")):
            args.fit_seg_source = source
            rng = np.random.default_rng(int(args.hcce_seed) + int(dataset_idx) * 1009 + (source == "gt") * 1000003)
            try:
                pred, diagnostics = hcce_eval.fit_pose_from_hcce(
                    output,
                    cad,
                    model_meta,
                    target_like,
                    args,
                    rng,
                )
                pred = surfvis.canonicalize_prediction(pred, args.canonical_eps)
                predictions[(int(dataset_idx), method)] = pred
                row = pose_row(dataset_idx, target, method, checkpoint_iter, pred, gt, extra=diagnostics)
            except Exception as exc:
                row = pose_row(
                    dataset_idx,
                    target,
                    method,
                    checkpoint_iter,
                    status=f"{type(exc).__name__}: {exc}",
                )
            rows.append(row)
            print(
                f"idx={dataset_idx} method={method} status={row['status']} "
                f"t={row.get('wrist_trans_err_mm', float('nan')):.2f} "
                f"r={row.get('wrist_rot_err_deg', float('nan')):.2f}",
                flush=True,
            )
        del output, x, K_crop_t

    del model, cad
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def failed_tile(rgb, title, status, panel_size):
    tile = surfvis._pad_square(rgb, panel_size)
    cv2.putText(tile, "FIT FAILED", (18, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 55, 55), 2, cv2.LINE_AA)
    return surfvis._title(tile, title, status[:58])


def render_comparison(dataset, args, predictions, rows):
    renderer = surfvis.WristGripperTrimeshRenderer(torch.device(args.device))
    by_key = {(int(row["dataset_idx"]), row["method"]): row for row in rows}
    panel_paths = []
    for dataset_idx in args.candidate_indices:
        _, target = dataset[int(dataset_idx)]
        rgb = target["orig_rgb"].cpu().numpy().astype(np.uint8)
        part = target["part_mask_orig"].cpu().numpy().astype(np.uint8)
        K_orig = target["K_orig"].cpu().numpy().astype(np.float32)
        gt = surfvis.target_pose(target)
        gt_overlay, gt_support = renderer.overlay(rgb, gt, K_orig, alpha=args.overlay_alpha)
        tiles = [
            surfvis._title(surfvis._pad_square(rgb, args.panel_size), "RGB"),
            surfvis._title(
                surfvis._pad_square(gt_overlay, args.panel_size),
                "GT wrist+gripper trimesh",
                f"SAM W/G IoU={surfvis._mask_iou(gt_support, part):.3f}",
            ),
        ]
        for method, title in METHODS:
            row = by_key[(int(dataset_idx), method)]
            pred = predictions.get((int(dataset_idx), method))
            if pred is None:
                tiles.append(failed_tile(rgb, title, row["status"], args.panel_size))
                continue
            overlay, support = renderer.overlay(rgb, pred, K_orig, alpha=args.overlay_alpha)
            grip_error = np.mean([row["theta_l_err_deg"], row["theta_r_err_deg"]])
            row["render_wg_iou"] = surfvis._mask_iou(support, part)
            subtitle = (
                f"t={row['wrist_trans_err_mm']:.1f}mm r={row['wrist_rot_err_deg']:.1f}deg "
                f"grip={grip_error:.1f}deg IoU={row['render_wg_iou']:.3f}"
            )
            tiles.append(surfvis._title(surfvis._pad_square(overlay, args.panel_size), title, subtitle))

        panel = np.concatenate(tiles, axis=1)
        banner_h = 46
        image = Image.fromarray(panel)
        canvas = Image.new("RGB", (image.width, image.height + banner_h), (28, 28, 28))
        canvas.paste(image, (0, banner_h))
        category = CATEGORIES.get(int(dataset_idx), "representative")
        ImageDraw.Draw(canvas).text(
            (12, 10),
            f"{category}  dataset_idx={dataset_idx}  {target['video_name']}  "
            f"frame={target['frame_id']}  instance={int(target['instance_id'].item())}",
            fill=(255, 255, 255),
            font=surfvis._font(20),
        )
        path = Path(args.out_dir) / f"{category}_idx{int(dataset_idx):04d}_hcce_vs_surfemb.jpg"
        canvas.save(path, quality=94)
        panel_paths.append(path)
        print(f"saved {path}", flush=True)

    surfvis._write_csv(Path(args.out_dir) / "comparison_metrics.csv", rows)
    surfvis._contact_sheet(panel_paths, Path(args.out_dir) / "hcce_vs_surfemb_contact_sheet.jpg")


def build_parser():
    parser = surfvis.build_parser()
    parser.set_defaults(
        out_dir=str(DEFAULT_OUT),
        candidate_indices=DEFAULT_INDICES,
        num_examples=5,
    )
    parser.add_argument("--hcce_checkpoint", default=str(DEFAULT_HCCE))
    parser.add_argument("--hcce_seed", type=int, default=20260802)
    parser.add_argument("--inst_thresh", type=float, default=0.5)
    parser.add_argument("--hcce_bits", type=int, default=8)
    parser.add_argument("--hcce_coord_min", type=float, default=-1.0)
    parser.add_argument("--hcce_coord_max", type=float, default=1.0)
    parser.add_argument("--hcce_bit_thresh", type=float, default=0.5)
    parser.add_argument("--hcce_axis_scale", default=None)
    parser.add_argument("--max_points_per_part", type=int, default=1200)
    parser.add_argument("--point_select", choices=["random"], default="random")
    parser.add_argument("--fit_seg_source", choices=["pred", "gt"], default="pred")
    parser.add_argument("--surface_snap_method", choices=["surface", "vertex"], default="surface")
    parser.add_argument("--surface_k_faces", type=int, default=0)
    parser.add_argument("--shaft_raw_x_min", type=float, default=-0.5)
    parser.add_argument("--min_wrist_points", type=int, default=12)
    parser.add_argument("--min_total_points", type=int, default=24)
    parser.add_argument("--min_pnp_inliers", type=int, default=8)
    parser.add_argument("--pnp_iters", type=int, default=300)
    parser.add_argument("--pnp_reproj_error", type=float, default=8.0)
    parser.add_argument("--pnp_confidence", type=float, default=0.99)
    parser.add_argument("--optim_strategy", choices=["single"], default="single")
    parser.add_argument("--optim_parts", choices=["wrist_gripper"], default="wrist_gripper")
    parser.add_argument("--freeze_wrist_after_pnp", type=int, choices=[0], default=0)
    parser.add_argument("--optim_loss", choices=["soft_l1"], default="soft_l1")
    parser.add_argument("--optim_f_scale", type=float, default=8.0)
    parser.add_argument("--optim_max_nfev", type=int, default=200)
    parser.add_argument("--min_depth", type=float, default=1e-4)
    parser.add_argument("--behind_camera_penalty", type=float, default=1e4)
    return parser


def main(args):
    args.out_dir = str(Path(args.out_dir).resolve())
    args.hcce_checkpoint = str(Path(args.hcce_checkpoint).resolve())
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dataset = surfvis.build_dataset(args)
    predictions = {}
    rows = []
    run_surfemb(dataset, args, device, predictions, rows)
    run_hcce(dataset, args, device, predictions, rows)
    render_comparison(dataset, args, predictions, rows)
    print(f"metrics={Path(args.out_dir) / 'comparison_metrics.csv'}", flush=True)
    print(f"contact_sheet={Path(args.out_dir) / 'hcce_vs_surfemb_contact_sheet.jpg'}", flush=True)


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    parsed.candidate_indices = [int(value) for value in parsed.candidate_indices.split(",") if value.strip()]
    if parsed.model is None:
        parsed.model = list(surfvis.DEFAULT_MODELS)
    main(parsed)
