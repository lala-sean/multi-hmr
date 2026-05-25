import argparse
import csv
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from datasets.surgical_instruments import collate_fn_instrument
from datasets.RarpInstanceDataset import RARPInstanceDataset
from estimate_rarp_cse_articulate_pose import (
    CAD_ROOT,
    DATASETS,
    PART_LABELS,
    REPO_ROOT,
    Correspondences,
    InstrumentCAD,
    _rotation_error_deg,
    compute_segmentation_metrics,
    draw_projected_mesh,
    image_geometry,
    indices_for_video,
    load_gt_pose,
    match_gt_to_predictions,
    optimize_pose,
    quarter_pixels_to_original,
    sample_mask_pixels,
    save_npz,
    select_videos,
    solve_wrist_pnp,
)
from multi_instrument.hcce_codec import hcce_decode_torch
from multi_instrument.multi_instrument_hcce_pose_dpt import MultiInstrumentHCCEPoseDPT


DATASETS_HCCE = {
    **DATASETS,
    "suturePulling": {
        "dataset_root": "/mnt/nas/share/shuojue/data/suturePulling_videos",
        "pose_root": None,
        "v2_force": True,
    },
}


def build_dataset_hcce(args):
    cfg = DATASETS_HCCE[args.dataset_name]
    dataset = RARPInstanceDataset(
        split="test",
        training=False,
        img_size=args.img_size,
        dataset_root=cfg["dataset_root"],
        pose_root=cfg["pose_root"],
        min_dice=[args.min_dice_shaft, args.min_dice_wrist, args.min_dice_gripper],
        train_ratio=args.train_ratio,
        subsample=1,
        v2_force=bool(cfg.get("v2_force", False)),
        cse_coord_root=None,
        render_on_the_fly=False,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"Dataset is empty: {args.dataset_name} test split")
    return dataset


def build_model_args(ckpt_args):
    defaults = dict(
        img_size=630,
        backbone="dinov2_vits14",
        xat_dim=512,
        xat_depth=4,
        xat_heads=16,
        xat_dim_head=32,
        xat_mlp_dim=2048,
        mask_dim=256,
        num_parts=4,
        hcce_feat_dim=256,
        hcce_bits=8,
        action_dim=3,
        use_pose_heads=0,
        pose_head_iter=4,
        pose_head_dropout=0.3,
    )
    coord_defaults = dict(hcce_coord_min=-1.0, hcce_coord_max=1.0)
    if ckpt_args is not None:
        if hasattr(ckpt_args, "__dict__"):
            src = vars(ckpt_args)
        elif isinstance(ckpt_args, dict):
            src = ckpt_args
        else:
            raise TypeError(f"Unsupported checkpoint args type: {type(ckpt_args)}")
        for key in defaults:
            if key in src:
                defaults[key] = src[key]
        for key in coord_defaults:
            if key in src:
                coord_defaults[key] = src[key]
    defaults["pretrained_backbone"] = False
    return defaults, coord_defaults


def load_model(checkpoint_path: Path, device: torch.device):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" not in ckpt:
        raise KeyError(f"Checkpoint missing model_state_dict: {checkpoint_path}")
    model_args, coord_args = build_model_args(ckpt.get("args"))
    model = MultiInstrumentHCCEPoseDPT(**model_args).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model, model_args, coord_args


def decode_hcce_logits(hcce_logits, bits=8, coord_min=-1.0, coord_max=1.0, threshold=0.5):
    """
    Follow the original HCCEPose inference path:
    sigmoid-threshold HCCE channels, hierarchical-decode to uint code, then scale.
    """
    if hcce_logits.ndim != 3:
        raise ValueError(f"Expected [C,H,W] HCCE logits, got {tuple(hcce_logits.shape)}")
    expected_channels = 3 * int(bits)
    if int(hcce_logits.shape[0]) != expected_channels:
        raise ValueError(
            f"Expected {expected_channels} HCCE channels for bits={bits}, "
            f"got {int(hcce_logits.shape[0])}"
        )
    bit_img = (torch.sigmoid(hcce_logits.detach()) > float(threshold)).float()
    bit_img = bit_img.permute(1, 2, 0).unsqueeze(0).contiguous()
    code = hcce_decode_torch(bit_img)[0] / 255.0
    xyz = code * float(coord_max - coord_min) + float(coord_min)
    return xyz


def decoded_coord_vis(xyz, inst_mask=None, coord_min=-1.0, coord_max=1.0):
    vis = ((xyz - float(coord_min)) / float(coord_max - coord_min)).clip(0.0, 1.0)
    if inst_mask is not None:
        vis = vis.copy()
        vis[~inst_mask.astype(bool)] *= 0.15
    return (vis * 255.0).astype(np.uint8)


def build_correspondences(pred, cad: InstrumentCAD, geom, args, rng):
    _, _, scale, pad_x, pad_y = geom
    inst_prob = torch.sigmoid(pred["inst_mask_logits"]).detach().cpu().numpy()
    part_label = torch.argmax(pred["part_mask_logits"], dim=0).detach().cpu().numpy()
    xyz = decode_hcce_logits(
        pred["hcce_logits"],
        bits=args.hcce_bits,
        coord_min=args.hcce_coord_min,
        coord_max=args.hcce_coord_max,
        threshold=args.hcce_bit_thresh,
    ).detach().cpu().numpy()
    q_size = inst_prob.shape[0]

    inst_mask = inst_prob >= args.inst_thresh
    all_uv = []
    all_points = []
    all_part_names = []
    wrist_uv = []
    wrist_points = []

    part_specs = [
        ("shaft", PART_LABELS["shaft"], args.max_points_per_part),
        ("wrist", PART_LABELS["wrist"], args.max_points_per_part),
        ("gripper", PART_LABELS["gripper"], args.max_points_per_part),
    ]

    for part_name, label, max_points in part_specs:
        part_mask = inst_mask & (part_label == label)
        xs, ys = sample_mask_pixels(part_mask, max_points, rng)
        if len(xs) == 0:
            continue

        uv = quarter_pixels_to_original(
            xs, ys, q_size, args.img_size, scale, pad_x, pad_y
        )
        valid_uv = (
            (uv[:, 0] >= 0.0)
            & (uv[:, 0] < geom[0].shape[1])
            & (uv[:, 1] >= 0.0)
            & (uv[:, 1] < geom[0].shape[0])
        )
        uv = uv[valid_uv]
        xs = xs[valid_uv]
        ys = ys[valid_uv]
        if len(xs) == 0:
            continue

        xyz_norm = xyz[ys, xs].astype(np.float64)
        finite = np.isfinite(xyz_norm).all(axis=1)
        uv = uv[finite]
        xyz_norm = xyz_norm[finite]
        if len(xyz_norm) == 0:
            continue

        if part_name == "gripper":
            points_part, names = cad.nearest_gripper_points(xyz_norm)
        else:
            points_part = cad.nearest_part_points(part_name, xyz_norm)
            names = np.full((len(points_part),), part_name, dtype=object)

        all_uv.append(uv)
        all_points.append(points_part)
        all_part_names.append(names)
        if part_name == "wrist":
            wrist_uv.append(uv)
            wrist_points.append(points_part)

    if not wrist_uv:
        raise RuntimeError("No predicted wrist HCCE pixels for this instance")

    wrist_uv = np.concatenate(wrist_uv, axis=0)
    wrist_points = np.concatenate(wrist_points, axis=0)
    if len(wrist_uv) < args.min_wrist_points:
        raise RuntimeError(
            f"Not enough wrist correspondences: {len(wrist_uv)} < {args.min_wrist_points}"
        )

    if not all_uv:
        raise RuntimeError("No HCCE correspondences for this instance")

    corr = Correspondences(
        uv=np.concatenate(all_uv, axis=0).astype(np.float64),
        points_part=np.concatenate(all_points, axis=0).astype(np.float64),
        part_names=np.concatenate(all_part_names, axis=0),
    )
    if len(corr) < args.min_total_points:
        raise RuntimeError(
            f"Not enough total correspondences: {len(corr)} < {args.min_total_points}"
        )
    return corr, wrist_uv.astype(np.float64), wrist_points.astype(np.float64), xyz, inst_mask


def write_part_pose_rows(part_writer, base_row, cad, opt_params):
    transforms = cad.fk(
        opt_params[:3],
        opt_params[3:6],
        opt_params[6],
        opt_params[7],
        opt_params[8],
    )
    for part_name in ("shaft", "wrist", "l_gripper", "r_gripper"):
        T = transforms[part_name]
        row = {
            "dataset": base_row["dataset"],
            "video": base_row["video"],
            "frame_id": base_row["frame_id"],
            "instance_id": base_row["instance_id"],
            "sample_idx": base_row["sample_idx"],
            "pred_idx": base_row["pred_idx"],
            "gt_idx": base_row["gt_idx"],
            "part_name": part_name,
        }
        row.update({f"T{i}{j}": float(T[i, j]) for i in range(4) for j in range(4)})
        part_writer.writerow(row)
    return transforms


def blank_segmentation_metrics():
    return {
        "inst_iou": float("nan"),
        "part_iou_mean": float("nan"),
        "part_iou_gripper": float("nan"),
        "part_iou_wrist": float("nan"),
        "part_iou_shaft": float("nan"),
        "pred_inst_area": float("nan"),
        "gt_inst_area": float("nan"),
    }


def match_predictions_for_visualization(predictions, gt_wrist_centers, match_max_dist):
    if len(predictions) == 0:
        return []
    if len(gt_wrist_centers) == 0:
        return [(None, pred_idx, float("nan")) for pred_idx in range(len(predictions))]

    pred_locs = np.stack([
        pred["loc"].detach().cpu().numpy().astype(np.float64)
        for pred in predictions
    ])
    used = set()
    matches = []
    for gt_idx, gt_loc in enumerate(gt_wrist_centers):
        dists = np.linalg.norm(pred_locs - gt_loc[None], axis=1)
        for pred_idx in np.argsort(dists):
            pred_idx = int(pred_idx)
            if pred_idx in used:
                continue
            if float(dists[pred_idx]) <= float(match_max_dist):
                used.add(pred_idx)
                matches.append((gt_idx, pred_idx, float(dists[pred_idx])))
            break

    for pred_idx in range(len(predictions)):
        if pred_idx not in used:
            matches.append((None, pred_idx, float("nan")))
    return matches


def save_failed_debug(dataset, sample_idx, model, device, args, reason):
    video_name, frame_id, _ = dataset.samples[sample_idx]
    fail_dir = args.output_dir / "failures" / f"{video_name}_{frame_id}"
    fail_dir.mkdir(parents=True, exist_ok=True)

    rgb, _, _, _, _ = image_geometry(dataset, video_name, frame_id)
    Image.fromarray(rgb).save(fail_dir / "original.png")
    (fail_dir / "reason.txt").write_text(str(reason), encoding="utf-8")

    img_array, annot = dataset[sample_idx]
    batch_img, _ = collate_fn_instrument([(img_array, annot)])
    with torch.no_grad():
        preds = model(
            batch_img.to(device),
            is_training=False,
            det_thresh=args.det_thresh,
            nms_kernel_size=args.nms_kernel_size,
        )
    if not isinstance(preds, list) or len(preds) == 0:
        return

    for pred_idx, pred in enumerate(preds):
        inst_mask = (
            torch.sigmoid(pred["inst_mask_logits"]).detach().cpu().numpy()
            >= args.inst_thresh
        )
        xyz = decode_hcce_logits(
            pred["hcce_logits"],
            bits=args.hcce_bits,
            coord_min=args.hcce_coord_min,
            coord_max=args.hcce_coord_max,
            threshold=args.hcce_bit_thresh,
        ).detach().cpu().numpy()
        coord = decoded_coord_vis(
            xyz,
            inst_mask=inst_mask,
            coord_min=args.hcce_coord_min,
            coord_max=args.hcce_coord_max,
        )
        coord_full = cv2.resize(
            coord,
            (rgb.shape[1], rgb.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        preview = np.concatenate(
            [rgb, np.full((rgb.shape[0], 6, 3), 255, dtype=np.uint8), coord_full],
            axis=1,
        )
        Image.fromarray(coord).save(fail_dir / f"pred{pred_idx}_decoded_xyz.png")
        Image.fromarray(preview).save(fail_dir / f"pred{pred_idx}_original_hcce.png")


def process_sample(dataset, sample_idx, model, cad, device, args, rng, writer, part_writer, do_vis):
    video_name, frame_id, instances_list = dataset.samples[sample_idx]
    img_array, annot = dataset[sample_idx]
    no_pose_gt = bool(args.no_pose_gt) or dataset.pose_root is None
    if len(annot["instruments"]) != len(instances_list) and not no_pose_gt:
        raise RuntimeError(
            f"Instrument count changed during __getitem__ for {video_name}/{frame_id}: "
            f"{len(annot['instruments'])} != {len(instances_list)}"
        )

    batch_img, y = collate_fn_instrument([(img_array, annot)])
    with torch.no_grad():
        preds = model(
            batch_img.to(device),
            is_training=False,
            det_thresh=args.det_thresh,
            nms_kernel_size=args.nms_kernel_size,
        )
    if not isinstance(preds, list) or len(preds) == 0:
        raise RuntimeError(f"No predicted instances for {video_name}/{frame_id}")

    valid_gt = y["valid_instruments"][0].bool().numpy()
    gt_wrist_centers = y["wrist_centers"][0][valid_gt].numpy().astype(np.float64)
    if no_pose_gt:
        matches = match_predictions_for_visualization(
            preds, gt_wrist_centers, args.match_max_dist
        )
    else:
        matches = match_gt_to_predictions(preds, gt_wrist_centers, args.match_max_dist)

    geom = image_geometry(dataset, video_name, frame_id)
    rgb, K = geom[0], geom[1]

    for gt_idx, pred_idx, match_dist in matches:
        if gt_idx is None:
            instance_id = int(pred_idx) + 1 if len(instances_list) == 0 else 1000 + int(pred_idx)
        elif no_pose_gt:
            instance_id = int(gt_idx) + 1
        else:
            instance_id = int(instances_list[gt_idx][0])
        pred = preds[pred_idx]
        seg_metrics = (
            blank_segmentation_metrics()
            if gt_idx is None
            else compute_segmentation_metrics(pred, y, gt_idx, args)
        )
        corr, wrist_uv, wrist_points, xyz, inst_mask = build_correspondences(
            pred, cad, geom, args, rng
        )
        rvec0, trans0, pnp_inliers = solve_wrist_pnp(wrist_uv, wrist_points, K, args)
        (
            opt_params,
            reproj_rmse,
            opt_nfev,
            reproj_rmse_wrist_gripper,
            reproj_rmse_shaft,
        ) = optimize_pose(cad, corr, rvec0, trans0, K, args)

        R_pred, _ = cv2.Rodrigues(opt_params[:3].reshape(3, 1))
        if no_pose_gt:
            gt_pose = None
            rot_err_deg = float("nan")
            trans_err_m = float("nan")
            alpha_err = float("nan")
            theta_l_err = float("nan")
            theta_r_err = float("nan")
        else:
            gt_pose = load_gt_pose(dataset.pose_root, video_name, instance_id, frame_id)
            rot_err_deg = _rotation_error_deg(R_pred, gt_pose["rot_mat"])
            trans_err_m = float(np.linalg.norm(opt_params[3:6] - gt_pose["trans"]))
            alpha_err = abs(float(opt_params[6]) - gt_pose["alpha"])
            theta_l_err = abs(float(opt_params[7]) - gt_pose["theta_l"])
            theta_r_err = abs(float(opt_params[8]) - gt_pose["theta_r"])

        row = {
            "dataset": args.dataset_name,
            "video": video_name,
            "frame_id": frame_id,
            "instance_id": instance_id,
            "sample_idx": sample_idx,
            "pred_idx": pred_idx,
            "gt_idx": -1 if gt_idx is None else gt_idx,
            "match_dist_px": match_dist,
            **seg_metrics,
            "corr_count": len(corr),
            "wrist_corr_count": len(wrist_uv),
            "pnp_inliers": pnp_inliers,
            "optim_nfev": opt_nfev,
            "reproj_rmse_px": reproj_rmse,
            "reproj_rmse_wrist_gripper_px": reproj_rmse_wrist_gripper,
            "reproj_rmse_shaft_px": reproj_rmse_shaft,
            "rot_err_deg": rot_err_deg,
            "trans_err_m": trans_err_m,
            "alpha_err_rad": alpha_err,
            "theta_l_err_rad": theta_l_err,
            "theta_r_err_rad": theta_r_err,
            "pred_rvec_x": opt_params[0],
            "pred_rvec_y": opt_params[1],
            "pred_rvec_z": opt_params[2],
            "pred_tx": opt_params[3],
            "pred_ty": opt_params[4],
            "pred_tz": opt_params[5],
            "pred_alpha": opt_params[6],
            "pred_theta_l": opt_params[7],
            "pred_theta_r": opt_params[8],
        }
        writer.writerow(row)
        part_transforms = write_part_pose_rows(part_writer, row, cad, opt_params)

        part_names = np.array(["shaft", "wrist", "l_gripper", "r_gripper"])
        part_mats = np.stack([part_transforms[name] for name in part_names], axis=0)
        save_npz(
            args.output_dir / "pred_poses" / f"{video_name}_{frame_id}_inst{instance_id}.npz",
            pred_params=opt_params,
            pnp_rvec=rvec0,
            pnp_trans=trans0,
            part_names=part_names,
            part_transforms=part_mats,
            gt_rot_mat=gt_pose["rot_mat"] if gt_pose is not None else np.full((3, 3), np.nan),
            gt_trans=gt_pose["trans"] if gt_pose is not None else np.full((3,), np.nan),
            gt_joints=(
                np.array([gt_pose["alpha"], gt_pose["theta_l"], gt_pose["theta_r"]])
                if gt_pose is not None
                else np.full((3,), np.nan)
            ),
            correspondences_uv=corr.uv,
            correspondences_points_part=corr.points_part,
            correspondences_part_names=corr.part_names.astype(str),
            wrist_uv=wrist_uv,
            wrist_points=wrist_points,
            decoded_hcce_xyz=xyz,
        )

        if do_vis:
            Image.fromarray(draw_projected_mesh(rgb, cad, opt_params, K, args)).save(
                args.output_dir / "vis" / f"{video_name}_{frame_id}_inst{instance_id}.png"
            )
            Image.fromarray(
                decoded_coord_vis(
                    xyz,
                    inst_mask=inst_mask,
                    coord_min=args.hcce_coord_min,
                    coord_max=args.hcce_coord_max,
                )
            ).save(
                args.output_dir
                / "coord_vis"
                / f"{video_name}_{frame_id}_inst{instance_id}_decoded_xyz.png"
            )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate articulate instrument pose from predicted HCCE coordinates."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("logs/instrument_rarp_hcce_pose_dpt_rarp50_2gpu_bs6_devicefix/checkpoints/best.pt"),
    )
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument(
        "--dataset_name",
        choices=sorted(DATASETS_HCCE.keys()),
        default="needlePuncture",
    )
    parser.add_argument("--video_select", choices=["random"], default="random")
    parser.add_argument("--video_name", type=str, default=None)
    parser.add_argument("--num_videos", type=int, default=1)
    parser.add_argument(
        "--frame_select",
        choices=["first", "random", "even"],
        default="first",
    )
    parser.add_argument("--frames_per_video", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_frames", type=int, default=-1)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--img_size", type=int, default=630)
    parser.add_argument("--train_ratio", type=float, default=0.95)
    parser.add_argument("--min_dice_shaft", type=float, default=0.8)
    parser.add_argument("--min_dice_wrist", type=float, default=0.6)
    parser.add_argument("--min_dice_gripper", type=float, default=0.6)
    parser.add_argument("--det_thresh", type=float, default=0.3)
    parser.add_argument("--nms_kernel_size", type=int, default=3)
    parser.add_argument("--match_max_dist", type=float, default=80.0)

    parser.add_argument("--hcce_bits", type=int, default=None)
    parser.add_argument("--hcce_coord_min", type=float, default=None)
    parser.add_argument("--hcce_coord_max", type=float, default=None)
    parser.add_argument("--hcce_bit_thresh", type=float, default=0.5)
    parser.add_argument("--inst_thresh", type=float, default=0.5)
    parser.add_argument("--max_points_per_part", type=int, default=1200)
    parser.add_argument("--min_wrist_points", type=int, default=12)
    parser.add_argument("--min_total_points", type=int, default=24)
    parser.add_argument("--min_shaft_points", type=int, default=24)
    parser.add_argument("--min_pnp_inliers", type=int, default=8)
    parser.add_argument("--pnp_iters", type=int, default=300)
    parser.add_argument("--pnp_reproj_error", type=float, default=8.0)
    parser.add_argument("--pnp_confidence", type=float, default=0.99)
    parser.add_argument(
        "--freeze_wrist_after_pnp",
        type=int,
        default=0,
        choices=[0, 1],
    )
    parser.add_argument(
        "--optim_strategy",
        choices=["decoupled", "single"],
        default="decoupled",
    )
    parser.add_argument(
        "--optim_parts",
        choices=["wrist_gripper", "all"],
        default="wrist_gripper",
    )
    parser.add_argument("--optim_loss", choices=["linear", "soft_l1", "huber", "cauchy", "arctan"], default="soft_l1")
    parser.add_argument("--optim_f_scale", type=float, default=8.0)
    parser.add_argument("--optim_max_nfev", type=int, default=200)
    parser.add_argument("--min_depth", type=float, default=1e-4)

    parser.add_argument("--vis_every", type=int, default=1)
    parser.add_argument("--skip_failed", type=int, default=0, choices=[0, 1])
    parser.add_argument(
        "--no_pose_gt",
        type=int,
        default=0,
        choices=[0, 1],
        help="Skip pose GT loading/errors. Automatically enabled when dataset pose_root is None.",
    )
    parser.add_argument("--line_width", type=int, default=1)
    parser.add_argument("--draw_margin", type=int, default=64)
    parser.add_argument("--overlay_alpha", type=float, default=0.85)
    parser.add_argument(
        "--mesh_render_backend",
        choices=["pyrender", "software"],
        default="pyrender",
    )
    parser.add_argument("--pyrender_znear", type=float, default=0.001)
    parser.add_argument("--pyrender_zfar", type=float, default=0.6)
    parser.add_argument("--pyrender_light_intensity", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = REPO_ROOT / "debug_vis" / "hcce_pose_single_video"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "pred_poses").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "vis").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "coord_vis").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "failures").mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    device = torch.device(args.device)
    model, model_args, coord_args = load_model(args.checkpoint, device)
    if int(model_args["img_size"]) != int(args.img_size):
        raise ValueError(
            f"Script --img_size={args.img_size} does not match checkpoint img_size={model_args['img_size']}"
        )
    args.hcce_bits = int(model_args["hcce_bits"] if args.hcce_bits is None else args.hcce_bits)
    args.hcce_coord_min = float(
        coord_args["hcce_coord_min"] if args.hcce_coord_min is None else args.hcce_coord_min
    )
    args.hcce_coord_max = float(
        coord_args["hcce_coord_max"] if args.hcce_coord_max is None else args.hcce_coord_max
    )

    cad = InstrumentCAD(CAD_ROOT)
    dataset = build_dataset_hcce(args)
    args.no_pose_gt = int(bool(args.no_pose_gt) or dataset.pose_root is None)
    video_names = select_videos(dataset, args)
    frames_per_video = args.frames_per_video if args.frames_per_video > 0 else args.max_frames
    sample_indices = []
    for video_name in video_names:
        sample_indices.extend(
            indices_for_video(
                dataset,
                video_name,
                frames_per_video,
                frame_select=args.frame_select,
                seed=args.seed,
            )
        )

    selected_path = args.output_dir / "selected_video.txt"
    selected_path.write_text(
        f"dataset={args.dataset_name}\n"
        f"videos={','.join(video_names)}\n"
        f"seed={args.seed}\n"
        f"frame_select={args.frame_select}\n"
        f"frames_per_video={frames_per_video}\n"
        f"num_frames={len(sample_indices)}\n"
        f"checkpoint={args.checkpoint}\n"
        f"no_pose_gt={args.no_pose_gt}\n"
        f"hcce_bits={args.hcce_bits}\n"
        f"hcce_coord_min={args.hcce_coord_min}\n"
        f"hcce_coord_max={args.hcce_coord_max}\n"
        f"hcce_bit_thresh={args.hcce_bit_thresh}\n",
        encoding="utf-8",
    )

    csv_path = args.output_dir / "pred_poses.csv"
    fieldnames = [
        "dataset", "video", "frame_id", "instance_id", "sample_idx",
        "pred_idx", "gt_idx", "match_dist_px", "corr_count",
        "inst_iou", "part_iou_mean", "part_iou_gripper",
        "part_iou_wrist", "part_iou_shaft", "pred_inst_area",
        "gt_inst_area",
        "wrist_corr_count", "pnp_inliers", "optim_nfev",
        "reproj_rmse_px", "reproj_rmse_wrist_gripper_px",
        "reproj_rmse_shaft_px", "rot_err_deg", "trans_err_m",
        "alpha_err_rad", "theta_l_err_rad", "theta_r_err_rad",
        "pred_rvec_x", "pred_rvec_y", "pred_rvec_z",
        "pred_tx", "pred_ty", "pred_tz",
        "pred_alpha", "pred_theta_l", "pred_theta_r",
    ]
    part_csv_path = args.output_dir / "part_poses.csv"
    part_fieldnames = [
        "dataset", "video", "frame_id", "instance_id", "sample_idx",
        "pred_idx", "gt_idx", "part_name",
    ] + [f"T{i}{j}" for i in range(4) for j in range(4)]

    with csv_path.open("w", newline="", encoding="utf-8") as f, part_csv_path.open(
        "w", newline="", encoding="utf-8"
    ) as part_f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        part_writer = csv.DictWriter(part_f, fieldnames=part_fieldnames)
        writer.writeheader()
        part_writer.writeheader()
        for count, sample_idx in enumerate(sample_indices):
            do_vis = args.vis_every > 0 and count % args.vis_every == 0
            try:
                process_sample(
                    dataset,
                    sample_idx,
                    model,
                    cad,
                    device,
                    args,
                    rng,
                    writer,
                    part_writer,
                    do_vis,
                )
            except Exception as exc:
                if not bool(args.skip_failed):
                    raise
                try:
                    save_failed_debug(dataset, sample_idx, model, device, args, exc)
                except Exception as debug_exc:
                    print(
                        f"[skip-debug-failed] {dataset.samples[sample_idx][0]}/"
                        f"{dataset.samples[sample_idx][1]}: "
                        f"{type(debug_exc).__name__}: {debug_exc}",
                        flush=True,
                    )
                print(
                    f"[skip] {dataset.samples[sample_idx][0]}/"
                    f"{dataset.samples[sample_idx][1]}: {type(exc).__name__}: {exc}",
                    flush=True,
                )
            f.flush()
            part_f.flush()
            print(
                f"[{count + 1}/{len(sample_indices)}] processed "
                f"{dataset.samples[sample_idx][0]}/{dataset.samples[sample_idx][1]}",
                flush=True,
            )

    print(f"[ok] selected videos: {', '.join(video_names)}")
    print(f"[ok] wrote {csv_path}")
    print(f"[ok] wrote {part_csv_path}")
    print(f"[ok] wrote visualizations to {args.output_dir / 'vis'}")


if __name__ == "__main__":
    main()
