import argparse
import csv
import importlib.util
import json
import math
import os
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

ROBOPEPP_ROOT = Path(__file__).resolve().parent
MULTIHMR_ROOT = ROBOPEPP_ROOT.parents[1]
if str(ROBOPEPP_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBOPEPP_ROOT))
if str(MULTIHMR_ROOT) not in sys.path:
    sys.path.insert(0, str(MULTIHMR_ROOT))

datasets_pkg = types.ModuleType("datasets")
datasets_pkg.__path__ = [str(MULTIHMR_ROOT / "datasets")]
sys.modules["datasets"] = datasets_pkg

from instrument_geometry import instrument_keypoints_camera_np, project_points_np  # noqa: E402
from pose_pnp import matrix_to_quat_wxyz_np, pose_from_keypoints_pnp  # noqa: E402
from predict_instrument_pose import (  # noqa: E402
    DATASETS,
    add_title,
    concat_panels,
    heatmap_argmax,
    load_model,
    overlay_part_mask,
    pad_rgb_to_square,
    square_image_geometry,
)


DEFAULT_QUANT_CSV = (
    ROBOPEPP_ROOT
    / "logs/robopepp_vs_hcce_quant_needleGrasping_suturePulling10_stride4_pnp/needleGrasping/per_instance.csv"
)
DEFAULT_HCCE_CSV = MULTIHMR_ROOT / "eval_outputs/best_iter55000_parallel_meshtexturefix/needleGrasping/per_instance.csv"
DEFAULT_CKPT = ROBOPEPP_ROOT / "logs/robopepp_instrument_pose_rarp_jepa_bs56_gpu0123/checkpoints/last.pt"

KEYPOINT_NAMES = ("shaft_axis", "wrist_shaft", "wrist_gripper", "left_tip", "right_tip")
SKELETON = ((0, 1), (1, 2), (2, 3), (2, 4))
GT_COLOR = (40, 220, 70)
ROBO_COLOR = (255, 70, 220)
HCCE_COLOR = (60, 210, 255)
MUTED = (180, 180, 180)


class Pose:
    def __setstate__(self, state):
        self.__dict__.update(state if isinstance(state, dict) else {})


def load_robopepp_rarp_dataset():
    path = ROBOPEPP_ROOT / "datasets" / "rarp_instrument.py"
    spec = importlib.util.spec_from_file_location("robopepp_vis_rarp_instrument", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.RoboPEPPRARPInstrument


def norm_frame_id(frame_id):
    return f"{int(frame_id):05d}"


def key_columns(df):
    out = df.copy()
    out["frame_id_norm"] = out["frame_id"].map(norm_frame_id)
    out["instance_id_int"] = out["instance_id"].astype(int)
    return out


def numeric(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def finite_row(row, cols):
    return all(math.isfinite(float(row[col])) for col in cols)


def selection_score(df):
    return (
        (df["hcce_reproj_rmse_px"] - df["robopepp_reproj_rmse_px"]) / 20.0
        + (df["hcce_trans_err_m"] - df["robopepp_trans_err_m"]) / 0.01
        + (df["hcce_rot_err_deg"] - df["robopepp_rot_err_deg"]) / 20.0
    )


def select_rows(args):
    quant = pd.read_csv(args.quant_csv)
    hcce = pd.read_csv(args.hcce_csv)
    quant = key_columns(quant)
    hcce = key_columns(hcce)
    metric_cols = [
        "robopepp_reproj_rmse_px",
        "hcce_reproj_rmse_px",
        "robopepp_trans_err_m",
        "hcce_trans_err_m",
        "robopepp_rot_err_deg",
        "hcce_rot_err_deg",
        "robopepp_action_rmse_rad",
        "hcce_action_mse",
    ]
    quant = numeric(quant, metric_cols)
    hcce_pose_cols = [
        "hcce_pred_rvec_x",
        "hcce_pred_rvec_y",
        "hcce_pred_rvec_z",
        "hcce_pred_tx",
        "hcce_pred_ty",
        "hcce_pred_tz",
        "hcce_pred_alpha",
        "hcce_pred_theta_l",
        "hcce_pred_theta_r",
    ]
    hcce = numeric(hcce, hcce_pose_cols)
    merged = quant.merge(
        hcce[["video", "frame_id_norm", "instance_id_int", *hcce_pose_cols]],
        on=["video", "frame_id_norm", "instance_id_int"],
        how="left",
        suffixes=("", "_orig_hcce"),
    )
    valid_cols = [
        "robopepp_reproj_rmse_px",
        "hcce_reproj_rmse_px",
        "robopepp_trans_err_m",
        "hcce_trans_err_m",
        "robopepp_rot_err_deg",
        "hcce_rot_err_deg",
        *hcce_pose_cols,
    ]
    ok = merged[
        merged["robopepp_status"].eq("ok")
        & merged["hcce_status"].eq("ok")
    ].copy()
    ok = ok.dropna(subset=valid_cols)
    ok = ok[np.isfinite(ok[valid_cols].to_numpy(dtype=np.float64)).all(axis=1)].copy()
    ok["reproj_gap"] = ok["hcce_reproj_rmse_px"] - ok["robopepp_reproj_rmse_px"]
    ok["trans_gap"] = ok["hcce_trans_err_m"] - ok["robopepp_trans_err_m"]
    ok["rot_gap"] = ok["hcce_rot_err_deg"] - ok["robopepp_rot_err_deg"]
    ok["score"] = selection_score(ok)

    worse = ok[
        (ok["reproj_gap"] > args.worse_reproj_gap)
        & ((ok["trans_gap"] > 0.0) | (ok["rot_gap"] > 0.0))
    ].sort_values(["score", "reproj_gap"], ascending=False)
    similar = ok[
        (ok["reproj_gap"].abs() <= args.similar_reproj_gap)
        & (ok["trans_gap"].abs() <= args.similar_trans_gap)
        & (ok["rot_gap"].abs() <= args.similar_rot_gap)
    ].assign(abs_score=lambda x: x["score"].abs()).sort_values(["abs_score", "reproj_gap"])
    better = ok[
        (ok["reproj_gap"] < -args.better_reproj_gap)
        & ((ok["trans_gap"] < 0.0) | (ok["rot_gap"] < 0.0))
    ].sort_values(["score", "reproj_gap"], ascending=True)

    selected = []
    used = set()

    def take(label, frame, count):
        taken = 0
        for _, row in frame.iterrows():
            key = (row["video"], row["frame_id_norm"], int(row["instance_id_int"]))
            if key in used:
                continue
            item = row.to_dict()
            item["selection_label"] = label
            selected.append(item)
            used.add(key)
            taken += 1
            if taken >= count:
                break
        return taken

    take("hcce_worse_than_robopepp", worse, args.num_worse)
    take("similar", similar, args.num_similar)
    take("hcce_better_than_robopepp", better, args.num_better)
    remaining = args.num_total - len(selected)
    if remaining > 0:
        fill = ok.sort_values("score", ascending=False)
        take("fill_high_signal", fill, remaining)
    return selected[: args.num_total], {
        "available_ok": int(len(ok)),
        "available_hcce_worse": int(len(worse)),
        "available_similar": int(len(similar)),
        "available_hcce_better": int(len(better)),
        "selected_total": int(min(len(selected), args.num_total)),
    }


def build_dataset(args):
    dataset_cls = load_robopepp_rarp_dataset()
    cfg = DATASETS["needleGrasping"]
    dataset = dataset_cls(
        cfg["dataset_root"],
        cfg["pose_root"],
        split="test",
        training=False,
        crop_size=args.crop_size,
        train_ratio=args.needle_train_ratio,
        subsample=1,
        min_dice=(args.min_dice_shaft, args.min_dice_wrist, args.min_dice_gripper),
        canonicalize_pose_symmetry=bool(args.canonicalize_pose_symmetry),
        canonical_eps=args.canonical_eps,
        cache_dir=args.dataset_cache_dir,
    )
    index = {
        (video, norm_frame_id(frame_id), int(inst)): idx
        for idx, (video, frame_id, inst, _) in enumerate(dataset.samples)
    }
    return dataset, index


def hccerow_to_pose(row):
    rvec = np.array(
        [row["hcce_pred_rvec_x"], row["hcce_pred_rvec_y"], row["hcce_pred_rvec_z"]],
        dtype=np.float64,
    )
    rot, _ = cv2.Rodrigues(rvec.reshape(3, 1))
    trans = np.array([row["hcce_pred_tx"], row["hcce_pred_ty"], row["hcce_pred_tz"]], dtype=np.float64)
    return {
        "rot": matrix_to_quat_wxyz_np(rot),
        "trans": trans,
        "alpha": float(row["hcce_pred_alpha"]),
        "theta_l": float(row["hcce_pred_theta_l"]),
        "theta_r": float(row["hcce_pred_theta_r"]),
    }


def target_pose(target):
    return {
        "rot": target["wrist_quat"].numpy().astype(np.float64),
        "trans": target["wrist_trans"].numpy().astype(np.float64),
        "alpha": float(target["action"][0].item()),
        "theta_l": float(target["action"][1].item()),
        "theta_r": float(target["action"][2].item()),
    }


def output_pose(model_out):
    action = model_out["action_pred"][0].detach().cpu().numpy().astype(np.float64)
    quat = model_out["wrist_quat_pred"][0].detach().cpu().numpy().astype(np.float64)
    trans = model_out["wrist_trans_pred"][0].detach().cpu().numpy().astype(np.float64)
    return {
        "rot": quat,
        "trans": trans,
        "alpha": float(action[0]),
        "theta_l": float(action[1]),
        "theta_r": float(action[2]),
    }


def robopepp_pose(model, image, target, device, args):
    x = image.unsqueeze(0).to(device, non_blocking=True)
    K_crop = target["K"].unsqueeze(0).to(device, non_blocking=True)
    with torch.inference_mode(), torch.amp.autocast(
        device_type="cuda", enabled=(device.type == "cuda"), dtype=torch.bfloat16
    ):
        out = model(x, K_crop, masks_enc=None, masks_pred=None)
    pred_hm_crop_t, hm_scores_t = heatmap_argmax(out["keypoint_heatmaps"].float().detach().cpu())
    pred_hm_crop = pred_hm_crop_t[0].numpy().astype(np.float32)
    scores = hm_scores_t[0].numpy().astype(np.float32)
    direct = output_pose(out)
    if args.pose_recovery == "pnp":
        return pose_from_keypoints_pnp(
            pred_hm_crop,
            [direct["alpha"], direct["theta_l"], direct["theta_r"]],
            target["K"].numpy(),
            scores=scores,
            min_score=args.pnp_min_score,
        )
    return direct


def pose_keypoints_uv(pose, K):
    action = np.array([pose["alpha"], pose["theta_l"], pose["theta_r"]], dtype=np.float64)
    pts = instrument_keypoints_camera_np(pose["rot"], pose["trans"], action)
    return project_points_np(pts, K).astype(np.float32)


def square_points(points, scale, pad_x, pad_y):
    pts = np.asarray(points, dtype=np.float32).copy()
    pts[:, 0] = pts[:, 0] * float(scale) + float(pad_x)
    pts[:, 1] = pts[:, 1] * float(scale) + float(pad_y)
    return pts


def draw_text(out, text, org, color=(255, 255, 255), scale=0.44, thickness=1):
    cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_points_and_skeleton(out, points, color, valid=None, prefix="", radius=5):
    points = np.asarray(points, dtype=np.float32)
    if valid is None:
        valid = np.ones((len(points),), dtype=bool)
    for a, b in SKELETON:
        if bool(valid[a]) and bool(valid[b]):
            pa = tuple(np.round(points[a]).astype(int))
            pb = tuple(np.round(points[b]).astype(int))
            cv2.line(out, pa, pb, color, 2, cv2.LINE_AA)
    for i, xy in enumerate(points):
        x, y = np.round(xy).astype(int)
        if bool(valid[i]):
            cv2.circle(out, (x, y), radius, color, -1, cv2.LINE_AA)
        else:
            cv2.circle(out, (x, y), radius, color, 1, cv2.LINE_AA)
        if prefix:
            draw_text(out, f"{prefix}{i}", (x + 5, y - 4), color, scale=0.34)


def draw_error_lines(out, gt_points, pred_points, valid, color):
    for g, p, ok in zip(gt_points, pred_points, valid):
        if not bool(ok):
            continue
        pg = tuple(np.round(g).astype(int))
        pp = tuple(np.round(p).astype(int))
        cv2.line(out, pg, pp, MUTED, 1, cv2.LINE_AA)
        cv2.circle(out, pp, 4, color, -1, cv2.LINE_AA)


def make_pose_panel(rgb_sq, gt_sq, pred_sq, valid, color, title, lines):
    out = rgb_sq.copy()
    draw_points_and_skeleton(out, gt_sq, GT_COLOR, valid=valid, prefix="g", radius=4)
    draw_error_lines(out, gt_sq, pred_sq, valid, color)
    draw_points_and_skeleton(out, pred_sq, color, valid=np.ones(len(pred_sq), dtype=bool), prefix="p", radius=4)
    y = 46
    for line in lines:
        draw_text(out, line, (10, y), (255, 255, 255), scale=0.42)
        y += 18
    return add_title(out, title)


def make_gt_panel(rgb_sq, gt_part_sq, gt_points_sq, valid):
    out = overlay_part_mask(rgb_sq, gt_part_sq, alpha=0.55)
    draw_points_and_skeleton(out, gt_points_sq, GT_COLOR, valid=valid, prefix="", radius=4)
    return add_title(out, "GT segmentation + visible keypoints")


def format_method_lines(row, prefix):
    if prefix == "robopepp":
        return [
            f"reproj {row['robopepp_reproj_rmse_px']:.2f}px",
            f"trans {row['robopepp_trans_err_m']*1000:.1f}mm",
            f"rot {row['robopepp_rot_err_deg']:.1f}deg",
            f"action {row['robopepp_action_rmse_rad']:.3f}rad",
        ]
    action_rmse = math.sqrt(max(float(row["hcce_action_mse"]), 0.0)) if math.isfinite(float(row["hcce_action_mse"])) else float("nan")
    return [
        f"reproj {row['hcce_reproj_rmse_px']:.2f}px",
        f"trans {row['hcce_trans_err_m']*1000:.1f}mm",
        f"rot {row['hcce_rot_err_deg']:.1f}deg",
        f"action {action_rmse:.3f}rad",
    ]


def save_one(out_path, row, image, target, robo_pose, hcce_pose, args):
    rgb = target["orig_rgb"].numpy()
    K = target["K_orig"].numpy()
    gt_part = target["part_mask_orig"].numpy().astype(np.uint8)
    gt_uv = target["keypoints_orig"].numpy().astype(np.float32)
    valid = target["keypoints_valid_orig"].numpy().astype(bool)
    robo_uv = pose_keypoints_uv(robo_pose, K)
    hcce_uv = pose_keypoints_uv(hcce_pose, K)

    scale, pad_x, pad_y = square_image_geometry(rgb, args.panel_size)
    rgb_sq = pad_rgb_to_square(rgb, args.panel_size, scale, pad_x, pad_y)
    gt_part_sq = np.zeros((args.panel_size, args.panel_size), dtype=np.uint8)
    new_w = int(rgb.shape[1] * scale)
    new_h = int(rgb.shape[0] * scale)
    gt_part_resized = np.asarray(
        Image.fromarray(gt_part).resize((new_w, new_h), Image.NEAREST),
        dtype=np.uint8,
    )
    gt_part_sq[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = gt_part_resized
    gt_sq = square_points(gt_uv, scale, pad_x, pad_y)
    robo_sq = square_points(robo_uv, scale, pad_x, pad_y)
    hcce_sq = square_points(hcce_uv, scale, pad_x, pad_y)

    gt_panel = make_gt_panel(rgb_sq, gt_part_sq, gt_sq, valid)
    robo_panel = make_pose_panel(
        rgb_sq,
        gt_sq,
        robo_sq,
        valid,
        ROBO_COLOR,
        "RoboPEPP pose vs GT keypoints",
        format_method_lines(row, "robopepp"),
    )
    hcce_panel = make_pose_panel(
        rgb_sq,
        gt_sq,
        hcce_sq,
        valid,
        HCCE_COLOR,
        "densepart_h2/HCCE pose vs GT keypoints",
        format_method_lines(row, "hcce"),
    )
    both = rgb_sq.copy()
    draw_points_and_skeleton(both, gt_sq, GT_COLOR, valid=valid, prefix="", radius=4)
    draw_points_and_skeleton(both, robo_sq, ROBO_COLOR, valid=np.ones(len(robo_sq), dtype=bool), prefix="", radius=4)
    draw_points_and_skeleton(both, hcce_sq, HCCE_COLOR, valid=np.ones(len(hcce_sq), dtype=bool), prefix="", radius=4)
    draw_text(both, "green=GT magenta=RoboPEPP cyan=HCCE", (10, 46), (255, 255, 255), scale=0.42)
    draw_text(
        both,
        f"gap reproj={row['reproj_gap']:.1f}px trans={row['trans_gap']*1000:.1f}mm rot={row['rot_gap']:.1f}deg",
        (10, 66),
        (255, 255, 255),
        scale=0.42,
    )
    both = add_title(both, f"{row['selection_label']} | {row['video']} {row['frame_id_norm']} inst{int(row['instance_id_int'])}")

    canvas = concat_panels([gt_panel, robo_panel, hcce_panel, both], separator=6)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(out_path)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main(args):
    selected, selection_meta = select_rows(args)
    dataset, index = build_dataset(args)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, _ = load_model(Path(args.checkpoint), device)
    out_dir = Path(args.output_dir)
    rows = []
    for out_i, row in enumerate(selected):
        key = (row["video"], row["frame_id_norm"], int(row["instance_id_int"]))
        if key not in index:
            print(f"[skip] not in dataset index: {key}", flush=True)
            continue
        image, target = dataset[index[key]]
        robo_pose = robopepp_pose(model, image, target, device, args)
        hcce_pose = hccerow_to_pose(row)
        stem = f"{out_i + 1:03d}_{row['selection_label']}_{row['video']}_{row['frame_id_norm']}_inst{int(row['instance_id_int'])}"
        rel = Path(row["selection_label"]) / f"{stem}.jpg"
        save_one(out_dir / "vis" / rel, row, image, target, robo_pose, hcce_pose, args)
        manifest_row = {
            "out_index": out_i + 1,
            "selection_label": row["selection_label"],
            "video": row["video"],
            "frame_id": row["frame_id_norm"],
            "instance_id": int(row["instance_id_int"]),
            "image_path": str(out_dir / "vis" / rel),
            "robopepp_reproj_rmse_px": float(row["robopepp_reproj_rmse_px"]),
            "hcce_reproj_rmse_px": float(row["hcce_reproj_rmse_px"]),
            "reproj_gap_hcce_minus_robopepp_px": float(row["reproj_gap"]),
            "robopepp_trans_err_m": float(row["robopepp_trans_err_m"]),
            "hcce_trans_err_m": float(row["hcce_trans_err_m"]),
            "trans_gap_hcce_minus_robopepp_m": float(row["trans_gap"]),
            "robopepp_rot_err_deg": float(row["robopepp_rot_err_deg"]),
            "hcce_rot_err_deg": float(row["hcce_rot_err_deg"]),
            "rot_gap_hcce_minus_robopepp_deg": float(row["rot_gap"]),
            "robopepp_action_rmse_rad": float(row["robopepp_action_rmse_rad"]),
            "hcce_action_rmse_rad": math.sqrt(max(float(row["hcce_action_mse"]), 0.0)),
        }
        rows.append(manifest_row)
        if (out_i + 1) == 1 or (out_i + 1) % args.print_freq == 0 or (out_i + 1) == len(selected):
            print(f"[{out_i + 1}/{len(selected)}] {stem}", flush=True)
    write_csv(out_dir / "manifest.csv", rows)
    labels = pd.Series([row["selection_label"] for row in rows]).value_counts().to_dict()
    summary = {**selection_meta, "written": len(rows), "written_by_label": labels}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {out_dir / 'manifest.csv'}", flush=True)
    print(f"wrote {out_dir / 'summary.json'}", flush=True)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quant_csv", type=str, default=str(DEFAULT_QUANT_CSV))
    parser.add_argument("--hcce_csv", type=str, default=str(DEFAULT_HCCE_CSV))
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CKPT))
    parser.add_argument("--output_dir", type=str, default=str(ROBOPEPP_ROOT / "logs/robopepp_hcce_pose_error_100_comparison"))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--panel_size", type=int, default=630)
    parser.add_argument("--pose_recovery", choices=["pnp", "direct"], default="pnp")
    parser.add_argument("--pnp_min_score", type=float, default=0.0)
    parser.add_argument("--num_total", type=int, default=100)
    parser.add_argument("--num_worse", type=int, default=60)
    parser.add_argument("--num_similar", type=int, default=20)
    parser.add_argument("--num_better", type=int, default=20)
    parser.add_argument("--worse_reproj_gap", type=float, default=20.0)
    parser.add_argument("--similar_reproj_gap", type=float, default=5.0)
    parser.add_argument("--similar_trans_gap", type=float, default=0.005)
    parser.add_argument("--similar_rot_gap", type=float, default=10.0)
    parser.add_argument("--better_reproj_gap", type=float, default=5.0)
    parser.add_argument("--needle_train_ratio", type=float, default=0.95)
    parser.add_argument("--canonicalize_pose_symmetry", type=int, choices=[0, 1], default=1)
    parser.add_argument("--canonical_eps", type=float, default=0.08)
    parser.add_argument("--min_dice_shaft", type=float, default=0.8)
    parser.add_argument("--min_dice_wrist", type=float, default=0.6)
    parser.add_argument("--min_dice_gripper", type=float, default=0.6)
    parser.add_argument("--dataset_cache_dir", type=str, default=str(ROBOPEPP_ROOT / "logs/robopepp_quant_cache/dataset_cache"))
    parser.add_argument("--print_freq", type=int, default=10)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
