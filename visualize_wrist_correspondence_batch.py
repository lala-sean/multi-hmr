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
from PIL import Image, ImageDraw, ImageFont

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

ROBOPEPP_ROOT = Path(__file__).resolve().parent
MULTIHMR_ROOT = ROBOPEPP_ROOT.parents[1]
if str(ROBOPEPP_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBOPEPP_ROOT))
if str(MULTIHMR_ROOT) not in sys.path:
    sys.path.insert(0, str(MULTIHMR_ROOT))

import compare_crop_hcce_robopepp_rarp as cmp  # noqa: E402
from instrument_geometry import quat_wxyz_to_matrix_np  # noqa: E402
from instrument_trimesh_renderer import GMSInstrumentTrimeshRenderer  # noqa: E402
from pose_pnp import matrix_to_quat_wxyz_np  # noqa: E402


def pose_to_params(pose):
    rot = quat_wxyz_to_matrix_np(pose["rot"])
    rvec, _ = cv2.Rodrigues(rot.astype(np.float64))
    return np.array(
        [
            rvec.reshape(3)[0],
            rvec.reshape(3)[1],
            rvec.reshape(3)[2],
            float(pose["trans"][0]),
            float(pose["trans"][1]),
            float(pose["trans"][2]),
            float(pose.get("alpha", 0.0)),
            float(pose.get("theta_l", 0.0)),
            float(pose.get("theta_r", 0.0)),
        ],
        dtype=np.float64,
    )


def pose_from_rvec_trans(rvec, trans):
    rot_mat, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    return {
        "rot": matrix_to_quat_wxyz_np(rot_mat),
        "trans": np.asarray(trans, dtype=np.float64).reshape(3),
        "alpha": 0.0,
        "theta_l": 0.0,
        "theta_r": 0.0,
    }


def label(panel, title, subtitle=None):
    panel = panel.astype(np.uint8)
    bar = 58 if subtitle else 36
    out = np.full((panel.shape[0] + bar, panel.shape[1], 3), 245, dtype=np.uint8)
    out[bar:] = panel
    cv2.putText(out, str(title), (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 1, cv2.LINE_AA)
    if subtitle:
        cv2.putText(out, str(subtitle), (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (35, 35, 35), 1, cv2.LINE_AA)
    return out


def resize_h(panel, height):
    if panel.shape[0] == height:
        return panel
    width = int(round(panel.shape[1] * float(height) / float(panel.shape[0])))
    return cv2.resize(panel, (width, height), interpolation=cv2.INTER_AREA)


def hstack(panels, gap=14):
    height = max(p.shape[0] for p in panels)
    panels = [resize_h(p, height) for p in panels]
    sep = np.full((height, gap, 3), 255, dtype=np.uint8)
    out = []
    for idx, panel in enumerate(panels):
        if idx:
            out.append(sep)
        out.append(panel)
    return np.concatenate(out, axis=1)


def vstack(panels, gap=14):
    width = max(p.shape[1] for p in panels)
    out = []
    for idx, panel in enumerate(panels):
        if panel.shape[1] != width:
            canvas = np.full((panel.shape[0], width, 3), 255, dtype=np.uint8)
            canvas[:, : panel.shape[1]] = panel
            panel = canvas
        if idx:
            out.append(np.full((gap, width, 3), 255, dtype=np.uint8))
        out.append(panel)
    return np.concatenate(out, axis=0)


def draw_point(image, uv, color, radius=1):
    h, w = image.shape[:2]
    x = int(round(float(uv[0])))
    y = int(round(float(uv[1])))
    if -20 <= x < w + 20 and -20 <= y < h + 20:
        cv2.circle(image, (x, y), radius, color, -1, lineType=cv2.LINE_AA)


def draw_connectors(pair, left_panel, uv_left, uv_right, colors, gap=14, title_bar_height=58, radius=1):
    left_w = left_panel.shape[1]
    yoff = int(title_bar_height)
    for u_left, u_right, color in zip(uv_left, uv_right, colors):
        p0 = (int(round(float(u_left[0]))), int(round(float(u_left[1]))) + yoff)
        p1 = (left_w + gap + int(round(float(u_right[0]))), int(round(float(u_right[1]))) + yoff)
        cv2.line(pair, p0, p1, color, 1, lineType=cv2.LINE_AA)
    for u_left, u_right, color in zip(uv_left, uv_right, colors):
        p0 = (int(round(float(u_left[0]))), int(round(float(u_left[1]))) + yoff)
        p1 = (left_w + gap + int(round(float(u_right[0]))), int(round(float(u_right[1]))) + yoff)
        cv2.circle(pair, p0, radius, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(pair, p1, radius, color, -1, lineType=cv2.LINE_AA)
    return pair


def crop_zoom(panel, uv_left, uv_right, margin=80, max_side=760):
    h, w = panel.shape[:2]
    pts = np.concatenate([uv_left, uv_right], axis=0)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) == 0:
        return panel.copy(), np.zeros(2, dtype=np.float64), 1.0
    x0 = max(0, int(np.floor(np.min(pts[:, 0]) - margin)))
    y0 = max(0, int(np.floor(np.min(pts[:, 1]) - margin)))
    x1 = min(w, int(np.ceil(np.max(pts[:, 0]) + margin)))
    y1 = min(h, int(np.ceil(np.max(pts[:, 1]) + margin)))
    if x1 <= x0 or y1 <= y0:
        return panel.copy(), np.zeros(2, dtype=np.float64), 1.0
    crop = panel[y0:y1, x0:x1].copy()
    scale = min(float(max_side) / max(1, crop.shape[1]), float(max_side) / max(1, crop.shape[0]))
    if scale > 1.0:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return crop, np.asarray([x0, y0], dtype=np.float64), float(max(1.0, scale))


def make_colors(n):
    colors = []
    for i in range(n):
        hue = int(round(179 * i / max(1, n - 1)))
        bgr = cv2.cvtColor(np.uint8([[[hue, 220, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
        colors.append(tuple(int(v) for v in bgr.tolist()))
    return colors


def render_mesh_panel(renderer, pose, K_orig, shape):
    mesh_color, depth, _ = renderer.render_pose(pose, K_orig, shape)
    panel = np.zeros((*shape, 3), dtype=np.uint8)
    support = np.asarray(depth) > 0
    panel[support] = mesh_color[support]
    return panel


def project_wrist_points(cad, points_part, pose, K_orig):
    params = pose_to_params(pose)
    transforms = cad.fk(params[:3], params[3:6], params[6], params[7], params[8])
    points_cam = cad.transform_points("wrist", points_part, transforms)
    return cmp.project_camera_points(points_cam, K_orig)


def make_pair(mesh_panel, rgb, uv_mesh, uv_rgb, colors, title, subtitle, zoom):
    mesh_draw = mesh_panel.copy()
    rgb_draw = rgb.copy()
    for a, b, color in zip(uv_mesh, uv_rgb, colors):
        draw_point(mesh_draw, a, color, radius=1)
        draw_point(rgb_draw, b, color, radius=1)

    if zoom:
        mesh_zoom, offset, scale = crop_zoom(mesh_draw, uv_mesh, uv_rgb)
        rgb_zoom, _, _ = crop_zoom(rgb_draw, uv_mesh, uv_rgb)
        uv_mesh = (uv_mesh - offset.reshape(1, 2)) * scale
        uv_rgb = (uv_rgb - offset.reshape(1, 2)) * scale
        mesh_draw = mesh_zoom
        rgb_draw = rgb_zoom

    left = label(mesh_draw, f"mesh {title}", subtitle)
    right = label(rgb_draw, "rgb original", "same random wrist pixels; no IDs")
    pair = hstack([left, right], gap=14)
    pair = draw_connectors(
        pair,
        left,
        uv_mesh,
        uv_rgb,
        colors,
        gap=14,
        title_bar_height=58,
        radius=1,
    )
    return pair


def stats_from_residuals(uv_mesh, uv_rgb):
    res = np.linalg.norm(np.asarray(uv_mesh) - np.asarray(uv_rgb), axis=1)
    if len(res) == 0:
        return {"mean": float("nan"), "median": float("nan"), "p90": float("nan")}
    return {
        "mean": float(np.mean(res)),
        "median": float(np.median(res)),
        "p90": float(np.percentile(res, 90)),
    }


def contact_sheet(paths, out_path, max_width=2200):
    rows = []
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    except Exception:
        font = ImageFont.load_default()
    for idx, path in enumerate(paths):
        im = Image.open(path).convert("RGB")
        if im.width > max_width:
            new_h = int(round(im.height * max_width / im.width))
            im = im.resize((max_width, new_h), Image.Resampling.LANCZOS)
        label_h = 38
        canvas = Image.new("RGB", (im.width, im.height + label_h), (255, 255, 255))
        canvas.paste(im, (0, label_h))
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 7), f"{idx:02d}  {path.name}", fill=(0, 0, 0), font=font)
        rows.append(canvas)
    if not rows:
        return
    sheet = Image.new("RGB", (max(r.width for r in rows), sum(r.height for r in rows)), (255, 255, 255))
    y = 0
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height
    sheet.save(out_path, quality=92, optimize=True)


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


def run(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = cmp.configure_device(args.device)
    base_args = cmp.build_parser().parse_args([])
    base_args.output_dir = str(out_dir)
    base_args.hcce_checkpoint = str(args.hcce_checkpoint)
    base_args.needle_manifest_csv = str(args.needle_manifest_csv)
    base_args.dataset_cache_dir = str(args.dataset_cache_dir)
    base_args.datasets = ["needleGrasping"]
    base_args.crop_size = 224
    base_args.surface_snap_method = "surface"
    base_args.surface_k_faces = int(args.surface_k_faces)
    base_args.point_select = "random"
    base_args.shaft_raw_x_min = float(args.shaft_raw_x_min)
    base_args.optim_strategy = "decoupled"
    base_args.optim_parts = "wrist_gripper"
    base_args.render_seg_metrics = 0
    base_args.vis_limit = 0
    base_args.crop_vis_limit = 0
    base_args.devices = [str(args.device)]

    dataset = cmp.build_needle_dataset(base_args)
    items, meta = cmp.build_needle_items(base_args)
    if args.max_items > 0:
        items = items[: int(args.max_items)]
    print(f"[manifest] {meta}")

    model, model_meta = cmp.load_hcce_model(args.hcce_checkpoint, device)
    cad = cmp.InstrumentCAD(cmp.CAD_ROOT)
    renderer = GMSInstrumentTrimeshRenderer(device)
    to_tensor = tv_transforms.Compose(
        [
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    combined_zoom_paths = []
    gt_zoom_paths = []
    pnp_zoom_paths = []
    summary_rows = []
    point_rows = []

    for ordinal, item in enumerate(items):
        image, target = dataset[item.dataset_idx]
        target_like = cmp.target_like_from_needle(target)
        gt_pose = cmp.pose_from_target(target)
        x = image.unsqueeze(0).to(device)
        K_crop_t = torch.from_numpy(target_like["K_crop"]).unsqueeze(0).to(device)
        with torch.inference_mode(), torch.amp.autocast(device_type="cuda", enabled=device.type == "cuda", dtype=torch.bfloat16):
            out = model(x, K_crop_t)

        rng = np.random.default_rng(int(args.seed) + ordinal * 9973)
        corr, wrist_uv, wrist_points, counts = cmp.build_crop_hcce_correspondences(
            out,
            cad,
            model_meta,
            target_like,
            base_args,
            rng,
        )
        rvec0, trans0, inliers = cmp.solve_wrist_pnp(
            wrist_uv,
            wrist_points,
            np.asarray(target_like["K_crop"], dtype=np.float64),
            base_args,
        )
        pnp_pose = pose_from_rvec_trans(rvec0, trans0)
        pnp_rot_err = cmp.rotation_error_deg(pnp_pose["rot"], gt_pose["rot"])
        pnp_trans_err = float(np.linalg.norm(np.asarray(pnp_pose["trans"]) - np.asarray(gt_pose["trans"])))

        wrist_mask = corr.part_names == "wrist"
        uv_crop_all = corr.uv[wrist_mask]
        points_all = corr.points_part[wrist_mask]
        if len(uv_crop_all) == 0:
            print(f"[skip] no wrist correspondences {item.video} {item.frame_id} inst{item.instance_id}")
            continue
        keep_n = min(int(args.num_points), len(uv_crop_all))
        keep = rng.choice(len(uv_crop_all), size=keep_n, replace=False)
        uv_crop = uv_crop_all[keep]
        points_part = points_all[keep]
        uv_rgb = cmp.crop_points_to_original(
            uv_crop,
            target_like["bbox_min"],
            target_like["scale"],
            target_like["pad"],
        ).astype(np.float64)

        rgb = target_like["orig_rgb"].astype(np.uint8)
        K_orig = np.asarray(target_like["K_orig"], dtype=np.float64)
        uv_gt = project_wrist_points(cad, points_part, gt_pose, K_orig)
        uv_pnp = project_wrist_points(cad, points_part, pnp_pose, K_orig)
        valid = np.isfinite(uv_gt).all(axis=1) & np.isfinite(uv_pnp).all(axis=1) & np.isfinite(uv_rgb).all(axis=1)
        uv_gt = uv_gt[valid]
        uv_pnp = uv_pnp[valid]
        uv_rgb = uv_rgb[valid]
        uv_crop = uv_crop[valid]
        points_part = points_part[valid]
        if len(uv_rgb) == 0:
            print(f"[skip] no valid projected wrist correspondences {item.video} {item.frame_id} inst{item.instance_id}")
            continue

        colors = make_colors(len(uv_rgb))
        gt_stats = stats_from_residuals(uv_gt, uv_rgb)
        pnp_stats = stats_from_residuals(uv_pnp, uv_rgb)
        shape = rgb.shape[:2]
        gt_mesh = render_mesh_panel(renderer, gt_pose, K_orig, shape)
        pnp_mesh = render_mesh_panel(renderer, pnp_pose, K_orig, shape)
        stem = f"{ordinal:03d}_{item.video}_{item.frame_id}_inst{int(item.instance_id)}"

        gt_subtitle = f"random wrist n={len(uv_rgb)}, residual mean/p90={gt_stats['mean']:.1f}/{gt_stats['p90']:.1f}px"
        pnp_subtitle = (
            f"dense wrist PnP init r={pnp_rot_err:.1f}deg t={pnp_trans_err:.3f}m, "
            f"residual mean/p90={pnp_stats['mean']:.1f}/{pnp_stats['p90']:.1f}px, inliers={int(inliers)}"
        )
        gt_zoom = make_pair(gt_mesh, rgb, uv_gt, uv_rgb, colors, "GT-pose wrist", gt_subtitle, zoom=True)
        pnp_zoom = make_pair(pnp_mesh, rgb, uv_pnp, uv_rgb, colors, "dense-PnP-init wrist", pnp_subtitle, zoom=True)
        combined = vstack([gt_zoom, pnp_zoom], gap=18)
        gt_full = make_pair(gt_mesh, rgb, uv_gt, uv_rgb, colors, "GT-pose wrist", gt_subtitle, zoom=False)
        pnp_full = make_pair(pnp_mesh, rgb, uv_pnp, uv_rgb, colors, "dense-PnP-init wrist", pnp_subtitle, zoom=False)

        item_dir = out_dir / "per_frame" / stem
        item_dir.mkdir(parents=True, exist_ok=True)
        gt_zoom_path = item_dir / "wrist_corr_gtpose_zoom.jpg"
        pnp_zoom_path = item_dir / "wrist_corr_pnpinit_zoom.jpg"
        combined_path = item_dir / "wrist_corr_gtpose_vs_pnpinit_zoom.jpg"
        Image.fromarray(gt_zoom).save(gt_zoom_path, quality=94)
        Image.fromarray(pnp_zoom).save(pnp_zoom_path, quality=94)
        Image.fromarray(combined).save(combined_path, quality=94)
        Image.fromarray(gt_full).save(item_dir / "wrist_corr_gtpose_full.jpg", quality=94)
        Image.fromarray(pnp_full).save(item_dir / "wrist_corr_pnpinit_full.jpg", quality=94)

        gt_zoom_paths.append(gt_zoom_path)
        pnp_zoom_paths.append(pnp_zoom_path)
        combined_zoom_paths.append(combined_path)
        summary_rows.append(
            {
                "ordinal": int(ordinal),
                "video": item.video,
                "frame_id": item.frame_id,
                "instance_id": int(item.instance_id),
                "dataset_idx": int(item.dataset_idx),
                "wrist_corr_total": int(len(uv_crop_all)),
                "wrist_corr_sampled": int(len(uv_rgb)),
                "dense_wrist_pnp_inliers": int(inliers),
                "dense_wrist_pnp_rot_err_deg": float(pnp_rot_err),
                "dense_wrist_pnp_trans_err_m": float(pnp_trans_err),
                "gtpose_residual_mean_px": gt_stats["mean"],
                "gtpose_residual_median_px": gt_stats["median"],
                "gtpose_residual_p90_px": gt_stats["p90"],
                "pnpinit_residual_mean_px": pnp_stats["mean"],
                "pnpinit_residual_median_px": pnp_stats["median"],
                "pnpinit_residual_p90_px": pnp_stats["p90"],
                "combined_zoom": str(combined_path),
            }
        )
        for point_idx, (uv_c, uv_o, u_gt, u_pnp, pt) in enumerate(zip(uv_crop, uv_rgb, uv_gt, uv_pnp, points_part)):
            point_rows.append(
                {
                    "ordinal": int(ordinal),
                    "video": item.video,
                    "frame_id": item.frame_id,
                    "instance_id": int(item.instance_id),
                    "point_idx": int(point_idx),
                    "uv_crop_x": float(uv_c[0]),
                    "uv_crop_y": float(uv_c[1]),
                    "uv_orig_x": float(uv_o[0]),
                    "uv_orig_y": float(uv_o[1]),
                    "gt_proj_x": float(u_gt[0]),
                    "gt_proj_y": float(u_gt[1]),
                    "pnp_proj_x": float(u_pnp[0]),
                    "pnp_proj_y": float(u_pnp[1]),
                    "gt_residual_px": float(np.linalg.norm(u_gt - uv_o)),
                    "pnp_residual_px": float(np.linalg.norm(u_pnp - uv_o)),
                    "wrist_surface_x": float(pt[0]),
                    "wrist_surface_y": float(pt[1]),
                    "wrist_surface_z": float(pt[2]),
                }
            )
        print(
            f"[ok] {ordinal + 1}/{len(items)} {stem} "
            f"pnp r={pnp_rot_err:.1f} t={pnp_trans_err:.3f} "
            f"gt-res={gt_stats['mean']:.1f}px pnp-res={pnp_stats['mean']:.1f}px"
        )

    contact_sheet(combined_zoom_paths, out_dir / "contact_sheet_wrist_gtpose_vs_pnpinit_zoom.jpg")
    contact_sheet(gt_zoom_paths, out_dir / "contact_sheet_wrist_gtpose_zoom.jpg")
    contact_sheet(pnp_zoom_paths, out_dir / "contact_sheet_wrist_pnpinit_zoom.jpg")
    write_csv(out_dir / "wrist_correspondence_summary.csv", summary_rows)
    write_csv(out_dir / "wrist_correspondence_points.csv", point_rows)
    (out_dir / "manifest.json").write_text(json.dumps(summary_rows, indent=2, allow_nan=True), encoding="utf-8")
    print(f"[done] wrote {len(summary_rows)} wrist debug cases to {out_dir}")


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--needle_manifest_csv", type=str, required=True)
    parser.add_argument(
        "--hcce_checkpoint",
        type=str,
        default=str(cmp.ROBOPEPP_ROOT / "logs/hcce_crop224_fixbf16_rarp_gpu0123_bs56_fromscratch/checkpoints/iter0046000.pt"),
    )
    parser.add_argument(
        "--dataset_cache_dir",
        type=str,
        default=str(cmp.ROBOPEPP_ROOT / "logs/crop_hcce_vs_robopepp_hccefixbf16_last_ng_suture10_stride4/dataset_cache"),
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--surface_k_faces", type=int, default=32)
    parser.add_argument("--shaft_raw_x_min", type=float, default=-0.5)
    parser.add_argument("--num_points", type=int, default=32)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max_items", type=int, default=0)
    return parser


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
