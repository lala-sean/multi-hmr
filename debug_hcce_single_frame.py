import argparse
import copy
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as tv_transforms
from PIL import Image
from scipy.optimize import least_squares

import compare_crop_hcce_robopepp_rarp as cmp
from instrument_geometry import quat_wxyz_to_matrix_np
from instrument_trimesh_renderer import GMSInstrumentTrimeshRenderer


def _pose_to_params(pose):
    rot = quat_wxyz_to_matrix_np(pose["rot"])
    rvec, _ = cv2.Rodrigues(rot.astype(np.float64))
    return np.array(
        [
            rvec.reshape(3)[0],
            rvec.reshape(3)[1],
            rvec.reshape(3)[2],
            pose["trans"][0],
            pose["trans"][1],
            pose["trans"][2],
            pose["alpha"],
            pose["theta_l"],
            pose["theta_r"],
        ],
        dtype=np.float64,
    )


def _subset_corr(corr, mask):
    return cmp.Correspondences(
        uv=corr.uv[mask],
        points_part=corr.points_part[mask],
        part_names=corr.part_names[mask],
    )


def _rmse_xy(res):
    res = np.asarray(res, dtype=np.float64).reshape(-1, 2)
    if len(res) == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.sum(res * res, axis=1))))


def _part_residual_stats(cad, corr, params, K, args):
    out = {}
    for part_key, part_names in (
        ("wrist", ["wrist"]),
        ("shaft", ["shaft"]),
        ("gripper", ["l_gripper", "r_gripper"]),
        ("wrist_gripper", ["wrist", "l_gripper", "r_gripper"]),
        ("all", ["wrist", "shaft", "l_gripper", "r_gripper"]),
    ):
        mask = np.isin(corr.part_names, part_names)
        if not bool(mask.any()):
            out[part_key] = {"n": 0, "rmse_px": float("nan")}
            continue
        active = _subset_corr(corr, mask)
        res = cmp.correspondence_residuals(
            cad,
            active,
            K,
            args,
            params[:3],
            params[3:6],
            params[6],
            params[7],
            params[8],
        ).reshape(-1, 2)
        norms = np.linalg.norm(res, axis=1)
        out[part_key] = {
            "n": int(mask.sum()),
            "rmse_px": _rmse_xy(res),
            "mean_dx_px": float(res[:, 0].mean()),
            "mean_dy_px": float(res[:, 1].mean()),
            "median_norm_px": float(np.median(norms)),
            "p90_norm_px": float(np.percentile(norms, 90)),
            "p95_norm_px": float(np.percentile(norms, 95)),
        }
    return out


def _fit_with_builtin(cad, corr, wrist_uv, wrist_points, K, args, strategy, parts):
    local_args = copy.copy(args)
    local_args.optim_strategy = strategy
    local_args.optim_parts = parts
    local_args.gripper_no_wrist_pose = 0
    rvec0, trans0, inliers = cmp.solve_wrist_pnp(wrist_uv, wrist_points, K.astype(np.float64), local_args)
    params, rmse_all, nfev, rmse_wg, rmse_shaft = cmp.optimize_pose(
        cad,
        corr,
        rvec0,
        trans0,
        K.astype(np.float64),
        local_args,
    )
    pose = cmp.pose_from_opt_params(params)
    return {
        "pose": pose,
        "params": params,
        "pnp_inliers": int(inliers),
        "nfev": int(nfev),
        "rmse_all_px": float(rmse_all),
        "rmse_wrist_gripper_px": float(rmse_wg),
        "rmse_shaft_px": float(rmse_shaft),
    }


def _fit_wrist_only_pose_then_joints(cad, corr, wrist_uv, wrist_points, K, args):
    wrist_mask = corr.part_names == "wrist"
    shaft_mask = corr.part_names == "shaft"
    gripper_mask = np.isin(corr.part_names, ["l_gripper", "r_gripper"])
    wrist_corr = _subset_corr(corr, wrist_mask)
    shaft_corr = _subset_corr(corr, shaft_mask)
    gripper_corr = _subset_corr(corr, gripper_mask)
    rvec0, trans0, inliers = cmp.solve_wrist_pnp(wrist_uv, wrist_points, K.astype(np.float64), args)

    def residual_wrist(params):
        return cmp.correspondence_residuals(
            cad,
            wrist_corr,
            K,
            args,
            params[:3],
            params[3:6],
            0.0,
            0.0,
            0.0,
        )

    x0 = np.concatenate([rvec0, trans0]).astype(np.float64)
    res_wrist = least_squares(
        residual_wrist,
        x0,
        loss=args.optim_loss,
        f_scale=args.optim_f_scale,
        max_nfev=args.optim_max_nfev,
        verbose=0,
    )
    if not res_wrist.success:
        raise RuntimeError(f"wrist-only pose failed: {res_wrist.message}")
    rvec = res_wrist.x[:3]
    trans = res_wrist.x[3:6]

    joint_lower = np.array(
        [-math.pi / 2.0, -80.0 / 180.0 * math.pi, -80.0 / 180.0 * math.pi],
        dtype=np.float64,
    )
    joint_upper = np.array(
        [math.pi / 2.0, 80.0 / 180.0 * math.pi, 80.0 / 180.0 * math.pi],
        dtype=np.float64,
    )

    def residual_alpha(alpha_arr):
        return cmp.correspondence_residuals(
            cad,
            shaft_corr,
            K,
            args,
            rvec,
            trans,
            float(alpha_arr[0]),
            0.0,
            0.0,
        )

    res_alpha = least_squares(
        residual_alpha,
        np.zeros(1, dtype=np.float64),
        bounds=(joint_lower[:1], joint_upper[:1]),
        loss=args.optim_loss,
        f_scale=args.optim_f_scale,
        max_nfev=args.optim_max_nfev,
        verbose=0,
    )
    if not res_alpha.success:
        raise RuntimeError(f"shaft-alpha failed: {res_alpha.message}")
    alpha = float(res_alpha.x[0])

    def residual_gripper(theta):
        return cmp.correspondence_residuals(
            cad,
            gripper_corr,
            K,
            args,
            rvec,
            trans,
            alpha,
            float(theta[0]),
            float(theta[1]),
        )

    res_gripper = least_squares(
        residual_gripper,
        np.zeros(2, dtype=np.float64),
        bounds=(joint_lower[1:], joint_upper[1:]),
        loss=args.optim_loss,
        f_scale=args.optim_f_scale,
        max_nfev=args.optim_max_nfev,
        verbose=0,
    )
    if not res_gripper.success:
        raise RuntimeError(f"gripper-theta failed: {res_gripper.message}")
    params = np.array(
        [rvec[0], rvec[1], rvec[2], trans[0], trans[1], trans[2], alpha, res_gripper.x[0], res_gripper.x[1]],
        dtype=np.float64,
    )
    stats = _part_residual_stats(cad, corr, params, K, args)
    return {
        "pose": cmp.pose_from_opt_params(params),
        "params": params,
        "pnp_inliers": int(inliers),
        "nfev": int(res_wrist.nfev + res_alpha.nfev + res_gripper.nfev),
        "rmse_all_px": stats["all"]["rmse_px"],
        "rmse_wrist_gripper_px": stats["wrist_gripper"]["rmse_px"],
        "rmse_shaft_px": stats["shaft"]["rmse_px"],
    }


def _add_pose_metric_rows(rows, name, fit, gt_pose, cad, corr, K, args):
    row = {"variant": name, "status": "ok"}
    cmp.add_pose_errors(row, fit["pose"], gt_pose, "pose")
    row.update(
        {
            "pnp_inliers": fit.get("pnp_inliers", ""),
            "nfev": fit.get("nfev", ""),
            "rmse_all_px": fit.get("rmse_all_px", float("nan")),
            "rmse_wrist_gripper_px": fit.get("rmse_wrist_gripper_px", float("nan")),
            "rmse_shaft_px": fit.get("rmse_shaft_px", float("nan")),
            "alpha_deg": math.degrees(float(fit["pose"]["alpha"])),
            "theta_l_deg": math.degrees(float(fit["pose"]["theta_l"])),
            "theta_r_deg": math.degrees(float(fit["pose"]["theta_r"])),
        }
    )
    part_stats = _part_residual_stats(cad, corr, fit["params"], K, args)
    for part_name, stats in part_stats.items():
        for k, v in stats.items():
            row[f"{part_name}_{k}"] = v
    rows.append(row)


def _part_mask_rgb(mask):
    colors = np.array(
        [
            [0, 0, 0],
            [70, 150, 255],
            [40, 210, 120],
            [235, 70, 60],
        ],
        dtype=np.uint8,
    )
    return colors[np.clip(mask.astype(np.int64), 0, 3)]


def _label(panel, title, subtitle=None):
    panel = panel.astype(np.uint8)
    bar = 54 if subtitle else 34
    out = np.full((panel.shape[0] + bar, panel.shape[1], 3), 245, dtype=np.uint8)
    out[bar:] = panel
    cv2.putText(out, str(title), (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 1, cv2.LINE_AA)
    if subtitle:
        cv2.putText(out, str(subtitle), (8, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (30, 30, 30), 1, cv2.LINE_AA)
    return out


def _resize_h(panel, height):
    if panel.shape[0] == height:
        return panel
    width = int(round(panel.shape[1] * float(height) / float(panel.shape[0])))
    return cv2.resize(panel, (width, height), interpolation=cv2.INTER_AREA)


def _hstack(panels, gap=8):
    h = max(p.shape[0] for p in panels)
    resized = [_resize_h(p, h) for p in panels]
    sep = np.full((h, gap, 3), 255, dtype=np.uint8)
    out = []
    for i, p in enumerate(resized):
        if i:
            out.append(sep)
        out.append(p)
    return np.concatenate(out, axis=1)


def _vstack(panels, gap=8):
    w = max(p.shape[1] for p in panels)
    out = []
    for i, p in enumerate(panels):
        if p.shape[1] != w:
            h = p.shape[0]
            canvas = np.full((h, w, 3), 255, dtype=np.uint8)
            canvas[:, : p.shape[1]] = p
            p = canvas
        if i:
            out.append(np.full((gap, w, 3), 255, dtype=np.uint8))
        out.append(p)
    return np.concatenate(out, axis=0)


def _write_csv(path, rows):
    if not rows:
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _decode_dense(out, args):
    inst_prob = torch.sigmoid(out["inst_mask_logits"][0].detach().float().cpu()).numpy()
    part_logits = out["part_mask_logits"][0].detach().float().cpu()
    dense_part = torch.argmax(part_logits, dim=0).numpy().astype(np.int64)
    part_conf = torch.softmax(part_logits, dim=0).max(dim=0).values.numpy().astype(np.float32)
    xyz = cmp.decode_hcce_logits(
        out["hcce_logits"][0].detach().float().cpu(),
        bits=int(args.hcce_bits),
        coord_min=float(args.hcce_coord_min),
        coord_max=float(args.hcce_coord_max),
        threshold=float(args.hcce_bit_thresh),
    ).detach().cpu().numpy()
    return inst_prob, dense_part, part_conf, xyz


def _draw_point(image, uv, color, radius=2):
    h, w = image.shape[:2]
    x = int(round(float(uv[0])))
    y = int(round(float(uv[1])))
    if -20 <= x < w + 20 and -20 <= y < h + 20:
        cv2.circle(image, (x, y), radius, color, -1, lineType=cv2.LINE_AA)


def _draw_pair_connectors(pair, left_labeled, uv_left, uv_right, colors, gap=18, title_bar_height=34, radius=2):
    left_w = left_labeled.shape[1]
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


def _random_part_correspondences(out, cad, pose, target_like, args, part_name, rng, num_points):
    rgb = target_like["orig_rgb"].astype(np.uint8)
    K_orig = target_like["K_orig"].astype(np.float64)
    inst_prob, dense_part, part_conf, xyz = _decode_dense(out, args)
    score = inst_prob * part_conf
    dense_label = {"wrist": 0, "gripper": 1, "shaft": 2}[part_name]
    mask = (inst_prob >= float(args.inst_thresh)) & (dense_part == dense_label) & np.isfinite(xyz).all(axis=2)
    if part_name == "shaft" and getattr(args, "shaft_raw_x_min", None) is not None:
        mask &= xyz[:, :, 0] > float(args.shaft_raw_x_min)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    keep_n = min(int(num_points), len(xs))
    keep = rng.choice(len(xs), size=keep_n, replace=False)
    xs = xs[keep]
    ys = ys[keep]
    scores = score[ys, xs]
    uv_crop = np.stack([xs.astype(np.float64) + 0.5, ys.astype(np.float64) + 0.5], axis=1)
    uv_orig = cmp.crop_points_to_original(uv_crop, target_like["bbox_min"], target_like["scale"], target_like["pad"])
    xyz_norm = xyz[ys, xs].astype(np.float64)
    if part_name == "gripper":
        points_part, transform_names = cmp.closest_gripper_surface_points(cad, xyz_norm, k_faces=int(args.surface_k_faces))
    else:
        transform_name = "wrist" if part_name == "wrist" else "shaft"
        points_part = cmp.closest_part_surface_points(cad, transform_name, xyz_norm, k_faces=int(args.surface_k_faces))
        transform_names = np.array([transform_name] * len(points_part), dtype=object)
    surface_xyz_norm = np.zeros_like(points_part, dtype=np.float64)
    for transform_name in sorted(set(transform_names.tolist())):
        part_mask = transform_names == transform_name
        surface_xyz_norm[part_mask] = (
            cmp._homogeneous_transform(points_part[part_mask], np.linalg.inv(cad.canon_to_part[transform_name]))
            / float(cad.canon_scale)
        )
    params = _pose_to_params(pose)
    transforms = cad.fk(params[:3], params[3:6], params[6], params[7], params[8])
    points_cam = np.zeros_like(points_part, dtype=np.float64)
    for transform_name in sorted(set(transform_names.tolist())):
        part_mask = transform_names == transform_name
        points_cam[part_mask] = cad.transform_points(transform_name, points_part[part_mask], transforms)
    uv_mesh = cmp.project_camera_points(points_cam, K_orig)
    valid_proj = np.isfinite(uv_mesh).all(axis=1)
    if not np.any(valid_proj):
        return None
    return {
        "part_name": part_name,
        "scores": scores[valid_proj],
        "uv_crop": uv_crop[valid_proj],
        "uv_orig": uv_orig[valid_proj],
        "uv_mesh": uv_mesh[valid_proj],
        "xyz_norm": xyz_norm[valid_proj],
        "surface_xyz_norm": surface_xyz_norm[valid_proj],
        "surface_part": transform_names[valid_proj],
    }


def _crop_zoom(panel, uv_a, uv_b, margin=90, max_side=760):
    h, w = panel.shape[:2]
    pts = np.concatenate([uv_a, uv_b], axis=0) if len(uv_a) and len(uv_b) else np.zeros((0, 2))
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


def _make_random_part_correspondence_visuals(out_dir, out, cad, pose, target_like, args, renderer, rng, num_points=24):
    rgb = target_like["orig_rgb"].astype(np.uint8)
    K_orig = target_like["K_orig"].astype(np.float64)

    mesh_color, depth, _ = renderer.render_pose(pose, K_orig, rgb.shape[:2])
    mesh_panel = np.zeros_like(rgb)
    support = np.asarray(depth) > 0
    mesh_panel[support] = mesh_color[support]

    all_rows = []
    full_rows = []
    zoom_rows = []
    for part_name in ("shaft", "wrist", "gripper"):
        sample = _random_part_correspondences(
            out, cad, pose, target_like, args, part_name, rng, int(num_points)
        )
        if sample is None:
            continue
        n = len(sample["uv_orig"])
        colors = []
        for i in range(n):
            hue = int(round(179 * i / max(1, n - 1)))
            bgr = cv2.cvtColor(np.uint8([[[hue, 220, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
            colors.append(tuple(int(x) for x in bgr.tolist()))

        part_mesh = mesh_panel.copy()
        part_rgb = rgb.copy()
        for i, (u_img, u_mesh, sc, xyz_norm, surf_xyz_norm, surf_part) in enumerate(
            zip(
                sample["uv_orig"],
                sample["uv_mesh"],
                sample["scores"],
                sample["xyz_norm"],
                sample["surface_xyz_norm"],
                sample["surface_part"],
            )
        ):
            color = colors[i]
            _draw_point(part_rgb, u_img, color)
            _draw_point(part_mesh, u_mesh, color)
            all_rows.append(
                {
                    "part": part_name,
                    "idx": i,
                    "score": float(sc),
                    "surface_part": str(surf_part),
                    "uv_crop_x": float(sample["uv_crop"][i, 0]),
                    "uv_crop_y": float(sample["uv_crop"][i, 1]),
                    "uv_orig_x": float(u_img[0]),
                    "uv_orig_y": float(u_img[1]),
                    "mesh_proj_x": float(u_mesh[0]),
                    "mesh_proj_y": float(u_mesh[1]),
                    "residual_orig_px": float(np.linalg.norm(u_mesh - u_img)),
                    "pred_xyz_x": float(xyz_norm[0]),
                    "pred_xyz_y": float(xyz_norm[1]),
                    "pred_xyz_z": float(xyz_norm[2]),
                    "surface_norm_x": float(surf_xyz_norm[0]),
                    "surface_norm_y": float(surf_xyz_norm[1]),
                    "surface_norm_z": float(surf_xyz_norm[2]),
                    "passes_pred_x_threshold": bool(
                        part_name != "shaft"
                        or getattr(args, "shaft_raw_x_min", None) is None
                        or float(xyz_norm[0]) > float(args.shaft_raw_x_min)
                    ),
                    "passes_surface_x_threshold": bool(
                        part_name != "shaft"
                        or getattr(args, "shaft_raw_x_min", None) is None
                        or float(surf_xyz_norm[0]) > float(args.shaft_raw_x_min)
                    ),
                }
            )

        subtitle = f"n={n}"
        if part_name == "shaft" and getattr(args, "shaft_raw_x_min", None) is not None:
            subtitle += f", sampled pred HCCE x>{float(args.shaft_raw_x_min):g}"
        full_left = _label(part_mesh, f"mesh: random {part_name} correspondences", subtitle)
        full_right = _label(part_rgb, f"rgb: matched random {part_name} pixels", subtitle)
        full_pair = _hstack([full_left, full_right], gap=18)
        full_pair = _draw_pair_connectors(
            full_pair,
            full_left,
            sample["uv_mesh"],
            sample["uv_orig"],
            colors,
            gap=18,
            title_bar_height=54,
            radius=2,
        )
        Image.fromarray(full_pair).save(out_dir / f"correspondence_random_{part_name}_full.jpg")
        full_rows.append(full_pair)

        zoom_mesh, zoom_offset, zoom_scale = _crop_zoom(part_mesh, sample["uv_mesh"], sample["uv_orig"])
        zoom_rgb, _, _ = _crop_zoom(part_rgb, sample["uv_mesh"], sample["uv_orig"])
        uv_mesh_zoom = (sample["uv_mesh"] - zoom_offset.reshape(1, 2)) * zoom_scale
        uv_orig_zoom = (sample["uv_orig"] - zoom_offset.reshape(1, 2)) * zoom_scale
        zoom_left = _label(zoom_mesh, f"mesh zoom: {part_name}")
        zoom_right = _label(zoom_rgb, f"rgb zoom: {part_name}")
        zoom_pair = _hstack([zoom_left, zoom_right], gap=18)
        zoom_pair = _draw_pair_connectors(
            zoom_pair,
            zoom_left,
            uv_mesh_zoom,
            uv_orig_zoom,
            colors,
            gap=18,
            title_bar_height=34,
            radius=2,
        )
        Image.fromarray(zoom_pair).save(out_dir / f"correspondence_random_{part_name}_zoom.jpg")
        zoom_rows.append(zoom_pair)

    if full_rows:
        Image.fromarray(_vstack(full_rows)).save(out_dir / "correspondence_random_all_parts_full.jpg")
    if zoom_rows:
        Image.fromarray(_vstack(zoom_rows)).save(out_dir / "correspondence_random_all_parts_zoom.jpg")
    _write_csv(out_dir / "correspondence_random_all_parts.csv", all_rows)
    return all_rows


def _make_hcce_error_visuals(out_dir, out, renderer, gt_pose, target_like, args):
    crop_rgb = target_like["crop_rgb"].astype(np.uint8)
    gt_part = target_like["gt_part_crop"].astype(np.uint8)
    inst_prob, dense_part, part_conf, pred_xyz = _decode_dense(out, args)
    pred_part = np.zeros_like(gt_part)
    pred_inst = inst_prob >= float(args.inst_thresh)
    pred_part[pred_inst & (dense_part == 0)] = 2
    pred_part[pred_inst & (dense_part == 1)] = 1
    pred_part[pred_inst & (dense_part == 2)] = 3

    gt_coord_orig = renderer.render_canonical_coords(
        gt_pose,
        target_like["K_orig"],
        target_like["orig_rgb"].shape[:2],
    )
    gt_coord = cmp.crop_resize_pad_map(
        gt_coord_orig.astype(np.float32),
        target_like["bbox_min"],
        target_like["bbox_max"],
        int(args.crop_size),
        cv2.INTER_NEAREST,
        value=0,
    ).astype(np.float32)
    valid = (gt_coord[..., 3] > 0) & np.isfinite(pred_xyz).all(axis=2)
    err = pred_xyz - gt_coord[..., :3]
    err_norm = np.linalg.norm(err, axis=2)

    def heat(values, mask, title, signed=False):
        vals = values.copy()
        if signed:
            lim = np.nanpercentile(np.abs(vals[mask]), 95) if np.any(mask) else 1.0
            lim = max(float(lim), 1e-6)
            norm = np.clip((vals / lim) * 0.5 + 0.5, 0, 1)
        else:
            lim = np.nanpercentile(vals[mask], 95) if np.any(mask) else 1.0
            lim = max(float(lim), 1e-6)
            norm = np.clip(vals / lim, 0, 1)
        img = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img[~mask] = crop_rgb[~mask]
        blend = (0.55 * crop_rgb.astype(np.float32) + 0.45 * img.astype(np.float32)).astype(np.uint8)
        return _label(blend, title)

    panels = [
        _label(crop_rgb, "crop rgb"),
        _label(cmp.overlay_part_mask(crop_rgb, gt_part), "GT part mask"),
        _label(cmp.overlay_part_mask(crop_rgb, pred_part), "pred part mask"),
        heat(err_norm, valid, "HCCE xyz error norm"),
        heat(err_norm, valid & (gt_part == 2), "wrist HCCE error norm"),
        heat(err[..., 0], valid, "signed error x", signed=True),
        heat(err[..., 1], valid, "signed error y", signed=True),
        heat(err[..., 2], valid, "signed error z", signed=True),
    ]
    Image.fromarray(_hstack(panels[:4])).save(out_dir / "hcce_seg_error_row1.jpg")
    Image.fromarray(_hstack(panels[4:])).save(out_dir / "hcce_error_row2.jpg")
    Image.fromarray(_vstack([_hstack(panels[:4]), _hstack(panels[4:])])).save(out_dir / "hcce_error_contactsheet.jpg")

    stats = {}
    for name, label in (("gripper", 1), ("wrist", 2), ("shaft", 3), ("all", -1)):
        mask = valid if label < 0 else (valid & (gt_part == label))
        if not np.any(mask):
            continue
        part_err = err[mask]
        norms = np.linalg.norm(part_err, axis=1)
        bias = part_err.mean(axis=0)
        centered = part_err - bias.reshape(1, 3)
        rmse = float(np.sqrt(np.mean(np.sum(part_err * part_err, axis=1))))
        centered_rmse = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))
        stats[name] = {
            "n": int(mask.sum()),
            "rmse_xyz": rmse,
            "median_norm": float(np.median(norms)),
            "p90_norm": float(np.percentile(norms, 90)),
            "bias_x": float(bias[0]),
            "bias_y": float(bias[1]),
            "bias_z": float(bias[2]),
            "bias_norm": float(np.linalg.norm(bias)),
            "bias_fraction_of_rmse": float(np.linalg.norm(bias) / max(rmse, 1e-12)),
            "centered_rmse_xyz": centered_rmse,
        }
    return stats


def _threshold_boundary(values, valid, threshold):
    values = np.asarray(values, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(values)
    sign = values > float(threshold)
    boundary = np.zeros(valid.shape, dtype=bool)
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        shifted_valid = np.zeros_like(valid)
        shifted_sign = np.zeros_like(sign)
        if dy == 1:
            shifted_valid[1:, :] = valid[:-1, :]
            shifted_sign[1:, :] = sign[:-1, :]
        elif dy == -1:
            shifted_valid[:-1, :] = valid[1:, :]
            shifted_sign[:-1, :] = sign[1:, :]
        elif dx == 1:
            shifted_valid[:, 1:] = valid[:, :-1]
            shifted_sign[:, 1:] = sign[:, :-1]
        else:
            shifted_valid[:, :-1] = valid[:, 1:]
            shifted_sign[:, :-1] = sign[:, 1:]
        boundary |= valid & shifted_valid & (sign != shifted_sign)
    return boundary


def _draw_orig_boundary(panel, boundary, color, thickness=2):
    if not np.any(boundary):
        return panel
    kernel = np.ones((int(thickness), int(thickness)), dtype=np.uint8)
    mask = cv2.dilate(boundary.astype(np.uint8), kernel, iterations=1) > 0
    panel[mask] = np.asarray(color, dtype=np.uint8)
    return panel


def _draw_crop_points_on_orig(panel, mask_crop, target_like, color, radius=1):
    ys, xs = np.where(mask_crop)
    if len(xs) == 0:
        return panel
    uv_crop = np.stack([xs.astype(np.float64) + 0.5, ys.astype(np.float64) + 0.5], axis=1)
    uv_orig = cmp.crop_points_to_original(
        uv_crop,
        target_like["bbox_min"],
        target_like["scale"],
        target_like["pad"],
    )
    h, w = panel.shape[:2]
    for xy in uv_orig:
        x, y = np.round(xy).astype(int)
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(panel, (int(x), int(y)), int(radius), color, -1, lineType=cv2.LINE_AA)
    return panel


def _value_to_rgb(values, vmin=-1.0, vmax=1.0, cmap=cv2.COLORMAP_TURBO):
    values = np.asarray(values)
    norm = np.clip((values.astype(np.float32) - float(vmin)) / max(float(vmax) - float(vmin), 1e-6), 0, 1)
    colors = cv2.applyColorMap((norm * 255).astype(np.uint8), cmap)
    colors = cv2.cvtColor(colors, cv2.COLOR_BGR2RGB)
    if values.ndim == 1 and colors.ndim == 3 and colors.shape[1] == 1:
        colors = colors[:, 0, :]
    return colors


def _heat_overlay_orig(rgb, values, mask, title, subtitle=None, vmin=-1.0, vmax=1.0, boundaries=None, alpha=0.68):
    valid = np.asarray(mask, dtype=bool) & np.isfinite(values)
    heat = _value_to_rgb(values, vmin=vmin, vmax=vmax)
    out = rgb.copy()
    if np.any(valid):
        out[valid] = (
            (1.0 - float(alpha)) * out[valid].astype(np.float32)
            + float(alpha) * heat[valid].astype(np.float32)
        ).clip(0, 255).astype(np.uint8)
    for boundary, color, thickness in boundaries or []:
        _draw_orig_boundary(out, boundary, color, thickness=thickness)
    return _label(out, title, subtitle)


def _heat_overlay_crop_scatter(rgb, target_like, values_crop, mask_crop, title, subtitle=None, vmin=-1.0, vmax=1.0, boundaries=None):
    valid = np.asarray(mask_crop, dtype=bool) & np.isfinite(values_crop)
    ys, xs = np.where(valid)
    out = rgb.copy()
    if len(xs) > 0:
        uv_crop = np.stack([xs.astype(np.float64) + 0.5, ys.astype(np.float64) + 0.5], axis=1)
        uv_orig = cmp.crop_points_to_original(
            uv_crop,
            target_like["bbox_min"],
            target_like["scale"],
            target_like["pad"],
        )
        colors = _value_to_rgb(values_crop[ys, xs], vmin=vmin, vmax=vmax)
        h, w = out.shape[:2]
        color_layer = np.zeros_like(out)
        alpha_mask = np.zeros((h, w), dtype=np.uint8)
        radius = max(1, int(round(0.65 / max(float(np.mean(target_like["scale"])), 1e-6))))
        for xy, color in zip(uv_orig, colors):
            x, y = np.round(xy).astype(int)
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(color_layer, (int(x), int(y)), radius, tuple(int(v) for v in color), -1, lineType=cv2.LINE_AA)
                cv2.circle(alpha_mask, (int(x), int(y)), radius, 255, -1, lineType=cv2.LINE_AA)
        mask = alpha_mask > 0
        out[mask] = (0.28 * out[mask].astype(np.float32) + 0.72 * color_layer[mask].astype(np.float32)).clip(0, 255).astype(np.uint8)
    for boundary, color, thickness in boundaries or []:
        if boundary.shape == rgb.shape[:2]:
            _draw_orig_boundary(out, boundary, color, thickness=thickness)
        else:
            _draw_crop_points_on_orig(out, boundary, target_like, color, radius=max(1, int(thickness)))
    return _label(out, title, subtitle)


def _binary_selection_overlay(rgb, target_like, gt_select_crop, pred_select_crop, title, boundaries=None):
    out = rgb.copy()
    both = gt_select_crop & pred_select_crop
    gt_only = gt_select_crop & ~pred_select_crop
    pred_only = pred_select_crop & ~gt_select_crop
    _draw_crop_points_on_orig(out, gt_only, target_like, (40, 230, 70), radius=1)
    _draw_crop_points_on_orig(out, pred_only, target_like, (255, 70, 60), radius=1)
    _draw_crop_points_on_orig(out, both, target_like, (255, 230, 40), radius=1)
    for boundary, color, thickness in boundaries or []:
        if boundary.shape == rgb.shape[:2]:
            _draw_orig_boundary(out, boundary, color, thickness=thickness)
        else:
            _draw_crop_points_on_orig(out, boundary, target_like, color, radius=max(1, int(thickness)))
    return _label(out, title, "green=GT x>-0.5, red=pred x>-0.5, yellow=overlap")


def _array_stats(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"count": 0}
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std()),
        "min": float(values.min()),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
        "max": float(values.max()),
    }


def _make_shaft_hcce_map_visuals(out_dir, out, renderer, gt_pose, target_like, args, model_meta):
    rgb = target_like["orig_rgb"].astype(np.uint8)
    K_orig = target_like["K_orig"].astype(np.float64)
    gt_part_crop = target_like["gt_part_crop"].astype(np.uint8)
    gt_part_orig = target_like["gt_part_orig"].astype(np.uint8)
    threshold = float(args.shaft_raw_x_min)

    inst_prob, dense_part, part_conf, pred_xyz = _decode_dense(out, args)
    pred_x = pred_xyz[..., 0].astype(np.float32)
    pred_y = pred_xyz[..., 1].astype(np.float32)
    pred_z = pred_xyz[..., 2].astype(np.float32)
    pred_inst = inst_prob >= float(args.inst_thresh)
    pred_shaft = pred_inst & (dense_part == 2) & np.isfinite(pred_xyz).all(axis=2)
    gt_shaft_crop = gt_part_crop == 3

    gt_coord_orig = renderer.render_canonical_coords(gt_pose, K_orig, rgb.shape[:2])
    gt_coord_crop = cmp.crop_resize_pad_map(
        gt_coord_orig.astype(np.float32),
        target_like["bbox_min"],
        target_like["bbox_max"],
        int(args.crop_size),
        cv2.INTER_NEAREST,
        value=0,
    ).astype(np.float32)
    gt_x_crop = gt_coord_crop[..., 0]
    gt_y_crop = gt_coord_crop[..., 1]
    gt_z_crop = gt_coord_crop[..., 2]
    gt_valid_crop = (gt_coord_crop[..., 3] > 0) & np.isfinite(gt_coord_crop[..., :3]).all(axis=2)
    eval_mask = gt_shaft_crop & gt_valid_crop & np.isfinite(pred_xyz).all(axis=2)

    gt_valid_orig = (gt_coord_orig[..., 3] > 0) & (gt_part_orig == 3) & np.isfinite(gt_coord_orig[..., 0])
    gt_boundary_orig = _threshold_boundary(gt_coord_orig[..., 0], gt_valid_orig, threshold)
    pred_boundary_crop = _threshold_boundary(pred_x, pred_shaft, threshold)
    gt_boundary_crop = _threshold_boundary(gt_x_crop, gt_shaft_crop & gt_valid_crop, threshold)
    boundaries = [
        (gt_boundary_orig, (255, 230, 30), 2),
        (pred_boundary_crop, (20, 230, 255), 2),
    ]

    pred_on_gt = eval_mask
    pred_on_pred = pred_shaft
    gt_select_crop = eval_mask & (gt_x_crop > threshold)
    pred_select_crop = pred_shaft & (pred_x > threshold)
    pred_select_on_eval = eval_mask & (pred_x > threshold)

    rgb_boundary = rgb.copy()
    _draw_orig_boundary(rgb_boundary, gt_boundary_orig, (255, 230, 30), thickness=2)
    _draw_crop_points_on_orig(rgb_boundary, pred_boundary_crop, target_like, (20, 230, 255), radius=2)
    cv2.putText(rgb_boundary, "yellow: GT x=-0.5", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 230, 30), 2, cv2.LINE_AA)
    cv2.putText(rgb_boundary, "cyan: pred x=-0.5", (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 230, 255), 2, cv2.LINE_AA)

    gt_x_orig_panel = _heat_overlay_orig(
        rgb,
        gt_coord_orig[..., 0],
        gt_valid_orig,
        "GT shaft HCCE x on RGB",
        "yellow=GT x=-0.5",
        boundaries=[(gt_boundary_orig, (255, 230, 30), 2)],
    )
    pred_x_gt_panel = _heat_overlay_crop_scatter(
        rgb,
        target_like,
        pred_x,
        pred_on_gt,
        "pred HCCE x sampled on GT shaft",
        "yellow=GT x=-0.5, cyan=pred x=-0.5",
        boundaries=boundaries,
    )
    pred_x_pred_panel = _heat_overlay_crop_scatter(
        rgb,
        target_like,
        pred_x,
        pred_on_pred,
        "pred HCCE x on predicted shaft",
        "this is the fitting candidate field",
        boundaries=boundaries,
    )
    err_x = pred_x - gt_x_crop
    err_panel = _heat_overlay_crop_scatter(
        rgb,
        target_like,
        err_x,
        eval_mask,
        "pred x - GT x on GT shaft",
        "blue/green low, yellow/red high",
        vmin=-0.8,
        vmax=0.8,
        boundaries=boundaries,
    )
    select_panel = _binary_selection_overlay(
        rgb,
        target_like,
        gt_select_crop,
        pred_select_on_eval,
        "threshold x > -0.5 on GT shaft",
        boundaries=boundaries,
    )
    fit_select_panel = _binary_selection_overlay(
        rgb,
        target_like,
        gt_select_crop,
        pred_select_crop,
        "fitting shaft selection vs GT front-half",
        boundaries=boundaries,
    )

    row1 = _hstack([
        _label(rgb_boundary, "RGB with x=-0.5 contours"),
        gt_x_orig_panel,
        pred_x_gt_panel,
    ])
    row2 = _hstack([pred_x_pred_panel, err_panel, select_panel])
    row3 = _hstack([
        fit_select_panel,
        _heat_overlay_crop_scatter(rgb, target_like, pred_y, pred_on_gt, "pred HCCE y on GT shaft", vmin=-1, vmax=1, boundaries=boundaries),
        _heat_overlay_crop_scatter(rgb, target_like, pred_z, pred_on_gt, "pred HCCE z on GT shaft", vmin=-1, vmax=1, boundaries=boundaries),
    ])
    Image.fromarray(_vstack([row1, row2, row3])).save(out_dir / "shaft_hcce_x_threshold_contactsheet.jpg")

    xyz_rows = [
        _hstack([
            _heat_overlay_crop_scatter(rgb, target_like, gt_x_crop, eval_mask, "GT x on GT shaft", boundaries=[(gt_boundary_orig, (255, 230, 30), 2)]),
            _heat_overlay_crop_scatter(rgb, target_like, gt_y_crop, eval_mask, "GT y on GT shaft"),
            _heat_overlay_crop_scatter(rgb, target_like, gt_z_crop, eval_mask, "GT z on GT shaft"),
        ]),
        _hstack([
            _heat_overlay_crop_scatter(rgb, target_like, pred_x, eval_mask, "pred x on GT shaft", boundaries=boundaries),
            _heat_overlay_crop_scatter(rgb, target_like, pred_y, eval_mask, "pred y on GT shaft"),
            _heat_overlay_crop_scatter(rgb, target_like, pred_z, eval_mask, "pred z on GT shaft"),
        ]),
        _hstack([
            _heat_overlay_crop_scatter(rgb, target_like, pred_x - gt_x_crop, eval_mask, "error x", vmin=-0.8, vmax=0.8),
            _heat_overlay_crop_scatter(rgb, target_like, pred_y - gt_y_crop, eval_mask, "error y", vmin=-0.8, vmax=0.8),
            _heat_overlay_crop_scatter(rgb, target_like, pred_z - gt_z_crop, eval_mask, "error z", vmin=-0.8, vmax=0.8),
        ]),
    ]
    Image.fromarray(_vstack(xyz_rows)).save(out_dir / "shaft_hcce_xyz_maps_contactsheet.jpg")

    def _safe_rate(num, den):
        return float(num) / float(den) if int(den) > 0 else float("nan")

    gt_pos = eval_mask & (gt_x_crop > threshold)
    pred_pos = eval_mask & (pred_x > threshold)
    tp = int(np.count_nonzero(gt_pos & pred_pos))
    fp = int(np.count_nonzero(~gt_pos & pred_pos & eval_mask))
    fn = int(np.count_nonzero(gt_pos & ~pred_pos & eval_mask))
    tn = int(np.count_nonzero(~gt_pos & ~pred_pos & eval_mask))
    selected_gt_parts = gt_part_crop[pred_select_crop]
    selected_gt_part_counts = {str(int(k)): int(v) for k, v in zip(*np.unique(selected_gt_parts, return_counts=True))} if selected_gt_parts.size else {}
    pred_shaft_gt_parts = gt_part_crop[pred_shaft]
    pred_shaft_gt_part_counts = {str(int(k)): int(v) for k, v in zip(*np.unique(pred_shaft_gt_parts, return_counts=True))} if pred_shaft_gt_parts.size else {}
    stats = {
        "threshold_x": threshold,
        "axis_scale": cmp.hcce_axis_scale_from_args(model_meta, args).astype(float).tolist(),
        "eval_gt_shaft_pixels": int(np.count_nonzero(eval_mask)),
        "pred_shaft_pixels": int(np.count_nonzero(pred_shaft)),
        "pred_selected_pixels": int(np.count_nonzero(pred_select_crop)),
        "pred_shaft_gt_part_counts": pred_shaft_gt_part_counts,
        "pred_selected_gt_part_counts": selected_gt_part_counts,
        "gt_x_stats_on_gt_shaft": _array_stats(gt_x_crop[eval_mask]),
        "pred_x_stats_on_gt_shaft": _array_stats(pred_x[eval_mask]),
        "pred_x_stats_on_pred_shaft": _array_stats(pred_x[pred_shaft]),
        "x_error_stats_on_gt_shaft": _array_stats(err_x[eval_mask]),
        "threshold_eval": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": _safe_rate(tp, tp + fp),
            "recall": _safe_rate(tp, tp + fn),
            "iou": _safe_rate(tp, tp + fp + fn),
            "accuracy": _safe_rate(tp + tn, tp + fp + fn + tn),
        },
        "pred_selected_overlap_gt_shaft": int(np.count_nonzero(pred_select_crop & eval_mask)),
        "pred_selected_gt_x_le_threshold": int(np.count_nonzero(pred_select_crop & eval_mask & (gt_x_crop <= threshold))),
        "gt_front_missed_by_pred_threshold": int(np.count_nonzero(gt_pos & ~pred_pos)),
    }
    if np.count_nonzero(eval_mask) > 2:
        stats["pred_gt_x_corr_on_gt_shaft"] = float(np.corrcoef(pred_x[eval_mask].reshape(-1), gt_x_crop[eval_mask].reshape(-1))[0, 1])
    else:
        stats["pred_gt_x_corr_on_gt_shaft"] = float("nan")
    (out_dir / "shaft_hcce_threshold_stats.json").write_text(json.dumps(stats, indent=2, allow_nan=True), encoding="utf-8")
    return stats


def _make_pose_overlay_sheet(out_dir, renderer, target_like, variants, gt_pose):
    rgb = target_like["orig_rgb"].astype(np.uint8)
    K = target_like["K_orig"]
    panels = [_label(renderer.render_pose_overlay(rgb, gt_pose, K, alpha=0.85), "GT pose")]
    for name, fit in variants.items():
        pose = fit.get("pose")
        subtitle = f"t={fit.get('trans_err', float('nan')):.4f} R={fit.get('rot_err', float('nan')):.1f} J={fit.get('joint_err', float('nan')):.1f}"
        panels.append(_label(renderer.render_pose_overlay(rgb, pose, K, alpha=0.85), name, subtitle))
    rows = []
    for i in range(0, len(panels), 3):
        rows.append(_hstack(panels[i : i + 3]))
    Image.fromarray(_vstack(rows)).save(out_dir / "pose_variant_overlays.jpg")


def _make_shaft_compare_sheet(out_dir, renderer, target_like, fits, gt_pose):
    rgb = target_like["orig_rgb"].astype(np.uint8)
    K = target_like["K_orig"]

    def panel_for_pose(title, pose, fit=None):
        subtitle = None
        if fit is not None:
            subtitle = "t={:.4f}m R={:.1f}deg J={:.1f}deg".format(
                float(fit.get("trans_err", float("nan"))),
                float(fit.get("rot_err", float("nan"))),
                float(fit.get("joint_err", float("nan"))),
            )
        return _label(renderer.render_pose_overlay(rgb, pose, K, alpha=0.85), title, subtitle)

    panels = [
        _label(rgb, "RGB input"),
        panel_for_pose("GT pose", gt_pose),
        panel_for_pose(
            "fit: shaft fixed for wrist pose",
            fits["current_decoupled_wg_pose_shaft_alpha"]["pose"],
            fits["current_decoupled_wg_pose_shaft_alpha"],
        ),
        panel_for_pose(
            "fit: shaft participates in wrist pose",
            fits["single_all_shaft_affects_wrist"]["pose"],
            fits["single_all_shaft_affects_wrist"],
        ),
    ]
    Image.fromarray(_hstack(panels, gap=18)).save(out_dir / "shaft_fixed_vs_shaft_optim_gt_concat.jpg")


def run(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = cmp.configure_device(args.device)
    base_args = cmp.build_parser().parse_args([])
    base_args.output_dir = str(out_dir)
    base_args.robopepp_checkpoint = str(args.robopepp_checkpoint)
    base_args.hcce_checkpoint = str(args.hcce_checkpoint)
    base_args.needle_manifest_csv = str(args.needle_manifest_csv)
    base_args.dataset_cache_dir = str(args.dataset_cache_dir)
    base_args.datasets = ["needleGrasping"]
    base_args.max_needle_samples = 0
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
    items, _ = cmp.build_needle_items(base_args)
    item = None
    for candidate in items:
        if candidate.video == args.video and candidate.frame_id == args.frame_id and int(candidate.instance_id) == int(args.instance_id):
            item = candidate
            break
    if item is None:
        raise RuntimeError(f"Could not find {args.video}/{args.frame_id}/inst{args.instance_id}")

    image, target = dataset[item.dataset_idx]
    target_like = cmp.target_like_from_needle(target)
    gt_pose = cmp.pose_from_target(target)
    model, model_meta = cmp.load_hcce_model(args.hcce_checkpoint, device)
    to_tensor = tv_transforms.Compose(
        [
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    x = image.unsqueeze(0).to(device)
    K_crop_t = torch.from_numpy(target_like["K_crop"]).unsqueeze(0).to(device)
    with torch.inference_mode(), torch.amp.autocast(device_type="cuda", enabled=device.type == "cuda", dtype=torch.bfloat16):
        out = model(x, K_crop_t)

    cad = cmp.InstrumentCAD(cmp.CAD_ROOT)
    renderer = GMSInstrumentTrimeshRenderer(device)
    rng = np.random.default_rng(int(args.seed))

    pred_args = copy.copy(base_args)
    pred_args.fit_seg_source = "pred"
    corr, wrist_uv, wrist_points, counts = cmp.build_crop_hcce_correspondences(
        out, cad, model_meta, target_like, pred_args, rng
    )

    gtseg_args = copy.copy(base_args)
    gtseg_args.fit_seg_source = "gt"
    gt_corr, gt_wrist_uv, gt_wrist_points, gt_counts = cmp.build_crop_hcce_correspondences(
        out, cad, model_meta, target_like, gtseg_args, np.random.default_rng(int(args.seed))
    )

    fits = {}
    fits["current_decoupled_wg_pose_shaft_alpha"] = _fit_with_builtin(
        cad, corr, wrist_uv, wrist_points, target_like["K_crop"], pred_args, "decoupled", "wrist_gripper"
    )
    fits["single_all_shaft_affects_wrist"] = _fit_with_builtin(
        cad, corr, wrist_uv, wrist_points, target_like["K_crop"], pred_args, "single", "all"
    )
    fits["single_wrist_gripper_no_shaft"] = _fit_with_builtin(
        cad, corr, wrist_uv, wrist_points, target_like["K_crop"], pred_args, "single", "wrist_gripper"
    )
    fits["wrist_only_pose_then_shaft_alpha_gripper_theta"] = _fit_wrist_only_pose_then_joints(
        cad, corr, wrist_uv, wrist_points, target_like["K_crop"], pred_args
    )
    fits["gtseg_current_decoupled"] = _fit_with_builtin(
        cad, gt_corr, gt_wrist_uv, gt_wrist_points, target_like["K_crop"], gtseg_args, "decoupled", "wrist_gripper"
    )

    direct_pose = cmp.pose_from_output(out)
    direct_params = _pose_to_params(direct_pose)
    fits["hcce_direct_head"] = {
        "pose": direct_pose,
        "params": direct_params,
        "pnp_inliers": "",
        "nfev": "",
        "rmse_all_px": _part_residual_stats(cad, corr, direct_params, target_like["K_crop"], pred_args)["all"]["rmse_px"],
        "rmse_wrist_gripper_px": _part_residual_stats(cad, corr, direct_params, target_like["K_crop"], pred_args)["wrist_gripper"]["rmse_px"],
        "rmse_shaft_px": _part_residual_stats(cad, corr, direct_params, target_like["K_crop"], pred_args)["shaft"]["rmse_px"],
    }

    rows = []
    for name, fit in fits.items():
        use_corr = gt_corr if name.startswith("gtseg") else corr
        use_args = gtseg_args if name.startswith("gtseg") else pred_args
        _add_pose_metric_rows(rows, name, fit, gt_pose, cad, use_corr, target_like["K_crop"], use_args)
        fit["trans_err"] = rows[-1]["pose_trans_err_m"]
        fit["rot_err"] = rows[-1]["pose_rot_err_deg"]
        fit["joint_err"] = rows[-1]["pose_joint_mae_deg"]

    _write_csv(out_dir / "pose_variant_metrics.csv", rows)
    (out_dir / "pose_variant_metrics.json").write_text(json.dumps(rows, indent=2, allow_nan=True), encoding="utf-8")

    random_corr_rows = _make_random_part_correspondence_visuals(
        out_dir,
        out,
        cad,
        fits["current_decoupled_wg_pose_shaft_alpha"]["pose"],
        target_like,
        pred_args,
        renderer,
        np.random.default_rng(int(args.seed) + 17),
        num_points=int(args.corr_points_per_part),
    )
    hcce_stats = _make_hcce_error_visuals(out_dir, out, renderer, gt_pose, target_like, pred_args)
    shaft_hcce_stats = _make_shaft_hcce_map_visuals(
        out_dir,
        out,
        renderer,
        gt_pose,
        target_like,
        pred_args,
        model_meta,
    )
    _make_pose_overlay_sheet(out_dir, renderer, target_like, fits, gt_pose)
    _make_shaft_compare_sheet(out_dir, renderer, target_like, fits, gt_pose)

    # Crop segmentation metrics and masks.
    pred_part = cmp.model_part_mask_crop(out, pred_args.inst_thresh)
    seg_row = {}
    cmp.add_model_crop_segmentation(seg_row, out, target_like, pred_args)
    Image.fromarray(_hstack([
        _label(target_like["crop_rgb"], "crop rgb"),
        _label(cmp.overlay_part_mask(target_like["crop_rgb"], target_like["gt_part_crop"]), "GT part crop"),
        _label(cmp.overlay_part_mask(target_like["crop_rgb"], pred_part), "pred part crop"),
    ])).save(out_dir / "part_segmentation_crop_debug.jpg")

    summary = {
        "frame": {
            "video": item.video,
            "frame_id": item.frame_id,
            "instance_id": int(item.instance_id),
            "dataset_idx": int(item.dataset_idx),
        },
        "config": {
            "surface_k_faces": int(pred_args.surface_k_faces),
            "shaft_raw_x_min": float(pred_args.shaft_raw_x_min),
            "point_select": pred_args.point_select,
            "fit_seg_source": pred_args.fit_seg_source,
        },
        "counts_predseg": counts,
        "counts_gtseg": gt_counts,
        "segmentation_metrics": seg_row,
        "hcce_xyz_error_stats": hcce_stats,
        "shaft_hcce_threshold_stats": shaft_hcce_stats,
        "random_correspondence_mean_residual_orig_px": float(np.mean([r["residual_orig_px"] for r in random_corr_rows])) if random_corr_rows else float("nan"),
        "random_correspondence_p90_residual_orig_px": float(np.percentile([r["residual_orig_px"] for r in random_corr_rows], 90)) if random_corr_rows else float("nan"),
    }
    (out_dir / "debug_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")

    md = [
        f"# HCCE Debug {item.video} {item.frame_id} inst{item.instance_id}",
        "",
        f"- surface_k_faces: {pred_args.surface_k_faces}",
        f"- shaft_raw_x_min: {pred_args.shaft_raw_x_min}",
        f"- predseg counts: {counts}",
        f"- gtseg counts: {gt_counts}",
        f"- segmentation: {seg_row}",
        "",
        "## Pose Variants",
        "",
        "| variant | trans m | rot deg | joint deg | all reproj px | wrist rmse | shaft rmse | gripper rmse |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        md.append(
            "| {variant} | {t:.6f} | {r:.3f} | {j:.3f} | {all:.3f} | {wr:.3f} | {sh:.3f} | {gr:.3f} |".format(
                variant=row["variant"],
                t=float(row["pose_trans_err_m"]),
                r=float(row["pose_rot_err_deg"]),
                j=float(row["pose_joint_mae_deg"]),
                all=float(row["all_rmse_px"]),
                wr=float(row["wrist_rmse_px"]),
                sh=float(row["shaft_rmse_px"]),
                gr=float(row["gripper_rmse_px"]),
            )
        )
    md.extend([
        "",
        "## HCCE XYZ Error",
        "",
        "```json",
        json.dumps(hcce_stats, indent=2, allow_nan=True),
        "```",
        "",
        "## Shaft HCCE Threshold",
        "",
        "```json",
        json.dumps(shaft_hcce_stats, indent=2, allow_nan=True),
        "```",
    ])
    (out_dir / "debug_summary.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[ok] wrote {out_dir}")


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=str(cmp.ROBOPEPP_ROOT / "logs/debug_hcce_needle00008_inst2"))
    parser.add_argument("--video", type=str, default="needleGrasping_971_video38")
    parser.add_argument("--frame_id", type=str, default="00008")
    parser.add_argument("--instance_id", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--robopepp_checkpoint", type=str, default=str(cmp.DEFAULT_ROBOPEPP_CKPT))
    parser.add_argument(
        "--hcce_checkpoint",
        type=str,
        default=str(cmp.ROBOPEPP_ROOT / "logs/hcce_crop224_fixbf16_rarp_gpu0123_bs56_fromscratch/checkpoints/iter0022000.pt"),
    )
    parser.add_argument("--needle_manifest_csv", type=str, default=str(cmp.DEFAULT_NEEDLE_MANIFEST))
    parser.add_argument(
        "--dataset_cache_dir",
        type=str,
        default=str(cmp.ROBOPEPP_ROOT / "logs/crop_hcce_vs_robopepp_hccefixbf16_last_ng_suture10_stride4/dataset_cache"),
    )
    parser.add_argument("--surface_k_faces", type=int, default=0)
    parser.add_argument("--shaft_raw_x_min", type=float, default=-0.5)
    parser.add_argument("--corr_points_per_part", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    return parser


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
