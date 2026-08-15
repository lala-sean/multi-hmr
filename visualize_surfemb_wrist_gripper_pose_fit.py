#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np
import pyrender
import torch
import torch.nn.functional as F
import trimesh
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

from eval_surfemb_articulated_rarp import (
    DEFAULT_MODELS,
    build_dataset,
    canonicalize_prediction,
    load_model,
    make_surfemb_crop,
    parse_model_specs,
    target_pose,
)
from instrument_geometry import (
    GRIPPER_JOINT_OFFSET_M,
    SURFEMB_GRIPPER_STATIC_THRESHOLD_M,
    fk_matrices_np,
    make_transform_np,
    quat_wxyz_to_matrix_np,
)
from instrument_trimesh_renderer import GMSInstrumentTrimeshRenderer
from surfemb_articulated_pose import (
    downsample_intrinsics,
    encode_surface_keys,
    estimate_chain_joint_angle,
    estimate_part_pose_from_context,
    load_part_surfaces,
    prepare_part_score_context,
)


ROBOPEPP_ROOT = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = ROBOPEPP_ROOT / "logs" / "surfemb_wrist_gripper_pose_fit_vis"
DEFAULT_CANDIDATES = "0,124,249,317,422,433,438,624,749,874,999,1124,1374,1572,1573,1578,1874,2264"
WG_PARTS = ("wrist", "l_gripper", "r_gripper")
PART_COLORS = {
    "wrist": (55, 225, 100, 255),
    "l_gripper": (55, 210, 250, 255),
    "r_gripper": (245, 80, 190, 255),
}


def _font(size):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _matrix_to_quat_wxyz(matrix):
    xyzw = Rotation.from_matrix(np.asarray(matrix, dtype=np.float64)).as_quat()
    quat = np.asarray([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64)
    return -quat if quat[0] < 0.0 else quat


def _rotation_error_deg(q_pred, q_gt):
    relative = quat_wxyz_to_matrix_np(q_pred) @ quat_wxyz_to_matrix_np(q_gt).T
    cosine = np.clip((float(np.trace(relative)) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _angle_error_deg(pred, gt):
    delta = math.atan2(math.sin(float(pred) - float(gt)), math.cos(float(pred) - float(gt)))
    return abs(math.degrees(delta))


def _probability_input(prob, h, w):
    prob = prob.reshape(-1).float().clamp(1e-7, 1.0 - 1e-7)
    log_prob = prob.log().reshape(h, w)
    neg_log_prob = torch.log1p(-prob).reshape(h, w)
    return {
        "prob": prob,
        "mask_log_prob": F.max_pool2d(log_prob[None, None], 3, 1, 1)[0, 0].reshape(-1),
        "neg_mask_log_prob": F.max_pool2d(neg_log_prob[None, None], 3, 1, 1)[0, 0].reshape(-1),
    }


@torch.inference_mode()
def build_wrist_gripper_contexts(mask_logits, query_chw, surfaces, K_crop, part_crop, down_sample_scale):
    del mask_logits  # SAM part masks define the W/G fit ROI for this diagnostic.
    scale = int(down_sample_scale)
    query = F.avg_pool2d(query_chw[None].float(), scale)[0]
    _, h, w = query.shape
    query_flat = query.permute(1, 2, 0).reshape(h * w, -1)

    ys = np.arange(h, dtype=np.int64) * scale + scale // 2
    xs = np.arange(w, dtype=np.int64) * scale + scale // 2
    part_ds = np.asarray(part_crop, dtype=np.uint8)[np.ix_(ys, xs)]
    wrist_roi = torch.from_numpy((part_ds == 2).reshape(-1)).to(query_flat.device)
    gripper_roi = torch.from_numpy((part_ds == 1).reshape(-1)).to(query_flat.device)

    wrist_prob = torch.where(
        wrist_roi,
        torch.full_like(wrist_roi, 0.995, dtype=torch.float32),
        torch.full_like(wrist_roi, 1e-7, dtype=torch.float32),
    )
    gripper_scores = []
    for name in ("l_gripper", "r_gripper"):
        keys = surfaces[name].mask_keys
        gripper_scores.append(torch.logsumexp(query_flat @ keys.T, dim=1) - math.log(float(len(keys))))
    gripper_split = torch.softmax(torch.stack(gripper_scores, dim=1), dim=1)
    gripper_support = gripper_roi.float()[:, None]
    gripper_prob = (gripper_support * gripper_split).clamp_min(1e-7)

    probability_inputs = {
        "wrist": _probability_input(wrist_prob, h, w),
        "l_gripper": _probability_input(gripper_prob[:, 0], h, w),
        "r_gripper": _probability_input(gripper_prob[:, 1], h, w),
    }
    contexts = {
        name: prepare_part_score_context(query_flat, probability_inputs[name], surfaces[name], (h, w))
        for name in WG_PARTS
    }
    visibility = {
        "wrist": int(wrist_roi.sum().item()),
        "gripper": int(gripper_roi.sum().item()),
    }
    return contexts, downsample_intrinsics(K_crop, scale), (h, w), visibility


@torch.inference_mode()
def fit_wrist_gripper_pose(model, surfaces, image, K_crop, part_crop, args, seed):
    device = next(model.parameters()).device
    x = image[None].to(device=device, non_blocking=True)
    K_tensor = torch.from_numpy(K_crop)[None].to(device=device, non_blocking=True)
    with torch.amp.autocast(
        device_type=device.type,
        enabled=device.type == "cuda" and bool(args.amp),
        dtype=torch.bfloat16,
    ):
        output = model(x, K_tensor)
    contexts, K_ds, image_hw, visibility = build_wrist_gripper_contexts(
        output["inst_mask_logits"][0].float(),
        output["surfemb_queries"][0].float(),
        surfaces,
        K_crop,
        part_crop,
        args.down_sample_scale,
    )
    if visibility["wrist"] < int(args.min_wrist_pixels_ds):
        raise RuntimeError(f"Only {visibility['wrist']} downsampled wrist pixels in SAM ROI")

    wrist_hypotheses = estimate_part_pose_from_context(
        contexts["wrist"],
        surfaces["wrist"],
        K_ds,
        image_hw,
        max_poses=args.max_poses,
        max_pose_evaluations=args.max_pose_evaluations,
        pose_batch_size=args.pose_batch_size,
        top_k=args.top_k,
        alpha=args.corr_alpha,
        dist_2d_min=args.dist_2d_min,
        seed=seed,
    )
    if not wrist_hypotheses:
        raise RuntimeError("No valid wrist AP3P hypothesis")

    fit_gripper = visibility["gripper"] >= int(args.min_gripper_pixels_ds)
    best = None
    for rank, wrist_hypothesis in enumerate(wrist_hypotheses):
        joint_results = {}
        if fit_gripper:
            try:
                for name in ("l_gripper", "r_gripper"):
                    joint_results[name] = estimate_chain_joint_angle(
                        contexts[name],
                        wrist_hypothesis.transform,
                        name,
                        K_ds,
                        image_hw,
                        pose_batch_size=args.pose_batch_size,
                        coarse_steps=args.coarse_steps,
                        fine_steps=args.fine_steps,
                    )
            except RuntimeError:
                continue
        wrist_weight = 0.75 if fit_gripper else 1.0
        gripper_weight = 0.125 if fit_gripper else 0.0
        total_score = wrist_weight * float(wrist_hypothesis.score)
        total_score += gripper_weight * sum(float(value["score"]) for value in joint_results.values())
        if best is None or total_score > best[0]:
            best = (total_score, rank, wrist_hypothesis, joint_results)
    if best is None:
        raise RuntimeError("Every wrist hypothesis failed W/G chain scoring")

    total_score, rank, wrist_hypothesis, joint_results = best
    wrist = wrist_hypothesis.transform
    pose = {
        "rot": _matrix_to_quat_wxyz(wrist[:3, :3]),
        "trans": wrist[:3, 3].copy(),
        "alpha": 0.0,
        "theta_l": float(joint_results.get("l_gripper", {}).get("angle", 0.0)),
        "theta_r": float(joint_results.get("r_gripper", {}).get("angle", 0.0)),
        "chain_cost": float(-total_score),
        "chain_nfev": int(sum(value.get("evaluations", 0) for value in joint_results.values())),
    }
    diagnostics = {
        "selected_wrist_rank": int(rank),
        "wrist_hypotheses": int(len(wrist_hypotheses)),
        "wrist_pixels_ds": visibility["wrist"],
        "gripper_pixels_ds": visibility["gripper"],
        "fit_gripper": bool(fit_gripper),
        "wg_score": float(total_score),
    }
    return canonicalize_prediction(pose, args.canonical_eps), diagnostics


def _clip_polygon_x(vertices, threshold, keep_lower):
    vertices = [np.asarray(vertex, dtype=np.float64) for vertex in vertices]

    def inside(vertex):
        return vertex[0] <= threshold if keep_lower else vertex[0] >= threshold

    clipped = []
    previous = vertices[-1]
    previous_inside = inside(previous)
    for current in vertices:
        current_inside = inside(current)
        if current_inside != previous_inside:
            delta = current - previous
            if abs(float(delta[0])) > 1e-12:
                ratio = (float(threshold) - float(previous[0])) / float(delta[0])
                clipped.append(previous + ratio * delta)
        if current_inside:
            clipped.append(current)
        previous = current
        previous_inside = current_inside
    return np.asarray(clipped, dtype=np.float64) if len(clipped) >= 3 else np.empty((0, 3), dtype=np.float64)


def _triangulate(vertices):
    vertices = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    triangles = []
    for index in range(1, len(vertices) - 1):
        triangle = np.stack((vertices[0], vertices[index], vertices[index + 1]))
        if np.linalg.norm(np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])) > 1e-14:
            triangles.append(triangle)
    return triangles


def _mesh_from_triangles(triangles):
    vertices = np.asarray(triangles, dtype=np.float64).reshape(-1, 3)
    faces = np.arange(len(vertices), dtype=np.int64).reshape(-1, 3)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def _split_mesh_x(mesh, threshold):
    lower = []
    upper = []
    for triangle in np.asarray(mesh.vertices[mesh.faces], dtype=np.float64):
        lower.extend(_triangulate(_clip_polygon_x(triangle, threshold, True)))
        upper.extend(_triangulate(_clip_polygon_x(triangle, threshold, False)))
    return _mesh_from_triangles(lower), _mesh_from_triangles(upper)


class WristGripperTrimeshRenderer:
    def __init__(self, device):
        base = GMSInstrumentTrimeshRenderer(device)
        self.wrist = base.part_meshes["wrist"].copy()
        self.grippers = {}
        for name in ("l_gripper", "r_gripper"):
            static, moving = _split_mesh_x(base.part_meshes[name], SURFEMB_GRIPPER_STATIC_THRESHOLD_M)
            self.grippers[name] = {"static": static, "moving": moving}

    @staticmethod
    def _transform(mesh, transform):
        out = mesh.copy()
        out.apply_transform(transform)
        return out

    def _world_meshes(self, pose):
        transforms = fk_matrices_np(
            pose["rot"], pose["trans"], pose.get("alpha", 0.0), pose["theta_l"], pose["theta_r"]
        )
        wrist_to_gripper = make_transform_np(np.eye(3), [GRIPPER_JOINT_OFFSET_M, 0.0, 0.0])
        meshes = [("wrist", self._transform(self.wrist, transforms["wrist"]))]
        for name in ("l_gripper", "r_gripper"):
            meshes.append(("wrist", self._transform(self.grippers[name]["static"], transforms["wrist"] @ wrist_to_gripper)))
            meshes.append((name, self._transform(self.grippers[name]["moving"], transforms[name])))
        return meshes

    def render(self, pose, K, image_shape):
        h, w = int(image_shape[0]), int(image_shape[1])
        scene = pyrender.Scene(bg_color=(0, 0, 0, 0), ambient_light=(0.28, 0.28, 0.28))
        for name, mesh in self._world_meshes(pose):
            rgba = PART_COLORS[name]
            material = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=tuple(float(value) / 255.0 for value in rgba),
                metallicFactor=0.05,
                roughnessFactor=0.62,
            )
            scene.add(pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False))
        K = np.asarray(K, dtype=np.float64)
        camera_pose = np.eye(4, dtype=np.float64)
        camera_pose[:, 1:3] *= -1.0
        scene.add(
            pyrender.IntrinsicsCamera(
                fx=float(K[0, 0]), fy=float(K[1, 1]), cx=float(K[0, 2]), cy=float(K[1, 2]), znear=0.001, zfar=1.0
            ),
            pose=camera_pose,
        )
        light_pose = camera_pose.copy()
        light_pose[0, 3] += 0.1
        light_pose[2, 3] += 0.1
        scene.add(pyrender.PointLight(color=np.ones(3), intensity=2.2), pose=light_pose)
        renderer = pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h)
        try:
            color, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        finally:
            renderer.delete()
        return color[..., :3].astype(np.uint8), np.asarray(depth)

    def overlay(self, rgb, pose, K, alpha=0.78):
        color, depth = self.render(pose, K, rgb.shape[:2])
        support = depth > 0.0
        out = np.asarray(rgb, dtype=np.uint8).copy()
        out[support] = np.clip(
            out[support].astype(np.float32) * (1.0 - alpha) + color[support].astype(np.float32) * alpha,
            0,
            255,
        ).astype(np.uint8)
        return out, support


def _pad_square(rgb, size):
    h, w = rgb.shape[:2]
    scale = min(float(size) / w, float(size) / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    out = np.zeros((size, size, 3), dtype=np.uint8)
    x0, y0 = (size - nw) // 2, (size - nh) // 2
    out[y0 : y0 + nh, x0 : x0 + nw] = resized
    return out


def _title(rgb, title, subtitle=""):
    image = Image.fromarray(rgb)
    header_h = 62
    out = Image.new("RGB", (image.width, image.height + header_h), (247, 247, 247))
    out.paste(image, (0, header_h))
    draw = ImageDraw.Draw(out)
    draw.text((9, 5), title, fill=(18, 18, 18), font=_font(18))
    if subtitle:
        draw.text((9, 34), subtitle, fill=(70, 70, 70), font=_font(13))
    return np.asarray(out)


def _overlay_part_mask(rgb, part_mask):
    out = np.asarray(rgb, dtype=np.uint8).copy()
    for label, color in ((2, np.array(PART_COLORS["wrist"][:3])), (1, np.array(PART_COLORS["r_gripper"][:3]))):
        mask = np.asarray(part_mask) == label
        out[mask] = np.clip(out[mask].astype(np.float32) * 0.40 + color * 0.60, 0, 255).astype(np.uint8)
    out[np.asarray(part_mask) == 3] = (out[np.asarray(part_mask) == 3].astype(np.float32) * 0.35).astype(np.uint8)
    return out


def _mask_iou(render_support, part_mask):
    gt = (np.asarray(part_mask) == 1) | (np.asarray(part_mask) == 2)
    pred = np.asarray(render_support, dtype=bool)
    return float(np.logical_and(gt, pred).sum() / max(1, np.logical_or(gt, pred).sum()))


def _pose_metrics(pred, gt):
    return {
        "wrist_trans_err_mm": float(np.linalg.norm(np.asarray(pred["trans"]) - np.asarray(gt["trans"])) * 1000.0),
        "wrist_rot_err_deg": _rotation_error_deg(pred["rot"], gt["rot"]),
        "theta_l_err_deg": _angle_error_deg(pred["theta_l"], gt["theta_l"]),
        "theta_r_err_deg": _angle_error_deg(pred["theta_r"], gt["theta_r"]),
    }


def _select_examples(rows, model_names, count):
    by_index = {}
    for row in rows:
        if row["status"] == "ok":
            by_index.setdefault(int(row["dataset_idx"]), {})[row["model"]] = row
    complete = {
        index: values
        for index, values in by_index.items()
        if all(name in values and bool(values[name].get("fit_gripper", False)) for name in model_names)
    }
    if not complete:
        return []

    def severity(row):
        return float(row["wrist_trans_err_mm"]) / 10.0 + float(row["wrist_rot_err_deg"]) / 15.0

    scored = []
    for index, values in complete.items():
        scores = [severity(values[name]) for name in model_names]
        scored.append((index, float(np.mean(scores)), float(max(scores)), float(abs(scores[0] - scores[-1]))))
    selected = []

    def add(index, label):
        if index not in {item[0] for item in selected}:
            selected.append((index, label))

    for item in sorted(scored, key=lambda value: value[1])[:2]:
        add(item[0], "success")
    add(max(scored, key=lambda item: item[3])[0], "model-contrast")
    add(max(scored, key=lambda item: item[2])[0], "failure")
    for item in sorted(scored, key=lambda value: value[1], reverse=True):
        add(item[0], "failure")
        if len(selected) >= int(count):
            break
    for item in sorted(scored, key=lambda value: value[1]):
        add(item[0], "success")
        if len(selected) >= int(count):
            break
    return selected[: int(count)]


def _write_csv(path, rows):
    keys = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _contact_sheet(paths, output_path):
    images = [Image.open(path).convert("RGB") for path in paths]
    width = max(image.width for image in images)
    canvas = Image.new("RGB", (width, sum(image.height for image in images)), "white")
    top = 0
    for image in images:
        canvas.paste(image, (0, top))
        top += image.height
    canvas.save(output_path, quality=94)


def main(args):
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = parse_model_specs(args.model or list(DEFAULT_MODELS))
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dataset = build_dataset(args)

    loaded = []
    for spec in specs:
        model, checkpoint_iter, _ = load_model(spec, device)
        surfaces = load_part_surfaces(args.surface_root, keys_per_part=args.surface_keys_per_part, seed=args.surface_seed)
        encode_surface_keys(model, surfaces, device, mask_keys_per_part=args.mask_keys_per_part)
        loaded.append((spec, model, surfaces, checkpoint_iter))

    rows = []
    predictions = {}
    targets = {}
    for dataset_idx in args.candidate_indices:
        _, target = dataset[int(dataset_idx)]
        image, K_crop, _, M_crop = make_surfemb_crop(target, args)
        part_orig = target["part_mask_orig"].cpu().numpy().astype(np.uint8)
        part_crop = cv2.warpAffine(
            part_orig,
            M_crop,
            (args.crop_size, args.crop_size),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        gt = target_pose(target)
        targets[int(dataset_idx)] = target
        for model_index, (spec, model, surfaces, checkpoint_iter) in enumerate(loaded):
            row = {
                "model": spec["name"],
                "checkpoint_iter": checkpoint_iter,
                "dataset_idx": int(dataset_idx),
                "video": str(target["video_name"]),
                "frame_id": str(target["frame_id"]),
                "instance_id": int(target["instance_id"].item()),
            }
            try:
                pred, diagnostics = fit_wrist_gripper_pose(
                    model,
                    surfaces,
                    image,
                    K_crop,
                    part_crop,
                    args,
                    seed=int(args.seed) + int(dataset_idx) * 1009 + model_index * 1000003,
                )
                row["status"] = "ok"
                row.update(_pose_metrics(pred, gt))
                row.update(diagnostics)
                if not diagnostics["fit_gripper"]:
                    row["theta_l_err_deg"] = float("nan")
                    row["theta_r_err_deg"] = float("nan")
                predictions[(int(dataset_idx), spec["name"])] = pred
            except Exception as exc:
                row["status"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
            print(
                f"idx={dataset_idx} model={spec['name']} status={row['status']} "
                f"t={row.get('wrist_trans_err_mm', float('nan')):.2f} "
                f"r={row.get('wrist_rot_err_deg', float('nan')):.2f}",
                flush=True,
            )

    _write_csv(out_dir / "all_candidate_fits.csv", rows)
    selected = _select_examples(rows, [spec["name"] for spec, *_ in loaded], args.num_examples)
    if not selected:
        raise RuntimeError("No common successful W/G fits to visualize")

    renderer = WristGripperTrimeshRenderer(device)
    panel_paths = []
    for dataset_idx, category in selected:
        target = targets[dataset_idx]
        rgb = target["orig_rgb"].cpu().numpy().astype(np.uint8)
        part = target["part_mask_orig"].cpu().numpy().astype(np.uint8)
        K_orig = target["K_orig"].cpu().numpy().astype(np.float32)
        gt = target_pose(target)
        gt_overlay, gt_support = renderer.overlay(rgb, gt, K_orig, alpha=args.overlay_alpha)
        tiles = [
            _title(_pad_square(rgb, args.panel_size), "RGB"),
            _title(_pad_square(_overlay_part_mask(rgb, part), args.panel_size), "SAM part ROI", "shaft dimmed; green=wrist, pink=gripper"),
            _title(
                _pad_square(gt_overlay, args.panel_size),
                "GT wrist+gripper trimesh",
                f"SAM W/G mask IoU={_mask_iou(gt_support, part):.3f}",
            ),
        ]
        for spec, *_ in loaded:
            pred = predictions[(dataset_idx, spec["name"])]
            overlay, support = renderer.overlay(rgb, pred, K_orig, alpha=args.overlay_alpha)
            metric_row = next(
                row for row in rows if row["model"] == spec["name"] and int(row["dataset_idx"]) == dataset_idx
            )
            subtitle = (
                f"t={metric_row['wrist_trans_err_mm']:.1f}mm r={metric_row['wrist_rot_err_deg']:.1f}deg "
                f"grip={np.mean([metric_row['theta_l_err_deg'], metric_row['theta_r_err_deg']]):.1f}deg "
                f"IoU={_mask_iou(support, part):.3f}"
            )
            tiles.append(_title(_pad_square(overlay, args.panel_size), f"{spec['name']} W/G fit", subtitle))
            metric_row["render_wg_iou"] = _mask_iou(support, part)
            metric_row["selection_category"] = category
        panel = np.concatenate(tiles, axis=1)
        banner_h = 46
        panel_image = Image.fromarray(panel)
        canvas = Image.new("RGB", (panel_image.width, panel_image.height + banner_h), (28, 28, 28))
        canvas.paste(panel_image, (0, banner_h))
        ImageDraw.Draw(canvas).text(
            (12, 10),
            f"{category}  dataset_idx={dataset_idx}  {target['video_name']}  "
            f"frame={target['frame_id']}  instance={int(target['instance_id'].item())}",
            fill=(255, 255, 255),
            font=_font(20),
        )
        path = out_dir / f"{category}_idx{dataset_idx:04d}_{target['video_name']}_frame{target['frame_id']}.jpg"
        canvas.save(path, quality=94)
        panel_paths.append(path)
        print(f"saved {path}", flush=True)

    _write_csv(out_dir / "all_candidate_fits.csv", rows)
    _contact_sheet(panel_paths, out_dir / "selected_success_failure_contact_sheet.jpg")
    print(f"fits={out_dir / 'all_candidate_fits.csv'}", flush=True)
    print(f"contact_sheet={out_dir / 'selected_success_failure_contact_sheet.jpg'}", flush=True)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", default=None, help="NAME=CHECKPOINT; repeat for multiple models")
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--candidate_indices", default=DEFAULT_CANDIDATES)
    parser.add_argument("--num_examples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--panel_size", type=int, default=400)
    parser.add_argument("--overlay_alpha", type=float, default=0.78)
    parser.add_argument("--amp", type=int, choices=[0, 1], default=1)
    parser.add_argument("--down_sample_scale", type=int, default=3)
    parser.add_argument("--surface_keys_per_part", type=int, default=4096)
    parser.add_argument("--mask_keys_per_part", type=int, default=512)
    parser.add_argument("--surface_seed", type=int, default=2026)
    parser.add_argument("--max_poses", type=int, default=4096)
    parser.add_argument("--max_pose_evaluations", type=int, default=512)
    parser.add_argument("--pose_batch_size", type=int, default=64)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--corr_alpha", type=float, default=1.5)
    parser.add_argument("--dist_2d_min", type=float, default=0.07)
    parser.add_argument("--coarse_steps", type=int, default=73)
    parser.add_argument("--fine_steps", type=int, default=17)
    parser.add_argument("--min_wrist_pixels_ds", type=int, default=4)
    parser.add_argument("--min_gripper_pixels_ds", type=int, default=3)
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
    parsed.candidate_indices = [int(value) for value in parsed.candidate_indices.split(",") if value.strip()]
    main(parsed)
