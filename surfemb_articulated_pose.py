import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import least_squares, minimize
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from instrument_geometry import (
    GRIPPER_JOINT_OFFSET_M,
    fk_matrices_np,
    make_transform_np,
    quat_wxyz_to_matrix_np,
)


PART_NAMES = ("shaft", "wrist", "l_gripper", "r_gripper")
PART_RASTER_IDS = {"shaft": 1, "wrist": 2, "l_gripper": 3, "r_gripper": 4}


@dataclass
class PartSurface:
    name: str
    points_m: np.ndarray
    normals: np.ndarray
    key_coords: np.ndarray
    diameter_m: float
    keys: torch.Tensor | None = None
    mask_keys: torch.Tensor | None = None


@dataclass
class PoseHypothesis:
    transform: np.ndarray
    score: float
    mask_score: float
    coord_score: float


@dataclass
class PartScoreContext:
    points: torch.Tensor
    corr_prob: torch.Tensor
    corr_log_score: torch.Tensor
    mask_log_prob: torch.Tensor
    neg_mask_log_prob: torch.Tensor


def _load_payload(root, part_name):
    path = Path(root) / f"{part_name}_surface_points.npy"
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.load(path, allow_pickle=True).item()


def _diameter(points):
    points = np.asarray(points, dtype=np.float64)
    return float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))


def _subsample_arrays(arrays, count, rng):
    n = len(arrays[0])
    if n <= int(count):
        return [np.asarray(array) for array in arrays]
    idx = np.sort(rng.choice(n, size=int(count), replace=False))
    return [np.asarray(array)[idx] for array in arrays]


def load_part_surfaces(surface_root, keys_per_part=4096, seed=2026):
    """Load the exact effective-part geometry used by SurfEmb supervision."""
    root = Path(surface_root)
    payloads = {name: _load_payload(root, name) for name in PART_NAMES}
    rng = np.random.default_rng(int(seed))
    surfaces = {}

    shaft = payloads["shaft"]
    arrays = _subsample_arrays(
        [shaft["points_part_m"], shaft["normals_part"], shaft["points_norm"]],
        keys_per_part,
        rng,
    )
    surfaces["shaft"] = PartSurface(
        "shaft", arrays[0].astype(np.float32), arrays[1].astype(np.float32), arrays[2].astype(np.float32), _diameter(arrays[0])
    )

    wrist_points = [np.asarray(payloads["wrist"]["points_part_m"], dtype=np.float32)]
    wrist_normals = [np.asarray(payloads["wrist"]["normals_part"], dtype=np.float32)]
    wrist_coords = [np.asarray(payloads["wrist"]["points_norm"], dtype=np.float32)]
    for name in ("l_gripper", "r_gripper"):
        payload = payloads[name]
        static = np.asarray(payload["static_wrist_mask"], dtype=bool)
        points = np.asarray(payload["points_part_m"], dtype=np.float32)[static].copy()
        points[:, 0] += float(GRIPPER_JOINT_OFFSET_M)
        wrist_points.append(points)
        wrist_normals.append(np.asarray(payload["normals_part"], dtype=np.float32)[static])
        wrist_coords.append(np.asarray(payload["points_norm"], dtype=np.float32)[static])
    arrays = _subsample_arrays(
        [np.concatenate(wrist_points), np.concatenate(wrist_normals), np.concatenate(wrist_coords)],
        keys_per_part,
        rng,
    )
    surfaces["wrist"] = PartSurface(
        "wrist", arrays[0].astype(np.float32), arrays[1].astype(np.float32), arrays[2].astype(np.float32), _diameter(arrays[0])
    )

    for name in ("l_gripper", "r_gripper"):
        payload = payloads[name]
        moving = ~np.asarray(payload["static_wrist_mask"], dtype=bool)
        arrays = _subsample_arrays(
            [
                np.asarray(payload["points_part_m"])[moving],
                np.asarray(payload["normals_part"])[moving],
                np.asarray(payload["points_norm"])[moving],
            ],
            keys_per_part,
            rng,
        )
        surfaces[name] = PartSurface(
            name, arrays[0].astype(np.float32), arrays[1].astype(np.float32), arrays[2].astype(np.float32), _diameter(arrays[0])
        )
    return surfaces


@torch.inference_mode()
def encode_surface_keys(model, surfaces, device, mask_keys_per_part=512, chunk_size=65536):
    for surface in surfaces.values():
        coords = torch.from_numpy(surface.key_coords).to(device=device, dtype=torch.float32)
        chunks = []
        for start in range(0, len(coords), int(chunk_size)):
            chunks.append(model.surface_key_mlp(coords[start : start + int(chunk_size)]).float())
        surface.keys = torch.cat(chunks, dim=0)
        n_mask = min(int(mask_keys_per_part), len(surface.keys))
        idx = torch.linspace(0, len(surface.keys) - 1, n_mask, device=device).round().long()
        surface.mask_keys = surface.keys[idx]
    return surfaces


def downsample_intrinsics(K, scale):
    K = np.asarray(K, dtype=np.float64).copy()
    K[:2, 2] += 0.5
    K[:2] /= float(scale)
    K[:2, 2] -= 0.5
    return K


def correspondence_logits(query, keys, similarity="raw_dot", temperature=1.0):
    """Compute query/key logits with an explicit train/test similarity contract."""
    similarity = str(similarity)
    temperature = float(temperature)
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if similarity == "cosine":
        query = F.normalize(query, dim=-1, eps=1e-6)
        keys = F.normalize(keys, dim=-1, eps=1e-6)
    elif similarity != "raw_dot":
        raise ValueError(f"Unknown correspondence similarity: {similarity!r}")
    return (query @ keys.T) / temperature


@torch.inference_mode()
def build_part_probability_inputs(
    mask_logits,
    query_chw,
    surfaces,
    K,
    down_sample_scale=3,
    similarity="raw_dot",
    temperature=1.0,
):
    """Build p(part | pixel, image) from the learned key likelihoods."""
    scale = int(down_sample_scale)
    query = F.avg_pool2d(query_chw[None].float(), scale)[0]
    _, h, w = query.shape
    query_flat = query.permute(1, 2, 0).reshape(h * w, -1)

    logits = mask_logits.float()[None, None]
    obj_log_prob = F.max_pool2d(F.logsigmoid(logits), scale)[0, 0]
    obj_prob = obj_log_prob.exp().reshape(-1)

    part_scores = []
    for name in PART_NAMES:
        keys = surfaces[name].mask_keys
        score = torch.logsumexp(
            correspondence_logits(query_flat, keys, similarity, temperature), dim=1
        ) - math.log(float(len(keys)))
        part_scores.append(score)
    part_log_prob = torch.log_softmax(torch.stack(part_scores, dim=1), dim=1)

    out = {}
    for index, name in enumerate(PART_NAMES):
        log_prob = (obj_log_prob.reshape(-1) + part_log_prob[:, index]).clamp(max=-1e-7)
        prob = log_prob.exp()
        neg_log_prob = torch.log1p(-prob.clamp(max=1.0 - 1e-7))
        log_prob_img = log_prob.reshape(h, w)
        neg_log_prob_img = neg_log_prob.reshape(h, w)
        log_prob_img = F.max_pool2d(log_prob_img[None, None], 3, 1, 1)[0, 0]
        neg_log_prob_img = F.max_pool2d(neg_log_prob_img[None, None], 3, 1, 1)[0, 0]
        out[name] = {
            "prob": prob,
            "mask_log_prob": log_prob_img.reshape(-1),
            "neg_mask_log_prob": neg_log_prob_img.reshape(-1),
        }
    return query_flat, out, downsample_intrinsics(K, scale), (h, w), obj_prob


def _sample_ap3p_hypotheses(
    corr_prob,
    img_points,
    object_points,
    object_normals,
    K,
    image_size,
    diameter,
    max_poses,
    alpha,
    dist_2d_min,
    rng,
):
    m = int(object_points.shape[0])
    weights = corr_prob.reshape(-1).float().clamp_min_(0.0).pow_(float(alpha))
    total = weights.sum()
    if not torch.isfinite(total) or float(total.item()) <= 0.0:
        return np.empty((0, 3, 4), dtype=np.float64)
    # Match original SurfEmb's inversion sampling. torch.multinomial cannot
    # address more than 2^24 categories, while pixel x surface distributions
    # exceed that limit with 4096 keys per articulated part.
    cumsum = torch.cumsum(weights, dim=0)
    generator = torch.Generator(device=weights.device)
    generator.manual_seed(int(rng.integers(0, np.iinfo(np.int64).max)))
    samples = torch.rand(
        int(max_poses) * 4,
        device=weights.device,
        dtype=weights.dtype,
        generator=generator,
    ) * cumsum[-1]
    corr_idx = torch.searchsorted(cumsum, samples).reshape(int(max_poses), 4)
    p2d_idx = torch.div(corr_idx, m, rounding_mode="floor")
    p3d_idx = corr_idx % m
    p2d = img_points[p2d_idx].float().cpu().numpy()
    p3d = object_points[p3d_idx].float().cpu().numpy()
    n3d = object_normals[p3d_idx.cpu().numpy()[:, :3]]

    poses = []
    kept_p2d = []
    kept_p3d = []
    kept_n3d = []
    for index in range(len(p2d)):
        try:
            ok, rvecs, tvecs = cv2.solveP3P(p3d[index], p2d[index], K, None, flags=cv2.SOLVEPNP_AP3P)
        except cv2.error:
            continue
        if not ok or len(rvecs) == 0:
            continue
        pick = int(rng.integers(0, len(rvecs)))
        R, _ = cv2.Rodrigues(np.asarray(rvecs[pick], dtype=np.float64))
        pose = np.concatenate([R, np.asarray(tvecs[pick], dtype=np.float64).reshape(3, 1)], axis=1)
        poses.append(pose)
        kept_p2d.append(p2d[index])
        kept_p3d.append(p3d[index])
        kept_n3d.append(n3d[index])
    if not poses:
        return np.empty((0, 3, 4), dtype=np.float64)

    poses = np.asarray(poses, dtype=np.float64)
    p2d = np.asarray(kept_p2d, dtype=np.float64)
    p3d = np.asarray(kept_p3d, dtype=np.float64)
    n3d = np.asarray(kept_n3d, dtype=np.float64)
    dist_2d = np.linalg.norm(p2d[:, :3, None] - p2d[:, None, :3], axis=-1).max(axis=(1, 2))
    z = poses[:, 2, 3]
    z_min = float(K[0, 0]) * float(diameter) / (float(image_size) * 20.0)
    z_max = float(K[0, 0]) * float(diameter) / (float(image_size) * 0.5)
    size_ok = (z_min < z) & (z < z_max)
    Rt = poses[:, :3, :3].transpose(0, 2, 1)
    normals_cam = n3d @ Rt
    points_cam = p3d[:, :3] @ Rt + poses[:, None, :3, 3]
    front_ok = np.all(np.sum(normals_cam * points_cam, axis=-1) < 0.0, axis=1)
    valid = (dist_2d >= float(dist_2d_min) * float(image_size)) & size_ok & front_ok
    return poses[valid]


@torch.inference_mode()
def _score_pose_batch(poses, points, corr_log, mask_log_prob, neg_mask_log_prob, K, image_hw):
    device = points.device
    h, w = image_hw
    n = int(h * w)
    R = torch.from_numpy(poses[:, :3, :3]).to(device=device, dtype=torch.float32)
    t = torch.from_numpy(poses[:, :3, 3]).to(device=device, dtype=torch.float32)
    points_cam = points[None] @ R.transpose(1, 2) + t[:, None]
    z = points_cam[..., 2]
    uvw = points_cam @ torch.as_tensor(K, device=device, dtype=torch.float32).T
    uv = torch.round(uvw[..., :2] / uvw[..., 2:].clamp_min(1e-8)).long()
    valid = (z > 0.0) & (uv[..., 0] >= 0) & (uv[..., 0] < w) & (uv[..., 1] >= 0) & (uv[..., 1] < h)
    pixel = uv[..., 1] * w + uv[..., 0]
    ignore = n
    pixel = torch.where(valid, pixel, torch.full_like(pixel, ignore))

    batch = len(poses)
    pose_offset = torch.arange(batch, device=device).view(-1, 1) * (n + 1)
    flat_pixel = pixel + pose_offset
    zbuf = torch.full((batch * (n + 1),), torch.inf, device=device)
    zbuf.scatter_reduce_(0, flat_pixel.reshape(-1), z.reshape(-1), reduce="amin", include_self=True)
    near = valid & (z <= zbuf[flat_pixel] + 1e-6)

    key_idx = torch.arange(points.shape[0], device=device).view(1, -1).expand(batch, -1)
    sentinel = int(points.shape[0])
    argbuf = torch.full((batch * (n + 1),), sentinel, dtype=torch.long, device=device)
    candidates = torch.where(near, key_idx, torch.full_like(key_idx, sentinel))
    argbuf.scatter_reduce_(0, flat_pixel.reshape(-1), candidates.reshape(-1), reduce="amin", include_self=True)
    argbuf = argbuf.reshape(batch, n + 1)[:, :n]
    pose_idx, pixel_idx = torch.where(argbuf < sentinel)
    if len(pose_idx) == 0:
        empty = torch.full((batch,), -torch.inf, device=device)
        return empty, empty.clone(), empty.clone()
    visible_key_idx = argbuf[pose_idx, pixel_idx]

    base = neg_mask_log_prob.sum()
    delta = mask_log_prob - neg_mask_log_prob
    mask_sum = torch.full((batch,), float(base.item()), device=device)
    mask_sum.scatter_add_(0, pose_idx, delta[pixel_idx])
    mask_score = mask_sum / float(n) / math.log(2.0)

    corr_values = corr_log[pixel_idx, visible_key_idx]
    coord_sum = torch.zeros((batch,), device=device)
    coord_count = torch.zeros((batch,), device=device)
    coord_sum.scatter_add_(0, pose_idx, corr_values)
    coord_count.scatter_add_(0, pose_idx, torch.ones_like(corr_values))
    coord_score = coord_sum / coord_count.clamp_min(1.0) / math.log(float(points.shape[0]))
    coord_score[coord_count == 0] = -torch.inf
    return mask_score + coord_score, mask_score, coord_score


@torch.inference_mode()
def prepare_part_score_context(
    query_flat,
    probability_input,
    surface,
    image_hw,
    similarity="raw_dot",
    temperature=1.0,
):
    h, w = image_hw
    keys = surface.keys
    points = torch.from_numpy(surface.points_m).to(device=query_flat.device, dtype=torch.float32)
    corr_log = torch.log_softmax(
        correspondence_logits(query_flat, keys, similarity, temperature), dim=1
    )
    corr_prob = corr_log.exp() * probability_input["prob"][:, None]
    corr_log_score = torch.empty_like(corr_log)
    corr_img = corr_log.reshape(h, w, -1).permute(2, 0, 1)
    for start in range(0, corr_img.shape[0], 1024):
        pooled = F.max_pool2d(corr_img[start : start + 1024, None], 3, 1, 1)[:, 0]
        corr_log_score[:, start : start + 1024] = pooled.permute(1, 2, 0).reshape(h * w, -1)
    return PartScoreContext(
        points=points,
        corr_prob=corr_prob,
        corr_log_score=corr_log_score,
        mask_log_prob=probability_input["mask_log_prob"],
        neg_mask_log_prob=probability_input["neg_mask_log_prob"],
    )


@torch.inference_mode()
def prepare_original_surfemb_score_context(
    mask_logits,
    query_chw,
    surface,
    K,
    down_sample_scale=3,
    similarity="raw_dot",
    temperature=1.0,
):
    """Build the exact tensors consumed by SurfEmb's original pose estimator.

    Sampling uses sigmoid(avg_pool(mask logits)); hypothesis scoring separately
    max-pools log-sigmoid foreground and background probabilities. This
    distinction is present in the original implementation and is intentionally
    kept separate from the articulated part-probability path above.
    """
    scale = int(down_sample_scale)
    if scale <= 0:
        raise ValueError(f"down_sample_scale must be positive, got {scale}")
    if mask_logits.ndim != 2 or query_chw.ndim != 3:
        raise ValueError(
            f"Expected mask (H,W) and queries (E,H,W), got "
            f"{tuple(mask_logits.shape)} and {tuple(query_chw.shape)}"
        )

    query = F.avg_pool2d(query_chw[None].float(), scale)[0]
    _, h, w = query.shape
    if h != w:
        raise ValueError(f"Original SurfEmb pose estimator expects a square crop, got {(h, w)}")
    query_flat = query.permute(1, 2, 0).reshape(h * w, -1)

    logits = mask_logits.float()[None, None]
    mask_prob = torch.sigmoid(F.avg_pool2d(logits, scale)[0, 0]).reshape(-1)
    mask_log_prob = F.max_pool2d(F.logsigmoid(logits), scale)[0, 0]
    neg_mask_log_prob = F.max_pool2d(F.logsigmoid(-logits), scale)[0, 0]
    mask_log_prob = F.max_pool2d(mask_log_prob[None, None], 3, 1, 1)[0, 0].reshape(-1)
    neg_mask_log_prob = F.max_pool2d(neg_mask_log_prob[None, None], 3, 1, 1)[0, 0].reshape(-1)

    keys = surface.keys
    points = torch.from_numpy(surface.points_m).to(device=query_flat.device, dtype=torch.float32)
    corr_log = torch.log_softmax(
        correspondence_logits(query_flat, keys, similarity, temperature), dim=1
    )
    corr_prob = corr_log.exp() * mask_prob[:, None]
    corr_log_score = torch.empty_like(corr_log)
    corr_img = corr_log.reshape(h, w, -1).permute(2, 0, 1)
    for start in range(0, corr_img.shape[0], 1024):
        pooled = F.max_pool2d(corr_img[start : start + 1024, None], 3, 1, 1)[:, 0]
        corr_log_score[:, start : start + 1024] = pooled.permute(1, 2, 0).reshape(h * w, -1)

    context = PartScoreContext(
        points=points,
        corr_prob=corr_prob,
        corr_log_score=corr_log_score,
        mask_log_prob=mask_log_prob,
        neg_mask_log_prob=neg_mask_log_prob,
    )
    return context, downsample_intrinsics(K, scale), (h, w), mask_prob


@torch.inference_mode()
def estimate_part_pose_from_context(
    context,
    surface,
    K,
    image_hw,
    max_poses=1024,
    max_pose_evaluations=256,
    pose_batch_size=64,
    top_k=5,
    alpha=1.5,
    dist_2d_min=0.1,
    seed=0,
):
    """SurfEmb probability sampling and mask+coordinate hypothesis scoring for one rigid part."""
    h, w = image_hw
    if h != w:
        raise ValueError(f"SurfEmb pose estimator expects a square crop, got {(h, w)}")
    points = context.points
    yy, xx = torch.meshgrid(
        torch.arange(h, device=points.device),
        torch.arange(w, device=points.device),
        indexing="ij",
    )
    image_points = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
    rng = np.random.default_rng(int(seed))
    poses = _sample_ap3p_hypotheses(
        context.corr_prob,
        image_points,
        points,
        surface.normals,
        K,
        h,
        surface.diameter_m,
        max_poses,
        alpha,
        dist_2d_min,
        rng,
    )
    if len(poses) == 0:
        return []
    poses = poses[: int(max_pose_evaluations)]
    all_score = []
    all_mask = []
    all_coord = []
    for start in range(0, len(poses), int(pose_batch_size)):
        score, mask_score, coord_score = _score_pose_batch(
            poses[start : start + int(pose_batch_size)],
            points,
            context.corr_log_score,
            context.mask_log_prob,
            context.neg_mask_log_prob,
            K,
            image_hw,
        )
        all_score.append(score.cpu())
        all_mask.append(mask_score.cpu())
        all_coord.append(coord_score.cpu())
    scores = torch.cat(all_score).numpy()
    mask_scores = torch.cat(all_mask).numpy()
    coord_scores = torch.cat(all_coord).numpy()
    finite = np.flatnonzero(np.isfinite(scores))
    if len(finite) == 0:
        return []
    order = finite[np.argsort(scores[finite])[::-1]][: int(top_k)]
    hypotheses = []
    for index in order:
        T = np.eye(4, dtype=np.float64)
        T[:3] = poses[index]
        hypotheses.append(PoseHypothesis(T, float(scores[index]), float(mask_scores[index]), float(coord_scores[index])))
    return hypotheses


def _rotation_vector_to_matrix(rotation_vector):
    """Differentiable Rodrigues formula for original SurfEmb pose refinement."""
    x, y, z = rotation_vector.unbind()
    zero = torch.zeros((), device=rotation_vector.device, dtype=rotation_vector.dtype)
    skew = torch.stack(
        (
            torch.stack((zero, -z, y)),
            torch.stack((z, zero, -x)),
            torch.stack((-y, x, zero)),
        )
    )
    theta2 = torch.dot(rotation_vector, rotation_vector)
    theta = torch.sqrt(theta2.clamp_min(1e-12))
    small = theta2 < 1e-8
    a = torch.where(small, 1.0 - theta2 / 6.0, torch.sin(theta) / theta)
    b = torch.where(small, 0.5 - theta2 / 24.0, (1.0 - torch.cos(theta)) / theta2.clamp_min(1e-12))
    eye = torch.eye(3, device=rotation_vector.device, dtype=rotation_vector.dtype)
    return eye + a * skew + b * (skew @ skew)


def refine_original_surfemb_pose(
    hypothesis,
    mask_logits,
    query_chw,
    model,
    surface,
    K,
    renderer,
    device,
    denominator_keys=4096,
    denominator_pixel_chunk=4096,
    min_visible_pixels=20,
    max_iterations=None,
    translation_units_per_meter=1000.0,
    optimization_float64=False,
    rotation_units_per_radian=1.0,
):
    """Apply SurfEmb's original BFGS query/key likelihood refinement.

    The initial visible wrist surface is obtained with the shared triangle
    rasterizer. Depth is unprojected back into the wrist frame, so refinement
    uses continuous triangle-surface points rather than nearest sampled keys.
    """
    del mask_logits  # The original refinement objective uses correspondence likelihood only.
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    translation_units_per_meter = float(translation_units_per_meter)
    if translation_units_per_meter <= 0.0:
        raise ValueError("translation_units_per_meter must be positive")
    optimization_dtype = torch.float64 if bool(optimization_float64) else torch.float32
    rotation_units_per_radian = float(rotation_units_per_radian)
    if rotation_units_per_radian <= 0.0:
        raise ValueError("rotation_units_per_radian must be positive")
    transform = np.asarray(hypothesis.transform, dtype=np.float64).reshape(4, 4)
    h, w = int(query_chw.shape[-2]), int(query_chw.shape[-1])
    coord_img, part_img, depth, valid = renderer.render_candidate_part_visibility(
        {"wrist": transform},
        K,
        (h, w),
        include_parts=("wrist",),
    )
    visible = valid & (part_img == PART_RASTER_IDS["wrist"]) & (depth > 0.0)
    if int(visible.sum()) < int(min_visible_pixels):
        raise RuntimeError(
            f"Original SurfEmb refinement has only {int(visible.sum())} visible wrist pixels"
        )

    vv, uu = np.nonzero(visible)
    z = depth[vv, uu].astype(np.float64)
    rays = np.stack((uu, vv, np.ones_like(uu)), axis=1).astype(np.float64) @ np.linalg.inv(K).T
    points_cam = rays * z[:, None]
    R0 = transform[:3, :3]
    t0 = transform[:3, 3]
    points_obj = (points_cam - t0[None]) @ R0

    with torch.no_grad():
        coord_norm = torch.from_numpy(coord_img[vv, uu]).to(device=device, dtype=torch.float32)
        visible_keys = model.surface_key_mlp(coord_norm).to(dtype=optimization_dtype)
        query_img = query_chw.permute(1, 2, 0).to(device=device, dtype=optimization_dtype)
        n_denom = min(int(denominator_keys), int(surface.keys.shape[0]))
        if n_denom <= 0:
            raise RuntimeError("No surface keys available for SurfEmb refinement")
        if n_denom == int(surface.keys.shape[0]):
            denominator_surface_keys = surface.keys.to(dtype=optimization_dtype)
        else:
            index = torch.linspace(
                0,
                int(surface.keys.shape[0]) - 1,
                n_denom,
                device=surface.keys.device,
            ).round().long()
            denominator_surface_keys = surface.keys[index].to(dtype=optimization_dtype)
        query_flat = query_img.reshape(h * w, -1)
        denominator_chunks = []
        for start in range(0, len(query_flat), int(denominator_pixel_chunk)):
            denominator_chunks.append(
                torch.logsumexp(
                    query_flat[start : start + int(denominator_pixel_chunk)]
                    @ denominator_surface_keys.T,
                    dim=1,
                )
            )
        denominator_img = torch.cat(denominator_chunks).reshape(h, w, 1)

    points_obj = torch.from_numpy(points_obj).to(device=device, dtype=optimization_dtype)
    K_tensor = torch.from_numpy(K).to(device=device, dtype=optimization_dtype)

    def sample(image, normalized_points):
        return F.grid_sample(
            image.permute(2, 0, 1)[None],
            normalized_points[None, None],
            align_corners=False,
            padding_mode="border",
            mode="bilinear",
        )[0, :, 0].T

    def objective(parameters, return_gradient=False):
        pose = torch.as_tensor(parameters, device=device, dtype=optimization_dtype).clone().detach()
        pose.requires_grad_(bool(return_gradient))
        rotation = _rotation_vector_to_matrix(pose[:3] / rotation_units_per_radian)
        translation_m = pose[3:] / translation_units_per_meter
        points_cam_current = points_obj @ rotation.T + translation_m[None]
        projected = points_cam_current @ K_tensor.T
        image_points = projected[:, :2] / projected[:, 2:].clamp_min(1e-8)
        normalized_points = (
            (image_points + 0.5)
            * (2.0 / torch.tensor((w, h), device=device, dtype=optimization_dtype))
            - 1.0
        )
        sampled_query = sample(query_img, normalized_points)
        sampled_denominator = sample(denominator_img, normalized_points)[:, 0]
        log_numerator = (visible_keys * sampled_query).sum(dim=1)
        value = -(log_numerator.mean() - sampled_denominator.mean()) / 2.0
        if return_gradient:
            value.backward()
            return pose.grad.detach().cpu().numpy().astype(np.float64)
        return float(value.detach().cpu().item())

    # Original SurfEmb optimizes BOP translations in millimeters. Keeping that
    # parameter scale is essential: with meter-valued geometry, an unscaled
    # BFGS unit step would move the object by one meter and leave the crop.
    initial = np.concatenate(
        (
            cv2.Rodrigues(R0)[0].reshape(3) * rotation_units_per_radian,
            t0 * translation_units_per_meter,
        ),
        axis=0,
    )
    initial_objective = objective(initial)
    initial_gradient = objective(initial, return_gradient=True)
    options = None if max_iterations is None else {"maxiter": int(max_iterations)}
    result = minimize(
        fun=objective,
        x0=initial,
        jac=lambda parameters: objective(parameters, return_gradient=True),
        method="BFGS",
        options=options,
    )
    refined = np.eye(4, dtype=np.float64)
    refined[:3, :3] = cv2.Rodrigues(
        np.asarray(result.x[:3], dtype=np.float64) / rotation_units_per_radian
    )[0]
    refined[:3, 3] = np.asarray(result.x[3:], dtype=np.float64) / translation_units_per_meter
    return (
        PoseHypothesis(
            transform=refined,
            score=float(-result.fun),
            mask_score=float(hypothesis.mask_score),
            coord_score=float(-result.fun),
        ),
        {
            "original_refine_success": bool(result.success),
            "original_refine_status": int(result.status),
            "original_refine_nit": int(result.nit),
            "original_refine_nfev": int(result.nfev),
            "original_refine_visible_pixels": int(visible.sum()),
            "original_refine_objective": float(result.fun),
            "original_refine_initial_objective": float(initial_objective),
            "original_refine_translation_units_per_meter": translation_units_per_meter,
            "original_refine_float64": bool(optimization_float64),
            "original_refine_initial_gradient_norm": float(np.linalg.norm(initial_gradient)),
            "original_refine_rotation_units_per_radian": rotation_units_per_radian,
        },
    )


@torch.inference_mode()
def estimate_part_pose_topk_ransac(
    context,
    surface,
    K,
    image_hw,
    pixel_mask=None,
    max_correspondences=512,
    min_correspondences=12,
    min_part_probability=0.05,
    margin_power=0.0,
    ransac_iterations=2000,
    ransac_reprojection_error=3.0,
    ransac_confidence=0.999,
    min_inliers=8,
    min_inlier_fraction=0.6,
    keys_per_pixel=1,
    return_correspondences=False,
):
    """Fit one rigid part from high-confidence dense correspondences.

    Each eligible pixel contributes its most likely surface key. Candidates
    are ranked by joint part-and-key probability, deduplicated in 3D-key
    space, and fitted jointly with robust PnP instead of sampling a minimal
    four-correspondence AP3P proposal.
    """
    h, w = image_hw
    if h != w:
        raise ValueError(f"SurfEmb pose estimator expects a square crop, got {(h, w)}")
    corr_prob = context.corr_prob
    if corr_prob.ndim != 2 or corr_prob.shape[0] != h * w:
        raise ValueError(f"Unexpected correspondence probability shape {tuple(corr_prob.shape)}")

    keys_per_pixel = max(1, int(keys_per_pixel))
    top_count = min(max(2, keys_per_pixel), int(corr_prob.shape[1]))
    top_prob, top_key = torch.topk(corr_prob, k=top_count, dim=1, largest=True, sorted=True)
    joint_confidence = top_prob[:, 0]
    part_probability = corr_prob.sum(dim=1)
    confidence = joint_confidence
    if float(margin_power) > 0.0 and top_count > 1:
        margin = (1.0 - top_prob[:, 1] / joint_confidence.clamp_min(1e-12)).clamp_min(0.0)
        confidence = confidence * margin.pow(float(margin_power))

    eligible = torch.isfinite(confidence) & (part_probability >= float(min_part_probability))
    if pixel_mask is not None:
        pixel_mask = torch.as_tensor(pixel_mask, device=eligible.device, dtype=torch.bool).reshape(-1)
        if pixel_mask.numel() != eligible.numel():
            raise ValueError(f"pixel_mask has {pixel_mask.numel()} values, expected {eligible.numel()}")
        eligible &= pixel_mask
    candidate_idx = torch.nonzero(eligible, as_tuple=False).reshape(-1)
    if len(candidate_idx) < int(min_correspondences):
        return None, {
            "topk_candidates": int(len(candidate_idx)),
            "topk_correspondences": 0,
            "topk_inliers": 0,
            "topk_status": "insufficient_candidates",
        }

    if keys_per_pixel == 1:
        candidate_pixels = candidate_idx
        candidate_keys = top_key[candidate_idx, 0]
        candidate_confidence = confidence[candidate_idx]
    else:
        pair_count = min(keys_per_pixel, top_count)
        candidate_pixels = candidate_idx[:, None].expand(-1, pair_count).reshape(-1)
        candidate_keys = top_key[candidate_idx, :pair_count].reshape(-1)
        candidate_confidence = top_prob[candidate_idx, :pair_count].reshape(-1)
    order = candidate_confidence.argsort(descending=True)
    candidate_idx_cpu = candidate_pixels[order].cpu().numpy()
    key_idx_cpu = candidate_keys[order].cpu().numpy()
    confidence_cpu = candidate_confidence[order].cpu().numpy()

    selected_pixels = []
    selected_keys = []
    selected_confidence = []
    seen_keys = set()
    for pixel_index, key_index, value in zip(candidate_idx_cpu, key_idx_cpu, confidence_cpu):
        key_index = int(key_index)
        if key_index in seen_keys:
            continue
        seen_keys.add(key_index)
        selected_pixels.append(int(pixel_index))
        selected_keys.append(key_index)
        selected_confidence.append(float(value))
        if len(selected_pixels) >= int(max_correspondences):
            break

    diagnostics = {
        "topk_candidates": int(len(candidate_idx)),
        "topk_candidate_pairs": int(len(candidate_pixels)),
        "topk_keys_per_pixel": int(keys_per_pixel),
        "topk_correspondences": int(len(selected_pixels)),
        "topk_inliers": 0,
        "topk_status": "insufficient_unique_keys",
    }
    if len(selected_pixels) < int(min_correspondences):
        return None, diagnostics

    selected_pixels = np.asarray(selected_pixels, dtype=np.int64)
    selected_keys = np.asarray(selected_keys, dtype=np.int64)
    object_points = np.asarray(surface.points_m[selected_keys], dtype=np.float64)
    image_points = np.stack((selected_pixels % w, selected_pixels // w), axis=1).astype(np.float64)
    try:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points,
            image_points,
            np.asarray(K, dtype=np.float64),
            None,
            iterationsCount=int(ransac_iterations),
            reprojectionError=float(ransac_reprojection_error),
            confidence=float(ransac_confidence),
            flags=cv2.SOLVEPNP_EPNP,
        )
    except cv2.error as exc:
        diagnostics["topk_status"] = f"opencv_error:{exc.code}"
        return None, diagnostics

    n_inliers = 0 if inliers is None else int(len(inliers))
    inlier_fraction = float(n_inliers / max(1, len(selected_pixels)))
    diagnostics["topk_inliers"] = n_inliers
    diagnostics["topk_inlier_fraction"] = inlier_fraction
    diagnostics["topk_confidence_median"] = float(np.median(selected_confidence))
    inlier_idx = (
        np.empty((0,), dtype=np.int64)
        if inliers is None
        else np.asarray(inliers, dtype=np.int64).reshape(-1)
    )
    if return_correspondences:
        diagnostics["topk_selected_pixel_indices"] = selected_pixels.copy()
        diagnostics["topk_selected_key_indices"] = selected_keys.copy()
        diagnostics["topk_inlier_selected_indices"] = inlier_idx.copy()
    if not ok or n_inliers < int(min_inliers) or inlier_fraction < float(min_inlier_fraction):
        diagnostics["topk_status"] = "ransac_failed"
        return None, diagnostics

    try:
        rvec, tvec = cv2.solvePnPRefineLM(
            object_points[inlier_idx],
            image_points[inlier_idx],
            np.asarray(K, dtype=np.float64),
            None,
            rvec,
            tvec,
        )
    except cv2.error:
        pass
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
    translation = np.asarray(tvec, dtype=np.float64).reshape(3)
    if not np.isfinite(rotation).all() or not np.isfinite(translation).all() or translation[2] <= 0.0:
        diagnostics["topk_status"] = "invalid_pose"
        return None, diagnostics

    projected, _ = cv2.projectPoints(
        object_points[inlier_idx],
        np.asarray(rvec, dtype=np.float64),
        translation,
        np.asarray(K, dtype=np.float64),
        None,
    )
    residual = np.linalg.norm(projected.reshape(-1, 2) - image_points[inlier_idx], axis=1)
    diagnostics["topk_reprojection_median_px"] = float(np.median(residual))
    diagnostics["topk_reprojection_p90_px"] = float(np.quantile(residual, 0.9))
    diagnostics["topk_status"] = "ok"

    pose = np.concatenate((rotation, translation[:, None]), axis=1)[None]
    score, mask_score, coord_score = _score_pose_batch(
        pose,
        context.points,
        context.corr_log_score,
        context.mask_log_prob,
        context.neg_mask_log_prob,
        K,
        image_hw,
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return (
        PoseHypothesis(transform, float(score[0]), float(mask_score[0]), float(coord_score[0])),
        diagnostics,
    )


@torch.inference_mode()
def _local_refinement_correspondences(
    context,
    surface,
    image_hw,
    seed_pixels,
    seed_keys,
    pixel_mask=None,
    neighbors_per_inlier=20,
    max_correspondences=4096,
    min_part_probability=0.05,
):
    """Rematch local 2D/3D neighborhoods around coarse inlier pairs."""
    h, w = (int(image_hw[0]), int(image_hw[1]))
    corr_prob = context.corr_prob
    seed_pixels = np.asarray(seed_pixels, dtype=np.int64).reshape(-1)
    seed_keys = np.asarray(seed_keys, dtype=np.int64).reshape(-1)
    if len(seed_pixels) != len(seed_keys):
        raise ValueError("seed pixel/key counts differ")
    if len(seed_pixels) == 0:
        return (
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.float64),
            {"fine_seed_inliers": 0, "fine_status": "no_coarse_inliers"},
        )

    part_probability = corr_prob.sum(dim=1)
    eligible = torch.isfinite(part_probability) & (
        part_probability >= float(min_part_probability)
    )
    if pixel_mask is not None:
        pixel_mask = torch.as_tensor(
            pixel_mask, device=eligible.device, dtype=torch.bool
        ).reshape(-1)
        if pixel_mask.numel() != eligible.numel():
            raise ValueError(
                f"pixel_mask has {pixel_mask.numel()} values, expected {eligible.numel()}"
            )
        eligible &= pixel_mask
    eligible_pixels = torch.nonzero(eligible, as_tuple=False).reshape(-1).cpu().numpy()
    if len(eligible_pixels) == 0:
        return (
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.float64),
            {"fine_seed_inliers": int(len(seed_pixels)), "fine_status": "no_eligible_pixels"},
        )

    neighbor_count_2d = min(int(neighbors_per_inlier), len(eligible_pixels))
    neighbor_count_3d = min(int(neighbors_per_inlier), len(surface.points_m))
    if neighbor_count_2d <= 0 or neighbor_count_3d <= 0:
        raise ValueError("neighbors_per_inlier must be positive")

    eligible_uv = np.stack(
        (eligible_pixels % w, eligible_pixels // w), axis=1
    ).astype(np.float64)
    seed_uv = np.stack((seed_pixels % w, seed_pixels // w), axis=1).astype(np.float64)
    pixel_tree = cKDTree(eligible_uv)
    _, pixel_neighbor_rows = pixel_tree.query(seed_uv, k=neighbor_count_2d)
    pixel_neighbor_rows = np.asarray(pixel_neighbor_rows, dtype=np.int64)
    if pixel_neighbor_rows.ndim == 1:
        pixel_neighbor_rows = pixel_neighbor_rows[:, None]
    local_pixels = eligible_pixels[pixel_neighbor_rows]

    surface_tree = getattr(surface, "_surfemb_neighbor_tree", None)
    if surface_tree is None:
        surface_tree = cKDTree(np.asarray(surface.points_m, dtype=np.float64))
        surface._surfemb_neighbor_tree = surface_tree
    _, local_keys = surface_tree.query(
        np.asarray(surface.points_m[seed_keys], dtype=np.float64),
        k=neighbor_count_3d,
    )
    local_keys = np.asarray(local_keys, dtype=np.int64)
    if local_keys.ndim == 1:
        local_keys = local_keys[:, None]

    pixel_tensor = torch.as_tensor(local_pixels, device=corr_prob.device, dtype=torch.long)
    key_tensor = torch.as_tensor(local_keys, device=corr_prob.device, dtype=torch.long)
    local_scores = corr_prob[pixel_tensor[:, :, None], key_tensor[:, None, :]]
    best_score, best_key_column = local_scores.max(dim=2)
    best_key_column_np = best_key_column.cpu().numpy()
    rematched_keys = np.take_along_axis(
        local_keys[:, None, :], best_key_column_np[:, :, None], axis=2
    )[:, :, 0]
    candidate_pixels = local_pixels.reshape(-1)
    candidate_keys = rematched_keys.reshape(-1)
    candidate_confidence = best_score.reshape(-1).cpu().numpy().astype(np.float64)

    finite = np.isfinite(candidate_confidence)
    candidate_pixels = candidate_pixels[finite]
    candidate_keys = candidate_keys[finite]
    candidate_confidence = candidate_confidence[finite]
    order = np.argsort(candidate_confidence)[::-1]
    selected_pixels = []
    selected_keys = []
    selected_confidence = []
    seen_pixels = set()
    seen_keys = set()
    for candidate_index in order:
        pixel_index = int(candidate_pixels[candidate_index])
        key_index = int(candidate_keys[candidate_index])
        if pixel_index in seen_pixels or key_index in seen_keys:
            continue
        seen_pixels.add(pixel_index)
        seen_keys.add(key_index)
        selected_pixels.append(pixel_index)
        selected_keys.append(key_index)
        selected_confidence.append(float(candidate_confidence[candidate_index]))
        if len(selected_pixels) >= int(max_correspondences):
            break

    diagnostics = {
        "fine_seed_inliers": int(len(seed_pixels)),
        "fine_neighbors_per_inlier": int(neighbors_per_inlier),
        "fine_2d_neighbor_count": int(neighbor_count_2d),
        "fine_3d_neighbor_count": int(neighbor_count_3d),
        "fine_local_candidate_pairs": int(len(candidate_pixels)),
        "fine_correspondences": int(len(selected_pixels)),
        "fine_status": "ok" if selected_pixels else "no_unique_local_correspondences",
    }
    return (
        np.asarray(selected_pixels, dtype=np.int64),
        np.asarray(selected_keys, dtype=np.int64),
        np.asarray(selected_confidence, dtype=np.float64),
        diagnostics,
    )


@torch.inference_mode()
def estimate_part_pose_coarse_fine_ransac(
    context,
    surface,
    K,
    image_hw,
    pixel_mask=None,
    coarse_max_correspondences=4096,
    coarse_keys_per_pixel=4,
    coarse_min_inlier_fraction=0.0,
    fine_neighbors_per_inlier=20,
    fine_max_correspondences=4096,
    min_correspondences=12,
    min_part_probability=0.05,
    margin_power=0.0,
    ransac_iterations=2000,
    coarse_ransac_reprojection_error=3.0,
    fine_ransac_reprojection_error=2.0,
    ransac_confidence=0.999,
    min_inliers=8,
    min_inlier_fraction=0.6,
    return_correspondences=False,
):
    """Two-stage local SurfEmb matching followed by robust PnP in each stage."""
    coarse, coarse_diagnostics = estimate_part_pose_topk_ransac(
        context,
        surface,
        K,
        image_hw,
        pixel_mask=pixel_mask,
        max_correspondences=int(coarse_max_correspondences),
        min_correspondences=int(min_correspondences),
        min_part_probability=float(min_part_probability),
        margin_power=float(margin_power),
        ransac_iterations=int(ransac_iterations),
        ransac_reprojection_error=float(coarse_ransac_reprojection_error),
        ransac_confidence=float(ransac_confidence),
        min_inliers=int(min_inliers),
        min_inlier_fraction=float(coarse_min_inlier_fraction),
        keys_per_pixel=int(coarse_keys_per_pixel),
        return_correspondences=True,
    )
    coarse_pixels = np.asarray(
        coarse_diagnostics.pop("topk_selected_pixel_indices", []), dtype=np.int64
    )
    coarse_keys = np.asarray(
        coarse_diagnostics.pop("topk_selected_key_indices", []), dtype=np.int64
    )
    coarse_inliers = np.asarray(
        coarse_diagnostics.pop("topk_inlier_selected_indices", []), dtype=np.int64
    )
    diagnostics = {
        f"coarse_{key}": value for key, value in coarse_diagnostics.items()
    }
    diagnostics["coarse_fine_status"] = "coarse_failed"
    if coarse is None:
        return None, diagnostics

    fine_pixels, fine_keys, fine_confidence, fine_diagnostics = (
        _local_refinement_correspondences(
            context,
            surface,
            image_hw,
            coarse_pixels[coarse_inliers],
            coarse_keys[coarse_inliers],
            pixel_mask=pixel_mask,
            neighbors_per_inlier=int(fine_neighbors_per_inlier),
            max_correspondences=int(fine_max_correspondences),
            min_part_probability=float(min_part_probability),
        )
    )
    diagnostics.update(fine_diagnostics)
    diagnostics.update(
        {
            "topk_candidates": int(fine_diagnostics.get("fine_local_candidate_pairs", 0)),
            "topk_correspondences": int(len(fine_pixels)),
            "topk_inliers": 0,
            "topk_status": "insufficient_fine_correspondences",
        }
    )
    diagnostics["coarse_fine_status"] = "fine_correspondence_failed"
    if len(fine_pixels) < int(min_correspondences):
        return None, diagnostics

    h, w = (int(image_hw[0]), int(image_hw[1]))
    image_points = np.stack((fine_pixels % w, fine_pixels // w), axis=1).astype(np.float64)
    object_points = np.asarray(surface.points_m[fine_keys], dtype=np.float64)
    try:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points,
            image_points,
            np.asarray(K, dtype=np.float64),
            None,
            iterationsCount=int(ransac_iterations),
            reprojectionError=float(fine_ransac_reprojection_error),
            confidence=float(ransac_confidence),
            flags=cv2.SOLVEPNP_EPNP,
        )
    except cv2.error as exc:
        diagnostics["topk_status"] = f"opencv_error:{exc.code}"
        diagnostics["coarse_fine_status"] = "fine_opencv_error"
        return None, diagnostics

    inlier_idx = (
        np.empty((0,), dtype=np.int64)
        if inliers is None
        else np.asarray(inliers, dtype=np.int64).reshape(-1)
    )
    n_inliers = int(len(inlier_idx))
    inlier_fraction = float(n_inliers / max(1, len(fine_pixels)))
    diagnostics.update(
        {
            "fine_inliers": n_inliers,
            "fine_inlier_fraction": inlier_fraction,
            "fine_confidence_median": float(np.median(fine_confidence)),
            "topk_inliers": n_inliers,
            "topk_inlier_fraction": inlier_fraction,
            "topk_confidence_median": float(np.median(fine_confidence)),
        }
    )
    if return_correspondences:
        diagnostics["topk_selected_pixel_indices"] = fine_pixels.copy()
        diagnostics["topk_selected_key_indices"] = fine_keys.copy()
        diagnostics["topk_inlier_selected_indices"] = inlier_idx.copy()
    if not ok or n_inliers < int(min_inliers) or inlier_fraction < float(min_inlier_fraction):
        diagnostics["topk_status"] = "ransac_failed"
        diagnostics["fine_status"] = "ransac_failed"
        diagnostics["coarse_fine_status"] = "fine_ransac_failed"
        return None, diagnostics

    try:
        rvec, tvec = cv2.solvePnPRefineLM(
            object_points[inlier_idx],
            image_points[inlier_idx],
            np.asarray(K, dtype=np.float64),
            None,
            rvec,
            tvec,
        )
    except cv2.error:
        pass
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
    translation = np.asarray(tvec, dtype=np.float64).reshape(3)
    if not np.isfinite(rotation).all() or not np.isfinite(translation).all() or translation[2] <= 0.0:
        diagnostics["topk_status"] = "invalid_pose"
        diagnostics["fine_status"] = "invalid_pose"
        diagnostics["coarse_fine_status"] = "fine_invalid_pose"
        return None, diagnostics

    projected, _ = cv2.projectPoints(
        object_points[inlier_idx],
        np.asarray(rvec, dtype=np.float64),
        translation,
        np.asarray(K, dtype=np.float64),
        None,
    )
    residual = np.linalg.norm(projected.reshape(-1, 2) - image_points[inlier_idx], axis=1)
    diagnostics.update(
        {
            "fine_reprojection_median_px": float(np.median(residual)),
            "fine_reprojection_p90_px": float(np.quantile(residual, 0.9)),
            "fine_status": "ok",
            "topk_reprojection_median_px": float(np.median(residual)),
            "topk_reprojection_p90_px": float(np.quantile(residual, 0.9)),
            "topk_status": "ok",
            "coarse_fine_status": "ok",
        }
    )

    pose = np.concatenate((rotation, translation[:, None]), axis=1)[None]
    score, mask_score, coord_score = _score_pose_batch(
        pose,
        context.points,
        context.corr_log_score,
        context.mask_log_prob,
        context.neg_mask_log_prob,
        K,
        image_hw,
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return (
        PoseHypothesis(transform, float(score[0]), float(mask_score[0]), float(coord_score[0])),
        diagnostics,
    )


@torch.inference_mode()
def estimate_part_pose_topk_roi_ransac(
    context,
    surface,
    K,
    image_hw,
    pixel_mask=None,
    max_correspondences=512,
    min_correspondences=12,
    min_part_probability=0.05,
    margin_power=0.0,
    ransac_iterations=2000,
    ransac_reprojection_error=3.0,
    ransac_confidence=0.999,
    min_inliers=8,
    min_inlier_fraction=0.0,
    roi_refit_max_iterations=5,
    return_correspondences=False,
):
    """Refit Top-K RANSAC using only inliers whose projections remain in the ROI.

    The base correspondence selection and RANSAC path are intentionally reused
    unchanged. This wrapper only removes RANSAC inliers whose projected 3D key
    falls outside ``pixel_mask`` and repeats LM until that inlier set is stable.
    """
    if pixel_mask is None:
        raise ValueError("topk_roi_ransac requires a pixel_mask ROI")
    best, diagnostics = estimate_part_pose_topk_ransac(
        context,
        surface,
        K,
        image_hw,
        pixel_mask=pixel_mask,
        max_correspondences=max_correspondences,
        min_correspondences=min_correspondences,
        min_part_probability=min_part_probability,
        margin_power=margin_power,
        ransac_iterations=ransac_iterations,
        ransac_reprojection_error=ransac_reprojection_error,
        ransac_confidence=ransac_confidence,
        min_inliers=min_inliers,
        min_inlier_fraction=min_inlier_fraction,
        return_correspondences=True,
    )
    if best is None:
        return None, diagnostics

    selected_pixels = np.asarray(
        diagnostics.pop("topk_selected_pixel_indices"), dtype=np.int64
    )
    selected_keys = np.asarray(
        diagnostics.pop("topk_selected_key_indices"), dtype=np.int64
    )
    current_inliers = np.asarray(
        diagnostics.pop("topk_inlier_selected_indices"), dtype=np.int64
    )
    initial_inlier_count = int(len(current_inliers))
    h, w = image_hw
    roi = (
        torch.as_tensor(pixel_mask, dtype=torch.bool)
        .reshape(h, w)
        .detach()
        .cpu()
        .numpy()
    )
    object_points = np.asarray(surface.points_m[selected_keys], dtype=np.float64)
    image_points = np.stack((selected_pixels % w, selected_pixels // w), axis=1).astype(np.float64)
    rvec = cv2.Rodrigues(np.asarray(best.transform[:3, :3], dtype=np.float64))[0]
    tvec = np.asarray(best.transform[:3, 3], dtype=np.float64).reshape(3, 1)

    converged = False
    refit_count = 0
    for _ in range(max(1, int(roi_refit_max_iterations))):
        projected, _ = cv2.projectPoints(
            object_points[current_inliers],
            rvec,
            tvec,
            np.asarray(K, dtype=np.float64),
            None,
        )
        projected = projected.reshape(-1, 2)
        projected_int = np.rint(projected).astype(np.int64)
        inside_image = (
            (projected_int[:, 0] >= 0)
            & (projected_int[:, 0] < w)
            & (projected_int[:, 1] >= 0)
            & (projected_int[:, 1] < h)
        )
        inside_roi = np.zeros((len(current_inliers),), dtype=bool)
        valid_index = np.nonzero(inside_image)[0]
        inside_roi[valid_index] = roi[
            projected_int[valid_index, 1], projected_int[valid_index, 0]
        ]
        filtered_inliers = current_inliers[inside_roi]
        if len(filtered_inliers) == len(current_inliers):
            converged = True
            break
        current_inliers = filtered_inliers
        n_inliers = int(len(current_inliers))
        inlier_fraction = float(n_inliers / max(1, len(selected_pixels)))
        if n_inliers < int(min_inliers) or inlier_fraction < float(min_inlier_fraction):
            diagnostics.update(
                {
                    "topk_status": "insufficient_roi_inliers",
                    "topk_ransac_inliers_before_roi": initial_inlier_count,
                    "topk_roi_inliers": n_inliers,
                    "topk_roi_retained_fraction": float(n_inliers / max(1, initial_inlier_count)),
                }
            )
            return None, diagnostics
        try:
            rvec, tvec = cv2.solvePnPRefineLM(
                object_points[current_inliers],
                image_points[current_inliers],
                np.asarray(K, dtype=np.float64),
                None,
                rvec,
                tvec,
            )
        except cv2.error:
            diagnostics["topk_status"] = "roi_refit_opencv_error"
            return None, diagnostics
        refit_count += 1

    if not converged:
        diagnostics["topk_status"] = "roi_inliers_not_stable"
        return None, diagnostics

    n_inliers = int(len(current_inliers))
    inlier_fraction = float(n_inliers / max(1, len(selected_pixels)))
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
    translation = np.asarray(tvec, dtype=np.float64).reshape(3)
    if not np.isfinite(rotation).all() or not np.isfinite(translation).all() or translation[2] <= 0.0:
        diagnostics["topk_status"] = "invalid_roi_refit_pose"
        return None, diagnostics

    projected, _ = cv2.projectPoints(
        object_points[current_inliers],
        np.asarray(rvec, dtype=np.float64),
        translation,
        np.asarray(K, dtype=np.float64),
        None,
    )
    residual = np.linalg.norm(
        projected.reshape(-1, 2) - image_points[current_inliers], axis=1
    )
    diagnostics.update(
        {
            "topk_inliers": n_inliers,
            "topk_inlier_fraction": inlier_fraction,
            "topk_ransac_inliers_before_roi": initial_inlier_count,
            "topk_roi_inliers": n_inliers,
            "topk_roi_removed": int(initial_inlier_count - n_inliers),
            "topk_roi_retained_fraction": float(n_inliers / max(1, initial_inlier_count)),
            "topk_roi_refit_count": int(refit_count),
            "topk_reprojection_median_px": float(np.median(residual)),
            "topk_reprojection_p90_px": float(np.quantile(residual, 0.9)),
            "topk_status": "ok",
        }
    )
    if return_correspondences:
        diagnostics["topk_selected_pixel_indices"] = selected_pixels.copy()
        diagnostics["topk_selected_key_indices"] = selected_keys.copy()
        diagnostics["topk_inlier_selected_indices"] = current_inliers.copy()

    pose = np.concatenate((rotation, translation[:, None]), axis=1)[None]
    score, mask_score, coord_score = _score_pose_batch(
        pose,
        context.points,
        context.corr_log_score,
        context.mask_log_prob,
        context.neg_mask_log_prob,
        K,
        image_hw,
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return (
        PoseHypothesis(transform, float(score[0]), float(mask_score[0]), float(coord_score[0])),
        diagnostics,
    )


@torch.inference_mode()
def estimate_part_pose_spatial_topk_ransac(
    context,
    surface,
    K,
    image_hw,
    pixel_mask=None,
    max_correspondences=512,
    min_correspondences=12,
    min_part_probability=0.05,
    margin_power=0.0,
    grid_size=8,
    max_per_grid_cell=12,
    fps_2d_weight=0.5,
    fps_3d_weight=0.5,
    fps_confidence_power=0.25,
    ransac_iterations=2000,
    ransac_reprojection_error=1.5,
    ransac_confidence=0.999,
    min_inliers=8,
    min_inlier_fraction=0.6,
    min_inlier_hull_fraction=0.15,
    min_inlier_axis_span=0.25,
    min_inlier_3d_span=0.20,
    max_reprojection_median=1.5,
    return_correspondences=False,
):
    """Fit a rigid part from confidence-ranked, spatially balanced matches.

    This is deliberately separate from ``estimate_part_pose_topk_ransac`` so
    existing evaluation remains bit-for-bit unchanged. Candidates are first
    stratified over the predicted 2D ROI, then selected by confidence-weighted
    farthest-point sampling in joint normalized 2D/3D space.
    """
    h, w = image_hw
    if h != w:
        raise ValueError(f"SurfEmb pose estimator expects a square crop, got {(h, w)}")
    if int(grid_size) <= 0 or int(max_per_grid_cell) <= 0:
        raise ValueError("grid_size and max_per_grid_cell must be positive")
    if float(fps_2d_weight) < 0.0 or float(fps_3d_weight) < 0.0:
        raise ValueError("FPS weights must be non-negative")
    if float(fps_2d_weight) + float(fps_3d_weight) <= 0.0:
        raise ValueError("At least one FPS distance weight must be positive")

    corr_prob = context.corr_prob
    if corr_prob.ndim != 2 or corr_prob.shape[0] != h * w:
        raise ValueError(f"Unexpected correspondence probability shape {tuple(corr_prob.shape)}")

    top_count = min(2, int(corr_prob.shape[1]))
    top_prob, top_key = torch.topk(corr_prob, k=top_count, dim=1, largest=True, sorted=True)
    joint_confidence = top_prob[:, 0]
    part_probability = corr_prob.sum(dim=1)
    confidence = joint_confidence
    if float(margin_power) > 0.0 and top_count > 1:
        margin = (1.0 - top_prob[:, 1] / joint_confidence.clamp_min(1e-12)).clamp_min(0.0)
        confidence = confidence * margin.pow(float(margin_power))

    eligible = torch.isfinite(confidence) & (part_probability >= float(min_part_probability))
    if pixel_mask is not None:
        pixel_mask = torch.as_tensor(pixel_mask, device=eligible.device, dtype=torch.bool).reshape(-1)
        if pixel_mask.numel() != eligible.numel():
            raise ValueError(f"pixel_mask has {pixel_mask.numel()} values, expected {eligible.numel()}")
        eligible &= pixel_mask
    candidate_idx = torch.nonzero(eligible, as_tuple=False).reshape(-1)
    diagnostics = {
        "topk_candidates": int(len(candidate_idx)),
        "topk_correspondences": 0,
        "topk_inliers": 0,
        "topk_status": "insufficient_candidates",
        "spatial_grid_size": int(grid_size),
        "spatial_max_per_grid_cell": int(max_per_grid_cell),
    }
    if len(candidate_idx) < int(min_correspondences):
        return None, diagnostics

    candidate_confidence = confidence[candidate_idx]
    order = candidate_confidence.argsort(descending=True)
    pixels_sorted = candidate_idx[order].cpu().numpy().astype(np.int64, copy=False)
    keys_sorted = top_key[candidate_idx[order], 0].cpu().numpy().astype(np.int64, copy=False)
    confidence_sorted = candidate_confidence[order].cpu().numpy().astype(np.float64, copy=False)

    # Keep the highest-confidence occurrence of each 3D key.
    unique_pixels = []
    unique_keys = []
    unique_confidence = []
    seen_keys = set()
    for pixel_index, key_index, value in zip(pixels_sorted, keys_sorted, confidence_sorted):
        key_index = int(key_index)
        if key_index in seen_keys:
            continue
        seen_keys.add(key_index)
        unique_pixels.append(int(pixel_index))
        unique_keys.append(key_index)
        unique_confidence.append(float(value))
    diagnostics["spatial_unique_keys"] = int(len(unique_pixels))
    if len(unique_pixels) < int(min_correspondences):
        diagnostics["topk_status"] = "insufficient_unique_keys"
        return None, diagnostics

    unique_pixels = np.asarray(unique_pixels, dtype=np.int64)
    unique_keys = np.asarray(unique_keys, dtype=np.int64)
    unique_confidence = np.asarray(unique_confidence, dtype=np.float64)
    unique_xy = np.stack((unique_pixels % w, unique_pixels // w), axis=1).astype(np.float64)

    # The grid is defined over the actual eligible ROI rather than the full crop.
    roi_pixels = candidate_idx.cpu().numpy().astype(np.int64, copy=False)
    roi_xy = np.stack((roi_pixels % w, roi_pixels // w), axis=1).astype(np.float64)
    roi_min = roi_xy.min(axis=0)
    roi_max = roi_xy.max(axis=0)
    roi_extent = np.maximum(roi_max - roi_min + 1.0, 1.0)
    grid_xy = np.floor((unique_xy - roi_min) / roi_extent * int(grid_size)).astype(np.int64)
    grid_xy = np.clip(grid_xy, 0, int(grid_size) - 1)
    grid_counts = np.zeros((int(grid_size), int(grid_size)), dtype=np.int64)
    grid_keep = []
    for index, (grid_x, grid_y) in enumerate(grid_xy):
        if grid_counts[grid_y, grid_x] >= int(max_per_grid_cell):
            continue
        grid_counts[grid_y, grid_x] += 1
        grid_keep.append(index)
    grid_keep = np.asarray(grid_keep, dtype=np.int64)
    diagnostics["spatial_occupied_grid_cells"] = int(np.count_nonzero(grid_counts))
    diagnostics["spatial_grid_candidates"] = int(len(grid_keep))
    if len(grid_keep) < int(min_correspondences):
        diagnostics["topk_status"] = "insufficient_grid_candidates"
        return None, diagnostics

    grid_pixels = unique_pixels[grid_keep]
    grid_keys = unique_keys[grid_keep]
    grid_confidence = unique_confidence[grid_keep]
    grid_xy_points = unique_xy[grid_keep]
    grid_xyz_points = np.asarray(surface.points_m[grid_keys], dtype=np.float64)

    # Joint 2D/3D FPS prevents a dense high-confidence patch from dominating.
    xy_scale = max(float(np.linalg.norm(roi_extent)), 1e-12)
    surface_points = np.asarray(surface.points_m, dtype=np.float64)
    xyz_scale = max(float(np.linalg.norm(np.ptp(surface_points, axis=0))), 1e-12)
    descriptor = np.concatenate(
        (
            math.sqrt(float(fps_2d_weight)) * grid_xy_points / xy_scale,
            math.sqrt(float(fps_3d_weight)) * grid_xyz_points / xyz_scale,
        ),
        axis=1,
    )
    target_count = min(int(max_correspondences), len(grid_keep))
    selected = [int(np.argmax(grid_confidence))]
    min_distance_sq = np.full((len(grid_keep),), np.inf, dtype=np.float64)
    confidence_weight = np.power(
        np.clip(grid_confidence / max(float(grid_confidence.max()), 1e-12), 1e-12, 1.0),
        float(fps_confidence_power),
    )
    while len(selected) < target_count:
        latest = descriptor[selected[-1]]
        distance_sq = np.sum((descriptor - latest[None]) ** 2, axis=1)
        min_distance_sq = np.minimum(min_distance_sq, distance_sq)
        min_distance_sq[np.asarray(selected, dtype=np.int64)] = -1.0
        score = min_distance_sq * confidence_weight
        next_index = int(np.argmax(score))
        if score[next_index] < 0.0:
            break
        selected.append(next_index)
    selected = np.asarray(selected, dtype=np.int64)
    selected_pixels = grid_pixels[selected]
    selected_keys = grid_keys[selected]
    selected_confidence = grid_confidence[selected]
    object_points = grid_xyz_points[selected]
    image_points = grid_xy_points[selected]
    diagnostics["topk_correspondences"] = int(len(selected_pixels))
    if len(selected_pixels) < int(min_correspondences):
        diagnostics["topk_status"] = "insufficient_spatial_correspondences"
        return None, diagnostics

    try:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points,
            image_points,
            np.asarray(K, dtype=np.float64),
            None,
            iterationsCount=int(ransac_iterations),
            reprojectionError=float(ransac_reprojection_error),
            confidence=float(ransac_confidence),
            flags=cv2.SOLVEPNP_EPNP,
        )
    except cv2.error as exc:
        diagnostics["topk_status"] = f"opencv_error:{exc.code}"
        return None, diagnostics

    inlier_idx = (
        np.empty((0,), dtype=np.int64)
        if inliers is None
        else np.asarray(inliers, dtype=np.int64).reshape(-1)
    )
    n_inliers = int(len(inlier_idx))
    inlier_fraction = float(n_inliers / max(1, len(selected_pixels)))
    diagnostics["topk_inliers"] = n_inliers
    diagnostics["topk_inlier_fraction"] = inlier_fraction
    diagnostics["topk_confidence_median"] = float(np.median(selected_confidence))
    if return_correspondences:
        diagnostics["topk_selected_pixel_indices"] = selected_pixels.copy()
        diagnostics["topk_selected_key_indices"] = selected_keys.copy()
        diagnostics["topk_inlier_selected_indices"] = inlier_idx.copy()
    if not ok or n_inliers < int(min_inliers) or inlier_fraction < float(min_inlier_fraction):
        diagnostics["topk_status"] = "ransac_failed"
        return None, diagnostics

    try:
        rvec, tvec = cv2.solvePnPRefineLM(
            object_points[inlier_idx],
            image_points[inlier_idx],
            np.asarray(K, dtype=np.float64),
            None,
            rvec,
            tvec,
        )
    except cv2.error:
        pass
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
    translation = np.asarray(tvec, dtype=np.float64).reshape(3)
    if not np.isfinite(rotation).all() or not np.isfinite(translation).all() or translation[2] <= 0.0:
        diagnostics["topk_status"] = "invalid_pose"
        return None, diagnostics

    inlier_image_points = image_points[inlier_idx]
    inlier_object_points = object_points[inlier_idx]
    projected, _ = cv2.projectPoints(
        inlier_object_points,
        np.asarray(rvec, dtype=np.float64),
        translation,
        np.asarray(K, dtype=np.float64),
        None,
    )
    residual = np.linalg.norm(projected.reshape(-1, 2) - inlier_image_points, axis=1)
    reprojection_median = float(np.median(residual))
    diagnostics["topk_reprojection_median_px"] = reprojection_median
    diagnostics["topk_reprojection_p90_px"] = float(np.quantile(residual, 0.9))

    if len(inlier_image_points) >= 3:
        hull = cv2.convexHull(inlier_image_points.astype(np.float32))
        hull_area = float(cv2.contourArea(hull))
    else:
        hull_area = 0.0
    roi_area = max(float(len(roi_pixels)), 1.0)
    hull_fraction = hull_area / roi_area
    inlier_extent = np.ptp(inlier_image_points, axis=0) if len(inlier_image_points) else np.zeros(2)
    axis_span = inlier_extent / roi_extent
    surface_span = max(float(np.linalg.norm(np.ptp(surface_points, axis=0))), 1e-12)
    inlier_3d_span = float(np.linalg.norm(np.ptp(inlier_object_points, axis=0)) / surface_span)
    diagnostics["spatial_inlier_hull_fraction"] = hull_fraction
    diagnostics["spatial_inlier_x_span"] = float(axis_span[0])
    diagnostics["spatial_inlier_y_span"] = float(axis_span[1])
    diagnostics["spatial_inlier_3d_span"] = inlier_3d_span

    coverage_ok = (
        hull_fraction >= float(min_inlier_hull_fraction)
        and float(axis_span[0]) >= float(min_inlier_axis_span)
        and float(axis_span[1]) >= float(min_inlier_axis_span)
        and inlier_3d_span >= float(min_inlier_3d_span)
    )
    if not coverage_ok:
        diagnostics["topk_status"] = "insufficient_inlier_coverage"
        return None, diagnostics
    if reprojection_median > float(max_reprojection_median):
        diagnostics["topk_status"] = "high_reprojection_error"
        return None, diagnostics

    diagnostics["topk_status"] = "ok"
    pose = np.concatenate((rotation, translation[:, None]), axis=1)[None]
    score, mask_score, coord_score = _score_pose_batch(
        pose,
        context.points,
        context.corr_log_score,
        context.mask_log_prob,
        context.neg_mask_log_prob,
        K,
        image_hw,
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return (
        PoseHypothesis(transform, float(score[0]), float(mask_score[0]), float(coord_score[0])),
        diagnostics,
    )


def filter_correspondences_by_triangle_visibility(
    surface,
    part_name,
    transform,
    selected_pixels,
    selected_keys,
    selected_indices,
    K,
    image_hw,
    raster_part_ids,
    raster_depth,
    raster_valid,
    depth_tolerance=0.0008,
):
    """Keep candidate correspondences whose 3D keys are front-most mesh points."""
    h, w = (int(image_hw[0]), int(image_hw[1]))
    if part_name not in PART_RASTER_IDS:
        raise ValueError(f"Unknown part {part_name!r}")
    selected_pixels = np.asarray(selected_pixels, dtype=np.int64).reshape(-1)
    selected_keys = np.asarray(selected_keys, dtype=np.int64).reshape(-1)
    selected_indices = np.asarray(selected_indices, dtype=np.int64).reshape(-1)
    if len(selected_pixels) != len(selected_keys):
        raise ValueError("selected_pixels and selected_keys must have equal length")
    if np.any((selected_indices < 0) | (selected_indices >= len(selected_pixels))):
        raise ValueError("selected_indices contains an out-of-range correspondence index")

    pixels = selected_pixels[selected_indices]
    keys = selected_keys[selected_indices]
    points = np.asarray(surface.points_m[keys], dtype=np.float64)
    normals = np.asarray(surface.normals[keys], dtype=np.float64)
    transform = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    points_cam = points @ transform[:3, :3].T + transform[:3, 3]
    normals_cam = normals @ transform[:3, :3].T
    z = points_cam[:, 2]
    uvw = points_cam @ np.asarray(K, dtype=np.float64).reshape(3, 3).T
    uv = uvw[:, :2] / np.maximum(uvw[:, 2:], 1e-12)
    u = np.rint(uv[:, 0]).astype(np.int64)
    v = np.rint(uv[:, 1]).astype(np.int64)
    finite = np.isfinite(uv).all(axis=1) & np.isfinite(z)
    in_frame = finite & (z > 0.0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    front_facing = np.sum(normals_cam * points_cam, axis=1) < 0.0

    raster_part_ids = np.asarray(raster_part_ids, dtype=np.uint8)
    raster_depth = np.asarray(raster_depth, dtype=np.float64)
    raster_valid = np.asarray(raster_valid, dtype=bool)
    if raster_part_ids.shape != (h, w) or raster_depth.shape != (h, w) or raster_valid.shape != (h, w):
        raise ValueError("Visibility raster shape does not match image_hw")
    raster_match = np.zeros(len(keys), dtype=bool)
    depth_match = np.zeros(len(keys), dtype=bool)
    in_frame_idx = np.flatnonzero(in_frame)
    if len(in_frame_idx):
        rv = v[in_frame_idx]
        ru = u[in_frame_idx]
        raster_match[in_frame_idx] = (
            raster_valid[rv, ru]
            & (raster_part_ids[rv, ru] == int(PART_RASTER_IDS[part_name]))
        )
        depth_at_key = raster_depth[rv, ru]
        depth_match[in_frame_idx] = (
            depth_at_key > 0.0
        ) & (z[in_frame_idx] <= depth_at_key + float(depth_tolerance))

    visible = in_frame & front_facing & raster_match & depth_match
    diagnostics = {
        "visibility_input_inliers": int(len(keys)),
        "visibility_front_facing": int(np.count_nonzero(in_frame & front_facing)),
        "visibility_backface_rejected": int(np.count_nonzero(in_frame & ~front_facing)),
        "visibility_out_of_frame_rejected": int(np.count_nonzero(~in_frame)),
        "visibility_occlusion_rejected": int(np.count_nonzero(in_frame & front_facing & ~(raster_match & depth_match))),
        "visibility_kept": int(np.count_nonzero(visible)),
        "visibility_keep_fraction": float(np.mean(visible)) if len(visible) else 0.0,
    }
    return pixels[visible], keys[visible], diagnostics


@torch.inference_mode()
def refine_part_pose_with_triangle_visibility(
    context,
    surface,
    part_name,
    initial_hypothesis,
    initial_diagnostics,
    K,
    image_hw,
    raster_part_ids,
    raster_depth,
    raster_valid,
    depth_tolerance=0.0008,
    min_correspondences=12,
    ransac_iterations=2000,
    ransac_reprojection_error=3.0,
    ransac_confidence=0.999,
    min_inliers=8,
    min_inlier_fraction=0.6,
):
    """Filter an initial PnP fit with mesh visibility, then rerun PnP once."""
    required = (
        "topk_selected_pixel_indices",
        "topk_selected_key_indices",
        "topk_inlier_selected_indices",
    )
    missing = [key for key in required if key not in initial_diagnostics]
    if missing:
        raise ValueError(f"Initial PnP diagnostics are missing {missing}; use return_correspondences=True")
    pixels, keys, diagnostics = filter_correspondences_by_triangle_visibility(
        surface,
        part_name,
        initial_hypothesis.transform,
        initial_diagnostics["topk_selected_pixel_indices"],
        initial_diagnostics["topk_selected_key_indices"],
        initial_diagnostics["topk_inlier_selected_indices"],
        K,
        image_hw,
        raster_part_ids,
        raster_depth,
        raster_valid,
        depth_tolerance=depth_tolerance,
    )
    diagnostics["visibility_status"] = "insufficient_visible_correspondences"
    if len(pixels) < int(min_correspondences):
        return None, diagnostics

    h, w = (int(image_hw[0]), int(image_hw[1]))
    image_points = np.stack((pixels % w, pixels // w), axis=1).astype(np.float64)
    object_points = np.asarray(surface.points_m[keys], dtype=np.float64)
    try:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points,
            image_points,
            np.asarray(K, dtype=np.float64),
            None,
            iterationsCount=int(ransac_iterations),
            reprojectionError=float(ransac_reprojection_error),
            confidence=float(ransac_confidence),
            flags=cv2.SOLVEPNP_EPNP,
        )
    except cv2.error as exc:
        diagnostics["visibility_status"] = f"opencv_error:{exc.code}"
        return None, diagnostics
    n_inliers = 0 if inliers is None else int(len(inliers))
    inlier_fraction = float(n_inliers / max(1, len(pixels)))
    diagnostics["visibility_refit_inliers"] = n_inliers
    diagnostics["visibility_refit_inlier_fraction"] = inlier_fraction
    if not ok or n_inliers < int(min_inliers) or inlier_fraction < float(min_inlier_fraction):
        diagnostics["visibility_status"] = "visibility_refit_ransac_failed"
        return None, diagnostics

    inlier_idx = np.asarray(inliers, dtype=np.int64).reshape(-1)
    try:
        rvec, tvec = cv2.solvePnPRefineLM(
            object_points[inlier_idx],
            image_points[inlier_idx],
            np.asarray(K, dtype=np.float64),
            None,
            rvec,
            tvec,
        )
    except cv2.error:
        pass
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
    translation = np.asarray(tvec, dtype=np.float64).reshape(3)
    if not np.isfinite(rotation).all() or not np.isfinite(translation).all() or translation[2] <= 0.0:
        diagnostics["visibility_status"] = "visibility_refit_invalid_pose"
        return None, diagnostics
    projected, _ = cv2.projectPoints(
        object_points[inlier_idx],
        np.asarray(rvec, dtype=np.float64),
        translation,
        np.asarray(K, dtype=np.float64),
        None,
    )
    residual = np.linalg.norm(projected.reshape(-1, 2) - image_points[inlier_idx], axis=1)
    diagnostics["visibility_refit_reprojection_median_px"] = float(np.median(residual))
    diagnostics["visibility_refit_reprojection_p90_px"] = float(np.quantile(residual, 0.9))
    diagnostics["visibility_status"] = "ok"

    pose = np.concatenate((rotation, translation[:, None]), axis=1)[None]
    score, mask_score, coord_score = _score_pose_batch(
        pose,
        context.points,
        context.corr_log_score,
        context.mask_log_prob,
        context.neg_mask_log_prob,
        K,
        image_hw,
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return (
        PoseHypothesis(transform, float(score[0]), float(mask_score[0]), float(coord_score[0])),
        diagnostics,
    )


@torch.inference_mode()
def estimate_part_pose(
    query_flat,
    probability_input,
    surface,
    K,
    image_hw,
    max_poses=1024,
    max_pose_evaluations=256,
    pose_batch_size=64,
    top_k=5,
    alpha=1.5,
    dist_2d_min=0.1,
    seed=0,
):
    context = prepare_part_score_context(query_flat, probability_input, surface, image_hw)
    return estimate_part_pose_from_context(
        context,
        surface,
        K,
        image_hw,
        max_poses=max_poses,
        max_pose_evaluations=max_pose_evaluations,
        pose_batch_size=pose_batch_size,
        top_k=top_k,
        alpha=alpha,
        dist_2d_min=dist_2d_min,
        seed=seed,
    )


def _wrap_angle(value):
    return math.atan2(math.sin(float(value)), math.cos(float(value)))


def _relative_joint(part_name, wrist_transform, part_transform):
    relative = np.linalg.inv(wrist_transform) @ part_transform
    if part_name == "shaft":
        alpha = -math.atan2(float(relative[0, 2]), float(relative[0, 0]))
        joint = make_transform_np(
            Rotation.from_rotvec([0.0, alpha, 0.0]).as_matrix(),
            [0.2159, 0.0, 0.0],
        )
        expected = np.linalg.inv(joint)
        return _wrap_angle(alpha), expected
    angle = math.atan2(float(relative[1, 0]), float(relative[0, 0]))
    if part_name == "l_gripper":
        theta = _wrap_angle(angle)
        expected = make_transform_np(Rotation.from_rotvec([0.0, 0.0, theta]).as_matrix(), [GRIPPER_JOINT_OFFSET_M, 0.0, 0.0])
        return theta, expected
    theta = _wrap_angle(-angle)
    expected = make_transform_np(Rotation.from_rotvec([0.0, 0.0, -theta]).as_matrix(), [GRIPPER_JOINT_OFFSET_M, 0.0, 0.0])
    return theta, expected


def _rotation_residual(R_a, R_b):
    return Rotation.from_matrix(np.asarray(R_a).T @ np.asarray(R_b)).as_rotvec()


def _compatibility_cost(part_name, wrist_hypothesis, part_hypothesis):
    _, expected = _relative_joint(part_name, wrist_hypothesis.transform, part_hypothesis.transform)
    relative = np.linalg.inv(wrist_hypothesis.transform) @ part_hypothesis.transform
    rot = np.linalg.norm(_rotation_residual(expected[:3, :3], relative[:3, :3]))
    trans = np.linalg.norm(expected[:3, 3] - relative[:3, 3]) / 0.01
    score_gap = max(0.0, -float(part_hypothesis.score)) * 0.02
    return float(rot + trans + score_gap)


def _matrix_to_quat_wxyz(matrix):
    q_xyzw = Rotation.from_matrix(np.asarray(matrix, dtype=np.float64)).as_quat()
    quat = np.asarray([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64)
    if quat[0] < 0.0:
        quat = -quat
    return quat


def select_chain_consistent_hypotheses(hypotheses):
    wrists = hypotheses.get("wrist", [])
    if not wrists:
        raise RuntimeError("No valid wrist SurfEmb pose hypothesis")
    best = None
    for wrist in wrists:
        selected = {"wrist": wrist}
        cost = max(0.0, -float(wrist.score)) * 0.02
        for name in ("shaft", "l_gripper", "r_gripper"):
            candidates = hypotheses.get(name, [])
            if not candidates:
                continue
            candidate = min(candidates, key=lambda item: _compatibility_cost(name, wrist, item))
            selected[name] = candidate
            cost += _compatibility_cost(name, wrist, candidate)
        if best is None or cost < best[0]:
            best = (cost, selected)
    return best[1], float(best[0])


def fit_kinematic_chain(selected):
    wrist = selected["wrist"].transform
    rvec = Rotation.from_matrix(wrist[:3, :3]).as_rotvec()
    action0 = {"shaft": 0.0, "l_gripper": 0.0, "r_gripper": 0.0}
    for name in action0:
        if name in selected:
            action0[name], _ = _relative_joint(name, wrist, selected[name].transform)
    x0 = np.concatenate(
        [rvec, wrist[:3, 3], [action0["shaft"], action0["l_gripper"], action0["r_gripper"]]]
    )
    weights = {"wrist": 2.0, "shaft": 1.0, "l_gripper": 0.75, "r_gripper": 0.75}
    translation_scales = {"wrist": 0.01, "shaft": 0.02, "l_gripper": 0.008, "r_gripper": 0.008}

    def residual(params):
        R = Rotation.from_rotvec(params[:3]).as_matrix()
        quat = _matrix_to_quat_wxyz(R)
        predicted = fk_matrices_np(quat, params[3:6], params[6], params[7], params[8])
        values = []
        for name, hypothesis in selected.items():
            weight = math.sqrt(weights[name])
            observed = hypothesis.transform
            values.extend((weight * _rotation_residual(predicted[name][:3, :3], observed[:3, :3])).tolist())
            values.extend((weight * (predicted[name][:3, 3] - observed[:3, 3]) / translation_scales[name]).tolist())
        return np.asarray(values, dtype=np.float64)

    lower = np.asarray([-np.inf] * 6 + [-math.pi, -math.pi, -math.pi], dtype=np.float64)
    upper = np.asarray([np.inf] * 6 + [math.pi, math.pi, math.pi], dtype=np.float64)
    result = least_squares(residual, x0, bounds=(lower, upper), loss="soft_l1", f_scale=1.0, max_nfev=100)
    R = Rotation.from_rotvec(result.x[:3]).as_matrix()
    return {
        "rot": _matrix_to_quat_wxyz(R),
        "trans": result.x[3:6].astype(np.float64),
        "alpha": _wrap_angle(result.x[6]),
        "theta_l": _wrap_angle(result.x[7]),
        "theta_r": _wrap_angle(result.x[8]),
        "chain_cost": float(np.mean(np.square(residual(result.x)))),
        "chain_nfev": int(result.nfev),
    }


@torch.inference_mode()
def score_part_transforms(context, transforms, K, image_hw, pose_batch_size=64):
    matrices = np.asarray(transforms, dtype=np.float64).reshape(-1, 4, 4)
    poses = matrices[:, :3, :4]
    score_chunks = []
    mask_chunks = []
    coord_chunks = []
    for start in range(0, len(poses), int(pose_batch_size)):
        score, mask_score, coord_score = _score_pose_batch(
            poses[start : start + int(pose_batch_size)],
            context.points,
            context.corr_log_score,
            context.mask_log_prob,
            context.neg_mask_log_prob,
            K,
            image_hw,
        )
        score_chunks.append(score.cpu())
        mask_chunks.append(mask_score.cpu())
        coord_chunks.append(coord_score.cpu())
    return (
        torch.cat(score_chunks).numpy(),
        torch.cat(mask_chunks).numpy(),
        torch.cat(coord_chunks).numpy(),
    )


def _part_transform_from_wrist(wrist_transform, part_name, angle):
    wrist_transform = np.asarray(wrist_transform, dtype=np.float64).reshape(4, 4)
    if part_name == "shaft":
        joint = make_transform_np(
            Rotation.from_rotvec([0.0, float(angle), 0.0]).as_matrix(),
            [0.2159, 0.0, 0.0],
        )
        return wrist_transform @ np.linalg.inv(joint)
    signed_angle = float(angle) if part_name == "l_gripper" else -float(angle)
    joint = make_transform_np(
        Rotation.from_rotvec([0.0, 0.0, signed_angle]).as_matrix(),
        [GRIPPER_JOINT_OFFSET_M, 0.0, 0.0],
    )
    return wrist_transform @ joint


@torch.inference_mode()
def estimate_chain_joint_angle(
    context,
    wrist_transform,
    part_name,
    K,
    image_hw,
    pose_batch_size=64,
    coarse_steps=73,
    fine_steps=17,
):
    coarse = np.linspace(-math.pi, math.pi, int(coarse_steps), endpoint=False, dtype=np.float64)
    transforms = [_part_transform_from_wrist(wrist_transform, part_name, angle) for angle in coarse]
    scores, mask_scores, coord_scores = score_part_transforms(
        context, transforms, K, image_hw, pose_batch_size=pose_batch_size
    )
    if not np.isfinite(scores).any():
        raise RuntimeError(f"No finite chain-constrained score for {part_name}")
    coarse_index = int(np.nanargmax(scores))
    step = 2.0 * math.pi / float(coarse_steps)
    fine = coarse[coarse_index] + np.linspace(-step, step, int(fine_steps), dtype=np.float64)
    fine = np.asarray([_wrap_angle(value) for value in fine], dtype=np.float64)
    fine_transforms = [_part_transform_from_wrist(wrist_transform, part_name, angle) for angle in fine]
    fine_scores, fine_mask, fine_coord = score_part_transforms(
        context, fine_transforms, K, image_hw, pose_batch_size=pose_batch_size
    )
    fine_index = int(np.nanargmax(fine_scores))
    return {
        "angle": _wrap_angle(fine[fine_index]),
        "score": float(fine_scores[fine_index]),
        "mask_score": float(fine_mask[fine_index]),
        "coord_score": float(fine_coord[fine_index]),
        "evaluations": int(len(coarse) + len(fine)),
    }


@torch.inference_mode()
def fit_chain_by_part_probability(wrist_hypotheses, contexts, K, image_hw, pose_batch_size=64):
    if not wrist_hypotheses:
        raise RuntimeError("No valid wrist SurfEmb pose hypothesis")
    best = None
    for wrist_rank, wrist_hypothesis in enumerate(wrist_hypotheses):
        joint_results = {}
        total_score = float(wrist_hypothesis.score)
        for part_name in ("shaft", "l_gripper", "r_gripper"):
            result = estimate_chain_joint_angle(
                contexts[part_name],
                wrist_hypothesis.transform,
                part_name,
                K,
                image_hw,
                pose_batch_size=pose_batch_size,
            )
            joint_results[part_name] = result
            total_score += result["score"]
        if best is None or total_score > best[0]:
            best = (total_score, wrist_rank, wrist_hypothesis, joint_results)
    total_score, wrist_rank, wrist_hypothesis, joint_results = best
    wrist = wrist_hypothesis.transform
    pose = {
        "rot": _matrix_to_quat_wxyz(wrist[:3, :3]),
        "trans": wrist[:3, 3].copy(),
        "alpha": joint_results["shaft"]["angle"],
        "theta_l": joint_results["l_gripper"]["angle"],
        "theta_r": joint_results["r_gripper"]["angle"],
        "chain_cost": float(-total_score),
        "chain_nfev": int(sum(item["evaluations"] for item in joint_results.values())),
    }
    diagnostics = {"selected_wrist_rank": int(wrist_rank), "chain_total_score": float(total_score)}
    for part_name, result in joint_results.items():
        diagnostics[f"{part_name}_chain_score"] = result["score"]
        diagnostics[f"{part_name}_chain_mask_score"] = result["mask_score"]
        diagnostics[f"{part_name}_chain_coord_score"] = result["coord_score"]
    return pose, diagnostics


@torch.inference_mode()
def estimate_articulated_pose(
    mask_logits,
    query_chw,
    model,
    surfaces,
    K,
    max_poses=1024,
    max_pose_evaluations=256,
    pose_batch_size=64,
    top_k=5,
    down_sample_scale=3,
    seed=0,
    require_all_parts=True,
    pose_estimator="probability_ap3p",
    topk_max_correspondences=512,
    topk_min_correspondences=12,
    topk_min_part_probability=0.05,
    topk_min_object_probability=0.5,
    topk_margin_power=0.0,
    topk_ransac_iterations=2000,
    topk_ransac_reprojection_error=3.0,
    topk_ransac_confidence=0.999,
    topk_min_inliers=8,
    topk_min_inlier_fraction=0.6,
):
    query_flat, part_inputs, K_ds, image_hw, object_prob = build_part_probability_inputs(
        mask_logits, query_chw, surfaces, K, down_sample_scale=down_sample_scale
    )
    contexts = {
        name: prepare_part_score_context(query_flat, part_inputs[name], surfaces[name], image_hw)
        for name in PART_NAMES
    }
    hypotheses = {}
    diagnostics = {"pred_mask_area_ds": float((object_prob > 0.5).sum().item())}
    if pose_estimator == "topk_ransac":
        part_probability = torch.stack([part_inputs[name]["prob"] for name in PART_NAMES], dim=1)
        predicted_wrist_roi = (
            (part_probability.argmax(dim=1) == PART_NAMES.index("wrist"))
            & (object_prob >= float(topk_min_object_probability))
        )
        wrist_hypothesis, topk_diagnostics = estimate_part_pose_topk_ransac(
            contexts["wrist"],
            surfaces["wrist"],
            K_ds,
            image_hw,
            pixel_mask=predicted_wrist_roi,
            max_correspondences=topk_max_correspondences,
            min_correspondences=topk_min_correspondences,
            min_part_probability=topk_min_part_probability,
            margin_power=topk_margin_power,
            ransac_iterations=topk_ransac_iterations,
            ransac_reprojection_error=topk_ransac_reprojection_error,
            ransac_confidence=topk_ransac_confidence,
            min_inliers=topk_min_inliers,
            min_inlier_fraction=topk_min_inlier_fraction,
        )
        diagnostics.update(topk_diagnostics)
        diagnostics["pred_wrist_area_ds"] = int(predicted_wrist_roi.sum().item())
        if wrist_hypothesis is None:
            raise RuntimeError(f"No valid wrist top-K RANSAC pose: {topk_diagnostics}")
        wrist_hypotheses = [wrist_hypothesis]
    elif pose_estimator == "probability_ap3p":
        wrist_hypotheses = estimate_part_pose_from_context(
            contexts["wrist"],
            surfaces["wrist"],
            K_ds,
            image_hw,
            max_poses=max_poses,
            max_pose_evaluations=max_pose_evaluations,
            pose_batch_size=pose_batch_size,
            top_k=top_k,
            seed=int(seed) + 100003,
        )
    else:
        raise ValueError(f"Unknown pose_estimator {pose_estimator!r}")
    diagnostics["pose_estimator"] = pose_estimator
    hypotheses["wrist"] = wrist_hypotheses
    diagnostics["wrist_num_hypotheses"] = len(wrist_hypotheses)
    diagnostics["wrist_best_score"] = float(wrist_hypotheses[0].score) if wrist_hypotheses else float("nan")
    pose, chain_diagnostics = fit_chain_by_part_probability(
        wrist_hypotheses,
        contexts,
        K_ds,
        image_hw,
        pose_batch_size=pose_batch_size,
    )
    diagnostics.update(chain_diagnostics)
    diagnostics["selected_parts"] = list(PART_NAMES)
    return pose, diagnostics, hypotheses


def part_pose_errors(pred_pose, gt_pose):
    pred = fk_matrices_np(
        pred_pose["rot"], pred_pose["trans"], pred_pose["alpha"], pred_pose["theta_l"], pred_pose["theta_r"]
    )
    gt = fk_matrices_np(gt_pose["rot"], gt_pose["trans"], gt_pose["alpha"], gt_pose["theta_l"], gt_pose["theta_r"])
    out = {}
    for name in PART_NAMES:
        out[f"{name}_trans_err_mm"] = float(np.linalg.norm(pred[name][:3, 3] - gt[name][:3, 3]) * 1000.0)
        out[f"{name}_rot_err_deg"] = float(
            np.degrees(np.linalg.norm(_rotation_residual(pred[name][:3, :3], gt[name][:3, :3])))
        )
    return out


def articulated_add_mm(pred_pose, gt_pose, surfaces):
    pred = fk_matrices_np(
        pred_pose["rot"], pred_pose["trans"], pred_pose["alpha"], pred_pose["theta_l"], pred_pose["theta_r"]
    )
    gt = fk_matrices_np(gt_pose["rot"], gt_pose["trans"], gt_pose["alpha"], gt_pose["theta_l"], gt_pose["theta_r"])
    distances = []
    per_part = {}
    for name, surface in surfaces.items():
        points = np.asarray(surface.points_m, dtype=np.float64)
        pred_points = points @ pred[name][:3, :3].T + pred[name][:3, 3]
        gt_points = points @ gt[name][:3, :3].T + gt[name][:3, 3]
        values = np.linalg.norm(pred_points - gt_points, axis=1) * 1000.0
        per_part[f"{name}_add_mm"] = float(values.mean())
        distances.append(values)
    per_part["articulated_add_mm"] = float(np.concatenate(distances).mean())
    return per_part
