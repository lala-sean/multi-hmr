import math
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as tv_transforms
from PIL import Image

ROBOPEPP_ROOT = Path(__file__).resolve().parents[1]
MULTIHMR_ROOT = ROBOPEPP_ROOT.parents[1]
SURFEMB_ROOT = MULTIHMR_ROOT / "submodules" / "surfemb"
if SURFEMB_ROOT.is_dir() and str(SURFEMB_ROOT) not in sys.path:
    sys.path.insert(0, str(SURFEMB_ROOT))

RGB_INTERPOLATIONS = (
    cv2.INTER_NEAREST,
    cv2.INTER_LINEAR,
    cv2.INTER_AREA,
    cv2.INTER_CUBIC,
)

IMAGENET_TENSOR = tv_transforms.Compose(
    [
        tv_transforms.ToTensor(),
        tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def _maybe(p=0.5):
    return random.random() < float(p)


def gaussian_blur(rgb, p=0.5):
    if not _maybe(p):
        return rgb
    # SurfEmb uses Albumentations GaussianBlur(blur_limit=(1, 3)).
    k = random.choice([1, 3])
    if k <= 1:
        return rgb
    return cv2.GaussianBlur(rgb, (k, k), 0)


def iso_noise(rgb, p=0.5):
    if not _maybe(p):
        return rgb
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    intensity = random.uniform(0.1, 0.5)
    color_shift = random.uniform(0.01, 0.05) * 180.0
    hsv[..., 0] = (hsv[..., 0] + np.random.normal(0.0, color_shift, hsv.shape[:2])) % 180.0
    hsv[..., 2] += np.random.poisson(np.maximum(hsv[..., 2], 0.0) * intensity) - hsv[..., 2] * intensity
    return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)


def gauss_noise(rgb, p=0.5):
    if not _maybe(p):
        return rgb
    sigma = random.uniform(10.0, 50.0)
    out = rgb.astype(np.float32) + np.random.normal(0.0, sigma, rgb.shape).astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def debayer_artefacts(rgb, p=0.5):
    if not _maybe(p):
        return rgb
    channel_idxs = np.random.permutation(3)
    channel_idxs_inv = np.empty(3, dtype=int)
    channel_idxs_inv[channel_idxs] = 0, 1, 2
    bayer = np.zeros(rgb.shape[:2], dtype=rgb.dtype)
    bayer[::2, ::2] = rgb[::2, ::2, channel_idxs[2]]
    bayer[1::2, ::2] = rgb[1::2, ::2, channel_idxs[1]]
    bayer[::2, 1::2] = rgb[::2, 1::2, channel_idxs[1]]
    bayer[1::2, 1::2] = rgb[1::2, 1::2, channel_idxs[0]]
    method = random.choice([cv2.COLOR_BAYER_BG2BGR, cv2.COLOR_BAYER_BG2BGR_EA])
    return cv2.cvtColor(bayer, method)[..., channel_idxs_inv]


def unsharpen(rgb, p=0.5):
    if not _maybe(p):
        return rgb
    k = random.randrange(3, 8, 2)
    sigma = k / 3.0
    strength = random.uniform(0.0, 2.0)
    blur = cv2.GaussianBlur(rgb, (k, k), sigma)
    return cv2.addWeighted(rgb, 1.0 + strength, blur, -strength, 0)


def clahe(rgb, p=0.5):
    if not _maybe(p):
        return rgb
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    aug = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    lab[..., 0] = aug.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def _sample_mask_center(mask):
    ys, xs = np.nonzero(np.asarray(mask).astype(bool))
    if len(xs) == 0:
        return None
    idx = random.randrange(len(xs))
    return int(xs[idx]), int(ys[idx])


def coarse_dropout(rgb, inst_mask=None, part_mask=None, focus_part_ids=(2, 3, 4), p=0.5):
    if not _maybe(p):
        return rgb
    out = rgb.copy()
    h, w = out.shape[:2]
    inst = np.asarray(inst_mask).astype(bool) if inst_mask is not None else None
    part = np.asarray(part_mask) if part_mask is not None else None
    focus = None
    if part is not None:
        focus = np.isin(part, np.asarray(focus_part_ids, dtype=part.dtype))
        if inst is not None:
            focus &= inst
    holes = random.randint(1, 8)
    for _ in range(holes):
        dh = random.randint(8, 18)
        dw = random.randint(8, 18)
        center = None
        r = random.random()
        if focus is not None and focus.any() and r < 0.85:
            center = _sample_mask_center(focus)
        elif inst is not None and inst.any() and r < 0.97:
            center = _sample_mask_center(inst)
        if center is None:
            cx = random.randint(0, max(0, w - 1))
            cy = random.randint(0, max(0, h - 1))
        else:
            cx, cy = center
            cx += random.randint(-max(1, dw // 3), max(1, dw // 3))
            cy += random.randint(-max(1, dh // 3), max(1, dh // 3))
        x0 = int(np.clip(cx - dw // 2, 0, max(0, w - dw)))
        y0 = int(np.clip(cy - dh // 2, 0, max(0, h - dh)))
        out[y0 : y0 + dh, x0 : x0 + dw] = 0
    return out


def color_jitter_hue(rgb, hue=0.1):
    if not _maybe():
        return rgb
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + random.uniform(-float(hue), float(hue)) * 180.0) % 180.0
    return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)


def iso_noise_surfemb(rgb, p=0.3):
    """Albumentations 1.4.1 ISONoise with an explicit application probability."""
    if not _maybe(p):
        return rgb
    image = rgb.astype(np.float32) / 255.0
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    _, stddev = cv2.meanStdDev(hls)
    intensity = random.uniform(0.1, 0.5)
    color_shift = random.uniform(0.01, 0.05)
    luminance_noise = np.random.poisson(float(stddev[1, 0]) * intensity * 255.0, size=hls.shape[:2])
    color_noise = np.random.normal(0.0, color_shift * 360.0 * intensity, size=hls.shape[:2])
    hls[..., 0] = np.mod(hls[..., 0] + color_noise, 360.0)
    hls[..., 1] += (luminance_noise / 255.0) * (1.0 - hls[..., 1])
    return np.clip(cv2.cvtColor(hls, cv2.COLOR_HLS2RGB) * 255.0, 0, 255).astype(np.uint8)


def gauss_noise_surfemb(rgb, p=0.5, variance_limits=(10.0, 50.0)):
    """Match SurfEmb's GaussNoise variance range instead of treating it as sigma."""
    if not _maybe(p):
        return rgb
    variance = random.uniform(float(variance_limits[0]), float(variance_limits[1]))
    noise = np.random.normal(0.0, math.sqrt(variance), rgb.shape).astype(np.float32)
    return np.clip(rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def clahe_surfemb(rgb, p=0.3):
    if not _maybe(p):
        return rgb
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    aug = cv2.createCLAHE(clipLimit=random.uniform(1.0, 4.0), tileGridSize=(8, 8))
    lab[..., 0] = aug.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def _adjust_brightness(rgb, factor):
    lut = np.clip(np.arange(256, dtype=np.float32) * float(factor), 0, 255).astype(np.uint8)
    return cv2.LUT(rgb, lut)


def _adjust_contrast(rgb, factor):
    mean = float(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).mean())
    lut = np.clip(np.arange(256, dtype=np.float32) * float(factor) + mean * (1.0 - float(factor)), 0, 255)
    return cv2.LUT(rgb, lut.astype(np.uint8))


def _adjust_saturation(rgb, factor):
    gray = cv2.cvtColor(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
    return cv2.addWeighted(rgb, float(factor), gray, 1.0 - float(factor), 0.0)


def _adjust_hue(rgb, factor):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lut = np.mod(np.arange(256, dtype=np.int16) + 180.0 * float(factor), 180).astype(np.uint8)
    hsv[..., 0] = cv2.LUT(hsv[..., 0], lut)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def color_jitter_surfemb(rgb, p=0.5, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
    """Match A.ColorJitter(hue=0.1), including its nonzero default B/C/S jitter."""
    if not _maybe(p):
        return rgb
    transforms = [
        (_adjust_brightness, random.uniform(1.0 - brightness, 1.0 + brightness)),
        (_adjust_contrast, random.uniform(1.0 - contrast, 1.0 + contrast)),
        (_adjust_saturation, random.uniform(1.0 - saturation, 1.0 + saturation)),
        (_adjust_hue, random.uniform(-hue, hue)),
    ]
    random.shuffle(transforms)
    out = rgb
    for transform, factor in transforms:
        out = transform(out, factor)
    return out


def surfemb_precrop_photometric(rgb, profile="legacy"):
    if profile == "original_p30":
        out = gaussian_blur(rgb, p=0.5)
        out = iso_noise_surfemb(out, p=0.3)
        out = gauss_noise_surfemb(out, p=0.5)
        out = debayer_artefacts(out, p=0.3)
        out = unsharpen(out, p=0.3)
        out = clahe_surfemb(out, p=0.3)
        return gaussian_blur(out, p=0.5)
    if profile != "legacy":
        raise ValueError(f"Unknown SurfEmb augmentation profile: {profile!r}")
    out = rgb.copy()
    for fn in (gaussian_blur, iso_noise, gauss_noise, debayer_artefacts, unsharpen, clahe, gaussian_blur):
        out = fn(out)
    return out


def surfemb_postcrop_photometric(rgb, inst_mask=None, part_mask=None, profile="legacy"):
    if profile == "original_p30":
        dropped = coarse_dropout(rgb, inst_mask=inst_mask, part_mask=part_mask, p=0.5)
        return color_jitter_surfemb(dropped, p=0.5, hue=0.1)
    if profile != "legacy":
        raise ValueError(f"Unknown SurfEmb augmentation profile: {profile!r}")
    return color_jitter_hue(coarse_dropout(rgb, inst_mask=inst_mask, part_mask=part_mask), hue=0.1)


def random_rotated_mask_crop_matrix(
    mask,
    crop_res,
    crop_scale=1.2,
    max_angle=math.pi,
    offset_scale=1.0,
    ensure_full_mask=True,
    max_attempts=12,
    min_mask_retention=0.985,
    source_visibility_fraction=1.0,
    offset_multiplier=1.0,
):
    mask = np.asarray(mask).astype(bool)
    if mask.any():
        mask_xy = np.argwhere(mask)[:, ::-1].astype(np.float32)
    else:
        h, w = mask.shape[:2]
        mask_xy = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]], dtype=np.float32)
    crop_res = int(crop_res)

    min_mask_retention = float(min_mask_retention)
    if not 0.0 < min_mask_retention <= 1.0:
        raise ValueError(f"min_mask_retention must be in (0, 1], got {min_mask_retention}.")
    source_visibility_fraction = float(source_visibility_fraction)
    offset_multiplier = float(offset_multiplier)
    if offset_multiplier < 0.0:
        raise ValueError(f"offset_multiplier must be non-negative, got {offset_multiplier}.")
    required_retention = 1.0 if source_visibility_fraction < min_mask_retention else min_mask_retention

    best = None
    best_count = -1
    attempts = int(max_attempts) if ensure_full_mask else 1
    for attempt in range(max(1, attempts)):
        theta = random.uniform(-float(max_angle), float(max_angle))
        s, c = math.sin(theta), math.cos(theta)
        R = np.array(((c, -s), (s, c)), dtype=np.float32)
        rotated = mask_xy @ R.T
        left, top = rotated.min(axis=0)
        right, bottom = rotated.max(axis=0)
        cx, cy = (left + right) * 0.5, (top + bottom) * 0.5
        extent = max(float(right - left), float(bottom - top), 1.0)
        scale = float(crop_res) / extent / float(crop_scale)
        scale *= random.uniform(1.0 - 0.05 * float(offset_scale), 1.0 + 0.05 * float(offset_scale))
        M = np.concatenate((R, np.array([[-cx], [-cy]], dtype=np.float32)), axis=1) * scale
        M[:, 2] += float(crop_res) / 2.0
        offset = (
            (float(crop_res) - float(crop_res) / float(crop_scale))
            * 0.5
            * float(offset_scale)
            * offset_multiplier
        )
        if offset > 0:
            M[:, 2] += np.random.uniform(-offset, offset, 2).astype(np.float32)

        if not ensure_full_mask:
            return M.astype(np.float32)
        xy1 = np.concatenate([mask_xy, np.ones((mask_xy.shape[0], 1), dtype=np.float32)], axis=1)
        dst = xy1 @ M.T
        inside = (
            (dst[:, 0] >= 0.0)
            & (dst[:, 0] < float(crop_res))
            & (dst[:, 1] >= 0.0)
            & (dst[:, 1] < float(crop_res))
        )
        count = int(inside.sum())
        if count > best_count:
            best = M.copy()
            best_count = count
        # Random rotation can slightly discretize mask area; this threshold keeps
        # the full instrument effectively inside the crop.
        required_count = int(math.ceil(mask_xy.shape[0] * required_retention - 1e-9))
        if count >= required_count:
            return M.astype(np.float32)
        if attempt == attempts - 2:
            # Last try: keep rotation but remove detector-like random offset.
            offset_scale = 0.0
    return best.astype(np.float32)


def warp_rgb(rgb, M, crop_res):
    interp = random.choice(RGB_INTERPOLATIONS)
    return cv2.warpAffine(
        rgb,
        M,
        (int(crop_res), int(crop_res)),
        flags=int(interp),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def warp_map(arr, M, crop_res, interpolation=cv2.INTER_NEAREST, value=0):
    return cv2.warpAffine(
        arr,
        M,
        (int(crop_res), int(crop_res)),
        flags=int(interpolation),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=value,
    )


def transform_points(points_xy, M):
    points = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    ones = np.ones((points.shape[0], 1), dtype=np.float32)
    return (np.concatenate([points, ones], axis=1) @ M.T).astype(np.float32)


def apply_surfemb_crop_augmentation(rgb, inst_mask, M, crop_res, part_mask=None):
    aug_rgb = surfemb_precrop_photometric(rgb)
    crop_rgb = warp_rgb(aug_rgb, M, crop_res)
    inst_crop = warp_map(inst_mask.astype(np.uint8), M, crop_res, interpolation=cv2.INTER_NEAREST, value=0)
    part_crop = None
    if part_mask is not None:
        part_crop = warp_map(np.asarray(part_mask).astype(np.uint8), M, crop_res, interpolation=cv2.INTER_NEAREST, value=0)
    crop_rgb = surfemb_postcrop_photometric(crop_rgb, inst_mask=inst_crop, part_mask=part_crop)
    if part_mask is None:
        return crop_rgb, inst_crop
    return crop_rgb, inst_crop, part_crop


def imagenet_tensor(rgb):
    return IMAGENET_TENSOR(Image.fromarray(rgb.astype(np.uint8)))


def matrix3_from_affine(M):
    M3 = np.eye(3, dtype=np.float32)
    M3[:2] = np.asarray(M, dtype=np.float32).reshape(2, 3)
    return M3


def sample_mask_yx(mask, n_samples):
    yx = np.argwhere(np.asarray(mask).astype(bool))
    if len(yx) == 0:
        yx = np.array([[0, 0]], dtype=np.int64)
    idx = np.random.choice(len(yx), int(n_samples), replace=int(n_samples) > len(yx))
    return yx[idx].astype(np.int64)


def load_surface_points(path):
    payload = np.load(path, allow_pickle=True).item()
    points = np.asarray(payload["points_norm"], dtype=np.float32)
    part_ids = np.asarray(
        payload.get("effective_part_ids", payload.get("part_ids", np.ones((len(points),), dtype=np.int64))),
        dtype=np.int64,
    )
    return points, part_ids


def sample_surface_points(points, n_samples):
    if len(points) == 0:
        raise RuntimeError("surface point pool is empty")
    idx = np.random.choice(len(points), int(n_samples), replace=int(n_samples) > len(points))
    return points[idx].astype(np.float32)


def projection_residual_from_target(target, project_points_np):
    valid = target["keypoints_valid"].detach().cpu().numpy().astype(bool)
    if not valid.any():
        return float("nan")
    kp3d = target["keypoints_3d_cam"].detach().cpu().numpy().astype(np.float32)
    K = target["K"].detach().cpu().numpy().astype(np.float32)
    proj = project_points_np(kp3d, K).astype(np.float32)
    kp = target["keypoints_crop"].detach().cpu().numpy().astype(np.float32)
    return float(np.max(np.abs(proj[valid] - kp[valid])))
