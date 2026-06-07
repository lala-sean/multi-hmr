import math

import cv2
import numpy as np


RGB_INTERPOLATIONS = (
    cv2.INTER_NEAREST,
    cv2.INTER_LINEAR,
    cv2.INTER_AREA,
    cv2.INTER_CUBIC,
)


def _resize_nn(arr, size):
    return cv2.resize(arr, (size, size), interpolation=cv2.INTER_NEAREST)


def _union_mask(instruments):
    union = None
    for inst in instruments:
        mask = np.asarray(inst["inst_mask"]) > 0
        union = mask if union is None else (union | mask)
    return union


def _center_pad_affine(width, height, img_size):
    scale = img_size / max(width, height)
    new_w = int(width * scale)
    new_h = int(height * scale)
    pad_x = (img_size - new_w) // 2
    pad_y = (img_size - new_h) // 2
    return np.array(
        [[scale, 0.0, float(pad_x)], [0.0, scale, float(pad_y)]],
        dtype=np.float32,
    )


def _random_rotated_mask_crop_affine(
    union_mask,
    img_size,
    crop_scale=1.2,
    max_angle=math.pi,
    offset_scale=1.0,
    rng=np.random,
):
    theta = float(rng.uniform(-max_angle, max_angle))
    s, c = math.sin(theta), math.cos(theta)
    rot = np.array(((c, -s), (s, c)), dtype=np.float32)

    if union_mask is None or not np.any(union_mask):
        h, w = union_mask.shape if union_mask is not None else (img_size, img_size)
        mask_xy = np.array(
            [[0.0, 0.0], [float(w - 1), 0.0], [0.0, float(h - 1)], [float(w - 1), float(h - 1)]],
            dtype=np.float32,
        )
    else:
        mask_xy = np.argwhere(union_mask)[:, ::-1].astype(np.float32)

    mask_xy_rot = mask_xy @ rot.T
    left, top = mask_xy_rot.min(axis=0)
    right, bottom = mask_xy_rot.max(axis=0)
    cx, cy = (left + right) * 0.5, (top + bottom) * 0.5

    extent = max(float(bottom - top), float(right - left), 1.0)
    scale = img_size / extent / float(crop_scale)
    scale *= float(rng.uniform(1.0 - 0.05 * offset_scale, 1.0 + 0.05 * offset_scale))

    affine = np.concatenate((rot, np.array([[-cx], [-cy]], dtype=np.float32)), axis=1)
    affine *= scale
    affine[:, 2] += img_size / 2.0

    offset = (img_size - img_size / float(crop_scale)) * 0.5 * float(offset_scale)
    affine[:, 2] += rng.uniform(-offset, offset, 2).astype(np.float32)
    return affine.astype(np.float32)


def _warp_rgb(rgb, affine, img_size, random_interpolation, rng=np.random):
    interp = int(rng.choice(RGB_INTERPOLATIONS)) if random_interpolation else cv2.INTER_LINEAR
    return cv2.warpAffine(
        rgb,
        affine,
        (img_size, img_size),
        flags=interp,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def _warp_mask(mask, affine, img_size):
    return cv2.warpAffine(
        mask,
        affine,
        (img_size, img_size),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _transform_point(xy, affine):
    x, y = float(xy[0]), float(xy[1])
    out = affine @ np.array([x, y, 1.0], dtype=np.float32)
    return out.astype(np.float32)


def apply_photometric_jitter(rgb, rng=np.random):
    out = rgb.astype(np.uint8, copy=True)

    if float(rng.rand()) < 0.5:
        out = cv2.GaussianBlur(out, (3, 3), 0)

    if float(rng.rand()) < 0.5:
        sigma = float(rng.uniform(3.0, 12.0))
        noise = rng.normal(0.0, sigma, out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if float(rng.rand()) < 0.5:
        lab = cv2.cvtColor(out, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    if float(rng.rand()) < 0.5:
        hsv = cv2.cvtColor(out, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + float(rng.uniform(-18.0, 18.0))) % 180.0
        hsv[:, :, 1] *= float(rng.uniform(0.85, 1.15))
        hsv[:, :, 2] *= float(rng.uniform(0.85, 1.15))
        out = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)

    if float(rng.rand()) < 0.5:
        h, w = out.shape[:2]
        for _ in range(int(rng.randint(1, 5))):
            drop_h = int(rng.randint(8, 17))
            drop_w = int(rng.randint(8, 17))
            y0 = int(rng.randint(0, max(1, h - drop_h + 1)))
            x0 = int(rng.randint(0, max(1, w - drop_w + 1)))
            out[y0:y0 + drop_h, x0:x0 + drop_w] = 0

    return out


def transform_rarp_sample(
    rgb,
    instruments,
    img_size,
    output_size,
    dense_output_size=None,
    training=False,
    aug_random_crop_rotate=False,
    aug_geom_prob=1.0,
    aug_crop_scale=1.2,
    aug_max_angle=math.pi,
    aug_offset_scale=1.0,
    aug_color_jitter=False,
    rng=np.random,
):
    """
    Transform RGB plus per-instance masks/coord maps together.

    HCCE coord maps are expected in original image coordinates and are warped
    before target downsampling, matching the old render-then-transform order.
    If dense_output_size is provided, extra H/2-style targets are stored as
    inst_mask_dense / part_mask_dense / coord_img_dense without changing the
    legacy inst_mask / part_mask / coord_img fields.
    """
    h, w = rgb.shape[:2]
    use_geom_aug = bool(
        training
        and aug_random_crop_rotate
        and float(rng.rand()) < float(aug_geom_prob)
    )
    if use_geom_aug:
        union = _union_mask(instruments)
        if union is None:
            union = np.ones((h, w), dtype=bool)
        affine = _random_rotated_mask_crop_affine(
            union,
            img_size=img_size,
            crop_scale=aug_crop_scale,
            max_angle=aug_max_angle,
            offset_scale=aug_offset_scale,
            rng=rng,
        )
        random_rgb_interp = True
    else:
        affine = _center_pad_affine(w, h, img_size)
        random_rgb_interp = False

    rgb_sq = _warp_rgb(rgb, affine, img_size, random_rgb_interp, rng=rng)
    if bool(training and aug_color_jitter):
        rgb_sq = apply_photometric_jitter(rgb_sq, rng=rng)

    out_instruments = []
    for inst in instruments:
        inst_full = _warp_mask(
            np.asarray(inst["inst_mask"], dtype=np.float32),
            affine,
            img_size,
        ).astype(np.float32)
        part_full = _warp_mask(
            np.asarray(inst["part_mask"], dtype=np.uint8),
            affine,
            img_size,
        ).astype(np.uint8)
        inst_target = _resize_nn(inst_full, output_size).astype(np.float32)
        part_target = _resize_nn(part_full, output_size).astype(np.int64)
        wrist_center = _transform_point(inst["wrist_center"], affine)

        if inst_target.sum() <= 0:
            continue
        if not (0.0 <= wrist_center[0] < img_size and 0.0 <= wrist_center[1] < img_size):
            continue

        out = dict(inst)
        out["inst_mask"] = inst_target
        out["part_mask"] = part_target
        out["wrist_center"] = wrist_center
        if dense_output_size is not None:
            out["inst_mask_dense"] = _resize_nn(inst_full, dense_output_size).astype(np.float32)
            out["part_mask_dense"] = _resize_nn(part_full, dense_output_size).astype(np.int64)

        coord_img = inst.get("coord_img")
        if coord_img is not None:
            coord_full = _warp_mask(
                np.asarray(coord_img, dtype=np.float32),
                affine,
                img_size,
            ).astype(np.float32)
            out["coord_img"] = _resize_nn(coord_full, output_size).astype(np.float32)
            if dense_output_size is not None:
                out["coord_img_dense"] = _resize_nn(coord_full, dense_output_size).astype(np.float32)
        else:
            out["coord_img"] = None
            if dense_output_size is not None:
                out["coord_img_dense"] = None
        out_instruments.append(out)

    return rgb_sq, out_instruments, affine
