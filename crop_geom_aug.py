import cv2
import numpy as np


RGB_INTERPOLATIONS = (
    cv2.INTER_NEAREST,
    cv2.INTER_LINEAR,
    cv2.INTER_AREA,
    cv2.INTER_CUBIC,
)


def transform_points_2d(points, affine):
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] == 0:
        return pts
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    return (np.concatenate([pts, ones], axis=1) @ np.asarray(affine, dtype=np.float32).T).astype(np.float32)


def maybe_random_rotate_crop(
    rgb,
    K,
    keypoints_crop,
    keypoints_valid,
    training,
    enabled,
    prob,
    max_angle,
    rng=np.random,
):
    """Rotate the crop in image space and keep intrinsics/keypoints aligned."""
    if not (bool(training) and bool(enabled) and float(rng.rand()) < float(prob)):
        affine = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        return rgb, np.asarray(K, dtype=np.float32), np.asarray(keypoints_crop, dtype=np.float32), keypoints_valid, affine

    h, w = rgb.shape[:2]
    angle_deg = float(rng.uniform(-float(max_angle), float(max_angle)) * 180.0 / np.pi)
    center = ((float(w) - 1.0) * 0.5, (float(h) - 1.0) * 0.5)
    affine = cv2.getRotationMatrix2D(center, angle_deg, 1.0).astype(np.float32)
    interp = int(rng.choice(RGB_INTERPOLATIONS))
    rgb_rot = cv2.warpAffine(
        rgb,
        affine,
        (w, h),
        flags=interp,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    H = np.eye(3, dtype=np.float32)
    H[:2] = affine
    K_rot = H @ np.asarray(K, dtype=np.float32)
    kp_rot = transform_points_2d(keypoints_crop, affine)
    valid = np.asarray(keypoints_valid, dtype=bool).reshape(-1)
    inside = (
        (kp_rot[:, 0] >= 0.0)
        & (kp_rot[:, 0] < float(w))
        & (kp_rot[:, 1] >= 0.0)
        & (kp_rot[:, 1] < float(h))
    )
    return rgb_rot, K_rot.astype(np.float32), kp_rot.astype(np.float32), valid & inside, affine
