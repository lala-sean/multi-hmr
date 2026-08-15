import cv2
import numpy as np

from instrument_geometry import instrument_keypoints_camera_np


def matrix_to_quat_wxyz_np(R):
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    trace = np.trace(R)
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    quat = np.asarray([w, x, y, z], dtype=np.float64)
    quat /= np.linalg.norm(quat)
    if quat[0] < 0.0:
        quat = -quat
    return quat


def pose_from_keypoints_pnp(keypoints_uv, action, K, scores=None, min_score=0.0):
    image_points = np.asarray(keypoints_uv, dtype=np.float64).reshape(-1, 2)
    action = np.asarray(action, dtype=np.float64).reshape(3)
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    object_points = instrument_keypoints_camera_np(
        np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
        action,
    ).astype(np.float64)
    valid = np.isfinite(image_points).all(axis=1)
    if scores is not None:
        valid &= np.asarray(scores, dtype=np.float64).reshape(-1) >= float(min_score)
    if int(valid.sum()) < 4:
        raise RuntimeError(f"PnP needs at least 4 valid keypoints, got {int(valid.sum())}")
    obj = np.ascontiguousarray(object_points[valid])
    img = np.ascontiguousarray(image_points[valid])
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, None, flags=cv2.SOLVEPNP_EPNP)
    if not ok:
        raise RuntimeError("cv2.solvePnP(EPNP) failed")
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, None, rvec, tvec, True, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise RuntimeError("cv2.solvePnP(ITERATIVE) refinement failed")
    R, _ = cv2.Rodrigues(rvec)
    return {
        "rot": matrix_to_quat_wxyz_np(R),
        "trans": tvec.reshape(3).astype(np.float64),
        "alpha": float(action[0]),
        "theta_l": float(action[1]),
        "theta_r": float(action[2]),
    }
