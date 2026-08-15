import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


RARP_FC = 587.54401824
SHAFT_WRIST_OFFSET_M = 0.2159
GRIPPER_JOINT_OFFSET_M = 0.009
SURFEMB_GRIPPER_STATIC_THRESHOLD_M = 0.00213
SURFEMB_SHAFT_NORM_X_MIN = -0.5

KEYPOINT_NAMES = (
    "shaft_axis",
    "wrist_shaft_joint",
    "wrist_gripper_joint",
    "left_tip",
    "right_tip",
)


def surfemb_surface_sampling_mask(coords_norm, effective_part_ids, shaft_norm_x_min=SURFEMB_SHAFT_NORM_X_MIN):
    coords_norm = np.asarray(coords_norm)
    effective_part_ids = np.asarray(effective_part_ids, dtype=np.int64).reshape(-1)
    if coords_norm.shape != (len(effective_part_ids), 3):
        raise ValueError(
            f"Expected surface coordinates shaped ({len(effective_part_ids)}, 3), got {coords_norm.shape}."
        )
    rear_shaft = (effective_part_ids == 1) & (
        coords_norm[:, 0] < float(shaft_norm_x_min)
    )
    return ~rear_shaft


def rodrigues_np(axis, theta):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    a = math.cos(float(theta) / 2.0)
    b, c, d = -axis * math.sin(float(theta) / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array(
        [
            [aa + bb - cc - dd, 2.0 * (bc + ad), 2.0 * (bd - ac)],
            [2.0 * (bc - ad), aa + cc - bb - dd, 2.0 * (cd + ab)],
            [2.0 * (bd + ac), 2.0 * (cd - ab), aa + dd - bb - cc],
        ],
        dtype=np.float64,
    )


def rodrigues_torch(axis, theta):
    axis = torch.as_tensor(axis, dtype=theta.dtype, device=theta.device)
    axis = axis / torch.linalg.norm(axis)
    theta = theta.reshape(-1)
    a = torch.cos(theta / 2.0)
    b = -axis[0] * torch.sin(theta / 2.0)
    c = -axis[1] * torch.sin(theta / 2.0)
    d = -axis[2] * torch.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    row0 = torch.stack([aa + bb - cc - dd, 2.0 * (bc + ad), 2.0 * (bd - ac)], dim=-1)
    row1 = torch.stack([2.0 * (bc - ad), aa + cc - bb - dd, 2.0 * (cd + ab)], dim=-1)
    row2 = torch.stack([2.0 * (bd + ac), 2.0 * (cd - ab), aa + dd - bb - cc], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def make_transform_np(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def quat_wxyz_to_matrix_np(quat):
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def quat_wxyz_to_matrix_torch(quat):
    q = F.normalize(quat.float(), p=2, dim=-1)
    w, x, y, z = q.unbind(dim=-1)
    row0 = torch.stack(
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
        dim=-1,
    )
    row1 = torch.stack(
        [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
        dim=-1,
    )
    row2 = torch.stack(
        [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        dim=-1,
    )
    return torch.stack([row0, row1, row2], dim=-2)


def quat_wxyz_to_rvec_np(quat):
    R = quat_wxyz_to_matrix_np(quat)
    rvec, _ = cv2.Rodrigues(R)
    return rvec.reshape(3).astype(np.float64)


def project_points_np(points_cam, K):
    points = np.asarray(points_cam, dtype=np.float64).reshape(-1, 3)
    z = points[:, 2:3]
    uvw = points @ np.asarray(K, dtype=np.float64).reshape(3, 3).T
    return uvw[:, :2] / z


def project_points_torch(points_cam, K):
    uvw = points_cam @ K.transpose(-1, -2)
    return uvw[..., :2] / uvw[..., 2:3].clamp_min(1e-8)


def rarp_intrinsics(width, height):
    return np.array(
        [[RARP_FC, 0.0, float(width) / 2.0], [0.0, RARP_FC, float(height) / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def crop_resize_pad_intrinsics(K, bbox_min, scale_xy, pad_xy):
    sx, sy = float(scale_xy[0]), float(scale_xy[1])
    pad_x, pad_y = float(pad_xy[0]), float(pad_xy[1])
    x0, y0 = float(bbox_min[0]), float(bbox_min[1])
    A = np.array([[sx, 0.0, pad_x - sx * x0], [0.0, sy, pad_y - sy * y0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return A @ np.asarray(K, dtype=np.float32)


def _bias2part_matrices():
    bias2world = np.eye(4, dtype=np.float64)
    bias2world[:3, :3] = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64).T
    flip_wrist = np.eye(4, dtype=np.float64)
    flip_wrist[:3, :3] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64).T

    shaft2world = np.eye(4, dtype=np.float64)
    wrist2world = np.eye(4, dtype=np.float64)
    l_gripper2world = np.eye(4, dtype=np.float64)
    r_gripper2world = np.eye(4, dtype=np.float64)
    shaft2world[:3, 3] = np.array([-SHAFT_WRIST_OFFSET_M, 0.0, 0.0])
    l_gripper2world[:3, 3] = np.array([GRIPPER_JOINT_OFFSET_M, 0.0, 0.0])
    r_gripper2world[:3, 3] = np.array([GRIPPER_JOINT_OFFSET_M, 0.0, 0.0])

    return {
        "shaft": np.linalg.inv(shaft2world) @ bias2world,
        "wrist": np.linalg.inv(wrist2world) @ flip_wrist @ bias2world,
        "l_gripper": np.linalg.inv(l_gripper2world) @ bias2world,
        "r_gripper": np.linalg.inv(r_gripper2world) @ bias2world,
    }


def _bias2world_matrix():
    bias2world = np.eye(4, dtype=np.float64)
    bias2world[:3, :3] = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64).T
    return bias2world


def _joint_keypoint_part_matrices():
    # GMS Part applies mesh vertices with bias2part, but its annotated joint
    # keypoints are stored in the post-bias world frame and use
    # bias2part @ inv(bias2world).  Keep this path identical to Instrument
    # Splatting so gripper tip keypoints and mesh FK share one convention.
    world2bias = np.linalg.inv(_bias2world_matrix())
    return {part_name: mat @ world2bias for part_name, mat in _bias2part_matrices().items()}


def instrument_keypoints_part_np():
    joint2part = _joint_keypoint_part_matrices()
    raw = {
        "shaft": np.array([[-0.007026, 0.0, 0.0]], dtype=np.float64),
        "wrist": np.array(
            [[0.0, 0.0, 0.0], [GRIPPER_JOINT_OFFSET_M, 0.0, 0.0]],
            dtype=np.float64,
        ),
        "l_gripper": np.array([[0.017935, 0.000635, 0.000548]], dtype=np.float64),
        "r_gripper": np.array([[0.017935, -0.000635, 0.000548]], dtype=np.float64),
    }

    def tx(part_name):
        pts = raw[part_name]
        T = joint2part[part_name]
        return pts @ T[:3, :3].T + T[:3, 3]

    shaft_axis = tx("shaft")[0]
    wrist_pts = tx("wrist")
    left_tip = tx("l_gripper")[0]
    right_tip = tx("r_gripper")[0]
    return {
        "shaft_axis": ("shaft", shaft_axis),
        "wrist_shaft_joint": ("wrist", wrist_pts[0]),
        "wrist_gripper_joint": ("wrist", wrist_pts[1]),
        "left_tip": ("l_gripper", left_tip),
        "right_tip": ("r_gripper", right_tip),
    }


def fk_matrices_np(quat_wxyz, trans, alpha, theta_l, theta_r):
    R = quat_wxyz_to_matrix_np(quat_wxyz)
    wrist2camera = make_transform_np(R, trans)
    wrist2shaft = make_transform_np(
        rodrigues_np([0.0, 1.0, 0.0], alpha),
        [SHAFT_WRIST_OFFSET_M, 0.0, 0.0],
    )
    shaft = wrist2camera @ np.linalg.inv(wrist2shaft)
    wrist = shaft @ wrist2shaft
    l_gripper = make_transform_np(
        rodrigues_np([0.0, 0.0, 1.0], theta_l),
        [GRIPPER_JOINT_OFFSET_M, 0.0, 0.0],
    )
    r_gripper = make_transform_np(
        rodrigues_np([0.0, 0.0, 1.0], -float(theta_r)),
        [GRIPPER_JOINT_OFFSET_M, 0.0, 0.0],
    )
    return {
        "shaft": shaft,
        "wrist": wrist,
        "l_gripper": wrist @ l_gripper,
        "r_gripper": wrist @ r_gripper,
    }


def instrument_keypoints_camera_np(quat_wxyz, trans, action):
    alpha, theta_l, theta_r = [float(v) for v in np.asarray(action).reshape(3)]
    transforms = fk_matrices_np(quat_wxyz, trans, alpha, theta_l, theta_r)
    points = instrument_keypoints_part_np()
    out = []
    for name in KEYPOINT_NAMES:
        part_name, point = points[name]
        T = transforms[part_name]
        out.append(point @ T[:3, :3].T + T[:3, 3])
    return np.asarray(out, dtype=np.float32)


def instrument_keypoints_camera_torch(quat_wxyz, trans, action):
    B = quat_wxyz.shape[0]
    dtype = quat_wxyz.dtype
    device = quat_wxyz.device
    R = quat_wxyz_to_matrix_torch(quat_wxyz).to(dtype)
    alpha, theta_l, theta_r = action[:, 0], action[:, 1], action[:, 2]

    wrist2shaft_R = rodrigues_torch([0.0, 1.0, 0.0], alpha).to(dtype)
    shaft_R = R @ wrist2shaft_R.transpose(1, 2)
    shaft_offset = (
        torch.tensor([SHAFT_WRIST_OFFSET_M, 0.0, 0.0], dtype=dtype, device=device)
        .view(1, 3, 1)
        .expand(B, -1, -1)
    )
    shaft_t = trans - torch.bmm(shaft_R, shaft_offset).squeeze(-1)
    wrist_R = R
    wrist_t = trans

    l_R = rodrigues_torch([0.0, 0.0, 1.0], theta_l).to(dtype)
    r_R = rodrigues_torch([0.0, 0.0, 1.0], -theta_r).to(dtype)
    gripper_offset = (
        torch.tensor([GRIPPER_JOINT_OFFSET_M, 0.0, 0.0], dtype=dtype, device=device)
        .view(1, 3, 1)
        .expand(B, -1, -1)
    )
    l_world_R = wrist_R @ l_R
    r_world_R = wrist_R @ r_R
    l_world_t = wrist_t + torch.bmm(wrist_R, gripper_offset).squeeze(-1)
    r_world_t = wrist_t + torch.bmm(wrist_R, gripper_offset).squeeze(-1)

    kp_part = instrument_keypoints_part_np()
    specs = {
        "shaft": (shaft_R, shaft_t),
        "wrist": (wrist_R, wrist_t),
        "l_gripper": (l_world_R, l_world_t),
        "r_gripper": (r_world_R, r_world_t),
    }
    out = []
    for name in KEYPOINT_NAMES:
        part, point_np = kp_part[name]
        point = torch.tensor(point_np, dtype=dtype, device=device).view(1, 3, 1).expand(B, -1, -1)
        part_R, part_t = specs[part]
        out.append(torch.bmm(part_R, point).squeeze(-1) + part_t)
    return torch.stack(out, dim=1)


def pose_dict_to_params(pose):
    quat = np.asarray(pose["rot"], dtype=np.float64).reshape(4)
    rvec = quat_wxyz_to_rvec_np(quat)
    trans = np.asarray(pose["trans"], dtype=np.float64).reshape(3)
    return np.concatenate(
        [
            rvec,
            trans,
            np.asarray([pose["alpha"], pose["theta_l"], pose["theta_r"]], dtype=np.float64),
        ],
        axis=0,
    )


def multihmr_root():
    return Path(__file__).resolve().parents[2]
