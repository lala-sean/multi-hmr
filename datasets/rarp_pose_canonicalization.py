import math

import numpy as np
import torch
import torch.nn.functional as F


SX_180 = torch.diag(torch.tensor([1.0, -1.0, -1.0], dtype=torch.float32))


def quat_to_matrix(quat: torch.Tensor) -> torch.Tensor:
    """Convert quaternion [w, x, y, z] to a 3x3 rotation matrix."""
    q = quat.detach().float().reshape(1, 4)
    q = F.normalize(q, p=2, dim=1)[0]
    w, x, y, z = q[0], q[1], q[2], q[3]
    rows = [
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)]),
        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)]),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]),
    ]
    return torch.stack(rows, dim=0)


def matrix_to_quat(matrix: torch.Tensor) -> torch.Tensor:
    """Convert a 3x3 rotation matrix to quaternion [w, x, y, z] with w >= 0."""
    m = matrix.detach().float()
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = torch.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = torch.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = torch.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = torch.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    quat = F.normalize(torch.stack([w, x, y, z], dim=0), p=2, dim=0)
    if quat[0] < 0:
        quat = -quat
    return quat


def canonicalize_pose_symmetry(pose: dict, eps: float = 0.08, enabled: bool = True) -> dict:
    """
    Canonicalize wrist-to-camera pose under the instrument RotX(pi) symmetry.

    The positive representative is the candidate with smaller R[2,2]; if that
    value is within eps, smaller R[1,2] breaks the tie.
    """
    rot = torch.as_tensor(pose["rot"]).detach().float().reshape(4)
    trans = torch.as_tensor(pose["trans"]).detach().float().reshape(3)
    alpha = torch.as_tensor(pose["alpha"]).detach().float().reshape(-1)
    theta_l = torch.as_tensor(pose["theta_l"]).detach().float().reshape(-1)
    theta_r = torch.as_tensor(pose["theta_r"]).detach().float().reshape(-1)

    R = quat_to_matrix(rot)
    out = {
        "rot": matrix_to_quat(R),
        "trans": trans.clone(),
        "alpha": alpha.clone(),
        "theta_l": theta_l.clone(),
        "theta_r": theta_r.clone(),
        "pose_sym_flipped": torch.tensor(False),
        "pose_sym_score": torch.tensor([R[2, 2], R[1, 2]], dtype=torch.float32),
    }
    if not enabled:
        return out

    sx = SX_180.to(R.device)
    R_flip = R @ sx
    choose_flip = bool(R_flip[2, 2] < R[2, 2])
    if abs(float(R_flip[2, 2] - R[2, 2])) <= float(eps):
        choose_flip = bool(R_flip[1, 2] < R[1, 2])

    if choose_flip:
        out["rot"] = matrix_to_quat(R_flip)
        out["alpha"] = -alpha.clone()
        out["theta_l"] = theta_r.clone()
        out["theta_r"] = theta_l.clone()
        out["pose_sym_flipped"] = torch.tensor(True)
        out["pose_sym_score"] = torch.tensor([R_flip[2, 2], R_flip[1, 2]], dtype=torch.float32)
    return out


def pose_to_numpy(pose: dict) -> dict:
    out = {}
    for key, value in pose.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.detach().cpu().numpy()
        else:
            out[key] = np.asarray(value)
    return out


def rot_y(angle: float) -> np.ndarray:
    c, s = math.cos(float(angle)), math.sin(float(angle))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def rot_z(angle: float) -> np.ndarray:
    c, s = math.cos(float(angle)), math.sin(float(angle))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def make_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def fk_matrices(R: np.ndarray, t: np.ndarray, alpha: float, theta_l: float, theta_r: float) -> dict:
    """Small numpy FK mirror of GMS Instrument.forward_kinematics for tests."""
    wrist2camera = make_transform(R, t)
    wrist2shaft = make_transform(rot_y(alpha), np.array([0.2159, 0.0, 0.0]))
    shaft = wrist2camera @ np.linalg.inv(wrist2shaft)

    l_gripper = make_transform(rot_z(theta_l), np.array([0.009, 0.0, 0.0]))
    r_gripper = make_transform(rot_z(-theta_r), np.array([0.009, 0.0, 0.0]))
    return {
        "shaft": shaft,
        "wrist": wrist2camera,
        "l_gripper": wrist2camera @ l_gripper,
        "r_gripper": wrist2camera @ r_gripper,
    }
