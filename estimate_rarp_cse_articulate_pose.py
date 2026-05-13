import argparse
import csv
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from PIL import Image
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from datasets.RarpInstanceDataset import RARPInstanceDataset
from datasets.surgical_instruments import collate_fn_instrument
from multi_instrument.multi_instrument_cse_dpt import MultiInstrumentCSEDPT


RARP_FC = 587.54401824
REPO_ROOT = Path(__file__).resolve().parent
GMS_ROOT = REPO_ROOT / "submodules" / "gaussian-mesh-splatting"
CAD_ROOT = GMS_ROOT / "instrument_mesh"

DATASETS = {
    "needlePuncture": {
        "dataset_root": "/mnt/nas/share/shuojue/data/needlePuncture_videos",
        "pose_root": "/mnt/nas/share/shuojue/data/needlePuncture_results",
    },
    "needleGrasping": {
        "dataset_root": "/mnt/nas/share/shuojue/data/needleGrasping_videos",
        "pose_root": "/mnt/nas/share/shuojue/data/needleGrasping_results",
    },
    "knotting": {
        "dataset_root": "/mnt/nas/share/shuojue/data/knotting_videos",
        "pose_root": "/mnt/nas/share/shuojue/data/knotting_results",
    },
}

PART_LABELS = {
    "shaft": 3,
    "wrist": 2,
    "gripper": 1,
}

DRAW_COLORS = {
    "shaft": (235, 70, 60),
    "wrist": (40, 210, 120),
    "l_gripper": (70, 150, 255),
    "r_gripper": (245, 170, 45),
}


class _PoseStub:
    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)


def _register_pose_stub():
    main_mod = sys.modules.get("__main__")
    if main_mod is not None and not hasattr(main_mod, "Pose"):
        setattr(main_mod, "Pose", _PoseStub)


def _load_mesh(path: Path) -> trimesh.Trimesh:
    if not path.is_file():
        raise FileNotFoundError(f"CAD mesh not found: {path}")
    mesh = trimesh.load_mesh(str(path), force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected Trimesh at {path}, got {type(mesh)}")
    return mesh


def _homogeneous_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def _rodrigues(axis: np.ndarray, theta: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array(
        [
            [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
        ],
        dtype=np.float64,
    )


def _make_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def _project(points_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    z = points_cam[:, 2:3]
    uvw = points_cam @ K.T
    return uvw[:, :2] / z


def _quaternion_to_matrix(quat: torch.Tensor) -> np.ndarray:
    q = F.normalize(quat.float().flatten().unsqueeze(0), p=2, dim=1)[0]
    r, x, y, z = [float(v) for v in q]
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - r * z), 2 * (x * z + r * y)],
            [2 * (x * y + r * z), 1 - 2 * (x * x + z * z), 2 * (y * z - r * x)],
            [2 * (x * z - r * y), 2 * (y * z + r * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _rotation_error_deg(R_pred: np.ndarray, R_gt: np.ndarray) -> float:
    R_delta = R_pred @ R_gt.T
    cos_angle = (np.trace(R_delta) - 1.0) * 0.5
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
    return math.degrees(math.acos(cos_angle))


@dataclass
class Correspondences:
    uv: np.ndarray
    points_part: np.ndarray
    part_names: np.ndarray

    def __len__(self):
        return int(self.uv.shape[0])


class InstrumentCAD:
    def __init__(self, cad_root: Path):
        self.cad_root = Path(cad_root)
        self.bias2world = np.eye(4, dtype=np.float64)
        self.bias2world[:3, :3] = np.array(
            [[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64
        ).T

        flip_wrist = np.eye(4, dtype=np.float64)
        flip_wrist[:3, :3] = np.array(
            [[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64
        ).T

        shaft2world = np.eye(4, dtype=np.float64)
        wrist2world = np.eye(4, dtype=np.float64)
        l_gripper2world = np.eye(4, dtype=np.float64)
        r_gripper2world = np.eye(4, dtype=np.float64)
        shaft2world[:3, 3] = np.array([-0.2159, 0.0, 0.0])
        l_gripper2world[:3, 3] = np.array([0.009, 0.0, 0.0])
        r_gripper2world[:3, 3] = np.array([0.009, 0.0, 0.0])

        self.bias2part = {
            "shaft": np.linalg.inv(shaft2world) @ self.bias2world,
            "wrist": np.linalg.inv(wrist2world) @ flip_wrist @ self.bias2world,
            "l_gripper": np.linalg.inv(l_gripper2world) @ self.bias2world,
            "r_gripper": np.linalg.inv(r_gripper2world) @ self.bias2world,
        }

        mesh_files = {
            "shaft": "transformed_shaft.obj",
            "wrist": "transformed_wrist.obj",
            "l_gripper": "transformed_gripper_left.obj",
            "r_gripper": "transformed_gripper_right.obj",
        }
        self.meshes = {}
        self.vertices_part = {}
        self.faces = {}
        self.canon_norm = {}
        self.kdtrees = {}

        for part_name, file_name in mesh_files.items():
            mesh = _load_mesh(self.cad_root / file_name).copy()
            mesh.apply_transform(self.bias2part[part_name])
            vertices = np.asarray(mesh.vertices, dtype=np.float64)
            faces = np.asarray(mesh.faces, dtype=np.int32)
            self.meshes[part_name] = mesh
            self.vertices_part[part_name] = vertices
            self.faces[part_name] = faces

        wrist_vertices = self.vertices_part["wrist"]
        self.canon_scale = float(2.0 * (wrist_vertices.max(0) - wrist_vertices.min(0)).max())
        if self.canon_scale <= 0:
            raise ValueError("Invalid CAD canonical scale")

        world2bias = np.linalg.inv(self.bias2world)
        self.canon_to_part = {}
        for part_name, vertices_part in self.vertices_part.items():
            part2world = self.bias2world @ np.linalg.inv(self.bias2part[part_name])
            canon_world = _homogeneous_transform(vertices_part, part2world)
            canon_norm = canon_world / self.canon_scale
            self.canon_norm[part_name] = canon_norm
            self.kdtrees[part_name] = cKDTree(canon_norm)
            self.canon_to_part[part_name] = self.bias2part[part_name] @ world2bias

    def nearest_part_points(self, part_name: str, xyz_norm: np.ndarray) -> np.ndarray:
        _, idx = self.kdtrees[part_name].query(xyz_norm, k=1)
        return self.vertices_part[part_name][idx]

    def nearest_gripper_points(self, xyz_norm: np.ndarray):
        dist_l, idx_l = self.kdtrees["l_gripper"].query(xyz_norm, k=1)
        dist_r, idx_r = self.kdtrees["r_gripper"].query(xyz_norm, k=1)
        use_l = dist_l <= dist_r
        points = np.empty((xyz_norm.shape[0], 3), dtype=np.float64)
        part_names = np.empty((xyz_norm.shape[0],), dtype=object)
        points[use_l] = self.vertices_part["l_gripper"][idx_l[use_l]]
        points[~use_l] = self.vertices_part["r_gripper"][idx_r[~use_l]]
        part_names[use_l] = "l_gripper"
        part_names[~use_l] = "r_gripper"
        return points, part_names

    def fk(self, rvec: np.ndarray, trans: np.ndarray, alpha: float,
           theta_l: float, theta_r: float):
        R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
        wrist2camera = _make_transform(R, trans)

        wrist2shaft = np.eye(4, dtype=np.float64)
        wrist2shaft[:3, :3] = _rodrigues(np.array([0.0, 1.0, 0.0]), float(alpha))
        wrist2shaft[:3, 3] = np.array([0.2159, 0.0, 0.0])
        shaft = wrist2camera @ np.linalg.inv(wrist2shaft)
        wrist = shaft @ wrist2shaft

        l_gripper_local = np.eye(4, dtype=np.float64)
        l_gripper_local[:3, :3] = _rodrigues(np.array([0.0, 0.0, 1.0]), float(theta_l))
        l_gripper_local[:3, 3] = np.array([0.009, 0.0, 0.0])

        r_gripper_local = np.eye(4, dtype=np.float64)
        r_gripper_local[:3, :3] = _rodrigues(np.array([0.0, 0.0, 1.0]), -float(theta_r))
        r_gripper_local[:3, 3] = np.array([0.009, 0.0, 0.0])

        return {
            "shaft": shaft,
            "wrist": wrist,
            "l_gripper": wrist @ l_gripper_local,
            "r_gripper": wrist @ r_gripper_local,
        }

    def transform_points(self, part_name: str, points_part: np.ndarray, transforms: dict):
        return _homogeneous_transform(points_part, transforms[part_name])

    def posed_mesh_vertices(self, rvec: np.ndarray, trans: np.ndarray,
                            alpha: float, theta_l: float, theta_r: float):
        transforms = self.fk(rvec, trans, alpha, theta_l, theta_r)
        return {
            part_name: self.transform_points(part_name, vertices, transforms)
            for part_name, vertices in self.vertices_part.items()
        }


def build_model_args(ckpt_args):
    defaults = dict(
        img_size=630,
        backbone="dinov2_vits14",
        xat_dim=512,
        xat_depth=4,
        xat_heads=16,
        xat_dim_head=32,
        xat_mlp_dim=2048,
        mask_dim=256,
        num_parts=4,
        nocs_feat_dim=256,
    )
    if ckpt_args is not None:
        if hasattr(ckpt_args, "__dict__"):
            src = vars(ckpt_args)
        elif isinstance(ckpt_args, dict):
            src = ckpt_args
        else:
            raise TypeError(f"Unsupported checkpoint args type: {type(ckpt_args)}")
        for key in defaults:
            if key in src:
                defaults[key] = src[key]
    defaults["pretrained_backbone"] = False
    return defaults


def load_model(checkpoint_path: Path, device: torch.device):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" not in ckpt:
        raise KeyError(f"Checkpoint missing model_state_dict: {checkpoint_path}")
    model_args = build_model_args(ckpt.get("args"))
    model = MultiInstrumentCSEDPT(**model_args).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model, model_args


def build_dataset(args):
    cfg = DATASETS[args.dataset_name]
    dataset = RARPInstanceDataset(
        split="test",
        training=False,
        img_size=args.img_size,
        dataset_root=cfg["dataset_root"],
        pose_root=cfg["pose_root"],
        min_dice=[args.min_dice_shaft, args.min_dice_wrist, args.min_dice_gripper],
        train_ratio=args.train_ratio,
        subsample=1,
        cse_coord_root=None,
        render_on_the_fly=False,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"Dataset is empty: {args.dataset_name} test split")
    return dataset


def select_videos(dataset, args):
    videos = sorted({video_name for video_name, _, _ in dataset.samples})
    if not videos:
        raise RuntimeError("No videos found in dataset")
    if args.video_name:
        if args.video_name not in videos:
            raise ValueError(
                f"Requested video_name={args.video_name!r} not in test split. "
                f"Available examples: {videos[:10]}"
            )
        return [args.video_name]
    if args.video_select != "random":
        raise ValueError(f"Unsupported video_select={args.video_select!r}")
    rng = random.Random(args.seed)
    n = min(int(args.num_videos), len(videos))
    if n <= 0:
        raise ValueError("--num_videos must be positive")
    return rng.sample(videos, n)


def indices_for_video(dataset, video_name: str, max_frames: int,
                      frame_select: str = "first", seed: int = 0):
    idxs = [
        idx for idx, sample in enumerate(dataset.samples)
        if sample[0] == video_name
    ]
    if not idxs:
        raise RuntimeError(f"No samples found for video {video_name}")
    if frame_select == "even":
        idxs = [
            idx for idx in idxs
            if int(dataset.samples[idx][1]) % 2 == 0
        ]
    elif frame_select == "random":
        rng = random.Random(f"{seed}:{video_name}")
        rng.shuffle(idxs)
    elif frame_select == "first":
        pass
    else:
        raise ValueError(f"Unsupported frame_select={frame_select!r}")
    if max_frames > 0:
        idxs = idxs[:max_frames]
    return idxs


def original_image_path(dataset, video_name: str, frame_id: str):
    video_folder = Path(dataset.dataset_root) / f"SARRARP502022_{video_name}"
    frames_folder = video_folder / "frames_v2" if (
        getattr(dataset, "v2_force", False) and (video_folder / "frames_v2").is_dir()
    ) else video_folder / "frames"
    png_path = frames_folder / f"{frame_id}.png"
    jpg_path = frames_folder / f"{frame_id}.jpg"
    if png_path.is_file():
        return png_path
    if jpg_path.is_file():
        return jpg_path
    raise FileNotFoundError(f"Frame image not found: {png_path} or {jpg_path}")


def image_geometry(dataset, video_name: str, frame_id: str):
    img_path = original_image_path(dataset, video_name, frame_id)
    rgb = np.asarray(Image.open(img_path).convert("RGB"))
    img_h, img_w = rgb.shape[:2]
    scale = dataset.img_size / max(img_w, img_h)
    new_w = int(img_w * scale)
    new_h = int(img_h * scale)
    pad_x = (dataset.img_size - new_w) // 2
    pad_y = (dataset.img_size - new_h) // 2
    K = np.array(
        [[RARP_FC, 0.0, img_w / 2.0],
         [0.0, RARP_FC, img_h / 2.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return rgb, K, scale, pad_x, pad_y


def quarter_pixels_to_original(xs, ys, q_size, img_size, scale, pad_x, pad_y):
    stride = img_size / float(q_size)
    x_pad = (xs.astype(np.float64) + 0.5) * stride
    y_pad = (ys.astype(np.float64) + 0.5) * stride
    x_orig = (x_pad - pad_x) / scale
    y_orig = (y_pad - pad_y) / scale
    return np.stack([x_orig, y_orig], axis=1)


def load_gt_pose(pose_root: str, video_name: str, instance_id: int, frame_id: str):
    _register_pose_stub()
    mem_path = (
        Path(pose_root)
        / f"SARRARP502022_{video_name}_instance{instance_id}"
        / "memory_pool.pth"
    )
    if not mem_path.is_file():
        raise FileNotFoundError(f"memory_pool.pth not found: {mem_path}")
    memory_pool = torch.load(mem_path, map_location="cpu", weights_only=False)
    if frame_id not in memory_pool:
        raise KeyError(f"frame {frame_id} not found in {mem_path}")
    entry = memory_pool[frame_id]
    pose = entry["pose_info"]
    rot_quat = pose.rot.detach().float().flatten()
    return {
        "rot_mat": _quaternion_to_matrix(rot_quat),
        "trans": pose.trans.detach().float().cpu().numpy().reshape(3).astype(np.float64),
        "alpha": float(pose.alpha.detach().float().flatten()[0]),
        "theta_l": float(pose.theta_l.detach().float().flatten()[0]),
        "theta_r": float(pose.theta_r.detach().float().flatten()[0]),
        "mem_path": str(mem_path),
    }


def match_gt_to_predictions(predictions, gt_wrist_centers, match_max_dist):
    if len(predictions) < len(gt_wrist_centers):
        raise RuntimeError(
            f"Predicted instances fewer than GT instances: "
            f"{len(predictions)} < {len(gt_wrist_centers)}"
        )
    pred_locs = np.stack([
        pred["loc"].detach().cpu().numpy().astype(np.float64)
        for pred in predictions
    ])
    used = set()
    matches = []
    for gt_idx, gt_loc in enumerate(gt_wrist_centers):
        dists = np.linalg.norm(pred_locs - gt_loc[None], axis=1)
        order = np.argsort(dists)
        chosen = None
        for pred_idx in order:
            if int(pred_idx) not in used:
                chosen = int(pred_idx)
                break
        if chosen is None:
            raise RuntimeError("Could not assign a prediction to every GT instance")
        if float(dists[chosen]) > match_max_dist:
            raise RuntimeError(
                f"Nearest prediction for GT instance {gt_idx} is too far: "
                f"{float(dists[chosen]):.2f}px > {match_max_dist:.2f}px"
            )
        used.add(chosen)
        matches.append((gt_idx, chosen, float(dists[chosen])))
    return matches


def _binary_iou(pred_mask, gt_mask):
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return float("nan")
    inter = np.logical_and(pred_mask, gt_mask).sum()
    return float(inter / union)


def compute_segmentation_metrics(pred, y, gt_idx, args):
    pred_inst = (
        torch.sigmoid(pred["inst_mask_logits"]).detach().cpu().numpy()
        >= args.inst_thresh
    )
    pred_part = torch.argmax(pred["part_mask_logits"], dim=0).detach().cpu().numpy()

    gt_inst = y["inst_masks"][0, gt_idx].numpy() > 0.5
    gt_part = y["part_masks"][0, gt_idx].numpy().astype(np.int64)

    metrics = {
        "inst_iou": _binary_iou(pred_inst, gt_inst),
        "pred_inst_area": int(pred_inst.sum()),
        "gt_inst_area": int(gt_inst.sum()),
    }
    part_ious = []
    for label, name in [(1, "gripper"), (2, "wrist"), (3, "shaft")]:
        # Measure part overlap inside the predicted/GT instance support so a
        # large background class cannot hide a bad part assignment.
        pred_mask = pred_inst & (pred_part == label)
        gt_mask = gt_inst & (gt_part == label)
        iou = _binary_iou(pred_mask, gt_mask)
        metrics[f"part_iou_{name}"] = iou
        if not np.isnan(iou):
            part_ious.append(iou)
    metrics["part_iou_mean"] = float(np.mean(part_ious)) if part_ious else float("nan")
    return metrics


def sample_mask_pixels(mask, max_points, rng):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return xs, ys
    if len(xs) > max_points:
        keep = rng.choice(len(xs), size=max_points, replace=False)
        xs = xs[keep]
        ys = ys[keep]
    return xs, ys


def build_correspondences(pred, cad: InstrumentCAD, geom, args, rng):
    _, _, scale, pad_x, pad_y = geom
    inst_prob = torch.sigmoid(pred["inst_mask_logits"]).detach().cpu().numpy()
    part_label = torch.argmax(pred["part_mask_logits"], dim=0).detach().cpu().numpy()
    nocs = pred["nocs_pred"].detach().cpu().numpy().transpose(1, 2, 0)
    q_size = inst_prob.shape[0]

    inst_mask = inst_prob >= args.inst_thresh
    all_uv = []
    all_points = []
    all_part_names = []
    wrist_uv = []
    wrist_points = []

    part_specs = [
        ("shaft", PART_LABELS["shaft"], args.max_points_per_part),
        ("wrist", PART_LABELS["wrist"], args.max_points_per_part),
        ("gripper", PART_LABELS["gripper"], args.max_points_per_part),
    ]

    for part_name, label, max_points in part_specs:
        part_mask = inst_mask & (part_label == label)
        xs, ys = sample_mask_pixels(part_mask, max_points, rng)
        if len(xs) == 0:
            continue

        uv = quarter_pixels_to_original(
            xs, ys, q_size, args.img_size, scale, pad_x, pad_y
        )
        valid_uv = (
            (uv[:, 0] >= 0.0)
            & (uv[:, 0] < geom[0].shape[1])
            & (uv[:, 1] >= 0.0)
            & (uv[:, 1] < geom[0].shape[0])
        )
        uv = uv[valid_uv]
        xs = xs[valid_uv]
        ys = ys[valid_uv]
        if len(xs) == 0:
            continue

        xyz_norm = nocs[ys, xs].astype(np.float64)
        if part_name == "gripper":
            points_part, names = cad.nearest_gripper_points(xyz_norm)
        else:
            points_part = cad.nearest_part_points(part_name, xyz_norm)
            names = np.full((len(points_part),), part_name, dtype=object)

        all_uv.append(uv)
        all_points.append(points_part)
        all_part_names.append(names)
        if part_name == "wrist":
            wrist_uv.append(uv)
            wrist_points.append(points_part)

    if not wrist_uv:
        raise RuntimeError("No predicted wrist CSE pixels for this instance")

    wrist_uv = np.concatenate(wrist_uv, axis=0)
    wrist_points = np.concatenate(wrist_points, axis=0)
    if len(wrist_uv) < args.min_wrist_points:
        raise RuntimeError(
            f"Not enough wrist correspondences: {len(wrist_uv)} < {args.min_wrist_points}"
        )

    if not all_uv:
        raise RuntimeError("No CSE correspondences for this instance")

    corr = Correspondences(
        uv=np.concatenate(all_uv, axis=0).astype(np.float64),
        points_part=np.concatenate(all_points, axis=0).astype(np.float64),
        part_names=np.concatenate(all_part_names, axis=0),
    )
    if len(corr) < args.min_total_points:
        raise RuntimeError(
            f"Not enough total correspondences: {len(corr)} < {args.min_total_points}"
        )
    return corr, wrist_uv.astype(np.float64), wrist_points.astype(np.float64)


def solve_wrist_pnp(wrist_uv, wrist_points, K, args):
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        wrist_points.astype(np.float32),
        wrist_uv.astype(np.float32),
        K.astype(np.float32),
        None,
        iterationsCount=args.pnp_iters,
        reprojectionError=args.pnp_reproj_error,
        confidence=args.pnp_confidence,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not ok or inliers is None:
        raise RuntimeError("Wrist PnP failed")
    inlier_count = int(len(inliers))
    if inlier_count < args.min_pnp_inliers:
        raise RuntimeError(
            f"Not enough PnP inliers: {inlier_count} < {args.min_pnp_inliers}"
        )
    return rvec.reshape(3).astype(np.float64), tvec.reshape(3).astype(np.float64), inlier_count


def optimize_pose(cad: InstrumentCAD, corr: Correspondences, rvec0, trans0, K, args):
    wg_mask = np.isin(corr.part_names, ["wrist", "l_gripper", "r_gripper"])
    shaft_mask = corr.part_names == "shaft"

    def subset_corr(mask):
        return Correspondences(
            uv=corr.uv[mask],
            points_part=corr.points_part[mask],
            part_names=corr.part_names[mask],
        )

    def residual_from_pose(active_corr, rvec, trans, alpha, theta_l, theta_r):
        transforms = cad.fk(rvec, trans, alpha, theta_l, theta_r)
        residuals = []
        part_indices = {
            part_name: np.where(active_corr.part_names == part_name)[0]
            for part_name in sorted(set(active_corr.part_names.tolist()))
        }
        for part_name, idx in part_indices.items():
            points_cam = cad.transform_points(
                part_name, active_corr.points_part[idx], transforms
            )
            z_ok = points_cam[:, 2] > args.min_depth
            if not np.all(z_ok):
                bad = np.where(~z_ok)[0][:5].tolist()
                raise FloatingPointError(
                    f"Projected points behind camera for {part_name}; bad indices {bad}"
                )
            proj = _project(points_cam, K)
            residuals.append((proj - active_corr.uv[idx]).reshape(-1))
        return np.concatenate(residuals, axis=0)

    def rmse_for(mask, params):
        if not bool(mask.any()):
            return float("nan")
        active = subset_corr(mask)
        res = residual_from_pose(
            active, params[:3], params[3:6], params[6], params[7], params[8]
        )
        return float(np.sqrt(np.mean(res.reshape(-1, 2) ** 2)))

    joint_lower = np.array(
        [-math.pi / 2.0, -80.0 / 180.0 * math.pi, -80.0 / 180.0 * math.pi],
        dtype=np.float64,
    )
    joint_upper = np.array(
        [math.pi / 2.0, 80.0 / 180.0 * math.pi, 80.0 / 180.0 * math.pi],
        dtype=np.float64,
    )

    if args.optim_strategy == "decoupled":
        if int(wg_mask.sum()) < args.min_total_points:
            raise RuntimeError(
                f"Not enough wrist/gripper correspondences: "
                f"{int(wg_mask.sum())} < {args.min_total_points}"
            )
        if int(shaft_mask.sum()) < args.min_shaft_points:
            raise RuntimeError(
                f"Not enough shaft correspondences: "
                f"{int(shaft_mask.sum())} < {args.min_shaft_points}"
            )

        wg_corr = subset_corr(wg_mask)
        shaft_corr = subset_corr(shaft_mask)

        def residual_wrist_gripper(params):
            rvec = params[:3]
            trans = params[3:6]
            theta_l, theta_r = params[6], params[7]
            return residual_from_pose(wg_corr, rvec, trans, 0.0, theta_l, theta_r)

        x0_wg = np.array(
            [rvec0[0], rvec0[1], rvec0[2],
             trans0[0], trans0[1], trans0[2],
             0.0, 0.0],
            dtype=np.float64,
        )
        lower_wg = np.array(
            [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf,
             joint_lower[1], joint_lower[2]],
            dtype=np.float64,
        )
        upper_wg = np.array(
            [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
             joint_upper[1], joint_upper[2]],
            dtype=np.float64,
        )
        res_wg = least_squares(
            residual_wrist_gripper,
            x0_wg,
            bounds=(lower_wg, upper_wg),
            loss=args.optim_loss,
            f_scale=args.optim_f_scale,
            max_nfev=args.optim_max_nfev,
            verbose=0,
        )
        if not res_wg.success:
            raise RuntimeError(f"Wrist/gripper optimization failed: {res_wg.message}")

        rvec_opt = res_wg.x[:3]
        trans_opt = res_wg.x[3:6]
        theta_l_opt = float(res_wg.x[6])
        theta_r_opt = float(res_wg.x[7])

        def residual_shaft(alpha_arr):
            return residual_from_pose(
                shaft_corr,
                rvec_opt,
                trans_opt,
                float(alpha_arr[0]),
                theta_l_opt,
                theta_r_opt,
            )

        res_shaft = least_squares(
            residual_shaft,
            np.zeros(1, dtype=np.float64),
            bounds=(np.array([joint_lower[0]]), np.array([joint_upper[0]])),
            loss=args.optim_loss,
            f_scale=args.optim_f_scale,
            max_nfev=args.optim_max_nfev,
            verbose=0,
        )
        if not res_shaft.success:
            raise RuntimeError(f"Shaft alpha optimization failed: {res_shaft.message}")

        opt_params = np.array(
            [rvec_opt[0], rvec_opt[1], rvec_opt[2],
             trans_opt[0], trans_opt[1], trans_opt[2],
             float(res_shaft.x[0]), theta_l_opt, theta_r_opt],
            dtype=np.float64,
        )
        nfev = int(res_wg.nfev + res_shaft.nfev)

    elif bool(args.freeze_wrist_after_pnp):
        if args.optim_parts == "wrist_gripper":
            optim_mask = wg_mask
        elif args.optim_parts == "all":
            optim_mask = np.ones(len(corr), dtype=bool)
        else:
            raise ValueError(f"Unsupported optim_parts={args.optim_parts!r}")
        if int(optim_mask.sum()) < args.min_total_points:
            raise RuntimeError(
                f"Not enough optimization correspondences for {args.optim_parts}: "
                f"{int(optim_mask.sum())} < {args.min_total_points}"
            )
        opt_corr = subset_corr(optim_mask)

        def residual_joints(joints):
            alpha, theta_l, theta_r = joints[0], joints[1], joints[2]
            return residual_from_pose(opt_corr, rvec0, trans0, alpha, theta_l, theta_r)

        x0 = np.zeros(3, dtype=np.float64)
        res = least_squares(
            residual_joints,
            x0,
            bounds=(joint_lower, joint_upper),
            loss=args.optim_loss,
            f_scale=args.optim_f_scale,
            max_nfev=args.optim_max_nfev,
            verbose=0,
        )
        opt_params = np.array(
            [rvec0[0], rvec0[1], rvec0[2],
             trans0[0], trans0[1], trans0[2],
             res.x[0], res.x[1], res.x[2]],
            dtype=np.float64,
        )
        if not res.success:
            raise RuntimeError(f"Pose optimization failed: {res.message}")
        nfev = int(res.nfev)
    else:
        if args.optim_parts == "wrist_gripper":
            optim_mask = wg_mask
        elif args.optim_parts == "all":
            optim_mask = np.ones(len(corr), dtype=bool)
        else:
            raise ValueError(f"Unsupported optim_parts={args.optim_parts!r}")
        if int(optim_mask.sum()) < args.min_total_points:
            raise RuntimeError(
                f"Not enough optimization correspondences for {args.optim_parts}: "
                f"{int(optim_mask.sum())} < {args.min_total_points}"
            )
        opt_corr = subset_corr(optim_mask)

        def residual_full(params):
            return residual_from_pose(
                opt_corr,
                params[:3],
                params[3:6],
                params[6],
                params[7],
                params[8],
            )

        x0 = np.array(
            [rvec0[0], rvec0[1], rvec0[2],
             trans0[0], trans0[1], trans0[2],
             0.0, 0.0, 0.0],
            dtype=np.float64,
        )
        lower = np.array(
            [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf,
             joint_lower[0], joint_lower[1], joint_lower[2]],
            dtype=np.float64,
        )
        upper = np.array(
            [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
             joint_upper[0], joint_upper[1], joint_upper[2]],
            dtype=np.float64,
        )
        res = least_squares(
            residual_full,
            x0,
            bounds=(lower, upper),
            loss=args.optim_loss,
            f_scale=args.optim_f_scale,
            max_nfev=args.optim_max_nfev,
            verbose=0,
        )
        opt_params = res.x
        if not res.success:
            raise RuntimeError(f"Pose optimization failed: {res.message}")
        nfev = int(res.nfev)

    rmse_all = rmse_for(np.ones(len(corr), dtype=bool), opt_params)
    rmse_wrist_gripper = rmse_for(wg_mask, opt_params)
    rmse_shaft = rmse_for(shaft_mask, opt_params)
    return opt_params, rmse_all, nfev, rmse_wrist_gripper, rmse_shaft


def _shade_color(base_color, normal_cam):
    normal = np.asarray(normal_cam, dtype=np.float64)
    norm = np.linalg.norm(normal)
    if norm < 1e-12:
        shade = 0.6
    else:
        normal = normal / norm
        light = np.array([-0.35, -0.25, -1.0], dtype=np.float64)
        light = light / np.linalg.norm(light)
        shade = 0.35 + 0.65 * max(0.0, float(np.dot(normal, -light)))
    return tuple(np.clip(np.asarray(base_color) * shade, 0, 255).astype(np.uint8).tolist())


def _paint_projected_triangles(cad, posed_vertices, K, args, shade_mesh=False):
    h = int(args.render_h)
    w = int(args.render_w)
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    triangles = []

    for part_name, vertices_cam in posed_vertices.items():
        faces = cad.faces[part_name]
        uv = np.zeros((vertices_cam.shape[0], 2), dtype=np.float64)
        valid_z = vertices_cam[:, 2] > args.min_depth
        uv[valid_z] = _project(vertices_cam[valid_z], K)
        for face in faces:
            if not np.all(valid_z[face]):
                continue
            pts = uv[face]
            if (
                pts[:, 0].max() < -args.draw_margin
                or pts[:, 1].max() < -args.draw_margin
                or pts[:, 0].min() >= w + args.draw_margin
                or pts[:, 1].min() >= h + args.draw_margin
            ):
                continue

            tri3d = vertices_cam[face]
            depth = float(tri3d[:, 2].mean())
            if shade_mesh:
                normal = np.cross(tri3d[1] - tri3d[0], tri3d[2] - tri3d[0])
                color = _shade_color((205, 205, 195), normal)
            else:
                color = DRAW_COLORS[part_name]
            triangles.append((depth, pts, color))

    # Painter pass: far triangles first, near triangles overwrite them.
    for _, pts, color in sorted(triangles, key=lambda item: item[0], reverse=True):
        poly = np.round(pts).astype(np.int32)
        cv2.fillConvexPoly(canvas, poly, color, lineType=cv2.LINE_AA)

    return canvas


def _render_trimesh_panel(cad, posed_vertices, K, h, w, args):
    try:
        import pyrender
    except ImportError as exc:
        raise ImportError("pyrender is required for --mesh_render_backend pyrender") from exc

    mesh_list = []
    for part_name in ("shaft", "wrist", "l_gripper", "r_gripper"):
        mesh = trimesh.Trimesh(
            vertices=posed_vertices[part_name],
            faces=cad.faces[part_name],
            process=False,
        )
        mesh_list.append(mesh)
    combined_mesh = trimesh.util.concatenate(mesh_list)

    extrinsic = np.eye(4, dtype=np.float64)
    extrinsic[:, 1:3] *= -1

    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.08, 0.08, 0.08])
    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        roughnessFactor=0.75,
        baseColorFactor=[0.78, 0.78, 0.72, 1.0],
        alphaMode="OPAQUE",
    )
    scene.add(pyrender.Mesh.from_trimesh(combined_mesh, material=material, smooth=False))

    camera = pyrender.IntrinsicsCamera(
        fx=float(K[0, 0]),
        fy=float(K[1, 1]),
        cx=float(K[0, 2]),
        cy=float(K[1, 2]),
        znear=args.pyrender_znear,
        zfar=args.pyrender_zfar,
    )
    scene.add(camera, pose=extrinsic)

    light = pyrender.PointLight(intensity=args.pyrender_light_intensity)
    light_pose = extrinsic.copy()
    light_pose[:, 1:3] *= -1
    light_pose[2, 3] += 0.1
    light_pose[0, 3] += 0.1
    scene.add(light, pose=light_pose)

    renderer = pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h)
    try:
        color, _ = renderer.render(scene)
    finally:
        renderer.delete()
    return color[:, :, :3].astype(np.uint8)


def draw_projected_mesh(rgb, cad: InstrumentCAD, pose_params, K, args):
    rvec = pose_params[:3]
    trans = pose_params[3:6]
    alpha, theta_l, theta_r = pose_params[6], pose_params[7], pose_params[8]
    posed_vertices = cad.posed_mesh_vertices(rvec, trans, alpha, theta_l, theta_r)

    args.render_h, args.render_w = rgb.shape[:2]
    semantic_panel = _paint_projected_triangles(cad, posed_vertices, K, args, shade_mesh=False)
    if args.mesh_render_backend == "pyrender":
        mesh_panel = _render_trimesh_panel(
            cad, posed_vertices, K, rgb.shape[0], rgb.shape[1], args
        )
    else:
        mesh_panel = _paint_projected_triangles(cad, posed_vertices, K, args, shade_mesh=True)
    overlay = cv2.addWeighted(rgb, 1.0, semantic_panel, args.overlay_alpha, 0)
    sep = np.full((rgb.shape[0], 6, 3), 255, dtype=np.uint8)
    return np.concatenate([rgb, sep, semantic_panel, sep, mesh_panel, sep, overlay], axis=1)


def save_npz(path, **kwargs):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **kwargs)


def process_sample(dataset, sample_idx, model, cad, device, args, rng, writer, do_vis):
    video_name, frame_id, instances_list = dataset.samples[sample_idx]
    img_array, annot = dataset[sample_idx]
    if len(annot["instruments"]) != len(instances_list):
        raise RuntimeError(
            f"Instrument count changed during __getitem__ for {video_name}/{frame_id}: "
            f"{len(annot['instruments'])} != {len(instances_list)}"
        )

    batch_img, y = collate_fn_instrument([(img_array, annot)])
    with torch.no_grad():
        preds = model(
            batch_img.to(device),
            is_training=False,
            det_thresh=args.det_thresh,
            nms_kernel_size=args.nms_kernel_size,
        )
    if not isinstance(preds, list) or len(preds) == 0:
        raise RuntimeError(f"No predicted instances for {video_name}/{frame_id}")

    valid_gt = y["valid_instruments"][0].bool().numpy()
    gt_wrist_centers = y["wrist_centers"][0][valid_gt].numpy().astype(np.float64)
    matches = match_gt_to_predictions(preds, gt_wrist_centers, args.match_max_dist)

    geom = image_geometry(dataset, video_name, frame_id)
    rgb, K = geom[0], geom[1]
    canvas = rgb.copy()
    frame_rows = []

    for gt_idx, pred_idx, match_dist in matches:
        instance_id = int(instances_list[gt_idx][0])
        pred = preds[pred_idx]
        seg_metrics = compute_segmentation_metrics(pred, y, gt_idx, args)
        corr, wrist_uv, wrist_points = build_correspondences(pred, cad, geom, args, rng)
        rvec0, trans0, pnp_inliers = solve_wrist_pnp(wrist_uv, wrist_points, K, args)
        (
            opt_params,
            reproj_rmse,
            opt_nfev,
            reproj_rmse_wrist_gripper,
            reproj_rmse_shaft,
        ) = optimize_pose(cad, corr, rvec0, trans0, K, args)

        gt_pose = load_gt_pose(dataset.pose_root, video_name, instance_id, frame_id)
        R_pred, _ = cv2.Rodrigues(opt_params[:3].reshape(3, 1))
        rot_err_deg = _rotation_error_deg(R_pred, gt_pose["rot_mat"])
        trans_err_m = float(np.linalg.norm(opt_params[3:6] - gt_pose["trans"]))
        alpha_err = abs(float(opt_params[6]) - gt_pose["alpha"])
        theta_l_err = abs(float(opt_params[7]) - gt_pose["theta_l"])
        theta_r_err = abs(float(opt_params[8]) - gt_pose["theta_r"])

        row = {
            "dataset": args.dataset_name,
            "video": video_name,
            "frame_id": frame_id,
            "instance_id": instance_id,
            "sample_idx": sample_idx,
            "pred_idx": pred_idx,
            "gt_idx": gt_idx,
            "match_dist_px": match_dist,
            **seg_metrics,
            "corr_count": len(corr),
            "wrist_corr_count": len(wrist_uv),
            "pnp_inliers": pnp_inliers,
            "optim_nfev": opt_nfev,
            "reproj_rmse_px": reproj_rmse,
            "reproj_rmse_wrist_gripper_px": reproj_rmse_wrist_gripper,
            "reproj_rmse_shaft_px": reproj_rmse_shaft,
            "rot_err_deg": rot_err_deg,
            "trans_err_m": trans_err_m,
            "alpha_err_rad": alpha_err,
            "theta_l_err_rad": theta_l_err,
            "theta_r_err_rad": theta_r_err,
            "pred_rvec_x": opt_params[0],
            "pred_rvec_y": opt_params[1],
            "pred_rvec_z": opt_params[2],
            "pred_tx": opt_params[3],
            "pred_ty": opt_params[4],
            "pred_tz": opt_params[5],
            "pred_alpha": opt_params[6],
            "pred_theta_l": opt_params[7],
            "pred_theta_r": opt_params[8],
        }
        writer.writerow(row)
        frame_rows.append(row)

        save_npz(
            args.output_dir / "pred_poses" / f"{video_name}_{frame_id}_inst{instance_id}.npz",
            pred_params=opt_params,
            pnp_rvec=rvec0,
            pnp_trans=trans0,
            gt_rot_mat=gt_pose["rot_mat"],
            gt_trans=gt_pose["trans"],
            gt_joints=np.array([gt_pose["alpha"], gt_pose["theta_l"], gt_pose["theta_r"]]),
            correspondences_uv=corr.uv,
            correspondences_points_part=corr.points_part,
            correspondences_part_names=corr.part_names.astype(str),
            wrist_uv=wrist_uv,
            wrist_points=wrist_points,
        )

        if do_vis:
            inst_canvas = draw_projected_mesh(rgb, cad, opt_params, K, args)
            Image.fromarray(inst_canvas).save(
                args.output_dir
                / "vis"
                / f"{video_name}_{frame_id}_inst{instance_id}.png"
            )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate articulate instrument pose from predicted CSE on one RARPInstanceDataset video."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("logs/instrument_rarp_cse_dpt_2gpu_bs8/checkpoints/best.pt"),
    )
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument(
        "--dataset_name",
        choices=sorted(DATASETS.keys()),
        default="needlePuncture",
    )
    parser.add_argument("--video_select", choices=["random"], default="random")
    parser.add_argument("--video_name", type=str, default=None)
    parser.add_argument("--num_videos", type=int, default=1)
    parser.add_argument(
        "--frame_select",
        choices=["first", "random", "even"],
        default="first",
    )
    parser.add_argument(
        "--frames_per_video",
        type=int,
        default=-1,
        help="Number of frames per selected video. Defaults to --max_frames for backward compatibility.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_frames", type=int, default=-1)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--img_size", type=int, default=630)
    parser.add_argument("--train_ratio", type=float, default=0.95)
    parser.add_argument("--min_dice_shaft", type=float, default=0.8)
    parser.add_argument("--min_dice_wrist", type=float, default=0.6)
    parser.add_argument("--min_dice_gripper", type=float, default=0.6)
    parser.add_argument("--det_thresh", type=float, default=0.3)
    parser.add_argument("--nms_kernel_size", type=int, default=3)
    parser.add_argument("--match_max_dist", type=float, default=80.0)

    parser.add_argument("--inst_thresh", type=float, default=0.5)
    parser.add_argument("--max_points_per_part", type=int, default=1200)
    parser.add_argument("--min_wrist_points", type=int, default=12)
    parser.add_argument("--min_total_points", type=int, default=24)
    parser.add_argument("--min_shaft_points", type=int, default=24)
    parser.add_argument("--min_pnp_inliers", type=int, default=8)
    parser.add_argument("--pnp_iters", type=int, default=300)
    parser.add_argument("--pnp_reproj_error", type=float, default=8.0)
    parser.add_argument("--pnp_confidence", type=float, default=0.99)
    parser.add_argument(
        "--freeze_wrist_after_pnp",
        type=int,
        default=0,
        choices=[0, 1],
        help="Legacy single-stage mode: keep wrist rot/trans fixed after wrist-CSE PnP and optimize only articulation joints.",
    )
    parser.add_argument(
        "--optim_strategy",
        choices=["decoupled", "single"],
        default="decoupled",
        help=(
            "decoupled: wrist+gripper optimize wrist pose and gripper joints, "
            "then shaft only optimizes alpha. single: use freeze_wrist_after_pnp/optim_parts."
        ),
    )
    parser.add_argument(
        "--optim_parts",
        choices=["wrist_gripper", "all"],
        default="wrist_gripper",
        help="Correspondence parts used by pose optimization. Default excludes shaft so shaft CSE cannot move wrist/gripper.",
    )
    parser.add_argument("--optim_loss", choices=["linear", "soft_l1", "huber", "cauchy", "arctan"], default="soft_l1")
    parser.add_argument("--optim_f_scale", type=float, default=8.0)
    parser.add_argument("--optim_max_nfev", type=int, default=200)
    parser.add_argument("--min_depth", type=float, default=1e-4)

    parser.add_argument("--vis_every", type=int, default=1)
    parser.add_argument(
        "--skip_failed",
        type=int,
        default=0,
        choices=[0, 1],
        help="For large debug sweeps, print failed frames and continue without fallback predictions.",
    )
    parser.add_argument("--line_width", type=int, default=1)
    parser.add_argument("--draw_margin", type=int, default=64)
    parser.add_argument("--overlay_alpha", type=float, default=0.85)
    parser.add_argument(
        "--mesh_render_backend",
        choices=["pyrender", "software"],
        default="pyrender",
        help="Mesh visualization panel backend. pyrender follows Instrument.render_trimesh.",
    )
    parser.add_argument("--pyrender_znear", type=float, default=0.001)
    parser.add_argument("--pyrender_zfar", type=float, default=0.6)
    parser.add_argument("--pyrender_light_intensity", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = REPO_ROOT / "debug_vis" / "cse_pose_single_video"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "pred_poses").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "vis").mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    device = torch.device(args.device)
    model, model_args = load_model(args.checkpoint, device)
    if int(model_args["img_size"]) != int(args.img_size):
        raise ValueError(
            f"Script --img_size={args.img_size} does not match checkpoint img_size={model_args['img_size']}"
        )

    cad = InstrumentCAD(CAD_ROOT)
    dataset = build_dataset(args)
    video_names = select_videos(dataset, args)
    frames_per_video = args.frames_per_video if args.frames_per_video > 0 else args.max_frames
    sample_indices = []
    for video_name in video_names:
        sample_indices.extend(
            indices_for_video(
                dataset,
                video_name,
                frames_per_video,
                frame_select=args.frame_select,
                seed=args.seed,
            )
        )

    selected_path = args.output_dir / "selected_video.txt"
    selected_path.write_text(
        f"dataset={args.dataset_name}\n"
        f"videos={','.join(video_names)}\n"
        f"seed={args.seed}\n"
        f"frame_select={args.frame_select}\n"
        f"frames_per_video={frames_per_video}\n"
        f"num_frames={len(sample_indices)}\n"
        f"checkpoint={args.checkpoint}\n",
        encoding="utf-8",
    )

    csv_path = args.output_dir / "pred_poses.csv"
    fieldnames = [
        "dataset", "video", "frame_id", "instance_id", "sample_idx",
        "pred_idx", "gt_idx", "match_dist_px", "corr_count",
        "inst_iou", "part_iou_mean", "part_iou_gripper",
        "part_iou_wrist", "part_iou_shaft", "pred_inst_area",
        "gt_inst_area",
        "wrist_corr_count", "pnp_inliers", "optim_nfev",
        "reproj_rmse_px", "reproj_rmse_wrist_gripper_px",
        "reproj_rmse_shaft_px", "rot_err_deg", "trans_err_m",
        "alpha_err_rad", "theta_l_err_rad", "theta_r_err_rad",
        "pred_rvec_x", "pred_rvec_y", "pred_rvec_z",
        "pred_tx", "pred_ty", "pred_tz",
        "pred_alpha", "pred_theta_l", "pred_theta_r",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for count, sample_idx in enumerate(sample_indices):
            do_vis = args.vis_every > 0 and count % args.vis_every == 0
            try:
                process_sample(dataset, sample_idx, model, cad, device, args, rng, writer, do_vis)
            except Exception as exc:
                if not bool(args.skip_failed):
                    raise
                print(
                    f"[skip] {dataset.samples[sample_idx][0]}/"
                    f"{dataset.samples[sample_idx][1]}: {type(exc).__name__}: {exc}",
                    flush=True,
                )
            f.flush()
            print(
                f"[{count + 1}/{len(sample_indices)}] processed "
                f"{dataset.samples[sample_idx][0]}/{dataset.samples[sample_idx][1]}",
                flush=True,
            )

    print(f"[ok] selected videos: {', '.join(video_names)}")
    print(f"[ok] wrote {csv_path}")
    print(f"[ok] wrote visualizations to {args.output_dir / 'vis'}")


if __name__ == "__main__":
    main()
