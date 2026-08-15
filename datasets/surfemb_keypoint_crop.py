import importlib.util
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

ROBOPEPP_ROOT = Path(__file__).resolve().parents[1]
MULTIHMR_ROOT = Path(__file__).resolve().parents[3]
if str(ROBOPEPP_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBOPEPP_ROOT))
if str(MULTIHMR_ROOT) not in sys.path:
    sys.path.insert(0, str(MULTIHMR_ROOT))

from instrument_geometry import (  # noqa: E402
    GRIPPER_JOINT_OFFSET_M,
    SURFEMB_SHAFT_NORM_X_MIN,
    fk_matrices_np,
    make_transform_np,
    project_points_np,
    surfemb_surface_sampling_mask,
)


def _load_local_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_rarp_instrument = _load_local_module(
    "robopepp_rarp_instrument_for_surfemb_keypoint",
    ROBOPEPP_ROOT / "datasets" / "rarp_instrument.py",
)
_lnd_instrument = _load_local_module(
    "robopepp_lnd_instrument_for_surfemb_keypoint",
    ROBOPEPP_ROOT / "datasets" / "surgripe_lnd_instrument.py",
)
_surf_aug = _load_local_module(
    "robopepp_surfemb_augment_for_keypoint",
    ROBOPEPP_ROOT / "datasets" / "surfemb_augment.py",
)

RoboPEPPRARPInstrument = _rarp_instrument.RoboPEPPRARPInstrument
RoboPEPPSurgripeLNDInstrument = _lnd_instrument.RoboPEPPSurgripeLNDInstrument


def _as_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _current_egl_device_id():
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        # Forked DataLoader workers must not touch the inherited CUDA runtime.
        # They only need the physical device index for their independent EGL context.
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if visible.strip():
            ids = [v.strip() for v in visible.split(",") if v.strip()]
            if local_rank < len(ids):
                try:
                    return int(ids[local_rank])
                except ValueError:
                    pass
        return local_rank
    if not torch.cuda.is_available():
        return 0
    current = int(torch.cuda.current_device())
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible.strip():
        ids = [v.strip() for v in visible.split(",") if v.strip()]
        if current < len(ids):
            try:
                return int(ids[current])
            except ValueError:
                pass
    return current


def _create_belief_maps(image_resolution, points, valid, sigma=2.0):
    width, height = image_resolution
    out = np.zeros((len(points), height, width), dtype=np.float32)
    radius = int(float(sigma) * 2.0)
    for i, (point, is_valid) in enumerate(zip(points, valid)):
        if not bool(is_valid):
            continue
        u = int(point[0])
        v = int(point[1])
        if not (0 <= u < width and 0 <= v < height):
            continue
        x0 = max(0, u - radius)
        x1 = min(width, u + radius + 1)
        y0 = max(0, v - radius)
        y1 = min(height, v + radius + 1)
        for x in range(x0, x1):
            for y in range(y0, y1):
                out[i, y, x] = np.exp(-(((x - u) ** 2 + (y - v) ** 2) / (2.0 * float(sigma) ** 2)))
    return out


class _NoProjectedSurfaceSamples(RuntimeError):
    pass


class _LowRenderIouSample(_NoProjectedSurfaceSamples):
    pass


class _MissingPartSampleGroup(_NoProjectedSurfaceSamples):
    pass


PART_SAMPLE_GROUPS = (
    ("shaft", (1,)),
    ("wrist", (2,)),
    ("gripper", (3, 4)),
)


def _validate_part_sample_ratios(ratios):
    ratios = np.asarray(ratios, dtype=np.float64).reshape(-1)
    if ratios.shape != (len(PART_SAMPLE_GROUPS),):
        raise ValueError(
            f"Expected {len(PART_SAMPLE_GROUPS)} part sampling ratios, got {ratios.tolist()}."
        )
    if np.any(ratios < 0.0) or not np.isclose(ratios.sum(), 1.0, atol=1e-6):
        raise ValueError(f"Part sampling ratios must be non-negative and sum to 1, got {ratios.tolist()}.")
    return ratios


def _allocate_part_sample_counts(n_samples, ratios):
    ratios = _validate_part_sample_ratios(ratios)
    raw = ratios * int(n_samples)
    counts = np.floor(raw).astype(np.int64)
    remainder = int(n_samples) - int(counts.sum())
    if remainder > 0:
        order = np.argsort(-(raw - counts), kind="stable")
        counts[order[:remainder]] += 1
    return counts


def _sample_part_balanced_indices(candidate_indices, effective_part_ids, n_samples, ratios, strict=True):
    candidate_indices = np.asarray(candidate_indices, dtype=np.int64).reshape(-1)
    effective_part_ids = np.asarray(effective_part_ids, dtype=np.int64).reshape(-1)
    ratios = _validate_part_sample_ratios(ratios)
    pools = [
        candidate_indices[np.isin(effective_part_ids[candidate_indices], np.asarray(part_ids))]
        for _, part_ids in PART_SAMPLE_GROUPS
    ]
    available = np.asarray([len(pool) > 0 for pool in pools], dtype=bool)
    requested = ratios > 0.0
    if strict and np.any(requested & ~available):
        missing = [PART_SAMPLE_GROUPS[i][0] for i in np.flatnonzero(requested & ~available)]
        raise _MissingPartSampleGroup(f"No visible samples for balanced part groups: {missing}.")
    if not available.any():
        raise _NoProjectedSurfaceSamples("No candidates available for part-balanced sampling.")

    effective_ratios = ratios.copy()
    if np.any(requested & ~available):
        effective_ratios[~available] = 0.0
        effective_ratios /= effective_ratios.sum()
    counts = _allocate_part_sample_counts(n_samples, effective_ratios)
    picked = []
    for pool, count in zip(pools, counts):
        if count <= 0:
            continue
        picked.append(np.random.choice(pool, int(count), replace=int(count) > len(pool)))
    picked = np.concatenate(picked).astype(np.int64)
    np.random.shuffle(picked)
    return picked


def _default_surface_points_path():
    refined = (
        ROBOPEPP_ROOT
        / "assets"
        / "instrument_surface_samples_surfemb_x2.13mm_wg1over3_shafttop30mm"
        / "instrument_surface_points_all.npy"
    )
    if refined.is_file():
        return refined
    return ROBOPEPP_ROOT / "assets" / "instrument_surface_samples" / "instrument_surface_points_all.npy"


PARTS = ("shaft", "wrist", "l_gripper", "r_gripper")
_OPENGL_RENDERER_CACHE = {}


def _load_surface_payloads(surface_points_path):
    root = Path(surface_points_path).resolve().parent
    payloads = {}
    for part in PARTS:
        path = root / f"{part}_surface_points.npy"
        if not path.is_file():
            raise FileNotFoundError(f"Surface payload not found: {path}")
        payloads[part] = np.load(path, allow_pickle=True).item()
    return payloads


def _transform_points_normals(points, normals, T):
    points_cam = points @ T[:3, :3].T + T[:3, 3]
    normals_cam = normals @ T[:3, :3].T
    normals_cam /= np.linalg.norm(normals_cam, axis=1, keepdims=True).clip(1e-8)
    return points_cam, normals_cam


def _transform_surface_payload(part_name, payload, transforms):
    points = np.asarray(payload["points_part_m"], dtype=np.float64)
    normals = np.asarray(payload["normals_part"], dtype=np.float64)
    points_cam = np.empty_like(points)
    normals_cam = np.empty_like(normals)
    static = np.asarray(payload.get("static_wrist_mask", np.zeros((len(points),), dtype=bool))).astype(bool)
    if part_name not in ("l_gripper", "r_gripper"):
        static[:] = False

    moving = ~static
    if moving.any():
        pts, nrm = _transform_points_normals(points[moving], normals[moving], transforms[part_name])
        points_cam[moving] = pts
        normals_cam[moving] = nrm
    if static.any():
        # Rear gripper points are embedded in the wrist in the current surface
        # version, so they keep the wrist frame and skip gripper rotation.
        wrist_to_static = make_transform_np(
            np.eye(3),
            [GRIPPER_JOINT_OFFSET_M, 0.0, 0.0],
        )
        pts, nrm = _transform_points_normals(points[static], normals[static], transforms["wrist"] @ wrist_to_static)
        points_cam[static] = pts
        normals_cam[static] = nrm
    return points_cam, normals_cam


class SurfEmbKeypointCropDataset(Dataset):
    """
    SurfEmb-style crop/correspondence dataset plus RoboPEPP joint keypoints.

    The crop order follows the original SurfEmb training path more closely than
    the HCCE wrapper: crop matrix is defined from the full visible instrument
    mask, RGB is warped into crop space, and SurfEmb positive pairs are sampled
    by projecting visible surface samples into the crop.  It deliberately does
    not render dense HCCE/canonical-coordinate maps in the dataloader.
    """

    def __init__(
        self,
        base_dataset,
        surface_points_path=None,
        crop_size=224,
        n_pos=1024,
        n_neg=1024,
        crop_scale=1.2,
        max_angle=math.pi,
        offset_scale=1.0,
        training=True,
        heatmap_sigma=2.0,
        min_depth=1e-4,
        depth_tolerance=0.0008,
        use_mesh_zbuffer=True,
        zbuffer_backend="opengl",
        min_render_iou=0.0,
        part_sample_ratios=(0.2, 0.6, 0.2),
        shaft_norm_x_min=SURFEMB_SHAFT_NORM_X_MIN,
        supervision_part_ids=None,
        min_supervision_pixels=1,
        negative_visible_only=False,
        fallback_to_other_sample=True,
        augmentation_profile="legacy",
        crop_min_mask_retention=0.985,
        crop_offset_multiplier=1.0,
    ):
        super().__init__()
        self.base_dataset = base_dataset
        self.surface_points_path = Path(surface_points_path) if surface_points_path else _default_surface_points_path()
        self.crop_size = int(crop_size)
        self.n_pos = int(n_pos)
        self.n_neg = int(n_neg)
        self.crop_scale = float(crop_scale)
        self.max_angle = float(max_angle)
        self.offset_scale = float(offset_scale)
        self.training = bool(training)
        self.heatmap_sigma = float(heatmap_sigma)
        self.surface_points, self.surface_part_ids = _surf_aug.load_surface_points(self.surface_points_path)
        self.shaft_norm_x_min = float(shaft_norm_x_min)
        self.surface_sample_indices = np.flatnonzero(
            surfemb_surface_sampling_mask(
                self.surface_points,
                self.surface_part_ids,
                self.shaft_norm_x_min,
            )
        )
        self.surface_payloads = _load_surface_payloads(self.surface_points_path)
        self.min_depth = float(min_depth)
        self.depth_tolerance = float(depth_tolerance)
        self.use_mesh_zbuffer = bool(use_mesh_zbuffer)
        self.zbuffer_backend = str(zbuffer_backend)
        if not self.use_mesh_zbuffer:
            raise ValueError(
                "SurfEmb positive supervision requires OpenGL mesh-triangle coordinate rasterization."
            )
        if self.zbuffer_backend != "opengl":
            raise ValueError(
                "SurfEmb visible-surface rendering is OpenGL-only; "
                f"got zbuffer_backend={self.zbuffer_backend!r}."
            )
        self.min_render_iou = float(min_render_iou)
        self.part_sample_ratios = _validate_part_sample_ratios(part_sample_ratios)
        self.supervision_part_ids = (
            None
            if supervision_part_ids is None
            else tuple(int(v) for v in supervision_part_ids)
        )
        self.min_supervision_pixels = int(min_supervision_pixels)
        if self.min_supervision_pixels < 1:
            raise ValueError(f"min_supervision_pixels must be positive, got {self.min_supervision_pixels}.")
        self.negative_visible_only = bool(negative_visible_only)
        self.fallback_to_other_sample = bool(fallback_to_other_sample)
        self.augmentation_profile = str(augmentation_profile)
        if self.augmentation_profile not in ("legacy", "original_p30"):
            raise ValueError(f"Unknown SurfEmb augmentation profile: {self.augmentation_profile!r}")
        self.crop_min_mask_retention = float(crop_min_mask_retention)
        if not 0.0 < self.crop_min_mask_retention <= 1.0:
            raise ValueError(
                f"crop_min_mask_retention must be in (0, 1], got {self.crop_min_mask_retention}."
            )
        self.crop_offset_multiplier = float(crop_offset_multiplier)
        if self.crop_offset_multiplier < 0.0:
            raise ValueError(f"crop_offset_multiplier must be non-negative, got {self.crop_offset_multiplier}.")
        self._mesh_renderer = None

    def __len__(self):
        return len(self.base_dataset)

    def __repr__(self):
        return (
            f"surfemb_keypoint_crop(training={self.training} N={len(self)} "
            f"crop_size={self.crop_size} n_pos={self.n_pos} n_neg={self.n_neg} "
            f"crop_scale={self.crop_scale} max_angle={self.max_angle:.3f} "
            f"offset_scale={self.offset_scale} mesh_zbuffer={self.use_mesh_zbuffer} "
            f"zbuffer_backend={self.zbuffer_backend} min_render_iou={self.min_render_iou:.3f} "
            f"shaft_norm_x_min={self.shaft_norm_x_min:g} "
            f"part_sample_ratios={self.part_sample_ratios.tolist()} "
            f"supervision_part_ids={self.supervision_part_ids} "
            f"min_supervision_pixels={self.min_supervision_pixels} "
            f"negative_visible_only={self.negative_visible_only} "
            f"fallback_to_other_sample={self.fallback_to_other_sample} "
            f"augmentation_profile={self.augmentation_profile} "
            f"crop_min_mask_retention={self.crop_min_mask_retention:.3f} "
            f"crop_offset_multiplier={self.crop_offset_multiplier:.3f} "
            f"base={self.base_dataset})"
        )

    def set_epoch(self, epoch):
        if hasattr(self.base_dataset, "set_epoch"):
            self.base_dataset.set_epoch(epoch)

    def _get_mesh_renderer(self):
        if self._mesh_renderer is None:
            if self.zbuffer_backend != "opengl":
                raise ValueError(
                    "SurfEmb visible-surface rendering is OpenGL-only; "
                    f"got zbuffer_backend={self.zbuffer_backend!r}."
                )
            from instrument_opengl_renderer import InstrumentOpenGLDepthRenderer

            device_idx = _current_egl_device_id()
            cache_key = (os.getpid(), device_idx, self.crop_size, self.crop_size)
            if cache_key not in _OPENGL_RENDERER_CACHE:
                _OPENGL_RENDERER_CACHE[cache_key] = InstrumentOpenGLDepthRenderer(
                    self.crop_size,
                    self.crop_size,
                    device_idx=device_idx,
                )
            self._mesh_renderer = _OPENGL_RENDERER_CACHE[cache_key]
        return self._mesh_renderer

    @staticmethod
    def _pose_from_target(target):
        action = _as_numpy(target["action"]).astype(np.float32).reshape(3)
        return {
            "rot": _as_numpy(target["wrist_quat"]).astype(np.float32),
            "trans": _as_numpy(target["wrist_trans"]).astype(np.float32),
            "alpha": float(action[0]),
            "theta_l": float(action[1]),
            "theta_r": float(action[2]),
        }

    def _make_crop_matrix(self, mask, source_visibility_fraction=1.0):
        max_angle = self.max_angle if self.training else 0.0
        offset_scale = self.offset_scale if self.training else 0.0
        return _surf_aug.random_rotated_mask_crop_matrix(
            mask,
            self.crop_size,
            crop_scale=self.crop_scale,
            max_angle=max_angle,
            offset_scale=offset_scale,
            ensure_full_mask=True,
            min_mask_retention=self.crop_min_mask_retention,
            source_visibility_fraction=source_visibility_fraction,
            offset_multiplier=self.crop_offset_multiplier,
        )

    def _warp_rgb(self, rgb, M, inst_crop=None, part_crop=None):
        rgb = np.asarray(rgb, dtype=np.uint8)
        if self.training:
            aug_rgb = _surf_aug.surfemb_precrop_photometric(rgb, profile=self.augmentation_profile)
            crop_rgb = _surf_aug.warp_rgb(aug_rgb, M, self.crop_size)
            return _surf_aug.surfemb_postcrop_photometric(
                crop_rgb,
                inst_mask=inst_crop,
                part_mask=part_crop,
                profile=self.augmentation_profile,
            )
        return cv2.warpAffine(
            rgb,
            M,
            (self.crop_size, self.crop_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

    def _sample_visible_surface_pairs(
        self,
        pose,
        K_crop,
        inst_crop,
        enforce_render_iou=True,
        min_supervision_pixels=None,
    ):
        h, w = inst_crop.shape[:2]
        renderer = self._get_mesh_renderer()
        coord_img, effective_part_img, depth, raster_valid = renderer.render_canonical_coordinates(
            pose,
            K_crop,
            (h, w),
        )
        mesh_support = raster_valid & (effective_part_img > 0) & (depth > self.min_depth)
        inst_bool = inst_crop > 0
        sample_eligible = surfemb_surface_sampling_mask(
            coord_img.reshape(-1, 3),
            effective_part_img.reshape(-1),
            self.shaft_norm_x_min,
        ).reshape(h, w)
        requested_part_ids = np.concatenate(
            [
                np.asarray(part_ids, dtype=np.int64)
                for ratio, (_, part_ids) in zip(self.part_sample_ratios, PART_SAMPLE_GROUPS)
                if ratio > 0.0
            ]
        )
        requested_support = mesh_support & np.isin(effective_part_img, requested_part_ids) & sample_eligible
        valid = requested_support & inst_bool
        iou_support = mesh_support if self.supervision_part_ids is None else requested_support
        union = np.count_nonzero(iou_support | inst_bool)
        render_iou = float(np.count_nonzero(iou_support & inst_bool) / max(1, union))
        if enforce_render_iou and self.training and render_iou < self.min_render_iou:
            raise _LowRenderIouSample(
                f"render-mask IoU {render_iou:.4f} < {self.min_render_iou:.4f}"
            )

        idx = np.flatnonzero(valid.reshape(-1))
        required_pixels = self.min_supervision_pixels if min_supervision_pixels is None else int(min_supervision_pixels)
        if len(idx) < required_pixels:
            raise _NoProjectedSurfaceSamples(
                f"Only {len(idx)} triangle-rasterized visible supervision pixels remain; "
                f"need at least {required_pixels}."
            )
        effective_flat = effective_part_img.reshape(-1)
        pick = _sample_part_balanced_indices(
            idx,
            effective_flat,
            self.n_pos,
            self.part_sample_ratios,
            strict=self.training,
        )
        v, u = np.unravel_index(pick, (h, w))
        yx = np.stack([v, u], axis=1).astype(np.int64)
        coords = coord_img[v, u].astype(np.float32)
        picked_part_ids = effective_part_img[v, u].astype(np.int64)
        expected_counts = _allocate_part_sample_counts(self.n_pos, self.part_sample_ratios)
        actual_counts = np.asarray(
            [np.isin(picked_part_ids, np.asarray(part_ids)).sum() for _, part_ids in PART_SAMPLE_GROUPS],
            dtype=np.int64,
        )
        part_ratio_exact = bool(np.array_equal(actual_counts, expected_counts))
        negative_candidates = idx[~np.isin(idx, np.unique(pick))]
        if self.negative_visible_only and len(negative_candidates) == 0:
            raise _NoProjectedSurfaceSamples(
                "No visible surface pixels remain for negative sampling after selecting positives."
            )
        nv, nu = np.unravel_index(negative_candidates, (h, w))
        negative_coords = coord_img[nv, nu].astype(np.float32)
        negative_part_ids = effective_part_img[nv, nu].astype(np.int64)
        return (
            yx,
            coords,
            picked_part_ids,
            render_iou,
            part_ratio_exact,
            negative_coords,
            negative_part_ids,
        )

    def _supervision_masks(self, target):
        inst_orig = (_as_numpy(target["inst_mask_orig"]) > 0).astype(np.uint8)
        part_orig = _as_numpy(target["part_mask_orig"]).astype(np.uint8)
        if self.supervision_part_ids is None:
            return inst_orig, part_orig
        supervision = inst_orig.astype(bool) & np.isin(
            part_orig,
            np.asarray(self.supervision_part_ids, dtype=part_orig.dtype),
        )
        part_supervision = np.where(supervision, part_orig, 0).astype(np.uint8)
        return supervision.astype(np.uint8), part_supervision

    def __getitem__(self, idx, _retry_depth=0):
        _, target = self.base_dataset[idx]
        rgb = _as_numpy(target["orig_rgb"]).astype(np.uint8)
        inst_orig, part_orig = self._supervision_masks(target)
        supervision_pixels = int(np.count_nonzero(inst_orig))
        if supervision_pixels < self.min_supervision_pixels:
            if self.fallback_to_other_sample and _retry_depth < 32 and len(self) > 1:
                if self.training:
                    offset = np.random.randint(1, len(self))
                else:
                    offset = 1
                return self.__getitem__((int(idx) + int(offset)) % len(self), _retry_depth=_retry_depth + 1)
            raise RuntimeError(
                f"Supervision mask has {supervision_pixels} pixels, below "
                f"min_supervision_pixels={self.min_supervision_pixels} for idx={idx}."
            )
        K_orig = _as_numpy(target["K_orig"]).astype(np.float32)
        pose = self._pose_from_target(target)
        source_visibility_fraction = float(
            _as_numpy(target.get("wrist_visibility_fraction", np.asarray(1.0, dtype=np.float32)))
        )
        low_source_visibility = source_visibility_fraction < self.crop_min_mask_retention

        attempts = 16 if self.training else 1
        last_error = None
        surfemb_valid = True
        for _ in range(attempts):
            M = self._make_crop_matrix(
                inst_orig,
                source_visibility_fraction=source_visibility_fraction,
            )
            M3 = _surf_aug.matrix3_from_affine(M)
            K_crop = M3 @ K_orig
            inst_crop = _surf_aug.warp_map(
                inst_orig,
                M,
                self.crop_size,
                interpolation=cv2.INTER_NEAREST,
                value=0,
            ).astype(np.float32)
            part_crop = _surf_aug.warp_map(
                part_orig,
                M,
                self.crop_size,
                interpolation=cv2.INTER_NEAREST,
                value=0,
            ).astype(np.int64)
            try:
                (
                    pos_yx,
                    pos_coords,
                    pos_part_ids,
                    render_iou,
                    part_ratio_exact,
                    visible_neg_coords,
                    visible_neg_part_ids,
                ) = self._sample_visible_surface_pairs(
                    pose,
                    K_crop,
                    inst_crop,
                    enforce_render_iou=not low_source_visibility,
                    min_supervision_pixels=1 if low_source_visibility else self.min_supervision_pixels,
                )
                break
            except _LowRenderIouSample as exc:
                last_error = exc
                if self.fallback_to_other_sample and self.training and _retry_depth < 32 and len(self) > 1:
                    offset = np.random.randint(1, len(self))
                    return self.__getitem__((int(idx) + int(offset)) % len(self), _retry_depth=_retry_depth + 1)
                if self.training:
                    # Strict mode may retry the crop augmentation for this same
                    # frame, but never substitutes a different training sample.
                    continue
            except _MissingPartSampleGroup as exc:
                last_error = exc
                if self.fallback_to_other_sample and self.training and _retry_depth < 32 and len(self) > 1:
                    offset = np.random.randint(1, len(self))
                    return self.__getitem__((int(idx) + int(offset)) % len(self), _retry_depth=_retry_depth + 1)
                raise RuntimeError(f"Could not satisfy SurfEmb part sampling ratios for idx={idx}.") from exc
            except _NoProjectedSurfaceSamples as exc:
                last_error = exc
                if (
                    self.fallback_to_other_sample
                    and self.supervision_part_ids is not None
                    and _retry_depth < 32
                    and len(self) > 1
                ):
                    if self.training:
                        offset = np.random.randint(1, len(self))
                    else:
                        offset = 1
                    return self.__getitem__(
                        (int(idx) + int(offset)) % len(self),
                        _retry_depth=_retry_depth + 1,
                    )
        else:
            if self.fallback_to_other_sample and self.training and _retry_depth < 3 and len(self) > 1:
                offset = np.random.randint(1, len(self))
                return self.__getitem__((int(idx) + int(offset)) % len(self), _retry_depth=_retry_depth + 1)
            raise RuntimeError(
                f"Could not sample SurfEmb positives after {attempts} crop augmentations for idx={idx}."
            ) from last_error

        crop_rgb = self._warp_rgb(rgb, M, inst_crop=inst_crop, part_crop=part_crop)
        keypoints_crop = _surf_aug.transform_points(_as_numpy(target["keypoints_orig"]), M)
        keypoints_valid = _as_numpy(target["keypoints_valid_orig"]).astype(bool)
        inside = (
            (keypoints_crop[:, 0] >= 0.0)
            & (keypoints_crop[:, 0] < float(self.crop_size))
            & (keypoints_crop[:, 1] >= 0.0)
            & (keypoints_crop[:, 1] < float(self.crop_size))
        )
        keypoints_valid = keypoints_valid & inside
        heatmaps = _create_belief_maps(
            (self.crop_size, self.crop_size),
            keypoints_crop,
            keypoints_valid,
            sigma=self.heatmap_sigma,
        )

        if self.negative_visible_only:
            visible_indices = np.arange(len(visible_neg_coords), dtype=np.int64)
            surf_idx = _sample_part_balanced_indices(
                visible_indices,
                visible_neg_part_ids,
                self.n_neg,
                self.part_sample_ratios,
                strict=True,
            )
            negative_coords = visible_neg_coords[surf_idx].astype(np.float32)
            negative_part_ids = visible_neg_part_ids[surf_idx].astype(np.int64)
        else:
            surf_idx = _sample_part_balanced_indices(
                self.surface_sample_indices,
                self.surface_part_ids,
                self.n_neg,
                self.part_sample_ratios,
                strict=True,
            )
            negative_coords = self.surface_points[surf_idx].astype(np.float32)
            negative_part_ids = self.surface_part_ids[surf_idx].astype(np.int64)
        target_out = dict(target)
        target_out.update(
            {
                "heatmaps": torch.from_numpy(heatmaps),
                "keypoints_crop": torch.from_numpy(keypoints_crop.astype(np.float32)),
                "keypoints_valid": torch.from_numpy(keypoints_valid.astype(np.bool_)),
                "K": torch.from_numpy(K_crop.astype(np.float32)),
                "crop_rgb": torch.from_numpy(crop_rgb.astype(np.uint8)),
                "inst_mask": torch.from_numpy(inst_crop.astype(np.float32)),
                "part_mask": torch.from_numpy(part_crop.astype(np.int64)),
                "has_cse": torch.tensor(bool(surfemb_valid), dtype=torch.bool),
                "surfemb_M_crop": torch.from_numpy(M.astype(np.float32)),
                "surfemb_obj_idx": torch.tensor(0, dtype=torch.long),
                "surfemb_mask_samples": torch.from_numpy(pos_yx),
                "surfemb_coords_pos": torch.from_numpy(pos_coords),
                "surfemb_positive_part_ids": torch.from_numpy(pos_part_ids),
                "surfemb_surface_samples": torch.from_numpy(negative_coords),
                "surfemb_surface_part_ids": torch.from_numpy(negative_part_ids),
                "surfemb_render_iou": torch.tensor(float(render_iou), dtype=torch.float32),
                "surfemb_part_ratio_exact": torch.tensor(part_ratio_exact, dtype=torch.bool),
                "surfemb_supervision_pixels_orig": torch.tensor(supervision_pixels, dtype=torch.int64),
                "surfemb_negative_visible_only": torch.tensor(self.negative_visible_only, dtype=torch.bool),
                "surfemb_source_visibility_fraction": torch.tensor(
                    source_visibility_fraction, dtype=torch.float32
                ),
                "surfemb_low_visibility_mode": torch.tensor(low_source_visibility, dtype=torch.bool),
            }
        )

        kp3d = _as_numpy(target["keypoints_3d_cam"]).astype(np.float32)
        uv = project_points_np(kp3d, K_crop).astype(np.float32)
        valid = keypoints_valid & np.isfinite(uv).all(axis=-1) & (kp3d[:, 2] > 1e-4)
        resid = float(np.max(np.linalg.norm(uv[valid] - keypoints_crop[valid], axis=-1))) if valid.any() else float("nan")
        target_out["surfemb_kpt_proj_resid_px"] = torch.tensor(resid, dtype=torch.float32)

        image_tensor = _surf_aug.imagenet_tensor(crop_rgb)
        return image_tensor, target_out


def collate_fn_surfemb_keypoint_crop(batch):
    images, targets = zip(*batch)
    x = torch.stack(images, dim=0)
    tensor_keys = [
        "action",
        "wrist_quat",
        "wrist_trans",
        "heatmaps",
        "keypoints_crop",
        "keypoints_orig",
        "keypoints_3d_cam",
        "keypoints_valid",
        "keypoints_valid_orig",
        "keypoint_hit_labels",
        "K",
        "K_orig",
        "bbox_min",
        "bbox_max",
        "scale",
        "pad",
        "pose_sym_flipped",
        "instance_id",
        "orig_size",
        "inst_mask",
        "part_mask",
        "has_cse",
        "surfemb_M_crop",
        "surfemb_obj_idx",
        "surfemb_mask_samples",
        "surfemb_coords_pos",
        "surfemb_positive_part_ids",
        "surfemb_surface_samples",
        "surfemb_surface_part_ids",
        "surfemb_render_iou",
        "surfemb_part_ratio_exact",
        "surfemb_supervision_pixels_orig",
        "surfemb_negative_visible_only",
        "surfemb_source_visibility_fraction",
        "surfemb_low_visibility_mode",
        "surfemb_kpt_proj_resid_px",
    ]
    y = {}
    for key in tensor_keys:
        y[key] = torch.stack([t[key] for t in targets], dim=0)
    y["video_name"] = [t["video_name"] for t in targets]
    y["frame_id"] = [t["frame_id"] for t in targets]
    y["img_path"] = [t["img_path"] for t in targets]
    return x, y


def _pose_from_debug_target(target):
    action = _as_numpy(target["action"]).astype(np.float32).reshape(3)
    return {
        "rot": _as_numpy(target["wrist_quat"]).astype(np.float32),
        "trans": _as_numpy(target["wrist_trans"]).astype(np.float32),
        "alpha": float(action[0]),
        "theta_l": float(action[1]),
        "theta_r": float(action[2]),
    }


def save_surfemb_keypoint_debug_panel(path, target, mesh_renderer=None):
    from PIL import Image

    rgb = _as_numpy(target["crop_rgb"]).astype(np.uint8)
    inst = (_as_numpy(target["inst_mask"]) > 0).astype(np.uint8)
    part = _as_numpy(target["part_mask"]).astype(np.uint8)
    heat = _as_numpy(target["heatmaps"]).max(axis=0)
    colors = np.array(
        [
            [0, 0, 0],
            [230, 80, 80],
            [80, 220, 120],
            [80, 150, 240],
            [230, 210, 80],
        ],
        dtype=np.uint8,
    )
    part_rgb = colors[np.clip(part, 0, len(colors) - 1)]
    overlay = rgb.copy()
    overlay[inst > 0] = (0.55 * overlay[inst > 0] + 0.45 * part_rgb[inst > 0]).astype(np.uint8)
    kp = _as_numpy(target["keypoints_crop"]).astype(np.float32)
    valid = _as_numpy(target["keypoints_valid"]).astype(bool)
    for u, v in kp[valid]:
        cv2.circle(overlay, (int(round(float(u))), int(round(float(v)))), 3, (255, 255, 0), -1)
    mesh_overlay = rgb.copy()
    if mesh_renderer is not None:
        try:
            pose = _pose_from_debug_target(target)
            if (
                hasattr(mesh_renderer, "render_pose_overlay_warped_crop")
                and "K_orig" in target
                and "surfemb_M_crop" in target
                and "orig_rgb" in target
            ):
                mesh_overlay = mesh_renderer.render_pose_overlay_warped_crop(
                    rgb,
                    pose,
                    _as_numpy(target["K_orig"]).astype(np.float32),
                    _as_numpy(target["surfemb_M_crop"]).astype(np.float32),
                    _as_numpy(target["orig_rgb"]).shape[:2],
                    alpha=0.70,
                )
            else:
                mesh_overlay = mesh_renderer.render_pose_overlay(
                    rgb,
                    pose,
                    _as_numpy(target["K"]).astype(np.float32),
                    alpha=0.70,
                )
        except Exception as exc:
            mesh_overlay = rgb.copy()
            cv2.putText(
                mesh_overlay,
                f"mesh render failed: {type(exc).__name__}",
                (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 80, 80),
                1,
                cv2.LINE_AA,
            )
    for u, v in kp[valid]:
        cv2.circle(mesh_overlay, (int(round(float(u))), int(round(float(v)))), 3, (255, 255, 0), -1)
    coord_rgb = np.zeros_like(rgb)
    yx = _as_numpy(target["surfemb_mask_samples"]).astype(np.int64)
    coords = _as_numpy(target["surfemb_coords_pos"]).astype(np.float32)
    pos_rgb = ((coords + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)
    for (y, x), color in zip(yx, pos_rgb):
        if 0 <= x < coord_rgb.shape[1] and 0 <= y < coord_rgb.shape[0]:
            coord_rgb[y, x] = color
    heat_rgb = cv2.applyColorMap((heat * 255.0).clip(0, 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
    heat_rgb = cv2.cvtColor(heat_rgb, cv2.COLOR_BGR2RGB)
    panel = np.concatenate([rgb, mesh_overlay, overlay, part_rgb, coord_rgb, heat_rgb], axis=1)
    Image.fromarray(panel).save(path)
