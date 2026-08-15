import sys
import types
import os
from pathlib import Path

import numpy as np
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import pyrender
import torch
import trimesh
import cv2

from instrument_geometry import _bias2part_matrices, _bias2world_matrix, fk_matrices_np


ROBOPEPP_ROOT = Path(__file__).resolve().parent
MULTIHMR_ROOT = ROBOPEPP_ROOT.parents[1]
GMS_ROOT = MULTIHMR_ROOT / "submodules" / "gaussian-mesh-splatting"
CAD_ROOT = GMS_ROOT / "instrument_mesh"
ROBOPEPP_DEPS = ROBOPEPP_ROOT / ".deps" / "python"
if ROBOPEPP_DEPS.is_dir() and str(ROBOPEPP_DEPS) not in sys.path:
    sys.path.insert(0, str(ROBOPEPP_DEPS))

PART_MESHES = {
    "shaft": "transformed_shaft.obj",
    "wrist": "transformed_wrist.obj",
    "l_gripper": "transformed_gripper_left.obj",
    "r_gripper": "transformed_gripper_right.obj",
}

PART_ORDER = ("shaft", "wrist", "l_gripper", "r_gripper")


class _DummyGaussianModel:
    def apply_transform_params(self, transformation, gs_grad=False):
        return {}


class _DummyMeshPointCloud:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


def _install_unused_gms_dependency_stubs():
    games = sys.modules.setdefault("games", types.ModuleType("games"))
    games.__path__ = []
    mesh_splatting = sys.modules.setdefault("games.mesh_splatting", types.ModuleType("games.mesh_splatting"))
    mesh_splatting.__path__ = []
    games.mesh_splatting = mesh_splatting

    utils_pkg = sys.modules.setdefault("games.mesh_splatting.utils", types.ModuleType("games.mesh_splatting.utils"))
    utils_pkg.__path__ = []
    mesh_splatting.utils = utils_pkg
    graphics_utils = types.ModuleType("games.mesh_splatting.utils.graphics_utils")
    graphics_utils.MeshPointCloud = _DummyMeshPointCloud
    sys.modules["games.mesh_splatting.utils.graphics_utils"] = graphics_utils
    utils_pkg.graphics_utils = graphics_utils

    scene_pkg = sys.modules.setdefault("games.mesh_splatting.scene", types.ModuleType("games.mesh_splatting.scene"))
    scene_pkg.__path__ = []
    mesh_splatting.scene = scene_pkg
    gaussian_model = types.ModuleType("games.mesh_splatting.scene.gaussian_mesh_model_3dgs")
    gaussian_model.GaussianMeshModel = _DummyGaussianModel
    sys.modules["games.mesh_splatting.scene.gaussian_mesh_model_3dgs"] = gaussian_model
    scene_pkg.gaussian_mesh_model_3dgs = gaussian_model


def _ensure_gms_utils_importable():
    gms_root = str(GMS_ROOT)
    if gms_root in sys.path:
        sys.path.remove(gms_root)
    sys.path.insert(0, gms_root)

    loaded_utils = sys.modules.get("utils")
    utils_file = Path(getattr(loaded_utils, "__file__", "")).resolve() if loaded_utils is not None else None
    if loaded_utils is not None and (utils_file is None or not str(utils_file).startswith(gms_root)):
        for name in list(sys.modules):
            if name == "utils" or name.startswith("utils."):
                del sys.modules[name]
    _install_unused_gms_dependency_stubs()


def _load_mesh(name):
    path = CAD_ROOT / name
    if not path.is_file():
        raise FileNotFoundError(f"Instrument mesh not found: {path}")
    mesh = trimesh.load_mesh(str(path), force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected trimesh.Trimesh at {path}, got {type(mesh)}")
    return mesh


class GMSInstrumentTrimeshRenderer:
    def __init__(self, device):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.bias2part = _bias2part_matrices()
        self.bias2world = _bias2world_matrix()
        self.part_meshes = {}
        for part_name in PART_ORDER:
            mesh = _load_mesh(PART_MESHES[part_name])
            mesh = mesh.copy()
            mesh.apply_transform(self.bias2part[part_name])
            self.part_meshes[part_name] = mesh
        wrist_vertices = np.asarray(self.part_meshes["wrist"].vertices, dtype=np.float64)
        self._canon_scale = float(2.0 * (wrist_vertices.max(axis=0) - wrist_vertices.min(axis=0)).max())
        self._coord_renderer = None
        self.new_mesh_dict = {}

    def _transformed_meshes(self, pose):
        transforms = fk_matrices_np(
            pose["rot"],
            pose["trans"],
            pose["alpha"],
            pose["theta_l"],
            pose["theta_r"],
        )
        out = {}
        for part_name in PART_ORDER:
            mesh = self.part_meshes[part_name].copy()
            mesh.apply_transform(transforms[part_name])
            out[part_name] = mesh
        self.new_mesh_dict = out
        return out

    def render_pose(self, pose, K, image_shape):
        meshes = self._transformed_meshes(pose)
        combined_mesh = trimesh.util.concatenate([meshes[part_name] for part_name in PART_ORDER])
        K = np.asarray(K, dtype=np.float32)

        extrinsic = np.eye(4)
        extrinsic[:, 1:3] *= -1
        scene = pyrender.Scene()
        scene.add(pyrender.Mesh.from_trimesh(combined_mesh))
        camera = pyrender.IntrinsicsCamera(
            fx=K[0, 0],
            fy=K[1, 1],
            cx=K[0, 2],
            cy=K[1, 2],
            znear=0.001,
            zfar=0.6,
        )
        scene.add(camera, pose=extrinsic)
        light = pyrender.PointLight(intensity=1)
        light_pose = extrinsic.copy()
        light_pose[:, 1:3] *= -1
        light_pose[2, 3] += 0.1
        light_pose[0, 3] += 0.1
        scene.add(light, pose=light_pose)

        height, width = int(image_shape[0]), int(image_shape[1])
        renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
        try:
            color, depth = renderer.render(scene)
        finally:
            renderer.delete()
        return color.astype(np.uint8), depth, combined_mesh

    def render_canonical_coords(self, pose, K, image_shape, device_idx=0):
        _ensure_gms_utils_importable()
        from utils.instrument import PartCoordRenderer

        h, w = int(image_shape[0]), int(image_shape[1])
        if self._coord_renderer is None or self._coord_renderer.h != h or self._coord_renderer.w != w:
            self._coord_renderer = PartCoordRenderer(h, w, int(device_idx))

        meshes = self._transformed_meshes(pose)
        world_v_list, canon_v_list, faces_list = [], [], []
        for part_name in PART_ORDER:
            world_mesh = meshes[part_name]
            canon_mesh = self.part_meshes[part_name]
            part2world = self.bias2world @ np.linalg.inv(self.bias2part[part_name])
            canon_verts = np.asarray(canon_mesh.vertices, dtype=np.float32)
            canon_world = canon_verts @ part2world[:3, :3].T + part2world[:3, 3]
            world_v_list.append(np.asarray(world_mesh.vertices, dtype=np.float32))
            canon_v_list.append(canon_world.astype(np.float32))
            faces_list.append(np.asarray(world_mesh.faces, dtype=np.int32))

        return self._coord_renderer.render(
            np.asarray(K, dtype=np.float32),
            world_v_list,
            canon_v_list,
            faces_list,
            self._canon_scale,
        ).astype(np.float32)

    def render_pose_overlay(self, rgb, pose, K, alpha=0.85):
        color, depth, _ = self.render_pose(pose, K, rgb.shape[:2])
        support = np.asarray(depth) > 0
        out = rgb.copy()
        if support.any():
            blended = (
                rgb[support].astype(np.float32) * (1.0 - float(alpha))
                + color[support].astype(np.float32) * float(alpha)
            )
            out[support] = np.clip(blended, 0, 255).astype(np.uint8)
        return out

    @staticmethod
    def _project(points_cam, K):
        z = points_cam[:, 2:3]
        uvw = points_cam @ np.asarray(K, dtype=np.float64).reshape(3, 3).T
        return uvw[:, :2] / z

    @staticmethod
    def _part_label(part_name):
        if part_name == "shaft":
            return 3
        if part_name == "wrist":
            return 2
        return 1

    def render_pose_mask(self, pose, K, image_shape, min_depth=1e-4, draw_margin=20.0):
        meshes = self._transformed_meshes(pose)
        h, w = int(image_shape[0]), int(image_shape[1])
        K = np.asarray(K, dtype=np.float32)
        extrinsic = np.eye(4)
        extrinsic[:, 1:3] *= -1
        scene = pyrender.Scene()
        seg_node_map = {}
        for part_name in PART_ORDER:
            node = scene.add(pyrender.Mesh.from_trimesh(meshes[part_name]))
            label = self._part_label(part_name)
            seg_node_map[node] = np.array([label, 0, 0], dtype=np.uint8)
        camera = pyrender.IntrinsicsCamera(
            fx=K[0, 0],
            fy=K[1, 1],
            cx=K[0, 2],
            cy=K[1, 2],
            znear=float(min_depth),
            zfar=0.6,
        )
        scene.add(camera, pose=extrinsic)
        renderer = pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h)
        try:
            color, depth = renderer.render(scene, flags=pyrender.RenderFlags.SEG, seg_node_map=seg_node_map)
        finally:
            renderer.delete()
        mask = color[..., 0].astype(np.uint8)
        mask[np.asarray(depth) <= float(min_depth)] = 0
        return mask
