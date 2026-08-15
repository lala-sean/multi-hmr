import os
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import trimesh

from instrument_geometry import (
    GRIPPER_JOINT_OFFSET_M,
    SHAFT_WRIST_OFFSET_M,
    SURFEMB_GRIPPER_STATIC_THRESHOLD_M,
    fk_matrices_np,
)


ROBOPEPP_ROOT = Path(__file__).resolve().parent
MULTIHMR_ROOT = ROBOPEPP_ROOT.parents[1]
GMS_ROOT = MULTIHMR_ROOT / "submodules" / "gaussian-mesh-splatting"
CAD_ROOT = GMS_ROOT / "instrument_mesh"

PART_MESHES = {
    "shaft": "transformed_shaft.obj",
    "wrist": "transformed_wrist.obj",
    "l_gripper": "transformed_gripper_left.obj",
    "r_gripper": "transformed_gripper_right.obj",
}


def _orthographic_matrix(left, right, bottom, top, near, far):
    return np.array(
        (
            (2.0 / (right - left), 0.0, 0.0, -(right + left) / (right - left)),
            (0.0, 2.0 / (top - bottom), 0.0, -(top + bottom) / (top - bottom)),
            (0.0, 0.0, -2.0 / (far - near), -(far + near) / (far - near)),
            (0.0, 0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )


def _projection_matrix(K, w, h, near=1e-4, far=10.0):
    # OpenGL is -Z forward, while the repo pose convention uses OpenCV camera
    # coordinates (+Z forward, +Y down).
    view = np.eye(4, dtype=np.float64)
    view[1:3] *= -1.0

    persp = np.zeros((4, 4), dtype=np.float64)
    persp[:2, :3] = np.asarray(K, dtype=np.float64).reshape(3, 3)[:2, :3]
    persp[2, 2:] = near + far, near * far
    persp[3, 2] = -1.0
    persp[:2, 1:3] *= -1.0

    orth = _orthographic_matrix(-0.5, float(w) - 0.5, -0.5, float(h) - 0.5, near, far)
    return orth @ persp @ view


def _bias_matrices():
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


def _bias_to_world_matrix():
    bias2world = np.eye(4, dtype=np.float64)
    bias2world[:3, :3] = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64).T
    return bias2world


def _part_to_canonical_matrix(part_name):
    return _bias_to_world_matrix() @ np.linalg.inv(_bias_matrices()[part_name])


def _transform_points(points, transform):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return points @ transform[:3, :3].T + transform[:3, 3]


def _triangle_vertex_normals(vertices):
    triangles = np.asarray(vertices, dtype=np.float64).reshape(-1, 3, 3)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True).clip(1e-12)
    return np.repeat(normals[:, None, :], 3, axis=1).reshape(-1, 3).astype(np.float32)


def _load_part_mesh(part_name):
    path = CAD_ROOT / PART_MESHES[part_name]
    mesh = trimesh.load_mesh(path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected Trimesh at {path}, got {type(mesh)}")
    mesh = mesh.copy()
    mesh.apply_transform(_bias_matrices()[part_name])
    return mesh


def _clip_polygon_x(vertices, threshold, keep_lower):
    vertices = [np.asarray(vertex, dtype=np.float64) for vertex in vertices]
    if not vertices:
        return np.empty((0, 3), dtype=np.float64)

    def inside(vertex):
        if keep_lower:
            return vertex[0] <= threshold
        return vertex[0] >= threshold

    clipped = []
    previous = vertices[-1]
    previous_inside = inside(previous)
    for current in vertices:
        current_inside = inside(current)
        if current_inside != previous_inside:
            delta = current - previous
            if abs(float(delta[0])) > 1e-12:
                ratio = (float(threshold) - float(previous[0])) / float(delta[0])
                clipped.append(previous + ratio * delta)
        if current_inside:
            clipped.append(current)
        previous = current
        previous_inside = current_inside
    if len(clipped) < 3:
        return np.empty((0, 3), dtype=np.float64)
    return np.asarray(clipped, dtype=np.float64)


def _triangulate_polygon(vertices):
    vertices = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    if len(vertices) < 3:
        return []
    triangles = []
    for index in range(1, len(vertices) - 1):
        triangle = np.stack((vertices[0], vertices[index], vertices[index + 1]), axis=0)
        twice_area = np.linalg.norm(np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0]))
        if twice_area > 1e-14:
            triangles.append(triangle)
    return triangles


def _split_gripper_mesh_vertices(mesh, threshold=SURFEMB_GRIPPER_STATIC_THRESHOLD_M):
    static_triangles = []
    moving_triangles = []
    for triangle in np.asarray(mesh.vertices[mesh.faces], dtype=np.float64):
        static_triangles.extend(
            _triangulate_polygon(_clip_polygon_x(triangle, float(threshold), keep_lower=True))
        )
        moving_triangles.extend(
            _triangulate_polygon(_clip_polygon_x(triangle, float(threshold), keep_lower=False))
        )

    def flatten(triangles):
        if not triangles:
            return np.empty((0, 3), dtype=np.float32)
        return np.asarray(triangles, dtype=np.float32).reshape(-1, 3)

    return flatten(static_triangles), flatten(moving_triangles)


def _wrist_to_gripper_origin():
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = np.array([GRIPPER_JOINT_OFFSET_M, 0.0, 0.0], dtype=np.float64)
    return transform


def _default_device_idx():
    if "EGL_DEVICE_ID" in os.environ:
        try:
            return int(os.environ["EGL_DEVICE_ID"])
        except ValueError:
            return 0
    visible = [v.strip() for v in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if v.strip()]
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if visible and local_rank < len(visible):
        try:
            return int(visible[local_rank])
        except ValueError:
            return local_rank
    return local_rank


class InstrumentOpenGLDepthRenderer:
    """Moderngl triangle renderer for instrument depth and SurfEmb coordinates.

    This is intentionally CPU/OpenGL based and does not call torch.cuda, so it is
    safe to lazily instantiate inside PyTorch DataLoader worker processes.
    """

    def __init__(self, w=224, h=224, device_idx=None, near=1e-4, far=10.0):
        import moderngl

        self.moderngl = moderngl
        self.w = int(w)
        self.h = int(h)
        self.near = float(near)
        self.far = float(far)
        self.device_idx = _default_device_idx() if device_idx is None else int(device_idx)
        self.ctx = moderngl.create_context(standalone=True, backend="egl", device_index=self.device_idx)
        self.ctx.disable(moderngl.CULL_FACE)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.fbo = None
        self._framebuffers = OrderedDict()
        self._framebuffer_cache_size = 4
        self._ensure_framebuffer(self.w, self.h)
        self.prog = self.ctx.program(
            vertex_shader="""
                #version 330
                uniform mat4 model;
                uniform vec3 k0;
                uniform vec3 k1;
                uniform vec2 image_size;
                uniform float near_z;
                uniform float far_z;
                uniform vec3 part_color;
                in vec3 in_vert;
                void main() {
                    vec4 cam = model * vec4(in_vert, 1.0);
                    float z = cam.z;
                    float u = dot(k0, cam.xyz) / z;
                    float v = dot(k1, cam.xyz) / z;
                    float x_ndc = 2.0 * (u + 0.5) / image_size.x - 1.0;
                    float y_ndc = 2.0 * (v + 0.5) / image_size.y - 1.0;
                    float a = (far_z + near_z) / (far_z - near_z);
                    float b = -2.0 * far_z * near_z / (far_z - near_z);
                    gl_Position = vec4(x_ndc * z, y_ndc * z, a * z + b, z);
                }
                """,
            fragment_shader="""
                #version 330
                uniform vec3 part_color;
                out vec4 fragColor;
                void main() {
                    fragColor = vec4(part_color, 1.0);
                }
                """,
        )
        self.coord_prog = self.ctx.program(
            vertex_shader="""
                #version 330
                uniform mat4 model;
                uniform vec3 k0;
                uniform vec3 k1;
                uniform vec2 image_size;
                uniform float near_z;
                uniform float far_z;
                in vec3 in_vert;
                in vec3 in_coord;
                in vec3 in_normal;
                out vec3 canonical_coord;
                out vec3 camera_point;
                out vec3 camera_normal;
                void main() {
                    vec4 cam = model * vec4(in_vert, 1.0);
                    float z = cam.z;
                    float u = dot(k0, cam.xyz) / z;
                    float v = dot(k1, cam.xyz) / z;
                    float x_ndc = 2.0 * (u + 0.5) / image_size.x - 1.0;
                    float y_ndc = 2.0 * (v + 0.5) / image_size.y - 1.0;
                    float a = (far_z + near_z) / (far_z - near_z);
                    float b = -2.0 * far_z * near_z / (far_z - near_z);
                    gl_Position = vec4(x_ndc * z, y_ndc * z, a * z + b, z);
                    canonical_coord = in_coord;
                    camera_point = cam.xyz;
                    camera_normal = mat3(model) * in_normal;
                }
                """,
            fragment_shader="""
                #version 330
                uniform float effective_part_id;
                in vec3 canonical_coord;
                in vec3 camera_point;
                in vec3 camera_normal;
                out vec4 fragColor;
                void main() {
                    if (dot(camera_normal, camera_point) >= 0.0) {
                        discard;
                    }
                    fragColor = vec4(canonical_coord, effective_part_id);
                }
                """,
        )
        self.mesh_vis_prog = self.ctx.program(
            vertex_shader="""
                #version 330
                uniform mat4 model;
                uniform vec3 k0;
                uniform vec3 k1;
                uniform vec2 image_size;
                uniform float near_z;
                uniform float far_z;
                in vec3 in_vert;
                in vec3 in_normal;
                in vec3 in_barycentric;
                out vec3 camera_point;
                out vec3 camera_normal;
                out vec3 barycentric;
                void main() {
                    vec4 cam = model * vec4(in_vert, 1.0);
                    float z = cam.z;
                    float u = dot(k0, cam.xyz) / z;
                    float v = dot(k1, cam.xyz) / z;
                    float x_ndc = 2.0 * (u + 0.5) / image_size.x - 1.0;
                    float y_ndc = 2.0 * (v + 0.5) / image_size.y - 1.0;
                    float a = (far_z + near_z) / (far_z - near_z);
                    float b = -2.0 * far_z * near_z / (far_z - near_z);
                    gl_Position = vec4(x_ndc * z, y_ndc * z, a * z + b, z);
                    camera_point = cam.xyz;
                    camera_normal = mat3(model) * in_normal;
                    barycentric = in_barycentric;
                }
                """,
            fragment_shader="""
                #version 330
                uniform vec3 base_color;
                uniform vec3 edge_color;
                in vec3 camera_point;
                in vec3 camera_normal;
                in vec3 barycentric;
                out vec4 fragColor;
                void main() {
                    if (dot(camera_normal, camera_point) >= 0.0) {
                        discard;
                    }
                    vec3 normal = normalize(camera_normal);
                    vec3 light_dir = normalize(vec3(-0.35, -0.55, -1.0));
                    float diffuse = 0.38 + 0.62 * abs(dot(normal, light_dir));
                    float face_variation = 0.90 + 0.10 * fract(
                        sin(float(gl_PrimitiveID + 1) * 12.9898) * 43758.5453
                    );
                    vec3 derivatives = max(fwidth(barycentric), vec3(1e-6));
                    vec3 interior = smoothstep(
                        vec3(0.0), 1.15 * derivatives, barycentric
                    );
                    float edge = 1.0 - min(min(interior.x, interior.y), interior.z);
                    vec3 surface = base_color * diffuse * face_variation;
                    fragColor = vec4(mix(surface, edge_color, 0.90 * edge), 1.0);
                }
                """,
        )
        self.vaos = {}
        self.vbos = {}
        self.coord_vaos = {}
        self.coord_vbos = {}
        self.mesh_vis_vaos = {}
        self.mesh_vis_vbos = {}
        self.draw_specs = []
        meshes = {part_name: _load_part_mesh(part_name) for part_name in PART_MESHES}
        wrist_vertices = np.asarray(meshes["wrist"].vertices, dtype=np.float64)
        self.canonical_scale = float(
            2.0 * (wrist_vertices.max(axis=0) - wrist_vertices.min(axis=0)).max()
        )
        for part_name, mesh in meshes.items():
            if part_name in ("l_gripper", "r_gripper"):
                static_vertices, moving_vertices = _split_gripper_mesh_vertices(mesh)
                static_coords = self._canonical_coords(part_name, static_vertices)
                moving_coords = self._canonical_coords(part_name, moving_vertices)
                self._add_vao(f"{part_name}_static_wrist", static_vertices, static_coords)
                self._add_vao(f"{part_name}_moving", moving_vertices, moving_coords)
                self.draw_specs.extend(
                    (
                        (f"{part_name}_static_wrist", "static_wrist", "wrist"),
                        (f"{part_name}_moving", part_name, "gripper"),
                    )
                )
            else:
                vertices = np.asarray(mesh.vertices[mesh.faces], dtype=np.float32).reshape(-1, 3)
                self._add_vao(part_name, vertices, self._canonical_coords(part_name, vertices))
                self.draw_specs.append((part_name, part_name, part_name))

    def _canonical_coords(self, part_name, vertices):
        coords_m = _transform_points(vertices, _part_to_canonical_matrix(part_name))
        return (coords_m / self.canonical_scale).astype(np.float32)

    def _add_vao(self, name, vertices, canonical_coords):
        vertices = np.asarray(vertices, dtype="f4").reshape(-1, 3)
        canonical_coords = np.asarray(canonical_coords, dtype="f4").reshape(-1, 3)
        normals = _triangle_vertex_normals(vertices)
        if len(vertices) == 0:
            raise RuntimeError(f"OpenGL mesh segment {name} is empty.")
        if vertices.shape != canonical_coords.shape:
            raise ValueError(
                f"Geometry/canonical coordinate shape mismatch for {name}: "
                f"{vertices.shape} vs {canonical_coords.shape}."
            )
        self.vbos[name] = self.ctx.buffer(vertices)
        self.vaos[name] = self.ctx.simple_vertex_array(
            self.prog,
            self.vbos[name],
            "in_vert",
        )
        packed = np.concatenate((vertices, canonical_coords, normals), axis=1).astype("f4")
        self.coord_vbos[name] = self.ctx.buffer(packed)
        self.coord_vaos[name] = self.ctx.vertex_array(
            self.coord_prog,
            [(self.coord_vbos[name], "3f 3f 3f", "in_vert", "in_coord", "in_normal")],
        )
        barycentric = np.tile(np.eye(3, dtype=np.float32), (len(vertices) // 3, 1))
        mesh_vis_packed = np.concatenate((vertices, normals, barycentric), axis=1).astype("f4")
        self.mesh_vis_vbos[name] = self.ctx.buffer(mesh_vis_packed)
        self.mesh_vis_vaos[name] = self.ctx.vertex_array(
            self.mesh_vis_prog,
            [(
                self.mesh_vis_vbos[name],
                "3f 3f 3f",
                "in_vert",
                "in_normal",
                "in_barycentric",
            )],
        )

    @staticmethod
    def _draw_transform(transform_kind, transforms):
        if transform_kind == "static_wrist":
            return transforms["wrist"] @ _wrist_to_gripper_origin()
        return transforms[transform_kind]

    def _draw_meshes(self, transforms, colors):
        for vao_name, transform_kind, semantic_part in self.draw_specs:
            transform = self._draw_transform(transform_kind, transforms)
            self.prog["model"].value = tuple(transform.T.astype("f4").reshape(-1))
            self.prog["part_color"].value = colors[semantic_part]
            self.vaos[vao_name].render(mode=self.moderngl.TRIANGLES)

    def _draw_coordinate_meshes(self, transforms, effective_part_ids=None, include_parts=None):
        if effective_part_ids is None:
            effective_part_ids = {"shaft": 1.0, "wrist": 2.0, "gripper": 3.0}
        include_parts = None if include_parts is None else set(include_parts)
        for vao_name, transform_kind, semantic_part in self.draw_specs:
            distinct_part = transform_kind if semantic_part == "gripper" else semantic_part
            if include_parts is not None and distinct_part not in include_parts:
                continue
            if transform_kind != "static_wrist" and transform_kind not in transforms:
                continue
            label_name = distinct_part if distinct_part in effective_part_ids else semantic_part
            transform = self._draw_transform(transform_kind, transforms)
            self.coord_prog["model"].value = tuple(transform.T.astype("f4").reshape(-1))
            self.coord_prog["effective_part_id"].value = float(effective_part_ids[label_name])
            self.coord_vaos[vao_name].render(mode=self.moderngl.TRIANGLES)

    def _draw_visual_meshes(self, transforms, include_parts=None):
        include_parts = None if include_parts is None else set(include_parts)
        for vao_name, transform_kind, semantic_part in self.draw_specs:
            distinct_part = transform_kind if semantic_part == "gripper" else semantic_part
            if include_parts is not None and distinct_part not in include_parts:
                continue
            if transform_kind != "static_wrist" and transform_kind not in transforms:
                continue
            transform = self._draw_transform(transform_kind, transforms)
            self.mesh_vis_prog["model"].value = tuple(transform.T.astype("f4").reshape(-1))
            self.mesh_vis_vaos[vao_name].render(mode=self.moderngl.TRIANGLES)

    @staticmethod
    def _set_camera_uniforms(program, K, w, h, near, far):
        K = np.asarray(K, dtype=np.float64).reshape(3, 3)
        program["k0"].value = tuple(K[0, :].astype("f4"))
        program["k1"].value = tuple(K[1, :].astype("f4"))
        program["image_size"].value = (float(w), float(h))
        program["near_z"].value = float(near)
        program["far_z"].value = float(far)

    def _ensure_framebuffer(self, w, h):
        w, h = int(w), int(h)
        if self.fbo is not None and self.w == w and self.h == h:
            return
        key = (w, h)
        resources = self._framebuffers.pop(key, None)
        if resources is None:
            color = self.ctx.renderbuffer(key, components=4, dtype="f4")
            depth = self.ctx.depth_renderbuffer(key)
            fbo = self.ctx.framebuffer(color_attachments=[color], depth_attachment=depth)
            resources = (fbo, color, depth)
        self._framebuffers[key] = resources
        self.w, self.h = w, h
        self.fbo = resources[0]
        while len(self._framebuffers) > self._framebuffer_cache_size:
            _, stale_resources = self._framebuffers.popitem(last=False)
            self._release_framebuffer(stale_resources)

    @staticmethod
    def _release_framebuffer(resources):
        fbo, color, depth = resources
        fbo.release()
        color.release()
        depth.release()

    def release(self):
        for resources in self._framebuffers.values():
            self._release_framebuffer(resources)
        self._framebuffers.clear()
        self.fbo = None
        for vao in self.vaos.values():
            vao.release()
        self.vaos.clear()
        for vao in self.coord_vaos.values():
            vao.release()
        self.coord_vaos.clear()
        for vao in self.mesh_vis_vaos.values():
            vao.release()
        self.mesh_vis_vaos.clear()
        for vbo in self.vbos.values():
            vbo.release()
        self.vbos.clear()
        for vbo in self.coord_vbos.values():
            vbo.release()
        self.coord_vbos.clear()
        for vbo in self.mesh_vis_vbos.values():
            vbo.release()
        self.mesh_vis_vbos.clear()
        self.prog.release()
        self.coord_prog.release()
        self.mesh_vis_prog.release()
        self.ctx.release()

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass

    def _read_depth(self):
        depth = np.frombuffer(
            self.fbo.read(attachment=-1, components=1, dtype="f4"),
            "f4",
        ).reshape(self.h, self.w)
        depth = np.nan_to_num(depth, nan=1.0, posinf=1.0, neginf=1.0)
        background = depth >= 1.0
        z_ndc = 2.0 * depth.astype(np.float64) - 1.0
        z = 2.0 * self.near * self.far / (
            self.far + self.near - z_ndc * (self.far - self.near)
        )
        z[background] = 0.0
        return z.astype(np.float32)

    def render_depth(self, pose, K, image_shape):
        h, w = int(image_shape[0]), int(image_shape[1])
        self._ensure_framebuffer(w, h)
        transforms = fk_matrices_np(
            pose["rot"],
            pose["trans"],
            pose["alpha"],
            pose["theta_l"],
            pose["theta_r"],
        )

        self.fbo.use()
        self.ctx.clear()
        self._set_camera_uniforms(self.prog, K, w, h, self.near, self.far)
        self._draw_meshes(
            transforms,
            {
                "shaft": (1.0, 1.0, 1.0),
                "wrist": (1.0, 1.0, 1.0),
                "gripper": (1.0, 1.0, 1.0),
            },
        )
        return self._read_depth()

    def render_canonical_coordinates(self, pose, K, image_shape):
        """Render perspective-correct canonical XYZ and effective part per pixel.

        This follows the original SurfEmb supervision path: triangle vertices
        carry normalized canonical coordinates, fragment interpolation produces
        the exact visible-surface coordinate, and the mesh depth buffer selects
        a single front-most surface for every pixel.
        """
        h, w = int(image_shape[0]), int(image_shape[1])
        self._ensure_framebuffer(w, h)
        transforms = fk_matrices_np(
            pose["rot"],
            pose["trans"],
            pose["alpha"],
            pose["theta_l"],
            pose["theta_r"],
        )

        self.fbo.use()
        self.ctx.clear()
        self._set_camera_uniforms(self.coord_prog, K, w, h, self.near, self.far)
        self._draw_coordinate_meshes(transforms)
        rgba = np.frombuffer(
            self.fbo.read(attachment=0, components=4, dtype="f4"),
            "f4",
        ).reshape(h, w, 4)
        depth = self._read_depth()
        support = depth > 0.0
        coords = rgba[..., :3].astype(np.float32, copy=True)
        coords[~support] = 0.0
        part_ids = np.zeros((h, w), dtype=np.uint8)
        part_ids[support] = np.rint(rgba[..., 3][support]).astype(np.uint8)
        valid_mask = support & np.isfinite(coords).all(axis=-1)
        return coords, part_ids, depth, valid_mask

    def render_candidate_part_visibility(self, transforms, K, image_shape, include_parts=None):
        """Rasterize arbitrary candidate part transforms with distinct part IDs.

        The coordinate fragment shader rejects back-facing triangles and the
        shared mesh depth attachment handles self- and cross-part occlusion.
        Static x<2.13 mm gripper triangles keep the wrist transform and label.
        """
        h, w = int(image_shape[0]), int(image_shape[1])
        self._ensure_framebuffer(w, h)
        self.fbo.use()
        self.ctx.clear()
        self._set_camera_uniforms(self.coord_prog, K, w, h, self.near, self.far)
        self._draw_coordinate_meshes(
            transforms,
            effective_part_ids={
                "shaft": 1.0,
                "wrist": 2.0,
                "static_wrist": 2.0,
                "l_gripper": 3.0,
                "r_gripper": 4.0,
            },
            include_parts=include_parts,
        )
        rgba = np.frombuffer(
            self.fbo.read(attachment=0, components=4, dtype="f4"),
            "f4",
        ).reshape(h, w, 4)
        depth = self._read_depth()
        support = depth > 0.0
        coords = rgba[..., :3].astype(np.float32, copy=True)
        coords[~support] = 0.0
        part_ids = np.zeros((h, w), dtype=np.uint8)
        part_ids[support] = np.rint(rgba[..., 3][support]).astype(np.uint8)
        valid_mask = support & np.isfinite(coords).all(axis=-1)
        return coords, part_ids, depth, valid_mask

    def render_candidate_mesh_visualization(
        self,
        transforms,
        K,
        image_shape,
        include_parts=None,
        base_color=(0.18, 0.82, 0.92),
        edge_color=(0.01, 0.14, 0.20),
    ):
        """Render front-facing visible triangles with explicit mesh edges."""
        h, w = int(image_shape[0]), int(image_shape[1])
        self._ensure_framebuffer(w, h)
        self.fbo.use()
        self.ctx.clear()
        self._set_camera_uniforms(self.mesh_vis_prog, K, w, h, self.near, self.far)
        self.mesh_vis_prog["base_color"].value = tuple(
            np.asarray(base_color, dtype=np.float32).reshape(3)
        )
        self.mesh_vis_prog["edge_color"].value = tuple(
            np.asarray(edge_color, dtype=np.float32).reshape(3)
        )
        self._draw_visual_meshes(transforms, include_parts=include_parts)
        rgba = np.frombuffer(
            self.fbo.read(attachment=0, components=4, dtype="f4"),
            "f4",
        ).reshape(h, w, 4)
        depth = self._read_depth()
        valid_mask = depth > 0.0
        rgb = np.clip(rgba[..., :3] * 255.0, 0.0, 255.0).astype(np.uint8)
        rgb[~valid_mask] = 0
        return rgb, depth, valid_mask

    def render_pose_mask(self, pose, K, image_shape):
        h, w = int(image_shape[0]), int(image_shape[1])
        self._ensure_framebuffer(w, h)
        transforms = fk_matrices_np(
            pose["rot"],
            pose["trans"],
            pose["alpha"],
            pose["theta_l"],
            pose["theta_r"],
        )

        self.fbo.use()
        self.ctx.clear()
        self._set_camera_uniforms(self.prog, K, w, h, self.near, self.far)

        colors = {
            "shaft": (0.0, 0.0, 1.0),
            "wrist": (0.0, 1.0, 0.0),
            "gripper": (1.0, 0.0, 0.0),
        }
        self._draw_meshes(transforms, colors)

        rgb = np.frombuffer(
            self.fbo.read(attachment=0, components=4, dtype="f4"),
            "f4",
        ).reshape(h, w, 4)[..., :3]
        depth = self._read_depth()
        mask = np.zeros((h, w), dtype=np.uint8)
        support = depth > 0.0
        if support.any():
            chan = np.argmax(rgb, axis=-1)
            mask[support & (chan == 0)] = 1
            mask[support & (chan == 1)] = 2
            mask[support & (chan == 2)] = 3
        return mask

    def render_pose(self, pose, K, image_shape):
        depth = self.render_depth(pose, K, image_shape)
        color = np.zeros((int(image_shape[0]), int(image_shape[1]), 3), dtype=np.uint8)
        color[depth > 0.0] = np.array([180, 220, 255], dtype=np.uint8)
        return color, depth, None

    def render_pose_overlay(self, rgb, pose, K, alpha=0.70):
        color, depth, _ = self.render_pose(pose, K, rgb.shape[:2])
        support = depth > 0.0
        out = np.asarray(rgb, dtype=np.uint8).copy()
        if support.any():
            blended = (
                out[support].astype(np.float32) * (1.0 - float(alpha))
                + color[support].astype(np.float32) * float(alpha)
            )
            out[support] = np.clip(blended, 0, 255).astype(np.uint8)
        return out

    def render_pose_overlay_warped_crop(self, rgb_crop, pose, K_orig, M_crop, orig_shape, alpha=0.70):
        color, depth, _ = self.render_pose(pose, K_orig, orig_shape)
        h, w = np.asarray(rgb_crop).shape[:2]
        M = np.asarray(M_crop, dtype=np.float32).reshape(2, 3)
        color_crop = cv2.warpAffine(
            color,
            M,
            (int(w), int(h)),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        support_crop = cv2.warpAffine(
            (depth > 0.0).astype(np.uint8),
            M,
            (int(w), int(h)),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).astype(bool)
        out = np.asarray(rgb_crop, dtype=np.uint8).copy()
        if support_crop.any():
            blended = (
                out[support_crop].astype(np.float32) * (1.0 - float(alpha))
                + color_crop[support_crop].astype(np.float32) * float(alpha)
            )
            out[support_crop] = np.clip(blended, 0, 255).astype(np.uint8)
        return out
