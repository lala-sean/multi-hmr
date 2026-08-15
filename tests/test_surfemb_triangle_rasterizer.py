import os
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import instrument_opengl_renderer as renderer_mod
from instrument_geometry import (
    SURFEMB_GRIPPER_STATIC_THRESHOLD_M,
    SURFEMB_SHAFT_NORM_X_MIN,
    fk_matrices_np,
    project_points_np,
    surfemb_surface_sampling_mask,
)
from surfemb_articulated_pose import (
    filter_correspondences_by_triangle_visibility,
    load_part_surfaces,
)


SURFACE_ASSET = (
    ROOT
    / "assets"
    / "instrument_surface_samples_surfemb_x2.13mm_wg1over3_shafttop30mm"
    / "instrument_surface_points_all.npy"
)


def _pose(alpha=0.4, theta_l=0.5, theta_r=-0.35):
    return {
        "rot": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        "trans": np.array([0.0, 0.0, 0.35], dtype=np.float64),
        "alpha": float(alpha),
        "theta_l": float(theta_l),
        "theta_r": float(theta_r),
    }


class SurfEmbTriangleRasterizerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.renderer = renderer_mod.InstrumentOpenGLDepthRenderer(224, 224, device_idx=0)
        cls.K = np.array(
            [[500.0, 0.0, 112.0], [0.0, 500.0, 112.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    @classmethod
    def tearDownClass(cls):
        cls.renderer.release()

    def test_gripper_threshold_splits_triangles_at_2p13_mm(self):
        threshold = float(SURFEMB_GRIPPER_STATIC_THRESHOLD_M)
        self.assertAlmostEqual(threshold, 0.00213, places=9)
        for part_name in ("l_gripper", "r_gripper"):
            mesh = renderer_mod._load_part_mesh(part_name)
            static, moving = renderer_mod._split_gripper_mesh_vertices(mesh, threshold)
            self.assertGreater(len(static), 0)
            self.assertGreater(len(moving), 0)
            self.assertLessEqual(float(static[:, 0].max()), threshold + 1e-8)
            self.assertGreaterEqual(float(moving[:, 0].min()), threshold - 1e-8)

        semantics = {name: semantic for name, _, semantic in self.renderer.draw_specs}
        self.assertEqual(semantics["l_gripper_static_wrist"], "wrist")
        self.assertEqual(semantics["r_gripper_static_wrist"], "wrist")
        self.assertEqual(semantics["l_gripper_moving"], "gripper")
        self.assertEqual(semantics["r_gripper_moving"], "gripper")

    def test_static_gripper_transform_does_not_follow_gripper_rotation(self):
        first = _pose(theta_l=0.1, theta_r=-0.2)
        second = _pose(theta_l=0.9, theta_r=-0.8)
        transforms_a = fk_matrices_np(**{
            "quat_wxyz": first["rot"],
            "trans": first["trans"],
            "alpha": first["alpha"],
            "theta_l": first["theta_l"],
            "theta_r": first["theta_r"],
        })
        transforms_b = fk_matrices_np(**{
            "quat_wxyz": second["rot"],
            "trans": second["trans"],
            "alpha": second["alpha"],
            "theta_l": second["theta_l"],
            "theta_r": second["theta_r"],
        })
        static_a = self.renderer._draw_transform("static_wrist", transforms_a)
        static_b = self.renderer._draw_transform("static_wrist", transforms_b)
        np.testing.assert_allclose(static_a, static_b, atol=1e-12)
        self.assertGreater(np.abs(transforms_a["l_gripper"] - transforms_b["l_gripper"]).max(), 1e-3)

    def test_canonical_scale_matches_negative_surface_asset(self):
        payload = np.load(SURFACE_ASSET, allow_pickle=True).item()
        self.assertAlmostEqual(
            self.renderer.canonical_scale,
            float(payload["canon_scale"]),
            places=8,
        )

    def test_surface_sampling_excludes_only_shaft_below_minus_half(self):
        payload = np.load(SURFACE_ASSET, allow_pickle=True).item()
        coords = payload["points_norm"]
        part_ids = payload["effective_part_ids"]
        keep = surfemb_surface_sampling_mask(coords, part_ids)

        self.assertAlmostEqual(SURFEMB_SHAFT_NORM_X_MIN, -0.5)
        self.assertFalse(np.any((part_ids[keep] == 1) & (coords[keep, 0] < -0.5)))
        self.assertTrue(np.all(keep[part_ids != 1]))
        self.assertGreater(int(np.count_nonzero(~keep)), 0)

    def test_coordinate_pass_is_finite_front_surface_with_shared_depth(self):
        pose = _pose()
        coords, part_ids, depth, valid = self.renderer.render_canonical_coordinates(
            pose,
            self.K,
            (224, 224),
        )
        all_face_mask = self.renderer.render_pose_mask(pose, self.K, (224, 224))
        self.assertGreater(int(valid.sum()), 500)
        self.assertTrue(np.isfinite(coords[valid]).all())
        self.assertTrue((depth[valid] > 0.0).all())
        self.assertTrue(np.array_equal(valid, part_ids > 0))
        self.assertTrue(np.all(valid <= (all_face_mask > 0)))
        self.assertSetEqual(set(np.unique(part_ids[valid]).tolist()), {1, 2, 3})

        sample_eligible = surfemb_surface_sampling_mask(
            coords.reshape(-1, 3), part_ids.reshape(-1)
        ).reshape(valid.shape)
        sampled = valid & sample_eligible
        self.assertFalse(np.any((part_ids[sampled] == 1) & (coords[sampled, 0] < -0.5)))
        self.assertGreater(int(np.count_nonzero(valid & ~sample_eligible)), 0)

    def test_candidate_visibility_has_distinct_gripper_ids_and_filters_backfaces(self):
        pose = _pose()
        transforms = fk_matrices_np(
            pose["rot"], pose["trans"], pose["alpha"], pose["theta_l"], pose["theta_r"]
        )
        _, part_ids, depth, valid = self.renderer.render_candidate_part_visibility(
            transforms,
            self.K,
            (224, 224),
        )
        self.assertSetEqual(set(np.unique(part_ids[valid]).tolist()), {1, 2, 3, 4})

        surfaces = load_part_surfaces(SURFACE_ASSET.parent, keys_per_part=4096, seed=17)
        for name in ("l_gripper", "r_gripper"):
            surface = surfaces[name]
            points_cam = surface.points_m @ transforms[name][:3, :3].T + transforms[name][:3, 3]
            uv = project_points_np(points_cam, self.K)
            u = np.rint(uv[:, 0]).astype(np.int64)
            v = np.rint(uv[:, 1]).astype(np.int64)
            pixels = v * 224 + u
            keys = np.arange(len(surface.points_m), dtype=np.int64)
            kept_pixels, kept_keys, diagnostics = filter_correspondences_by_triangle_visibility(
                surface,
                name,
                transforms[name],
                pixels,
                keys,
                np.arange(len(keys), dtype=np.int64),
                self.K,
                (224, 224),
                part_ids,
                depth,
                valid,
                depth_tolerance=0.0008,
            )
            self.assertEqual(len(kept_pixels), len(kept_keys))
            self.assertGreater(diagnostics["visibility_kept"], 50)
            self.assertGreater(diagnostics["visibility_backface_rejected"], 50)
            self.assertGreater(diagnostics["visibility_occlusion_rejected"], 0)
            self.assertLess(diagnostics["visibility_keep_fraction"], 0.75)

    def test_rotated_crop_intrinsics_match_affine_warped_raster(self):
        pose = _pose()
        K_orig = np.array(
            [[700.0, 0.0, 320.0], [0.0, 700.0, 240.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        angle = 0.73
        c, s = np.cos(angle), np.sin(angle)
        M = np.array([[c, -s, 0.0], [s, c, 0.0]], dtype=np.float64) * 0.72
        M[:, 2] = np.array([112.0, 112.0]) - M[:, :2] @ np.array([320.0, 240.0])
        K_crop = np.vstack((M, [0.0, 0.0, 1.0])) @ K_orig

        original = self.renderer.render_pose_mask(pose, K_orig, (480, 640)) > 0
        warped = cv2.warpAffine(
            original.astype(np.uint8),
            M.astype(np.float32),
            (224, 224),
            flags=cv2.INTER_NEAREST,
        ).astype(bool)
        _, _, _, direct = self.renderer.render_canonical_coordinates(pose, K_crop, (224, 224))
        iou = np.count_nonzero(warped & direct) / np.count_nonzero(warped | direct)
        self.assertGreater(iou, 0.95)

    def test_rasterized_coordinates_reproject_to_their_pixels(self):
        pose = _pose()
        coords, part_ids, depth, valid = self.renderer.render_canonical_coordinates(
            pose,
            self.K,
            (224, 224),
        )
        transforms = fk_matrices_np(
            pose["rot"], pose["trans"], pose["alpha"], pose["theta_l"], pose["theta_r"]
        )
        candidates = {
            1: (("shaft", transforms["shaft"]),),
            2: (
                ("wrist", transforms["wrist"]),
                ("l_gripper", transforms["wrist"] @ renderer_mod._wrist_to_gripper_origin()),
                ("r_gripper", transforms["wrist"] @ renderer_mod._wrist_to_gripper_origin()),
            ),
            3: (
                ("l_gripper", transforms["l_gripper"]),
                ("r_gripper", transforms["r_gripper"]),
            ),
        }
        ys, xs = np.where(valid)
        rng = np.random.default_rng(7)
        chosen = rng.choice(len(ys), min(256, len(ys)), replace=False)
        uv_errors = []
        depth_errors = []
        for y, x in zip(ys[chosen], xs[chosen]):
            canonical_m = coords[y, x].astype(np.float64) * self.renderer.canonical_scale
            options = []
            for part_name, draw_transform in candidates[int(part_ids[y, x])]:
                canonical_to_part = np.linalg.inv(renderer_mod._part_to_canonical_matrix(part_name))
                point_part = renderer_mod._transform_points(canonical_m[None], canonical_to_part)
                point_cam = renderer_mod._transform_points(point_part, draw_transform)
                uv = project_points_np(point_cam, self.K)[0]
                options.append(
                    (
                        float(np.linalg.norm(uv - np.array([x, y], dtype=np.float64))),
                        float(abs(point_cam[0, 2] - depth[y, x])),
                    )
                )
            uv_error, depth_error = min(options, key=lambda value: value[0] + 100.0 * value[1])
            uv_errors.append(uv_error)
            depth_errors.append(depth_error)

        self.assertLess(max(uv_errors), 0.01)
        self.assertLess(max(depth_errors), 2e-4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
