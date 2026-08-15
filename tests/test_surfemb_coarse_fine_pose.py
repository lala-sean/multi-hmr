import unittest

import numpy as np
import torch

from surfemb_articulated_pose import (
    PartScoreContext,
    PartSurface,
    _local_refinement_correspondences,
)


class LocalRefinementCorrespondenceTest(unittest.TestCase):
    def setUp(self):
        yy, xx = np.mgrid[:5, :5]
        points = np.stack((xx.reshape(-1), yy.reshape(-1), np.zeros(25)), axis=1).astype(
            np.float32
        )
        corr_prob = torch.full((25, 25), 1e-6, dtype=torch.float32)
        corr_prob[torch.arange(25), torch.arange(25)] = 0.9
        self.surface = PartSurface(
            "wrist",
            points,
            np.tile(np.asarray([[0.0, 0.0, -1.0]], dtype=np.float32), (25, 1)),
            points.copy(),
            1.0,
        )
        self.context = PartScoreContext(
            points=torch.from_numpy(points),
            corr_prob=corr_prob,
            corr_log_score=torch.log(corr_prob),
            mask_log_prob=torch.zeros(25),
            neg_mask_log_prob=torch.zeros(25),
        )

    def test_rematches_twenty_style_local_neighborhoods(self):
        pixels, keys, confidence, diagnostics = _local_refinement_correspondences(
            self.context,
            self.surface,
            (5, 5),
            seed_pixels=np.asarray([12]),
            seed_keys=np.asarray([12]),
            pixel_mask=np.ones(25, dtype=bool),
            neighbors_per_inlier=5,
            max_correspondences=20,
        )
        self.assertEqual(diagnostics["fine_local_candidate_pairs"], 5)
        self.assertEqual(diagnostics["fine_correspondences"], 5)
        self.assertEqual(len(np.unique(pixels)), len(pixels))
        self.assertEqual(len(np.unique(keys)), len(keys))
        np.testing.assert_array_equal(pixels, keys)
        self.assertTrue(np.all(confidence > 0.8))

    def test_respects_pixel_roi(self):
        roi = np.zeros(25, dtype=bool)
        roi[[7, 11, 12, 13, 17]] = True
        pixels, _, _, diagnostics = _local_refinement_correspondences(
            self.context,
            self.surface,
            (5, 5),
            seed_pixels=np.asarray([12]),
            seed_keys=np.asarray([12]),
            pixel_mask=roi,
            neighbors_per_inlier=5,
            max_correspondences=20,
        )
        self.assertEqual(diagnostics["fine_correspondences"], 5)
        self.assertTrue(np.all(roi[pixels]))


if __name__ == "__main__":
    unittest.main()
