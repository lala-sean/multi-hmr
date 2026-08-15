from surfemb_keypoint_crop import SurfEmbKeypointCropDataset


class SurfEmbWristOnlyCropDataset(SurfEmbKeypointCropDataset):
    """Wrist-ROI SurfEmb supervision with configurable wrist negatives."""

    def __init__(self, *args, min_wrist_pixels=20, negative_visible_only=True, **kwargs):
        kwargs["part_sample_ratios"] = (0.0, 1.0, 0.0)
        kwargs["supervision_part_ids"] = (2,)
        kwargs["min_supervision_pixels"] = int(min_wrist_pixels)
        kwargs["negative_visible_only"] = bool(negative_visible_only)
        super().__init__(*args, **kwargs)

    def __repr__(self):
        return "wrist_only_" + super().__repr__()
