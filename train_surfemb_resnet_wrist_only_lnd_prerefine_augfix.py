#!/usr/bin/env python3
"""LND wrist-only SurfEmb run with corrected original-style RGB augmentation."""

from pathlib import Path

import train_surfemb_resnet_wrist_only_rarp_lnd as wrist_runner
from torch.utils.data import Dataset


_train = wrist_runner._train
_wrist_dataset = wrist_runner._wrist_dataset
PREREFINE_MEMORY = (
    Path(__file__).resolve().parents[1]
    / "gaussian-mesh-splatting"
    / "Results2"
    / "surgripe_lnd_action_gt_full"
    / "TRAIN"
    / "memory_pool.json"
)
TRAIN_EXCLUDED_FRAME_IDS = (340, 408, 779, 1125)
VAL_EXCLUDED_FRAME_IDS = (210,)


class _FrameFilteredDataset(Dataset):
    def __init__(self, dataset, excluded_frame_ids):
        self.dataset = dataset
        self.excluded_frame_ids = tuple(sorted(int(v) for v in excluded_frame_ids))
        excluded = set(self.excluded_frame_ids)
        samples = dataset.base_dataset.samples
        self.indices = [i for i, sample in enumerate(samples) if int(sample[0]) not in excluded]
        found = {int(samples[i][0]) for i in range(len(samples)) if int(samples[i][0]) in excluded}
        if found != excluded:
            raise ValueError(f"Missing explicitly excluded frame IDs: {sorted(excluded - found)}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]

    def set_epoch(self, epoch):
        self.dataset.set_epoch(epoch)

    def __repr__(self):
        return (
            f"strict_frame_filter(N={len(self)} excluded={self.excluded_frame_ids} "
            f"dataset={self.dataset})"
        )


def _build_strict_wrist_dataset_factory(args):
    def dataset_factory(*factory_args, **factory_kwargs):
        return _wrist_dataset.SurfEmbWristOnlyCropDataset(
            *factory_args,
            min_wrist_pixels=args.wrist_min_visible_pixels,
            negative_visible_only=args.wrist_negative_source == "visible",
            fallback_to_other_sample=False,
            **factory_kwargs,
        )

    return dataset_factory


def main():
    parser = wrist_runner.build_parser()
    parser.set_defaults(
        name="surfemb_resnet_wristonly_lnd_prerefine_augfix_p1024_b56_gpu0123",
        train_dataset_names="",
        include_lnd_train=1,
        include_lnd_val=1,
        lnd_sampling_rate=1.0,
        lnd_refine_memory=str(PREREFINE_MEMORY),
        surfemb_n_pos=1024,
        surfemb_n_neg=1024,
        wrist_negative_source="full_surface",
        surfemb_augmentation_profile="original_p30",
        surfemb_crop_min_mask_retention=0.70,
        surfemb_crop_offset_multiplier=2.5,
        validate_before_train=1,
    )
    args = parser.parse_args()
    if Path(args.lnd_refine_memory).resolve() != PREREFINE_MEMORY.resolve():
        raise ValueError(
            "This augmentation control must use the pre-refinement TRAIN memory: "
            f"{PREREFINE_MEMORY}"
        )
    if args.surfemb_augmentation_profile != "original_p30":
        raise ValueError("The augmentation control requires surfemb_augmentation_profile=original_p30.")
    if abs(float(args.surfemb_crop_min_mask_retention) - 0.70) > 1e-9:
        raise ValueError("The augmentation control requires crop retention 0.70.")
    if abs(float(args.surfemb_crop_offset_multiplier) - 2.5) > 1e-9:
        raise ValueError("The augmentation control requires crop offset multiplier 2.5.")
    if args.surfemb_n_pos <= 0 or args.surfemb_n_neg <= 0:
        raise ValueError("SurfEmb n_pos and n_neg must both be positive.")
    if (
        args.surfemb_shaft_sample_ratio,
        args.surfemb_wrist_sample_ratio,
        args.surfemb_gripper_sample_ratio,
    ) != (0.0, 1.0, 0.0):
        raise ValueError("LND-only wrist sampling ratios must remain 0/1/0.")
    if not bool(args.include_lnd_train) or not bool(args.include_lnd_val):
        raise ValueError("The LND-only control requires LND TRAIN and TEST validation.")

    _train._base_train.SurfEmbKeypointCropDataset = _build_strict_wrist_dataset_factory(args)

    def make_lnd_only_train(run_args):
        return [
            _FrameFilteredDataset(
                _train._base_train.make_lnd_dataset(
                    run_args,
                    split="TRAIN",
                    training=True,
                    use_memory_pose=True,
                    subsample=run_args.lnd_train_subsample,
                ),
                TRAIN_EXCLUDED_FRAME_IDS,
            )
        ]

    def make_lnd_primary_validation(run_args):
        validation = _FrameFilteredDataset(
            _train._base_train.make_lnd_dataset(
                run_args,
                split="TEST",
                training=False,
                use_memory_pose=False,
                subsample=run_args.lnd_val_subsample,
            ),
            VAL_EXCLUDED_FRAME_IDS,
        )
        wrist_dataset = validation.dataset
        base_dataset = wrist_dataset.base_dataset
        if not isinstance(wrist_dataset, _wrist_dataset.SurfEmbWristOnlyCropDataset):
            raise TypeError("Primary validation must be SurfEmbWristOnlyCropDataset.")
        if base_dataset.split != "TEST" or base_dataset.use_memory_pose:
            raise RuntimeError("Primary validation must use direct-pose LND/TEST.")
        if wrist_dataset.supervision_part_ids != (2,):
            raise RuntimeError("Primary validation must supervise wrist part ID 2 only.")
        if tuple(wrist_dataset.part_sample_ratios.tolist()) != (0.0, 1.0, 0.0):
            raise RuntimeError("Primary validation must sample shaft/wrist/gripper at 0/1/0.")
        if wrist_dataset.negative_visible_only:
            raise RuntimeError("Primary validation must use full-surface wrist negatives.")
        return validation

    _train.make_train_datasets = make_lnd_only_train
    _train.make_rarp_val_dataset = make_lnd_primary_validation
    _train.make_lnd_val_dataset = lambda _args: None
    _train.PRIMARY_VAL_NAME = "LND"

    print(
        "LND_PREREFINE_AUGFIX: GaussNoise variance=10..50; full ColorJitter B/C/S=0.2 hue=0.1; "
        "ISO/CLAHE/Debayer/Unsharpen p=0.30; focused dropout retained; "
        "crop retention=0.70 with 2.5x offset; "
        "source wrist visibility <0.70 forces full visible-wrist crop and bypasses render-IoU rejection; "
        "full-surface wrist negatives; sample fallback disabled; "
        f"TRAIN excluded={TRAIN_EXCLUDED_FRAME_IDS}; VAL excluded={VAL_EXCLUDED_FRAME_IDS}; RARP disabled",
        flush=True,
    )
    print(f"PREREFINE_MEMORY: {PREREFINE_MEMORY}", flush=True)
    print(
        "PRIMARY_VALIDATION_HARD_RULE: LND/TEST, direct wrist GT, wrist mask only, "
        "wrist positive/negative correspondence only",
        flush=True,
    )
    _train.main(args)


if __name__ == "__main__":
    main()
