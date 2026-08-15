#!/usr/bin/env python3
"""Train the wrist-only SurfEmb ResNet using SurgRIPE-LND exclusively."""

import train_surfemb_resnet_wrist_only_rarp_lnd as wrist_runner


_train = wrist_runner._train
_wrist_dataset = wrist_runner._wrist_dataset


def _build_wrist_dataset_factory(args):
    def dataset_factory(*factory_args, **factory_kwargs):
        return _wrist_dataset.SurfEmbWristOnlyCropDataset(
            *factory_args,
            min_wrist_pixels=args.wrist_min_visible_pixels,
            negative_visible_only=args.wrist_negative_source == "visible",
            **factory_kwargs,
        )

    return dataset_factory


def main():
    parser = wrist_runner.build_parser()
    parser.set_defaults(
        name="surfemb_resnet_wristonly_lnd_visible_p1024_b56_gpu4567",
        train_dataset_names="",
        include_lnd_train=1,
        include_lnd_val=1,
        lnd_sampling_rate=1.0,
        wrist_negative_source="full_surface",
        validate_before_train=1,
    )
    args = parser.parse_args()
    if args.surfemb_n_pos <= 0 or args.surfemb_n_neg <= 0:
        raise ValueError("SurfEmb n_pos and n_neg must both be positive.")
    if (
        args.surfemb_shaft_sample_ratio,
        args.surfemb_wrist_sample_ratio,
        args.surfemb_gripper_sample_ratio,
    ) != (0.0, 1.0, 0.0):
        raise ValueError("LND-only wrist sampling ratios must remain 0/1/0.")
    if not bool(args.include_lnd_train) or not bool(args.include_lnd_val):
        raise ValueError("The LND-only runner requires both LND TRAIN and TEST validation.")

    dataset_factory = _build_wrist_dataset_factory(args)
    _train._base_train.SurfEmbKeypointCropDataset = dataset_factory

    def make_lnd_only_train(run_args):
        return [
            _train._base_train.make_lnd_dataset(
                run_args,
                split="TRAIN",
                training=True,
                use_memory_pose=True,
                subsample=run_args.lnd_train_subsample,
            )
        ]

    def make_lnd_primary_validation(run_args):
        return _train._base_train.make_lnd_dataset(
            run_args,
            split="TEST",
            training=False,
            use_memory_pose=False,
            subsample=run_args.lnd_val_subsample,
        )

    _train.make_train_datasets = make_lnd_only_train
    _train.make_rarp_val_dataset = make_lnd_primary_validation
    _train.make_lnd_val_dataset = lambda _args: None
    _train.PRIMARY_VAL_NAME = "LND"

    print(
        "LND_ONLY_EXPERIMENT: LND TRAIN only; LND TEST primary validation; "
        "wrist ROI crop + wrist binary mask + visible wrist positives + "
        f"{args.wrist_negative_source} wrist negatives; "
        "RARP disabled",
        flush=True,
    )
    _train.main(args)


if __name__ == "__main__":
    main()
