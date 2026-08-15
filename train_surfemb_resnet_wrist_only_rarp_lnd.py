import importlib.util
import sys
from argparse import ArgumentParser
from pathlib import Path


ROBOPEPP_ROOT = Path(__file__).resolve().parent
if str(ROBOPEPP_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBOPEPP_ROOT))
DATASETS_ROOT = ROBOPEPP_ROOT / "datasets"
if str(DATASETS_ROOT) not in sys.path:
    sys.path.insert(0, str(DATASETS_ROOT))


def _load_local_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_train = _load_local_module(
    "robopepp_surfemb_resnet_full_runner_for_wrist_only",
    ROBOPEPP_ROOT / "train_surfemb_resnet_crop_rarp_lnd_refinemem.py",
)
_wrist_dataset = _load_local_module(
    "robopepp_surfemb_wrist_only_dataset",
    DATASETS_ROOT / "surfemb_wrist_only_crop.py",
)


def build_parser():
    parser = ArgumentParser()
    parser.add_argument("--save_dir", default=str(ROBOPEPP_ROOT / "logs"))
    parser.add_argument("--name", default="surfemb_resnet_wrist_only_rarp_lnd")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--resume_optimizer", type=int, default=1, choices=[0, 1])
    parser.add_argument("--resume_scheduler", type=int, default=1, choices=[0, 1])
    parser.add_argument("--reset_iter_on_resume", type=int, default=0, choices=[0, 1])
    parser.add_argument("--inspect_datasets_only", type=int, default=0, choices=[0, 1])
    parser.add_argument("--inspect_vis_dir", default=None)
    parser.add_argument("--inspect_samples_per_dataset", type=int, default=3)
    parser.add_argument("--inspect_render_mesh", type=int, default=1, choices=[0, 1])

    parser.add_argument("--needle_puncture_data_dir", default=_train.PUNCTURE_DATASET_ROOT)
    parser.add_argument("--needle_puncture_pose_dir", default=_train.PUNCTURE_POSE_ROOT)
    parser.add_argument("--needle_grasping_data_dir", default=_train.GRASPING_DATASET_ROOT)
    parser.add_argument("--needle_grasping_pose_dir", default=_train.GRASPING_POSE_ROOT)
    parser.add_argument("--knotting_data_dir", default=_train.KNOTTING_DATASET_ROOT)
    parser.add_argument("--knotting_pose_dir", default=_train.KNOTTING_POSE_ROOT)
    parser.add_argument("--train_dataset_names", default="needlePuncture,needleGrasping,knotting")
    parser.add_argument("--min_dice_shaft", type=float, default=0.8)
    parser.add_argument("--min_dice_wrist", type=float, default=0.6)
    parser.add_argument("--min_dice_gripper", type=float, default=0.6)
    parser.add_argument("--needle_train_ratio", type=float, default=0.95)
    parser.add_argument("--canonicalize_pose_symmetry", type=int, default=1, choices=[0, 1])
    parser.add_argument("--canonical_eps", type=float, default=0.08)
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--bbox_padding_frac", type=float, default=0.12)
    parser.add_argument("--dataset_cache_dir", default=None)
    parser.add_argument("--surface_points_path", default=str(_train.DEFAULT_SURFACE_POINTS))

    # The full-instrument baseline has exactly 614 wrist positives and 614 wrist negatives.
    parser.add_argument("--surfemb_n_pos", type=int, default=614)
    parser.add_argument("--surfemb_n_neg", type=int, default=614)
    parser.add_argument("--surfemb_shaft_sample_ratio", type=float, default=0.0)
    parser.add_argument("--surfemb_wrist_sample_ratio", type=float, default=1.0)
    parser.add_argument("--surfemb_gripper_sample_ratio", type=float, default=0.0)
    parser.add_argument("--wrist_min_visible_pixels", type=int, default=20)
    parser.add_argument(
        "--wrist_negative_source",
        choices=("visible", "full_surface"),
        default="visible",
        help="Sample wrist negatives from visible raster pixels or the full effective wrist surface.",
    )
    parser.add_argument("--surfemb_key_noise", type=float, default=1e-3)
    parser.add_argument("--surfemb_similarity", choices=("raw_dot", "cosine"), default="raw_dot")
    parser.add_argument("--surfemb_temperature", type=float, default=1.0)
    parser.add_argument("--surfemb_crop_scale", type=float, default=1.2)
    parser.add_argument("--surfemb_max_angle", type=float, default=3.141592653589793)
    parser.add_argument("--surfemb_offset_scale", type=float, default=1.0)
    parser.add_argument(
        "--surfemb_augmentation_profile",
        choices=("legacy", "original_p30"),
        default="legacy",
    )
    parser.add_argument("--surfemb_crop_min_mask_retention", type=float, default=0.985)
    parser.add_argument("--surfemb_crop_offset_multiplier", type=float, default=1.0)
    parser.add_argument("--surfemb_min_depth", type=float, default=1e-4)
    parser.add_argument("--surfemb_depth_tolerance", type=float, default=8e-4)
    parser.add_argument("--surfemb_use_mesh_zbuffer", type=int, default=1, choices=[1])
    parser.add_argument("--surfemb_zbuffer_backend", default="opengl", choices=["opengl"])
    parser.add_argument("--surfemb_min_train_render_iou", type=float, default=0.35)
    parser.add_argument("--heatmap_sigma", type=float, default=2.0)

    parser.add_argument("--include_lnd_train", type=int, default=1, choices=[0, 1])
    parser.add_argument("--include_lnd_val", type=int, default=1, choices=[0, 1])
    parser.add_argument("--lnd_root", default=_train.DEFAULT_LND_ROOT)
    parser.add_argument("--lnd_refine_memory", default=_train.DEFAULT_LND_REFINE_MEMORY)
    parser.add_argument("--lnd_sampling_rate", type=float, default=0.30)
    parser.add_argument("--lnd_train_subsample", type=int, default=1)
    parser.add_argument("--lnd_val_subsample", type=int, default=1)

    parser.add_argument("--batch_size", type=int, default=56)
    parser.add_argument("--val_batch_size", type=int, default=56)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--train_subsample", type=int, default=1)
    parser.add_argument("--val_subsample", type=int, default=10)
    parser.add_argument("--max_iter", type=int, default=60000)
    parser.add_argument("--log_freq", type=int, default=10)
    parser.add_argument("--val_freq", type=int, default=1000)
    parser.add_argument("--val_max_batches", type=int, default=0)
    parser.add_argument("--val_score_mode", default="rarp_total", choices=["rarp_total", "mean_total"])
    parser.add_argument("--ckpt_freq", type=int, default=1000)
    parser.add_argument("--validate_before_train", type=int, default=1, choices=[0, 1])
    parser.add_argument("--save_best_ckpt", type=int, default=1, choices=[0, 1])
    parser.add_argument("--save_val_iter_ckpt", type=int, default=0, choices=[0, 1])
    parser.add_argument("--amp", type=int, default=1, choices=[0, 1])
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--dist_timeout_sec", type=int, default=7200)

    parser.add_argument("--surfemb_emb_dim", type=int, default=12)
    parser.add_argument("--surfemb_mlp_hidden_features", type=int, default=256)
    parser.add_argument("--surfemb_mlp_hidden_layers", type=int, default=2)
    parser.add_argument("--resnet_feat_preultimate", type=int, default=64)
    parser.add_argument("--lr_cnn", type=float, default=1e-4)
    parser.add_argument("--lr_surfemb_mlp", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    return parser


def main():
    args = build_parser().parse_args()
    if (args.surfemb_n_pos, args.surfemb_n_neg) != (614, 614):
        raise ValueError("Fair comparison requires wrist-only n_pos=n_neg=614.")
    if (
        args.surfemb_shaft_sample_ratio,
        args.surfemb_wrist_sample_ratio,
        args.surfemb_gripper_sample_ratio,
    ) != (0.0, 1.0, 0.0):
        raise ValueError("Wrist-only sampling ratios must remain 0/1/0.")

    def dataset_factory(*factory_args, **factory_kwargs):
        return _wrist_dataset.SurfEmbWristOnlyCropDataset(
            *factory_args,
            min_wrist_pixels=args.wrist_min_visible_pixels,
            negative_visible_only=args.wrist_negative_source == "visible",
            **factory_kwargs,
        )

    _train._base_train.SurfEmbKeypointCropDataset = dataset_factory
    print(
        "WRIST_ONLY_EXPERIMENT: wrist ROI + wrist binary mask + visible wrist positives; "
        f"negative_source={args.wrist_negative_source}; "
        f"min_original_wrist_pixels={args.wrist_min_visible_pixels}",
        flush=True,
    )
    _train.main(args)


if __name__ == "__main__":
    main()
