#!/usr/bin/env python3
"""Export small, public-safe experiment records; never copy datasets or weights."""

import argparse
import csv
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGES = (
    "torch", "torchvision", "numpy", "scipy", "opencv-python", "Pillow",
    "moderngl", "glcontext", "PyOpenGL", "trimesh", "pyrender", "timm",
    "rtree", "matplotlib", "pytest", "einops", "omegaconf", "PyYAML",
    "pytorch-lightning", "albumentations", "pymeshlab", "torch-scatter",
)
CSV_FILES = (
    "lnd_all_evaluations.csv", "rarp_all_evaluations.csv",
    "training_loss_endpoints.csv",
)
RECIPE_FIELDS = (
    "name", "resume", "resume_optimizer", "resume_scheduler", "reset_iter_on_resume",
    "train_dataset_names", "include_lnd_train", "include_lnd_val", "lnd_sampling_rate",
    "lnd_refine_memory", "surface_points_path", "crop_size", "bbox_padding_frac",
    "canonicalize_pose_symmetry", "canonical_eps", "surfemb_n_pos", "surfemb_n_neg",
    "wrist_negative_source", "wrist_min_visible_pixels", "surfemb_key_noise",
    "surfemb_similarity", "surfemb_temperature", "surfemb_crop_scale",
    "surfemb_max_angle", "surfemb_offset_scale", "surfemb_augmentation_profile",
    "surfemb_crop_min_mask_retention", "surfemb_crop_offset_multiplier",
    "surfemb_zbuffer_backend", "surfemb_min_train_render_iou", "batch_size",
    "val_batch_size", "num_workers", "max_iter", "val_freq", "ckpt_freq",
    "validate_before_train", "amp", "grad_clip", "surfemb_emb_dim",
    "surfemb_mlp_hidden_features", "surfemb_mlp_hidden_layers", "resnet_feat_preultimate",
    "lr_cnn", "lr_surfemb_mlp", "weight_decay", "warmup_steps",
)


def portable(value):
    if isinstance(value, str):
        return value.replace(str(ROOT) + "/", "")
    if isinstance(value, list):
        return [portable(v) for v in value]
    if isinstance(value, dict):
        return {portable(k): portable(v) for k, v in value.items()}
    return value


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.write_text(json.dumps(portable(value), indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "logs/surfemb_experiment_summary_20260908")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/experiment_memory")
    args = parser.parse_args()
    audit = read_json(args.source / "audit.json")
    if audit["csv_exceptions"]:
        raise ValueError("The evaluation inventory contains unresolved CSV mismatches.")
    tables = {}
    for name in CSV_FILES:
        with (args.source / name).open(newline="") as handle:
            reader = csv.DictReader(handle)
            tables[name] = (reader.fieldnames, list(reader))
    if len(tables[CSV_FILES[0]][1]) != audit["lnd_rows"]:
        raise ValueError("LND record count differs from source audit.")
    if len(tables[CSV_FILES[1]][1]) != audit["rarp_rows"]:
        raise ValueError("RARP record count differs from source audit.")

    checkpoint_meta = read_json(args.source / "checkpoint_metadata.json")
    recipes = {}
    for path, meta in checkpoint_meta.items():
        if "surfemb_debug_current_lnd_augfix_" not in path:
            continue
        run_args = meta.get("args", {})
        recipes[path] = {
            "iter_at_20260908_inventory": meta.get("iter_now"),
            "args": {key: run_args[key] for key in RECIPE_FIELDS if key in run_args},
        }

    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    import torch

    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total", "--format=csv"],
        capture_output=True, text=True, timeout=15, check=True,
    )
    environment = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Current local runtime, not a clean-install lockfile or historical package proof.",
        "python": platform.python_version(),
        "executable": sys.executable,
        "os_release": platform.freedesktop_os_release(),
        "torch_cuda_build": torch.version.cuda,
        "packages": versions,
        "gpus": list(csv.DictReader(gpu.stdout.splitlines(), skipinitialspace=True)),
    }
    delta = read_json(ROOT / "logs/lnd_refine_delta_summary/summary.json")
    delta_stats = {key: value for key, value in delta.items() if key.endswith("_delta_deg") or key == "trans_delta_mm"}
    args.output.mkdir(parents=True, exist_ok=True)
    for name, (fields, rows) in tables.items():
        with (args.output / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(portable(rows))
    write_json(args.output / "environment_snapshot.json", environment)
    write_json(args.output / "augfix_checkpoint_recipes.json", recipes)
    write_json(args.output / "source_audit.json", audit)
    write_json(args.output / "lnd_refine_delta_summary.json", delta_stats)
    print(json.dumps({
        "output": str(args.output),
        "csv_rows": {name: len(rows) for name, (_, rows) in tables.items()},
        "augfix_checkpoints": len(recipes),
        "copied_images_or_weights": False,
    }, indent=2))


if __name__ == "__main__":
    main()
