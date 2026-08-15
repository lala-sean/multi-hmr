import argparse
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import visualize_surfemb_opengl_zbuffer_occlusion as base


ROBOPEPP_ROOT = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = ROBOPEPP_ROOT / "logs" / "surfemb_wrist_opengl_zbuffer_occlusion_vis"
WRIST_PART_ID = 2
VISIBLE_COLOR = np.array((40, 240, 80), dtype=np.uint8)
OCCLUDED_COLOR = np.array((255, 60, 60), dtype=np.uint8)


def effective_part_ids(dataset):
    ids = []
    static = []
    for part_name in base.PARTS:
        payload = dataset.surface_payloads[part_name]
        n_points = len(payload["points_part_m"])
        fallback_id = {"shaft": 3, "wrist": 2, "l_gripper": 1, "r_gripper": 1}[part_name]
        ids.append(
            np.asarray(
                payload.get("effective_part_ids", np.full((n_points,), fallback_id, dtype=np.int64)),
                dtype=np.int64,
            )
        )
        static.append(
            np.asarray(
                payload.get("static_wrist_mask", np.zeros((n_points,), dtype=bool)),
                dtype=bool,
            )
        )
    return np.concatenate(ids), np.concatenate(static)


def classify_wrist_points(dataset, target, renderer):
    stats = base.classify_points(dataset, target, renderer)
    part_ids, static_mask = effective_part_ids(dataset)
    if len(part_ids) != len(stats["z"]):
        raise RuntimeError(
            f"Surface metadata length mismatch: effective_part_ids={len(part_ids)} points={len(stats['z'])}"
        )

    wrist = part_ids == WRIST_PART_ID
    visible = wrist & stats["final_visible"]
    mesh_occluded = wrist & stats["mesh_occluded"]
    sample_occluded = wrist & stats["sample_occluded"]
    occluded = mesh_occluded | sample_occluded
    return {
        **stats,
        "wrist": wrist,
        "wrist_static_gripper": wrist & static_mask,
        "wrist_visible": visible,
        "wrist_mesh_occluded": mesh_occluded,
        "wrist_sample_occluded": sample_occluded,
        "wrist_occluded": occluded,
    }


def draw_wrist_points(image, stats, rng, *, visible=True, occluded=True, max_points=30000):
    out = image.copy()
    if occluded:
        out = base.draw_points(
            out,
            stats["u"],
            stats["v"],
            stats["wrist_occluded"],
            OCCLUDED_COLOR,
            rng,
            max_points=max_points,
            radius=1,
        )
    if visible:
        out = base.draw_points(
            out,
            stats["u"],
            stats["v"],
            stats["wrist_visible"],
            VISIBLE_COLOR,
            rng,
            max_points=max_points,
            radius=1,
        )
    return out


def make_panel(stats, label, rng, max_points):
    rgb = stats["rgb"]
    black = np.zeros_like(rgb)
    visible_count = int(stats["wrist_visible"].sum())
    mesh_occluded_count = int(stats["wrist_mesh_occluded"].sum())
    sample_occluded_count = int(stats["wrist_sample_occluded"].sum())
    occluded_count = int(stats["wrist_occluded"].sum())
    classified_count = visible_count + occluded_count
    visible_ratio = visible_count / max(1, classified_count)
    static_count = int(
        (
            stats["wrist_static_gripper"]
            & (stats["wrist_visible"] | stats["wrist_occluded"])
        ).sum()
    )

    overlay = draw_wrist_points(
        rgb,
        stats,
        rng,
        visible=True,
        occluded=True,
        max_points=max_points,
    )
    occluded_only = draw_wrist_points(
        rgb,
        stats,
        rng,
        visible=False,
        occluded=True,
        max_points=max_points,
    )
    status_black = draw_wrist_points(
        black,
        stats,
        rng,
        visible=True,
        occluded=True,
        max_points=max_points,
    )

    panels = [
        base.draw_label(rgb, [label, "crop rgb"]),
        base.draw_label(
            overlay,
            [
                "effective wrist points",
                f"green visible={visible_count}",
                f"red occluded={occluded_count}",
            ],
        ),
        base.draw_label(
            occluded_only,
            [
                "occluded wrist only",
                f"mesh={mesh_occluded_count}",
                f"sample={sample_occluded_count}",
            ],
        ),
        base.draw_label(
            status_black,
            [
                f"visible ratio={visible_ratio:.3f}",
                f"classified static rear={static_count}",
                "no shaft/gripper points shown",
            ],
        ),
    ]
    return np.concatenate(panels, axis=1)


def make_datasets(args):
    return [
        ("rarp_train", base.make_rarp("train", bool(args.training_crop), args)),
        ("rarp_val", base.make_rarp("test", False, args)),
        ("lnd_train", base.make_lnd("TRAIN", bool(args.training_crop), True, args)),
        ("lnd_val", base.make_lnd("TEST", False, False, args)),
    ]


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    renderer = base.InstrumentOpenGLDepthRenderer(args.crop_size, args.crop_size)
    rng = np.random.default_rng(args.seed + 17)
    written = []
    for set_name, dataset in make_datasets(args):
        print(dataset, flush=True)
        n_samples = min(int(args.samples_per_set), len(dataset))
        for index in range(n_samples):
            _, target = dataset[index]
            stats = classify_wrist_points(dataset, target, renderer)
            frame_id = target.get("frame_id", index)
            if hasattr(frame_id, "item"):
                frame_id = frame_id.item()
            label = f"{set_name}_{index:02d}_frame{frame_id}"
            panel = make_panel(stats, label, rng, int(args.max_points_per_class))
            out_path = out_dir / f"{label}_wrist_opengl_zbuffer_occlusion.jpg"
            Image.fromarray(panel).save(out_path, quality=92)
            written.append(out_path)
            print(
                f"WROTE {out_path} "
                f"visible={int(stats['wrist_visible'].sum())} "
                f"mesh_occluded={int(stats['wrist_mesh_occluded'].sum())} "
                f"sample_occluded={int(stats['wrist_sample_occluded'].sum())} "
                f"classified_static_rear={int((stats['wrist_static_gripper'] & (stats['wrist_visible'] | stats['wrist_occluded'])).sum())}",
                flush=True,
            )

    contact_sheet = out_dir / "contact_sheet.jpg"
    base.make_contact_sheet(written, contact_sheet, cols=1)
    print(f"CONTACT_SHEET {contact_sheet}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--surface_points_path", type=str, default=str(base.DEFAULT_SURFACE_POINTS))
    parser.add_argument("--lnd_root", type=str, default=base.DEFAULT_LND_ROOT)
    parser.add_argument("--lnd_memory", type=str, default=base.DEFAULT_LND_MEMORY)
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--n_pos", type=int, default=1024)
    parser.add_argument("--n_neg", type=int, default=1024)
    parser.add_argument("--crop_scale", type=float, default=1.2)
    parser.add_argument("--max_angle", type=float, default=np.pi)
    parser.add_argument("--offset_scale", type=float, default=1.0)
    parser.add_argument("--min_depth", type=float, default=1e-4)
    parser.add_argument("--depth_tolerance", type=float, default=8e-4)
    parser.add_argument("--rarp_subsample", type=int, default=137)
    parser.add_argument("--lnd_subsample", type=int, default=97)
    parser.add_argument("--samples_per_set", type=int, default=3)
    parser.add_argument("--training_crop", type=int, default=0, choices=[0, 1])
    parser.add_argument("--max_points_per_class", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=21)
    main(parser.parse_args())
