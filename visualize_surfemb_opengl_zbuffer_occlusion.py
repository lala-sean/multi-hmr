import argparse
import importlib.util
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

ROBOPEPP_ROOT = Path(__file__).resolve().parent
if str(ROBOPEPP_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBOPEPP_ROOT))


def load_local_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dataset_mod = load_local_module(
    "robopepp_surfemb_keypoint_crop_zbuf_vis",
    ROBOPEPP_ROOT / "datasets" / "surfemb_keypoint_crop.py",
)

from instrument_geometry import fk_matrices_np, project_points_np  # noqa: E402
from instrument_opengl_renderer import InstrumentOpenGLDepthRenderer  # noqa: E402


DEFAULT_SURFACE_POINTS = (
    ROBOPEPP_ROOT
    / "assets"
    / "instrument_surface_samples_surfemb_x2.13mm_wg1over3_shafttop30mm"
    / "instrument_surface_points_all.npy"
)
DEFAULT_LND_ROOT = "/mnt/iMVR/daiyun/Dataset/LND"
DEFAULT_LND_MEMORY = (
    "/mnt/iMVR/daiyun/shuojue-temp/code/multi-hmr/submodules/gaussian-mesh-splatting/"
    "Results2/surgripe_lnd_refine_memory_train/TRAIN/refine_memory_pool.json"
)
PUNCTURE_DATASET_ROOT = "/mnt/nas/share/shuojue/data/needlePuncture_videos"
PUNCTURE_POSE_ROOT = "/mnt/nas/share/shuojue/data/needlePuncture_results"

PARTS = ("shaft", "wrist", "l_gripper", "r_gripper")
PART_COLORS = {
    0: (0, 0, 0),
    1: (230, 80, 80),
    2: (80, 220, 120),
    3: (80, 150, 240),
}
STATUS_COLORS = {
    "visible": (40, 240, 80),
    "mesh_occluded": (255, 60, 60),
    "sample_occluded": (255, 180, 40),
    "backface": (80, 170, 255),
    "missing_depth": (220, 80, 255),
}


def as_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def make_rarp(split, training, args):
    base = dataset_mod.RoboPEPPRARPInstrument(
        PUNCTURE_DATASET_ROOT,
        PUNCTURE_POSE_ROOT,
        split=split,
        training=False,
        crop_size=args.crop_size,
        train_ratio=0.95,
        subsample=args.rarp_subsample,
        min_dice=(0.8, 0.6, 0.6),
        canonicalize_pose_symmetry=True,
        canonical_eps=0.08,
        heatmap_sigma=2.0,
        bbox_padding_frac=0.12,
        color_jitter=False,
        rgb_augmentation=False,
        occlusion_augmentation=False,
        occlusion_prob=0.0,
        bbox_jitter=False,
        bbox_shift=False,
        cache_dir=str(Path(args.out_dir) / "dataset_cache"),
    )
    return dataset_mod.SurfEmbKeypointCropDataset(
        base,
        surface_points_path=args.surface_points_path,
        crop_size=args.crop_size,
        n_pos=args.n_pos,
        n_neg=args.n_neg,
        crop_scale=args.crop_scale,
        max_angle=args.max_angle,
        offset_scale=args.offset_scale,
        training=training,
        heatmap_sigma=2.0,
        min_depth=args.min_depth,
        depth_tolerance=args.depth_tolerance,
        use_mesh_zbuffer=True,
        zbuffer_backend="opengl",
    )


def make_lnd(split, training, use_memory_pose, args):
    base = dataset_mod.RoboPEPPSurgripeLNDInstrument(
        root=args.lnd_root,
        split=split,
        training=False,
        crop_size=args.crop_size,
        memory_path=args.lnd_memory if use_memory_pose else None,
        use_memory_pose=use_memory_pose,
        canonicalize_pose_symmetry=True,
        canonical_eps=0.08,
        heatmap_sigma=2.0,
        bbox_padding_frac=0.12,
        color_jitter=False,
        rgb_augmentation=False,
        occlusion_augmentation=False,
        occlusion_prob=0.0,
        bbox_jitter=False,
        bbox_shift=False,
        subsample=args.lnd_subsample,
    )
    return dataset_mod.SurfEmbKeypointCropDataset(
        base,
        surface_points_path=args.surface_points_path,
        crop_size=args.crop_size,
        n_pos=args.n_pos,
        n_neg=args.n_neg,
        crop_scale=args.crop_scale,
        max_angle=args.max_angle,
        offset_scale=args.offset_scale,
        training=training,
        heatmap_sigma=2.0,
        min_depth=args.min_depth,
        depth_tolerance=args.depth_tolerance,
        use_mesh_zbuffer=True,
        zbuffer_backend="opengl",
    )


def draw_label(img, lines):
    out = img.copy()
    h = 18 * len(lines) + 8
    cv2.rectangle(out, (0, 0), (out.shape[1] - 1, h), (0, 0, 0), -1)
    for i, line in enumerate(lines):
        cv2.putText(
            out,
            str(line),
            (6, 17 + 18 * i),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return out


def overlay_mask(rgb, mask, color, alpha=0.55):
    out = rgb.copy()
    color_arr = np.asarray(color, dtype=np.float32)
    m = np.asarray(mask).astype(bool)
    out[m] = (out[m].astype(np.float32) * (1.0 - alpha) + color_arr * alpha).clip(0, 255).astype(np.uint8)
    return out


def part_mask_rgb(part_mask):
    out = np.zeros((*part_mask.shape, 3), dtype=np.uint8)
    for part_id, color in PART_COLORS.items():
        out[part_mask == part_id] = np.asarray(color, dtype=np.uint8)
    return out


def draw_points(base, u, v, mask, color, rng, max_points=25000, radius=1):
    out = base.copy()
    idx = np.flatnonzero(mask)
    if len(idx) > int(max_points):
        idx = rng.choice(idx, int(max_points), replace=False)
    color_bgr = tuple(int(c) for c in color[::-1])
    out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    for i in idx:
        cv2.circle(out_bgr, (int(u[i]), int(v[i])), int(radius), color_bgr, -1, cv2.LINE_AA)
    return cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)


def classify_points(ds, target, renderer):
    pose = ds._pose_from_target(target)
    K_orig = as_numpy(target["K_orig"]).astype(np.float64)
    M_crop = as_numpy(target["surfemb_M_crop"]).astype(np.float32)
    rgb = as_numpy(target["crop_rgb"]).astype(np.uint8)
    inst = as_numpy(target["inst_mask"]).astype(np.float32) > 0
    part = as_numpy(target["part_mask"]).astype(np.uint8)
    h, w = inst.shape
    orig_rgb = as_numpy(target["orig_rgb"]).astype(np.uint8)
    orig_h, orig_w = orig_rgb.shape[:2]
    transforms = fk_matrices_np(pose["rot"], pose["trans"], pose["alpha"], pose["theta_l"], pose["theta_r"])

    points_cam_all = []
    normals_cam_all = []
    part_name_all = []
    for part_name in PARTS:
        payload = ds.surface_payloads[part_name]
        points_cam, normals_cam = dataset_mod._transform_surface_payload(part_name, payload, transforms)
        points_cam_all.append(points_cam)
        normals_cam_all.append(normals_cam)
        part_name_all.extend([part_name] * len(points_cam))
    points_cam = np.concatenate(points_cam_all, axis=0)
    normals_cam = np.concatenate(normals_cam_all, axis=0)
    part_name_all = np.asarray(part_name_all)

    z = points_cam[:, 2]
    uv_orig = project_points_np(points_cam, K_orig)
    uv_crop = dataset_mod._surf_aug.transform_points(uv_orig, M_crop)
    finite_orig = np.isfinite(uv_orig).all(axis=1)
    finite_crop = np.isfinite(uv_crop).all(axis=1)
    u_orig = np.rint(np.nan_to_num(uv_orig[:, 0], nan=-1e9)).astype(np.int64)
    v_orig = np.rint(np.nan_to_num(uv_orig[:, 1], nan=-1e9)).astype(np.int64)
    u = np.rint(np.nan_to_num(uv_crop[:, 0], nan=-1e9)).astype(np.int64)
    v = np.rint(np.nan_to_num(uv_crop[:, 1], nan=-1e9)).astype(np.int64)
    in_orig = (
        finite_orig
        & (z > ds.min_depth)
        & (u_orig >= 0)
        & (u_orig < orig_w)
        & (v_orig >= 0)
        & (v_orig < orig_h)
    )
    in_img = in_orig & finite_crop & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    pix = np.zeros((len(points_cam),), dtype=np.int64)
    pix[in_img] = v[in_img] * w + u[in_img]
    pix_orig = np.zeros((len(points_cam),), dtype=np.int64)
    pix_orig[in_orig] = v_orig[in_orig] * orig_w + u_orig[in_orig]
    in_inst = np.zeros((len(points_cam),), dtype=bool)
    in_inst[in_img] = inst[v[in_img], u[in_img]]
    front = np.sum(normals_cam * points_cam, axis=1) < 0.0
    candidate = in_img & in_inst & front
    backface = in_img & in_inst & ~front

    _, mesh_depth_orig, _ = renderer.render_pose(pose, K_orig, (orig_h, orig_w))
    mesh_part_orig = None
    if hasattr(renderer, "render_pose_mask"):
        mesh_part_orig = renderer.render_pose_mask(pose, K_orig, (orig_h, orig_w))
    mesh_depth = cv2.warpAffine(
        (mesh_depth_orig > 0.0).astype(np.uint8),
        M_crop,
        (int(w), int(h)),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(np.float32)
    if mesh_part_orig is not None:
        mesh_part = cv2.warpAffine(
            mesh_part_orig.astype(np.uint8),
            M_crop,
            (int(w), int(h)),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).astype(np.uint8)
    else:
        mesh_part = mesh_depth.astype(np.uint8)
    depth_flat = mesh_depth_orig.reshape(-1).astype(np.float64)
    depth_at = np.zeros((len(points_cam),), dtype=np.float64)
    depth_at[in_orig] = depth_flat[pix_orig[in_orig]]
    mesh_hit = np.zeros((len(points_cam),), dtype=bool)
    mesh_hit[in_img] = depth_at[in_img] > ds.min_depth

    mesh_visible = candidate & mesh_hit & (z <= depth_at + ds.depth_tolerance)
    mesh_occluded = candidate & mesh_hit & (z > depth_at + ds.depth_tolerance)
    missing_depth = candidate & ~mesh_hit

    zbuf = np.full((h * w,), np.inf, dtype=np.float64)
    if mesh_visible.any():
        np.minimum.at(zbuf, pix[mesh_visible], z[mesh_visible])
    final_visible = np.zeros((len(points_cam),), dtype=bool)
    final_visible[mesh_visible] = z[mesh_visible] <= zbuf[pix[mesh_visible]] + ds.depth_tolerance
    sample_occluded = mesh_visible & ~final_visible

    return {
        "rgb": rgb,
        "inst": inst,
        "part": part,
        "mesh_depth": mesh_depth,
        "mesh_part": mesh_part,
        "u": u,
        "v": v,
        "z": z,
        "depth_at": depth_at,
        "part_name": part_name_all,
        "in_img": in_img,
        "in_inst": in_inst,
        "front": front,
        "candidate": candidate,
        "final_visible": final_visible,
        "mesh_occluded": mesh_occluded,
        "sample_occluded": sample_occluded,
        "backface": backface,
        "missing_depth": missing_depth,
    }


def make_depth_diff_panel(stats, rng, max_points):
    rgb = stats["rgb"]
    out = np.zeros_like(rgb)
    mask = (stats["candidate"] | stats["mesh_occluded"] | stats["final_visible"]) & (stats["depth_at"] > 0)
    idx = np.flatnonzero(mask)
    if len(idx) > int(max_points):
        idx = rng.choice(idx, int(max_points), replace=False)
    diff_mm = (stats["z"][idx] - stats["depth_at"][idx]) * 1000.0
    mag = np.clip((diff_mm + 2.0) / 18.0, 0.0, 1.0)
    colors = cv2.applyColorMap((mag * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
    out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    for point_idx, color in zip(idx, colors.reshape(-1, 3)):
        cv2.circle(out_bgr, (int(stats["u"][point_idx]), int(stats["v"][point_idx])), 1, tuple(int(c) for c in color), -1)
    out = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
    if len(diff_mm):
        label = f"z-depth mm: p50={np.median(diff_mm):.2f} p95={np.percentile(diff_mm, 95):.2f}"
    else:
        label = "z-depth mm: no candidate"
    return draw_label(out, [label, "blue/green near visible, red means behind mesh"])


def make_panel(stats, label, rng, max_points):
    rgb = stats["rgb"]
    part_rgb = part_mask_rgb(stats["part"])
    depth_mask = stats["mesh_depth"] > 0
    mesh_part_rgb = part_mask_rgb(stats["mesh_part"])
    depth_overlay = rgb.copy()
    depth_overlay[depth_mask] = (
        0.45 * depth_overlay[depth_mask].astype(np.float32)
        + 0.55 * mesh_part_rgb[depth_mask].astype(np.float32)
    ).astype(np.uint8)

    status_rgb = rgb.copy()
    status_black = np.zeros_like(rgb)
    draw_order = [
        ("missing_depth", STATUS_COLORS["missing_depth"]),
        ("backface", STATUS_COLORS["backface"]),
        ("mesh_occluded", STATUS_COLORS["mesh_occluded"]),
        ("sample_occluded", STATUS_COLORS["sample_occluded"]),
        ("final_visible", STATUS_COLORS["visible"]),
    ]
    for key, color in draw_order:
        status_rgb = draw_points(status_rgb, stats["u"], stats["v"], stats[key], np.asarray(color), rng, max_points=max_points, radius=1)
        status_black = draw_points(status_black, stats["u"], stats["v"], stats[key], np.asarray(color), rng, max_points=max_points, radius=1)

    counts = {key: int(np.count_nonzero(stats[key])) for key, _ in draw_order}
    cand = int(np.count_nonzero(stats["candidate"]))
    back = counts["backface"]
    lines = [
        label,
        f"cand={cand} vis={counts['final_visible']} mesh_occ={counts['mesh_occluded']}",
        f"sample_occ={counts['sample_occluded']} backface={back} no_depth={counts['missing_depth']}",
    ]
    legend = "green vis | red mesh-occ | orange sample-occ | blue back | magenta no-depth"

    panels = [
        draw_label(rgb, [label, "crop rgb"]),
        draw_label(part_rgb, ["part mask", "red grip green wrist blue shaft"]),
        draw_label(depth_overlay, ["OpenGL part z-buffer", f"depth_px={int(depth_mask.sum())}"]),
        draw_label(status_rgb, lines),
        draw_label(status_black, [legend]),
        make_depth_diff_panel(stats, rng, max_points=max_points),
    ]
    return np.concatenate(panels, axis=1)


def make_contact_sheet(paths, out_path, cols=1):
    images = [Image.open(p).convert("RGB") for p in paths]
    widths = [im.width for im in images]
    heights = [im.height for im in images]
    cols = max(1, int(cols))
    rows = (len(images) + cols - 1) // cols
    cell_w = max(widths)
    cell_h = max(heights)
    sheet = Image.new("RGB", (cell_w * cols, cell_h * rows), (25, 25, 25))
    for i, im in enumerate(images):
        sheet.paste(im, ((i % cols) * cell_w, (i // cols) * cell_h))
    sheet.save(out_path, quality=92)


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_sets = [
        ("rarp_train", make_rarp("train", bool(args.training_crop), args)),
        ("rarp_val", make_rarp("test", False, args)),
        ("lnd_train", make_lnd("TRAIN", bool(args.training_crop), True, args)),
        ("lnd_val", make_lnd("TEST", False, False, args)),
    ]
    renderer = InstrumentOpenGLDepthRenderer(args.crop_size, args.crop_size)
    rng = np.random.default_rng(args.seed + 17)
    written = []
    for set_name, ds in sample_sets:
        print(ds, flush=True)
        n = min(int(args.samples_per_set), len(ds))
        for idx in range(n):
            _, target = ds[idx]
            stats = classify_points(ds, target, renderer)
            frame_id = str(target.get("frame_id", idx))
            if hasattr(frame_id, "item"):
                frame_id = str(frame_id.item())
            label = f"{set_name}_{idx:02d}_frame{frame_id}"
            panel = make_panel(stats, label, rng, int(args.max_points_per_class))
            out_path = out_dir / f"{label}_opengl_zbuffer_occlusion.jpg"
            Image.fromarray(panel).save(out_path, quality=92)
            written.append(out_path)
            print(
                f"WROTE {out_path} "
                f"visible={int(stats['final_visible'].sum())} "
                f"mesh_occluded={int(stats['mesh_occluded'].sum())} "
                f"sample_occluded={int(stats['sample_occluded'].sum())} "
                f"backface={int(stats['backface'].sum())}",
                flush=True,
            )
    make_contact_sheet(written, out_dir / "contact_sheet.jpg", cols=1)
    print(f"CONTACT_SHEET {out_dir / 'contact_sheet.jpg'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default=str(ROBOPEPP_ROOT / "logs" / "surfemb_opengl_zbuffer_occlusion_vis"))
    parser.add_argument("--surface_points_path", type=str, default=str(DEFAULT_SURFACE_POINTS))
    parser.add_argument("--lnd_root", type=str, default=DEFAULT_LND_ROOT)
    parser.add_argument("--lnd_memory", type=str, default=DEFAULT_LND_MEMORY)
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--n_pos", type=int, default=1024)
    parser.add_argument("--n_neg", type=int, default=1024)
    parser.add_argument("--crop_scale", type=float, default=1.2)
    parser.add_argument("--max_angle", type=float, default=np.pi)
    parser.add_argument("--offset_scale", type=float, default=1.0)
    parser.add_argument("--min_depth", type=float, default=1e-4)
    parser.add_argument("--depth_tolerance", type=float, default=8e-4)
    parser.add_argument("--rarp_subsample", type=int, default=500)
    parser.add_argument("--lnd_subsample", type=int, default=200)
    parser.add_argument("--samples_per_set", type=int, default=2)
    parser.add_argument("--training_crop", type=int, default=1, choices=[0, 1])
    parser.add_argument("--max_points_per_class", type=int, default=25000)
    parser.add_argument("--seed", type=int, default=7)
    main(parser.parse_args())
