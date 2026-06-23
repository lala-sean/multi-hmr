import json
import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("EGL_DEVICE_ID", "7")

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
from PIL import Image

from estimate_rarp_cse_articulate_pose import CAD_ROOT, InstrumentCAD, RARP_FC
from eval_suturepulling_hcce_unified_debug import (
    add_title,
    estimate_pose,
    load_model_unified,
    overlay_part_mask,
    pad_rgb_to_square,
    pred_part_label,
    rasterize_pose_part_mask,
    render_mesh_overlay_direct,
)
from multi_instrument import Multi_Instrument
from utils import normalize_rgb


DENSEPART_H2_BEST = (
    "logs/instrument_rarp_hcce_densepart_dpt_h2_sym_poseheads_2gpu_bs6_part05_gpu56"
    "/checkpoints/best.pt"
)
SURGPOSE_PROCESSED_ROOT = Path(
    "/mnt/nas/share/shuojue/data/surgpose/000000/processed_stereo_640"
)
SURGPOSE_LEFT_FRAMES = SURGPOSE_PROCESSED_ROOT / "left_frames"
SURGPOSE_SAM3_SEGMENTATION = SURGPOSE_PROCESSED_ROOT / "sam3_segmentation"
SURGPOSE_METADATA = SURGPOSE_PROCESSED_ROOT / "metadata.json"


def _namespace_to_dict(value):
    if value is None:
        return {}
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    if isinstance(value, dict):
        return dict(value)
    raise TypeError(f"Unsupported checkpoint args type: {type(value)}")


def _cuda_index(device):
    if isinstance(device, torch.device):
        device = str(device)
    if not str(device).startswith("cuda"):
        return None
    parts = str(device).split(":", 1)
    return int(parts[1]) if len(parts) == 2 and parts[1] else 0


def configure_device(device):
    cuda_idx = _cuda_index(device)
    if cuda_idx is not None:
        os.environ["EGL_DEVICE_ID"] = str(cuda_idx)
        torch.cuda.set_device(cuda_idx)
    return torch.device(device)


def load_camera_K(args):
    if args.camera_metadata:
        meta_path = Path(args.camera_metadata)
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if args.camera_key in meta:
                return np.asarray(meta[args.camera_key], dtype=np.float64)
    return None


def preprocess(img_path, img_size, focal, cx, cy, K_override=None):
    img_pil = Image.open(img_path).convert("RGB")
    rgb_orig = np.asarray(img_pil)
    img_h, img_w = rgb_orig.shape[:2]
    scale = img_size / float(max(img_w, img_h))
    new_w = int(img_w * scale)
    new_h = int(img_h * scale)
    pad_x = (img_size - new_w) // 2
    pad_y = (img_size - new_h) // 2

    resized = img_pil.resize((new_w, new_h), Image.BILINEAR)
    padded = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = np.asarray(resized, dtype=np.uint8)
    img_tensor = torch.from_numpy(normalize_rgb(padded)).unsqueeze(0)

    if K_override is not None:
        K = np.asarray(K_override, dtype=np.float64).copy()
    else:
        K = np.array(
            [
                [float(focal), 0.0, img_w / 2.0 if cx is None else float(cx)],
                [0.0, float(focal), img_h / 2.0 if cy is None else float(cy)],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
    geom = (rgb_orig, K, scale, pad_x, pad_y)
    return img_tensor, padded, geom


def _seg_model_kwargs(args, ckpt_args):
    src = _namespace_to_dict(ckpt_args)
    keys = (
        "backbone",
        "xat_dim",
        "xat_depth",
        "xat_heads",
        "xat_dim_head",
        "xat_mlp_dim",
        "mask_dim",
        "num_parts",
        "img_size",
    )
    out = {key: getattr(args, key) for key in keys}
    for key in keys:
        if key in src:
            out[key] = src[key]
    out["pretrained_backbone"] = False
    return out


def load_demo_model(args, device):
    checkpoint_path = Path(args.pretrained)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"]
    has_hcce = (
        any(k.startswith("dense_dpt_head.") for k in state)
        or any(k.startswith("hcce_dpt_head.") for k in state)
    )
    if has_hcce:
        model, meta = load_model_unified(checkpoint_path, device)
        meta["has_hcce"] = True
        return model, meta

    model = Multi_Instrument(**_seg_model_kwargs(args, ckpt.get("args"))).to(device)
    log = model.load_state_dict(state, strict=False)
    print(f"[seg-only] load_state_dict: {log}", flush=True)
    model.eval()
    return model, {
        "variant": "seg_only",
        "has_hcce": False,
        "checkpoint_iter": int(ckpt.get("iter", -1)),
        "checkpoint_epoch": int(ckpt.get("epoch", -1)),
        "img_size": int(getattr(args, "img_size", 630)),
    }


def normalize_predictions(output):
    if isinstance(output, tuple):
        output = output[-1]
    if isinstance(output, dict) or output is None:
        return []
    return [p for p in output if int(p.get("batch_idx", 0)) == 0]


def external_mask_path_for_image(img_path, args):
    if not args.external_inst_dir:
        return None
    root = Path(args.external_inst_dir)
    candidates = [
        root / Path(img_path).name,
        root / f"{Path(img_path).stem}.png",
        root / f"{Path(img_path).stem}.jpg",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return candidates[0]


def pad_original_mask(mask, img_size, scale, pad_x, pad_y):
    h, w = mask.shape[:2]
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = Image.fromarray(mask.astype(np.uint8)).resize((new_w, new_h), Image.NEAREST)
    padded = np.zeros((img_size, img_size), dtype=np.uint8)
    padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = np.asarray(resized, dtype=np.uint8)
    return padded


def load_external_instances(img_path, rgb_shape, geom, args, patch_size):
    mask_path = external_mask_path_for_image(img_path, args)
    if mask_path is None:
        return None
    if not Path(mask_path).is_file():
        if args.require_external_inst:
            raise FileNotFoundError(f"External instance mask not found: {mask_path}")
        return []

    mask = np.asarray(Image.open(mask_path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.shape[:2] != tuple(rgb_shape[:2]):
        mask = np.asarray(
            Image.fromarray(mask.astype(np.uint8)).resize(
                (int(rgb_shape[1]), int(rgb_shape[0])),
                Image.NEAREST,
            ),
            dtype=np.uint8,
        )

    _, _, scale, pad_x, pad_y = geom
    instances = []
    for label in sorted(int(x) for x in np.unique(mask) if int(x) > 0):
        inst_orig = mask == label
        if int(inst_orig.sum()) < int(args.external_min_area):
            continue
        inst_pad = pad_original_mask(inst_orig.astype(np.uint8), args.img_size, scale, pad_x, pad_y)
        ys, xs = np.where(inst_pad > 0)
        if len(xs) == 0:
            continue
        cx = float(xs.mean())
        cy = float(ys.mean())
        grid_size = args.img_size // int(patch_size)
        patch_x = int(np.clip(cx / float(patch_size), 0, grid_size - 1))
        patch_y = int(np.clip(cy / float(patch_size), 0, grid_size - 1))
        instances.append(
            {
                "label": label,
                "mask_orig": inst_orig.astype(np.uint8),
                "mask_padded": inst_pad.astype(np.uint8),
                "center_xy": (cx, cy),
                "idx_yx": (patch_y, patch_x),
                "mask_path": str(mask_path),
                "area": int(inst_orig.sum()),
            }
        )
    return instances


def external_idx_from_instances(instances, device):
    if not instances:
        return None
    return (
        torch.zeros(len(instances), dtype=torch.long, device=device),
        torch.tensor([inst["idx_yx"][0] for inst in instances], dtype=torch.long, device=device),
        torch.tensor([inst["idx_yx"][1] for inst in instances], dtype=torch.long, device=device),
    )


def apply_external_instances(preds, instances, args):
    if instances is None:
        return preds
    out = []
    for pred, inst in zip(preds, instances):
        pred = dict(pred)
        target_hw = tuple(int(x) for x in pred["part_mask_logits"].shape[-2:])
        mask = np.asarray(
            Image.fromarray(inst["mask_padded"].astype(np.uint8)).resize(
                (target_hw[1], target_hw[0]),
                Image.NEAREST,
            ),
            dtype=np.uint8,
        ) > 0
        logits = torch.full(
            target_hw,
            -20.0,
            dtype=pred["part_mask_logits"].dtype,
            device=pred["part_mask_logits"].device,
        )
        logits[torch.from_numpy(mask).to(logits.device)] = 20.0
        pred["inst_mask_logits"] = logits
        pred["loc"] = torch.tensor(
            inst["center_xy"],
            dtype=pred["part_mask_logits"].dtype,
            device=pred["part_mask_logits"].device,
        )
        pred["external_instance_label"] = int(inst["label"])
        pred["external_instance_area"] = int(inst["area"])
        out.append(pred)
    return out


def draw_wrist_centers(panel, preds):
    out = panel.copy()
    h, w = out.shape[:2]
    for pred in preds:
        loc = pred.get("loc")
        if loc is None or torch.isnan(loc).any():
            continue
        cx, cy = int(round(float(loc[0]))), int(round(float(loc[1])))
        if not (0 <= cx < w and 0 <= cy < h):
            continue
        cv2.drawMarker(
            out,
            (cx, cy),
            (255, 255, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=18,
            thickness=2,
            line_type=cv2.LINE_AA,
        )
    return out


KEYPOINT_COLORS = (
    (255, 0, 255),
    (0, 255, 255),
    (255, 128, 0),
    (0, 255, 0),
    (255, 0, 0),
    (0, 128, 255),
    (180, 80, 255),
)


def draw_keypoints(panel, preds, color=None):
    out = panel.copy()
    h, w = out.shape[:2]
    for pred in preds:
        keypoint = pred.get("keypoint_xy")
        if keypoint is None:
            continue
        if torch.is_tensor(keypoint):
            keypoint = keypoint.detach().cpu().numpy()
        keypoint = np.asarray(keypoint, dtype=np.float64).reshape(-1, 2)
        for j, xy in enumerate(keypoint):
            if not np.isfinite(xy).all():
                continue
            x, y = int(round(float(xy[0]))), int(round(float(xy[1])))
            if not (0 <= x < w and 0 <= y < h):
                continue
            c = KEYPOINT_COLORS[j % len(KEYPOINT_COLORS)] if color is None else color
            cv2.circle(out, (x, y), 5, c, 2, cv2.LINE_AA)
            cv2.drawMarker(
                out,
                (x, y),
                c,
                markerType=cv2.MARKER_CROSS,
                markerSize=14,
                thickness=2,
                line_type=cv2.LINE_AA,
            )
    return out


def model_seg_panel(img_np, preds, args):
    canvas = img_np.copy()
    for pred in preds:
        part_label, _ = pred_part_label(pred, args)
        part_sq = np.asarray(
            Image.fromarray(part_label).resize((args.img_size, args.img_size), Image.NEAREST),
            dtype=np.uint8,
        )
        canvas = overlay_part_mask(canvas, part_sq, alpha=args.alpha)
    return draw_keypoints(draw_wrist_centers(canvas, preds), preds)


def recover_hcce_poses(preds, cad, geom, model_meta, args, image_seed):
    poses = []
    rows = []
    for pred_idx, pred in enumerate(preds):
        if "hcce_logits" not in pred:
            rows.append(
                {
                    "pred_idx": pred_idx,
                    "status": "missing_hcce_logits",
                }
            )
            continue
        rng = np.random.default_rng(int(image_seed) * 1009 + pred_idx)
        try:
            pose, _, debug = estimate_pose(pred, cad, geom, model_meta, args, rng)
            poses.append(pose)
            rows.append(
                {
                    "pred_idx": pred_idx,
                    "status": "ok",
                    **debug,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "pred_idx": pred_idx,
                    "status": f"{type(exc).__name__}: {exc}",
                }
            )
    return poses, rows


def hcce_panels(geom, cad, poses, args):
    rgb_orig, K, scale, pad_x, pad_y = geom
    mesh_rgb = rgb_orig.copy()
    proj_mask = np.zeros(rgb_orig.shape[:2], dtype=np.uint8)
    for pose in poses:
        mesh_rgb = render_mesh_overlay_direct(mesh_rgb, cad, pose, K, args)
        part_mask = rasterize_pose_part_mask(
            cad, pose, K, rgb_orig.shape[0], rgb_orig.shape[1], args
        )
        proj_mask[part_mask > 0] = part_mask[part_mask > 0]

    mesh_sq = pad_rgb_to_square(mesh_rgb, args.img_size, scale, pad_x, pad_y)
    rgb_sq = pad_rgb_to_square(rgb_orig, args.img_size, scale, pad_x, pad_y)
    proj_sq = pad_original_mask(proj_mask, args.img_size, scale, pad_x, pad_y)
    proj_panel = overlay_part_mask(rgb_sq, proj_sq, alpha=args.alpha)
    return mesh_sq, proj_panel


def make_visualization(img_np, preds, geom, cad, poses, args, hcce_enabled):
    rgb_panel = add_title(draw_keypoints(img_np, preds), "rgb")
    seg_panel = add_title(model_seg_panel(img_np, preds, args), "model-seg")
    if hcce_enabled:
        mesh_panel, proj_panel = hcce_panels(geom, cad, poses, args)
        mesh_panel = draw_keypoints(mesh_panel, preds)
        proj_panel = draw_keypoints(proj_panel, preds)
        mesh_panel = add_title(mesh_panel, "hcce-trimesh")
        proj_panel = add_title(proj_panel, "hcce-proj-seg")
    else:
        mesh_panel = add_title(img_np.copy(), "hcce unavailable")
        proj_panel = add_title(img_np.copy(), "hcce-proj unavailable")
    sep = np.full((args.img_size, 6, 3), 255, dtype=np.uint8)
    return np.concatenate([rgb_panel, sep, seg_panel, sep, mesh_panel, sep, proj_panel], axis=1)


def image_paths_from_input(input_path, recursive=False, max_images=-1):
    input_path = Path(input_path)
    if input_path.is_dir():
        iterator = input_path.rglob("*") if recursive else input_path.iterdir()
        paths = sorted(
            p for p in iterator
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
    else:
        paths = [input_path]
    if max_images and max_images > 0:
        paths = paths[: int(max_images)]
    return paths


def split_evenly(items, n):
    n = max(1, int(n))
    return [items[i::n] for i in range(n)]


def worker_main(rank, args, img_paths):
    device_name = args.devices[rank]
    device = configure_device(device_name)
    args.device = device_name
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    model, model_meta = load_demo_model(args, device)
    cad = InstrumentCAD(CAD_ROOT) if model_meta.get("has_hcce", False) else None
    camera_K = load_camera_K(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[worker {rank}] device={device_name} images={len(img_paths)} "
        f"variant={model_meta.get('variant')} iter={model_meta.get('checkpoint_iter')}",
        flush=True,
    )
    for local_idx, img_path in enumerate(img_paths):
        img_tensor, img_np, geom = preprocess(
            img_path, args.img_size, args.focal, args.cx, args.cy, K_override=camera_K
        )
        external_instances = load_external_instances(
            img_path,
            geom[0].shape,
            geom,
            args,
            patch_size=getattr(model, "patch_size", 14),
        )
        external_idx = external_idx_from_instances(external_instances, device)
        img_tensor = img_tensor.to(device)
        if external_instances is not None and not external_instances:
            output = []
        else:
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                    output = model(
                        img_tensor,
                        idx=external_idx,
                        is_training=False,
                        det_thresh=args.det_thresh,
                        nms_kernel_size=args.nms_kernel_size,
                    )
        preds = apply_external_instances(
            normalize_predictions(output),
            external_instances,
            args,
        )
        image_seed = args.seed + rank * 100000 + local_idx
        if model_meta.get("has_hcce", False):
            poses, pose_rows = recover_hcce_poses(preds, cad, geom, model_meta, args, image_seed)
        else:
            poses, pose_rows = [], []

        vis = make_visualization(
            img_np,
            preds,
            geom,
            cad,
            poses,
            args,
            hcce_enabled=bool(model_meta.get("has_hcce", False)),
        )
        stem = Path(img_path).stem
        out_path = output_dir / f"{stem}_zeroshot_hcce.jpg"
        Image.fromarray(vis).save(out_path)
        meta = {
            "image": str(img_path),
            "output": str(out_path),
            "device": device_name,
            "checkpoint": str(args.pretrained),
            "model_variant": model_meta.get("variant"),
            "checkpoint_iter": model_meta.get("checkpoint_iter"),
            "num_predictions": len(preds),
            "external_instance_mode": external_instances is not None,
            "external_instance_count": 0 if external_instances is None else len(external_instances),
            "external_instances": [
                {
                    "label": int(inst["label"]),
                    "area": int(inst["area"]),
                    "center_xy": [float(inst["center_xy"][0]), float(inst["center_xy"][1])],
                    "idx_yx": [int(inst["idx_yx"][0]), int(inst["idx_yx"][1])],
                    "mask_path": inst["mask_path"],
                }
                for inst in (external_instances or [])
            ],
            "num_hcce_poses": len(poses),
            "pose_recovery": pose_rows,
            "camera": {
                "focal": args.focal,
                "cx": args.cx,
                "cy": args.cy,
                "K": geom[1].tolist(),
            },
            "point_select": args.point_select,
            "max_points_per_part": args.max_points_per_part,
        }
        (output_dir / f"{stem}_zeroshot_hcce.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        print(
            f"[worker {rank}] {Path(img_path).name}: "
            f"{len(preds)} det, {len(poses)} hcce pose -> {out_path.name}",
            flush=True,
        )


def run(args):
    img_paths = image_paths_from_input(args.input, args.recursive, args.max_images)
    if not img_paths:
        raise RuntimeError(f"No images found: {args.input}")
    args.devices = args.devices or [args.device]
    args.devices = [d for d in args.devices if d]
    if not args.devices:
        args.devices = ["cuda:7" if torch.cuda.is_available() else "cpu"]

    print(
        f"Processing {len(img_paths)} image(s) -> {args.output_dir}; "
        f"devices={args.devices}; checkpoint={args.pretrained}",
        flush=True,
    )
    chunks = [chunk for chunk in split_evenly(img_paths, len(args.devices)) if chunk]
    if len(chunks) == 1:
        args.devices = [args.devices[0]]
        worker_main(0, args, chunks[0])
        return

    ctx = mp.get_context("spawn")
    procs = []
    for rank, chunk in enumerate(chunks):
        proc = ctx.Process(target=worker_main, args=(rank, args, chunk))
        proc.start()
        procs.append(proc)
    failed = []
    for proc in procs:
        proc.join()
        if proc.exitcode != 0:
            failed.append(proc.exitcode)
    if failed:
        raise RuntimeError(f"Worker failure exit codes: {failed}")


def add_pose_recovery_args(parser):
    parser.add_argument("--focal", type=float, default=RARP_FC)
    parser.add_argument("--cx", type=float, default=None)
    parser.add_argument("--cy", type=float, default=None)
    parser.add_argument("--inst_thresh", type=float, default=0.5)
    parser.add_argument("--hcce_bits", type=int, default=None)
    parser.add_argument("--hcce_bit_thresh", type=float, default=0.5)
    parser.add_argument("--max_points_per_part", type=int, default=512)
    parser.add_argument("--point_select", choices=["top", "random"], default="top")
    parser.add_argument("--surface_k_faces", type=int, default=0)
    parser.add_argument("--shaft_raw_x_min", type=float, default=-0.5)
    parser.add_argument("--min_wrist_points", type=int, default=12)
    parser.add_argument("--min_total_points", type=int, default=24)
    parser.add_argument("--min_shaft_points", type=int, default=24)
    parser.add_argument("--min_pnp_inliers", type=int, default=8)
    parser.add_argument("--pnp_iters", type=int, default=300)
    parser.add_argument("--pnp_reproj_error", type=float, default=8.0)
    parser.add_argument("--pnp_confidence", type=float, default=0.99)
    parser.add_argument("--freeze_wrist_after_pnp", type=int, default=0, choices=[0, 1])
    parser.add_argument("--optim_strategy", choices=["decoupled", "single"], default="single")
    parser.add_argument("--optim_parts", choices=["wrist_gripper", "all"], default="all")
    parser.add_argument(
        "--optim_loss",
        choices=["linear", "soft_l1", "huber", "cauchy", "arctan"],
        default="soft_l1",
    )
    parser.add_argument("--optim_f_scale", type=float, default=8.0)
    parser.add_argument("--optim_max_nfev", type=int, default=200)
    parser.add_argument("--min_depth", type=float, default=1e-4)
    parser.add_argument("--behind_camera_penalty", type=float, default=1e4)
    parser.add_argument("--draw_margin", type=int, default=64)
    parser.add_argument("--mesh_render_backend", choices=["pyrender", "software"], default="pyrender")
    parser.add_argument("--line_width", type=int, default=1)
    parser.add_argument("--pyrender_znear", type=float, default=0.001)
    parser.add_argument("--pyrender_zfar", type=float, default=0.6)
    parser.add_argument("--pyrender_light_intensity", type=float, default=3.0)
    parser.add_argument("--pyrender_ambient_light", type=float, default=0.35)
    parser.add_argument("--mesh_render_smooth", type=int, default=1)
    parser.add_argument("--mesh_overlay_edge_px", type=int, default=0)
    parser.add_argument("--render_seg_mode", choices=["surface_splat", "hybrid", "triangles"], default="surface_splat")
    parser.add_argument("--render_surface_points_shaft", type=int, default=20000)
    parser.add_argument("--render_surface_points_wrist", type=int, default=6000)
    parser.add_argument("--render_surface_points_gripper", type=int, default=8000)
    parser.add_argument("--render_splat_radius_px", type=int, default=3)
    parser.add_argument("--render_shaft_splat_radius_px", type=int, default=5)
    parser.add_argument("--render_shaft_close_px", type=int, default=0)
    parser.add_argument("--render_shaft_dilate_px", type=int, default=0)
    return parser


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default=str(SURGPOSE_LEFT_FRAMES),
        help="Path to a single image or a folder of PNG/JPG images",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="demo_outputs/demo_output_surgpose_sam3_densepart_h2_top512",
    )
    parser.add_argument("--pretrained", type=str, default=DENSEPART_H2_BEST)
    parser.add_argument("--external_inst_dir", type=str, default=str(SURGPOSE_SAM3_SEGMENTATION))
    parser.add_argument("--require_external_inst", type=int, default=1, choices=[0, 1])
    parser.add_argument("--external_min_area", type=int, default=64)
    parser.add_argument("--camera_metadata", type=str, default=str(SURGPOSE_METADATA))
    parser.add_argument("--camera_key", type=str, default="K_left_rectified_scaled")

    parser.add_argument("--img_size", type=int, default=630)
    parser.add_argument("--det_thresh", type=float, default=0.5)
    parser.add_argument("--nms_kernel_size", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--amp", type=int, default=1, choices=[0, 1])
    parser.add_argument("--device", type=str, default="cuda:7")
    parser.add_argument("--devices", nargs="*", default=["cuda:7", "cuda:1"])
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--max_images", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--backbone",
        type=str,
        default="dinov2_vits14",
        choices=["dinov2_vitl14", "dinov2_vitb14", "dinov2_vits14"],
    )
    parser.add_argument("--xat_dim", type=int, default=512)
    parser.add_argument("--xat_depth", type=int, default=4)
    parser.add_argument("--xat_heads", type=int, default=16)
    parser.add_argument("--xat_dim_head", type=int, default=32)
    parser.add_argument("--xat_mlp_dim", type=int, default=2048)
    parser.add_argument("--mask_dim", type=int, default=256)
    parser.add_argument("--num_parts", type=int, default=4)
    parser = add_pose_recovery_args(parser)

    run(parser.parse_args())
