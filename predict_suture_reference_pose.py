import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
from PIL import Image
import torchvision.transforms as tv_transforms

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

ROBOPEPP_ROOT = Path(__file__).resolve().parent
MULTIHMR_ROOT = ROBOPEPP_ROOT.parents[1]
if str(ROBOPEPP_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBOPEPP_ROOT))
if str(MULTIHMR_ROOT) not in sys.path:
    sys.path.insert(0, str(MULTIHMR_ROOT))

from instrument_geometry import (  # noqa: E402
    crop_resize_pad_intrinsics,
    instrument_keypoints_camera_np,
    project_points_np,
)
from instrument_trimesh_renderer import GMSInstrumentTrimeshRenderer  # noqa: E402
from pose_pnp import pose_from_keypoints_pnp  # noqa: E402
from predict_instrument_pose import (  # noqa: E402
    add_title,
    concat_panels,
    heatmap_argmax,
    load_model,
    overlay_part_mask,
    pad_mask_to_square,
    pad_rgb_to_square,
    square_image_geometry,
)

from datasets.RarpInstanceDataset import RARPInstanceDataset, _resolve_mask_subfolder  # noqa: E402


SUTURE_ROOT = "/mnt/nas/share/shuojue/data/suturePulling_videos"
REF_VIS = (
    "/mnt/iMVR/daiyun/shuojue-temp/code/multi-hmr/eval_outputs/"
    "suturepulling_best55000_ref_current_best_three_exp_full_stride4/"
    "densepart_h2/vis/suturePulling"
)


def _cuda_index(device):
    if not str(device).startswith("cuda"):
        return None
    parts = str(device).split(":", 1)
    return int(parts[1]) if len(parts) == 2 and parts[1] else 0


def configure_device(device):
    cuda_idx = _cuda_index(device)
    if cuda_idx is not None:
        os.environ["EGL_DEVICE_ID"] = str(cuda_idx)
        torch.cuda.set_device(cuda_idx)
    return torch.device(device if torch.cuda.is_available() else "cpu")


def parse_reference_files(ref_dir):
    pattern = re.compile(r"^(suturePulling_\d+_video\d+)_(\d+)_inst(\d+)\.jpg$")
    samples = []
    for path in sorted(Path(ref_dir).glob("*.jpg")):
        match = pattern.match(path.name)
        if not match:
            continue
        video, frame_id, ref_inst = match.groups()
        samples.append(
            {
                "video": video,
                "frame_id": frame_id,
                "ref_inst": int(ref_inst),
                "ref_path": path,
                "name": path.name,
            }
        )
    if not samples:
        raise RuntimeError(f"No reference samples parsed from {ref_dir}")
    return samples


def build_dataset_index(args):
    dataset = RARPInstanceDataset(
        split="test",
        training=False,
        img_size=args.panel_size,
        dataset_root=args.dataset_root,
        pose_root=None,
        min_dice=[args.min_dice_shaft, args.min_dice_wrist, args.min_dice_gripper],
        train_ratio=args.train_ratio,
        subsample=1,
        v2_force=True,
        cse_coord_root=None,
        render_on_the_fly=False,
    )
    return dataset, {(video, frame_id): idx for idx, (video, frame_id, _) in enumerate(dataset.samples)}


def frame_image_path(dataset_root, video, frame_id):
    video_folder = Path(dataset_root) / f"SARRARP502022_{video}"
    frames_dir = video_folder / "frames_v2"
    for ext in ("png", "jpg"):
        path = frames_dir / f"{frame_id}.{ext}"
        if path.is_file():
            return path
    raise FileNotFoundError(f"Frame not found for {video}/{frame_id}")


def load_instance_masks(dataset_root, video, frame_id, actual_instance_id):
    video_folder = Path(dataset_root) / f"SARRARP502022_{video}"
    instance_folder = video_folder / f"instance{actual_instance_id}"
    mask_frame_id = f"{int(frame_id) - 1:05d}"
    part_mask = None
    inst_mask = None
    for part, label in (("shaft", 3), ("wrist", 2), ("gripper", 1)):
        folder = _resolve_mask_subfolder(str(instance_folder), part, True)
        if folder is None:
            continue
        path = Path(folder) / f"{mask_frame_id}.png"
        if not path.is_file():
            continue
        mask = np.asarray(Image.open(path).convert("L")) > 0
        if part_mask is None:
            part_mask = np.zeros(mask.shape, dtype=np.uint8)
            inst_mask = np.zeros(mask.shape, dtype=bool)
        part_mask[mask] = label
        inst_mask |= mask
    if part_mask is None or inst_mask is None or not inst_mask.any():
        raise RuntimeError(f"Empty masks for {video}/{frame_id}/actual_instance{actual_instance_id}")
    return part_mask, inst_mask


def crop_instance(rgb, K_orig, inst_mask, crop_size):
    ys, xs = np.where(inst_mask)
    if len(xs) == 0:
        raise RuntimeError("Empty instance mask")
    h, w = rgb.shape[:2]
    bbox_min = np.array([float(xs.min()), float(ys.min())], dtype=np.float32)
    bbox_max = np.array([float(xs.max() + 1), float(ys.max() + 1)], dtype=np.float32)
    bbox_min = np.clip(bbox_min, [0.0, 0.0], [float(w - 1), float(h - 1)])
    bbox_max = np.clip(bbox_max, [1.0, 1.0], [float(w), float(h)])
    x0, y0 = bbox_min.astype(np.int64)
    x1, y1 = np.ceil(bbox_max).astype(np.int64)
    crop = rgb[y0:y1, x0:x1]
    crop_h, crop_w = crop.shape[:2]
    if crop_w > crop_h:
        new_w = int(crop_size)
        new_h = int(crop_size * crop_h / crop_w)
    else:
        new_h = int(crop_size)
        new_w = int(crop_size * crop_w / crop_h)
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_x = (crop_size - new_w) // 2
    pad_y = (crop_size - new_h) // 2
    crop_square = np.pad(
        resized,
        ((pad_y, crop_size - new_h - pad_y), (pad_x, crop_size - new_w - pad_x), (0, 0)),
        mode="edge",
    )
    scale_x = float(new_w) / float(bbox_max[0] - bbox_min[0])
    scale_y = float(new_h) / float(bbox_max[1] - bbox_min[1])
    K_crop = crop_resize_pad_intrinsics(K_orig, bbox_min=bbox_min, scale_xy=(scale_x, scale_y), pad_xy=(pad_x, pad_y))
    return crop_square.astype(np.uint8), K_crop.astype(np.float32), bbox_min, bbox_max


def pose_from_output(out):
    action = out["action_pred"][0].detach().cpu().numpy().astype(np.float64)
    quat = out["wrist_quat_pred"][0].detach().cpu().numpy().astype(np.float64)
    trans = out["wrist_trans_pred"][0].detach().cpu().numpy().astype(np.float64)
    return {
        "rot": quat,
        "trans": trans,
        "alpha": float(action[0]),
        "theta_l": float(action[1]),
        "theta_r": float(action[2]),
    }


def extract_ref_panel(ref_img, index, panel_size=630, separator=6):
    x0 = index * (panel_size + separator)
    return ref_img[:, x0:x0 + panel_size].copy()


def make_output_from_reference(ref_img, pose_mesh_rgb, pred_pose_mask, orig_rgb, panel_size):
    scale, pad_x, pad_y = square_image_geometry(orig_rgb, panel_size)
    pose_mesh_sq = pad_rgb_to_square(pose_mesh_rgb, panel_size, scale, pad_x, pad_y)
    pred_mask_sq = pad_mask_to_square(pred_pose_mask, panel_size, scale, pad_x, pad_y)
    rgb_sq = extract_ref_panel(ref_img, 0, panel_size)
    panels = [
        rgb_sq,
        extract_ref_panel(ref_img, 1, panel_size),
        add_title(pose_mesh_sq, "posehead-trimesh"),
        extract_ref_panel(ref_img, 3, panel_size),
        extract_ref_panel(ref_img, 4, panel_size),
        add_title(overlay_part_mask(rgb_sq, pred_mask_sq), "posehead-proj-seg"),
    ]
    return concat_panels(panels)


def process_sample(sample, dataset, dataset_index, model, renderer, device, args, to_tensor):
    key = (sample["video"], sample["frame_id"])
    if key not in dataset_index:
        raise KeyError(f"Reference sample not in dataset index: {key}")
    _, annot = dataset[dataset_index[key]]
    valid_instruments = annot["instruments"]
    ref_slot = int(sample["ref_inst"]) - 1
    if ref_slot < 0 or ref_slot >= len(valid_instruments):
        raise RuntimeError(
            f"Reference {sample['name']} asks inst{sample['ref_inst']} but loader has {len(valid_instruments)} valid instruments"
        )
    actual_instance_id = int(valid_instruments[ref_slot]["instance_id"])
    rgb = np.asarray(Image.open(frame_image_path(args.dataset_root, sample["video"], sample["frame_id"])).convert("RGB"))
    h, w = rgb.shape[:2]
    K_orig = np.array(
        [[args.focal, 0.0, w / 2.0], [0.0, args.focal, h / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    _, inst_mask = load_instance_masks(args.dataset_root, sample["video"], sample["frame_id"], actual_instance_id)
    crop_rgb, K_crop, _, _ = crop_instance(rgb, K_orig, inst_mask, args.crop_size)
    x = to_tensor(Image.fromarray(crop_rgb)).unsqueeze(0).to(device)
    K_crop_t = torch.from_numpy(K_crop).unsqueeze(0).to(device)
    with torch.inference_mode(), torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda"), dtype=torch.bfloat16):
        out = model(x, K_crop_t, masks_enc=None, masks_pred=None)
    pred_hm_crop_t, hm_scores_t = heatmap_argmax(out["keypoint_heatmaps"].float().detach().cpu())
    direct_pose = pose_from_output(out)
    if args.pose_recovery == "pnp":
        pose = pose_from_keypoints_pnp(
            pred_hm_crop_t[0].numpy().astype(np.float32),
            [direct_pose["alpha"], direct_pose["theta_l"], direct_pose["theta_r"]],
            K_crop,
            scores=hm_scores_t[0].numpy().astype(np.float32),
            min_score=args.pnp_min_score,
        )
    else:
        pose = direct_pose
    pred_kp_cam = instrument_keypoints_camera_np(pose["rot"], pose["trans"], [pose["alpha"], pose["theta_l"], pose["theta_r"]])
    pred_pose_orig = project_points_np(pred_kp_cam, K_orig).astype(np.float32)
    pose_mesh_rgb = renderer.render_pose_overlay(rgb, pose, K_orig, alpha=args.overlay_alpha)
    pred_pose_mask = renderer.render_pose_mask(pose, K_orig, rgb.shape[:2])
    ref_img = np.asarray(Image.open(sample["ref_path"]).convert("RGB"))
    canvas = make_output_from_reference(ref_img, pose_mesh_rgb, pred_pose_mask, rgb, args.panel_size)
    out_path = Path(args.output_dir) / "densepart_h2" / "vis" / "suturePulling" / sample["name"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(out_path)
    return {
        "video": sample["video"],
        "frame_id": sample["frame_id"],
        "ref_inst": int(sample["ref_inst"]),
        "actual_instance_id": actual_instance_id,
        "output": str(out_path),
        "reference": str(sample["ref_path"]),
        "hm_score_mean": float(hm_scores_t[0].numpy().mean()),
        "pose_recovery": args.pose_recovery,
        "alpha": pose["alpha"],
        "theta_l": pose["theta_l"],
        "theta_r": pose["theta_r"],
        "trans_x": float(pose["trans"][0]),
        "trans_y": float(pose["trans"][1]),
        "trans_z": float(pose["trans"][2]),
    }


def worker_main(rank, args, samples):
    device = configure_device(args.devices[rank])
    model, _ = load_model(args.checkpoint, device)
    renderer = GMSInstrumentTrimeshRenderer(device)
    dataset, dataset_index = build_dataset_index(args)
    to_tensor = tv_transforms.Compose(
        [
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    rows = []
    for i, sample in enumerate(samples):
        try:
            row = process_sample(sample, dataset, dataset_index, model, renderer, device, args, to_tensor)
            rows.append(row)
            if i == 0 or (i + 1) % int(args.print_freq) == 0 or i + 1 == len(samples):
                print(f"[worker {rank}] {i + 1}/{len(samples)} {sample['name']} -> ok", flush=True)
        except Exception as exc:
            row = {
                "video": sample["video"],
                "frame_id": sample["frame_id"],
                "ref_inst": int(sample["ref_inst"]),
                "output": "",
                "reference": str(sample["ref_path"]),
                "error": f"{type(exc).__name__}: {exc}",
            }
            rows.append(row)
            print(f"[worker {rank}] {sample['name']} -> {row['error']}", flush=True)
            if args.fail_fast:
                raise
    manifest = Path(args.output_dir) / "densepart_h2" / f"worker_{rank}_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def split_evenly(items, n):
    return [items[i::n] for i in range(n)]


def run(args):
    samples = parse_reference_files(args.reference_vis_dir)
    if int(args.max_samples) > 0:
        samples = samples[: int(args.max_samples)]
    args.devices = args.devices or [args.device]
    chunks = [chunk for chunk in split_evenly(samples, len(args.devices)) if chunk]
    args.devices = args.devices[: len(chunks)]
    print(f"Processing {len(samples)} reference sample(s) -> {args.output_dir}; devices={args.devices}", flush=True)
    if len(chunks) == 1:
        worker_main(0, args, chunks[0])
    else:
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

    rows = []
    for path in sorted((Path(args.output_dir) / "densepart_h2").glob("worker_*_manifest.json")):
        rows.extend(json.loads(path.read_text(encoding="utf-8")))
    if rows:
        csv_path = Path(args.output_dir) / "densepart_h2" / "prediction_summary.csv"
        with csv_path.open("w", newline="") as f:
            fieldnames = sorted({key for row in rows for key in row.keys()})
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {csv_path}", flush=True)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="logs/robopepp_instrument_pose_rarp_jepa_bs56_gpu0123/checkpoints/last.pt")
    parser.add_argument("--reference_vis_dir", type=str, default=REF_VIS)
    parser.add_argument("--dataset_root", type=str, default=SUTURE_ROOT)
    parser.add_argument("--output_dir", type=str, default="logs/robopepp_suturepulling_refmatched")
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--panel_size", type=int, default=630)
    parser.add_argument("--focal", type=float, default=587.54401824)
    parser.add_argument("--overlay_alpha", type=float, default=0.85)
    parser.add_argument("--pose_recovery", type=str, default="pnp", choices=["pnp", "direct"])
    parser.add_argument("--pnp_min_score", type=float, default=0.0)
    parser.add_argument("--train_ratio", type=float, default=0.95)
    parser.add_argument("--min_dice_shaft", type=float, default=0.8)
    parser.add_argument("--min_dice_wrist", type=float, default=0.6)
    parser.add_argument("--min_dice_gripper", type=float, default=0.6)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--print_freq", type=int, default=10)
    parser.add_argument("--fail_fast", type=int, default=0, choices=[0, 1])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--devices", nargs="*", default=["cuda:0"])
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
