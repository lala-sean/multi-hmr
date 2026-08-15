import argparse
import csv
import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

ROBOPEPP_ROOT = Path(__file__).resolve().parent
if str(ROBOPEPP_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBOPEPP_ROOT))


def _load_local_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_base = _load_local_module(
    "robopepp_surfemb_keypoint_base_for_lnd_debug",
    ROBOPEPP_ROOT / "train_surfemb_keypoint_crop_rarp_lnd_refinemem.py",
)
_resnet_model = _load_local_module(
    "robopepp_surfemb_resnet_model_for_lnd_debug",
    ROBOPEPP_ROOT / "models" / "surfemb_resnet_crop_model.py",
)

collate_fn_surfemb_keypoint_crop = _base.collate_fn_surfemb_keypoint_crop
make_lnd_dataset = _base.make_lnd_dataset
make_lnd_val_dataset = _base.make_lnd_val_dataset
make_rarp_val_dataset = _base.make_rarp_val_dataset
make_surfemb_key_coords = _base.make_surfemb_key_coords
save_surfemb_keypoint_debug_panel = _base.save_surfemb_keypoint_debug_panel
to_device = _base.to_device
_pose_from_debug_target = _base._dataset_module._pose_from_debug_target
SurfEmbResNetCropModel = _resnet_model.SurfEmbResNetCropModel


def _as_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _default_args():
    return SimpleNamespace(
        needle_puncture_data_dir=_base.PUNCTURE_DATASET_ROOT,
        needle_puncture_pose_dir=_base.PUNCTURE_POSE_ROOT,
        needle_grasping_data_dir=_base.GRASPING_DATASET_ROOT,
        needle_grasping_pose_dir=_base.GRASPING_POSE_ROOT,
        knotting_data_dir=_base.KNOTTING_DATASET_ROOT,
        knotting_pose_dir=_base.KNOTTING_POSE_ROOT,
        train_dataset_names="needlePuncture,needleGrasping,knotting",
        min_dice_shaft=0.8,
        min_dice_wrist=0.6,
        min_dice_gripper=0.6,
        needle_train_ratio=0.95,
        canonicalize_pose_symmetry=1,
        canonical_eps=0.08,
        crop_size=224,
        bbox_padding_frac=0.12,
        dataset_cache_dir=None,
        surface_points_path=str(_base.DEFAULT_SURFACE_POINTS),
        surfemb_n_pos=1024,
        surfemb_n_neg=1024,
        surfemb_shaft_sample_ratio=0.20,
        surfemb_wrist_sample_ratio=0.60,
        surfemb_gripper_sample_ratio=0.20,
        surfemb_key_noise=0.0,
        surfemb_crop_scale=1.2,
        surfemb_max_angle=np.pi,
        surfemb_offset_scale=1.0,
        surfemb_min_depth=1e-4,
        surfemb_depth_tolerance=8e-4,
        surfemb_use_mesh_zbuffer=1,
        surfemb_zbuffer_backend="opengl",
        surfemb_min_train_render_iou=0.35,
        heatmap_sigma=2.0,
        include_lnd_train=1,
        include_lnd_val=1,
        lnd_root=_base.DEFAULT_LND_ROOT,
        lnd_refine_memory=_base.DEFAULT_LND_REFINE_MEMORY,
        lnd_sampling_rate=0.30,
        lnd_train_subsample=1,
        lnd_val_subsample=1,
        train_subsample=1,
        val_subsample=1,
        surfemb_emb_dim=12,
        surfemb_mlp_hidden_features=256,
        surfemb_mlp_hidden_layers=2,
        resnet_feat_preultimate=64,
        amp=1,
    )


def _dataset_from_name(args, name):
    if name == "lnd_val":
        return make_lnd_val_dataset(args)
    if name == "lnd_train_memory":
        return make_lnd_dataset(args, "TRAIN", False, True, args.lnd_train_subsample)
    if name == "rarp_val":
        return make_rarp_val_dataset(args)
    raise ValueError(f"unknown dataset {name!r}")


def _load_resnet(checkpoint, device, args):
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    ckpt_args = ckpt.get("args", {})
    args.surfemb_emb_dim = int(ckpt_args.get("surfemb_emb_dim", args.surfemb_emb_dim))
    args.surfemb_mlp_hidden_features = int(
        ckpt_args.get("surfemb_mlp_hidden_features", args.surfemb_mlp_hidden_features)
    )
    args.surfemb_mlp_hidden_layers = int(ckpt_args.get("surfemb_mlp_hidden_layers", args.surfemb_mlp_hidden_layers))
    args.resnet_feat_preultimate = int(ckpt_args.get("resnet_feat_preultimate", args.resnet_feat_preultimate))
    model = SurfEmbResNetCropModel(
        img_size=args.crop_size,
        surfemb_emb_dim=args.surfemb_emb_dim,
        surfemb_mlp_hidden_features=args.surfemb_mlp_hidden_features,
        surfemb_mlp_hidden_layers=args.surfemb_mlp_hidden_layers,
        resnet_feat_preultimate=args.resnet_feat_preultimate,
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device).eval()
    return model


@torch.no_grad()
def _batch_resnet_metrics(model, x, y, args, device):
    x = x.to(device, non_blocking=True)
    y = to_device(y, device)
    with torch.amp.autocast(device.type, enabled=bool(args.amp) and device.type == "cuda", dtype=torch.bfloat16):
        out = model(x, y["K"], surfemb_key_coords=make_surfemb_key_coords(y, args))

    queries = out["surfemb_queries"].float()
    keys = out["surfemb_keys"].float()
    inst = y["inst_mask"].float()
    yx = y["surfemb_mask_samples"].long()
    B, _, H, W = queries.shape
    y_idx = yx[..., 0].clamp(0, H - 1)
    x_idx = yx[..., 1].clamp(0, W - 1)
    batch_idx = torch.arange(B, device=device).view(B, 1)

    queries_pos = queries[batch_idx, :, y_idx, x_idx]
    n_pos = int(y["surfemb_coords_pos"].shape[1])
    keys_pos = keys[:, :n_pos]
    keys_neg = keys[:, n_pos:]
    sim_pos = (queries_pos * keys_pos).sum(dim=-1, keepdim=True)
    sim_neg = queries_pos @ keys_neg.permute(0, 2, 1)
    logits = torch.cat((sim_pos, sim_neg), dim=-1).permute(0, 2, 1)
    target = torch.zeros((B, n_pos), device=device, dtype=torch.long)
    nce = F.cross_entropy(logits, target, reduction="none").mean(dim=1)

    mask_prob = torch.sigmoid(out["inst_mask_logits"].float())
    mask_bce = F.binary_cross_entropy(mask_prob, inst, reduction="none").flatten(1).mean(dim=1)
    pred = mask_prob > 0.5
    gt = inst > 0.5
    inter = (pred & gt).float().flatten(1).sum(dim=1)
    union = (pred | gt).float().flatten(1).sum(dim=1).clamp_min(1.0)
    miou = inter / union
    return {
        "nce": nce.detach().cpu().numpy(),
        "mask_bce": mask_bce.detach().cpu().numpy(),
        "mask_iou": miou.detach().cpu().numpy(),
    }


def _tensor_item(value, i):
    if torch.is_tensor(value):
        return value[i].detach().cpu()
    return value[i]


def _target_from_batch(y, i):
    out = {}
    for k, v in y.items():
        out[k] = _tensor_item(v, i) if isinstance(v, (list, tuple)) or torch.is_tensor(v) else v
    return out


def _mask_rgb(mask, color):
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    out[mask.astype(bool)] = np.asarray(color, dtype=np.uint8)
    return out


def _mesh_support_crop(target, renderer, min_depth=1e-4):
    pose = _pose_from_debug_target(target)
    K_orig = _as_numpy(target["K_orig"]).astype(np.float32)
    M = _as_numpy(target["surfemb_M_crop"]).astype(np.float32)
    orig_shape = _as_numpy(target["orig_rgb"]).shape[:2]
    _, depth, _ = renderer.render_pose(pose, K_orig, orig_shape)
    support_orig = (np.asarray(depth) > float(min_depth)).astype(np.uint8)
    h, w = _as_numpy(target["inst_mask"]).shape[:2]
    return cv2.warpAffine(
        support_orig,
        M,
        (int(w), int(h)),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)


def _save_case_panel(path, target, renderer):
    from PIL import Image

    rgb = _as_numpy(target["crop_rgb"]).astype(np.uint8)
    gt = (_as_numpy(target["inst_mask"]) > 0).astype(bool)
    part = _as_numpy(target["part_mask"]).astype(np.uint8)
    mesh = _mesh_support_crop(target, renderer)

    gt_overlay = rgb.copy()
    gt_overlay[gt] = (0.55 * gt_overlay[gt] + np.array([30, 230, 80]) * 0.45).astype(np.uint8)
    mesh_overlay = rgb.copy()
    mesh_overlay[mesh] = (0.55 * mesh_overlay[mesh] + np.array([240, 60, 60]) * 0.45).astype(np.uint8)

    mismatch = rgb.copy()
    both = gt & mesh
    gt_only = gt & ~mesh
    mesh_only = mesh & ~gt
    mismatch[both] = (0.55 * mismatch[both] + np.array([250, 220, 40]) * 0.45).astype(np.uint8)
    mismatch[gt_only] = (0.55 * mismatch[gt_only] + np.array([40, 230, 80]) * 0.45).astype(np.uint8)
    mismatch[mesh_only] = (0.55 * mismatch[mesh_only] + np.array([240, 60, 60]) * 0.45).astype(np.uint8)

    colors = np.array(
        [[0, 0, 0], [80, 150, 240], [80, 220, 120], [230, 80, 80], [230, 210, 80]],
        dtype=np.uint8,
    )
    part_rgb = colors[np.clip(part, 0, len(colors) - 1)]
    point_rgb = rgb.copy()
    yx = _as_numpy(target["surfemb_mask_samples"]).astype(np.int64)
    pid = _as_numpy(target["surfemb_positive_part_ids"]).astype(np.int64)
    point_colors = colors[np.clip(pid, 0, len(colors) - 1)]
    for (yy, xx), color in zip(yx, point_colors):
        if 0 <= xx < point_rgb.shape[1] and 0 <= yy < point_rgb.shape[0]:
            cv2.circle(point_rgb, (int(xx), int(yy)), 1, tuple(int(c) for c in color.tolist()), -1)

    mesh_pose_overlay = rgb.copy()
    try:
        mesh_pose_overlay = renderer.render_pose_overlay_warped_crop(
            rgb,
            _pose_from_debug_target(target),
            _as_numpy(target["K_orig"]).astype(np.float32),
            _as_numpy(target["surfemb_M_crop"]).astype(np.float32),
            _as_numpy(target["orig_rgb"]).shape[:2],
            alpha=0.70,
        )
    except Exception as exc:
        cv2.putText(
            mesh_pose_overlay,
            f"render failed {type(exc).__name__}",
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 80, 80),
            1,
            cv2.LINE_AA,
        )

    panel = np.concatenate([rgb, gt_overlay, mesh_overlay, mismatch, mesh_pose_overlay, part_rgb, point_rgb], axis=1)
    Image.fromarray(panel).save(path)


def _scan(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_args = _default_args()
    base_args.dataset_cache_dir = args.dataset_cache_dir
    base_args.lnd_val_subsample = args.subsample
    base_args.lnd_train_subsample = args.subsample
    base_args.val_subsample = args.subsample
    base_args.surfemb_n_pos = args.n_pos
    base_args.surfemb_n_neg = args.n_neg
    base_args.amp = int(args.amp)

    dataset = _dataset_from_name(base_args, args.dataset)
    print(dataset, flush=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn_surfemb_keypoint_crop,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.checkpoint else "cpu")
    model = _load_resnet(args.checkpoint, device, base_args) if args.checkpoint else None

    rows = []
    sample_offset = 0
    for x, y in loader:
        model_metrics = None
        if model is not None:
            model_metrics = _batch_resnet_metrics(model, x, y, base_args, device)
        B = int(x.shape[0])
        for i in range(B):
            target = _target_from_batch(y, i)
            action = _as_numpy(target["action"]).reshape(-1)
            row = {
                "dataset": args.dataset,
                "idx": sample_offset + i,
                "video_name": target["video_name"],
                "frame_id": target["frame_id"],
                "render_iou": float(_as_numpy(target["surfemb_render_iou"])),
                "part_ratio_exact": int(bool(_as_numpy(target["surfemb_part_ratio_exact"]))),
                "action_alpha": float(action[0]),
                "action_theta_l": float(action[1]),
                "action_theta_r": float(action[2]),
                "inst_px": int((_as_numpy(target["inst_mask"]) > 0).sum()),
            }
            if model_metrics is not None:
                row.update(
                    {
                        "nce": float(model_metrics["nce"][i]),
                        "mask_bce": float(model_metrics["mask_bce"][i]),
                        "mask_iou": float(model_metrics["mask_iou"][i]),
                    }
                )
            rows.append(row)
        sample_offset += B
        if args.max_samples > 0 and sample_offset >= args.max_samples:
            break

    rows = rows[: args.max_samples] if args.max_samples > 0 else rows
    csv_path = out_dir / f"{args.dataset}_summary.csv"
    fieldnames = list(rows[0].keys()) if rows else []
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    def _mean(key):
        vals = [float(r[key]) for r in rows if key in r]
        return float(np.mean(vals)) if vals else float("nan")

    print(
        f"SUMMARY dataset={args.dataset} N={len(rows)} "
        f"render_iou_mean={_mean('render_iou'):.4f} "
        f"render_iou_min={min(float(r['render_iou']) for r in rows):.4f} "
        f"zero_action_frac={np.mean([abs(r['action_alpha']) < 1e-8 and abs(r['action_theta_l']) < 1e-8 and abs(r['action_theta_r']) < 1e-8 for r in rows]):.4f}",
        flush=True,
    )
    if model is not None:
        print(
            f"MODEL_MEAN nce={_mean('nce'):.4f} mask_iou={_mean('mask_iou'):.4f} mask_bce={_mean('mask_bce'):.4f}",
            flush=True,
        )

    sort_key = "nce" if model is not None and args.sort_by == "nce" else args.sort_by
    ranked = sorted(rows, key=lambda r: float(r[sort_key]), reverse=sort_key in {"nce", "mask_bce"})
    if sort_key == "render_iou":
        ranked = sorted(rows, key=lambda r: float(r["render_iou"]))
    top_path = out_dir / f"{args.dataset}_top_{sort_key}.csv"
    with open(top_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ranked[: args.num_vis])

    renderer = dataset._get_mesh_renderer()
    for rank, row in enumerate(ranked[: args.num_vis]):
        _, target = dataset[int(row["idx"])]
        stem = (
            f"{rank:02d}_idx{int(row['idx']):04d}_frame{row['frame_id']}"
            f"_riou{float(row['render_iou']):.3f}"
        )
        if "nce" in row:
            stem += f"_nce{float(row['nce']):.2f}"
        _save_case_panel(out_dir / f"{stem}_case.jpg", target, renderer)
        save_surfemb_keypoint_debug_panel(out_dir / f"{stem}_debugpanel.jpg", target, mesh_renderer=renderer)
    print(f"WROTE {csv_path}", flush=True)
    print(f"WROTE {top_path}", flush=True)
    print(f"WROTE_VIS {out_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="lnd_val", choices=["lnd_val", "lnd_train_memory", "rarp_val"])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output_dir", default=str(ROBOPEPP_ROOT / "logs" / "debug_lnd_val_label_cases"))
    parser.add_argument("--dataset_cache_dir", default=str(ROBOPEPP_ROOT / "logs" / "surfemb_keypoint_crop_part206020_sharedgl_p2048_b32_gpu0123" / "dataset_cache"))
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--subsample", type=int, default=1)
    parser.add_argument("--n_pos", type=int, default=1024)
    parser.add_argument("--n_neg", type=int, default=1024)
    parser.add_argument("--num_vis", type=int, default=12)
    parser.add_argument("--sort_by", default="render_iou", choices=["render_iou", "nce", "mask_bce"])
    parser.add_argument("--amp", type=int, default=1, choices=[0, 1])
    _scan(parser.parse_args())


if __name__ == "__main__":
    main()
