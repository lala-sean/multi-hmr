import os
import sys
import time
import importlib.util
from argparse import ArgumentParser
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

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


_base_train = _load_local_module(
    "robopepp_surfemb_keypoint_train_base_for_resnet",
    ROBOPEPP_ROOT / "train_surfemb_keypoint_crop_rarp_lnd_refinemem.py",
)
_model_module = _load_local_module(
    "robopepp_surfemb_resnet_crop_model",
    ROBOPEPP_ROOT / "models" / "surfemb_resnet_crop_model.py",
)

AverageMeter = _base_train.AverageMeter
DEFAULT_LND_REFINE_MEMORY = _base_train.DEFAULT_LND_REFINE_MEMORY
DEFAULT_LND_ROOT = _base_train.DEFAULT_LND_ROOT
DEFAULT_SURFACE_POINTS = _base_train.DEFAULT_SURFACE_POINTS
GRASPING_DATASET_ROOT = _base_train.GRASPING_DATASET_ROOT
GRASPING_POSE_ROOT = _base_train.GRASPING_POSE_ROOT
KNOTTING_DATASET_ROOT = _base_train.KNOTTING_DATASET_ROOT
KNOTTING_POSE_ROOT = _base_train.KNOTTING_POSE_ROOT
PUNCTURE_DATASET_ROOT = _base_train.PUNCTURE_DATASET_ROOT
PUNCTURE_POSE_ROOT = _base_train.PUNCTURE_POSE_ROOT
collate_fn_surfemb_keypoint_crop = _base_train.collate_fn_surfemb_keypoint_crop
dist_barrier = _base_train.dist_barrier
is_main_process = _base_train.is_main_process
make_lnd_val_dataset = _base_train.make_lnd_val_dataset
make_rarp_val_dataset = _base_train.make_rarp_val_dataset
make_surfemb_key_coords = _base_train.make_surfemb_key_coords
make_train_datasets = _base_train.make_train_datasets
reduce_float = _base_train.reduce_float
save_checkpoint = _base_train.save_checkpoint
save_surfemb_keypoint_debug_panel = _base_train.save_surfemb_keypoint_debug_panel
set_epoch_recursive = _base_train.set_epoch_recursive
setup_dist = _base_train.setup_dist
to_device = _base_train.to_device

SurfEmbResNetCropModel = _model_module.SurfEmbResNetCropModel
PRIMARY_VAL_NAME = "RARP"


def _display_val_score_mode(args):
    if args.val_score_mode == "rarp_total":
        return f"{PRIMARY_VAL_NAME.lower()}_total"
    return args.val_score_mode


def _mean_val_total(*results):
    vals = []
    for res in results:
        if res and "val_total" in res:
            vals.append(float(res["val_total"]))
    if not vals:
        return float("inf")
    return float(sum(vals) / len(vals))


def _val_score_from_results(args, rarp_val, lnd_val):
    if args.val_score_mode == "mean_total":
        return _mean_val_total(rarp_val, lnd_val)
    if not rarp_val or "val_total" not in rarp_val:
        return float("inf")
    return float(rarp_val["val_total"])


def _lnd_val_has_wrist_only_pose(dataset):
    if dataset is None:
        return False
    return "pose_source=direct_lnd_gt_wrist" in repr(dataset)


def compute_surfemb_resnet_losses(out, y, args):
    inst_mask = y["inst_mask"].float()
    coords_pos = y["surfemb_coords_pos"].float()
    coords_neg = y["surfemb_surface_samples"].float()
    mask_samples = y["surfemb_mask_samples"].long()
    queries = out["surfemb_queries"].float()
    B, _, H, W = queries.shape
    if coords_pos.shape[1] != int(args.surfemb_n_pos):
        raise RuntimeError(f"coords_pos n={coords_pos.shape[1]} does not match surfemb_n_pos={args.surfemb_n_pos}")
    if coords_neg.shape[1] != int(args.surfemb_n_neg):
        raise RuntimeError(f"coords_neg n={coords_neg.shape[1]} does not match surfemb_n_neg={args.surfemb_n_neg}")
    if mask_samples.shape[1] != int(args.surfemb_n_pos):
        raise RuntimeError(f"mask_samples n={mask_samples.shape[1]} does not match surfemb_n_pos={args.surfemb_n_pos}")

    with torch.amp.autocast(out["inst_mask_logits"].device.type, enabled=False):
        mask_prob = torch.sigmoid(out["inst_mask_logits"].float())
        mask_loss = F.binary_cross_entropy(mask_prob, inst_mask.float())

    valid_cse = y.get("has_cse")
    if valid_cse is None:
        valid_cse = torch.ones((B,), dtype=torch.bool, device=queries.device)
    else:
        valid_cse = valid_cse.to(device=queries.device).bool().view(B)
    if not valid_cse.any():
        nce_loss = queries.sum() * 0.0
        total = mask_loss + nce_loss
        return total, {
            "total": total.detach(),
            "mask_bce": mask_loss.detach(),
            "nce": nce_loss.detach(),
            "px": inst_mask.sum().detach(),
        }

    queries = queries[valid_cse]
    inst_mask_valid = inst_mask[valid_cse]
    coords_pos = coords_pos[valid_cse]
    coords_neg = coords_neg[valid_cse]
    mask_samples = mask_samples[valid_cse]
    keys = out["surfemb_keys"][valid_cse]
    Bv = queries.shape[0]

    yx = mask_samples.clamp_min(0)
    y_idx = yx[..., 0].clamp_max(H - 1)
    x_idx = yx[..., 1].clamp_max(W - 1)
    batch_idx = torch.arange(Bv, device=queries.device).view(Bv, 1)

    ratios = (
        float(args.surfemb_shaft_sample_ratio),
        float(args.surfemb_wrist_sample_ratio),
        float(args.surfemb_gripper_sample_ratio),
    )
    if ratios == (0.0, 1.0, 0.0):
        positive_part_ids = y["surfemb_positive_part_ids"][valid_cse]
        negative_part_ids = y["surfemb_surface_part_ids"][valid_cse]
        sampled_inst = inst_mask_valid[batch_idx, y_idx, x_idx]
        sampled_parts = y["part_mask"][valid_cse][batch_idx, y_idx, x_idx]
        if not torch.all(positive_part_ids == 2):
            raise RuntimeError("Wrist-only SurfEmb positives contain a non-wrist part ID.")
        if not torch.all(negative_part_ids == 2):
            raise RuntimeError("Wrist-only SurfEmb negatives contain a non-wrist part ID.")
        if not torch.all(sampled_inst > 0.5):
            raise RuntimeError("Wrist-only SurfEmb positives contain pixels outside the wrist mask.")
        if not torch.all(sampled_parts == 2):
            raise RuntimeError("Wrist-only SurfEmb positive pixels contain a non-wrist label.")

    queries_pos = queries[batch_idx, :, y_idx, x_idx]
    keys_pos = keys[:, : coords_pos.shape[1]]
    keys_neg = keys[:, coords_pos.shape[1] :]
    query_norm = queries_pos.norm(dim=-1).mean()
    positive_key_norm = keys_pos.norm(dim=-1).mean()
    negative_key_norm = keys_neg.norm(dim=-1).mean()
    similarity = str(getattr(args, "surfemb_similarity", "raw_dot"))
    temperature = float(getattr(args, "surfemb_temperature", 1.0))
    if temperature <= 0.0:
        raise ValueError(f"surfemb_temperature must be positive, got {temperature}.")
    if similarity == "cosine":
        queries_pos = F.normalize(queries_pos, dim=-1, eps=1e-6)
        keys_pos = F.normalize(keys_pos, dim=-1, eps=1e-6)
        keys_neg = F.normalize(keys_neg, dim=-1, eps=1e-6)
    elif similarity != "raw_dot":
        raise ValueError(f"Unknown SurfEmb similarity: {similarity!r}")
    sim_pos = (queries_pos * keys_pos).sum(dim=-1, keepdim=True)
    sim_neg = queries_pos @ keys_neg.permute(0, 2, 1)
    logits = (torch.cat((sim_pos, sim_neg), dim=-1) / temperature).permute(0, 2, 1)
    target = torch.zeros(Bv, int(args.surfemb_n_pos), device=queries.device, dtype=torch.long)
    nce_loss = F.cross_entropy(logits, target)
    total = mask_loss + nce_loss

    with torch.no_grad():
        pred_mask = mask_prob > 0.5
        gt_mask = inst_mask > 0.5
        inter = (pred_mask & gt_mask).float().sum()
        union = (pred_mask | gt_mask).float().sum().clamp_min(1.0)
        iou = inter / union
        dice = (2.0 * inter) / (pred_mask.float().sum() + gt_mask.float().sum()).clamp_min(1.0)
    metrics = {
        "total": torch.nan_to_num(total.detach(), nan=0.0, posinf=0.0, neginf=0.0),
        "mask_bce": torch.nan_to_num(mask_loss.detach(), nan=0.0, posinf=0.0, neginf=0.0),
        "nce": torch.nan_to_num(nce_loss.detach(), nan=0.0, posinf=0.0, neginf=0.0),
        "query_norm": query_norm.detach(),
        "positive_key_norm": positive_key_norm.detach(),
        "negative_key_norm": negative_key_norm.detach(),
        "mask_iou": iou.detach(),
        "mask_dice": dice.detach(),
        "px": inst_mask_valid.sum().detach(),
    }
    if "surfemb_render_iou" in y:
        metrics["render_iou"] = torch.nan_to_num(
            y["surfemb_render_iou"].float().mean(), nan=0.0, posinf=0.0, neginf=0.0
        ).detach()
    if "surfemb_part_ratio_exact" in y:
        metrics["part_ratio_exact_frac"] = y["surfemb_part_ratio_exact"].float().mean().detach()
    return total, metrics


@torch.no_grad()
def evaluate(model, loader, device, args, max_batches=None):
    model.eval()
    meters = {}
    sample_count = 0
    batch_count = 0
    for x, y in loader:
        batch_size = int(x.shape[0])
        x = x.to(device, non_blocking=True)
        y = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in y.items()}
        with torch.amp.autocast(device.type, enabled=bool(args.amp) and device.type == "cuda", dtype=torch.bfloat16):
            out = model(x, y["K"], surfemb_key_coords=make_surfemb_key_coords(y, args))
            _, metrics = compute_surfemb_resnet_losses(out, y, args)
        for key, value in metrics.items():
            if key == "px":
                meters[key] = meters.get(key, 0.0) + float(value.item())
            else:
                meters[key] = meters.get(key, 0.0) + float(value.item()) * batch_size
        sample_count += batch_size
        batch_count += 1
        if max_batches is not None and batch_count >= int(max_batches):
            break
    if sample_count == 0:
        return {}

    keys = sorted(meters)
    packed = torch.tensor(
        [meters[key] for key in keys] + [float(sample_count)],
        dtype=torch.float64,
        device=device,
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    global_count = max(1.0, float(packed[-1].item()))
    return {
        f"val_{key}": float(packed[i].item()) / global_count
        for i, key in enumerate(keys)
    }


def main(args):
    if args.crop_size != 224:
        raise ValueError("This SurfEmb-ResNet runner is fixed to --crop_size 224.")
    device, local_rank, world_size = setup_dist(args.dist_timeout_sec)
    use_ddp = world_size > 1
    torch.backends.cudnn.benchmark = True

    log_dir = Path(args.save_dir) / args.name
    args.log_dir = str(log_dir)
    if args.dataset_cache_dir is None:
        args.dataset_cache_dir = str(log_dir / "dataset_cache")
    if is_main_process():
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    if use_ddp:
        if is_main_process():
            train_datasets = make_train_datasets(args)
            rarp_val_dataset = make_rarp_val_dataset(args)
            lnd_val_dataset = make_lnd_val_dataset(args)
            dist_barrier(local_rank)
        else:
            dist_barrier(local_rank)
            train_datasets = make_train_datasets(args)
            rarp_val_dataset = make_rarp_val_dataset(args)
            lnd_val_dataset = make_lnd_val_dataset(args)
    else:
        train_datasets = make_train_datasets(args)
        rarp_val_dataset = make_rarp_val_dataset(args)
        lnd_val_dataset = make_lnd_val_dataset(args)

    train_dataset = ConcatDataset(train_datasets)
    if bool(args.inspect_datasets_only):
        if is_main_process():
            print("INSPECT_DATASETS", flush=True)
            for ds in train_datasets:
                print(ds, flush=True)
            print(rarp_val_dataset, flush=True)
            if lnd_val_dataset is not None:
                print(lnd_val_dataset, flush=True)
            sample_sets = [(f"train{i}", ds) for i, ds in enumerate(train_datasets)]
            sample_sets.append((f"{PRIMARY_VAL_NAME.lower()}_val", rarp_val_dataset))
            if lnd_val_dataset is not None:
                sample_sets.append(("lnd_val", lnd_val_dataset))
            mesh_renderer = None
            if args.inspect_vis_dir and bool(args.inspect_render_mesh):
                try:
                    from instrument_opengl_renderer import InstrumentOpenGLDepthRenderer

                    mesh_renderer = InstrumentOpenGLDepthRenderer(args.crop_size, args.crop_size)
                except Exception as exc:
                    print(f"INSPECT_MESH_RENDERER_FAILED {type(exc).__name__}: {exc}", flush=True)
            for label, ds in sample_sets:
                n_inspect = min(int(args.inspect_samples_per_dataset), len(ds))
                for sample_idx in range(n_inspect):
                    x, y = ds[sample_idx]
                    sample_label = f"{label}_{sample_idx:02d}"
                    print(
                        f"INSPECT_SAMPLE {sample_label}: x={tuple(x.shape)} "
                        f"inst={tuple(y['inst_mask'].shape)} visible_px={int(y['inst_mask'].sum())} "
                        f"surf_pos={tuple(y['surfemb_mask_samples'].shape)} "
                        f"surf_pos_xyz={tuple(y['surfemb_coords_pos'].shape)} "
                        f"surf_neg={tuple(y['surfemb_surface_samples'].shape)} "
                        f"K00={float(y['K'][0, 0]):.3f} "
                        f"render_iou={float(y['surfemb_render_iou']):.4f}",
                        flush=True,
                    )
                    if args.inspect_vis_dir:
                        vis_dir = Path(args.inspect_vis_dir)
                        vis_dir.mkdir(parents=True, exist_ok=True)
                        save_surfemb_keypoint_debug_panel(
                            vis_dir / f"{sample_label}_surfemb_resnet_crop_data.jpg",
                            y,
                            mesh_renderer=mesh_renderer,
                        )
        if use_ddp:
            dist.destroy_process_group()
        return

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if use_ddp else None
    rarp_val_sampler = DistributedSampler(rarp_val_dataset, shuffle=False) if use_ddp else None
    lnd_val_sampler = DistributedSampler(lnd_val_dataset, shuffle=False) if (use_ddp and lnd_val_dataset is not None) else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn_surfemb_keypoint_crop,
        persistent_workers=False,
    )
    rarp_val_loader = DataLoader(
        rarp_val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        sampler=rarp_val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn_surfemb_keypoint_crop,
        persistent_workers=False,
    )
    lnd_val_loader = None
    if lnd_val_dataset is not None:
        lnd_val_loader = DataLoader(
            lnd_val_dataset,
            batch_size=args.val_batch_size,
            shuffle=False,
            sampler=lnd_val_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn_surfemb_keypoint_crop,
            persistent_workers=False,
        )

    model = SurfEmbResNetCropModel(
        img_size=224,
        surfemb_emb_dim=args.surfemb_emb_dim,
        surfemb_mlp_hidden_features=args.surfemb_mlp_hidden_features,
        surfemb_mlp_hidden_layers=args.surfemb_mlp_hidden_layers,
        resnet_feat_preultimate=args.resnet_feat_preultimate,
    ).to(device)

    resume_ckpt = None
    resume_epoch = 0
    resume_iter = 0
    if args.resume:
        resume_ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        log = model.load_state_dict(resume_ckpt["model_state_dict"], strict=False)
        resume_epoch = int(resume_ckpt.get("epoch", 0))
        resume_iter = int(resume_ckpt.get("iter", 0))
        if bool(args.reset_iter_on_resume):
            resume_epoch = 0
            resume_iter = 0
        if is_main_process():
            print(f"Loading checkpoint from {args.resume}", flush=True)
            print(log, flush=True)

    if use_ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    raw = model.module if isinstance(model, DDP) else model

    optimizer = torch.optim.Adam(
        [
            {"params": raw.cnn.parameters(), "lr": args.lr_cnn},
            {"params": raw.surface_key_mlp.parameters(), "lr": args.lr_surfemb_mlp},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda i: min(1.0, float(i + 1) / max(1.0, float(args.warmup_steps))),
    )
    if resume_ckpt is not None and bool(args.resume_optimizer):
        optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        if resume_ckpt.get("scheduler_state_dict") is not None and bool(args.resume_scheduler):
            scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])

    if is_main_process():
        print(f"LOG_DIR: {log_dir}", flush=True)
        print(f"WORLD_SIZE: {world_size}", flush=True)
        print("MODEL: SurfEmbResNetCropModel", flush=True)
        print("BACKBONE: original SurfEmb ResNet18 U-Net, ImageNet pretrained by torchvision", flush=True)
        print("SUPERVISION: binary mask BCE + original SurfEmb-style InfoNCE correspondence", flush=True)
        print(
            f"SURFEMB: n_pos={args.surfemb_n_pos} n_neg={args.surfemb_n_neg} "
            f"emb_dim={args.surfemb_emb_dim} key_noise={args.surfemb_key_noise} "
            f"similarity={getattr(args, 'surfemb_similarity', 'raw_dot')} "
            f"temperature={getattr(args, 'surfemb_temperature', 1.0)} "
            "part_ratios="
            f"shaft:{args.surfemb_shaft_sample_ratio:.3f},"
            f"wrist:{args.surfemb_wrist_sample_ratio:.3f},"
            f"gripper:{args.surfemb_gripper_sample_ratio:.3f}",
            flush=True,
        )
        print(f"SURFACE_POINTS: {args.surface_points_path}", flush=True)
        print(f"LND_SAMPLING_RATE: {args.lnd_sampling_rate:.3f}", flush=True)
        for ds in train_datasets:
            print(ds, flush=True)
        print(rarp_val_dataset, flush=True)
        if lnd_val_dataset is not None:
            print(lnd_val_dataset, flush=True)
        print(f"PRIMARY_VALIDATION: {PRIMARY_VAL_NAME}", flush=True)
        print(f"VAL_SCORE_MODE: {_display_val_score_mode(args)}", flush=True)
        if _lnd_val_has_wrist_only_pose(lnd_val_dataset):
            print(
                "LND_VAL_NOTE: TEST has wrist-only GT pose and zero articulation action; "
                "LND SurfEmb NCE/total are diagnostic-only unless full action GT is provided.",
                flush=True,
            )

    iteration = resume_iter
    epoch = resume_epoch
    last_log = time.time()
    best_val_score = float("inf")
    if bool(args.validate_before_train):
        max_batches = None if int(args.val_max_batches) <= 0 else int(args.val_max_batches)
        rarp_val = evaluate(model, rarp_val_loader, device, args, max_batches=max_batches)
        if is_main_process():
            print(
                f"PRE_VAL_{PRIMARY_VAL_NAME} "
                + " | ".join(f"{k} {v:.4f}" for k, v in rarp_val.items()),
                flush=True,
            )
        lnd_val = None
        if lnd_val_loader is not None:
            lnd_val = evaluate(model, lnd_val_loader, device, args, max_batches=max_batches)
            if is_main_process():
                print("PRE_VAL_LND " + " | ".join(f"{k} {v:.4f}" for k, v in lnd_val.items()), flush=True)
        if is_main_process():
            score = _val_score_from_results(args, rarp_val, lnd_val)
            print(f"PRE_VAL_SCORE {_display_val_score_mode(args)} {score:.4f}", flush=True)
        model.train()

    while iteration < args.max_iter:
        set_epoch_recursive(train_dataset, epoch)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        meters = {}
        for x, y in train_loader:
            iteration += 1
            x = x.to(device, non_blocking=True)
            y = to_device(y, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=bool(args.amp) and device.type == "cuda", dtype=torch.bfloat16):
                out = model(x, y["K"], surfemb_key_coords=make_surfemb_key_coords(y, args))
                loss, metrics = compute_surfemb_resnet_losses(out, y, args)
            if not torch.isfinite(loss):
                if is_main_process():
                    bad = {
                        k: float(v.detach().float().cpu().item())
                        for k, v in metrics.items()
                        if torch.is_tensor(v) and v.numel() == 1
                    }
                    print(f"NONFINITE LOSS at iter {iteration}: {bad}", flush=True)
                raise RuntimeError(f"non-finite training loss at iter {iteration}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            for key, value in metrics.items():
                meters.setdefault(key, AverageMeter()).update(float(value.item()))

            if iteration % args.log_freq == 0 and is_main_process():
                elapsed = max(time.time() - last_log, 1e-6)
                last_log = time.time()
                msg = [
                    f"iter {iteration:07d}",
                    f"epoch {epoch}",
                    f"loss {meters['total'].avg:.4f}",
                    f"mask {meters['mask_bce'].avg:.4f}",
                    f"nce {meters['nce'].avg:.4f}",
                    f"qnorm {meters.get('query_norm', AverageMeter()).avg:.3f}",
                    f"kpnorm {meters.get('positive_key_norm', AverageMeter()).avg:.3f}",
                    f"knnorm {meters.get('negative_key_norm', AverageMeter()).avg:.3f}",
                    f"miou {meters.get('mask_iou', AverageMeter()).avg:.4f}",
                    f"mdice {meters.get('mask_dice', AverageMeter()).avg:.4f}",
                    f"px {meters['px'].avg:.0f}",
                    f"ratio {meters.get('part_ratio_exact_frac', AverageMeter()).avg:.4f}",
                    f"riou {meters.get('render_iou', AverageMeter()).avg:.4f}",
                    f"{args.log_freq * args.batch_size * max(world_size, 1) / elapsed:.1f} img/s",
                ]
                print(" | ".join(msg), flush=True)

            if iteration % args.val_freq == 0:
                max_batches = None if int(args.val_max_batches) <= 0 else int(args.val_max_batches)
                rarp_val = evaluate(model, rarp_val_loader, device, args, max_batches=max_batches)
                if is_main_process():
                    print(
                        f"VAL_{PRIMARY_VAL_NAME} "
                        + " | ".join(f"{k} {v:.4f}" for k, v in rarp_val.items()),
                        flush=True,
                    )
                lnd_val = None
                if lnd_val_loader is not None:
                    lnd_val = evaluate(model, lnd_val_loader, device, args, max_batches=max_batches)
                    if is_main_process():
                        print("VAL_LND " + " | ".join(f"{k} {v:.4f}" for k, v in lnd_val.items()), flush=True)
                if is_main_process():
                    val_score = _val_score_from_results(args, rarp_val, lnd_val)
                    print(f"VAL_SCORE {_display_val_score_mode(args)} {val_score:.4f}", flush=True)
                    if bool(args.save_best_ckpt) and val_score < best_val_score:
                        best_val_score = val_score
                        save_checkpoint(
                            log_dir / "checkpoints" / "best_val_total.pt",
                            model,
                            optimizer,
                            scheduler,
                            args,
                            epoch,
                            iteration,
                        )
                        print(f"BEST_VAL_SCORE iter {iteration:07d} score {best_val_score:.4f}", flush=True)
                    if bool(args.save_val_iter_ckpt):
                        save_checkpoint(log_dir / "checkpoints" / f"iter{iteration:07d}.pt", model, optimizer, scheduler, args, epoch, iteration)
                model.train()

            if iteration % args.ckpt_freq == 0 and is_main_process():
                save_checkpoint(log_dir / "checkpoints" / "last.pt", model, optimizer, scheduler, args, epoch, iteration)

            if iteration >= args.max_iter:
                break
        epoch += 1

    if is_main_process():
        save_checkpoint(log_dir / "checkpoints" / "last.pt", model, optimizer, scheduler, args, epoch, iteration)
        print(f"Finished training at iter {iteration}", flush=True)
    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--save_dir", type=str, default=str(ROBOPEPP_ROOT / "logs"))
    parser.add_argument("--name", type=str, default="surfemb_resnet_crop224_rarp_lnd_refinemem")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--resume_optimizer", type=int, default=1, choices=[0, 1])
    parser.add_argument("--resume_scheduler", type=int, default=1, choices=[0, 1])
    parser.add_argument("--reset_iter_on_resume", type=int, default=0, choices=[0, 1])
    parser.add_argument("--inspect_datasets_only", type=int, default=0, choices=[0, 1])
    parser.add_argument("--inspect_vis_dir", type=str, default=None)
    parser.add_argument("--inspect_samples_per_dataset", type=int, default=3)
    parser.add_argument("--inspect_render_mesh", type=int, default=1, choices=[0, 1])

    parser.add_argument("--needle_puncture_data_dir", type=str, default=PUNCTURE_DATASET_ROOT)
    parser.add_argument("--needle_puncture_pose_dir", type=str, default=PUNCTURE_POSE_ROOT)
    parser.add_argument("--needle_grasping_data_dir", type=str, default=GRASPING_DATASET_ROOT)
    parser.add_argument("--needle_grasping_pose_dir", type=str, default=GRASPING_POSE_ROOT)
    parser.add_argument("--knotting_data_dir", type=str, default=KNOTTING_DATASET_ROOT)
    parser.add_argument("--knotting_pose_dir", type=str, default=KNOTTING_POSE_ROOT)
    parser.add_argument("--train_dataset_names", type=str, default="needlePuncture,needleGrasping,knotting")
    parser.add_argument("--min_dice_shaft", type=float, default=0.8)
    parser.add_argument("--min_dice_wrist", type=float, default=0.6)
    parser.add_argument("--min_dice_gripper", type=float, default=0.6)
    parser.add_argument("--needle_train_ratio", type=float, default=0.95)
    parser.add_argument("--canonicalize_pose_symmetry", type=int, default=1, choices=[0, 1])
    parser.add_argument("--canonical_eps", type=float, default=0.08)
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--bbox_padding_frac", type=float, default=0.12)
    parser.add_argument("--dataset_cache_dir", type=str, default=None)
    parser.add_argument("--surface_points_path", type=str, default=str(DEFAULT_SURFACE_POINTS))
    parser.add_argument("--surfemb_n_pos", type=int, default=1024)
    parser.add_argument("--surfemb_n_neg", type=int, default=1024)
    parser.add_argument("--surfemb_shaft_sample_ratio", type=float, default=0.20)
    parser.add_argument("--surfemb_wrist_sample_ratio", type=float, default=0.60)
    parser.add_argument("--surfemb_gripper_sample_ratio", type=float, default=0.20)
    parser.add_argument("--surfemb_key_noise", type=float, default=1e-3)
    parser.add_argument("--surfemb_similarity", choices=("raw_dot", "cosine"), default="raw_dot")
    parser.add_argument("--surfemb_temperature", type=float, default=1.0)
    parser.add_argument("--surfemb_crop_scale", type=float, default=1.2)
    parser.add_argument("--surfemb_max_angle", type=float, default=3.141592653589793)
    parser.add_argument("--surfemb_offset_scale", type=float, default=1.0)
    parser.add_argument("--surfemb_min_depth", type=float, default=1e-4)
    parser.add_argument("--surfemb_depth_tolerance", type=float, default=8e-4)
    parser.add_argument("--surfemb_use_mesh_zbuffer", type=int, default=1, choices=[0, 1])
    parser.add_argument("--surfemb_zbuffer_backend", type=str, default="opengl", choices=["opengl"])
    parser.add_argument("--surfemb_min_train_render_iou", type=float, default=0.35)
    parser.add_argument("--heatmap_sigma", type=float, default=2.0)

    parser.add_argument("--include_lnd_train", type=int, default=1, choices=[0, 1])
    parser.add_argument("--include_lnd_val", type=int, default=1, choices=[0, 1])
    parser.add_argument("--lnd_root", type=str, default=DEFAULT_LND_ROOT)
    parser.add_argument("--lnd_refine_memory", type=str, default=DEFAULT_LND_REFINE_MEMORY)
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
    parser.add_argument("--val_score_mode", type=str, default="rarp_total", choices=["rarp_total", "mean_total"])
    parser.add_argument("--ckpt_freq", type=int, default=1000)
    parser.add_argument("--validate_before_train", type=int, default=0, choices=[0, 1])
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
    main(parser.parse_args())
