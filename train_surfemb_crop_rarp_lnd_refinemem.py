import os
import sys
import time
import importlib.util
from argparse import ArgumentParser
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import ConcatDataset, DataLoader, Dataset
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


_surfemb_dataset_module = _load_local_module(
    "robopepp_surfemb_crop_dataset_lndrun",
    ROBOPEPP_ROOT / "datasets" / "surfemb_hcce_crop.py",
)
_loss_module = _load_local_module("robopepp_surfemb_crop_loss_lndrun", ROBOPEPP_ROOT / "loss_surfemb_crop.py")
_model_module = _load_local_module(
    "robopepp_surfemb_crop_model_lndrun",
    ROBOPEPP_ROOT / "models" / "surfemb_crop_model.py",
)

RARPCropHCCEDataset = _surfemb_dataset_module.RARPCropHCCEDataset
SurgripeLNDHCCECropDataset = _surfemb_dataset_module.SurgripeLNDHCCECropDataset
SurfEmbCropAugmentedDataset = _surfemb_dataset_module.SurfEmbCropAugmentedDataset
collate_fn_surfemb_crop = _surfemb_dataset_module.collate_fn_surfemb_crop
save_surfemb_crop_debug_panel = _surfemb_dataset_module.save_surfemb_crop_debug_panel
compute_crop_surfemb_losses = _loss_module.compute_crop_surfemb_losses
SurfEmbCropDenseKeypointDPT = _model_module.SurfEmbCropDenseKeypointDPT


PUNCTURE_DATASET_ROOT = "/mnt/nas/share/shuojue/data/needlePuncture_videos"
PUNCTURE_POSE_ROOT = "/mnt/nas/share/shuojue/data/needlePuncture_results"
GRASPING_DATASET_ROOT = "/mnt/nas/share/shuojue/data/needleGrasping_videos"
GRASPING_POSE_ROOT = "/mnt/nas/share/shuojue/data/needleGrasping_results"
KNOTTING_DATASET_ROOT = "/mnt/nas/share/shuojue/data/knotting_videos"
KNOTTING_POSE_ROOT = "/mnt/nas/share/shuojue/data/knotting_results"
DEFAULT_LND_ROOT = "/mnt/iMVR/daiyun/Dataset/LND"
DEFAULT_LND_REFINE_MEMORY = (
    "/mnt/iMVR/daiyun/shuojue-temp/code/multi-hmr/submodules/gaussian-mesh-splatting/"
    "Results2/surgripe_lnd_refine_memory_train/TRAIN/refine_memory_pool.json"
)


class RepeatDataset(Dataset):
    def __init__(self, dataset, repeats):
        self.dataset = dataset
        self.repeats = max(1, int(repeats))

    def __len__(self):
        return len(self.dataset) * self.repeats

    def __getitem__(self, idx):
        return self.dataset[int(idx) % len(self.dataset)]

    def set_epoch(self, epoch):
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)

    def __repr__(self):
        return f"repeat_dataset(repeats={self.repeats}, base={self.dataset})"


class AverageMeter:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    @property
    def avg(self):
        return self.sum / max(1, self.count)

    def update(self, value, n=1):
        self.sum += float(value) * int(n)
        self.count += int(n)


def setup_dist(timeout_sec):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", timeout=timedelta(seconds=int(timeout_sec)))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    return device, local_rank, world_size


def dist_barrier(local_rank):
    if dist.is_available() and dist.is_initialized():
        ids = [int(local_rank)] if torch.cuda.is_available() else None
        dist.barrier(device_ids=ids)


def is_main_process():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def reduce_float(value, device):
    t = torch.tensor(float(value), device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= dist.get_world_size()
    return float(t.item())


def set_epoch_recursive(dataset, epoch):
    if hasattr(dataset, "set_epoch"):
        dataset.set_epoch(epoch)
    if isinstance(dataset, ConcatDataset):
        for child in dataset.datasets:
            set_epoch_recursive(child, epoch)


def make_rarp_dataset(args, name, split, training, root, pose_root, subsample):
    base = RARPCropHCCEDataset(
        root,
        pose_root,
        split=split,
        training=training,
        crop_size=args.crop_size,
        train_ratio=args.needle_train_ratio,
        subsample=subsample,
        min_dice=(args.min_dice_shaft, args.min_dice_wrist, args.min_dice_gripper),
        canonicalize_pose_symmetry=bool(args.canonicalize_pose_symmetry),
        canonical_eps=args.canonical_eps,
        heatmap_sigma=args.heatmap_sigma,
        bbox_padding_frac=args.bbox_padding_frac,
        color_jitter=False,
        rgb_augmentation=False,
        occlusion_augmentation=False,
        occlusion_prob=0.0,
        bbox_jitter=False,
        bbox_shift=False,
        cache_dir=args.dataset_cache_dir,
        cse_coord_root=args.cse_coord_root,
        render_on_the_fly=bool(args.render_on_the_fly),
        coord_render_backend=args.coord_render_backend,
        require_cse=True,
    )
    return SurfEmbCropAugmentedDataset(
        base,
        surface_points_path=args.surface_points_path,
        crop_size=args.crop_size,
        n_pos=args.surfemb_n_pos,
        n_neg=args.surfemb_n_neg,
        crop_scale=args.surfemb_crop_scale,
        max_angle=args.surfemb_max_angle,
        offset_scale=args.surfemb_offset_scale,
        training=training,
        heatmap_sigma=args.heatmap_sigma,
        ensure_full_mask=bool(args.surfemb_ensure_full_mask),
    )


def make_lnd_dataset(args, split, training, use_memory_pose, subsample):
    base = SurgripeLNDHCCECropDataset(
        root=args.lnd_root,
        split=split,
        training=training,
        crop_size=args.crop_size,
        memory_path=args.lnd_refine_memory if use_memory_pose else None,
        use_memory_pose=use_memory_pose,
        canonicalize_pose_symmetry=bool(args.canonicalize_pose_symmetry),
        canonical_eps=args.canonical_eps,
        heatmap_sigma=args.heatmap_sigma,
        bbox_padding_frac=args.bbox_padding_frac,
        color_jitter=False,
        rgb_augmentation=False,
        occlusion_augmentation=False,
        occlusion_prob=0.0,
        bbox_jitter=False,
        bbox_shift=False,
        subsample=subsample,
        render_on_the_fly=bool(args.render_on_the_fly),
        coord_render_backend=args.coord_render_backend,
        require_cse=True,
    )
    return SurfEmbCropAugmentedDataset(
        base,
        surface_points_path=args.surface_points_path,
        crop_size=args.crop_size,
        n_pos=args.surfemb_n_pos,
        n_neg=args.surfemb_n_neg,
        crop_scale=args.surfemb_crop_scale,
        max_angle=args.surfemb_max_angle,
        offset_scale=args.surfemb_offset_scale,
        training=training,
        heatmap_sigma=args.heatmap_sigma,
        ensure_full_mask=bool(args.surfemb_ensure_full_mask),
    )


def make_train_datasets(args):
    specs = {
        "needlePuncture": (args.needle_puncture_data_dir, args.needle_puncture_pose_dir),
        "needleGrasping": (args.needle_grasping_data_dir, args.needle_grasping_pose_dir),
        "knotting": (args.knotting_data_dir, args.knotting_pose_dir),
    }
    names = [n.strip() for n in args.train_dataset_names.split(",") if n.strip()]
    unknown = sorted(set(names) - set(specs))
    if unknown:
        raise ValueError(f"Unknown train datasets: {unknown}; choices={sorted(specs)}")
    rarp_datasets = [
        make_rarp_dataset(args, name, "train", True, specs[name][0], specs[name][1], args.train_subsample)
        for name in names
    ]
    datasets = list(rarp_datasets)
    if bool(args.include_lnd_train):
        lnd_train = make_lnd_dataset(args, "TRAIN", True, True, args.lnd_train_subsample)
        rarp_len = sum(len(ds) for ds in rarp_datasets)
        target_lnd_len = (float(args.lnd_sampling_rate) / max(1e-8, 1.0 - float(args.lnd_sampling_rate))) * rarp_len
        repeats = max(1, int(round(target_lnd_len / max(1, len(lnd_train)))))
        datasets.append(RepeatDataset(lnd_train, repeats))
    return datasets


def make_rarp_val_dataset(args):
    return make_rarp_dataset(
        args,
        "needlePuncture",
        "test",
        False,
        args.needle_puncture_data_dir,
        args.needle_puncture_pose_dir,
        args.val_subsample,
    )


def make_lnd_val_dataset(args):
    if not bool(args.include_lnd_val):
        return None
    return make_lnd_dataset(args, "TEST", False, False, args.lnd_val_subsample)


def save_checkpoint(path, model, optimizer, scheduler, args, epoch, iteration):
    raw = model.module if isinstance(model, DDP) else model
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": raw.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "args": vars(args),
            "epoch": epoch,
            "iter": iteration,
        },
        path,
    )


@torch.no_grad()
def evaluate(model, loader, device, args, max_batches=None):
    model.eval()
    meters = {}
    count = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in y.items()}
        with torch.amp.autocast("cuda", enabled=bool(args.amp), dtype=torch.bfloat16):
            out = model(x, y["K"])
            _, metrics = compute_crop_surfemb_losses(out, y, args)
        for key, value in metrics.items():
            meters[key] = meters.get(key, 0.0) + float(value.item())
        count += 1
        if max_batches is not None and count >= int(max_batches):
            break
    if count == 0:
        return {}
    return {f"val_{k}": reduce_float(v / count, device) for k, v in meters.items()}


def to_device(batch_y, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch_y.items()}


def main(args):
    if args.crop_size != 224:
        raise ValueError("This crop-SurfEmb runner is fixed to --crop_size 224.")
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
            sample_sets.append(("rarp_val", rarp_val_dataset))
            if lnd_val_dataset is not None:
                sample_sets.append(("lnd_val", lnd_val_dataset))
            for label, ds in sample_sets:
                x, y = ds[0]
                print(
                    f"INSPECT_SAMPLE {label}: x={tuple(x.shape)} "
                    f"inst={tuple(y['inst_mask'].shape)} part={tuple(y['part_mask'].shape)} "
                    f"coord={tuple(y['coord_img'].shape)} has_cse={bool(y['has_cse'])} "
                    f"surf_pos={tuple(y['surfemb_mask_samples'].shape)} "
                    f"surf_neg={tuple(y['surfemb_surface_samples'].shape)} "
                    f"K00={float(y['K'][0, 0]):.3f} "
                    f"kpt_proj_resid={float(y['surfemb_kpt_proj_resid_px']):.6f}px",
                    flush=True,
                )
                if args.inspect_vis_dir:
                    vis_dir = Path(args.inspect_vis_dir)
                    vis_dir.mkdir(parents=True, exist_ok=True)
                    save_surfemb_crop_debug_panel(vis_dir / f"{label}_surfemb_crop.jpg", y)
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
        collate_fn=collate_fn_surfemb_crop,
        # Dataset epoch drives bbox jitter; persistent worker copies would keep
        # the epoch value from worker initialization.
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
        collate_fn=collate_fn_surfemb_crop,
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
            collate_fn=collate_fn_surfemb_crop,
            persistent_workers=False,
        )

    model = SurfEmbCropDenseKeypointDPT(
        img_size=224,
        backbone=args.backbone,
        pretrained_backbone=bool(args.pretrained_backbone),
        dense_feat_dim=args.dense_feat_dim,
        surfemb_emb_dim=args.surfemb_emb_dim,
        surfemb_mlp_hidden_features=args.surfemb_mlp_hidden_features,
        surfemb_mlp_hidden_layers=args.surfemb_mlp_hidden_layers,
        num_keypoints=5,
        pose_head_iter=args.pose_head_iter,
        pose_head_dropout=args.pose_head_dropout,
        keypoint_feat_size=args.keypoint_feat_size,
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
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    raw = model.module if isinstance(model, DDP) else model

    optimizer = torch.optim.AdamW(
        [
            {"params": raw.encoder.parameters(), "lr": args.lr_backbone},
            {"params": raw.dense_head.parameters(), "lr": args.lr_dense},
            {"params": raw.surface_key_mlp.parameters(), "lr": args.lr_surfemb_mlp},
            {"params": raw.keypoint_net.parameters(), "lr": args.lr_keypoint},
            {
                "params": list(raw.action_head.parameters()) + list(raw.wrist_pose_head.parameters()),
                "lr": args.lr_pose,
            },
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[args.lr_backbone, args.lr_dense, args.lr_surfemb_mlp, args.lr_keypoint, args.lr_pose],
        total_steps=args.max_iter,
        pct_start=0.0,
        final_div_factor=args.final_div_factor,
        cycle_momentum=False,
    )
    if resume_ckpt is not None and bool(args.resume_optimizer):
        optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        if resume_ckpt.get("scheduler_state_dict") is not None and bool(args.resume_scheduler):
            scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])

    if is_main_process():
        print(f"LOG_DIR: {log_dir}", flush=True)
        print(f"WORLD_SIZE: {world_size}", flush=True)
        print("MODEL: SurfEmbCropDenseKeypointDPT, RARP+SurgRIPE-LND refine-memory SurfEmb crop training", flush=True)
        print("DENSE RESOLUTION: 224x224 for instance/part/SurfEmb/keypoint", flush=True)
        print(
            "SURFEMB AUG: GaussianBlur/ISONoise/GaussNoise/DebayerArtefacts/"
            "Unsharpen/CLAHE/CoarseDropout/ColorJitter(hue)/RandomRotatedMaskCrop",
            flush=True,
        )
        print(f"SURFACE_POINTS: {args.surface_points_path}", flush=True)
        print(f"LND_SAMPLING_RATE: {args.lnd_sampling_rate:.3f}", flush=True)
        for ds in train_datasets:
            print(ds, flush=True)
        print(rarp_val_dataset, flush=True)
        if lnd_val_dataset is not None:
            print(lnd_val_dataset, flush=True)

    iteration = resume_iter
    epoch = resume_epoch
    last_log = time.time()
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
            with torch.amp.autocast("cuda", enabled=bool(args.amp), dtype=torch.bfloat16):
                out = model(x, y["K"])
                loss, metrics = compute_crop_surfemb_losses(out, y, args)
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
                    f"dice {meters['dice'].avg:.4f}",
                    f"part {meters['part_ce'].avg:.4f}",
                    f"part_acc {meters['part_acc'].avg:.3f}",
                    f"surf {meters['surfemb'].avg:.4f}",
                    f"nce {meters['surfemb_nce'].avg:.4f}",
                    f"hm {meters['heatmap'].avg:.4f}",
                    f"kp {meters['kp_err_px'].avg:.2f}px",
                    f"uv {meters['uv_err_px'].avg:.2f}px",
                    f"proj {meters.get('kpt_proj_resid_px', AverageMeter()).avg:.4f}px",
                    f"{args.log_freq * args.batch_size * max(world_size, 1) / elapsed:.1f} img/s",
                ]
                print(" | ".join(msg), flush=True)

            if iteration % args.val_freq == 0:
                max_batches = None if int(args.val_max_batches) <= 0 else int(args.val_max_batches)
                rarp_val = evaluate(model, rarp_val_loader, device, args, max_batches=max_batches)
                if is_main_process():
                    print("VAL_RARP " + " | ".join(f"{k} {v:.4f}" for k, v in rarp_val.items()), flush=True)
                if lnd_val_loader is not None:
                    lnd_val = evaluate(model, lnd_val_loader, device, args, max_batches=max_batches)
                    if is_main_process():
                        print("VAL_LND " + " | ".join(f"{k} {v:.4f}" for k, v in lnd_val.items()), flush=True)
                if is_main_process():
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
    parser.add_argument("--name", type=str, default="surfemb_crop224_rarp_lnd_refinemem_bs56")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--resume_optimizer", type=int, default=1, choices=[0, 1])
    parser.add_argument("--resume_scheduler", type=int, default=1, choices=[0, 1])
    parser.add_argument("--reset_iter_on_resume", type=int, default=0, choices=[0, 1])
    parser.add_argument("--inspect_datasets_only", type=int, default=0, choices=[0, 1])
    parser.add_argument("--inspect_vis_dir", type=str, default=None)

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
    parser.add_argument("--color_jitter", type=int, default=1, choices=[0, 1])
    parser.add_argument("--rgb_augmentation", type=int, default=1, choices=[0, 1])
    parser.add_argument("--occlusion_augmentation", type=int, default=1, choices=[0, 1])
    parser.add_argument("--occlusion_prob", type=float, default=0.5)
    parser.add_argument("--cse_coord_root", type=str, default=None)
    parser.add_argument("--render_on_the_fly", type=int, default=1, choices=[0, 1])
    parser.add_argument("--coord_render_backend", type=str, default="trimesh", choices=["trimesh"])
    parser.add_argument("--dataset_cache_dir", type=str, default=None)
    parser.add_argument(
        "--surface_points_path",
        type=str,
        default=str(
            ROBOPEPP_ROOT
            / "assets"
            / "instrument_surface_samples_surfemb_x2.13mm_wg1over3_shafttop30mm"
            / "instrument_surface_points_all.npy"
        ),
    )
    parser.add_argument("--surfemb_n_pos", type=int, default=1024)
    parser.add_argument("--surfemb_n_neg", type=int, default=1024)
    parser.add_argument("--surfemb_key_noise", type=float, default=1e-3)
    parser.add_argument("--surfemb_crop_scale", type=float, default=1.2)
    parser.add_argument("--surfemb_max_angle", type=float, default=3.141592653589793)
    parser.add_argument("--surfemb_offset_scale", type=float, default=1.0)
    parser.add_argument("--surfemb_ensure_full_mask", type=int, default=1, choices=[0, 1])

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
    parser.add_argument("--ckpt_freq", type=int, default=1000)
    parser.add_argument("--amp", type=int, default=1, choices=[0, 1])
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--dist_timeout_sec", type=int, default=7200)

    parser.add_argument("--backbone", type=str, default="dinov2_vits14", choices=["dinov2_vits14", "dinov2_vitb14"])
    parser.add_argument("--pretrained_backbone", type=int, default=1, choices=[0, 1])
    parser.add_argument("--dense_feat_dim", type=int, default=256)
    parser.add_argument("--surfemb_emb_dim", type=int, default=12)
    parser.add_argument("--surfemb_mlp_hidden_features", type=int, default=256)
    parser.add_argument("--surfemb_mlp_hidden_layers", type=int, default=2)
    parser.add_argument("--pose_head_iter", type=int, default=4)
    parser.add_argument("--pose_head_dropout", type=float, default=0.3)
    parser.add_argument("--keypoint_feat_size", type=int, default=14)

    parser.add_argument("--lr_backbone", type=float, default=1e-4)
    parser.add_argument("--lr_dense", type=float, default=1e-4)
    parser.add_argument("--lr_surfemb_mlp", type=float, default=3e-5)
    parser.add_argument("--lr_keypoint", type=float, default=1e-4)
    parser.add_argument("--lr_pose", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-7)
    parser.add_argument("--final_div_factor", type=float, default=1e4)

    parser.add_argument("--alpha_dice", type=float, default=5.0)
    parser.add_argument("--alpha_bce_mask", type=float, default=2.0)
    parser.add_argument("--alpha_part", type=float, default=2.0)
    parser.add_argument("--alpha_surfemb", type=float, default=1.0)
    parser.add_argument("--alpha_heatmap", type=float, default=1.0)
    parser.add_argument("--alpha_action_l1", type=float, default=1.0)
    parser.add_argument("--alpha_wrist_quat_l1", type=float, default=1.0)
    parser.add_argument("--alpha_wrist_trans_l1", type=float, default=10.0)
    parser.add_argument("--alpha_keypoint_2d", type=float, default=0.0)
    parser.add_argument("--alpha_keypoint_3d", type=float, default=0.0)
    parser.add_argument("--heatmap_sigma", type=float, default=2.0)
    main(parser.parse_args())
