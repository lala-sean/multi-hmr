import os
import sys
import time
import importlib.util
from argparse import ArgumentParser
from pathlib import Path

import torch
import torch.distributed as dist
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


_dataset_module = _load_local_module("robopepp_hcce_crop_dataset", ROBOPEPP_ROOT / "datasets" / "rarp_hcce_crop.py")
_loss_module = _load_local_module("robopepp_hcce_crop_loss", ROBOPEPP_ROOT / "loss_hcce_crop.py")
_model_module = _load_local_module("robopepp_hcce_crop_model", ROBOPEPP_ROOT / "models" / "hcce_crop_model.py")
RARPCropHCCEDataset = _dataset_module.RARPCropHCCEDataset
collate_fn_rarp_crop_hcce = _dataset_module.collate_fn_rarp_crop_hcce
compute_crop_hcce_losses = _loss_module.compute_crop_hcce_losses
CropHCCEDenseKeypointDPT = _model_module.CropHCCEDenseKeypointDPT


PUNCTURE_DATASET_ROOT = "/mnt/nas/share/shuojue/data/needlePuncture_videos"
PUNCTURE_POSE_ROOT = "/mnt/nas/share/shuojue/data/needlePuncture_results"
GRASPING_DATASET_ROOT = "/mnt/nas/share/shuojue/data/needleGrasping_videos"
GRASPING_POSE_ROOT = "/mnt/nas/share/shuojue/data/needleGrasping_results"
KNOTTING_DATASET_ROOT = "/mnt/nas/share/shuojue/data/knotting_videos"
KNOTTING_POSE_ROOT = "/mnt/nas/share/shuojue/data/knotting_results"


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


def setup_dist():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    return device, local_rank, world_size


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


def make_dataset(args, name, split, training, root, pose_root, subsample):
    return RARPCropHCCEDataset(
        root,
        pose_root,
        split=split,
        training=training,
        crop_size=224,
        train_ratio=args.needle_train_ratio,
        subsample=subsample,
        min_dice=(args.min_dice_shaft, args.min_dice_wrist, args.min_dice_gripper),
        canonicalize_pose_symmetry=bool(args.canonicalize_pose_symmetry),
        canonical_eps=args.canonical_eps,
        heatmap_sigma=args.heatmap_sigma,
        color_jitter=bool(args.color_jitter),
        rgb_augmentation=bool(args.rgb_augmentation),
        occlusion_augmentation=bool(args.occlusion_augmentation),
        occlusion_prob=args.occlusion_prob,
        cache_dir=args.dataset_cache_dir,
        cse_coord_root=args.cse_coord_root,
        render_on_the_fly=bool(args.render_on_the_fly),
        coord_render_backend=args.coord_render_backend,
        require_cse=True,
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
    return [
        make_dataset(args, name, "train", True, specs[name][0], specs[name][1], args.train_subsample)
        for name in names
    ]


def save_checkpoint(path, model, optimizer, scheduler, scaler, args, epoch, iteration):
    raw = model.module if isinstance(model, DDP) else model
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": raw.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "args": vars(args),
            "epoch": epoch,
            "iter": iteration,
        },
        path,
    )


@torch.no_grad()
def evaluate(model, loader, device, args, max_batches=20):
    model.eval()
    meters = {}
    count = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in y.items()}
        with torch.cuda.amp.autocast(enabled=bool(args.amp)):
            out = model(x, y["K"])
            _, metrics = compute_crop_hcce_losses(out, y, args)
        for key, value in metrics.items():
            meters[key] = meters.get(key, 0.0) + float(value.item())
        count += 1
        if count >= max_batches:
            break
    if count == 0:
        return {}
    return {f"val_{k}": reduce_float(v / count, device) for k, v in meters.items()}


def main(args):
    if args.img_size != 224:
        raise ValueError("This crop-HCCE runner is fixed to --img_size 224.")
    device, local_rank, world_size = setup_dist()
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
            val_dataset = make_dataset(
                args,
                "needlePuncture",
                "test",
                False,
                args.needle_puncture_data_dir,
                args.needle_puncture_pose_dir,
                args.val_subsample,
            )
            dist.barrier()
        else:
            dist.barrier()
            train_datasets = make_train_datasets(args)
            val_dataset = make_dataset(
                args,
                "needlePuncture",
                "test",
                False,
                args.needle_puncture_data_dir,
                args.needle_puncture_pose_dir,
                args.val_subsample,
            )
    else:
        train_datasets = make_train_datasets(args)
        val_dataset = make_dataset(
            args,
            "needlePuncture",
            "test",
            False,
            args.needle_puncture_data_dir,
            args.needle_puncture_pose_dir,
            args.val_subsample,
        )

    train_dataset = ConcatDataset(train_datasets)
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if use_ddp else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if use_ddp else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn_rarp_crop_hcce,
        # Dataset epoch drives bbox jitter; persistent worker copies would keep
        # the epoch value from worker initialization.
        persistent_workers=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn_rarp_crop_hcce,
        persistent_workers=False,
    )

    model = CropHCCEDenseKeypointDPT(
        img_size=224,
        backbone=args.backbone,
        pretrained_backbone=bool(args.pretrained_backbone),
        hcce_feat_dim=args.hcce_feat_dim,
        hcce_bits=args.hcce_bits,
        num_keypoints=5,
        pose_head_iter=args.pose_head_iter,
        pose_head_dropout=args.pose_head_dropout,
    ).to(device)
    resume_ckpt = None
    resume_epoch = 0
    resume_iter = 0
    if args.resume is not None:
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
        max_lr=[args.lr_backbone, args.lr_dense, args.lr_keypoint, args.lr_pose],
        total_steps=args.max_iter,
        pct_start=0.0,
        final_div_factor=args.final_div_factor,
        cycle_momentum=False,
    )
    scaler = None
    if resume_ckpt is not None and bool(args.resume_optimizer):
        if "optimizer_state_dict" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in resume_ckpt and resume_ckpt["scheduler_state_dict"] is not None and bool(args.resume_scheduler):
            scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])
        if scaler is not None and "scaler_state_dict" in resume_ckpt and resume_ckpt["scaler_state_dict"] is not None:
            scaler.load_state_dict(resume_ckpt["scaler_state_dict"])

    if is_main_process():
        print(f"LOG_DIR: {log_dir}", flush=True)
        print(f"WORLD_SIZE: {world_size}", flush=True)
        if resume_ckpt is not None:
            print(
                f"RESUME: epoch={resume_epoch} iter={resume_iter} "
                f"optimizer={bool(args.resume_optimizer)} scheduler={bool(args.resume_scheduler)}",
                flush=True,
            )
        print("MODEL: CropHCCEDenseKeypointDPT (no query/detection branch)", flush=True)
        print("DENSE RESOLUTION: 224x224 for instance/part/HCCE/keypoint", flush=True)
        print("PART LABELS: dataset 1=gripper,2=wrist,3=shaft; logits [wrist,gripper,shaft]", flush=True)
        for ds in train_datasets:
            print(ds, flush=True)
        print(val_dataset, flush=True)

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
            y = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in y.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=bool(args.amp), dtype=torch.bfloat16):
                out = model(x, y["K"])
                loss, metrics = compute_crop_hcce_losses(out, y, args)
            if not torch.isfinite(loss):
                if is_main_process():
                    bad = {k: float(v.detach().float().cpu().item()) for k, v in metrics.items() if torch.is_tensor(v) and v.numel() == 1}
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
                    f"hcce {meters['hcce'].avg:.4f}",
                    f"bit {meters['hcce_bit_acc'].avg:.3f}",
                    f"hm {meters['heatmap'].avg:.4f}",
                    f"kp {meters['kp_err_px'].avg:.2f}px",
                    f"uv {meters['uv_err_px'].avg:.2f}px",
                    f"{args.log_freq * args.batch_size * max(world_size, 1) / elapsed:.1f} img/s",
                ]
                print(" | ".join(msg), flush=True)

            if iteration % args.val_freq == 0:
                val = evaluate(model, val_loader, device, args, max_batches=args.val_batches)
                if is_main_process():
                    print("VAL " + " | ".join(f"{k} {v:.4f}" for k, v in val.items()), flush=True)
                    save_checkpoint(
                        log_dir / "checkpoints" / f"iter{iteration:07d}.pt",
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        args,
                        epoch,
                        iteration,
                    )
                model.train()

            if iteration % args.ckpt_freq == 0 and is_main_process():
                save_checkpoint(
                    log_dir / "checkpoints" / "last.pt",
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    args,
                    epoch,
                    iteration,
                )

            if iteration >= args.max_iter:
                break
        epoch += 1

    if is_main_process():
        save_checkpoint(log_dir / "checkpoints" / "last.pt", model, optimizer, scheduler, scaler, args, epoch, iteration)
        print(f"Finished training at iter {iteration}", flush=True)
    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--save_dir", type=str, default=str(ROBOPEPP_ROOT / "logs"))
    parser.add_argument("--name", type=str, default="hcce_crop224_keypointnet_rarp_gpu0123")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--resume_optimizer", type=int, default=1, choices=[0, 1])
    parser.add_argument("--resume_scheduler", type=int, default=1, choices=[0, 1])
    parser.add_argument("--reset_iter_on_resume", type=int, default=0, choices=[0, 1])
    parser.add_argument("--img_size", type=int, default=224)
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
    parser.add_argument("--cse_coord_root", type=str, default=None)
    parser.add_argument("--render_on_the_fly", type=int, default=1, choices=[0, 1])
    parser.add_argument("--coord_render_backend", type=str, default="trimesh", choices=["trimesh", "gaussian"])
    parser.add_argument("--dataset_cache_dir", type=str, default=None)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--val_batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--train_subsample", type=int, default=1)
    parser.add_argument("--val_subsample", type=int, default=20)
    parser.add_argument("--max_iter", type=int, default=60000)
    parser.add_argument("--log_freq", type=int, default=10)
    parser.add_argument("--val_freq", type=int, default=1000)
    parser.add_argument("--val_batches", type=int, default=20)
    parser.add_argument("--ckpt_freq", type=int, default=1000)
    parser.add_argument("--amp", type=int, default=1, choices=[0, 1])
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--backbone", type=str, default="dinov2_vits14", choices=["dinov2_vits14", "dinov2_vitb14"])
    parser.add_argument("--pretrained_backbone", type=int, default=1, choices=[0, 1])
    parser.add_argument("--hcce_feat_dim", type=int, default=256)
    parser.add_argument("--hcce_bits", type=int, default=8)
    parser.add_argument("--pose_head_iter", type=int, default=4)
    parser.add_argument("--pose_head_dropout", type=float, default=0.3)

    parser.add_argument("--lr_backbone", type=float, default=5e-5)
    parser.add_argument("--lr_dense", type=float, default=1e-4)
    parser.add_argument("--lr_keypoint", type=float, default=1e-4)
    parser.add_argument("--lr_pose", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-7)
    parser.add_argument("--final_div_factor", type=float, default=1e4)

    parser.add_argument("--alpha_dice", type=float, default=5.0)
    parser.add_argument("--alpha_bce_mask", type=float, default=2.0)
    parser.add_argument("--alpha_part", type=float, default=2.0)
    parser.add_argument("--alpha_hcce", type=float, default=1.0)
    parser.add_argument("--alpha_heatmap", type=float, default=1.0)
    parser.add_argument("--alpha_action_l1", type=float, default=1.0)
    parser.add_argument("--alpha_wrist_quat_l1", type=float, default=1.0)
    parser.add_argument("--alpha_wrist_trans_l1", type=float, default=10.0)
    parser.add_argument("--alpha_keypoint_2d", type=float, default=1.0)
    parser.add_argument("--alpha_keypoint_3d", type=float, default=0.0)
    parser.add_argument("--hcce_coord_min", type=float, default=-1.0)
    parser.add_argument("--hcce_coord_max", type=float, default=1.0)
    parser.add_argument("--heatmap_sigma", type=float, default=2.0)

    parser.add_argument("--color_jitter", type=int, default=1, choices=[0, 1])
    parser.add_argument("--rgb_augmentation", type=int, default=1, choices=[0, 1])
    parser.add_argument("--occlusion_augmentation", type=int, default=1, choices=[0, 1])
    parser.add_argument("--occlusion_prob", type=float, default=0.5)
    main(parser.parse_args())
