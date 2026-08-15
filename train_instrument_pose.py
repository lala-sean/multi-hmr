import math
import os
import sys
import time
import importlib.util
from argparse import ArgumentParser
from pathlib import Path
from multiprocessing import Value

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

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


_rarp_module = _load_local_module("robopepp_rarp_instrument", ROBOPEPP_ROOT / "datasets" / "rarp_instrument.py")
_lnd_module = _load_local_module("robopepp_surgripe_lnd_instrument", ROBOPEPP_ROOT / "datasets" / "surgripe_lnd_instrument.py")
_model_module = _load_local_module("robopepp_instrument_model", ROBOPEPP_ROOT / "models" / "instrument_model.py")
_geom_module = _load_local_module("robopepp_instrument_geometry", ROBOPEPP_ROOT / "instrument_geometry.py")
RoboPEPPRARPInstrument = _rarp_module.RoboPEPPRARPInstrument
RoboPEPPSurgripeLNDInstrument = _lnd_module.RoboPEPPSurgripeLNDInstrument
make_robopepp_instrument_posenet = _model_module.make_robopepp_instrument_posenet
KEYPOINT_NAMES = _geom_module.KEYPOINT_NAMES


PUNCTURE_DATASET_ROOT = "/mnt/nas/share/shuojue/data/needlePuncture_videos"
PUNCTURE_POSE_ROOT = "/mnt/nas/share/shuojue/data/needlePuncture_results"
GRASPING_DATASET_ROOT = "/mnt/nas/share/shuojue/data/needleGrasping_videos"
GRASPING_POSE_ROOT = "/mnt/nas/share/shuojue/data/needleGrasping_results"
KNOTTING_DATASET_ROOT = "/mnt/nas/share/shuojue/data/knotting_videos"
KNOTTING_POSE_ROOT = "/mnt/nas/share/shuojue/data/knotting_results"
DEFAULT_JEPA_PATH = str(ROBOPEPP_ROOT / "pretrained" / "robopepp_original" / "jepa_joints-ep200.pth.tar")
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


class RoboPEPPMaskCollator:
    def __init__(
        self,
        input_size=224,
        patch_size=16,
        enc_mask_scale=(0.85, 1.0),
        pred_mask_scale=(0.0, 0.2),
        aspect_ratio=(0.75, 1.0),
        nenc=1,
        npred=1,
        min_keep=-1,
    ):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.patch_size = int(patch_size)
        self.height = input_size[0] // patch_size
        self.width = input_size[1] // patch_size
        self.enc_mask_scale = enc_mask_scale
        self.pred_mask_scale = pred_mask_scale
        self.aspect_ratio = aspect_ratio
        self.nenc = nenc
        self.npred = npred
        self.min_keep = min_keep
        self._itr_counter = Value("i", -1)

    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            return i.value

    def _sample_block_size(self, generator, scale, aspect_ratio_scale):
        rand = torch.rand(1, generator=generator).item()
        min_s, max_s = scale
        mask_scale = min_s + rand * (max_s - min_s)
        max_keep = int(self.height * self.width * mask_scale)
        min_ar, max_ar = aspect_ratio_scale
        aspect = min_ar + rand * (max_ar - min_ar)
        h = int(round(math.sqrt(max_keep * aspect)))
        w = int(round(math.sqrt(max_keep / aspect)))
        while h >= self.height:
            h -= 1
        while w >= self.width:
            w -= 1
        return h, w

    def _sample_block_mask(self, block_size):
        h, w = block_size
        valid = False
        timeout = 20
        while not valid:
            top = torch.randint(0, self.height - h, (1,))
            left = torch.randint(0, self.width - w, (1,))
            mask = torch.zeros((self.height, self.width), dtype=torch.int64)
            mask[top : top + h, left : left + w] = 1
            mask = torch.nonzero(mask.flatten()).squeeze(-1)
            valid = len(mask) > self.min_keep
            timeout -= 1
            if timeout <= 0:
                raise RuntimeError("RoboPEPP mask generator failed to sample a valid block")
        return mask

    def _collate_targets(self, targets):
        keys = [
            "action",
            "wrist_quat",
            "wrist_trans",
            "heatmaps",
            "keypoints_crop",
            "keypoints_3d_cam",
            "keypoints_valid",
            "K",
            "pose_sym_flipped",
        ]
        return {key: torch.stack([t[key] for t in targets], dim=0) for key in keys}

    def __call__(self, batch):
        images, targets = zip(*batch)
        x = torch.stack(images, dim=0)
        y = self._collate_targets(targets)

        B = len(batch)
        total_indices = torch.arange(self.height * self.width)
        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)
        pred_size = self._sample_block_size(g, self.pred_mask_scale, self.aspect_ratio)
        collated_masks_pred, collated_masks_enc = [], []
        for _ in range(B):
            masks_p = []
            for _ in range(self.npred):
                masks_p.append(self._sample_block_mask(pred_size))
            masks_p = torch.unique(torch.cat(masks_p, dim=0), dim=0)
            masks_p, _ = torch.sort(masks_p, dim=0)
            masks_c = torch.tensor([i for i in total_indices if i not in masks_p], dtype=torch.int64)
            collated_masks_pred.append([masks_p])
            collated_masks_enc.append([masks_c])
        masks_pred = torch.utils.data.default_collate(collated_masks_pred)
        masks_enc = torch.utils.data.default_collate(collated_masks_enc)
        return (x, y), masks_enc, masks_pred


def focal_heatmap_loss(output, target):
    pos_inds = target.eq(1).float()
    neg_inds = target.lt(1).float()
    neg_weights = torch.pow(1.0 - target, 4)
    output = torch.clamp(output.float(), 1e-3, 1.0 - 1e-3)
    pos_loss = torch.log(output) * torch.pow(1.0 - output, 2) * pos_inds
    neg_loss = torch.log(1.0 - output) * torch.pow(output, 2) * neg_weights * neg_inds
    num_pos = pos_inds.float().sum()
    if num_pos == 0:
        loss = -neg_loss.sum()
    else:
        loss = -(pos_loss.sum() + neg_loss.sum()) / num_pos
    return torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)


def heatmap_argmax(heatmaps):
    B, C, H, W = heatmaps.shape
    flat = heatmaps.view(B, C, -1)
    inds = flat.argmax(dim=-1)
    y = torch.div(inds, W, rounding_mode="floor")
    x = inds % W
    return torch.stack([x, y], dim=-1).float()


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


def set_epoch_recursive(dataset, epoch):
    if hasattr(dataset, "set_epoch"):
        dataset.set_epoch(epoch)
    if isinstance(dataset, ConcatDataset):
        for sub in dataset.datasets:
            set_epoch_recursive(sub, epoch)


def reduce_scalar(value, device):
    t = torch.tensor(float(value), device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= dist.get_world_size()
    return float(t.item())


def make_train_datasets(args):
    common = dict(
        split="train",
        training=True,
        crop_size=args.crop_size,
        train_ratio=args.needle_train_ratio,
        subsample=args.train_subsample,
        min_dice=(args.min_dice_shaft, args.min_dice_wrist, args.min_dice_gripper),
        canonicalize_pose_symmetry=bool(args.canonicalize_pose_symmetry),
        canonical_eps=args.canonical_eps,
        heatmap_sigma=args.heatmap_sigma,
        bbox_padding_frac=args.bbox_padding_frac,
        bbox_jitter=bool(args.bbox_jitter),
        bbox_shift=bool(args.bbox_shift),
        bbox_shift_max_px=args.bbox_shift_max_px,
        aug_random_crop_rotate=bool(args.aug_random_crop_rotate),
        aug_geom_prob=args.aug_geom_prob,
        aug_max_angle=args.aug_max_angle,
        color_jitter=bool(args.color_jitter),
        rgb_augmentation=bool(args.rgb_augmentation),
        occlusion_augmentation=bool(args.occlusion_augmentation),
        occlusion_prob=args.occlusion_prob,
        cache_dir=args.dataset_cache_dir,
    )
    specs = {
        "needlePuncture": (args.needle_puncture_data_dir, args.needle_puncture_pose_dir),
        "needleGrasping": (args.needle_grasping_data_dir, args.needle_grasping_pose_dir),
        "knotting": (args.knotting_data_dir, args.knotting_pose_dir),
    }
    names = [name.strip() for name in args.train_dataset_names.split(",") if name.strip()]
    unknown = sorted(set(names) - set(specs))
    if unknown:
        raise ValueError(f"Unknown --train_dataset_names entries: {unknown}; choices={sorted(specs)}")
    rarp_datasets = [RoboPEPPRARPInstrument(*specs[name], **common) for name in names]
    datasets = list(rarp_datasets)
    if bool(args.include_lnd_train):
        lnd_train = RoboPEPPSurgripeLNDInstrument(
            root=args.lnd_root,
            split="TRAIN",
            training=True,
            crop_size=args.crop_size,
            memory_path=args.lnd_refine_memory,
            use_memory_pose=True,
            canonicalize_pose_symmetry=bool(args.canonicalize_pose_symmetry),
            canonical_eps=args.canonical_eps,
            heatmap_sigma=args.heatmap_sigma,
            bbox_padding_frac=args.bbox_padding_frac,
            bbox_jitter=bool(args.lnd_bbox_jitter),
            bbox_shift=bool(args.bbox_shift),
            bbox_shift_max_px=args.bbox_shift_max_px,
            aug_random_crop_rotate=bool(args.aug_random_crop_rotate),
            aug_geom_prob=args.aug_geom_prob,
            aug_max_angle=args.aug_max_angle,
            color_jitter=bool(args.color_jitter),
            rgb_augmentation=bool(args.rgb_augmentation),
            occlusion_augmentation=bool(args.occlusion_augmentation),
            occlusion_prob=args.occlusion_prob,
            subsample=args.lnd_train_subsample,
        )
        rarp_len = sum(len(ds) for ds in rarp_datasets)
        target_lnd_len = (float(args.lnd_sampling_rate) / max(1e-8, 1.0 - float(args.lnd_sampling_rate))) * rarp_len
        repeats = max(1, int(round(target_lnd_len / max(1, len(lnd_train)))))
        datasets.append(RepeatDataset(lnd_train, repeats))
    return datasets


def make_val_dataset(args):
    return RoboPEPPRARPInstrument(
        args.needle_puncture_data_dir,
        args.needle_puncture_pose_dir,
        split="test",
        training=False,
        crop_size=args.crop_size,
        train_ratio=args.needle_train_ratio,
        subsample=args.val_subsample,
        min_dice=(args.min_dice_shaft, args.min_dice_wrist, args.min_dice_gripper),
        canonicalize_pose_symmetry=bool(args.canonicalize_pose_symmetry),
        canonical_eps=args.canonical_eps,
        heatmap_sigma=args.heatmap_sigma,
        bbox_padding_frac=args.bbox_padding_frac,
        aug_random_crop_rotate=False,
        color_jitter=bool(args.color_jitter),
        rgb_augmentation=bool(args.rgb_augmentation),
        occlusion_augmentation=bool(args.occlusion_augmentation),
        occlusion_prob=args.occlusion_prob,
        cache_dir=args.dataset_cache_dir,
    )


def make_lnd_val_dataset(args):
    return RoboPEPPSurgripeLNDInstrument(
        root=args.lnd_root,
        split="TEST",
        training=False,
        crop_size=args.crop_size,
        memory_path=None,
        use_memory_pose=False,
        canonicalize_pose_symmetry=bool(args.canonicalize_pose_symmetry),
        canonical_eps=args.canonical_eps,
        heatmap_sigma=args.heatmap_sigma,
        bbox_padding_frac=args.bbox_padding_frac,
        aug_random_crop_rotate=False,
        color_jitter=bool(args.color_jitter),
        rgb_augmentation=bool(args.rgb_augmentation),
        occlusion_augmentation=bool(args.occlusion_augmentation),
        occlusion_prob=args.occlusion_prob,
        subsample=args.lnd_val_subsample,
    )


def compute_losses(out, y, args):
    action_loss = F.l1_loss(out["action_pred"], y["action"])
    pred_quat = F.normalize(out["wrist_quat_pred"], p=2, dim=1)
    gt_quat = F.normalize(y["wrist_quat"], p=2, dim=1)
    sign = torch.where((pred_quat * gt_quat).sum(dim=1, keepdim=True) < 0.0, -1.0, 1.0)
    quat_loss = F.l1_loss(pred_quat, gt_quat * sign)
    trans_loss = F.l1_loss(out["wrist_trans_pred"], y["wrist_trans"])
    heatmap_loss = focal_heatmap_loss(out["keypoint_heatmaps"], y["heatmaps"])

    kp_pred = heatmap_argmax(out["keypoint_heatmaps"])
    valid = y["keypoints_valid"].bool()
    if valid.any():
        kp_err = torch.linalg.norm(kp_pred[valid] - y["keypoints_crop"][valid], dim=-1).mean()
    else:
        kp_err = torch.tensor(0.0, device=out["keypoint_heatmaps"].device)
    kp3d_loss = F.l1_loss(out["keypoints_3d_cam"], y["keypoints_3d_cam"])
    uv_pred = out["keypoints_uv"]
    if valid.any():
        crop_size = float(args.crop_size)
        uv_loss = F.smooth_l1_loss(uv_pred[valid] / crop_size, y["keypoints_crop"][valid] / crop_size)
        uv_err = torch.linalg.norm(uv_pred[valid] - y["keypoints_crop"][valid], dim=-1).mean()
    else:
        uv_loss = torch.tensor(0.0, device=uv_pred.device)
        uv_err = torch.tensor(0.0, device=uv_pred.device)

    loss = (
        args.alpha_action_l1 * action_loss
        + args.alpha_wrist_quat_l1 * quat_loss
        + args.alpha_wrist_trans_l1 * trans_loss
        + args.alpha_heatmap * heatmap_loss
        + args.alpha_keypoint_3d * kp3d_loss
        + args.alpha_keypoint_2d * uv_loss
    )
    return loss, {
        "loss": loss.detach(),
        "action_l1": action_loss.detach(),
        "quat_l1": quat_loss.detach(),
        "trans_l1": trans_loss.detach(),
        "heatmap": heatmap_loss.detach(),
        "kp_err": kp_err.detach(),
        "kp3d_l1": kp3d_loss.detach(),
        "uv_loss": uv_loss.detach(),
        "uv_err": uv_err.detach(),
        "kp_visible": valid.float().sum().detach(),
    }


def compute_wrist_only_losses(out, y, args):
    pred_quat = F.normalize(out["wrist_quat_pred"], p=2, dim=1)
    gt_quat = F.normalize(y["wrist_quat"], p=2, dim=1)
    sign = torch.where((pred_quat * gt_quat).sum(dim=1, keepdim=True) < 0.0, -1.0, 1.0)
    quat_loss = F.l1_loss(pred_quat, gt_quat * sign)
    trans_loss = F.l1_loss(out["wrist_trans_pred"], y["wrist_trans"])
    loss = args.alpha_wrist_quat_l1 * quat_loss + args.alpha_wrist_trans_l1 * trans_loss
    return loss, {
        "loss": loss.detach(),
        "quat_l1": quat_loss.detach(),
        "trans_l1": trans_loss.detach(),
    }


@torch.no_grad()
def evaluate(model, loader, device, args, max_batches=20, wrist_only=False):
    model.eval()
    meters = {}
    count = 0
    for (x, y), masks_enc, masks_pred in loader:
        x = x.to(device, non_blocking=True)
        y = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in y.items()}
        masks_enc = [m.to(device, non_blocking=True) for m in masks_enc]
        masks_pred = [m.to(device, non_blocking=True) for m in masks_pred]
        with torch.amp.autocast("cuda", enabled=bool(args.amp), dtype=torch.bfloat16):
            out = model(x, y["K"], masks_enc=masks_enc, masks_pred=masks_pred)
            if wrist_only:
                _, metrics = compute_wrist_only_losses(out, y, args)
            else:
                _, metrics = compute_losses(out, y, args)
        for key, value in metrics.items():
            meters[key] = meters.get(key, 0.0) + float(value.item())
        count += 1
        if max_batches is not None and max_batches > 0 and count >= max_batches:
            break
    if count == 0:
        raise RuntimeError("Validation loader produced no batches")
    return {f"val_{k}": reduce_scalar(v / count, device) for k, v in meters.items()}


def save_checkpoint(path, model, optimizer, scheduler, args, epoch, iteration):
    raw_model = model.module if isinstance(model, DDP) else model
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "args": vars(args),
            "epoch": epoch,
            "iter": iteration,
        },
        path,
    )


def _draw_keypoints(rgb, points, valid, color, radius=4, prefix=""):
    out = rgb.copy()
    for i, (xy, ok) in enumerate(zip(points, valid)):
        if not np.isfinite(xy).all():
            continue
        x, y = int(round(float(xy[0]))), int(round(float(xy[1])))
        if not (0 <= x < out.shape[1] and 0 <= y < out.shape[0]):
            continue
        c = color if bool(ok) else (230, 50, 50)
        cv2.circle(out, (x, y), radius, c, -1, lineType=cv2.LINE_AA)
        cv2.putText(out, f"{prefix}{i}", (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, c, 1, cv2.LINE_AA)
    return out


def _add_title(img, title):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(out, title, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _heatmap_argmax_np(heatmaps):
    c, h, w = heatmaps.shape
    flat = heatmaps.reshape(c, -1)
    inds = flat.argmax(axis=1)
    y = inds // w
    x = inds % w
    return np.stack([x, y], axis=1).astype(np.float32), flat.max(axis=1)


def _unnormalize_image_tensor(image_tensor):
    arr = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    arr = np.clip((arr * std + mean) * 255.0, 0, 255)
    return arr.astype(np.uint8)


@torch.no_grad()
def save_smoke_visuals(model, datasets_for_vis, device, args, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_model = model.module if isinstance(model, DDP) else model
    raw_model.eval()
    for ds_name, dataset in datasets_for_vis:
        base = dataset.dataset if isinstance(dataset, RepeatDataset) else dataset
        for local_idx in range(min(int(args.smoke_vis_samples), len(base))):
            image, target = base[local_idx]
            x = image.unsqueeze(0).to(device)
            K = target["K"].unsqueeze(0).to(device)
            with torch.amp.autocast("cuda", enabled=bool(args.amp), dtype=torch.bfloat16):
                out = raw_model(x, K, masks_enc=None, masks_pred=None)
            input_rgb = _unnormalize_image_tensor(image)
            gt_points = target["keypoints_crop"].numpy()
            gt_valid = target["keypoints_valid"].numpy().astype(bool)
            pred_hm = out["keypoint_heatmaps"][0].detach().float().cpu().numpy()
            pred_points, pred_scores = _heatmap_argmax_np(pred_hm)
            pred_valid = pred_scores > 0.05
            pred_uv = out["keypoints_uv"][0].detach().float().cpu().numpy()
            pred_uv_valid = np.isfinite(pred_uv).all(axis=1)

            gt_panel = _draw_keypoints(input_rgb, gt_points, gt_valid, (30, 220, 60), prefix="g")
            hm_panel = _draw_keypoints(input_rgb, pred_points, pred_valid, (255, 220, 30), prefix="h")
            uv_panel = _draw_keypoints(input_rgb, pred_uv, pred_uv_valid, (40, 180, 255), prefix="u")
            heat = np.clip(pred_hm.max(axis=0), 0.0, 1.0)
            heat = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
            heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
            panel = np.concatenate(
                [
                    _add_title(gt_panel, f"{ds_name} input GT keypoints"),
                    _add_title(hm_panel, "input pred heatmap argmax"),
                    _add_title(uv_panel, "input pred FK projection"),
                    _add_title(heat, "max pred heatmap"),
                ],
                axis=1,
            )
            frame_id = str(target.get("frame_id", local_idx))
            Image.fromarray(panel).save(out_dir / f"{ds_name}_{local_idx:03d}_{frame_id}.jpg", quality=92)


def main(args):
    device, local_rank, world_size = setup_dist()
    use_ddp = world_size > 1
    torch.backends.cudnn.benchmark = True

    log_dir = Path(args.save_dir) / args.name
    if args.dataset_cache_dir is None:
        args.dataset_cache_dir = str(log_dir / "dataset_cache")
    if is_main_process():
        log_dir.mkdir(parents=True, exist_ok=True)

    if use_ddp:
        if is_main_process():
            train_datasets = make_train_datasets(args)
            val_dataset = make_val_dataset(args)
            lnd_val_dataset = make_lnd_val_dataset(args) if bool(args.include_lnd_val) else None
            dist.barrier()
        else:
            dist.barrier()
            train_datasets = make_train_datasets(args)
            val_dataset = make_val_dataset(args)
            lnd_val_dataset = make_lnd_val_dataset(args) if bool(args.include_lnd_val) else None
        dist.barrier()
    else:
        train_datasets = make_train_datasets(args)
        val_dataset = make_val_dataset(args)
        lnd_val_dataset = make_lnd_val_dataset(args) if bool(args.include_lnd_val) else None
    train_dataset = ConcatDataset(train_datasets)

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if use_ddp else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if use_ddp else None
    lnd_val_sampler = DistributedSampler(lnd_val_dataset, shuffle=False) if (use_ddp and lnd_val_dataset is not None) else None
    collator = RoboPEPPMaskCollator(input_size=args.crop_size, patch_size=args.patch_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collator,
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
        collate_fn=collator,
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
            collate_fn=collator,
            persistent_workers=False,
        )

    model = make_robopepp_instrument_posenet(
        backbone=args.backbone,
        input_shape=(args.crop_size, args.crop_size),
        patch_size=args.patch_size,
        pred_emb_dim=args.pred_emb_dim,
        pred_depth=args.pred_depth,
        num_keypoints=5,
        pose_head_iter=args.pose_head_iter,
        pose_head_dropout=args.pose_head_dropout,
        jepa_path=args.jepa_path,
    ).to(device)
    if use_ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    raw_model = model.module if isinstance(model, DDP) else model
    optimizer = torch.optim.AdamW(
        [
            {"params": raw_model.context_backbone.parameters(), "lr": args.lr_backbone},
            {"params": raw_model.predictor_backbone.parameters(), "lr": args.lr_predictor},
            {"params": list(raw_model.action_net.parameters()) + list(raw_model.wrist_pose_net.parameters()), "lr": args.lr_pose},
            {"params": raw_model.keypoint_net.parameters(), "lr": args.lr_keypoint},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        pct_start=0.0,
        final_div_factor=args.final_div_factor,
        max_lr=[args.lr_backbone, args.lr_predictor, args.lr_pose, args.lr_keypoint],
        total_steps=args.max_iter,
        cycle_momentum=False,
    )
    iteration = 0
    epoch = 0
    best_val_loss = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model_state = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
        raw_model.load_state_dict(model_state, strict=True)
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        iteration = int(ckpt.get("iter", ckpt.get("iteration", 0)))
        epoch = int(ckpt.get("epoch", 0))
        best_val_loss = float(ckpt.get("best_val_loss", best_val_loss))
        if is_main_process():
            print(f"RESUME: {args.resume} | iter {iteration} | epoch {epoch}", flush=True)

    ckpt_dir = log_dir / "checkpoints"
    if is_main_process():
        print(f"LOG_DIR: {log_dir}")
        print(f"WORLD_SIZE: {world_size}")
        for ds in train_datasets:
            print(ds)
        print(val_dataset)
        if lnd_val_dataset is not None:
            print(lnd_val_dataset)

    if args.smoke_vis_dir and is_main_process():
        vis_datasets = [("rarp_train", train_datasets[0])]
        if bool(args.include_lnd_train):
            vis_datasets.append(("lnd_train", train_datasets[-1]))
        vis_datasets.append(("rarp_val", val_dataset))
        if lnd_val_dataset is not None:
            vis_datasets.append(("lnd_val", lnd_val_dataset))
        save_smoke_visuals(model, vis_datasets, device, args, args.smoke_vis_dir)

    if bool(args.full_val_before_train):
        val_metrics = evaluate(model, val_loader, device, args, max_batches=None, wrist_only=False)
        if is_main_process():
            print("PRETRAIN_VAL_RARP " + " | ".join(f"{k} {v:.4f}" for k, v in val_metrics.items()), flush=True)
        if lnd_val_loader is not None:
            lnd_metrics = evaluate(model, lnd_val_loader, device, args, max_batches=None, wrist_only=True)
            if is_main_process():
                print("PRETRAIN_VAL_LND_WRIST " + " | ".join(f"{k} {v:.4f}" for k, v in lnd_metrics.items()), flush=True)
        if bool(args.smoke_only):
            if use_ddp:
                dist.destroy_process_group()
            return

    last_log = time.time()
    while iteration < args.max_iter:
        set_epoch_recursive(train_dataset, epoch)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        for (x, y), masks_enc, masks_pred in train_loader:
            iteration += 1
            x = x.to(device, non_blocking=True)
            y = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in y.items()}
            masks_enc = [m.to(device, non_blocking=True) for m in masks_enc]
            masks_pred = [m.to(device, non_blocking=True) for m in masks_pred]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=bool(args.amp), dtype=torch.bfloat16):
                out = model(x, y["K"], masks_enc=masks_enc, masks_pred=masks_pred)
                loss, metrics = compute_losses(out, y, args)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            if iteration % args.log_freq == 0 and is_main_process():
                elapsed = max(time.time() - last_log, 1e-6)
                last_log = time.time()
                msg = [
                    f"iter {iteration:07d}",
                    f"epoch {epoch}",
                    f"loss {float(metrics['loss'].item()):.4f}",
                    f"action {float(metrics['action_l1'].item()):.4f}",
                    f"quat {float(metrics['quat_l1'].item()):.4f}",
                    f"trans {float(metrics['trans_l1'].item()):.4f}",
                    f"hm {float(metrics['heatmap'].item()):.4f}",
                    f"kp {float(metrics['kp_err'].item()):.2f}px",
                    f"uv {float(metrics['uv_err'].item()):.2f}px",
                    f"{args.log_freq * args.batch_size * max(world_size, 1) / elapsed:.1f} img/s",
                ]
                print(" | ".join(msg), flush=True)

            if iteration % args.val_freq == 0:
                max_batches = None if int(args.val_max_batches) <= 0 else int(args.val_max_batches)
                val_metrics = evaluate(model, val_loader, device, args, max_batches=max_batches, wrist_only=False)
                if is_main_process():
                    print("VAL_RARP " + " | ".join(f"{k} {v:.4f}" for k, v in val_metrics.items()), flush=True)
                if lnd_val_loader is not None:
                    lnd_metrics = evaluate(model, lnd_val_loader, device, args, max_batches=max_batches, wrist_only=True)
                    if is_main_process():
                        print("VAL_LND_WRIST " + " | ".join(f"{k} {v:.4f}" for k, v in lnd_metrics.items()), flush=True)
                if is_main_process():
                    val_loss = float(val_metrics.get("val_loss", float("inf")))
                    save_checkpoint(ckpt_dir / "last.pt", model, optimizer, scheduler, args, epoch, iteration)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_checkpoint(ckpt_dir / "best.pt", model, optimizer, scheduler, args, epoch, iteration)
                        print(f"NEW_BEST val_loss {best_val_loss:.4f} at iter {iteration}", flush=True)

            if iteration >= args.max_iter:
                break
        epoch += 1

    if is_main_process():
        save_checkpoint(ckpt_dir / "last.pt", model, optimizer, scheduler, args, epoch, iteration)
        print(f"Finished training at iter {iteration}")
    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="logs")
    parser.add_argument("--name", type=str, default="robopepp_instrument_pose_rarp")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--needle_puncture_data_dir", type=str, default=PUNCTURE_DATASET_ROOT)
    parser.add_argument("--needle_puncture_pose_dir", type=str, default=PUNCTURE_POSE_ROOT)
    parser.add_argument("--needle_grasping_data_dir", type=str, default=GRASPING_DATASET_ROOT)
    parser.add_argument("--needle_grasping_pose_dir", type=str, default=GRASPING_POSE_ROOT)
    parser.add_argument("--knotting_data_dir", type=str, default=KNOTTING_DATASET_ROOT)
    parser.add_argument("--knotting_pose_dir", type=str, default=KNOTTING_POSE_ROOT)
    parser.add_argument("--min_dice_shaft", type=float, default=0.8)
    parser.add_argument("--min_dice_wrist", type=float, default=0.6)
    parser.add_argument("--min_dice_gripper", type=float, default=0.6)
    parser.add_argument("--needle_train_ratio", type=float, default=0.95)
    parser.add_argument("--canonicalize_pose_symmetry", type=int, default=1, choices=[0, 1])
    parser.add_argument("--canonical_eps", type=float, default=0.08)
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--bbox_padding_frac", type=float, default=0.12)
    parser.add_argument("--bbox_jitter", type=int, default=1, choices=[0, 1])
    parser.add_argument("--bbox_shift", type=int, default=1, choices=[0, 1])
    parser.add_argument("--bbox_shift_max_px", type=float, default=8.0)
    parser.add_argument("--aug_random_crop_rotate", type=int, default=1, choices=[0, 1])
    parser.add_argument("--aug_geom_prob", type=float, default=0.3)
    parser.add_argument("--aug_max_angle", type=float, default=float(math.pi / 6.0))
    parser.add_argument("--color_jitter", type=int, default=1, choices=[0, 1])
    parser.add_argument("--rgb_augmentation", type=int, default=1, choices=[0, 1])
    parser.add_argument("--occlusion_augmentation", type=int, default=1, choices=[0, 1])
    parser.add_argument("--occlusion_prob", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--val_batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--train_subsample", type=int, default=1)
    parser.add_argument("--val_subsample", type=int, default=10)
    parser.add_argument("--dataset_cache_dir", type=str, default=None)
    parser.add_argument("--train_dataset_names", type=str, default="needlePuncture,needleGrasping,knotting")
    parser.add_argument("--include_lnd_train", type=int, default=0, choices=[0, 1])
    parser.add_argument("--include_lnd_val", type=int, default=0, choices=[0, 1])
    parser.add_argument("--lnd_root", type=str, default=DEFAULT_LND_ROOT)
    parser.add_argument("--lnd_refine_memory", type=str, default=DEFAULT_LND_REFINE_MEMORY)
    parser.add_argument("--lnd_sampling_rate", type=float, default=0.30)
    parser.add_argument("--lnd_train_subsample", type=int, default=1)
    parser.add_argument("--lnd_val_subsample", type=int, default=1)
    parser.add_argument("--lnd_bbox_jitter", type=int, default=0, choices=[0, 1])
    parser.add_argument("--backbone", type=str, default="vit_base", choices=["vit_tiny", "vit_small", "vit_base", "vit_large"])
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--pred_emb_dim", type=int, default=384)
    parser.add_argument("--pred_depth", type=int, default=12)
    parser.add_argument("--jepa_path", type=str, default=DEFAULT_JEPA_PATH)
    parser.add_argument("--pose_head_iter", type=int, default=4)
    parser.add_argument("--pose_head_dropout", type=float, default=0.3)
    parser.add_argument("--max_iter", type=int, default=60000)
    parser.add_argument("--log_freq", type=int, default=10)
    parser.add_argument("--val_freq", type=int, default=1000)
    parser.add_argument("--val_max_batches", type=int, default=20, help="<=0 runs full validation at periodic val_freq")
    parser.add_argument("--full_val_before_train", type=int, default=0, choices=[0, 1])
    parser.add_argument("--smoke_only", type=int, default=0, choices=[0, 1])
    parser.add_argument("--smoke_vis_dir", type=str, default="")
    parser.add_argument("--smoke_vis_samples", type=int, default=3)
    parser.add_argument("--amp", type=int, default=1, choices=[0, 1])
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=1e-7)
    parser.add_argument("--final_div_factor", type=float, default=1e4)
    parser.add_argument("--lr_backbone", type=float, default=1e-4)
    parser.add_argument("--lr_predictor", type=float, default=1e-4)
    parser.add_argument("--lr_pose", type=float, default=1e-4)
    parser.add_argument("--lr_keypoint", type=float, default=1e-4)
    parser.add_argument("--alpha_action_l1", type=float, default=1.0)
    parser.add_argument("--alpha_wrist_quat_l1", type=float, default=1.0)
    parser.add_argument("--alpha_wrist_trans_l1", type=float, default=10.0)
    parser.add_argument("--alpha_heatmap", type=float, default=1.0)
    parser.add_argument("--alpha_keypoint_3d", type=float, default=0.0)
    parser.add_argument("--alpha_keypoint_2d", type=float, default=0.0)
    parser.add_argument("--heatmap_sigma", type=float, default=2.0)
    main(parser.parse_args())
