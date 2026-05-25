import os
import sys
import time

os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["EGL_DEVICE_ID"] = "0"

from argparse import ArgumentParser

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from PIL import Image
from tqdm import tqdm

from datasets.rarp_hcce_pose_dataset import (
    RARPHCCEPoseDataset,
    collate_fn_instrument_hcce_pose,
)
from datasets.surgical_instruments import RARP50_DIR, SurgicalInstruments
from loss_instrument_hcce_pose import LossInstrumentHCCEPose
from multi_instrument.multi_instrument_hcce_pose_dpt import MultiInstrumentHCCEPoseDPT
from train_instrument_cse import TrainerCSE, is_main_process, setup_ddp
from utils import AverageMeter


def _hcce_first_level_vis(coord_img, hcce_logits, inst_mask, bits=8, coord_min=-1.0, coord_max=1.0):
    gt_xyz = coord_img[:, :, :3].detach().float().cpu()
    coord_part = coord_img[:, :, 3].detach().cpu().long()
    valid = (coord_part > 0) & (inst_mask.detach().cpu() > 0.5)

    gt_rgb = ((gt_xyz - coord_min) / (coord_max - coord_min)).clamp(0, 1).numpy()
    pred_rgb = (
        ((hcce_logits[[0, bits, bits * 2]].detach().float().clamp(-1.0, 1.0) + 1.0) * 0.5)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )

    gt_panel = np.zeros_like(gt_rgb)
    pred_panel = np.zeros_like(pred_rgb)
    err_panel = np.zeros_like(pred_rgb)
    valid_np = valid.numpy()
    gt_panel[valid_np] = gt_rgb[valid_np]
    pred_panel[valid_np] = pred_rgb[valid_np]
    err = np.abs(pred_rgb - gt_rgb).mean(axis=2)
    if valid_np.any():
        scale = np.percentile(err[valid_np], 95) + 1e-8
        val = np.clip(err / scale, 0, 1)
        err_panel[..., 0] = val
        err_panel[..., 1] = 1.0 - val
        err_panel[~valid_np] = 0

    out = np.concatenate([gt_panel, pred_panel, err_panel], axis=1)
    return (out * 255.0).clip(0, 255).astype(np.uint8)


class TrainerHCCEPose(TrainerCSE):
    """
    New trainer target preparation for HCCE + action/wrist pose supervision.
    """

    def save_checkpoint(self, name="last"):
        if not is_main_process():
            return
        save_dict = {
            "epoch": self.current_epoch,
            "iter": self.current_iter,
            "model_state_dict": self.raw_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "args": self.args,
            "best_val": self.best_val,
        }
        torch.save(save_dict, os.path.join(self.args.ckpt_dir, f"{name}.pt"))

    def fit(self, data_train, l_data_val):
        for epoch in range(self.current_epoch, self.args.max_epochs):
            max_iter = getattr(self.args, "max_iter", None)
            if max_iter is not None and self.current_iter >= max_iter:
                break
            self.current_epoch = epoch
            self.train_n_iters(data_train, l_data_val)
            self.save_checkpoint("last")
            if max_iter is not None and self.current_iter >= max_iter:
                break
        return 1

    def train_n_iters(self, data, l_data_val):
        if is_main_process():
            print(f"\nTRAIN EPOCH {self.current_epoch}: ")
        self.model.train()

        if hasattr(data, "sampler") and isinstance(data.sampler, DistributedSampler):
            data.sampler.set_epoch(self.current_epoch)

        grad_accum_steps = max(1, int(getattr(self.args, "grad_accum_steps", 1)))
        meters = {k: AverageMeter(k) for k in ["workload/data", "workload/batch", "workload/ratio_data"]}

        self.optimizer.zero_grad(set_to_none=True)
        pending_backward = 0
        timer_end = time.time()
        iterator = tqdm(data) if is_main_process() else data
        for i, (x, y) in enumerate(iterator):
            y = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in y.items()}
            data_time = time.time() - timer_end

            gt = self.prepare_gt(y=y)
            if gt is None:
                timer_end = time.time()
                continue

            x = x.to(self.device)

            with torch.cuda.amp.autocast(enabled=bool(self.args.amp)):
                pred = self.model(x, is_training=True, idx=gt["idx"])
                loss, dict_loss = self.loss(pred, gt, epoch=self.current_epoch, img_size=self.args.img_size)
                loss_for_backward = loss / grad_accum_steps

            self.scaler.scale(loss_for_backward).backward()
            pending_backward += 1

            should_step = pending_backward >= grad_accum_steps
            if should_step:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                pending_backward = 0

            batch_time = time.time() - timer_end
            meters["workload/data"].update(data_time)
            meters["workload/batch"].update(batch_time)
            meters["workload/ratio_data"].update(data_time / batch_time)
            if "pose_sym_flipped_inst" in gt and gt["pose_sym_flipped_inst"].numel() > 0:
                if "pose_sym/flipped_ratio" not in meters:
                    meters["pose_sym/flipped_ratio"] = AverageMeter("pose_sym/flipped_ratio")
                meters["pose_sym/flipped_ratio"].update(gt["pose_sym_flipped_inst"].float().mean().item())
            for k, v in dict_loss.items():
                k_name = f"loss/{k}"
                if k_name not in meters:
                    meters[k_name] = AverageMeter(k_name)
                meters[k_name].update(v.item())

            if is_main_process() and i % self.args.log_freq == 0:
                nocs_str = (
                    f" - nocs={meters['loss/nocs_reg'].avg:.4f}"
                    if "loss/nocs_reg" in meters
                    else ""
                )
                hcce_str = (
                    f" - hcce={meters['loss/hcce'].avg:.4f} - bit_acc={meters['loss/hcce_bit_acc'].avg:.4f}"
                    if "loss/hcce" in meters
                    else ""
                )
                print(
                    f"EPOCH={self.current_epoch:03d} - i={i:05d}/{len(data):05d} - "
                    f"iter={self.current_iter} - "
                    f"loss={meters['loss/total'].avg:.4f} - "
                    f"det={meters['loss/det'].avg:.4f} - "
                    f"dice={meters['loss/dice'].avg:.4f} - "
                    f"part={meters['loss/part_ce'].avg:.4f}"
                    f"{nocs_str}{hcce_str} - accum={grad_accum_steps}"
                )

                if self.writer is not None:
                    for k, v in meters.items():
                        self.writer.add_scalar(f"{k}", v.avg, self.current_iter)
                    self.writer.flush()
                sys.stdout.flush()

            if is_main_process() and self.writer is not None and self._should_log_train_vis():
                self._log_train_vis(x, gt, pred)
                self.writer.flush()

            if should_step and self.current_iter > 0 and self.current_iter % self.args.val_freq == 0:
                val_metric = 0.0
                for data_val in l_data_val:
                    val_metric = max(val_metric, self.evaluate(data_val))
                if val_metric > self.best_val:
                    self.best_val = val_metric
                    self.save_checkpoint("best")
                    if is_main_process():
                        print(f"*** New best: IoU={val_metric:.4f} ***")
                self.model.train()

            self.current_iter += 1
            max_iter = getattr(self.args, "max_iter", None)
            if max_iter is not None and self.current_iter >= max_iter:
                if pending_backward > 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                break
            timer_end = time.time()

        return 1

    def prepare_gt(self, y):
        valid = y["valid_instruments"].bool()
        idx_h = torch.where(valid)
        n_valid = int(valid.sum().item())
        if n_valid == 0:
            return None

        wrist_centers = y["wrist_centers"][idx_h[0], idx_h[1]]
        n_patch = self.args.img_size // self.raw_model.patch_size
        pk_coarse = (wrist_centers / self.raw_model.patch_size).int()
        pk_idx = torch.clamp(pk_coarse, 0, n_patch - 1)
        pk_offset = (
            wrist_centers - (pk_idx + 0.5) * self.raw_model.patch_size
        ) / self.raw_model.patch_size

        scores = torch.zeros((valid.shape[0], n_patch, n_patch), device=self.device)
        keep = torch.ones(n_valid, dtype=torch.bool, device=self.device)

        batch_all = idx_h[0].to(self.device)
        pk_idx_dev = pk_idx.to(self.device)

        for k in range(n_valid):
            bi = int(batch_all[k].item())
            px = pk_idx_dev[k, 1]
            py = pk_idx_dev[k, 0]
            if scores[bi, px, py] == 1:
                keep[k] = False
            else:
                scores[bi, px, py] = 1

        idx_vis = torch.where(keep)[0]
        if idx_vis.numel() == 0:
            return None

        target = {
            "scores": scores,
            "offset": pk_offset.to(self.device)[idx_vis],
            "loc": wrist_centers.to(self.device)[idx_vis],
            "inst_masks": y["inst_masks"][idx_h[0], idx_h[1]][idx_vis].to(self.device),
            "part_masks": y["part_masks"][idx_h[0], idx_h[1]][idx_vis].to(self.device),
            "idx": (
                batch_all[idx_vis],
                pk_idx_dev[idx_vis, 1],
                pk_idx_dev[idx_vis, 0],
                torch.zeros_like(batch_all[idx_vis]),
            ),
        }

        if "dataset_source" in y:
            target["det_source_mask"] = (y["dataset_source"] == 0).to(self.device)
            target["rarp50_inst_mask"] = target["det_source_mask"][target["idx"][0]]

        if "has_cse" in y and "coord_imgs" in y:
            target["has_cse_inst"] = y["has_cse"][idx_h[0], idx_h[1]][idx_vis].to(self.device)
            target["coord_imgs"] = y["coord_imgs"][idx_h[0], idx_h[1]][idx_vis].to(self.device)

        if "has_pose" in y:
            target["has_pose_inst"] = y["has_pose"][idx_h[0], idx_h[1]][idx_vis].to(self.device)
            target["action_gt_inst"] = y["action_gt"][idx_h[0], idx_h[1]][idx_vis].to(self.device)
            target["wrist_quat_gt_inst"] = y["wrist_quat_gt"][idx_h[0], idx_h[1]][idx_vis].to(self.device)
            target["wrist_trans_gt_inst"] = y["wrist_trans_gt"][idx_h[0], idx_h[1]][idx_vis].to(self.device)
        if "pose_sym_flipped" in y:
            target["pose_sym_flipped_inst"] = y["pose_sym_flipped"][idx_h[0], idx_h[1]][idx_vis].to(self.device)

        return target

    def _log_train_vis(self, x, gt, pred):
        super()._log_train_vis(x, gt, pred)
        if "hcce_logits" not in pred or "coord_imgs" not in gt or "has_cse_inst" not in gt:
            return
        with torch.no_grad():
            logged = 0
            max_logged = 4
            for j in range(pred["hcce_logits"].shape[0]):
                if logged >= max_logged:
                    break
                if not bool(gt["has_cse_inst"][j].item()):
                    continue
                vis = _hcce_first_level_vis(
                    gt["coord_imgs"][j],
                    pred["hcce_logits"][j],
                    gt["inst_masks"][j],
                    bits=self.args.hcce_bits,
                    coord_min=self.args.hcce_coord_min,
                    coord_max=self.args.hcce_coord_max,
                )
                if self.writer is not None:
                    self.writer.add_image(
                        f"train_hcce_first_level/inst_{j}",
                        vis,
                        self.current_iter,
                        dataformats="HWC",
                    )
                fn = os.path.join(
                    self.args.visu_dir,
                    f"train_hcce_first_level_iter{self.current_iter:07d}_{j:02d}.jpg",
                )
                Image.fromarray(vis).save(fn)
                logged += 1


def make_rarp_dataset(args, split, training, dataset_root, pose_root, subsample=1):
    return RARPHCCEPoseDataset(
        split=split,
        training=training,
        img_size=args.img_size,
        dataset_root=dataset_root,
        pose_root=pose_root,
        min_dice=[
            args.min_dice_shaft,
            args.min_dice_wrist,
            args.min_dice_gripper,
        ],
        train_ratio=args.needle_train_ratio,
        subsample=subsample,
        cse_coord_root=args.cse_coord_root,
        render_on_the_fly=bool(args.render_on_the_fly),
        canonicalize_pose_symmetry=bool(args.canonicalize_pose_symmetry),
        canonical_eps=args.canonical_eps,
    )


def main(args):
    if args.log_freq <= 0:
        raise ValueError("--log_freq must be positive")
    if args.train_vis_freq <= 0:
        raise ValueError("--train_vis_freq must be positive")
    if args.train_vis_freq % args.log_freq != 0:
        raise ValueError("--train_vis_freq must be an integer multiple of --log_freq")

    device, rank, world_size = setup_ddp()
    use_ddp = world_size > 1

    model = MultiInstrumentHCCEPoseDPT(pretrained_backbone=True, **vars(args)).to(device)
    resume_ckpt = None
    resume_epoch = 0
    resume_iter = 0
    resume_best_val = 0.0
    load_path = args.resume if args.resume is not None else args.pretrained
    if load_path is not None and os.path.isfile(load_path):
        if is_main_process():
            mode = "resume" if args.resume is not None else "pretrained"
            print(f"Loading {mode} checkpoint from {load_path}")
        ckpt = torch.load(load_path, map_location=device, weights_only=False)
        log = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if args.resume is not None:
            resume_ckpt = ckpt
            resume_epoch = int(ckpt.get("epoch", 0))
            resume_iter = int(ckpt.get("iter", 0))
            resume_best_val = float(ckpt.get("best_val", 0.0))
        if is_main_process():
            print(log)
            if args.resume is not None:
                print(
                    f"Resumed epoch={resume_epoch}, iter={resume_iter}, "
                    f"best_val={resume_best_val:.4f}"
                )

    if use_ddp:
        model = DDP(model, device_ids=[device], find_unused_parameters=True)

    l_val_data = []
    if args.use_rarp50:
        l_val_data.append(
            DataLoader(
                SurgicalInstruments(
                    split="test",
                    training=False,
                    img_size=args.img_size,
                    root_dir=args.data_dir,
                    subsample=args.val_subsample,
                ),
                batch_size=args.val_batch_size,
                num_workers=args.num_workers,
                shuffle=False,
                drop_last=False,
                collate_fn=collate_fn_instrument_hcce_pose,
            )
        )
    l_val_data.append(
        DataLoader(
            make_rarp_dataset(
                args,
                split="test",
                training=False,
                dataset_root=args.needle_puncture_data_dir,
                pose_root=args.needle_puncture_pose_dir,
                subsample=args.val_subsample,
            ),
            batch_size=args.val_batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn_instrument_hcce_pose,
        )
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss = LossInstrumentHCCEPose(args)
    trainer = TrainerHCCEPose(
        model=model,
        loss=loss,
        optimizer=optimizer,
        device=device,
        args=args,
        best_val=resume_best_val,
    )
    trainer.current_epoch = resume_epoch
    trainer.current_iter = resume_iter
    if resume_ckpt is not None and "optimizer_state_dict" in resume_ckpt:
        optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
    if resume_ckpt is not None and "scaler_state_dict" in resume_ckpt:
        trainer.scaler.load_state_dict(resume_ckpt["scaler_state_dict"])

    if is_main_process():
        print()
        print(f"ARGS: {trainer.args}")
        print(f"LOG_DIR: {trainer.args.log_dir}")
        print("MODEL: MultiInstrumentHCCEPoseDPT")
        print("LOSS:  LossInstrumentHCCEPose")
        print()

    if args.eval_only:
        for vd in l_val_data:
            trainer.evaluate(vd)
    else:
        train_datasets = []
        if args.use_rarp50:
            train_datasets.append(
                SurgicalInstruments(
                    split="train",
                    training=True,
                    img_size=args.img_size,
                    root_dir=args.data_dir,
                    subsample=args.train_subsample,
                )
            )
        train_datasets.extend([
            make_rarp_dataset(
                args,
                split="train",
                training=True,
                dataset_root=args.needle_puncture_data_dir,
                pose_root=args.needle_puncture_pose_dir,
                subsample=args.train_subsample,
            ),
            make_rarp_dataset(
                args,
                split="train",
                training=True,
                dataset_root=args.needle_grasping_data_dir,
                pose_root=args.needle_grasping_pose_dir,
                subsample=args.train_subsample,
            ),
            make_rarp_dataset(
                args,
                split="train",
                training=True,
                dataset_root=args.knotting_data_dir,
                pose_root=args.knotting_pose_dir,
                subsample=args.train_subsample,
            ),
        ])
        if is_main_process():
            for ds in train_datasets:
                print(ds)
            print(f"CSE coord source: {'render_on_the_fly' if args.render_on_the_fly else args.cse_coord_root}")

        train_dataset = ConcatDataset(train_datasets)
        train_sampler = DistributedSampler(train_dataset, shuffle=True) if use_ddp else None
        train_data = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=(train_sampler is None),
            drop_last=True,
            collate_fn=collate_fn_instrument_hcce_pose,
            sampler=train_sampler,
        )
        trainer.fit(train_data, l_val_data)

    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    PUNCTURE_DATASET_ROOT = "/mnt/nas/share/shuojue/data/needlePuncture_videos"
    PUNCTURE_POSE_ROOT = "/mnt/nas/share/shuojue/data/needlePuncture_results"
    GRASPING_DATASET_ROOT = "/mnt/nas/share/shuojue/data/needleGrasping_videos"
    GRASPING_POSE_ROOT = "/mnt/nas/share/shuojue/data/needleGrasping_results"
    KNOTTING_DATASET_ROOT = "/mnt/nas/share/shuojue/data/knotting_videos"
    KNOTTING_POSE_ROOT = "/mnt/nas/share/shuojue/data/knotting_results"

    parser = ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="logs")
    parser.add_argument("--name", type=str, default="instrument_rarp_hcce_pose_dpt")
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=RARP50_DIR)
    parser.add_argument("--use_rarp50", type=int, default=1, choices=[0, 1])

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

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--num_workers", "-j", type=int, default=0)
    parser.add_argument("--img_size", type=int, default=630)
    parser.add_argument(
        "--backbone",
        type=str,
        default="dinov2_vits14",
        choices=["dinov2_vitl14", "dinov2_vitb14", "dinov2_vits14"],
    )
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--log_freq", type=int, default=100)
    parser.add_argument("--train_vis_freq", type=int, default=1000)
    parser.add_argument("--max_iter", type=int, default=60000)
    parser.add_argument("--nb_max_ckpt", type=int, default=10)
    parser.add_argument("--amp", type=int, default=1, choices=[0, 1])
    parser.add_argument("--learning_rate", "-lr", type=float, default=5e-5)
    parser.add_argument("--eval_only", type=int, default=0, choices=[0, 1])
    parser.add_argument("--det_thresh", type=float, default=0.3)
    parser.add_argument("--nms_kernel_size", type=int, default=3)
    parser.add_argument("--val_freq", type=int, default=1000)
    parser.add_argument("--val_batch_size", type=int, default=8)
    parser.add_argument("--train_subsample", type=int, default=1)
    parser.add_argument("--val_subsample", type=int, default=10)

    parser.add_argument("--xat_dim", type=int, default=512)
    parser.add_argument("--xat_depth", type=int, default=4)
    parser.add_argument("--xat_heads", type=int, default=16)
    parser.add_argument("--xat_dim_head", type=int, default=32)
    parser.add_argument("--xat_mlp_dim", type=int, default=2048)
    parser.add_argument("--mask_dim", type=int, default=256)
    parser.add_argument("--num_parts", type=int, default=4)
    parser.add_argument("--hcce_feat_dim", type=int, default=256)
    parser.add_argument("--action_dim", type=int, default=3)
    parser.add_argument("--use_pose_heads", type=int, default=0, choices=[0, 1])
    parser.add_argument("--pose_head_iter", type=int, default=4)
    parser.add_argument("--pose_head_dropout", type=float, default=0.3)

    parser.add_argument("--cse_coord_root", type=str, default=None)
    parser.add_argument("--render_on_the_fly", type=int, default=1, choices=[0, 1])
    parser.add_argument("--canonicalize_pose_symmetry", type=int, default=1, choices=[0, 1])
    parser.add_argument("--canonical_eps", type=float, default=0.08)

    parser = LossInstrumentHCCEPose.add_specific_args(parser)
    main(parser.parse_args())
