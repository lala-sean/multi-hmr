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

from datasets.rarp_hcce_pose_dataset import collate_fn_instrument_hcce_pose
from datasets.surgical_instruments import RARP50_DIR, SurgicalInstruments
from loss_instrument_hcce_densepart_pose import LossInstrumentHCCEDensePartPose
from multi_instrument.multi_instrument_hcce_densepart_dpt import MultiInstrumentHCCEDensePartDPT
from train_instrument_cse import (
    AverageMeter,
    TrainerCSE,
    is_main_process,
    setup_ddp,
    visualize_segmentation,
)
from train_instrument_hcce_pose_dpt import _hcce_first_level_vis, make_rarp_dataset


def _dense_part_to_dataset_labels(part_idx):
    out = torch.zeros_like(part_idx)
    out[part_idx == 0] = 2  # wrist
    out[part_idx == 1] = 1  # gripper
    out[part_idx == 2] = 3  # shaft
    return out


def _masked_dense_part_labels(part_logits, inst_mask_logits, inst_thresh=0.5):
    labels = _dense_part_to_dataset_labels(part_logits.argmax(dim=0))
    inst_prob = torch.sigmoid(inst_mask_logits.float())
    if inst_prob.shape[-2:] != labels.shape[-2:]:
        inst_prob = F.interpolate(
            inst_prob[None, None],
            size=labels.shape[-2:],
            mode="nearest",
        )[0, 0]
    return torch.where(inst_prob > inst_thresh, labels, torch.zeros_like(labels))


def _resize_mask_like(mask, size):
    if mask.shape[-2:] == size:
        return mask
    return F.interpolate(mask.unsqueeze(1).float(), size=size, mode="nearest")[:, 0]


class TrainerHCCEDensePartPose(TrainerCSE):
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
            for k, v in dict_loss.items():
                k_name = f"loss/{k}"
                if k_name not in meters:
                    meters[k_name] = AverageMeter(k_name)
                meters[k_name].update(v.item())

            if is_main_process() and i % self.args.log_freq == 0:
                hcce_str = (
                    f" - hcce={meters['loss/hcce'].avg:.4f} - bit_acc={meters['loss/hcce_bit_acc'].avg:.4f}"
                    if "loss/hcce" in meters
                    else ""
                )
                part_acc_str = (
                    f" - part_acc={meters['loss/part_acc'].avg:.4f}"
                    if "loss/part_acc" in meters
                    else ""
                )
                print(
                    f"EPOCH={self.current_epoch:03d} - i={i:05d}/{len(data):05d} - "
                    f"iter={self.current_iter} - "
                    f"loss={meters['loss/total'].avg:.4f} - "
                    f"det={meters['loss/det'].avg:.4f} - "
                    f"dice={meters['loss/dice'].avg:.4f} - "
                    f"part={meters['loss/part_ce'].avg:.4f}{part_acc_str}"
                    f"{hcce_str} - accum={grad_accum_steps}"
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
        inst_all = idx_h[1].to(self.device)
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

        b_sel = batch_all[idx_vis]
        i_sel = inst_all[idx_vis]
        target = {
            "scores": scores,
            "offset": pk_offset.to(self.device)[idx_vis],
            "loc": wrist_centers.to(self.device)[idx_vis],
            "inst_masks": y["inst_masks"][idx_h[0], idx_h[1]][idx_vis].to(self.device),
            "part_masks": y["part_masks"][idx_h[0], idx_h[1]][idx_vis].to(self.device),
            "idx": (
                b_sel,
                pk_idx_dev[idx_vis, 1],
                pk_idx_dev[idx_vis, 0],
                torch.zeros_like(b_sel),
            ),
        }

        if "part_masks_dense" in y:
            target["part_masks_dense"] = y["part_masks_dense"][b_sel, i_sel].to(self.device)
        if "inst_masks_dense" in y:
            target["inst_masks_dense"] = y["inst_masks_dense"][b_sel, i_sel].to(self.device)

        if "dataset_source" in y:
            target["det_source_mask"] = (y["dataset_source"] == 0).to(self.device)
            target["rarp50_inst_mask"] = target["det_source_mask"][target["idx"][0]]

        if "has_cse" in y and "coord_imgs" in y:
            target["has_cse_inst"] = y["has_cse"][idx_h[0], idx_h[1]][idx_vis].to(self.device)
            target["coord_imgs"] = y["coord_imgs"][idx_h[0], idx_h[1]][idx_vis].to(self.device)
        if "has_cse_dense" in y and "coord_imgs_dense" in y:
            target["has_cse_dense_inst"] = y["has_cse_dense"][b_sel, i_sel].to(self.device)
            target["coord_imgs_dense"] = y["coord_imgs_dense"][b_sel, i_sel].to(self.device)

        if "has_pose" in y:
            target["has_pose_inst"] = y["has_pose"][idx_h[0], idx_h[1]][idx_vis].to(self.device)
            target["action_gt_inst"] = y["action_gt"][idx_h[0], idx_h[1]][idx_vis].to(self.device)
            target["wrist_quat_gt_inst"] = y["wrist_quat_gt"][idx_h[0], idx_h[1]][idx_vis].to(self.device)
            target["wrist_trans_gt_inst"] = y["wrist_trans_gt"][idx_h[0], idx_h[1]][idx_vis].to(self.device)
        if "pose_sym_flipped" in y:
            target["pose_sym_flipped_inst"] = y["pose_sym_flipped"][idx_h[0], idx_h[1]][idx_vis].to(self.device)

        return target

    def _log_train_vis(self, x, gt, pred):
        with torch.no_grad():
            n_vis = min(x.shape[0], 10)
            for vi in range(n_vis):
                img_inds = torch.where(gt["idx"][0] == vi)[0]
                if len(img_inds) == 0:
                    continue
                gt_masks_np = [gt["inst_masks"][j].cpu().numpy() for j in img_inds]
                gt_parts_tensor = gt.get("part_masks_dense", gt["part_masks"])
                gt_parts_np = [gt_parts_tensor[j].cpu().numpy() for j in img_inds]
                pred_masks_np = [
                    torch.sigmoid(pred["inst_mask_logits"][j]).float().cpu().numpy()
                    for j in img_inds
                ]
                pred_parts_np = [
                    _masked_dense_part_labels(
                        pred["part_mask_logits"][j],
                        pred["inst_mask_logits"][j],
                    ).cpu().numpy()
                    for j in img_inds
                ]
                vis = visualize_segmentation(
                    x[vi],
                    gt_masks_np,
                    gt_parts_np,
                    pred_masks_np,
                    pred_parts_np,
                    gt_locs=gt["loc"][img_inds],
                    pred_locs=pred["loc"][img_inds],
                )
                self.writer.add_image(f"train/sample_{vi}", vis, self.current_iter, dataformats="HWC")
                Image.fromarray(vis).save(
                    os.path.join(self.args.visu_dir, f"train_iter{self.current_iter:07d}_{vi:02d}.jpg")
                )

            if "hcce_logits" not in pred or "coord_imgs_dense" not in gt:
                return
            logged = 0
            for j in range(pred["hcce_logits"].shape[0]):
                if logged >= 4:
                    break
                has_cse = gt.get("has_cse_dense_inst", gt.get("has_cse_inst"))[j]
                if not bool(has_cse.item()):
                    continue
                inst_mask = gt.get("inst_masks_dense", gt["inst_masks"])[j]
                vis = _hcce_first_level_vis(
                    gt["coord_imgs_dense"][j],
                    pred["hcce_logits"][j],
                    inst_mask,
                    bits=self.args.hcce_bits,
                    coord_min=self.args.hcce_coord_min,
                    coord_max=self.args.hcce_coord_max,
                )
                self.writer.add_image(
                    f"train_hcce_first_level/inst_{j}",
                    vis,
                    self.current_iter,
                    dataformats="HWC",
                )
                Image.fromarray(vis).save(
                    os.path.join(
                        self.args.visu_dir,
                        f"train_hcce_first_level_iter{self.current_iter:07d}_{j:02d}.jpg",
                    )
                )
                logged += 1

    @torch.no_grad()
    def evaluate(self, data):
        if not is_main_process():
            return 0.0
        print(f"\nEVAL (iter={self.current_iter}): ")
        self.model.eval()

        meters = {k: AverageMeter(k) for k in ["inst_iou", "part_acc", "precision", "recall", "f1_score"]}
        count, miss, fp = 0, 0, 0
        vis_count = 0

        for i, (x, y) in enumerate(tqdm(data)):
            y = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in y.items()}
            gt = self.prepare_gt(y=y)
            if gt is None:
                continue
            x = x.to(self.device)
            with torch.cuda.amp.autocast(enabled=bool(self.args.amp)):
                pred = self.raw_model(
                    x,
                    is_training=False,
                    det_thresh=self.args.det_thresh,
                    nms_kernel_size=self.args.nms_kernel_size,
                )

            bs = x.shape[0]
            all_pred = pred if isinstance(pred, list) else []
            for bi in range(bs):
                gt_inds = torch.where(gt["idx"][0] == bi)[0]
                n_gt_i = len(gt_inds)
                count += n_gt_i
                pred_i = [p for p in all_pred if p["batch_idx"] == bi]
                n_pred_i = len(pred_i)

                if n_pred_i == 0:
                    miss += n_gt_i
                    continue

                pred_locs = torch.stack([p["loc"] for p in pred_i])
                gt_locs = gt["loc"][gt_inds]
                dist_matrix = torch.cdist(pred_locs.unsqueeze(0).float(), gt_locs.unsqueeze(0).float())[0]
                matched_pred = set()
                matched_gt = set()
                dist_thresh = self.raw_model.patch_size * 2

                for _ in range(min(n_pred_i, n_gt_i)):
                    min_val = dist_matrix.min()
                    if min_val > dist_thresh:
                        break
                    pi, gi = torch.where(dist_matrix == min_val)
                    pi, gi = pi[0].item(), gi[0].item()
                    if pi in matched_pred or gi in matched_gt:
                        dist_matrix[pi, gi] = float("inf")
                        continue
                    matched_pred.add(pi)
                    matched_gt.add(gi)
                    dist_matrix[pi, :] = float("inf")
                    dist_matrix[:, gi] = float("inf")

                    pred_mask = torch.sigmoid(pred_i[pi]["inst_mask_logits"]) > 0.5
                    gt_mask = gt["inst_masks"][gt_inds[gi]] > 0.5
                    if pred_mask.shape == gt_mask.shape:
                        intersection = (pred_mask & gt_mask).sum().float()
                        union = (pred_mask | gt_mask).sum().float()
                        meters["inst_iou"].update((intersection / (union + 1e-6)).item())

                    pred_parts = _masked_dense_part_labels(
                        pred_i[pi]["part_mask_logits"],
                        pred_i[pi]["inst_mask_logits"],
                    )
                    gt_parts = gt.get("part_masks_dense", gt["part_masks"])[gt_inds[gi]]
                    if pred_parts.shape != gt_parts.shape:
                        gt_parts = F.interpolate(
                            gt_parts[None, None].float(),
                            size=pred_parts.shape[-2:],
                            mode="nearest",
                        )[0, 0].long()
                    valid_part = (gt_parts > 0) & (pred_parts > 0)
                    if valid_part.any():
                        correct = (pred_parts[valid_part] == gt_parts[valid_part]).float().mean()
                        meters["part_acc"].update(correct.item())

                miss += n_gt_i - len(matched_gt)
                fp += n_pred_i - len(matched_pred)

                if vis_count < 10:
                    gt_masks_np = [gt["inst_masks"][j].cpu().numpy() for j in gt_inds]
                    gt_parts_tensor = gt.get("part_masks_dense", gt["part_masks"])
                    gt_parts_np = [gt_parts_tensor[j].cpu().numpy() for j in gt_inds]
                    pred_masks_np = [
                        (torch.sigmoid(p["inst_mask_logits"]) > 0.5).cpu().numpy()
                        for p in pred_i
                    ]
                    pred_parts_np = [
                        _masked_dense_part_labels(
                            p["part_mask_logits"],
                            p["inst_mask_logits"],
                        ).cpu().numpy()
                        for p in pred_i
                    ]
                    pred_locs_vis = torch.stack([p["loc"] for p in pred_i]) if pred_i else None
                    vis = visualize_segmentation(
                        x[bi],
                        gt_masks_np,
                        gt_parts_np,
                        pred_masks_np,
                        pred_parts_np,
                        gt_locs=gt["loc"][gt_inds],
                        pred_locs=pred_locs_vis,
                    )
                    self._save_eval_vis(vis, data, vis_count)
                    vis_count += 1

            if i % self.args.log_freq == 0:
                tp = count - miss
                precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
                recall = tp / count * 100 if count > 0 else 0
                print(
                    f"i={i} - Recall={recall:.1f}% - "
                    f"IoU={meters['inst_iou'].avg:.3f} - PartAcc={meters['part_acc'].avg:.3f}"
                )
                sys.stdout.flush()

        print(f"***EVAL - {data.dataset.name}-{data.dataset.split}***")
        if count > 0:
            tp = count - miss
            precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
            recall = tp / count * 100
            f1 = 2 * precision * recall / (precision + recall + 1e-6)
        else:
            precision = recall = f1 = 0
        meters["precision"].update(precision)
        meters["recall"].update(recall)
        meters["f1_score"].update(f1)
        for k, v in meters.items():
            if self.writer is not None:
                self.writer.add_scalar(f"{data.dataset.name}-{data.dataset.split}/{k}", v.avg, self.current_iter)
            print(f"    - {k}: {v.avg:.3f}")
        if self.writer is not None:
            self.writer.flush()
        sys.stdout.flush()
        return meters["inst_iou"].avg


def main(args):
    if args.log_freq <= 0:
        raise ValueError("--log_freq must be positive")
    if args.train_vis_freq <= 0:
        raise ValueError("--train_vis_freq must be positive")
    if args.train_vis_freq % args.log_freq != 0:
        raise ValueError("--train_vis_freq must be an integer multiple of --log_freq")
    if args.dense_output_stride != 2:
        raise ValueError("This model expects --dense_output_stride 2 for H/2 HCCE/part targets.")

    device, rank, world_size = setup_ddp()
    use_ddp = world_size > 1

    model = MultiInstrumentHCCEDensePartDPT(pretrained_backbone=True, **vars(args)).to(device)
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
                print(f"Resumed epoch={resume_epoch}, iter={resume_iter}, best_val={resume_best_val:.4f}")

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
                    dense_output_stride=args.dense_output_stride,
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
    loss = LossInstrumentHCCEDensePartPose(args)
    trainer = TrainerHCCEDensePartPose(
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
        print("MODEL: MultiInstrumentHCCEDensePartDPT")
        print("LOSS:  LossInstrumentHCCEDensePartPose")
        print("DENSE PART ORDER: [wrist, gripper, shaft]")
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
                    aug_random_crop_rotate=bool(args.aug_random_crop_rotate),
                    aug_geom_prob=args.aug_geom_prob,
                    aug_crop_scale=args.aug_crop_scale,
                    aug_max_angle=args.aug_max_angle,
                    aug_offset_scale=args.aug_offset_scale,
                    aug_color_jitter=bool(args.aug_color_jitter),
                    dense_output_stride=args.dense_output_stride,
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
    parser.add_argument("--name", type=str, default="instrument_rarp_hcce_densepart_dpt_h2")
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
    parser.add_argument("--dense_output_stride", type=int, default=2)

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
    parser.add_argument("--aug_random_crop_rotate", type=int, default=1, choices=[0, 1])
    parser.add_argument("--aug_crop_scale", type=float, default=1.2)
    parser.add_argument("--aug_geom_prob", type=float, default=0.3)
    parser.add_argument("--aug_max_angle", type=float, default=float(np.pi / 6.0))
    parser.add_argument("--aug_offset_scale", type=float, default=1.0)
    parser.add_argument("--aug_color_jitter", type=int, default=1, choices=[0, 1])

    parser = LossInstrumentHCCEDensePartPose.add_specific_args(parser)
    main(parser.parse_args())
