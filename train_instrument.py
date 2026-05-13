import os
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ['EGL_DEVICE_ID'] = '0'

from argparse import ArgumentParser
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from datasets.surgical_instruments import SurgicalInstruments, collate_fn_instrument, RARP50_DIR
from multi_instrument import Multi_Instrument
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import time
from utils import AverageMeter, denormalize_rgb
from torch.utils.tensorboard import SummaryWriter
from loss_instrument import Loss_Instrument
from PIL import Image
import torch.nn.functional as F


# Colors for instance overlay (up to 6 instruments)
INST_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (0, 255, 255), (255, 0, 255),
]
# Colors for part overlay: 0=background(skip), 1=gripper=red, 2=wrist=green, 3=shaft=blue
PART_COLORS = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255)]


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def visualize_segmentation(img_tensor, gt_masks, gt_parts, pred_masks, pred_parts,
                           gt_locs=None, pred_locs=None, alpha=0.45):
    """
    Create a side-by-side visualization: [original | GT overlay | pred overlay].
    All masks at 1/4 resolution are upsampled to match image size.

    Args:
        img_tensor: [3, H, W] normalized image tensor (on CPU or GPU)
        gt_masks: list of [h, w] binary GT instance masks
        gt_parts: list of [h, w] GT part label maps (0=wrist,1=shaft,2=body)
        pred_masks: list of [h, w] binary predicted instance masks
        pred_parts: list of [h, w] predicted part label maps
        gt_locs: [n_gt, 2] wrist center locations (x,y) or None
        pred_locs: [n_pred, 2] predicted wrist center locations (x,y) or None
        alpha: overlay transparency
    Returns:
        np.array [H, 3*W, 3] uint8 (side-by-side image)
    """
    # Denormalize image
    img_np = denormalize_rgb(img_tensor.cpu().numpy())  # [H, W, 3] uint8
    H, W = img_np.shape[:2]

    def overlay_masks(base_img, masks, parts, locs, is_part=True):
        canvas = base_img.copy().astype(np.float32)
        for idx, (m, p) in enumerate(zip(masks, parts)):
            # Upsample mask to full resolution
            m_full = np.array(Image.fromarray(m.astype(np.uint8)).resize((W, H), Image.NEAREST))
            p_full = np.array(Image.fromarray(p.astype(np.uint8)).resize((W, H), Image.NEAREST))
            region = m_full > 0
            if region.sum() == 0:
                continue
            if is_part:
                # Color by part label
                color_map = np.zeros((H, W, 3), dtype=np.float32)
                for part_id, color in enumerate(PART_COLORS):
                    part_region = region & (p_full == part_id)
                    for c in range(3):
                        color_map[:, :, c][part_region] = color[c]
            else:
                # Color by instance
                color = INST_COLORS[idx % len(INST_COLORS)]
                color_map = np.zeros((H, W, 3), dtype=np.float32)
                for c in range(3):
                    color_map[:, :, c][region] = color[c]
            canvas[region] = canvas[region] * (1 - alpha) + color_map[region] * alpha

        # Draw wrist center markers
        if locs is not None:
            canvas_uint8 = canvas.clip(0, 255).astype(np.uint8)
            for loc in locs:
                if torch.isnan(loc).any():
                    continue
                cx, cy = int(loc[0].item()), int(loc[1].item())
                # Draw crosshair
                r = 6
                for dx in range(-r, r + 1):
                    for dy in [-1, 0, 1]:
                        px, py = cx + dx, cy + dy
                        if 0 <= px < W and 0 <= py < H:
                            canvas_uint8[py, px] = [255, 255, 255]
                        px, py = cx + dy, cy + dx
                        if 0 <= px < W and 0 <= py < H:
                            canvas_uint8[py, px] = [255, 255, 255]
            return canvas_uint8
        return canvas.clip(0, 255).astype(np.uint8)

    gt_overlay = overlay_masks(img_np, gt_masks, gt_parts, gt_locs)
    pred_overlay = overlay_masks(img_np, pred_masks, pred_parts, pred_locs)

    # Side-by-side: [original | GT | prediction]
    combined = np.concatenate([img_np, gt_overlay, pred_overlay], axis=1)
    return combined


class Trainer(object):
    def __init__(self, model, loss, optimizer, device, args, best_val=0.0):
        self.model = model
        self.loss = loss
        self.device = device
        self.args = args
        self.optimizer = optimizer
        self.best_val = best_val
        self.scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp))
        self.current_epoch = 0
        self.current_iter = 0

        # Access the underlying model (unwrap DDP)
        self.raw_model = model.module if isinstance(model, DDP) else model

        self.args.log_dir = os.path.join(self.args.save_dir, self.args.name)
        os.makedirs(self.args.log_dir, exist_ok=True)

        self.args.ckpt_dir = os.path.join(self.args.log_dir, 'checkpoints')
        os.makedirs(self.args.ckpt_dir, exist_ok=True)

        self.args.visu_dir = os.path.join(self.args.log_dir, 'visu')
        os.makedirs(self.args.visu_dir, exist_ok=True)

        self.writer = SummaryWriter(self.args.log_dir) if is_main_process() else None

    def prepare_gt(self, y):
        """
        Prepare ground truth for instrument detection + segmentation.
        Converts wrist centers to patch-level detection heatmap + offsets,
        and gathers instance/part masks for visible instruments.
        """
        bs, nh_max = y['valid_instruments'].shape

        valid = y['valid_instruments']  # [bs, max_inst]
        idx_h = torch.where(valid)      # (batch_indices, inst_indices)
        n_valid = int(valid.sum())

        if n_valid == 0:
            return None

        # Wrist centers in pixel coordinates
        wrist_centers = y['wrist_centers'][idx_h[0], idx_h[1]]  # [n_valid, 2] (x, y)

        # Convert to patch coordinates
        n_patch = self.args.img_size // self.raw_model.patch_size
        pk_coarse = (wrist_centers / self.raw_model.patch_size).int()  # [n_valid, 2]
        pk_idx = torch.clamp(pk_coarse, 0, n_patch - 1)

        # Sub-patch offset
        pk_offset = (wrist_centers - (pk_idx + 0.5) * self.raw_model.patch_size) / self.raw_model.patch_size

        # Detection heatmap + occlusion handling (same logic as train.py)
        scores = torch.zeros((bs, n_patch, n_patch)).to(self.device)
        visible = torch.ones(n_valid).to(self.device)

        for k in range(n_valid):
            i = int(idx_h[0][k])
            j = int(idx_h[1][k])
            _x = pk_idx[k, 1]  # note: swapped for the scores grid
            _y = pk_idx[k, 0]
            if scores[i, _x, _y] == 1:
                valid[i, j] = 0
                visible[k] = 0
            else:
                scores[i, _x, _y] = 1

        # Filter to visible instruments only
        idx_vis = torch.where(visible)[0]

        target = {}
        target['scores'] = scores
        target['offset'] = pk_offset[idx_vis]
        target['loc'] = wrist_centers[idx_vis]

        # Instance masks and part masks for visible instruments
        target['inst_masks'] = y['inst_masks'][idx_h[0], idx_h[1]][idx_vis].to(self.device)  # [n_vis, H/4, W/4]
        target['part_masks'] = y['part_masks'][idx_h[0], idx_h[1]][idx_vis].to(self.device)  # [n_vis, H/4, W/4]

        # Detection indices (batch, x_patch, y_patch, dummy)
        target['idx'] = tuple([
            idx_h[0].to(self.device)[idx_vis],
            pk_idx[:, 1].to(self.device)[idx_vis],
            pk_idx[:, 0].to(self.device)[idx_vis],
            torch.zeros_like(idx_h[0].to(self.device)[idx_vis])
        ])

        return target

    def save_checkpoint(self, name='last'):
        """Save checkpoint (rank 0 only). name='last' or 'best'."""
        if not is_main_process():
            return
        save_dict = {
            'epoch': self.current_epoch,
            'iter': self.current_iter,
            'model_state_dict': self.raw_model.state_dict(),
            'args': self.args,
            'best_val': self.best_val,
        }
        torch.save(save_dict, os.path.join(self.args.ckpt_dir, f"{name}.pt"))

    def fit(self, data_train, l_data_val):
        for epoch in range(self.args.max_epochs):
            self.current_epoch = epoch
            self.train_n_iters(data_train, l_data_val)
            self.save_checkpoint('last')
        return 1

    def train_n_iters(self, data, l_data_val):
        if is_main_process():
            print(f"\nTRAIN EPOCH {self.current_epoch}: ")
        self.model.train()

        # Set epoch for DistributedSampler
        if hasattr(data, 'sampler') and isinstance(data.sampler, DistributedSampler):
            data.sampler.set_epoch(self.current_epoch)

        meters = {k: AverageMeter(k) for k in ['workload/data', 'workload/batch', 'workload/ratio_data']}

        timer_end = time.time()
        iterator = tqdm(data) if is_main_process() else data
        for i, (x, y) in enumerate(iterator):
            # Move to device
            y = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in y.items()}
            data_time = time.time() - timer_end

            # Prepare GT
            gt = self.prepare_gt(y=y)
            if gt is None:
                timer_end = time.time()
                continue

            # Image to GPU
            x = x.to(self.device)

            # Forward
            with torch.cuda.amp.autocast(enabled=bool(self.args.amp)):
                pred = self.model(x, is_training=True, idx=gt['idx'])

                # Loss
                loss, dict_loss = self.loss(pred, gt, epoch=self.current_epoch, img_size=self.args.img_size)

            # Optim step (outside autocast)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

            batch_time = time.time() - timer_end

            # Meters
            meters['workload/data'].update(data_time)
            meters['workload/batch'].update(batch_time)
            meters['workload/ratio_data'].update(data_time / batch_time)
            for k, v in dict_loss.items():
                k_name = f"loss/{k}"
                if k_name not in meters:
                    meters[k_name] = AverageMeter(k_name)
                meters[k_name].update(dict_loss[k].item())

            # Log (rank 0 only)
            if is_main_process() and i % self.args.log_freq == 0:
                print(f"EPOCH={self.current_epoch:03d} - i={i:05d}/{len(data):05d} - "
                      f"iter={self.current_iter} - "
                      f"loss={meters['loss/total'].avg:.4f} - "
                      f"det={meters['loss/det'].avg:.4f} - "
                      f"dice={meters['loss/dice'].avg:.4f} - "
                      f"part={meters['loss/part_ce'].avg:.4f}")

                if self.writer is not None:
                    for k, v in meters.items():
                        self.writer.add_scalar(f"{k}", v.avg, self.current_iter)

                    # Log training sample visualization
                    self._log_train_vis(x, gt, pred)

                    self.writer.flush()
                sys.stdout.flush()

            # Validation by iteration
            if self.current_iter > 0 and self.current_iter % self.args.val_freq == 0:
                val_metric = 0.0
                for data_val in l_data_val:
                    val_metric = self.evaluate(data_val)
                # Save best checkpoint
                if val_metric > self.best_val:
                    self.best_val = val_metric
                    self.save_checkpoint('best')
                    if is_main_process():
                        print(f"*** New best: IoU={val_metric:.4f} ***")
                self.model.train()

            self.current_iter += 1
            timer_end = time.time()

        return 1

    def _log_train_vis(self, x, gt, pred):
        """Log training visualization for up to 10 images in the batch."""
        with torch.no_grad():
            n_vis = min(x.shape[0], 10)
            for vi in range(n_vis):
                img_inds = torch.where(gt['idx'][0] == vi)[0]
                if len(img_inds) == 0:
                    continue

                gt_masks_np = [gt['inst_masks'][j].cpu().numpy() for j in img_inds]
                gt_parts_np = [gt['part_masks'][j].cpu().numpy() for j in img_inds]
                gt_locs_vis = gt['loc'][img_inds]

                pred_masks_np = [torch.sigmoid(pred['inst_mask_logits'][j]).float().cpu().numpy()
                                 for j in img_inds]
                pred_parts_np = [pred['part_mask_logits'][j].argmax(dim=0).cpu().numpy()
                                 for j in img_inds]
                pred_locs_vis = pred['loc'][img_inds]

                vis = visualize_segmentation(
                    x[vi], gt_masks_np, gt_parts_np, pred_masks_np, pred_parts_np,
                    gt_locs=gt_locs_vis, pred_locs=pred_locs_vis,
                )
                self.writer.add_image(f"train/sample_{vi}", vis, self.current_iter, dataformats='HWC')

                fn = os.path.join(self.args.visu_dir,
                                  f"train_iter{self.current_iter:07d}_{vi:02d}.jpg")
                Image.fromarray(vis).save(fn)

    @torch.no_grad()
    def evaluate(self, data):
        if not is_main_process():
            return 0.0
        print(f"\nEVAL (iter={self.current_iter}): ")
        self.model.eval()

        meters = {k: AverageMeter(k) for k in ['inst_iou', 'part_acc', 'precision', 'recall', 'f1_score']}
        count, miss, fp = 0, 0, 0
        vis_count = 0

        for i, (x, y) in enumerate(tqdm(data)):
            y = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in y.items()}

            gt = self.prepare_gt(y=y)
            if gt is None:
                continue

            with torch.cuda.amp.autocast(enabled=bool(self.args.amp)):
                x = x.to(self.device)
                pred = self.raw_model(x, is_training=False,
                                      det_thresh=self.args.det_thresh,
                                      nms_kernel_size=self.args.nms_kernel_size)

            bs = x.shape[0]
            all_pred = pred if isinstance(pred, list) else []

            # Per-image evaluation
            for bi in range(bs):
                # GT for this image
                gt_inds = torch.where(gt['idx'][0] == bi)[0]
                n_gt_i = len(gt_inds)
                count += n_gt_i

                # Pred for this image
                pred_i = [p for p in all_pred if p['batch_idx'] == bi]
                n_pred_i = len(pred_i)

                if n_gt_i == 0 and n_pred_i == 0:
                    continue
                if n_pred_i == 0:
                    miss += n_gt_i
                    # Visualization (no predictions)
                    if vis_count < 10:
                        gt_masks_np = [gt['inst_masks'][j].cpu().numpy() for j in gt_inds]
                        gt_parts_np = [gt['part_masks'][j].cpu().numpy() for j in gt_inds]
                        gt_locs_vis = gt['loc'][gt_inds]
                        vis = visualize_segmentation(
                            x[bi], gt_masks_np, gt_parts_np, [], [],
                            gt_locs=gt_locs_vis, pred_locs=None,
                        )
                        self._save_eval_vis(vis, data, vis_count)
                        vis_count += 1
                    continue

                # Greedy matching
                pred_locs = torch.stack([p['loc'] for p in pred_i])
                gt_locs = gt['loc'][gt_inds]

                dist_matrix = torch.cdist(pred_locs.unsqueeze(0).float(),
                                          gt_locs.unsqueeze(0).float())[0]

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
                        dist_matrix[pi, gi] = float('inf')
                        continue
                    matched_pred.add(pi)
                    matched_gt.add(gi)
                    dist_matrix[pi, :] = float('inf')
                    dist_matrix[:, gi] = float('inf')

                    pred_mask = torch.sigmoid(pred_i[pi]['inst_mask_logits']) > 0.5
                    gt_mask = gt['inst_masks'][gt_inds[gi]] > 0.5
                    if pred_mask.shape != gt_mask.shape:
                        continue
                    intersection = (pred_mask & gt_mask).sum().float()
                    union = (pred_mask | gt_mask).sum().float()
                    iou = intersection / (union + 1e-6)
                    meters['inst_iou'].update(iou.item())

                    pred_parts = pred_i[pi]['part_mask_logits'].argmax(dim=0)
                    gt_parts = gt['part_masks'][gt_inds[gi]]
                    within_mask = gt_mask
                    if within_mask.sum() > 0:
                        correct = (pred_parts[within_mask] == gt_parts[within_mask]).float().mean()
                        meters['part_acc'].update(correct.item())

                miss += n_gt_i - len(matched_gt)
                fp += n_pred_i - len(matched_pred)

                # Visualization (up to 10 images)
                if vis_count < 10:
                    gt_masks_np = [gt['inst_masks'][j].cpu().numpy() for j in gt_inds]
                    gt_parts_np = [gt['part_masks'][j].cpu().numpy() for j in gt_inds]
                    gt_locs_vis = gt['loc'][gt_inds]

                    pred_masks_np = [(torch.sigmoid(p['inst_mask_logits']) > 0.5).cpu().numpy() for p in pred_i]
                    pred_parts_np = [p['part_mask_logits'].argmax(dim=0).cpu().numpy() for p in pred_i]
                    pred_locs_vis = torch.stack([p['loc'] for p in pred_i])

                    vis = visualize_segmentation(
                        x[bi], gt_masks_np, gt_parts_np, pred_masks_np, pred_parts_np,
                        gt_locs=gt_locs_vis, pred_locs=pred_locs_vis,
                    )
                    self._save_eval_vis(vis, data, vis_count)
                    vis_count += 1

            if i % self.args.log_freq == 0:
                precision = count / (count + fp) * 100 if (count + fp) > 0 else 0
                recall = (count - miss) / count * 100 if count > 0 else 0
                print(f"i={i} - Recall={recall:.1f}% - IoU={meters['inst_iou'].avg:.3f} - PartAcc={meters['part_acc'].avg:.3f}")
                sys.stdout.flush()

        # Final metrics
        print(f"***EVAL - {data.dataset.name}-{data.dataset.split}***")
        if count > 0:
            precision = (count - fp) / count * 100
            recall = (count - miss) / count * 100
            f1 = 2 * precision * recall / (precision + recall + 1e-6)
        else:
            precision = recall = f1 = 0
        meters['precision'].update(precision)
        meters['recall'].update(recall)
        meters['f1_score'].update(f1)
        for k, v in meters.items():
            if self.writer is not None:
                self.writer.add_scalar(f"{data.dataset.name}-{data.dataset.split}/{k}", v.avg, self.current_iter)
            print(f"    - {k}: {v.avg:.3f}")
        if self.writer is not None:
            self.writer.flush()
        sys.stdout.flush()

        return meters['inst_iou'].avg

    def _save_eval_vis(self, vis, data, vis_idx):
        """Save eval visualization to disk and TensorBoard."""
        fn = os.path.join(self.args.visu_dir,
                          f"eval_iter{self.current_iter:07d}_{data.dataset.name}_{vis_idx:02d}.jpg")
        Image.fromarray(vis).save(fn)
        if self.writer is not None:
            self.writer.add_image(
                f"{data.dataset.name}-{data.dataset.split}/sample_{vis_idx}",
                vis, self.current_iter, dataformats='HWC')


def setup_ddp():
    """Initialize DDP if launched via torchrun."""
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        return torch.device(f'cuda:{local_rank}'), rank, world_size
    else:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu'), 0, 1


def main(args):
    device, rank, world_size = setup_ddp()
    use_ddp = world_size > 1

    model = Multi_Instrument(pretrained_backbone=True, **vars(args))
    model = model.to(device)

    # Load from pretrained
    if args.pretrained is not None and os.path.isfile(args.pretrained):
        if is_main_process():
            print(f"Loading weights from {args.pretrained}")
        ckpt = torch.load(args.pretrained, map_location=device)
        log = model.load_state_dict(ckpt['model_state_dict'], strict=False)
        if is_main_process():
            print(f"{log}")

    # Wrap with DDP
    if use_ddp:
        model = DDP(model, device_ids=[device], find_unused_parameters=True)

    # Validation data (only on rank 0)
    l_val_data = []
    val_data = DataLoader(
        SurgicalInstruments(split='test', training=False, img_size=args.img_size,
                            subsample=args.val_subsample),
        batch_size=args.val_batch_size, num_workers=args.num_workers,
        shuffle=False, drop_last=False,
        collate_fn=collate_fn_instrument,
    )
    l_val_data.append(val_data)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss = Loss_Instrument(args)
    trainer = Trainer(model=model, loss=loss, optimizer=optimizer, device=device, args=args)

    if is_main_process():
        print()
        print(f"ARGS: {trainer.args}")
        print(f"LOG_DIR: {trainer.args.log_dir}")
        print()

    if args.eval_only:
        for vd in l_val_data:
            trainer.evaluate(vd)
    else:
        train_dataset = SurgicalInstruments(
            split='train', training=True, img_size=args.img_size,
            subsample=args.train_subsample,
        )
        train_sampler = DistributedSampler(train_dataset, shuffle=True) if use_ddp else None
        train_data = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=(train_sampler is None),
            drop_last=True,
            collate_fn=collate_fn_instrument,
            sampler=train_sampler,
        )
        trainer.fit(train_data, l_val_data)

    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = ArgumentParser()

    # Data
    parser.add_argument('--save_dir', type=str, default='logs')
    parser.add_argument('--name', type=str, default='instrument_stage1')
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--data_dir', type=str, default=RARP50_DIR)

    # Training
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', '-j', type=int, default=4)
    parser.add_argument('--img_size', type=int, default=896)
    parser.add_argument('--backbone', type=str, default='dinov2_vits14',
                        choices=['dinov2_vitl14', 'dinov2_vitb14', 'dinov2_vits14'])
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--log_freq', type=int, default=100)
    parser.add_argument('--max_iter', type=int, default=60000, help='(unused, kept for compat)')
    parser.add_argument('--nb_max_ckpt', type=int, default=10)
    parser.add_argument('--amp', type=int, default=1, choices=[0, 1])
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-5)
    parser.add_argument('--eval_only', type=int, default=0, choices=[0, 1])
    parser.add_argument('--det_thresh', type=float, default=0.3)
    parser.add_argument('--nms_kernel_size', type=int, default=3)
    parser.add_argument('--val_freq', type=int, default=1000, help='validate every N iterations')
    parser.add_argument('--val_batch_size', type=int, default=8)
    parser.add_argument('--train_subsample', type=int, default=1)
    parser.add_argument('--val_subsample', type=int, default=10)

    # Model
    parser.add_argument('--xat_dim', type=int, default=512)
    parser.add_argument('--xat_depth', type=int, default=4)
    parser.add_argument('--xat_heads', type=int, default=16)
    parser.add_argument('--xat_dim_head', type=int, default=32)
    parser.add_argument('--xat_mlp_dim', type=int, default=2048)
    parser.add_argument('--mask_dim', type=int, default=256)
    parser.add_argument('--num_parts', type=int, default=4)

    parser = Loss_Instrument.add_specific_args(parser)
    args = parser.parse_args()
    main(args)

