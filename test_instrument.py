import os
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ['EGL_DEVICE_ID'] = '0'

from argparse import ArgumentParser
import torch
import numpy as np
from datasets.surgical_instruments import SurgicalInstruments, collate_fn_instrument, RARP50_DIR
from datasets.endovis_dataset import EndoVisDataset
from multi_instrument import Multi_Instrument
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
from utils import AverageMeter, denormalize_rgb
from PIL import Image

# Colors for part overlay: 0=background(skip), 1=gripper=red, 2=wrist=green, 3=shaft=blue
PART_COLORS = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255)]
PART_NAMES = {1: 'gripper', 2: 'wrist', 3: 'shaft'}


def visualize_segmentation(img_tensor, gt_masks, gt_parts, pred_masks, pred_parts,
                           gt_locs=None, pred_locs=None, alpha=0.45):
    """Create [original | GT overlay | pred overlay] side-by-side."""
    img_np = denormalize_rgb(img_tensor.cpu().numpy())
    H, W = img_np.shape[:2]

    def overlay_masks(base_img, masks, parts, locs):
        canvas = base_img.copy().astype(np.float32)
        for m, p in zip(masks, parts):
            m_full = np.array(Image.fromarray(m.astype(np.uint8)).resize((W, H), Image.NEAREST))
            p_full = np.array(Image.fromarray(p.astype(np.uint8)).resize((W, H), Image.NEAREST))
            region = m_full > 0
            if region.sum() == 0:
                continue
            color_map = np.zeros((H, W, 3), dtype=np.float32)
            for part_id, color in enumerate(PART_COLORS):
                part_region = region & (p_full == part_id)
                for c in range(3):
                    color_map[:, :, c][part_region] = color[c]
            canvas[region] = canvas[region] * (1 - alpha) + color_map[region] * alpha

        if locs is not None:
            canvas_uint8 = canvas.clip(0, 255).astype(np.uint8)
            for loc in locs:
                if torch.isnan(loc).any():
                    continue
                cx, cy = int(loc[0].item()), int(loc[1].item())
                r = 6
                for dx in range(-r, r + 1):
                    for dy in [-1, 0, 1]:
                        for px, py in [(cx + dx, cy + dy), (cx + dy, cy + dx)]:
                            if 0 <= px < W and 0 <= py < H:
                                canvas_uint8[py, px] = [255, 255, 255]
            return canvas_uint8
        return canvas.clip(0, 255).astype(np.uint8)

    gt_overlay = overlay_masks(img_np, gt_masks, gt_parts, gt_locs)
    pred_overlay = overlay_masks(img_np, pred_masks, pred_parts, pred_locs)
    return np.concatenate([img_np, gt_overlay, pred_overlay], axis=1)


def compute_dice(pred_mask, gt_mask, eps=1e-6):
    """Dice score between two binary masks."""
    pred = pred_mask.float().flatten()
    gt = gt_mask.float().flatten()
    intersection = (pred * gt).sum()
    return (2 * intersection + eps) / (pred.sum() + gt.sum() + eps)


def compute_part_dice(pred_parts, gt_parts, gt_inst_mask, num_parts=4):
    """
    Per-part Dice within the GT instance mask region.
    Returns dict {part_id: dice} for parts 1,2,3.
    """
    within = gt_inst_mask > 0
    result = {}
    for pid in range(1, num_parts):
        gt_region = within & (gt_parts == pid)
        pred_region = within & (pred_parts == pid)
        if gt_region.sum() == 0 and pred_region.sum() == 0:
            continue  # skip parts not present
        result[pid] = compute_dice(pred_region, gt_region).item()
    return result


class Evaluator:
    def __init__(self, model, device, args):
        self.model = model
        self.device = device
        self.args = args
        self.raw_model = model

    def prepare_gt(self, y):
        """Same as Trainer.prepare_gt but without selective loss gating."""
        bs, nh_max = y['valid_instruments'].shape

        valid = y['valid_instruments']
        idx_h = torch.where(valid)
        n_valid = int(valid.sum())

        if n_valid == 0:
            return None

        wrist_centers = y['wrist_centers'][idx_h[0], idx_h[1]]

        n_patch = self.args.img_size // self.raw_model.patch_size
        pk_coarse = (wrist_centers / self.raw_model.patch_size).int()
        pk_idx = torch.clamp(pk_coarse, 0, n_patch - 1)

        scores = torch.zeros((bs, n_patch, n_patch)).to(self.device)
        visible = torch.ones(n_valid).to(self.device)

        for k in range(n_valid):
            i = int(idx_h[0][k])
            j = int(idx_h[1][k])
            _x = pk_idx[k, 1]
            _y = pk_idx[k, 0]
            if scores[i, _x, _y] == 1:
                valid[i, j] = 0
                visible[k] = 0
            else:
                scores[i, _x, _y] = 1

        idx_vis = torch.where(visible)[0]

        target = {
            'scores': scores,
            'loc': wrist_centers[idx_vis],
            'inst_masks': y['inst_masks'][idx_h[0], idx_h[1]][idx_vis].to(self.device),
            'part_masks': y['part_masks'][idx_h[0], idx_h[1]][idx_vis].to(self.device),
            'idx': (
                idx_h[0].to(self.device)[idx_vis],
                pk_idx[:, 1].to(self.device)[idx_vis],
                pk_idx[:, 0].to(self.device)[idx_vis],
                torch.zeros_like(idx_h[0].to(self.device)[idx_vis]),
            ),
        }
        return target

    @torch.no_grad()
    def evaluate(self, data, visu_dir=None):
        """
        Evaluate on a dataloader. Returns dict of metrics.
        """
        self.model.eval()

        meters = {k: AverageMeter(k) for k in [
            'inst_dice',
            'part_dice_gripper', 'part_dice_wrist', 'part_dice_shaft',
        ]}
        # Detection counters
        n_gt_total = 0
        n_pred_total = 0
        n_tp = 0        # matched pairs
        vis_count = 0
        max_vis = self.args.max_vis

        for i, (x, y) in enumerate(tqdm(data, desc=f"Eval {data.dataset.name}-{data.dataset.split}")):
            y = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                 for k, v in y.items()}

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

            for bi in range(bs):
                gt_inds = torch.where(gt['idx'][0] == bi)[0]
                n_gt_i = len(gt_inds)
                pred_i = [p for p in all_pred if p['batch_idx'] == bi]
                n_pred_i = len(pred_i)

                n_gt_total += n_gt_i
                n_pred_total += n_pred_i

                if n_gt_i == 0 and n_pred_i == 0:
                    continue
                if n_pred_i == 0:
                    # Save vis for missed detections
                    if visu_dir and vis_count < max_vis:
                        gt_masks_np = [gt['inst_masks'][j].cpu().numpy() for j in gt_inds]
                        gt_parts_np = [gt['part_masks'][j].cpu().numpy() for j in gt_inds]
                        vis = visualize_segmentation(
                            x[bi], gt_masks_np, gt_parts_np, [], [],
                            gt_locs=gt['loc'][gt_inds], pred_locs=None)
                        Image.fromarray(vis).save(
                            os.path.join(visu_dir, f"{vis_count:04d}.jpg"))
                        vis_count += 1
                    continue

                # Greedy matching by distance
                pred_locs = torch.stack([p['loc'] for p in pred_i])
                gt_locs = gt['loc'][gt_inds]
                dist_matrix = torch.cdist(
                    pred_locs.unsqueeze(0).float(),
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

                    # Instance dice
                    pred_mask = torch.sigmoid(pred_i[pi]['inst_mask_logits']) > 0.5
                    gt_mask = gt['inst_masks'][gt_inds[gi]] > 0.5
                    if pred_mask.shape != gt_mask.shape:
                        continue
                    dice = compute_dice(pred_mask, gt_mask)
                    meters['inst_dice'].update(dice.item())

                    # Per-part dice
                    pred_parts = pred_i[pi]['part_mask_logits'].argmax(dim=0)
                    gt_parts = gt['part_masks'][gt_inds[gi]]
                    part_dices = compute_part_dice(pred_parts, gt_parts, gt_mask)
                    for pid, d in part_dices.items():
                        meters[f'part_dice_{PART_NAMES[pid]}'].update(d)

                n_tp += len(matched_gt)

                # Save visualization
                if visu_dir and vis_count < max_vis:
                    gt_masks_np = [gt['inst_masks'][j].cpu().numpy() for j in gt_inds]
                    gt_parts_np = [gt['part_masks'][j].cpu().numpy() for j in gt_inds]
                    pred_masks_np = [(torch.sigmoid(p['inst_mask_logits']) > 0.5).cpu().numpy()
                                     for p in pred_i]
                    pred_parts_np = [p['part_mask_logits'].argmax(dim=0).cpu().numpy()
                                     for p in pred_i]
                    vis = visualize_segmentation(
                        x[bi], gt_masks_np, gt_parts_np, pred_masks_np, pred_parts_np,
                        gt_locs=gt['loc'][gt_inds],
                        pred_locs=torch.stack([p['loc'] for p in pred_i]))
                    Image.fromarray(vis).save(
                        os.path.join(visu_dir, f"{vis_count:04d}.jpg"))
                    vis_count += 1

        # Compute detection metrics
        precision = n_tp / n_pred_total * 100 if n_pred_total > 0 else 0.0
        recall = n_tp / n_gt_total * 100 if n_gt_total > 0 else 0.0
        accuracy = n_tp / (n_gt_total + n_pred_total - n_tp) * 100 if (n_gt_total + n_pred_total - n_tp) > 0 else 0.0

        results = {
            'dataset': f"{data.dataset.name}-{data.dataset.split}",
            'n_images': len(data.dataset),
            'n_gt': n_gt_total,
            'n_pred': n_pred_total,
            'n_tp': n_tp,
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy,
            'inst_dice': meters['inst_dice'].avg if meters['inst_dice'].count > 0 else 0.0,
            'part_dice_gripper': meters['part_dice_gripper'].avg if meters['part_dice_gripper'].count > 0 else 0.0,
            'part_dice_wrist': meters['part_dice_wrist'].avg if meters['part_dice_wrist'].count > 0 else 0.0,
            'part_dice_shaft': meters['part_dice_shaft'].avg if meters['part_dice_shaft'].count > 0 else 0.0,
        }

        # Print
        print(f"\n{'='*60}")
        print(f"  {results['dataset']}  ({results['n_images']} images)")
        print(f"  GT instruments: {n_gt_total}  Predicted: {n_pred_total}  Matched: {n_tp}")
        print(f"  Detection  - Precision: {precision:.1f}%  Recall: {recall:.1f}%  Accuracy: {accuracy:.1f}%")
        print(f"  Inst Dice  : {results['inst_dice']:.4f}")
        print(f"  Part Dice  - gripper: {results['part_dice_gripper']:.4f}"
              f"  wrist: {results['part_dice_wrist']:.4f}"
              f"  shaft: {results['part_dice_shaft']:.4f}")
        print(f"{'='*60}")
        sys.stdout.flush()

        return results


def save_results_table(all_results, save_path, args):
    """Save results as a markdown table."""
    lines = []
    lines.append(f"# Evaluation Results")
    lines.append(f"")
    lines.append(f"- Checkpoint: `{args.pretrained}`")
    lines.append(f"- img_size: {args.img_size}")
    lines.append(f"- backbone: {args.backbone}")
    lines.append(f"- det_thresh: {args.det_thresh}")
    lines.append(f"- nms_kernel_size: {args.nms_kernel_size}")
    lines.append(f"")

    # Detection table
    lines.append(f"## Detection")
    lines.append(f"")
    lines.append(f"| Dataset | #Images | #GT | #Pred | #TP | Precision | Recall | Accuracy |")
    lines.append(f"|---------|---------|-----|-------|-----|-----------|--------|----------|")
    for r in all_results:
        lines.append(f"| {r['dataset']} | {r['n_images']} | {r['n_gt']} | {r['n_pred']} | {r['n_tp']} "
                      f"| {r['precision']:.1f}% | {r['recall']:.1f}% | {r['accuracy']:.1f}% |")
    lines.append(f"")

    # Segmentation table
    lines.append(f"## Segmentation (matched instruments only)")
    lines.append(f"")
    lines.append(f"| Dataset | Inst Dice | Gripper Dice | Wrist Dice | Shaft Dice |")
    lines.append(f"|---------|-----------|--------------|------------|------------|")
    for r in all_results:
        lines.append(f"| {r['dataset']} | {r['inst_dice']:.4f} "
                      f"| {r['part_dice_gripper']:.4f} | {r['part_dice_wrist']:.4f} "
                      f"| {r['part_dice_shaft']:.4f} |")
    lines.append(f"")

    table_str = '\n'.join(lines)
    with open(save_path, 'w') as f:
        f.write(table_str)
    print(f"\nResults saved to {save_path}")
    print(table_str)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build model
    model_kwargs = {k: v for k, v in vars(args).items()
                    if k in ('backbone', 'xat_dim', 'xat_depth', 'xat_heads',
                             'xat_dim_head', 'xat_mlp_dim', 'mask_dim', 'num_parts', 'img_size')}
    model = Multi_Instrument(pretrained_backbone=False, **model_kwargs)
    model = model.to(device)

    # Load checkpoint
    if args.pretrained is not None and os.path.isfile(args.pretrained):
        print(f"Loading weights from {args.pretrained}")
        ckpt = torch.load(args.pretrained, map_location=device, weights_only=False)
        log = model.load_state_dict(ckpt['model_state_dict'], strict=False)
        print(f"  {log}")
        if 'args' in ckpt:
            ckpt_args = ckpt['args']
            print(f"  Checkpoint trained at iter={getattr(ckpt_args, 'max_iter', '?')}, "
                  f"best_val={ckpt.get('best_val', '?')}")
    else:
        print(f"WARNING: checkpoint not found at {args.pretrained}")
    model.eval()

    evaluator = Evaluator(model, device, args)

    # Determine output directory: parent of checkpoint dir (experiment dir)
    ckpt_dir = os.path.dirname(os.path.abspath(args.pretrained))
    exp_dir = os.path.dirname(ckpt_dir)
    visu_base = os.path.join(exp_dir, 'eval_visu')

    # Build evaluation datasets
    eval_configs = []
    
    # 2. EndoVis17 valid
    eval_configs.append({
        'name': 'endovis17_valid',
        'dataset': EndoVisDataset(
            root_dir=args.endovis17_root, split='valid',
            training=False, img_size=args.img_size, subsample=args.subsample),
    })
    # 1. RARP50 test
    eval_configs.append({
        'name': 'rarp50_test',
        'dataset': SurgicalInstruments(
            split='test', training=False, img_size=args.img_size,
            root_dir=args.rarp50_root, subsample=args.subsample),
    })



    # Run evaluation
    all_results = []
    for cfg in eval_configs:
        ds = cfg['dataset']
        print(f"\n>>> Evaluating {ds.name}-{ds.split} ({len(ds)} samples)")

        loader = DataLoader(
            ds, batch_size=args.batch_size, num_workers=args.num_workers,
            shuffle=False, drop_last=False, collate_fn=collate_fn_instrument)

        visu_dir = os.path.join(visu_base, cfg['name'])
        os.makedirs(visu_dir, exist_ok=True)

        results = evaluator.evaluate(loader, visu_dir=visu_dir)
        all_results.append(results)

    # Save results table
    ckpt_name = os.path.splitext(os.path.basename(args.pretrained))[0]
    table_path = os.path.join(exp_dir, f'eval_{ckpt_name}.md')
    save_results_table(all_results, table_path, args)


if __name__ == "__main__":
    parser = ArgumentParser()

    # Checkpoint
    parser.add_argument('--pretrained', type=str,
                        default='logs/rarp_endovis/checkpoints/best.pt')

    # Data roots
    parser.add_argument('--rarp50_root', type=str, default=RARP50_DIR) # RARP50_DIR 
    parser.add_argument('--endovis17_root', type=str,
                        default='/mnt/nas/haofeng/data/Ref-Endovis18')

    # Eval settings
    parser.add_argument('--img_size', type=int, default=630)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', '-j', type=int, default=4)
    parser.add_argument('--amp', type=int, default=1, choices=[0, 1])
    parser.add_argument('--det_thresh', type=float, default=0.3)
    parser.add_argument('--nms_kernel_size', type=int, default=3)
    parser.add_argument('--subsample', type=int, default=1,
                        help='Keep every Nth sample (1 = full eval)')
    parser.add_argument('--max_vis', type=int, default=500,
                        help='Max number of visualization images to save per dataset')

    # Model architecture (must match checkpoint)
    parser.add_argument('--backbone', type=str, default='dinov2_vits14',
                        choices=['dinov2_vitl14', 'dinov2_vitb14', 'dinov2_vits14',
                                 'dinov3_vits16', 'dinov3_vitb16', 'dinov3_vitl16'])
    parser.add_argument('--xat_dim', type=int, default=512)
    parser.add_argument('--xat_depth', type=int, default=4)
    parser.add_argument('--xat_heads', type=int, default=16)
    parser.add_argument('--xat_dim_head', type=int, default=32)
    parser.add_argument('--xat_mlp_dim', type=int, default=2048)
    parser.add_argument('--mask_dim', type=int, default=256)
    parser.add_argument('--num_parts', type=int, default=4)

    args = parser.parse_args()
    main(args)
