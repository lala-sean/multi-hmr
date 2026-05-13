import torch
import torch.nn.functional as F
from argparse import ArgumentParser


def _neg_loss(pred, gt):
    """
    CenterNet focal loss variant.
    Code modified from: https://github.com/xingyizhou/CenterNet
    Args:
        pred (batch x h x w): predicted detection scores (sigmoid activated)
        gt (batch x h x w): ground truth heatmap (0 or 1)
    """
    assert pred.shape == gt.shape

    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()
    neg_weights = torch.pow(1 - gt, 4)

    loss = 0
    eps = 1e-7

    pos_loss = torch.log(pred + eps) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred + eps) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


def dice_loss(pred, target, eps=1e-6):
    """
    Compute dice loss per instance, then average.
    Args:
        pred: [N, H, W] sigmoid-activated predictions
        target: [N, H, W] binary ground truth
    Returns:
        scalar loss
    """
    pred = pred.flatten(1)   # [N, H*W]
    target = target.flatten(1)

    intersection = (pred * target).sum(1)
    union = pred.sum(1) + target.sum(1)

    dice = 1 - (2 * intersection + eps) / (union + eps)
    return dice.mean()


def part_ce_loss(part_logits, gt_part_labels):
    """
    Cross-entropy loss for part segmentation on all pixels.
    Args:
        part_logits: [N, num_parts, H, W] raw logits
        gt_part_labels: [N, H, W] part labels (long, 0=bg, 1=wrist, 2=shaft, 3=gripper)
    Returns:
        scalar loss
    """
    N = part_logits.shape[0]
    if N == 0:
        return torch.tensor(0.0, device=part_logits.device)

    return F.cross_entropy(part_logits, gt_part_labels, reduction='mean')


class Loss_Instrument(torch.nn.Module):
    def __init__(self, parser_args, *args, **kwargs):
        super().__init__()
        self.parser_args = parser_args

    def forward(self, y_hat, y, epoch=None, img_size=None):
        # Detection (focal loss) — gated to fully-annotated (RARP50) samples only
        gt_scores = (y['scores'] >= 1).float()
        if 'det_source_mask' in y and y['det_source_mask'].any():
            m = y['det_source_mask']
            bce = _neg_loss(y_hat['scores'][m], gt_scores[m])
        elif 'det_source_mask' in y:
            bce = torch.tensor(0., device=y_hat['scores'].device)
        else:
            bce = _neg_loss(y_hat['scores'], gt_scores)

        # Offset (L1) — gated to fully-annotated instruments only
        if 'rarp50_inst_mask' in y and y['rarp50_inst_mask'].any():
            m = y['rarp50_inst_mask']
            reg_offset = (y_hat['offset'][m] - y['offset'][m]).abs().sum(-1).mean(0)
        elif 'rarp50_inst_mask' in y:
            reg_offset = torch.tensor(0., device=y_hat['offset'].device)
        else:
            reg_offset = (y_hat['offset'] - y['offset']).abs().sum(-1).mean(0)

        # Instance mask (Dice + BCE)
        pred_inst_mask = torch.sigmoid(y_hat['inst_mask_logits'])  # [n_inst, H/4, W/4]
        gt_inst_mask = y['inst_masks']  # [n_inst, H/4, W/4]

        loss_dice = dice_loss(pred_inst_mask, gt_inst_mask)
        loss_bce_mask = F.binary_cross_entropy_with_logits(
            y_hat['inst_mask_logits'], gt_inst_mask, reduction='mean'
        )

        # Part segmentation (CE within instance mask)
        loss_part = part_ce_loss(
            y_hat['part_mask_logits'],  # [n_inst, num_parts, H/4, W/4]
            y['part_masks']             # [n_inst, H/4, W/4] long
        )

        # Handle nan/inf
        bce = torch.nan_to_num(bce, nan=0.0, posinf=0.0, neginf=0.0)
        reg_offset = torch.nan_to_num(reg_offset, nan=0.0, posinf=0.0, neginf=0.0)
        loss_dice = torch.nan_to_num(loss_dice, nan=0.0, posinf=0.0, neginf=0.0)
        loss_bce_mask = torch.nan_to_num(loss_bce_mask, nan=0.0, posinf=0.0, neginf=0.0)
        loss_part = torch.nan_to_num(loss_part, nan=0.0, posinf=0.0, neginf=0.0)

        # Total loss
        total_loss = (self.parser_args.alpha_det * bce +
                      self.parser_args.alpha_offset * reg_offset +
                      self.parser_args.alpha_dice * loss_dice +
                      self.parser_args.alpha_bce_mask * loss_bce_mask +
                      self.parser_args.alpha_part * loss_part)

        dict_loss = {
            'total': total_loss,
            'det': bce,
            'offset': reg_offset,
            'dice': loss_dice,
            'bce_mask': loss_bce_mask,
            'part_ce': loss_part,
        }

        return total_loss, dict_loss

    @staticmethod
    def add_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # Detection
        parser.add_argument('--alpha_det', type=float, default=10.0)
        parser.add_argument('--alpha_offset', type=float, default=1.0)
        # Instance segmentation
        parser.add_argument('--alpha_dice', type=float, default=5.0)
        parser.add_argument('--alpha_bce_mask', type=float, default=2.0)
        # Part segmentation
        parser.add_argument('--alpha_part', type=float, default=2.0)
        return parser
