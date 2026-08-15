import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROBOPEPP_ROOT = Path(__file__).resolve().parent
MULTIHMR_ROOT = Path(__file__).resolve().parents[2]
if str(ROBOPEPP_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBOPEPP_ROOT))
if str(MULTIHMR_ROOT) not in sys.path:
    sys.path.insert(0, str(MULTIHMR_ROOT))

from instrument_geometry import instrument_keypoints_camera_torch, project_points_torch  # noqa: E402
from loss_instrument import dice_loss  # noqa: E402
from multi_instrument.hcce_codec import normalized_xyz_to_hcce  # noqa: E402


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
    b, c, h, w = heatmaps.shape
    flat = heatmaps.reshape(b, c, -1)
    inds = flat.argmax(dim=-1)
    y = torch.div(inds, w, rounding_mode="floor")
    x = inds % w
    return torch.stack([x, y], dim=-1).float()


def _part_target_to_dense_order(part_mask):
    target = torch.full_like(part_mask.long(), -1)
    target[part_mask == 2] = 0  # wrist
    target[part_mask == 1] = 1  # gripper
    target[part_mask == 3] = 2  # shaft
    return target


def compute_crop_hcce_losses(out, y, args):
    inst_gt = y["inst_mask"].float()
    inst_loss_dice = dice_loss(torch.sigmoid(out["inst_mask_logits"]), inst_gt)
    inst_loss_bce = F.binary_cross_entropy_with_logits(out["inst_mask_logits"], inst_gt)

    part_target = _part_target_to_dense_order(y["part_mask"])
    part_valid = (part_target >= 0) & (inst_gt > 0.5)
    if part_valid.any():
        part_logits_flat = out["part_mask_logits"].permute(0, 2, 3, 1)[part_valid]
        part_target_flat = part_target[part_valid]
        part_ce = F.cross_entropy(part_logits_flat, part_target_flat)
        part_acc = (part_logits_flat.detach().argmax(dim=1) == part_target_flat).float().mean()
    else:
        part_ce = torch.tensor(0.0, device=inst_gt.device)
        part_acc = torch.tensor(0.0, device=inst_gt.device)

    hcce_loss, hcce_bit_acc, hcce_px = _compute_hcce_loss(out, y, args)
    heatmap_loss = focal_heatmap_loss(out["keypoint_heatmaps"], y["heatmaps"].float())

    pred_quat = F.normalize(out["wrist_quat_pred"], p=2, dim=1)
    gt_quat = F.normalize(y["wrist_quat"], p=2, dim=1)
    sign = torch.where((pred_quat * gt_quat).sum(dim=1, keepdim=True) < 0.0, -1.0, 1.0)
    quat_l1 = F.l1_loss(pred_quat, gt_quat * sign)
    action_l1 = F.l1_loss(out["action_pred"], y["action"])
    trans_l1 = F.l1_loss(out["wrist_trans_pred"], y["wrist_trans"])

    kp_pred = heatmap_argmax(out["keypoint_heatmaps"])
    valid = y["keypoints_valid"].bool()
    if valid.any():
        kp_err = torch.linalg.norm(kp_pred[valid] - y["keypoints_crop"][valid], dim=-1).mean()
    else:
        kp_err = torch.tensor(0.0, device=inst_gt.device)

    keypoints_3d_pred = instrument_keypoints_camera_torch(
        pred_quat,
        out["wrist_trans_pred"],
        out["action_pred"],
    )
    kp3d_l1 = F.l1_loss(keypoints_3d_pred, y["keypoints_3d_cam"])
    uv_pred = project_points_torch(keypoints_3d_pred, y["K"])
    valid_uv = (
        valid
        & torch.isfinite(uv_pred).all(dim=-1)
        & torch.isfinite(keypoints_3d_pred).all(dim=-1)
        & (keypoints_3d_pred[..., 2] > 1e-4)
    )
    if valid_uv.any():
        uv_loss = F.smooth_l1_loss(uv_pred[valid_uv] / 224.0, y["keypoints_crop"][valid_uv] / 224.0)
        uv_err = torch.linalg.norm(uv_pred[valid_uv] - y["keypoints_crop"][valid_uv], dim=-1).mean()
    else:
        uv_loss = torch.tensor(0.0, device=inst_gt.device)
        uv_err = torch.tensor(0.0, device=inst_gt.device)

    total = (
        args.alpha_dice * inst_loss_dice
        + args.alpha_bce_mask * inst_loss_bce
        + args.alpha_part * part_ce
        + args.alpha_hcce * hcce_loss
        + args.alpha_heatmap * heatmap_loss
        + args.alpha_action_l1 * action_l1
        + args.alpha_wrist_quat_l1 * quat_l1
        + args.alpha_wrist_trans_l1 * trans_l1
        + args.alpha_keypoint_2d * uv_loss
        + args.alpha_keypoint_3d * kp3d_l1
    )
    metrics = {
        "total": total.detach(),
        "dice": inst_loss_dice.detach(),
        "bce_mask": inst_loss_bce.detach(),
        "part_ce": part_ce.detach(),
        "part_acc": part_acc.detach(),
        "hcce": hcce_loss.detach(),
        "hcce_bit_acc": hcce_bit_acc.detach(),
        "hcce_px": hcce_px.detach(),
        "heatmap": heatmap_loss.detach(),
        "kp_err_px": kp_err.detach(),
        "action_l1": action_l1.detach(),
        "wrist_quat_l1": quat_l1.detach(),
        "wrist_trans_l1": trans_l1.detach(),
        "uv_loss": uv_loss.detach(),
        "uv_err_px": uv_err.detach(),
        "kp3d_l1": kp3d_l1.detach(),
    }
    return total, metrics


def _compute_hcce_loss(out, y, args):
    pred = out["hcce_logits"]
    coord_imgs = y["coord_img"].float()
    inst = y["inst_mask"].float()
    has_cse = y["has_cse"].bool()
    coord_min = float(getattr(args, "hcce_coord_min", -1.0))
    coord_max = float(getattr(args, "hcce_coord_max", 1.0))
    losses = []
    bit_correct = torch.tensor(0.0, device=pred.device)
    bit_count = torch.tensor(0.0, device=pred.device)
    px_count = torch.tensor(0.0, device=pred.device)
    for k in torch.where(has_cse)[0]:
        coord = coord_imgs[k]
        valid = (coord[..., 3].long() > 0) & (inst[k] > 0.5)
        if not valid.any():
            continue
        gt_hcce = normalized_xyz_to_hcce(
            coord[..., :3],
            iteration=args.hcce_bits,
            coord_min=coord_min,
            coord_max=coord_max,
        )
        target = gt_hcce.permute(2, 0, 1)
        target_signed = target * 2.0 - 1.0
        losses.append(F.l1_loss(pred[k, :, valid], target_signed[:, valid]))

        pred_bits = (torch.sigmoid(pred[k].detach()) > 0.5).float()
        gt_bits = (target.detach() > 0.5).float()
        bit_correct = bit_correct + (pred_bits[:, valid] == gt_bits[:, valid]).float().sum()
        bit_count = bit_count + gt_bits[:, valid].numel()
        px_count = px_count + valid.sum().float()
    loss = torch.stack(losses).mean() if losses else torch.tensor(0.0, device=pred.device)
    return (
        torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0),
        bit_correct / bit_count.clamp_min(1.0),
        px_count,
    )
