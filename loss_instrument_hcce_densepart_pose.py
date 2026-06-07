import torch
import torch.nn.functional as F

from loss_instrument import Loss_Instrument, _neg_loss, dice_loss
from loss_instrument_hcce_pose import LossInstrumentHCCEPose
from multi_instrument.hcce_codec import normalized_xyz_to_hcce


def _nearest_resize_label(x, size):
    if x.shape[-2:] == size:
        return x
    return F.interpolate(x.unsqueeze(1).float(), size=size, mode="nearest")[:, 0].long()


def _nearest_resize_mask(x, size):
    if x.shape[-2:] == size:
        return x
    return F.interpolate(x.unsqueeze(1).float(), size=size, mode="nearest")[:, 0]


class LossInstrumentHCCEDensePartPose(LossInstrumentHCCEPose):
    """
    Loss for MultiInstrumentHCCEDensePartDPT.

    Instance detection/mask losses are kept at H/4.  Dense part CE and HCCE
    bit regression are evaluated only on valid per-instance foreground pixels.
    Dense part channel order is [wrist, gripper, shaft].
    """

    def forward(self, y_hat, y, epoch=None, img_size=None):
        total_loss, dict_loss = self._compute_detection_and_instance_loss(y_hat, y)

        if "part_mask_logits" in y_hat:
            part_loss, part_metrics = self._compute_dense_part_loss(y_hat, y)
            total_loss = total_loss + self.parser_args.alpha_part * part_loss
            dict_loss["part_ce"] = part_loss
            dict_loss.update(part_metrics)

        if "hcce_logits" in y_hat and "has_cse_inst" in y:
            hcce_loss, hcce_metrics = self._compute_hcce_loss(y_hat, y)
            total_loss = total_loss + self.parser_args.alpha_hcce * hcce_loss
            dict_loss["hcce"] = hcce_loss
            dict_loss.update(hcce_metrics)

        if "action_pred" in y_hat and "has_pose_inst" in y:
            pose_loss, pose_metrics = self._compute_pose_loss(y_hat, y)
            total_loss = total_loss + pose_loss
            dict_loss.update(pose_metrics)

        dict_loss["total"] = total_loss
        return total_loss, dict_loss

    def _compute_detection_and_instance_loss(self, y_hat, y):
        gt_scores = (y["scores"] >= 1).float()
        if "det_source_mask" in y and y["det_source_mask"].any():
            m = y["det_source_mask"]
            det = _neg_loss(y_hat["scores"][m], gt_scores[m])
        elif "det_source_mask" in y:
            det = torch.tensor(0.0, device=y_hat["scores"].device)
        else:
            det = _neg_loss(y_hat["scores"], gt_scores)

        if "rarp50_inst_mask" in y and y["rarp50_inst_mask"].any():
            m = y["rarp50_inst_mask"]
            offset = (y_hat["offset"][m] - y["offset"][m]).abs().sum(-1).mean(0)
        elif "rarp50_inst_mask" in y:
            offset = torch.tensor(0.0, device=y_hat["offset"].device)
        else:
            offset = (y_hat["offset"] - y["offset"]).abs().sum(-1).mean(0)

        pred_inst = torch.sigmoid(y_hat["inst_mask_logits"])
        gt_inst = y["inst_masks"]
        dice = dice_loss(pred_inst, gt_inst)
        bce_mask = F.binary_cross_entropy_with_logits(
            y_hat["inst_mask_logits"],
            gt_inst,
            reduction="mean",
        )

        det = torch.nan_to_num(det, nan=0.0, posinf=0.0, neginf=0.0)
        offset = torch.nan_to_num(offset, nan=0.0, posinf=0.0, neginf=0.0)
        dice = torch.nan_to_num(dice, nan=0.0, posinf=0.0, neginf=0.0)
        bce_mask = torch.nan_to_num(bce_mask, nan=0.0, posinf=0.0, neginf=0.0)

        total = (
            self.parser_args.alpha_det * det
            + self.parser_args.alpha_offset * offset
            + self.parser_args.alpha_dice * dice
            + self.parser_args.alpha_bce_mask * bce_mask
        )
        return total, {
            "det": det,
            "offset": offset,
            "dice": dice,
            "bce_mask": bce_mask,
        }

    def _compute_dense_part_loss(self, y_hat, y):
        pred = y_hat["part_mask_logits"]
        device = pred.device
        part_gt = y.get("part_masks_dense", y["part_masks"]).long()
        part_gt = _nearest_resize_label(part_gt, pred.shape[-2:])
        inst = y.get("inst_masks_dense")
        inst = _nearest_resize_mask(inst.float(), pred.shape[-2:]) if inst is not None else None

        valid = part_gt > 0
        if inst is not None:
            valid = valid & (inst > 0.5)

        # Dataset labels: 1=gripper, 2=wrist, 3=shaft.
        # Dense logits:    0=wrist,   1=gripper, 2=shaft.
        target = torch.full_like(part_gt, -1)
        target[part_gt == 2] = 0
        target[part_gt == 1] = 1
        target[part_gt == 3] = 2
        valid = valid & (target >= 0)

        if not valid.any():
            zero = torch.tensor(0.0, device=device)
            return zero, {
                "part_px": zero,
                "part_acc": zero,
            }

        logits_flat = pred.permute(0, 2, 3, 1)[valid]
        target_flat = target[valid]
        loss = F.cross_entropy(logits_flat, target_flat, reduction="mean")
        acc = (logits_flat.detach().argmax(dim=1) == target_flat).float().mean()
        return torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0), {
            "part_px": valid.sum().float().detach(),
            "part_acc": acc.detach(),
        }

    def _compute_hcce_loss(self, y_hat, y):
        device = y_hat["hcce_logits"].device
        has_cse = y.get("has_cse_dense_inst", y["has_cse_inst"]).bool()
        if not has_cse.any():
            zero = torch.tensor(0.0, device=device)
            return zero, {
                "hcce_px": zero,
                "hcce_bit_acc": zero,
            }

        coord_imgs = y.get("coord_imgs_dense", y["coord_imgs"])
        inst_masks = y.get("inst_masks_dense", y["inst_masks"])
        pred = y_hat["hcce_logits"]
        if coord_imgs.shape[1:3] != pred.shape[-2:]:
            coord_imgs = F.interpolate(
                coord_imgs.permute(0, 3, 1, 2).float(),
                size=pred.shape[-2:],
                mode="nearest",
            ).permute(0, 2, 3, 1)
        if inst_masks.shape[-2:] != pred.shape[-2:]:
            inst_masks = _nearest_resize_mask(inst_masks.float(), pred.shape[-2:])

        use_inst_mask = bool(getattr(self.parser_args, "hcce_use_inst_mask", 1))
        coord_min = float(getattr(self.parser_args, "hcce_coord_min", -1.0))
        coord_max = float(getattr(self.parser_args, "hcce_coord_max", 1.0))

        losses = []
        bit_correct = torch.tensor(0.0, device=device)
        bit_count = torch.tensor(0.0, device=device)
        px_count = torch.tensor(0.0, device=device)
        for k in torch.where(has_cse)[0]:
            gt_xyz = coord_imgs[k, :, :, :3]
            coord_part = coord_imgs[k, :, :, 3].long()
            valid = coord_part > 0
            if use_inst_mask:
                valid = valid & (inst_masks[k] > 0.5)
            if not valid.any():
                continue
            gt_hcce = normalized_xyz_to_hcce(
                gt_xyz,
                iteration=self.parser_args.hcce_bits,
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

        loss = torch.stack(losses).mean() if losses else torch.tensor(0.0, device=device)
        return torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0), {
            "hcce_px": px_count.detach(),
            "hcce_bit_acc": bit_correct / bit_count.clamp_min(1.0),
        }

    @staticmethod
    def add_specific_args(parent_parser):
        parser = Loss_Instrument.add_specific_args(parent_parser)
        parser.add_argument("--alpha_hcce", type=float, default=1.0)
        parser.add_argument("--alpha_action_l1", type=float, default=1.0)
        parser.add_argument("--alpha_wrist_quat_l1", type=float, default=1.0)
        parser.add_argument("--alpha_wrist_trans_l1", type=float, default=10.0)
        parser.add_argument("--hcce_use_inst_mask", type=int, default=1, choices=[0, 1])
        parser.add_argument("--hcce_bits", type=int, default=8)
        parser.add_argument("--hcce_coord_min", type=float, default=-1.0)
        parser.add_argument("--hcce_coord_max", type=float, default=1.0)
        return parser
