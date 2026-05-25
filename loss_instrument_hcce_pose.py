import torch
import torch.nn.functional as F

from loss_instrument import Loss_Instrument
from multi_instrument.hcce_codec import normalized_xyz_to_hcce


class LossInstrumentHCCEPose(Loss_Instrument):
    """
    New loss for front-HCCE CSE plus direct instrument pose supervision.
    """

    def forward(self, y_hat, y, epoch=None, img_size=None):
        total_loss, dict_loss = Loss_Instrument.forward(
            self, y_hat, y, epoch=epoch, img_size=img_size
        )

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

    def _compute_hcce_loss(self, y_hat, y):
        device = y_hat["hcce_logits"].device
        has_cse = y["has_cse_inst"].bool()
        if not has_cse.any():
            return torch.tensor(0.0, device=device), {
                "hcce_px": torch.tensor(0.0, device=device),
                "hcce_bit_acc": torch.tensor(0.0, device=device),
            }

        coord_imgs = y["coord_imgs"]
        inst_masks = y["inst_masks"]
        pred = y_hat["hcce_logits"]
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

        if not losses:
            loss = torch.tensor(0.0, device=device)
        else:
            loss = torch.stack(losses).mean()
        metrics = {
            "hcce_px": px_count.detach(),
            "hcce_bit_acc": bit_correct / bit_count.clamp_min(1.0),
        }
        return loss, metrics

    def _compute_pose_loss(self, y_hat, y):
        device = y_hat["action_pred"].device
        has_pose = y["has_pose_inst"].bool()
        if not has_pose.any():
            zero = torch.tensor(0.0, device=device)
            return zero, {
                "action_l1": zero,
                "wrist_quat_l1": zero,
                "wrist_trans_l1": zero,
            }

        pred_action = y_hat["action_pred"][has_pose]
        gt_action = y["action_gt_inst"][has_pose]
        action_l1 = F.l1_loss(pred_action, gt_action)

        pred_quat = F.normalize(y_hat["wrist_quat_pred"][has_pose], p=2, dim=1)
        gt_quat = F.normalize(y["wrist_quat_gt_inst"][has_pose], p=2, dim=1)
        sign = torch.where((pred_quat * gt_quat).sum(dim=1, keepdim=True) < 0.0, -1.0, 1.0)
        gt_quat = gt_quat * sign
        wrist_quat_l1 = F.l1_loss(pred_quat, gt_quat)

        wrist_trans_l1 = F.l1_loss(
            y_hat["wrist_trans_pred"][has_pose],
            y["wrist_trans_gt_inst"][has_pose],
        )
        total = (
            self.parser_args.alpha_action_l1 * action_l1
            + self.parser_args.alpha_wrist_quat_l1 * wrist_quat_l1
            + self.parser_args.alpha_wrist_trans_l1 * wrist_trans_l1
        )
        return total, {
            "action_l1": action_l1,
            "wrist_quat_l1": wrist_quat_l1,
            "wrist_trans_l1": wrist_trans_l1,
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
