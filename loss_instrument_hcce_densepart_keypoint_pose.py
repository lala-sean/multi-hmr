import torch
import torch.nn.functional as F

from loss_instrument_hcce_densepart_pose import (
    LossInstrumentHCCEDensePartPose,
    _nearest_resize_mask,
)


class LossInstrumentHCCEDensePartKeypointPose(LossInstrumentHCCEDensePartPose):
    """
    Densepart/HCCE loss plus query keypoint and optional masked depth loss.
    """

    def forward(self, y_hat, y, epoch=None, img_size=None):
        total_loss, dict_loss = super().forward(y_hat, y, epoch=epoch, img_size=img_size)

        if "keypoint_heatmap_logits" in y_hat and "has_keypoint_inst" in y:
            kp_loss, kp_metrics = self._compute_keypoint_loss(y_hat, y)
            total_loss = total_loss + self.parser_args.alpha_keypoint * kp_loss
            dict_loss["keypoint_heatmap"] = kp_loss
            dict_loss.update(kp_metrics)

        if "depth_pred" in y_hat and "has_depth_inst" in y:
            depth_loss, depth_metrics = self._compute_depth_loss(y_hat, y)
            total_loss = total_loss + self.parser_args.alpha_depth * depth_loss
            dict_loss["depth_l1"] = depth_loss
            dict_loss.update(depth_metrics)

        dict_loss["total"] = total_loss
        return total_loss, dict_loss

    def _compute_keypoint_loss(self, y_hat, y):
        pred = y_hat["keypoint_heatmap_logits"]
        device = pred.device
        target_xy = y["keypoint_gt_inst"].float().to(device)
        if target_xy.ndim == 2:
            target_xy = target_xy[:, None, :]

        if "keypoint_valid_inst" in y:
            valid = y["keypoint_valid_inst"].bool().to(device)
        else:
            has_keypoint = y["has_keypoint_inst"].bool().to(device)
            valid = has_keypoint[:, None].expand(target_xy.shape[:2])

        k_pred = pred.shape[1]
        k_tgt = target_xy.shape[1]
        k = min(k_pred, k_tgt)
        pred = pred[:, :k]
        target_xy = target_xy[:, :k]
        valid = valid[:, :k]

        if not valid.any():
            zero = torch.tensor(0.0, device=device)
            return zero, {"keypoint_visible": zero, "keypoint_err_px": zero}

        n, _, h, w = pred.shape
        img_size = float(getattr(self.parser_args, "img_size", 630))
        sigma = float(getattr(self.parser_args, "keypoint_heatmap_sigma", 2.0))
        yy = torch.arange(h, device=device, dtype=torch.float32).view(1, 1, h, 1)
        xx = torch.arange(w, device=device, dtype=torch.float32).view(1, 1, 1, w)
        xh = target_xy[..., 0].view(n, k, 1, 1) / img_size * float(w) - 0.5
        yh = target_xy[..., 1].view(n, k, 1, 1) / img_size * float(h) - 0.5
        target = torch.exp(-((xx - xh) ** 2 + (yy - yh) ** 2) / (2.0 * sigma * sigma))
        target = target * valid[:, :, None, None].float()

        loss = F.mse_loss(torch.sigmoid(pred)[valid], target[valid], reduction="mean")

        flat = pred.detach().flatten(2)
        inds = flat.argmax(dim=-1)
        px = ((inds % w).float() + 0.5) * (img_size / float(w))
        py = (torch.div(inds, w, rounding_mode="floor").float() + 0.5) * (img_size / float(h))
        pred_xy = torch.stack([px, py], dim=-1)
        err = torch.linalg.norm(pred_xy[valid] - target_xy[valid], dim=-1).mean()
        return torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0), {
            "keypoint_visible": valid.float().sum().detach(),
            "keypoint_err_px": err.detach(),
        }

    def _compute_depth_loss(self, y_hat, y):
        device = y_hat["depth_pred"].device
        has_depth = y["has_depth_inst"].bool()
        if not has_depth.any():
            zero = torch.tensor(0.0, device=device)
            return zero, {"depth_px": zero}

        pred = y_hat["depth_pred"]
        depth = y["depth_maps_dense_inst"].float().to(device)
        valid = y["depth_valid_dense_inst"].float().to(device)
        if depth.shape[-2:] != pred.shape[-2:]:
            depth = _nearest_resize_mask(depth, pred.shape[-2:])
            valid = _nearest_resize_mask(valid, pred.shape[-2:])
        if "inst_masks_dense" in y:
            inst = _nearest_resize_mask(y["inst_masks_dense"].float().to(device), pred.shape[-2:])
            valid = valid * (inst > 0.5).float()

        depth_scale = float(getattr(self.parser_args, "depth_loss_scale", 250.0))
        losses = []
        px = torch.tensor(0.0, device=device)
        for k in torch.where(has_depth)[0]:
            mask = (valid[k] > 0.5) & torch.isfinite(depth[k])
            if not mask.any():
                continue
            target = depth[k][mask] / depth_scale
            losses.append(F.smooth_l1_loss(pred[k][mask], target, reduction="mean"))
            px = px + mask.sum().float()
        loss = torch.stack(losses).mean() if losses else torch.tensor(0.0, device=device)
        return torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0), {
            "depth_px": px.detach(),
        }

    @staticmethod
    def add_specific_args(parent_parser):
        parser = LossInstrumentHCCEDensePartPose.add_specific_args(parent_parser)
        parser.add_argument("--alpha_keypoint", type=float, default=1.0)
        parser.add_argument("--alpha_depth", type=float, default=1.0)
        parser.add_argument("--depth_loss_scale", type=float, default=1.0)
        parser.add_argument("--keypoint_heatmap_sigma", type=float, default=2.0)
        return parser
