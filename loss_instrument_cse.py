import torch
import torch.nn.functional as F
from argparse import ArgumentParser

from loss_instrument import Loss_Instrument


class LossInstrumentCSE(Loss_Instrument):
    """
    Extends Loss_Instrument with a smooth-L1 NOCS regression loss.

    New loss term:
        nocs_reg — smooth-L1 between predicted xyz and GT xyz,
                   computed only for instruments that have a pre-cached coord_img
                   (has_cse_inst == True), and only at pixels where inst_mask > 0.5,
                   the rendered coord image has a valid part label,
                   and coord_img x >= -0.6.

    The existing part CE loss (from part_head) serves as the CE component;
    the instance-conditioned CSE head is supervised with xyz regression.
    """

    def forward(self, y_hat, y, epoch=None, img_size=None):
        # Run the existing losses unchanged
        total_loss, dict_loss = super().forward(y_hat, y, epoch=epoch, img_size=img_size)

        # NOCS regression loss (only when predictions and GT are both present)
        if 'nocs_pred' not in y_hat or 'has_cse_inst' not in y:
            return total_loss, dict_loss

        nocs_loss = self._compute_nocs_loss(y_hat, y)
        nocs_loss = torch.nan_to_num(nocs_loss, nan=0.0, posinf=0.0, neginf=0.0)

        alpha_nocs = self.parser_args.alpha_nocs_reg
        total_loss = total_loss + alpha_nocs * nocs_loss
        dict_loss['nocs_reg'] = nocs_loss
        dict_loss['total']    = total_loss
        return total_loss, dict_loss

    def _compute_nocs_loss(self, y_hat, y):
        """
        Compute per-instrument smooth-L1 loss between predicted and GT xyz,
        averaged over instruments that have CSE coords.

        y_hat['nocs_pred']:   [n_vis_inst, 3, H/4, W/4]
        y['has_cse_inst']:    [n_vis_inst] bool
        y['coord_imgs']:      [n_vis_inst, H/4, W/4, 4]  ch 0-2=xyz, ch 3=part_id
        y['inst_masks']:      [n_vis_inst, H/4, W/4] float32
        """
        has_cse   = y['has_cse_inst']         # [n_vis_inst] bool
        if not has_cse.any():
            return torch.tensor(0.0, device=y_hat['nocs_pred'].device)

        coord_imgs = y['coord_imgs']           # [n_vis_inst, H/4, W/4, 4]
        inst_masks = y['inst_masks']           # [n_vis_inst, H/4, W/4]
        nocs_pred  = y_hat['nocs_pred']        # [n_vis_inst, 3, H/4, W/4]

        total = torch.tensor(0.0, device=nocs_pred.device)
        count = 0
        for k in range(len(has_cse)):
            if not has_cse[k]:
                continue
            gt_xyz   = coord_imgs[k, :, :, :3]            # [H/4, W/4, 3]
            coord_ok = coord_imgs[k, :, :, 3] > 0         # [H/4, W/4] bool
            x_ok     = gt_xyz[:, :, 0] >= -0.6            # [H/4, W/4] bool
            mask_px  = (inst_masks[k] > 0.5) & coord_ok & x_ok  # [H/4, W/4] bool
            if not mask_px.any():
                continue
            pred_xyz = nocs_pred[k].permute(1, 2, 0)      # [H/4, W/4, 3]
            total   += F.smooth_l1_loss(pred_xyz[mask_px], gt_xyz[mask_px])
            count   += 1

        return total / max(count, 1)

    @staticmethod
    def add_specific_args(parent_parser):
        parser = Loss_Instrument.add_specific_args(parent_parser)
        parser.add_argument('--alpha_nocs_reg', type=float, default=1.0,
                            help='Weight for the NOCS smooth-L1 regression loss.')
        return parser
