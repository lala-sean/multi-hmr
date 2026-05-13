import torch
import torch.nn.functional as F

from loss_instrument import Loss_Instrument
from loss_instrument_cse import LossInstrumentCSE


class LossInstrumentCSEVisibleMetrics(LossInstrumentCSE):
    """
    CSE/NOCS loss variant for debugging actual visible-coordinate supervision.

    Differences from LossInstrumentCSE:
      - no x >= -0.6 cutoff by default;
      - logs supervised/visible pixel counts per rendered CSE part;
      - logs pixel MAE globally and per part.
    """

    PARTS = {
        1: "shaft",
        2: "wrist",
        3: "l_gripper",
        4: "r_gripper",
    }

    def forward(self, y_hat, y, epoch=None, img_size=None):
        total_loss, dict_loss = Loss_Instrument.forward(
            self, y_hat, y, epoch=epoch, img_size=img_size
        )

        if "nocs_pred" not in y_hat or "has_cse_inst" not in y:
            return total_loss, dict_loss

        nocs_loss, nocs_metrics = self._compute_nocs_loss_and_metrics(y_hat, y)
        nocs_loss = torch.nan_to_num(nocs_loss, nan=0.0, posinf=0.0, neginf=0.0)

        total_loss = total_loss + self.parser_args.alpha_nocs_reg * nocs_loss
        dict_loss["nocs_reg"] = nocs_loss
        dict_loss.update(nocs_metrics)
        dict_loss["total"] = total_loss
        return total_loss, dict_loss

    def _compute_nocs_loss(self, y_hat, y):
        nocs_loss, _ = self._compute_nocs_loss_and_metrics(y_hat, y)
        return nocs_loss

    def _compute_nocs_loss_and_metrics(self, y_hat, y):
        device = y_hat["nocs_pred"].device
        has_cse = y["has_cse_inst"]
        if not has_cse.any():
            return torch.tensor(0.0, device=device), self._empty_metrics(device)

        coord_imgs = y["coord_imgs"]
        inst_masks = y["inst_masks"]
        nocs_pred = y_hat["nocs_pred"]

        use_inst_mask = bool(getattr(self.parser_args, "nocs_use_inst_mask", 1))
        nocs_min_x = getattr(self.parser_args, "nocs_min_x", None)

        total_loss = torch.tensor(0.0, device=device)
        inst_count = 0

        total_abs = torch.tensor(0.0, device=device)
        total_px = torch.tensor(0.0, device=device)
        total_coord_px = torch.tensor(0.0, device=device)

        part_abs = {
            pid: torch.tensor(0.0, device=device)
            for pid in self.PARTS
        }
        part_sup_px = {
            pid: torch.tensor(0.0, device=device)
            for pid in self.PARTS
        }
        part_coord_px = {
            pid: torch.tensor(0.0, device=device)
            for pid in self.PARTS
        }

        for k in range(len(has_cse)):
            if not bool(has_cse[k].item()):
                continue

            gt_xyz = coord_imgs[k, :, :, :3]
            coord_part = coord_imgs[k, :, :, 3].long()
            coord_ok = coord_part > 0
            if nocs_min_x is not None:
                coord_ok = coord_ok & (gt_xyz[:, :, 0] >= float(nocs_min_x))

            if use_inst_mask:
                mask_px = coord_ok & (inst_masks[k] > 0.5)
            else:
                mask_px = coord_ok

            total_coord_px = total_coord_px + coord_ok.sum().float()

            for pid in self.PARTS:
                part_coord_mask = coord_part == pid
                part_coord_px[pid] = part_coord_px[pid] + part_coord_mask.sum().float()

            if not mask_px.any():
                continue

            pred_xyz = nocs_pred[k].permute(1, 2, 0)
            total_loss = total_loss + F.smooth_l1_loss(
                pred_xyz[mask_px],
                gt_xyz[mask_px],
            )
            inst_count += 1

            abs_err = (pred_xyz - gt_xyz).abs().mean(dim=-1)
            total_abs = total_abs + abs_err[mask_px].sum()
            total_px = total_px + mask_px.sum().float()

            for pid in self.PARTS:
                part_mask = mask_px & (coord_part == pid)
                part_sup_px[pid] = part_sup_px[pid] + part_mask.sum().float()
                if part_mask.any():
                    part_abs[pid] = part_abs[pid] + abs_err[part_mask].sum()

        nocs_loss = total_loss / max(inst_count, 1)
        metrics = {
            "nocs_mae": total_abs / total_px.clamp_min(1.0),
            "nocs_supervised_px": total_px.detach(),
            "nocs_coord_px": total_coord_px.detach(),
            "nocs_coord_inst": torch.tensor(float(has_cse.sum().item()), device=device),
            "nocs_loss_inst": torch.tensor(float(inst_count), device=device),
        }

        for pid, name in self.PARTS.items():
            coord_px = part_coord_px[pid]
            sup_px = part_sup_px[pid]
            metrics[f"nocs_mae_{name}"] = part_abs[pid] / sup_px.clamp_min(1.0)
            metrics[f"nocs_supervised_px_{name}"] = sup_px.detach()
            metrics[f"nocs_coord_px_{name}"] = coord_px.detach()
            metrics[f"nocs_supervised_ratio_{name}"] = (
                sup_px / coord_px.clamp_min(1.0)
            ).detach()

        return nocs_loss, metrics

    def _empty_metrics(self, device):
        metrics = {
            "nocs_mae": torch.tensor(0.0, device=device),
            "nocs_supervised_px": torch.tensor(0.0, device=device),
            "nocs_coord_px": torch.tensor(0.0, device=device),
            "nocs_coord_inst": torch.tensor(0.0, device=device),
            "nocs_loss_inst": torch.tensor(0.0, device=device),
        }
        for name in self.PARTS.values():
            metrics[f"nocs_mae_{name}"] = torch.tensor(0.0, device=device)
            metrics[f"nocs_supervised_px_{name}"] = torch.tensor(0.0, device=device)
            metrics[f"nocs_coord_px_{name}"] = torch.tensor(0.0, device=device)
            metrics[f"nocs_supervised_ratio_{name}"] = torch.tensor(0.0, device=device)
        return metrics

    @staticmethod
    def add_specific_args(parent_parser):
        parser = LossInstrumentCSE.add_specific_args(parent_parser)
        parser.add_argument(
            "--nocs_use_inst_mask",
            type=int,
            default=1,
            choices=[0, 1],
            help="Intersect rendered coord pixels with dataset instance masks.",
        )
        parser.add_argument(
            "--nocs_min_x",
            type=float,
            default=None,
            help="Optional canonical x cutoff. Default None removes the old x>=-0.6 filter.",
        )
        return parser
