import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from multi_instrument.encoder_cse import EncoderCSE
from multi_instrument.hph import HPH
from multi_instrument.pixel_decoder import PixelDecoder
from multi_instrument.pos_embed import get_2d_sincos_pos_embed
from multi_instrument.dpt_hcce_head import DPTHCCEHead
from multi_instrument.instrument_pose_heads import (
    InstrumentActionHead,
    InstrumentWristPoseHead,
)
from utils import unpatch


class MultiInstrumentHCCEDensePartDPT(nn.Module):
    """
    Variant of MultiInstrumentHCCEPoseDPT with dense DPT part segmentation.

    The existing query-based instance mask path is kept unchanged.  The DPT
    decoder emits three dense part logits followed by HCCE logits at H/2:

        dense logits: [B, 3 + 3 * hcce_bits, H/2, W/2]
        part order:   [wrist, gripper, shaft]
    """

    def __init__(
        self,
        img_size=630,
        backbone="dinov2_vits14",
        pretrained_backbone=False,
        xat_dim=512,
        xat_depth=4,
        xat_heads=16,
        xat_dim_head=32,
        xat_mlp_dim=2048,
        xat_dropout=0.0,
        mask_dim=256,
        num_parts=4,
        hcce_feat_dim=256,
        hcce_bits=8,
        action_dim=3,
        use_pose_heads=0,
        pose_head_iter=4,
        pose_head_dropout=0.3,
        *args,
        **kwargs,
    ):
        super().__init__()
        if backbone.startswith("dinov3_") or "dinov3" in backbone:
            raise ValueError("MultiInstrumentHCCEDensePartDPT requires DINOv2 EncoderCSE.")
        self.img_size = img_size
        self.num_parts = 3
        self.mask_dim = mask_dim
        self.hcce_bits = hcce_bits
        self.action_dim = action_dim
        self.use_pose_heads = bool(use_pose_heads)

        self.encoder = EncoderCSE(backbone, pretrained=pretrained_backbone)
        self.patch_size = self.encoder.patch_size
        if self.img_size % self.patch_size != 0:
            raise ValueError("img_size must be divisible by encoder patch size")

        grid_size = self.img_size // self.patch_size
        dec_pos_emb = get_2d_sincos_pos_embed(embed_dim=xat_dim, grid_size=grid_size)
        self.register_buffer("dec_pos_emb", torch.from_numpy(dec_pos_emb).float())
        self.dec_to_token = nn.Linear(self.encoder.embed_dim, xat_dim)
        self.decoder = HPH(
            dim=xat_dim,
            depth=xat_depth,
            heads=xat_heads,
            dim_head=xat_dim_head,
            mlp_dim=xat_mlp_dim,
            dropout=xat_dropout,
        )
        self.mlp_offset = nn.Sequential(
            nn.Linear(self.decoder.dim, self.decoder.dim),
            nn.ReLU(),
            nn.Linear(self.decoder.dim, 2),
        )
        self.pixel_decoder = PixelDecoder(self.encoder.embed_dim, mask_dim)
        self.mask_head = nn.Linear(xat_dim, mask_dim)

        self.dense_dpt_head = DPTHCCEHead(
            embed_dim=self.encoder.embed_dim,
            feat_dim=hcce_feat_dim,
            out_channels=3 + 3 * hcce_bits,
        )
        if self.use_pose_heads:
            self.action_head = InstrumentActionHead(
                feature_dim=xat_dim,
                n_actions=action_dim,
                n_iter=pose_head_iter,
                dropout=pose_head_dropout,
            )
            self.wrist_pose_head = InstrumentWristPoseHead(
                feature_dim=xat_dim,
                n_iter=pose_head_iter,
                dropout=pose_head_dropout,
            )
        else:
            self.action_head = None
            self.wrist_pose_head = None

    def forward(
        self,
        x,
        idx=None,
        is_training=False,
        det_thresh=0.3,
        nms_kernel_size=3,
        *args,
        **kwargs,
    ):
        z = self.encoder(x)
        if not is_training:
            if nms_kernel_size > 1:
                pad = (
                    (nms_kernel_size - 1) // 2
                    if nms_kernel_size not in [2, 4]
                    else nms_kernel_size // 2
                )
                scores_max = F.max_pool2d(
                    z["scores"].unsqueeze(1),
                    (nms_kernel_size, nms_kernel_size),
                    stride=1,
                    padding=pad,
                )[:, 0]
                z["scores"] = z["scores"] * (scores_max == z["scores"]).float()
            idx = torch.where(z["scores"] >= det_thresh) if idx is None else idx
            if len(idx[0]) == 0:
                return []

        y, counts, values = self._decode_instance_queries(z, idx, x.device)
        offset = self.mlp_offset(y)
        loc_coarse = torch.stack([idx[2], idx[1]], dim=1)
        loc = (loc_coarse + 0.5 + offset) * self.encoder.patch_size

        pixel_feat = self.pixel_decoder(z["feat"], self.img_size)
        mask_emb = self.mask_head(y)
        inst_mask_logits = self._compute_masks(mask_emb, pixel_feat, counts, values)

        dense_logits = self.dense_dpt_head(
            self.encoder.last_intermediate_feats,
            target_size=self.img_size // 2,
        )
        part_mask_logits_global = dense_logits[:, :3]
        hcce_global_logits = dense_logits[:, 3:]
        part_mask_logits = part_mask_logits_global[idx[0]]
        hcce_logits = hcce_global_logits[idx[0]]

        out = {
            "scores": z["scores"],
            "scores_logits": z["scores_logits"],
            "loc": loc,
            "offset": offset,
            "inst_mask_logits": inst_mask_logits,
            "part_mask_logits_global": part_mask_logits_global,
            "part_mask_logits": part_mask_logits,
            "hcce_logits_global": hcce_global_logits,
            "hcce_logits": hcce_logits,
            "dense_logits": dense_logits,
            "feat": z["feat"],
        }
        if self.use_pose_heads:
            action_pred = self.action_head(y)
            wrist_quat_pred, wrist_trans_pred = self.wrist_pose_head(y)
            out["action_pred"] = action_pred
            out["wrist_quat_pred"] = wrist_quat_pred
            out["wrist_trans_pred"] = wrist_trans_pred
        if is_training:
            return out

        instruments = []
        for i in range(idx[0].shape[0]):
            inst = {
                "batch_idx": idx[0][i].item(),
                "loc": out["loc"][i],
                "inst_mask_logits": out["inst_mask_logits"][i],
                "part_mask_logits": out["part_mask_logits"][i],
                "hcce_logits": out["hcce_logits"][i],
            }
            if self.use_pose_heads:
                inst["action_pred"] = out["action_pred"][i]
                inst["wrist_quat_pred"] = out["wrist_quat_pred"][i]
                inst["wrist_trans_pred"] = out["wrist_trans_pred"][i]
            instruments.append(inst)
        return instruments

    def _decode_instance_queries(self, z, idx, device):
        dec_pos_emb = unpatch(
            self.dec_pos_emb.unsqueeze(0),
            patch_size=1,
            c=self.decoder.dim,
            img_size=int(np.sqrt(self.dec_pos_emb.shape[0])),
        ).permute(0, 2, 3, 1)
        dec_emb = self.dec_to_token(z["feat"]) + dec_pos_emb
        queries = dec_emb[idx[0], idx[1], idx[2]]
        values, counts = torch.unique(idx[0], sorted=True, return_counts=True)
        max_count = int(max(counts).item())
        queries = torch.stack(
            [
                F.pad(q, (0, 0, 0, max_count - int(counts[i].item())), mode="constant", value=0)
                for i, q in enumerate(torch.split(queries, tuple(counts), dim=0))
            ]
        )
        mask = torch.cat(
            [
                F.pad(
                    torch.ones(1, int(c.item()), 1, device=device),
                    (0, 0, 0, max_count - int(c.item())),
                    mode="constant",
                    value=0,
                )
                for c in counts
            ],
            dim=0,
        )[..., 0]
        context = dec_emb[values].flatten(1, 2)
        y = self.decoder(x=queries, context=context, mask=mask)
        y = torch.cat([y[i, : int(c.item())] for i, c in enumerate(counts)], dim=0)
        return y, counts, values

    def _compute_masks(self, mask_emb, pixel_feat, counts, values):
        split_sizes = [int(c.item()) for c in counts]
        embs_split = torch.split(mask_emb, split_sizes, dim=0)
        all_masks = []
        for emb, batch_idx in zip(embs_split, values):
            pf = pixel_feat[batch_idx]
            all_masks.append(torch.einsum("nd,dhw->nhw", emb, pf))
        return torch.cat(all_masks, dim=0)
