import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from multi_instrument.dpt_hcce_head import DPTHCCEHead
from multi_instrument.encoder_cse import EncoderCSE
from multi_instrument.hph import HPH
from multi_instrument.instrument_pose_heads import InstrumentActionHead, InstrumentWristPoseHead
from multi_instrument.pixel_decoder import PixelDecoder
from multi_instrument.pos_embed import get_2d_sincos_pos_embed
from utils import unpatch


class MultiInstrumentHCCEDensePartKeypointDPT(nn.Module):
    """
    Dense part/HCCE model with a query-level keypoint branch.

    Instance masks remain query-based. Dense DPT logits are global and are
    gathered per query by batch index:
        part logits: [N, 3, H/2, W/2], order [wrist, gripper, shaft]
        HCCE logits: [N, 24, H/2, W/2]
        optional depth: [N, H/2, W/2], normalized target space
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
        use_depth_head=0,
        num_keypoints=7,
        pose_head_iter=4,
        pose_head_dropout=0.3,
        *args,
        **kwargs,
    ):
        super().__init__()
        if backbone.startswith("dinov3_") or "dinov3" in backbone:
            raise ValueError("MultiInstrumentHCCEDensePartKeypointDPT requires DINOv2 EncoderCSE.")
        self.img_size = int(img_size)
        self.num_parts = 3
        self.mask_dim = int(mask_dim)
        self.hcce_bits = int(hcce_bits)
        self.action_dim = int(action_dim)
        self.use_pose_heads = bool(use_pose_heads)
        self.use_depth_head = bool(use_depth_head)
        self.num_keypoints = int(num_keypoints)

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
        self.keypoint_heatmap_head = nn.Sequential(
            nn.Linear(self.decoder.dim, self.decoder.dim),
            nn.ReLU(),
            nn.Linear(self.decoder.dim, self.num_keypoints * self.mask_dim),
        )

        self.dense_dpt_head = DPTHCCEHead(
            embed_dim=self.encoder.embed_dim,
            feat_dim=hcce_feat_dim,
            out_channels=3 + 3 * self.hcce_bits,
        )
        self.depth_dpt_head = (
            DPTHCCEHead(
                embed_dim=self.encoder.embed_dim,
                feat_dim=hcce_feat_dim,
                out_channels=1,
            )
            if self.use_depth_head
            else None
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
        keypoint_emb = self.keypoint_heatmap_head(y).view(-1, self.num_keypoints, self.mask_dim)
        keypoint_heatmap_logits = self._compute_keypoint_heatmaps(
            keypoint_emb,
            pixel_feat,
            counts,
            values,
        )
        keypoint_heatmap_logits = F.interpolate(
            keypoint_heatmap_logits,
            size=(self.img_size // 2, self.img_size // 2),
            mode="bilinear",
            align_corners=False,
        )
        keypoint_xy, keypoint_scores = self._decode_keypoint_heatmaps(keypoint_heatmap_logits)

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
            "keypoint_xy": keypoint_xy,
            "keypoint_scores": keypoint_scores,
            "keypoint_heatmap_logits": keypoint_heatmap_logits,
            "inst_mask_logits": inst_mask_logits,
            "part_mask_logits_global": part_mask_logits_global,
            "part_mask_logits": part_mask_logits,
            "hcce_logits_global": hcce_global_logits,
            "hcce_logits": hcce_logits,
            "dense_logits": dense_logits,
            "feat": z["feat"],
        }

        if self.depth_dpt_head is not None:
            depth_global = self.depth_dpt_head(
                self.encoder.last_intermediate_feats,
                target_size=self.img_size // 2,
            )
            out["depth_logits_global"] = depth_global
            out["depth_pred"] = depth_global[idx[0], 0]

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
                "keypoint_xy": out["keypoint_xy"][i],
                "keypoint_scores": out["keypoint_scores"][i],
                "keypoint_heatmap_logits": out["keypoint_heatmap_logits"][i],
                "inst_mask_logits": out["inst_mask_logits"][i],
                "part_mask_logits": out["part_mask_logits"][i],
                "hcce_logits": out["hcce_logits"][i],
            }
            if "depth_pred" in out:
                inst["depth_pred"] = out["depth_pred"][i]
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

    def _compute_keypoint_heatmaps(self, keypoint_emb, pixel_feat, counts, values):
        split_sizes = [int(c.item()) for c in counts]
        embs_split = torch.split(keypoint_emb, split_sizes, dim=0)
        all_heatmaps = []
        for emb, batch_idx in zip(embs_split, values):
            pf = pixel_feat[batch_idx]
            all_heatmaps.append(torch.einsum("nkd,dhw->nkhw", emb, pf))
        return torch.cat(all_heatmaps, dim=0)

    def _decode_keypoint_heatmaps(self, heatmap_logits):
        n, k, h, w = heatmap_logits.shape
        flat = heatmap_logits.flatten(2)
        vals, inds = flat.max(dim=-1)
        xs = (inds % w).float()
        ys = torch.div(inds, w, rounding_mode="floor").float()
        scale_x = float(self.img_size) / float(w)
        scale_y = float(self.img_size) / float(h)
        xy = torch.stack([(xs + 0.5) * scale_x, (ys + 0.5) * scale_y], dim=-1)
        return xy, torch.sigmoid(vals)
