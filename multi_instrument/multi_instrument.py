import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

from multi_instrument.hph import HPH
from multi_instrument.pos_embed import get_2d_sincos_pos_embed
from multi_instrument.encoder import Encoder
from multi_instrument.encoder_dinov3 import EncoderDINOv3
from multi_instrument.pixel_decoder import PixelDecoder

from utils import unpatch
from utils.image import IMG_NORM_MEAN, IMG_NORM_STD


class Multi_Instrument(nn.Module):
    """
    DINOv2 backbone + patch-level detection (wrist center) + HPH decoder
    + instance segmentation head + part-level segmentation head.

    Stage 1: detect and segment multiple surgical instruments per image.
    Each instrument is localized by its wrist center point.
    """

    def __init__(self,
                 img_size=896,
                 # Backbone
                 backbone='dinov2_vits14',
                 pretrained_backbone=False,
                 backbone_weights=None,
                 # HPH
                 xat_dim=512,
                 xat_depth=4,
                 xat_heads=16,
                 xat_dim_head=32,
                 xat_mlp_dim=2048,
                 xat_dropout=0.0,
                 # Segmentation
                 mask_dim=256,
                 num_parts=4,  # 0=background, 1=wrist, 2=shaft, 3=gripper
                 *args, **kwargs):
        super().__init__()

        # Encoder (DINOv2 or DINOv3 + patch-level detection)
        self.img_size = img_size
        if backbone.startswith('dinov3_') or 'dinov3' in backbone:
            self.encoder = EncoderDINOv3(backbone, pretrained=pretrained_backbone)
        else:
            self.encoder = Encoder(backbone, pretrained=pretrained_backbone)
        assert self.img_size % self.encoder.patch_size == 0, "Invalid img size"
        self.patch_size = self.encoder.patch_size

        # Positional embedding for decoder tokens
        grid_size = self.img_size // self.patch_size
        dec_pos_emb = get_2d_sincos_pos_embed(embed_dim=xat_dim, grid_size=grid_size)
        self.register_buffer('dec_pos_emb', torch.from_numpy(dec_pos_emb).float())
        self.dec_to_token = nn.Linear(self.encoder.embed_dim, xat_dim)

        # HPH decoder (cross-attention transformer)
        self.decoder = HPH(dim=xat_dim, depth=xat_depth, heads=xat_heads,
                           dim_head=xat_dim_head, mlp_dim=xat_mlp_dim, dropout=xat_dropout)

        # 2D offset head (sub-patch wrist center refinement)
        self.mlp_offset = nn.Sequential(
            nn.Linear(self.decoder.dim, self.decoder.dim),
            nn.ReLU(),
            nn.Linear(self.decoder.dim, 2)
        )

        # Pixel decoder (upsample backbone features for mask prediction)
        self.mask_dim = mask_dim
        self.num_parts = num_parts
        self.pixel_decoder = PixelDecoder(self.encoder.embed_dim, mask_dim)

        # Instance segmentation head: per-instrument binary mask
        self.mask_head = nn.Linear(xat_dim, mask_dim)

        # Part segmentation head: per-instrument part masks (wrist/shaft/body)
        self.part_head = nn.Linear(xat_dim, mask_dim * num_parts)

    def forward(self, x,
                idx=None,
                is_training=False,
                det_thresh=0.3, nms_kernel_size=3,
                visualize_pca=False,
                *args, **kwargs):
        out = {}

        # 1. Image encoder
        z = self.encoder(x)

        # Debug: PCA on patch features to visualize attention patterns (first image in batch).
        # Saves side-by-side [original | PCA] to pca_debug.png (relative to cwd)
        if visualize_pca:
            from torchvision.utils import save_image
            _feat = z['feat']                              # [bs, H_p, W_p, emb]
            _, H_p, W_p, _ = _feat.shape

            # --- Detect content bounding box (crop black padding at patch level) ---
            # Black pixels after ImageNet norm ≈ (-2.118, -2.036, -1.804)
            _black = torch.tensor([-2.118, -2.036, -1.804],
                                  device=x.device).view(1, 3, 1, 1)
            _xp = F.avg_pool2d(x[0:1].float(), kernel_size=self.patch_size,
                               stride=self.patch_size)          # [1, 3, H_p, W_p]
            _is_content = (_xp - _black).abs().mean(dim=1)[0] > 0.15  # [H_p, W_p]
            _rows = _is_content.any(dim=1)
            _cols = _is_content.any(dim=0)
            r0 = int(_rows.nonzero(as_tuple=False)[0])
            r1 = int(_rows.nonzero(as_tuple=False)[-1]) + 1
            c0 = int(_cols.nonzero(as_tuple=False)[0])
            c1 = int(_cols.nonzero(as_tuple=False)[-1]) + 1

            # --- Crop and denormalize original image ---
            _mean = torch.tensor(IMG_NORM_MEAN, device=x.device).view(1, 3, 1, 1)
            _std  = torch.tensor(IMG_NORM_STD,  device=x.device).view(1, 3, 1, 1)
            _img_crop = (x[0:1].float() * _std + _mean).clamp(0, 1)
            # crop to content region (pixel coords = patch coords * patch_size)
            _img_crop = _img_crop[:, :,
                                  r0 * self.patch_size : r1 * self.patch_size,
                                  c0 * self.patch_size : c1 * self.patch_size]
            # [1, 3, H_px, W_px]

            # --- PCA on content patches (.cpu() avoids fp16 QR decomp error under AMP) ---
            _f = _feat[0, r0:r1, c0:c1, :].reshape(-1, _feat.shape[-1]).detach().cpu().float()
            _f_c = _f - _f.mean(0, keepdim=True)
            _, _S, _V = torch.pca_lowrank(_f_c, q=3, niter=4)
            _proj = _f_c @ _V                                    # [N_crop, 3]
            _proj = _proj / (_proj.std(0, keepdim=True) + 1e-6)  # whiten
            _rgb = torch.sigmoid(_proj * 2.0)                    # vibrant rainbow colors
            H_crop, W_crop = r1 - r0, c1 - c0
            _rgb = _rgb.reshape(1, H_crop, W_crop, 3).permute(0, 3, 1, 2)  # [1, 3, H_p, W_p]

            # --- Upsample PCA map to match original image resolution ---
            _H_px, _W_px = _img_crop.shape[2], _img_crop.shape[3]
            _rgb_up = F.interpolate(_rgb, size=(_H_px, _W_px), mode='bilinear',
                                    align_corners=False)         # [1, 3, H_px, W_px]

            # --- Save side-by-side ---
            _vis = torch.cat([_img_crop.cpu(), _rgb_up], dim=3)  # [1, 3, H_px, 2*W_px]
            save_image(_vis, 'pca_debug.png')
            out['pca_feat'] = _rgb_up                            # [1, 3, H_px, W_px]

        # 2. NMS + threshold for detection
        if not is_training:
            if nms_kernel_size > 1:
                pad = (nms_kernel_size - 1) // 2 if nms_kernel_size not in [2, 4] else nms_kernel_size // 2
                scores_max = nn.functional.max_pool2d(
                    z['scores'].unsqueeze(1),
                    (nms_kernel_size, nms_kernel_size), stride=1, padding=pad
                )[:, 0]
                z['scores'] = z['scores'] * (scores_max == z['scores']).float()
            idx = torch.where(z['scores'] >= det_thresh) if idx is None else idx
            if len(idx[0]) == 0:
                return {}, []

        # 3. Decoder token embeddings
        dec_pos_emb = unpatch(
            self.dec_pos_emb.unsqueeze(0), patch_size=1,
            c=self.decoder.dim,
            img_size=int(np.sqrt(self.dec_pos_emb.shape[0]))
        ).permute(0, 2, 3, 1)  # [1, sqrt(np), sqrt(np), d]
        dec_emb = self.dec_to_token(z['feat']) + dec_pos_emb  # [bs, np, np, D]

        # 4. Extract queries at detected locations
        queries = dec_emb[idx[0], idx[1], idx[2]]
        values, counts = torch.unique(idx[0], sorted=True, return_counts=True)
        queries = torch.stack([
            F.pad(q, (0, 0, 0, max(counts) - counts[i]), mode='constant', value=0)
            for i, q in enumerate(torch.split(queries, tuple(counts), dim=0))
        ])  # [bs, max(counts), emb]
        mask = torch.cat([
            F.pad(torch.ones(1, c, 1).to(x.device), (0, 0, 0, max(counts) - c), mode='constant', value=0)
            for c in tuple(counts)
        ], dim=0)[..., 0]  # [bs, max(counts)]

        # 5. Context = all patch features
        context = dec_emb[values].flatten(1, 2)  # [bs', np*np, D]

        # 6. HPH decoder
        y = self.decoder(x=queries, context=context, mask=mask)  # [bs, max(counts), emb]
        y = torch.cat([y[i, :c] for i, c in enumerate(tuple(counts))], dim=0)  # [n_inst, emb]

        # 7. Wrist center 2D location
        offset = self.mlp_offset(y)
        loc_coarse = torch.stack([idx[2], idx[1]], dim=1)  # swap x,y
        loc = (loc_coarse + 0.5 + offset) * self.encoder.patch_size

        # 8. Pixel features (shared for both heads)
        pixel_feat = self.pixel_decoder(z['feat'], self.img_size)  # [bs, mask_dim, H/4, W/4]

        # 9. Instance mask prediction
        mask_emb = self.mask_head(y)  # [n_inst, mask_dim]
        inst_mask_logits = self._compute_masks(mask_emb, pixel_feat, idx, counts, values)

        # 10. Part mask prediction
        part_embs = self.part_head(y)  # [n_inst, mask_dim * num_parts]
        part_embs = part_embs.reshape(-1, self.num_parts, self.mask_dim)  # [n_inst, num_parts, mask_dim]
        part_mask_logits = self._compute_part_masks(part_embs, pixel_feat, idx, counts, values)

        # Output
        out = {
            'scores': z['scores'],
            'scores_logits': z['scores_logits'],
            'loc': loc,
            'offset': offset,
            'inst_mask_logits': inst_mask_logits,   # list of [n_i, H/4, W/4] per batch
            'part_mask_logits': part_mask_logits,    # list of [n_i, num_parts, H/4, W/4] per batch
            'feat': z['feat'],
        }

        if is_training:
            return out
        else:
            instruments = []
            for i in range(idx[0].shape[0]):
                inst = {
                    'batch_idx': idx[0][i].item(),
                    'loc': out['loc'][i],
                    'inst_mask_logits': out['inst_mask_logits'][i],   # [H/4, W/4]
                    'part_mask_logits': out['part_mask_logits'][i],   # [num_parts, H/4, W/4]
                }
                instruments.append(inst)
            return instruments

    def _compute_masks(self, mask_emb, pixel_feat, idx, counts, values):
        """
        Compute per-instrument instance masks via dot product.
        Args:
            mask_emb: [n_inst, mask_dim]
            pixel_feat: [bs, mask_dim, H/4, W/4]
            idx: detection indices
            counts: number of instruments per image
            values: batch indices
        Returns:
            list of mask logits [n_i, H/4, W/4] per batch image,
            or concatenated [n_inst, H/4, W/4] for training
        """
        # Split mask_emb by batch
        embs_split = torch.split(mask_emb, tuple(counts), dim=0)
        all_masks = []
        for i, (emb, batch_idx) in enumerate(zip(embs_split, values)):
            pf = pixel_feat[batch_idx]  # [mask_dim, H/4, W/4]
            # einsum: [n_i, d] x [d, h, w] -> [n_i, h, w]
            m = torch.einsum('nd,dhw->nhw', emb, pf)
            all_masks.append(m)

        return torch.cat(all_masks, dim=0)  # [n_inst, H/4, W/4]

    def _compute_part_masks(self, part_embs, pixel_feat, idx, counts, values):
        """
        Compute per-instrument part masks via dot product.
        Args:
            part_embs: [n_inst, num_parts, mask_dim]
            pixel_feat: [bs, mask_dim, H/4, W/4]
        Returns:
            concatenated [n_inst, num_parts, H/4, W/4] for training
        """
        embs_split = torch.split(part_embs, tuple(counts), dim=0)
        all_masks = []
        for i, (emb, batch_idx) in enumerate(zip(embs_split, values)):
            pf = pixel_feat[batch_idx]  # [mask_dim, H/4, W/4]
            # einsum: [n_i, p, d] x [d, h, w] -> [n_i, p, h, w]
            m = torch.einsum('npd,dhw->nphw', emb, pf)
            all_masks.append(m)

        return torch.cat(all_masks, dim=0)  # [n_inst, num_parts, H/4, W/4]
