import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

from multi_instrument.multi_instrument import Multi_Instrument
from utils import unpatch
from utils.image import IMG_NORM_MEAN, IMG_NORM_STD


class MultiInstrumentCSE(Multi_Instrument):
    """
    Multi_Instrument with an instance-conditioned CSE/NOCS head.

    This follows the same design pattern as the existing instance/part mask
    heads:
      DINO features -> HPH instance queries -> per-instance head embeddings
      pixel decoder features -> dense spatial support
      einsum(query head, pixel feature) -> per-instance dense prediction

    During training, adds:
        nocs_pred: [n_inst, 3, H/4, W/4]

    This is intentionally not a standalone DPT branch. The NOCS prediction is
    conditioned on the HPH output for each detected/GT instrument.
    """

    def __init__(self, nocs_feat_dim: int = 256, *args, **kwargs):
        # nocs_feat_dim is kept only for old CLI/checkpoint compatibility.
        super().__init__(*args, **kwargs)

        # Per-instrument CSE head: HPH query -> 3 xyz embeddings over pixel_feat.
        self.cse_head = nn.Linear(self.decoder.dim, self.mask_dim * 3)

    def forward(self, x,
                idx=None,
                is_training=False,
                det_thresh=0.3, nms_kernel_size=3,
                visualize_pca=False,
                *args, **kwargs):
        out = {}

        # 1. Image encoder
        z = self.encoder(x)

        # Keep the same PCA debug path as Multi_Instrument.
        if visualize_pca:
            from torchvision.utils import save_image
            _feat = z['feat']
            _, H_p, W_p, _ = _feat.shape

            _black = torch.tensor([-2.118, -2.036, -1.804],
                                  device=x.device).view(1, 3, 1, 1)
            _xp = F.avg_pool2d(x[0:1].float(), kernel_size=self.patch_size,
                               stride=self.patch_size)
            _is_content = (_xp - _black).abs().mean(dim=1)[0] > 0.15
            _rows = _is_content.any(dim=1)
            _cols = _is_content.any(dim=0)
            r0 = int(_rows.nonzero(as_tuple=False)[0])
            r1 = int(_rows.nonzero(as_tuple=False)[-1]) + 1
            c0 = int(_cols.nonzero(as_tuple=False)[0])
            c1 = int(_cols.nonzero(as_tuple=False)[-1]) + 1

            _mean = torch.tensor(IMG_NORM_MEAN, device=x.device).view(1, 3, 1, 1)
            _std = torch.tensor(IMG_NORM_STD, device=x.device).view(1, 3, 1, 1)
            _img_crop = (x[0:1].float() * _std + _mean).clamp(0, 1)
            _img_crop = _img_crop[:, :,
                                  r0 * self.patch_size:r1 * self.patch_size,
                                  c0 * self.patch_size:c1 * self.patch_size]

            _f = _feat[0, r0:r1, c0:c1, :].reshape(-1, _feat.shape[-1]).detach().cpu().float()
            _f_c = _f - _f.mean(0, keepdim=True)
            _, _S, _V = torch.pca_lowrank(_f_c, q=3, niter=4)
            _proj = _f_c @ _V
            _proj = _proj / (_proj.std(0, keepdim=True) + 1e-6)
            _rgb = torch.sigmoid(_proj * 2.0)
            H_crop, W_crop = r1 - r0, c1 - c0
            _rgb = _rgb.reshape(1, H_crop, W_crop, 3).permute(0, 3, 1, 2)

            _H_px, _W_px = _img_crop.shape[2], _img_crop.shape[3]
            _rgb_up = F.interpolate(_rgb, size=(_H_px, _W_px), mode='bilinear',
                                    align_corners=False)
            _vis = torch.cat([_img_crop.cpu(), _rgb_up], dim=3)
            save_image(_vis, 'pca_debug.png')
            out['pca_feat'] = _rgb_up

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
        ).permute(0, 2, 3, 1)
        dec_emb = self.dec_to_token(z['feat']) + dec_pos_emb

        # 4. Extract HPH queries at detected/GT locations
        queries = dec_emb[idx[0], idx[1], idx[2]]
        values, counts = torch.unique(idx[0], sorted=True, return_counts=True)
        queries = torch.stack([
            F.pad(q, (0, 0, 0, max(counts) - counts[i]), mode='constant', value=0)
            for i, q in enumerate(torch.split(queries, tuple(counts), dim=0))
        ])
        mask = torch.cat([
            F.pad(torch.ones(1, c, 1).to(x.device), (0, 0, 0, max(counts) - c), mode='constant', value=0)
            for c in tuple(counts)
        ], dim=0)[..., 0]

        # 5. Context = all patch features
        context = dec_emb[values].flatten(1, 2)

        # 6. HPH decoder
        y = self.decoder(x=queries, context=context, mask=mask)
        y = torch.cat([y[i, :c] for i, c in enumerate(tuple(counts))], dim=0)

        # 7. Wrist center 2D location
        offset = self.mlp_offset(y)
        loc_coarse = torch.stack([idx[2], idx[1]], dim=1)
        loc = (loc_coarse + 0.5 + offset) * self.encoder.patch_size

        # 8. Pixel features shared by segmentation and CSE heads
        pixel_feat = self.pixel_decoder(z['feat'], self.img_size)

        # 9. Instance mask prediction
        mask_emb = self.mask_head(y)
        inst_mask_logits = self._compute_masks(mask_emb, pixel_feat, idx, counts, values)

        # 10. Part mask prediction
        part_embs = self.part_head(y)
        part_embs = part_embs.reshape(-1, self.num_parts, self.mask_dim)
        part_mask_logits = self._compute_part_masks(part_embs, pixel_feat, idx, counts, values)

        # 11. NOCS prediction, conditioned on HPH instance queries
        cse_embs = self.cse_head(y).reshape(-1, 3, self.mask_dim)
        nocs_pred = self._compute_nocs(cse_embs, pixel_feat, idx, counts, values)

        out = {
            'scores': z['scores'],
            'scores_logits': z['scores_logits'],
            'loc': loc,
            'offset': offset,
            'inst_mask_logits': inst_mask_logits,
            'part_mask_logits': part_mask_logits,
            'nocs_pred': nocs_pred,
            'feat': z['feat'],
        }

        if is_training:
            return out

        instruments = []
        for i in range(idx[0].shape[0]):
            inst = {
                'batch_idx': idx[0][i].item(),
                'loc': out['loc'][i],
                'inst_mask_logits': out['inst_mask_logits'][i],
                'part_mask_logits': out['part_mask_logits'][i],
                'nocs_pred': out['nocs_pred'][i],
            }
            instruments.append(inst)
        return instruments

    def _compute_nocs(self, cse_embs, pixel_feat, idx, counts, values):
        """
        Args:
            cse_embs: [n_inst, 3, mask_dim]
            pixel_feat: [bs, mask_dim, H/4, W/4]
        Returns:
            [n_inst, 3, H/4, W/4]
        """
        embs_split = torch.split(cse_embs, tuple(counts), dim=0)
        all_nocs = []
        for emb, batch_idx in zip(embs_split, values):
            pf = pixel_feat[batch_idx]
            nocs = torch.einsum('ncd,dhw->nchw', emb, pf)
            all_nocs.append(nocs)
        return torch.cat(all_nocs, dim=0)
