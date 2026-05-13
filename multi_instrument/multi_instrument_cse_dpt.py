import torch
from torch import nn

from multi_instrument.dpt_nocs_head import DPTNocsHead
from multi_instrument.encoder_cse import EncoderCSE
from multi_instrument.multi_instrument_cse import MultiInstrumentCSE


class MultiInstrumentCSEDPT(MultiInstrumentCSE):
    """
    DPT-backed NOCS variant that leaves the existing instance/part segmentation
    path intact and replaces only the exported nocs_pred with a dense DPT map.

    The DPT map is per image: [B, 3, H/4, W/4]. For compatibility with the
    existing CSE loss, it is gathered by each visible/detected instance's batch
    index and returned as [N_inst, 3, H/4, W/4].
    """

    def __init__(self, nocs_feat_dim: int = 256, *args, **kwargs):
        backbone = kwargs.get("backbone", "dinov2_vits14")
        pretrained_backbone = kwargs.get("pretrained_backbone", False)
        if backbone.startswith("dinov3_") or "dinov3" in backbone:
            raise ValueError(
                "MultiInstrumentCSEDPT currently requires a DINOv2 backbone, "
                "because EncoderCSE provides DINOv2 intermediate features."
            )

        super().__init__(nocs_feat_dim=nocs_feat_dim, *args, **kwargs)

        # Swap the vanilla encoder for the hook-based encoder that exposes
        # four intermediate ViT feature maps for DPT.
        self.encoder = EncoderCSE(backbone, pretrained=pretrained_backbone)
        self.patch_size = self.encoder.patch_size

        if self.dec_to_token.in_features != self.encoder.embed_dim:
            self.dec_to_token = nn.Linear(self.encoder.embed_dim, self.decoder.dim)

        self.nocs_dpt_head = DPTNocsHead(
            embed_dim=self.encoder.embed_dim,
            feat_dim=nocs_feat_dim,
        )

    def _compute_dpt_nocs_map(self):
        feats = self.encoder.last_intermediate_feats
        if feats is None:
            raise RuntimeError(
                "EncoderCSE did not expose intermediate features. "
                "Call the encoder before computing the DPT NOCS map."
            )
        return self.nocs_dpt_head(feats, target_size=self.img_size // 4)

    def forward(self, x, idx=None, is_training=False, *args, **kwargs):
        base_out = super().forward(
            x,
            idx=idx,
            is_training=is_training,
            *args,
            **kwargs,
        )

        # No detections in inference follows the parent return convention.
        if not is_training and not isinstance(base_out, list):
            return base_out

        nocs_map = self._compute_dpt_nocs_map()

        if is_training:
            if idx is None:
                raise ValueError("Training with MultiInstrumentCSEDPT requires idx.")
            base_out["nocs_pred_query"] = base_out.get("nocs_pred")
            base_out["nocs_pred_global"] = nocs_map
            base_out["nocs_pred"] = nocs_map[idx[0]]
            return base_out

        for inst in base_out:
            batch_idx = int(inst["batch_idx"])
            inst["nocs_pred_query"] = inst.get("nocs_pred")
            inst["nocs_pred"] = nocs_map[batch_idx]
        return base_out
