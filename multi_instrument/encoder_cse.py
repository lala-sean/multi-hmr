import numpy as np
import torch
from multi_instrument.encoder import Encoder
from utils import unpatch


class EncoderCSE(Encoder):
    """
    DINOv2 encoder that captures 4 evenly-spaced intermediate ViT-block features
    via forward hooks — backbone runs exactly once.

    Additional key in returned dict:
        'intermediate_feats': list of 4 tensors [B, H_p, W_p, D],
                              from shallowest to deepest ViT layer.

    Also stored as self.last_intermediate_feats for access after forward().
    """

    def __init__(self, name='dinov2_vits14', pretrained=False, *args, **kwargs):
        super().__init__(name, pretrained, *args, **kwargs)

        n = len(self.backbone.blocks)
        # 3 evenly-spaced hook layers; the 4th (final) comes from Encoder's normal output
        self._hook_idxs = [n // 4 - 1, n // 2 - 1, 3 * n // 4 - 1]

        self._hooked = []
        for i in self._hook_idxs:
            self.backbone.blocks[i].register_forward_hook(
                lambda m, inp, out, _i=i: self._hooked.append(out)
            )

        self.last_intermediate_feats = None

    def forward(self, x):
        self._hooked.clear()
        z = super().forward(x)          # backbone runs once; 3 hooks fire

        B, H_p, W_p, D = z['feat'].shape
        feats = []
        for raw in self._hooked:
            # DINOv2 blocks return a plain tensor [B, 1+N_patches, D]
            tokens = raw[0] if isinstance(raw, tuple) else raw
            patch_tokens = tokens[:, 1:, :]            # strip cls token → [B, N, D]
            feats.append(patch_tokens.reshape(B, H_p, W_p, D))
        feats.append(z['feat'])                         # 4th = final layer (already [B,H_p,W_p,D])

        z['intermediate_feats'] = feats
        self.last_intermediate_feats = feats            # accessible after forward()
        return z
