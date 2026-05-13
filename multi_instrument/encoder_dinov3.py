import math
import torch
from torch import nn
import timm


class EncoderDINOv3(nn.Module):
    """
    Drop-in replacement for Encoder that uses a DINOv3 ViT backbone via timm.

    Exposes the same attributes and forward() return dict as Encoder so that
    Multi_Instrument requires no further changes beyond selecting this class.

    Default model: vit_large_patch16_dinov3.lvd1689m
        embed_dim=1024, patch_size=16
    """

    def __init__(self, name='vit_large_patch16_dinov3.lvd1689m', pretrained=True):
        super().__init__()
        self.name = name

        self.backbone = timm.create_model(name, pretrained=pretrained, num_classes=0)

        self.patch_size = self.backbone.patch_embed.patch_size[0]  # 16
        self.embed_dim  = self.backbone.embed_dim                   # 1024

        # Patch-level detection head (identical to Encoder)
        self.mlp_det = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1),
        )

        # FOV prediction head (identical to Encoder)
        self.mlp_fov_unique = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1),
        )
        self.register_buffer("fov_max", torch.tensor([math.pi]))

    def forward(self, x):
        """
        Args:
            x: [bs, 3, H, W]  (H and W divisible by patch_size)
        Returns:
            dict with keys: scores_logits, scores, K, fov, feat
            (identical layout to Encoder.forward)
        """
        assert len(x.shape) == 4
        H, W = x.shape[2], x.shape[3]

        # Eva (timm) returns [bs, num_prefix_tokens + N_patches, C]
        # prefix tokens: [cls, reg1, reg2, ...]; patch tokens follow
        tokens = self.backbone.forward_features(x)   # [bs, P+N, C]
        n_prefix = self.backbone.num_prefix_tokens    # 5 for lvd1689m
        cls = tokens[:, 0, :]                         # [bs, C]
        patch = tokens[:, n_prefix:, :]               # [bs, N, C]

        H_p = H // self.patch_size
        W_p = W // self.patch_size
        y = patch.reshape(patch.shape[0], H_p, W_p, patch.shape[-1])  # [bs, H_p, W_p, C]

        # FOV / camera intrinsics (use H as the reference dimension, consistent with Encoder)
        fov = self.fov_max * torch.sigmoid(self.mlp_fov_unique(cls))
        focal_length = (H / 2) / torch.tan(fov / 2)
        K = torch.eye(3).float().to(cls.device).reshape(1, 3, 3).repeat(cls.shape[0], 1, 1)
        K[:, [0, 1], [0, 1]] = focal_length[:, [0]]
        K[:, [0, 1], [-1, -1]] = H / 2.

        # Patch-level detection
        scores_logits = self.mlp_det(y)[..., 0]   # [bs, H_p, W_p]
        scores = torch.sigmoid(scores_logits)

        return {
            'scores_logits': scores_logits,
            'scores': scores,
            'K': K,
            'fov': fov,
            'feat': y,
        }
