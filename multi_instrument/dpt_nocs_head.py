"""
Dense Prediction Transformer (DPT) head for per-pixel NOCS coordinate regression.

Faithful re-implementation of the DPT decoder from:
    Ranftl et al., "Vision Transformers for Dense Prediction", ICCV 2021
    https://github.com/isl-org/DPT

Architecture:
    4 intermediate ViT features (shallowest → deepest)
    → ReassembleBlock × 4   (project channels + spatial resize)
    → FeatureFusionBlock × 4 (bottom-up, each 2× bilinear upsample)
    → Head conv               (project to 3 xyz channels)
    → F.interpolate to (target_size, target_size)

Output: [B, 3, target_size, target_size]  raw xyz per pixel, no activation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualConvUnit(nn.Module):
    """Two conv layers with residual connection (no BN, matching DPT paper)."""

    def __init__(self, features: int):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, 3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(features, features, 3, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        return out + x


class FeatureFusionBlock(nn.Module):
    """
    Fuses the output of the previous fusion stage with the current reassemble
    output, then doubles spatial resolution via bilinear 2× upsample.

    forward(x, res=None):
        x   — current-depth reassembled feature [B, C, H, W]
        res — output from previous (deeper) fusion stage (optional)
    """

    def __init__(self, features: int):
        super().__init__()
        self.resconf1 = ResidualConvUnit(features)
        self.resconf2 = ResidualConvUnit(features)

    def forward(self, x, res=None):
        if res is not None:
            # Align res to x's spatial size before adding (handles non-power-of-2 H_p)
            if res.shape[-2:] != x.shape[-2:]:
                res = F.interpolate(res, size=x.shape[-2:],
                                    mode='bilinear', align_corners=True)
            x = x + self.resconf1(res)
        x = self.resconf2(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return x


class ReassembleBlock(nn.Module):
    """
    Projects embed_dim → feat_dim and spatially resizes.

    stride_factor:
        4   → 4× upsample  (two ConvTranspose2d stride-2)
        2   → 2× upsample  (one ConvTranspose2d stride-2)
        1   → unchanged    (Conv1×1 only)
        0.5 → 2× downsample (Conv stride-2)
    """

    def __init__(self, embed_dim: int, feat_dim: int, stride_factor):
        super().__init__()
        self.proj = nn.Conv2d(embed_dim, feat_dim, kernel_size=1)

        if stride_factor == 4:
            self.spatial = nn.Sequential(
                nn.ConvTranspose2d(feat_dim, feat_dim, kernel_size=2, stride=2),
                nn.ConvTranspose2d(feat_dim, feat_dim, kernel_size=2, stride=2),
            )
        elif stride_factor == 2:
            self.spatial = nn.ConvTranspose2d(feat_dim, feat_dim, kernel_size=2, stride=2)
        elif stride_factor == 1:
            self.spatial = nn.Identity()
        elif stride_factor == 0.5:
            self.spatial = nn.Conv2d(feat_dim, feat_dim, kernel_size=3, stride=2, padding=1)
        else:
            raise ValueError(f"Unsupported stride_factor: {stride_factor}")

    def forward(self, x):
        # x: [B, H_p, W_p, D]
        x = x.permute(0, 3, 1, 2).contiguous()   # [B, D, H_p, W_p]
        x = self.proj(x)                           # [B, feat_dim, H_p, W_p]
        x = self.spatial(x)
        return x


class DPTNocsHead(nn.Module):
    """
    DPT decoder that regresses per-pixel NOCS (canonical xyz) coordinates.

    Args:
        embed_dim:   ViT token dimension (e.g. 384 for ViT-S/14)
        feat_dim:    internal DPT feature dimension (default 256, matches paper)

    forward(feats, target_size) → [B, 3, target_size, target_size]
        feats: list of 4 tensors [B, H_p, W_p, embed_dim],
               ordered shallowest → deepest (same order as EncoderCSE.last_intermediate_feats)
        target_size: output spatial size (e.g. img_size // 4)
    """

    def __init__(self, embed_dim: int, feat_dim: int = 256):
        super().__init__()

        # Reassemble: 4 spatial scales (shallowest first)
        self.reassemble = nn.ModuleList([
            ReassembleBlock(embed_dim, feat_dim, stride_factor=4),    # feat0 → ×4
            ReassembleBlock(embed_dim, feat_dim, stride_factor=2),    # feat1 → ×2
            ReassembleBlock(embed_dim, feat_dim, stride_factor=1),    # feat2 → ×1
            ReassembleBlock(embed_dim, feat_dim, stride_factor=0.5),  # feat3 → ×0.5
        ])

        # Fusion: 4 blocks (indexed deepest → shallowest)
        self.fusion = nn.ModuleList([
            FeatureFusionBlock(feat_dim),   # fusion[0]: deepest, no residual input
            FeatureFusionBlock(feat_dim),   # fusion[1]: + r2
            FeatureFusionBlock(feat_dim),   # fusion[2]: + r1
            FeatureFusionBlock(feat_dim),   # fusion[3]: + r0 (shallowest)
        ])

        # Prediction head
        self.head = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim // 2, 3, kernel_size=1),
        )

    def forward(self, feats, target_size: int):
        """
        feats: list of 4 × [B, H_p, W_p, D], shallowest first
        target_size: int
        """
        assert len(feats) == 4, f"Expected 4 feature maps, got {len(feats)}"

        # Reassemble all 4 scales
        r = [self.reassemble[i](feats[i]) for i in range(4)]
        # r[0]: [B, C, 4·H_p, 4·W_p]
        # r[1]: [B, C, 2·H_p, 2·W_p]
        # r[2]: [B, C, H_p, W_p]
        # r[3]: [B, C, H_p//2, W_p//2]

        # Fusion bottom-up: start from deepest (r[3])
        f = self.fusion[0](r[3])          # no residual, then 2× upsample → [B,C,H_p,W_p]
        f = self.fusion[1](r[2], res=f)   # fuse r2, then 2× upsample → [B,C,2·H_p,2·W_p]
        f = self.fusion[2](r[1], res=f)   # fuse r1, then 2× upsample → [B,C,4·H_p,4·W_p]
        f = self.fusion[3](r[0], res=f)   # fuse r0, then 2× upsample → [B,C,8·H_p,8·W_p]

        # Head
        out = self.head(f)                # [B, 3, 8·H_p, 8·W_p]

        # Resize to target
        out = F.interpolate(out, size=(target_size, target_size),
                            mode='bilinear', align_corners=False)
        return out                        # [B, 3, target_size, target_size]
