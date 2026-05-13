import torch
from torch import nn
import torch.nn.functional as F


class PixelDecoder(nn.Module):
    """
    Lightweight upsampling module that converts DINOv2 patch features
    from 1/14 resolution to 1/4 resolution for mask prediction.

    Input:  [bs, sqrt_np, sqrt_np, embed_dim]
    Output: [bs, mask_dim, H/4, W/4]
    """

    def __init__(self, embed_dim, mask_dim=256):
        super().__init__()
        self.mask_dim = mask_dim

        # Project from backbone dim to mask dim
        self.input_proj = nn.Linear(embed_dim, mask_dim)

        # Upsample 2x via transposed convolution
        self.upsample = nn.ConvTranspose2d(mask_dim, mask_dim, kernel_size=2, stride=2)

        # Refinement conv
        self.refine = nn.Conv2d(mask_dim, mask_dim, kernel_size=3, stride=1, padding=1)

        self.norm1 = nn.GroupNorm(32, mask_dim)
        self.norm2 = nn.GroupNorm(32, mask_dim)
        self.act = nn.GELU()

    def forward(self, feat, img_size):
        """
        Args:
            feat: [bs, sqrt_np, sqrt_np, embed_dim] - backbone patch features
            img_size: int - original input image size (e.g. 896)
        Returns:
            pixel_feat: [bs, mask_dim, H/4, W/4]
        """
        # Project to mask_dim
        x = self.input_proj(feat)  # [bs, h, w, mask_dim]
        x = x.permute(0, 3, 1, 2)  # [bs, mask_dim, h, w]

        # Upsample 2x: e.g. 64x64 -> 128x128
        x = self.act(self.norm1(self.upsample(x)))

        # Refine
        x = self.act(self.norm2(self.refine(x)))

        # Interpolate to exact 1/4 resolution
        target_size = img_size // 4
        if x.shape[-1] != target_size or x.shape[-2] != target_size:
            x = F.interpolate(x, size=(target_size, target_size), mode='bilinear', align_corners=False)

        return x
