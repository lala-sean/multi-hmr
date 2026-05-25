import torch.nn as nn
import torch.nn.functional as F


class HCCEResidualConvUnit(nn.Module):
    def __init__(self, features):
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


class HCCEFeatureFusionBlock(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.resconf1 = HCCEResidualConvUnit(features)
        self.resconf2 = HCCEResidualConvUnit(features)

    def forward(self, x, res=None):
        if res is not None:
            if res.shape[-2:] != x.shape[-2:]:
                res = F.interpolate(res, size=x.shape[-2:], mode="bilinear", align_corners=True)
            x = x + self.resconf1(res)
        x = self.resconf2(x)
        return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)


class HCCEReassembleBlock(nn.Module):
    def __init__(self, embed_dim, feat_dim, stride_factor):
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
        x = x.permute(0, 3, 1, 2).contiguous()
        return self.spatial(self.proj(x))


class DPTHCCEHead(nn.Module):
    """
    New DPT decoder for front-only HCCE code logits.

    This is intentionally separate from DPTNocsHead so existing CSE/NOCS models
    and functions remain untouched.
    """

    def __init__(self, embed_dim, feat_dim=256, out_channels=24):
        super().__init__()
        self.reassemble = nn.ModuleList(
            [
                HCCEReassembleBlock(embed_dim, feat_dim, stride_factor=4),
                HCCEReassembleBlock(embed_dim, feat_dim, stride_factor=2),
                HCCEReassembleBlock(embed_dim, feat_dim, stride_factor=1),
                HCCEReassembleBlock(embed_dim, feat_dim, stride_factor=0.5),
            ]
        )
        self.fusion = nn.ModuleList(
            [
                HCCEFeatureFusionBlock(feat_dim),
                HCCEFeatureFusionBlock(feat_dim),
                HCCEFeatureFusionBlock(feat_dim),
                HCCEFeatureFusionBlock(feat_dim),
            ]
        )
        self.head = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim // 2, out_channels, kernel_size=1),
        )

    def forward(self, feats, target_size):
        if len(feats) != 4:
            raise ValueError(f"Expected 4 feature maps, got {len(feats)}")
        r = [self.reassemble[i](feats[i]) for i in range(4)]
        f = self.fusion[0](r[3])
        f = self.fusion[1](r[2], res=f)
        f = self.fusion[2](r[1], res=f)
        f = self.fusion[3](r[0], res=f)
        out = self.head(f)
        return F.interpolate(out, size=(target_size, target_size), mode="bilinear", align_corners=False)
