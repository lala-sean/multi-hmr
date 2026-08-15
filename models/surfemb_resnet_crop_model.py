import sys
from pathlib import Path

import torch
from torch import nn

ROBOPEPP_ROOT = Path(__file__).resolve().parents[1]
MULTIHMR_ROOT = Path(__file__).resolve().parents[3]
SURFEMB_ROOT = MULTIHMR_ROOT / "submodules" / "surfemb"
if str(ROBOPEPP_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBOPEPP_ROOT))
if str(MULTIHMR_ROOT) not in sys.path:
    sys.path.insert(0, str(MULTIHMR_ROOT))
if SURFEMB_ROOT.is_dir() and str(SURFEMB_ROOT) not in sys.path:
    sys.path.insert(0, str(SURFEMB_ROOT))

from surfemb.dep.siren import Siren  # noqa: E402
from surfemb.dep.unet import ResNetUNet  # noqa: E402


class SurfEmbResNetCropModel(nn.Module):
    """
    Original-SurfEmb-style crop model.

    The CNN is the ResNet18 U-Net from the SurfEmb repo.  It predicts one
    binary object-mask logit channel plus a dense query embedding image.  A
    SIREN key MLP maps canonical surface coordinates to the same embedding
    space for InfoNCE correspondence supervision.
    """

    def __init__(
        self,
        img_size=224,
        surfemb_emb_dim=12,
        surfemb_mlp_hidden_features=256,
        surfemb_mlp_hidden_layers=2,
        resnet_feat_preultimate=64,
    ):
        super().__init__()
        if int(img_size) != 224:
            raise ValueError("SurfEmbResNetCropModel is fixed to 224x224 crops.")
        self.img_size = int(img_size)
        self.surfemb_emb_dim = int(surfemb_emb_dim)
        self.cnn = ResNetUNet(
            n_class=1 + self.surfemb_emb_dim,
            feat_preultimate=int(resnet_feat_preultimate),
            n_decoders=1,
        )
        # The original SurfEmb U-Net reuses ResNet feature layers but leaves the
        # torchvision classification FC registered.  It is not on the forward
        # path, so remove its trainable parameters for DDP.
        if hasattr(self.cnn, "base_model") and hasattr(self.cnn.base_model, "fc"):
            self.cnn.base_model.fc = nn.Identity()
        self.surface_key_mlp = Siren(
            in_features=3,
            out_features=self.surfemb_emb_dim,
            hidden_features=int(surfemb_mlp_hidden_features),
            hidden_layers=int(surfemb_mlp_hidden_layers),
        )

    def forward(self, x, K=None, surfemb_key_coords=None):
        dense = self.cnn(x)
        if dense.shape[-2:] != (self.img_size, self.img_size):
            raise RuntimeError(f"unexpected SurfEmb dense output shape: {tuple(dense.shape)}")
        out = {
            "inst_mask_logits": dense[:, 0],
            "surfemb_queries": dense[:, 1:],
            "dense_logits": dense,
        }
        if surfemb_key_coords is not None:
            out["surfemb_keys"] = self.surface_key_mlp(surfemb_key_coords)
        return out
