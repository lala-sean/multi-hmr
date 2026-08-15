import ast
import sys
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

ROBOPEPP_ROOT = Path(__file__).resolve().parents[1]
MULTIHMR_ROOT = Path(__file__).resolve().parents[3]
if str(ROBOPEPP_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBOPEPP_ROOT))
if str(MULTIHMR_ROOT) not in sys.path:
    sys.path.insert(0, str(MULTIHMR_ROOT))

from multi_instrument.dpt_hcce_head import DPTHCCEHead  # noqa: E402
from multi_instrument.encoder_cse import EncoderCSE  # noqa: E402
from multi_instrument.instrument_pose_heads import InstrumentActionHead, InstrumentWristPoseHead  # noqa: E402


def _load_keypoint_net():
    path = ROBOPEPP_ROOT / "models" / "model.py"
    tree = ast.parse(path.read_text())
    keep = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in {"SpatialSoftmax", "KeypointNet"}:
            keep.append(node)
    if not any(isinstance(node, ast.ClassDef) and node.name == "KeypointNet" for node in keep):
        raise ImportError(f"Could not find RoboPEPP KeypointNet in {path}")
    module = ast.Module(body=keep, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"torch": torch, "nn": nn, "F": F}
    exec(compile(module, str(path), "exec"), namespace)
    keypoint_net = namespace["KeypointNet"]
    keypoint_net.__module__ = "robopepp_original_keypointnet"
    return keypoint_net


KeypointNet = _load_keypoint_net()


class CropHCCEDenseKeypointDPT(nn.Module):
    """
    Crop-based single-instrument HCCE model.

    There is no query/detection branch.  A shared DINOv2 encoder feeds:
      - dense instance logit at 224x224,
      - dense part logits [wrist, gripper, shaft] at 224x224,
      - dense HCCE bit logits at 224x224,
      - RoboPEPP KeypointNet heatmaps at 224x224,
      - global action and wrist SE(3) regression heads.
    """

    def __init__(
        self,
        img_size=224,
        backbone="dinov2_vits14",
        pretrained_backbone=True,
        hcce_feat_dim=256,
        hcce_bits=8,
        num_keypoints=5,
        action_dim=3,
        pose_head_iter=4,
        pose_head_dropout=0.3,
        keypoint_feat_size=14,
        *args,
        **kwargs,
    ):
        super().__init__()
        if img_size != 224:
            raise ValueError("CropHCCEDenseKeypointDPT is intentionally fixed to 224x224 crops.")
        if backbone.startswith("dinov3_") or "dinov3" in backbone:
            raise ValueError("CropHCCEDenseKeypointDPT currently expects DINOv2 EncoderCSE.")
        self.img_size = int(img_size)
        self.hcce_bits = int(hcce_bits)
        self.num_keypoints = int(num_keypoints)
        self.keypoint_feat_size = None if keypoint_feat_size is None else int(keypoint_feat_size)
        if self.keypoint_feat_size is not None and self.keypoint_feat_size * 16 != self.img_size:
            raise ValueError(
                "RoboPEPP KeypointNet uses four stride-2 deconvs, so "
                f"keypoint_feat_size*16 must equal img_size; got {self.keypoint_feat_size} and {self.img_size}."
            )

        self.encoder = EncoderCSE(backbone, pretrained=pretrained_backbone)
        for aux_name in ("mlp_det", "mlp_fov_unique"):
            aux = getattr(self.encoder, aux_name, None)
            if aux is not None:
                for p in aux.parameters():
                    p.requires_grad_(False)
        self.patch_size = self.encoder.patch_size
        self.dense_head = DPTHCCEHead(
            embed_dim=self.encoder.embed_dim,
            feat_dim=hcce_feat_dim,
            out_channels=1 + 3 + 3 * self.hcce_bits,
        )
        self.keypoint_net = KeypointNet(self.encoder.embed_dim, self.num_keypoints)
        self.action_head = InstrumentActionHead(
            feature_dim=self.encoder.embed_dim,
            n_actions=action_dim,
            n_iter=pose_head_iter,
            dropout=pose_head_dropout,
        )
        self.wrist_pose_head = InstrumentWristPoseHead(
            feature_dim=self.encoder.embed_dim,
            n_iter=pose_head_iter,
            dropout=pose_head_dropout,
        )

    def forward(self, x, K=None):
        z = self.encoder(x)
        dense = self.dense_head(self.encoder.last_intermediate_feats, target_size=self.img_size)
        inst_mask_logits = dense[:, 0]
        part_mask_logits = dense[:, 1:4]
        hcce_logits = dense[:, 4:]

        feat = z["feat"].permute(0, 3, 1, 2).contiguous()
        if feat.ndim != 4 or feat.shape[1] != self.encoder.embed_dim:
            raise RuntimeError(f"unexpected keypoint feature shape: {tuple(feat.shape)}")
        if self.keypoint_feat_size is None:
            kp_feat = feat
        else:
            kp_feat = F.interpolate(
                feat,
                size=(self.keypoint_feat_size, self.keypoint_feat_size),
                mode="bilinear",
                align_corners=False,
            )
        keypoint_heatmaps = self.keypoint_net(kp_feat)
        if keypoint_heatmaps.shape[-2:] != (self.img_size, self.img_size):
            keypoint_heatmaps = F.interpolate(
                keypoint_heatmaps,
                size=(self.img_size, self.img_size),
                mode="bilinear",
                align_corners=False,
            ).clamp_(0.0, 1.0)
        if keypoint_heatmaps.shape[1:] != (self.num_keypoints, self.img_size, self.img_size):
            raise RuntimeError(f"unexpected RoboPEPP KeypointNet output shape: {tuple(keypoint_heatmaps.shape)}")

        pooled = z["feat"].mean(dim=(1, 2))
        action_pred = self.action_head(pooled)
        wrist_quat_pred, wrist_trans_pred = self.wrist_pose_head(pooled)
        return {
            "inst_mask_logits": inst_mask_logits,
            "part_mask_logits": part_mask_logits,
            "hcce_logits": hcce_logits,
            "keypoint_heatmaps": keypoint_heatmaps,
            "action_pred": action_pred,
            "wrist_quat_pred": wrist_quat_pred,
            "wrist_trans_pred": wrist_trans_pred,
            "dense_logits": dense,
        }
