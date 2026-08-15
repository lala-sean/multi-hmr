import importlib.util
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ROBOPEPP_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROBOPEPP_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBOPEPP_ROOT))

from instrument_geometry import instrument_keypoints_camera_torch, project_points_torch  # noqa: E402


def _load_robopepp_vit():
    root = Path(__file__).resolve().parent / "backbones" / "vit.py"
    spec = importlib.util.spec_from_file_location("robopepp_vit", root)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load RoboPEPP vit module: {root}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


vit = _load_robopepp_vit()


class IterativeRegressionHead(nn.Module):
    def __init__(self, feature_dim, output_dim, init_value=None, hidden_dim=1024, n_iter=4, dropout=0.3):
        super().__init__()
        self.n_iter = int(n_iter)
        if init_value is None:
            init = torch.zeros(output_dim, dtype=torch.float32)
        else:
            init = torch.as_tensor(init_value, dtype=torch.float32).reshape(output_dim)
        self.register_buffer("init_value", init.unsqueeze(0))
        self.fc_pose_1 = nn.Linear(feature_dim + output_dim, hidden_dim)
        self.fc_pose_2 = nn.Linear(hidden_dim, hidden_dim)
        self.decpose = nn.Linear(hidden_dim, output_dim)
        self.drop1 = nn.Dropout(p=dropout)
        self.drop2 = nn.Dropout(p=dropout)
        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.zeros_(self.decpose.bias)

    def forward(self, xf):
        pred = self.init_value.expand(xf.shape[0], -1)
        for _ in range(self.n_iter):
            x = torch.cat([xf, pred], 1).to(torch.float32)
            x = self.fc_pose_1(x)
            x = self.drop1(x)
            x = self.fc_pose_2(x)
            x = self.drop2(x)
            pred = self.decpose(x) + pred
        return pred


class KeypointNet(nn.Module):
    def __init__(self, in_channels, num_keypoints, dropout_prob=0.4):
        super().__init__()
        self.deconv1 = nn.ConvTranspose2d(in_channels, 256, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.deconv4 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.out_layer1 = nn.Conv2d(256, num_keypoints, kernel_size=1, stride=1)
        self.dropout = nn.Dropout(p=dropout_prob)
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in (self.deconv1, self.deconv2, self.deconv3, self.deconv4):
            nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
            nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        x = self.dropout(F.relu(self.bn1(self.deconv1(x)), inplace=True))
        x = self.dropout(F.relu(self.bn2(self.deconv2(x)), inplace=True))
        x = self.dropout(F.relu(self.bn3(self.deconv3(x)), inplace=True))
        x = self.dropout(F.relu(self.bn4(self.deconv4(x)), inplace=True))
        return torch.sigmoid(self.out_layer1(x.contiguous()))


class RoboPEPPInstrumentPoseNet(nn.Module):
    def __init__(
        self,
        backbone="vit_base",
        input_shape=(224, 224),
        patch_size=16,
        pred_emb_dim=384,
        pred_depth=12,
        num_keypoints=5,
        pose_head_iter=4,
        pose_head_dropout=0.3,
        jepa_path=None,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.context_backbone = vit.__dict__[backbone](img_size=[input_shape[0]], patch_size=patch_size)
        self.predictor_backbone = vit.__dict__["vit_predictor"](
            num_patches=self.context_backbone.patch_embed.num_patches,
            embed_dim=self.context_backbone.embed_dim,
            predictor_embed_dim=pred_emb_dim,
            depth=pred_depth,
            num_heads=self.context_backbone.num_heads,
        )
        self.feature_channel = self.context_backbone.embed_dim
        self.action_net = IterativeRegressionHead(
            self.feature_channel,
            3,
            n_iter=pose_head_iter,
            dropout=pose_head_dropout,
        )
        self.wrist_pose_net = IterativeRegressionHead(
            self.feature_channel,
            7,
            init_value=torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            n_iter=pose_head_iter,
            dropout=pose_head_dropout,
        )
        self.keypoint_net = KeypointNet(self.feature_channel, num_keypoints)
        self.num_keypoints = int(num_keypoints)
        if jepa_path is not None:
            self.load_jepa(jepa_path)

    def load_jepa(self, jepa_path):
        checkpoint = torch.load(jepa_path, map_location=torch.device("cpu"), weights_only=True)
        for attr, key in (("context_backbone", "encoder"), ("predictor_backbone", "predictor")):
            pretrained = checkpoint[key]
            state = {k.replace("module.", ""): v for k, v in pretrained.items()}
            getattr(self, attr).load_state_dict(state)
        print(f"Loaded RoboPEPP JEPA weights from {jepa_path}")

    def forward(self, x, K=None, masks_enc=None, masks_pred=None):
        B = x.shape[0]
        if masks_enc is None:
            img_feat = self.context_backbone(x)
            img_feat = self.predictor_backbone(img_feat, None, None)
        else:
            img_feat_context = self.context_backbone(x, masks_enc)
            img_feat = self.predictor_backbone(img_feat_context, masks_enc, masks_pred)

        xf = img_feat.mean(dim=1)
        action_pred = self.action_net(xf)
        wrist_raw = self.wrist_pose_net(xf)
        wrist_quat_pred = F.normalize(wrist_raw[:, :4], p=2, dim=1)
        wrist_trans_pred = wrist_raw[:, 4:7]

        num_patch = int(np.sqrt(img_feat.shape[1]))
        if num_patch * num_patch != img_feat.shape[1]:
            raise RuntimeError(f"RoboPEPP predictor produced non-square token count: {img_feat.shape[1]}")
        feat_map = img_feat.permute(0, 2, 1).view(B, self.feature_channel, num_patch, num_patch)
        keypoint_heatmaps = self.keypoint_net(feat_map)

        keypoints_3d_cam = instrument_keypoints_camera_torch(wrist_quat_pred, wrist_trans_pred, action_pred)
        out = {
            "action_pred": action_pred,
            "wrist_quat_pred": wrist_quat_pred,
            "wrist_trans_pred": wrist_trans_pred,
            "keypoint_heatmaps": keypoint_heatmaps,
            "keypoints_3d_cam": keypoints_3d_cam,
        }
        if K is not None:
            out["keypoints_uv"] = project_points_torch(keypoints_3d_cam, K)
        return out


def make_robopepp_instrument_posenet(**kwargs):
    return RoboPEPPInstrumentPoseNet(**kwargs)

