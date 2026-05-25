from types import SimpleNamespace

import numpy as np
import torch

from loss_instrument_hcce_pose import LossInstrumentHCCEPose
from multi_instrument.hcce_codec import (
    hcce_decode_torch,
    hcce_encode_numpy,
    normalized_xyz_to_hcce,
)
from multi_instrument.instrument_pose_heads import (
    InstrumentActionHead,
    InstrumentWristPoseHead,
)
from datasets.rarp_pose_canonicalization import (
    SX_180,
    canonicalize_pose_symmetry,
    fk_matrices,
    matrix_to_quat,
    quat_to_matrix,
)


def _reference_hcce_encode(code_img, iteration=8):
    code_img = code_img.copy().astype(np.int64)
    code_img = [code_img[:, :, 0], code_img[:, :, 1], code_img[:, :, 2]]
    hcce_images = np.zeros((code_img[0].shape[0], code_img[0].shape[1], iteration * 3))
    for i in range(iteration):
        temp1 = np.array(code_img[0] % (2 ** (iteration - i)), dtype="int") / (
            2 ** (iteration - i) - 1
        )
        hcce_images[:, :, i] = temp1
        temp1 = np.array(code_img[1] % (2 ** (iteration - i)), dtype="int") / (
            2 ** (iteration - i) - 1
        )
        hcce_images[:, :, i + iteration] = temp1
        temp1 = np.array(code_img[2] % (2 ** (iteration - i)), dtype="int") / (
            2 ** (iteration - i) - 1
        )
        hcce_images[:, :, i + iteration * 2] = temp1
    check_hcce_images = hcce_images.copy()
    k_ = iteration
    for i in range(k_ - 1):
        temp = hcce_images[:, :, i + 1].copy()
        temp[hcce_images[:, :, i] >= 0.5] = -temp[hcce_images[:, :, i] >= 0.5] + 1
        check_hcce_images[:, :, i + 1] = temp
    for i in range(k_ - 1):
        temp = hcce_images[:, :, i + 1 + k_].copy()
        temp[hcce_images[:, :, i + k_] >= 0.5] = -temp[hcce_images[:, :, i + k_] >= 0.5] + 1
        check_hcce_images[:, :, i + k_ + 1] = temp
    for i in range(k_ - 1):
        temp = hcce_images[:, :, i + 1 + k_ * 2].copy()
        temp[hcce_images[:, :, i + k_ * 2] >= 0.5] = (
            -temp[hcce_images[:, :, i + k_ * 2] >= 0.5] + 1
        )
        check_hcce_images[:, :, i + k_ * 2 + 1] = temp
    return check_hcce_images


def test_hcce_encoding_matches_hccepose_reference():
    rng = np.random.default_rng(0)
    code_img = rng.integers(0, 256, size=(13, 17, 3), dtype=np.uint8)
    ours = hcce_encode_numpy(code_img)
    ref = _reference_hcce_encode(code_img)
    assert np.allclose(ours, ref)


def test_hcce_decode_round_trip_after_threshold():
    rng = np.random.default_rng(1)
    code_img = rng.integers(0, 256, size=(9, 8, 3), dtype=np.uint8)
    enc = hcce_encode_numpy(code_img)
    bits = (torch.from_numpy(enc)[None] >= 0.5).float()
    dec = hcce_decode_torch(bits).round().cpu().numpy()[0].astype(np.uint8)
    assert np.array_equal(dec, code_img)


def test_normalized_xyz_to_hcce_matches_uint8_encoding():
    xyz = torch.tensor(
        [
            [[[-1.0, 0.0, 1.0], [0.5, -0.5, 0.25]]],
        ],
        dtype=torch.float32,
    )
    code = torch.round(((xyz + 1.0) / 2.0).clamp(0, 1) * 255.0).byte().numpy()[0]
    ref = torch.from_numpy(hcce_encode_numpy(code)).float()
    ours = normalized_xyz_to_hcce(xyz[0])
    assert torch.allclose(ours, ref)


def test_hcce_first_level_matches_original_cse_quantized_continuous_xyz():
    xyz = torch.rand(19, 23, 3) * 2.0 - 1.0
    hcce = normalized_xyz_to_hcce(xyz, iteration=8)
    first_level = hcce[..., [0, 8, 16]]
    cse_quantized = torch.round(((xyz + 1.0) * 0.5).clamp(0, 1) * 255.0) / 255.0
    assert torch.allclose(first_level, cse_quantized)


def test_action_and_wrist_heads_output_shapes():
    x = torch.randn(5, 32)
    action_head = InstrumentActionHead(32, n_actions=3, hidden_dim=64, n_iter=2, dropout=0.0)
    wrist_head = InstrumentWristPoseHead(32, hidden_dim=64, n_iter=2, dropout=0.0)
    action = action_head(x)
    quat, trans = wrist_head(x)
    assert action.shape == (5, 3)
    assert quat.shape == (5, 4)
    assert trans.shape == (5, 3)
    assert torch.allclose(quat.norm(dim=1), torch.ones(5), atol=1e-5)


def test_pose_loss_zero_for_supervised_targets():
    args = SimpleNamespace(
        alpha_action_l1=1.0,
        alpha_wrist_quat_l1=1.0,
        alpha_wrist_trans_l1=1.0,
    )
    loss_fn = LossInstrumentHCCEPose(args)
    y_hat = {
        "action_pred": torch.tensor([[0.1, 0.2, 0.3], [1.0, 2.0, 3.0]]),
        "wrist_quat_pred": torch.tensor([[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]]),
        "wrist_trans_pred": torch.tensor([[0.4, 0.5, 0.6], [4.0, 5.0, 6.0]]),
    }
    y = {
        "has_pose_inst": torch.tensor([True, True]),
        "action_gt_inst": torch.tensor([[0.1, 0.2, 0.3], [1.0, 2.0, 3.0]]),
        "wrist_quat_gt_inst": torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
        "wrist_trans_gt_inst": torch.tensor([[0.4, 0.5, 0.6], [4.0, 5.0, 6.0]]),
    }
    total, metrics = loss_fn._compute_pose_loss(y_hat, y)
    assert torch.allclose(total, torch.tensor(0.0))
    assert torch.allclose(metrics["action_l1"], torch.tensor(0.0))
    assert torch.allclose(metrics["wrist_quat_l1"], torch.tensor(0.0))
    assert torch.allclose(metrics["wrist_trans_l1"], torch.tensor(0.0))


def test_wrist_pose_symmetry_canonicalization_mapping():
    pose = {
        "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
        "trans": torch.tensor([0.01, -0.02, 0.3]),
        "alpha": torch.tensor([0.31]),
        "theta_l": torch.tensor([0.11]),
        "theta_r": torch.tensor([0.27]),
    }
    out = canonicalize_pose_symmetry(pose, eps=0.08, enabled=True)
    R = quat_to_matrix(pose["rot"])
    R_out = quat_to_matrix(out["rot"])
    assert bool(out["pose_sym_flipped"].item())
    assert torch.allclose(R_out, R @ SX_180, atol=1e-5)
    assert torch.allclose(out["trans"], pose["trans"])
    assert torch.allclose(out["alpha"], -pose["alpha"])
    assert torch.allclose(out["theta_l"], pose["theta_r"])
    assert torch.allclose(out["theta_r"], pose["theta_l"])
    assert out["rot"][0] >= 0
    assert torch.allclose(out["rot"].norm(), torch.tensor(1.0), atol=1e-6)


def test_wrist_pose_symmetry_fk_equivalence_matrices():
    pose = {
        "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
        "trans": torch.tensor([0.02, 0.01, 0.42]),
        "alpha": torch.tensor([0.37]),
        "theta_l": torch.tensor([0.12]),
        "theta_r": torch.tensor([0.24]),
    }
    out = canonicalize_pose_symmetry(pose, eps=0.08, enabled=True)
    R0 = quat_to_matrix(pose["rot"]).numpy()
    R1 = quat_to_matrix(out["rot"]).numpy()
    t = pose["trans"].numpy()
    S4 = np.diag([1.0, -1.0, -1.0, 1.0])
    fk0 = fk_matrices(
        R0,
        t,
        float(pose["alpha"][0]),
        float(pose["theta_l"][0]),
        float(pose["theta_r"][0]),
    )
    fk1 = fk_matrices(
        R1,
        t,
        float(out["alpha"][0]),
        float(out["theta_l"][0]),
        float(out["theta_r"][0]),
    )
    assert np.allclose(fk1["wrist"], fk0["wrist"] @ S4, atol=1e-8)
    assert np.allclose(fk1["shaft"], fk0["shaft"] @ S4, atol=1e-8)
    assert np.allclose(fk1["l_gripper"], fk0["r_gripper"] @ S4, atol=1e-8)
    assert np.allclose(fk1["r_gripper"], fk0["l_gripper"] @ S4, atol=1e-8)
