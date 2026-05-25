import numpy as np
import torch


def hcce_encode_numpy(code_img, iteration=8):
    """
    HCCE front-surface encoding aligned with HCCEPose.bop_loader.hcce_encode.

    Args:
        code_img: uint-like image [H, W, 3], values in [0, 255].
        iteration: number of hierarchy bits per coordinate axis.

    Returns:
        [H, W, 3 * iteration] float array.
    """
    code_img = code_img.copy().astype(np.int64)
    code_img = [code_img[:, :, 0], code_img[:, :, 1], code_img[:, :, 2]]
    hcce_images = np.zeros(
        (code_img[0].shape[0], code_img[0].shape[1], iteration * 3),
        dtype=np.float32,
    )
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
        temp[hcce_images[:, :, i + k_] >= 0.5] = (
            -temp[hcce_images[:, :, i + k_] >= 0.5] + 1
        )
        check_hcce_images[:, :, i + k_ + 1] = temp
    for i in range(k_ - 1):
        temp = hcce_images[:, :, i + 1 + k_ * 2].copy()
        temp[hcce_images[:, :, i + k_ * 2] >= 0.5] = (
            -temp[hcce_images[:, :, i + k_ * 2] >= 0.5] + 1
        )
        check_hcce_images[:, :, i + k_ * 2 + 1] = temp

    return check_hcce_images


def hcce_decode_torch(class_code_images):
    """
    Torch HCCE decode aligned with HCCEPose_BF_Net.hcce_decode.

    Args:
        class_code_images: [B, H, W, C] with C divisible by 3.

    Returns:
        [B, H, W, 3] decoded scalar codes in [0, 255] for 8-bit inputs.
    """
    class_base = 2
    _, _, _, channels = class_code_images.shape
    codes_length = channels // 3
    class_id_image = torch.zeros_like(class_code_images[..., :3])
    powers = torch.pow(
        torch.tensor(class_base, device=class_code_images.device, dtype=torch.float32),
        torch.arange(codes_length - 1, -1, -1, device=class_code_images.device),
    )
    for c in range(3):
        start_idx = c * codes_length
        end_idx = start_idx + codes_length
        codes = class_code_images[..., start_idx:end_idx]
        diffs = torch.zeros_like(codes)
        diffs[..., 0] = codes[..., 0]
        for k in range(1, codes_length):
            diffs[..., k] = torch.abs(codes[..., k] - diffs[..., k - 1])
        class_id_image[..., c] = torch.sum(diffs * powers, dim=-1)
    return class_id_image


def normalized_xyz_to_hcce(xyz, iteration=8, coord_min=-1.0, coord_max=1.0):
    """
    Convert normalized visible canonical xyz to front HCCE bits.

    Args:
        xyz: torch tensor [..., 3].

    Returns:
        torch tensor [..., 3 * iteration] with HCCE values in [0, 1].
    """
    denom = float(coord_max - coord_min)
    if denom <= 0:
        raise ValueError("coord_max must be greater than coord_min")
    code = ((xyz - coord_min) / denom).clamp(0.0, 1.0) * 255.0
    code = torch.round(code).to(torch.long)

    channels = []
    for c in range(3):
        axis = code[..., c]
        axis_codes = []
        for i in range(iteration):
            mod = 2 ** (iteration - i)
            temp = (axis % mod).to(torch.float32) / float(mod - 1)
            axis_codes.append(temp)
        axis_codes = torch.stack(axis_codes, dim=-1)
        check = axis_codes.clone()
        for i in range(iteration - 1):
            temp = axis_codes[..., i + 1].clone()
            temp = torch.where(axis_codes[..., i] >= 0.5, -temp + 1.0, temp)
            check[..., i + 1] = temp
        channels.append(check)
    return torch.cat(channels, dim=-1)
