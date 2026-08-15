import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROBOPEPP_ROOT = Path(__file__).resolve().parent
MULTIHMR_ROOT = Path(__file__).resolve().parents[2]
if str(ROBOPEPP_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBOPEPP_ROOT))
if str(MULTIHMR_ROOT) not in sys.path:
    sys.path.insert(0, str(MULTIHMR_ROOT))

from instrument_geometry import instrument_keypoints_camera_torch, project_points_torch  # noqa: E402
from loss_instrument import dice_loss  # noqa: E402
from loss_hcce_crop import focal_heatmap_loss, heatmap_argmax  # noqa: E402


def _part_target_to_dense_order(part_mask):
    target = torch.full_like(part_mask.long(), -1)
    target[part_mask == 2] = 0  # wrist
    target[part_mask == 1] = 1  # gripper
    target[part_mask == 3] = 2  # shaft
    return target


def compute_crop_surfemb_losses(out, y, args, model=None):
    inst_gt = y["inst_mask"].float()
    inst_loss_dice = dice_loss(torch.sigmoid(out["inst_mask_logits"]), inst_gt)
    inst_loss_bce = F.binary_cross_entropy_with_logits(out["inst_mask_logits"], inst_gt)

    part_target = _part_target_to_dense_order(y["part_mask"])
    part_valid = (part_target >= 0) & (inst_gt > 0.5)
    if part_valid.any():
        part_logits_flat = out["part_mask_logits"].permute(0, 2, 3, 1)[part_valid]
        part_target_flat = part_target[part_valid]
        part_ce = F.cross_entropy(part_logits_flat, part_target_flat)
        part_acc = (part_logits_flat.detach().argmax(dim=1) == part_target_flat).float().mean()
    else:
        part_ce = torch.tensor(0.0, device=inst_gt.device)
        part_acc = torch.tensor(0.0, device=inst_gt.device)

    surfemb_loss, surfemb_mask_bce, surfemb_nce, surfemb_px = _compute_surfemb_loss(out, y, args, model=model)
    heatmap_loss = focal_heatmap_loss(out["keypoint_heatmaps"], y["heatmaps"].float())

    pred_quat = F.normalize(out["wrist_quat_pred"], p=2, dim=1)
    gt_quat = F.normalize(y["wrist_quat"], p=2, dim=1)
    sign = torch.where((pred_quat * gt_quat).sum(dim=1, keepdim=True) < 0.0, -1.0, 1.0)
    quat_l1 = F.l1_loss(pred_quat, gt_quat * sign)
    action_l1 = F.l1_loss(out["action_pred"], y["action"])
    trans_l1 = F.l1_loss(out["wrist_trans_pred"], y["wrist_trans"])

    kp_pred = heatmap_argmax(out["keypoint_heatmaps"])
    valid = y["keypoints_valid"].bool()
    if valid.any():
        kp_err = torch.linalg.norm(kp_pred[valid] - y["keypoints_crop"][valid], dim=-1).mean()
    else:
        kp_err = torch.tensor(0.0, device=inst_gt.device)

    keypoints_3d_pred = instrument_keypoints_camera_torch(
        pred_quat,
        out["wrist_trans_pred"],
        out["action_pred"],
    )
    kp3d_l1 = F.l1_loss(keypoints_3d_pred, y["keypoints_3d_cam"])
    uv_pred = project_points_torch(keypoints_3d_pred, y["K"])
    valid_uv = (
        valid
        & torch.isfinite(uv_pred).all(dim=-1)
        & torch.isfinite(keypoints_3d_pred).all(dim=-1)
        & (keypoints_3d_pred[..., 2] > 1e-4)
    )
    if valid_uv.any():
        uv_loss = F.smooth_l1_loss(uv_pred[valid_uv] / 224.0, y["keypoints_crop"][valid_uv] / 224.0)
        uv_err = torch.linalg.norm(uv_pred[valid_uv] - y["keypoints_crop"][valid_uv], dim=-1).mean()
    else:
        uv_loss = torch.tensor(0.0, device=inst_gt.device)
        uv_err = torch.tensor(0.0, device=inst_gt.device)

    total = (
        args.alpha_dice * inst_loss_dice
        + args.alpha_bce_mask * inst_loss_bce
        + args.alpha_part * part_ce
        + args.alpha_surfemb * surfemb_loss
        + args.alpha_heatmap * heatmap_loss
        + args.alpha_action_l1 * action_l1
        + args.alpha_wrist_quat_l1 * quat_l1
        + args.alpha_wrist_trans_l1 * trans_l1
        + args.alpha_keypoint_2d * uv_loss
        + args.alpha_keypoint_3d * kp3d_l1
    )
    metrics = {
        "total": total.detach(),
        "dice": inst_loss_dice.detach(),
        "bce_mask": inst_loss_bce.detach(),
        "part_ce": part_ce.detach(),
        "part_acc": part_acc.detach(),
        "surfemb": surfemb_loss.detach(),
        "surfemb_mask_bce": surfemb_mask_bce.detach(),
        "surfemb_nce": surfemb_nce.detach(),
        "surfemb_px": surfemb_px.detach(),
        "heatmap": heatmap_loss.detach(),
        "kp_err_px": kp_err.detach(),
        "action_l1": action_l1.detach(),
        "wrist_quat_l1": quat_l1.detach(),
        "wrist_trans_l1": trans_l1.detach(),
        "uv_loss": uv_loss.detach(),
        "uv_err_px": uv_err.detach(),
        "kp3d_l1": kp3d_l1.detach(),
    }
    if "surfemb_kpt_proj_resid_px" in y:
        metrics["kpt_proj_resid_px"] = torch.nan_to_num(
            y["surfemb_kpt_proj_resid_px"].float().mean(), nan=0.0, posinf=0.0, neginf=0.0
        ).detach()
    return total, metrics


def _compute_surfemb_loss(out, y, args, model=None):
    key_mlp = getattr(model, "surface_key_mlp", None) if model is not None else None
    if key_mlp is None:
        key_mlp = out.get("surface_key_mlp")
    if key_mlp is None:
        raise RuntimeError("SurfEmb loss requires model.surface_key_mlp or out['surface_key_mlp'].")

    coord_img = y["coord_img"].float()
    mask = coord_img[..., 3] == 1.0
    with torch.amp.autocast(out["inst_mask_logits"].device.type, enabled=False):
        mask_prob = torch.sigmoid(out["inst_mask_logits"].float())
        surfemb_mask_bce = F.binary_cross_entropy(mask_prob, mask.type_as(mask_prob).float())

    queries = out["surfemb_queries"].float()
    coords_neg = y["surfemb_surface_samples"].float()
    mask_samples = y["surfemb_mask_samples"].long()
    B, emb_dim, H, W = queries.shape
    yx = mask_samples.clamp_min(0)
    y_idx = yx[..., 0].clamp_max(H - 1)
    x_idx = yx[..., 1].clamp_max(W - 1)
    batch_idx = torch.arange(B, device=queries.device).view(B, 1)

    queries_pos = queries[batch_idx, :, y_idx, x_idx]
    coords_pos = coord_img[batch_idx, y_idx, x_idx, :3]
    key_noise = float(getattr(args, "surfemb_key_noise", 1e-3))
    coords_pos = coords_pos + torch.randn_like(coords_pos) * key_noise
    keys_pos = key_mlp(coords_pos)
    sim_pos = (queries_pos * keys_pos).sum(dim=-1, keepdim=True)

    coords_neg = coords_neg + torch.randn_like(coords_neg) * key_noise
    keys_neg = key_mlp(coords_neg)
    sim_neg = queries_pos @ keys_neg.permute(0, 2, 1)

    logits = torch.cat((sim_pos, sim_neg), dim=-1).permute(0, 2, 1)
    target = torch.zeros(B, mask_samples.shape[1], device=queries.device, dtype=torch.long)
    nce_loss = F.cross_entropy(logits, target)
    surfemb_loss = surfemb_mask_bce + nce_loss
    px_count = mask.sum().float()
    return (
        torch.nan_to_num(surfemb_loss, nan=0.0, posinf=0.0, neginf=0.0),
        torch.nan_to_num(surfemb_mask_bce, nan=0.0, posinf=0.0, neginf=0.0),
        torch.nan_to_num(nce_loss, nan=0.0, posinf=0.0, neginf=0.0),
        px_count.detach(),
    )
