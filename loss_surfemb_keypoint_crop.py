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

from loss_hcce_crop import focal_heatmap_loss, heatmap_argmax  # noqa: E402
from instrument_geometry import instrument_keypoints_camera_torch, project_points_torch  # noqa: E402


def compute_surfemb_keypoint_losses(out, y, args, model=None):
    surfemb_loss, mask_loss, nce_loss, px_count = _compute_surfemb_loss(out, y, args, model=model)
    heatmap_loss = focal_heatmap_loss(out["keypoint_heatmaps"], y["heatmaps"].float())

    kp_pred = heatmap_argmax(out["keypoint_heatmaps"])
    valid = y["keypoints_valid"].bool()
    if valid.any():
        kp_err = torch.linalg.norm(kp_pred[valid] - y["keypoints_crop"][valid], dim=-1).mean()
    else:
        kp_err = torch.tensor(0.0, device=out["keypoint_heatmaps"].device)

    pred_quat = F.normalize(out["wrist_quat_pred"], p=2, dim=1)
    gt_quat = F.normalize(y["wrist_quat"], p=2, dim=1)
    sign = torch.where((pred_quat * gt_quat).sum(dim=1, keepdim=True) < 0.0, -1.0, 1.0)
    quat_l1 = F.l1_loss(pred_quat, gt_quat * sign)
    action_l1 = F.l1_loss(out["action_pred"], y["action"])
    trans_l1 = F.l1_loss(out["wrist_trans_pred"], y["wrist_trans"])

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
        uv_loss = torch.tensor(0.0, device=out["keypoint_heatmaps"].device)
        uv_err = torch.tensor(0.0, device=out["keypoint_heatmaps"].device)

    total = (
        args.alpha_surfemb * surfemb_loss
        + args.alpha_heatmap * heatmap_loss
        + args.alpha_action_l1 * action_l1
        + args.alpha_wrist_quat_l1 * quat_l1
        + args.alpha_wrist_trans_l1 * trans_l1
        + args.alpha_keypoint_2d * uv_loss
        + args.alpha_keypoint_3d * kp3d_l1
    )
    metrics = {
        "total": total.detach(),
        "surfemb": surfemb_loss.detach(),
        "surfemb_mask_bce": mask_loss.detach(),
        "surfemb_nce": nce_loss.detach(),
        "surfemb_px": px_count.detach(),
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
    if "has_cse" in y:
        metrics["surfemb_valid_frac"] = y["has_cse"].float().mean().detach()
    if "surfemb_render_iou" in y:
        metrics["surfemb_render_iou"] = torch.nan_to_num(
            y["surfemb_render_iou"].float().mean(), nan=0.0, posinf=0.0, neginf=0.0
        ).detach()
    if "surfemb_part_ratio_exact" in y:
        metrics["surfemb_part_ratio_exact_frac"] = y["surfemb_part_ratio_exact"].float().mean().detach()
    return total, metrics


def _compute_surfemb_loss(out, y, args, model=None):
    inst_mask = y["inst_mask"].float()
    coords_pos = y["surfemb_coords_pos"].float()
    coords_neg = y["surfemb_surface_samples"].float()
    mask_samples = y["surfemb_mask_samples"].long()
    queries = out["surfemb_queries"].float()
    B, _, H, W = queries.shape
    if coords_neg.shape[1] != int(args.surfemb_n_neg):
        raise RuntimeError(f"coords_neg n={coords_neg.shape[1]} does not match surfemb_n_neg={args.surfemb_n_neg}")
    if mask_samples.shape[1] != int(args.surfemb_n_pos):
        raise RuntimeError(f"mask_samples n={mask_samples.shape[1]} does not match surfemb_n_pos={args.surfemb_n_pos}")

    with torch.amp.autocast(out["inst_mask_logits"].device.type, enabled=False):
        mask_prob = torch.sigmoid(out["inst_mask_logits"].float())
        mask_loss = F.binary_cross_entropy(mask_prob, inst_mask.float())

    valid_cse = y.get("has_cse")
    if valid_cse is None:
        valid_cse = torch.ones((B,), dtype=torch.bool, device=queries.device)
    else:
        valid_cse = valid_cse.to(device=queries.device).bool().view(B)
    if not valid_cse.any():
        nce_loss = queries.sum() * 0.0
        surfemb_loss = mask_loss + nce_loss
        px_count = inst_mask.sum().float()
        return (
            torch.nan_to_num(surfemb_loss, nan=0.0, posinf=0.0, neginf=0.0),
            torch.nan_to_num(mask_loss, nan=0.0, posinf=0.0, neginf=0.0),
            torch.nan_to_num(nce_loss, nan=0.0, posinf=0.0, neginf=0.0),
            px_count.detach(),
        )

    queries = queries[valid_cse]
    coords_pos = coords_pos[valid_cse]
    coords_neg = coords_neg[valid_cse]
    mask_samples = mask_samples[valid_cse]
    B = queries.shape[0]

    yx = mask_samples.clamp_min(0)
    y_idx = yx[..., 0].clamp_max(H - 1)
    x_idx = yx[..., 1].clamp_max(W - 1)
    batch_idx = torch.arange(B, device=queries.device).view(B, 1)

    # Same contrastive target as SurfEmb: for every visible surface query,
    # class 0 is its true 3D coordinate key and classes 1..N are surface negatives.
    queries_pos = queries[batch_idx, :, y_idx, x_idx]
    keys = out.get("surfemb_keys")
    if keys is None:
        key_mlp = getattr(model, "surface_key_mlp", None) if model is not None else None
        if key_mlp is None:
            raise RuntimeError("SurfEmb loss requires out['surfemb_keys'] or model.surface_key_mlp.")
        key_noise = float(getattr(args, "surfemb_key_noise", 1e-3))
        coords_pos = coords_pos + torch.randn_like(coords_pos) * key_noise
        coords_neg = coords_neg + torch.randn_like(coords_neg) * key_noise
        keys = key_mlp(torch.cat((coords_pos, coords_neg), dim=1))
    else:
        keys = keys[valid_cse]
    keys_pos = keys[:, : coords_pos.shape[1]]
    keys_neg = keys[:, coords_pos.shape[1] :]
    sim_pos = (queries_pos * keys_pos).sum(dim=-1, keepdim=True)

    sim_neg = queries_pos @ keys_neg.permute(0, 2, 1)

    logits = torch.cat((sim_pos, sim_neg), dim=-1).permute(0, 2, 1)
    target = torch.zeros(B, int(args.surfemb_n_pos), device=queries.device, dtype=torch.long)
    nce_loss = F.cross_entropy(logits, target)
    surfemb_loss = mask_loss + nce_loss
    px_count = inst_mask.sum().float()
    return (
        torch.nan_to_num(surfemb_loss, nan=0.0, posinf=0.0, neginf=0.0),
        torch.nan_to_num(mask_loss, nan=0.0, posinf=0.0, neginf=0.0),
        torch.nan_to_num(nce_loss, nan=0.0, posinf=0.0, neginf=0.0),
        px_count.detach(),
    )
