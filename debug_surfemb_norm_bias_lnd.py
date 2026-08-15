#!/usr/bin/env python3
"""Measure whether raw key norms pull LND TEST correspondences off the GT point."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import debug_surfemb_nce_generalization as nce_debug


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "eval_outputs/surfemb_norm_bias_lnd/norm_bias.json"


class Accumulator:
    def __init__(self):
        self.count = 0
        self.sums = {}
        self.raw_errors_mm = []
        self.cos_errors_mm = []

    def add(self, name, value):
        self.sums[name] = self.sums.get(name, 0.0) + float(value)

    def update(self, metrics):
        n = int(metrics.pop("count"))
        self.count += n
        self.raw_errors_mm.append(metrics.pop("raw_errors_mm"))
        self.cos_errors_mm.append(metrics.pop("cos_errors_mm"))
        for name, value in metrics.items():
            self.add(name, value)

    def result(self):
        out = {name: value / max(1, self.count) for name, value in self.sums.items()}
        for prefix, chunks in (("raw", self.raw_errors_mm), ("cosine", self.cos_errors_mm)):
            values = np.concatenate(chunks).astype(np.float64)
            out[f"{prefix}_3d_error_mean_mm"] = float(values.mean())
            out[f"{prefix}_3d_error_median_mm"] = float(np.median(values))
            out[f"{prefix}_3d_error_p90_mm"] = float(np.percentile(values, 90))
        out["correspondences"] = self.count
        return out


@torch.inference_mode()
def analyze_batch(model, x, y, device, canon_scale_mm, key_noise):
    x = x.to(device, non_blocking=True)
    pos_coords = y["surfemb_coords_pos"].to(device).float()
    neg_coords = y["surfemb_surface_samples"].to(device).float()
    all_coords = torch.cat((pos_coords, neg_coords), dim=1)
    if key_noise > 0.0:
        all_coords = all_coords + torch.randn_like(all_coords) * key_noise
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        output = model(x, surfemb_key_coords=all_coords)
    queries = output["surfemb_queries"].float()
    keys = output["surfemb_keys"].float()
    yx = y["surfemb_mask_samples"].to(device).long()
    batch = torch.arange(len(x), device=device)[:, None]
    query = queries[batch, :, yx[..., 0], yx[..., 1]]
    n_pos = pos_coords.shape[1]
    pos_key = keys[:, :n_pos]
    neg_key = keys[:, n_pos:]

    raw_pos = (query * pos_key).sum(-1)
    raw_neg = query @ neg_key.transpose(1, 2)
    cos_query = F.normalize(query, dim=-1, eps=1e-6)
    cos_pos_key = F.normalize(pos_key, dim=-1, eps=1e-6)
    cos_neg_key = F.normalize(neg_key, dim=-1, eps=1e-6)
    cos_pos = (cos_query * cos_pos_key).sum(-1)
    cos_neg = cos_query @ cos_neg_key.transpose(1, 2)

    raw_index = torch.cat((raw_pos[..., None], raw_neg), dim=-1).argmax(-1)
    cos_index = torch.cat((cos_pos[..., None], cos_neg), dim=-1).argmax(-1)
    raw_wrong = raw_index != 0
    cos_wrong = cos_index != 0
    raw_neg_index = (raw_index - 1).clamp_min(0)
    cos_neg_index = (cos_index - 1).clamp_min(0)
    raw_selected_coord = neg_coords.gather(1, raw_neg_index[..., None].expand(-1, -1, 3))
    cos_selected_coord = neg_coords.gather(1, cos_neg_index[..., None].expand(-1, -1, 3))
    raw_selected_coord = torch.where(raw_wrong[..., None], raw_selected_coord, pos_coords)
    cos_selected_coord = torch.where(cos_wrong[..., None], cos_selected_coord, pos_coords)
    raw_error = (raw_selected_coord - pos_coords).norm(dim=-1) * canon_scale_mm
    cos_error = (cos_selected_coord - pos_coords).norm(dim=-1) * canon_scale_mm

    raw_selected_key_norm = neg_key.norm(dim=-1).gather(1, raw_neg_index)
    raw_selected_cos = cos_neg.gather(2, raw_neg_index[..., None]).squeeze(-1)
    positive_key_norm = pos_key.norm(dim=-1)
    norm_caused_inversion = raw_wrong & (raw_selected_cos < cos_pos) & (raw_selected_key_norm > positive_key_norm)
    n = int(raw_wrong.numel())
    return {
        "count": n,
        "raw_errors_mm": raw_error.cpu().numpy().reshape(-1),
        "cos_errors_mm": cos_error.cpu().numpy().reshape(-1),
        "raw_top1": int((~raw_wrong).sum()),
        "cosine_top1": int((~cos_wrong).sum()),
        "raw_wrong_cosine_correct": int((raw_wrong & ~cos_wrong).sum()),
        "raw_correct_cosine_wrong": int((~raw_wrong & cos_wrong).sum()),
        "cosine_reduces_3d_error": int((cos_error < raw_error).sum()),
        "cosine_increases_3d_error": int((cos_error > raw_error).sum()),
        "raw_error_selected_key_has_larger_norm": int((raw_wrong & (raw_selected_key_norm > positive_key_norm)).sum()),
        "norm_caused_inversion": int(norm_caused_inversion.sum()),
        "positive_key_norm_sum": float(positive_key_norm.sum()),
        "raw_selected_key_norm_sum": float(raw_selected_key_norm.sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", type=Path, default=list(nce_debug.DEFAULT_CHECKPOINTS))
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--key_noise", type=float, default=1e-3)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0")
    checkpoint_args, models = nce_debug.load_models(args.checkpoints, device)
    dataset = nce_debug.build_dataset(checkpoint_args, "TEST")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=nce_debug.prerefine._train.collate_fn_surfemb_keypoint_crop,
    )
    payload = np.load(checkpoint_args.surface_points_path, allow_pickle=True).item()
    canon_scale_mm = float(payload["canon_scale"]) * 1000.0
    accumulators = {name: Accumulator() for name, _ in models}
    for batch_index, (x, y) in enumerate(loader):
        for name, model in models:
            torch.manual_seed(args.seed + batch_index)
            metrics = analyze_batch(model, x, y, device, canon_scale_mm, args.key_noise)
            accumulators[name].update(metrics)
        if (batch_index + 1) % 10 == 0:
            print(f"processed {min(len(dataset), (batch_index + 1) * args.batch_size)}/{len(dataset)}", flush=True)
    results = {
        "dataset": "SurgRIPE-LND/TEST wrist-only",
        "candidate_keys_per_query": int(checkpoint_args.surfemb_n_neg) + 1,
        "canon_scale_mm": canon_scale_mm,
        "checkpoints": {name: accumulator.result() for name, accumulator in accumulators.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2), flush=True)
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
