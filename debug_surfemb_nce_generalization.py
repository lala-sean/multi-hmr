#!/usr/bin/env python3
"""Decompose SurfEmb train/test NCE into ranking and feature-scale terms."""

import argparse
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

import train_surfemb_resnet_wrist_only_lnd_prerefine as prerefine


ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINTS = (
    ROOT / "logs/surfemb_wristonly_lnd_prerefine_fullsurfneg_eval_20260807/checkpoints/best2k_snapshot.pt",
    ROOT / "logs/surfemb_wristonly_lnd_prerefine_fullsurfneg_eval_20260808_iter28k/checkpoints/last28k_snapshot.pt",
    ROOT / "logs/surfemb_wristonly_lnd_prerefine_fullsurfneg_eval_20260809_iter54k/checkpoints/last54k_snapshot.pt",
)


class FrameFilter(Dataset):
    def __init__(self, dataset, excluded):
        self.dataset = dataset
        excluded = {int(v) for v in excluded}
        self.indices = [
            i for i, sample in enumerate(dataset.base_dataset.samples)
            if int(sample[0]) not in excluded
        ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataset(args, split):
    base_train = prerefine._train._base_train

    def wrist_factory(*factory_args, **factory_kwargs):
        return prerefine._wrist_dataset.SurfEmbWristOnlyCropDataset(
            *factory_args,
            min_wrist_pixels=args.wrist_min_visible_pixels,
            negative_visible_only=False,
            fallback_to_other_sample=False,
            **factory_kwargs,
        )

    old_factory = base_train.SurfEmbKeypointCropDataset
    base_train.SurfEmbKeypointCropDataset = wrist_factory
    try:
        dataset = base_train.make_lnd_dataset(
            args,
            split=split,
            training=False,
            use_memory_pose=split == "TRAIN",
            subsample=1,
        )
    finally:
        base_train.SurfEmbKeypointCropDataset = old_factory
    excluded = prerefine.TRAIN_EXCLUDED_FRAME_IDS if split == "TRAIN" else prerefine.VAL_EXCLUDED_FRAME_IDS
    return FrameFilter(dataset, excluded)


def load_models(checkpoint_paths, device):
    models = []
    checkpoint_args = None
    for path in checkpoint_paths:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint_args is None:
            checkpoint_args = SimpleNamespace(**ckpt["args"])
        model = prerefine._train.SurfEmbResNetCropModel(
            img_size=224,
            surfemb_emb_dim=ckpt["args"]["surfemb_emb_dim"],
            surfemb_mlp_hidden_features=ckpt["args"]["surfemb_mlp_hidden_features"],
            surfemb_mlp_hidden_layers=ckpt["args"]["surfemb_mlp_hidden_layers"],
            resnet_feat_preultimate=ckpt["args"]["resnet_feat_preultimate"],
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        model.eval()
        models.append((path.stem, model))
    return checkpoint_args, models


class MetricAccumulator:
    KEYS = (
        "raw_nce", "cos_nce_t1", "cos_nce_t007", "raw_top1", "cos_top1",
        "raw_margin", "cos_margin", "query_norm", "positive_key_norm",
        "negative_key_norm", "raw_positive", "raw_hard_negative",
        "cos_positive", "cos_hard_negative",
    )

    def __init__(self):
        self.sums = {key: 0.0 for key in self.KEYS}
        self.count = 0
        self.frame_raw_nce = []
        self.frame_cos_nce = []

    def update(self, values, batch_size):
        for key in self.KEYS:
            self.sums[key] += float(values[key]) * batch_size
        self.count += batch_size
        self.frame_raw_nce.extend(values["frame_raw_nce"])
        self.frame_cos_nce.extend(values["frame_cos_nce"])

    def result(self):
        result = {key: value / max(1, self.count) for key, value in self.sums.items()}
        for name, values in (("raw_nce", self.frame_raw_nce), ("cos_nce_t007", self.frame_cos_nce)):
            arr = np.asarray(values, dtype=np.float64)
            result[f"frame_{name}_p50"] = float(np.percentile(arr, 50))
            result[f"frame_{name}_p90"] = float(np.percentile(arr, 90))
            result[f"frame_{name}_p99"] = float(np.percentile(arr, 99))
        result["frames"] = self.count
        return result


def batch_metrics(model, x, y, device, key_noise):
    x = x.to(device, non_blocking=True)
    coords_pos = y["surfemb_coords_pos"].to(device).float()
    coords_neg = y["surfemb_surface_samples"].to(device).float()
    coords = torch.cat((coords_pos, coords_neg), dim=1)
    if key_noise:
        coords = coords + torch.randn_like(coords) * key_noise

    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model(x, surfemb_key_coords=coords)
    queries = out["surfemb_queries"].float()
    keys = out["surfemb_keys"].float()
    yx = y["surfemb_mask_samples"].to(device).long()
    batch_idx = torch.arange(queries.shape[0], device=device)[:, None]
    q = queries[batch_idx, :, yx[..., 0], yx[..., 1]]
    kp = keys[:, : coords_pos.shape[1]]
    kn = keys[:, coords_pos.shape[1] :]

    raw_pos = (q * kp).sum(-1)
    raw_neg = q @ kn.transpose(1, 2)
    raw_logits = torch.cat((raw_pos[..., None], raw_neg), dim=-1)
    raw_loss = -F.log_softmax(raw_logits, dim=-1)[..., 0]

    qn = F.normalize(q, dim=-1)
    kpn = F.normalize(kp, dim=-1)
    knn = F.normalize(kn, dim=-1)
    cos_pos = (qn * kpn).sum(-1)
    cos_neg = qn @ knn.transpose(1, 2)
    cos_logits = torch.cat((cos_pos[..., None], cos_neg), dim=-1)
    cos_loss_t1 = -F.log_softmax(cos_logits, dim=-1)[..., 0]
    cos_loss_t007 = -F.log_softmax(cos_logits / 0.07, dim=-1)[..., 0]

    raw_hard = raw_neg.max(-1).values
    cos_hard = cos_neg.max(-1).values
    values = {
        "raw_nce": raw_loss.mean().item(),
        "cos_nce_t1": cos_loss_t1.mean().item(),
        "cos_nce_t007": cos_loss_t007.mean().item(),
        "raw_top1": (raw_pos > raw_hard).float().mean().item(),
        "cos_top1": (cos_pos > cos_hard).float().mean().item(),
        "raw_margin": (raw_pos - raw_hard).mean().item(),
        "cos_margin": (cos_pos - cos_hard).mean().item(),
        "query_norm": q.norm(dim=-1).mean().item(),
        "positive_key_norm": kp.norm(dim=-1).mean().item(),
        "negative_key_norm": kn.norm(dim=-1).mean().item(),
        "raw_positive": raw_pos.mean().item(),
        "raw_hard_negative": raw_hard.mean().item(),
        "cos_positive": cos_pos.mean().item(),
        "cos_hard_negative": cos_hard.mean().item(),
        "frame_raw_nce": raw_loss.mean(1).cpu().tolist(),
        "frame_cos_nce": cos_loss_t007.mean(1).cpu().tolist(),
    }
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", type=Path, default=list(DEFAULT_CHECKPOINTS))
    parser.add_argument("--splits", nargs="+", choices=("TRAIN", "TEST"), default=("TRAIN", "TEST"))
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--key_noise", type=float, default=1e-3)
    parser.add_argument("--output", type=Path, default=ROOT / "eval_outputs/surfemb_nce_debug/nce_decomposition.json")
    cli = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This diagnostic requires CUDA for the ResNet and OpenGL rasterizer.")

    seed_everything(cli.seed)
    device = torch.device("cuda:0")
    ckpt_args, models = load_models(cli.checkpoints, device)
    results = {}
    for split in cli.splits:
        seed_everything(cli.seed + (0 if split == "TRAIN" else 1))
        dataset = build_dataset(ckpt_args, split)
        if cli.max_samples > 0:
            dataset = Subset(dataset, range(min(cli.max_samples, len(dataset))))
        loader = DataLoader(
            dataset,
            batch_size=cli.batch_size,
            shuffle=False,
            num_workers=cli.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=prerefine._train.collate_fn_surfemb_keypoint_crop,
        )
        split_acc = {name: MetricAccumulator() for name, _ in models}
        for batch_idx, (x, y) in enumerate(loader):
            for model_idx, (name, model) in enumerate(models):
                torch.manual_seed(cli.seed + batch_idx)
                values = batch_metrics(model, x, y, device, cli.key_noise)
                split_acc[name].update(values, x.shape[0])
            if (batch_idx + 1) % 10 == 0:
                print(f"{split}: {min(len(dataset), (batch_idx + 1) * cli.batch_size)}/{len(dataset)}", flush=True)
        results[split] = {name: acc.result() for name, acc in split_acc.items()}
        print(json.dumps({split: results[split]}, indent=2), flush=True)

    cli.output.parent.mkdir(parents=True, exist_ok=True)
    cli.output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Wrote {cli.output}", flush=True)


if __name__ == "__main__":
    main()
