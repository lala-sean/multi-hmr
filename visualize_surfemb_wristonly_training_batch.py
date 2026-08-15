#!/usr/bin/env python3
import argparse
import math
import random
from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

import train_surfemb_resnet_wrist_only_rarp_lnd as wrist_train
from instrument_opengl_renderer import InstrumentOpenGLDepthRenderer


ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = (
    ROOT
    / "logs/surfemb_resnet_wristonly_lnd_best_last_predmask_20260804"
    / "checkpoints/last_iter17000.pt"
)
DEFAULT_OUTPUT = ROOT / "logs/surfemb_resnet_wristonly_batch_correspondence_vis"


def original_surfemb_embedding_rgb(embedding, mask=None, demean=False):
    """Match SurfaceEmbeddingModel.get_emb_vis from the original SurfEmb repo."""
    embedding = embedding.float().clone()
    if demean is True:
        if mask is None or not bool(mask.any()):
            raise ValueError("demean=True requires a non-empty mask")
        demean = embedding[mask].reshape(-1, embedding.shape[-1]).mean(dim=0)
    if demean is not False:
        embedding = embedding - torch.as_tensor(demean, device=embedding.device, dtype=embedding.dtype)
    shape = embedding.shape[:-1]
    embedding = embedding.reshape(*shape, 3, -1).mean(dim=-1)
    if mask is not None:
        embedding[~mask] = 0.0
    embedding /= embedding.abs().max().clamp_min(1e-9)
    embedding.mul_(0.5).add_(0.5)
    return (embedding.clamp(0.0, 1.0) * 255.0).byte().cpu().numpy()


def load_model_and_args(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved_args = checkpoint.get("args", {})
    if not isinstance(saved_args, dict):
        saved_args = vars(saved_args)
    args = Namespace(**saved_args)
    model = wrist_train._train.SurfEmbResNetCropModel(
        img_size=int(saved_args.get("crop_size", 224)),
        surfemb_emb_dim=int(saved_args.get("surfemb_emb_dim", 12)),
        surfemb_mlp_hidden_features=int(saved_args.get("surfemb_mlp_hidden_features", 256)),
        surfemb_mlp_hidden_layers=int(saved_args.get("surfemb_mlp_hidden_layers", 2)),
        resnet_feat_preultimate=int(saved_args.get("resnet_feat_preultimate", 64)),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval().to(device)
    return model, args, int(checkpoint.get("iter", -1)), int(checkpoint.get("epoch", 0))


def build_training_loader(args, batch_size, seed, epoch):
    def dataset_factory(*factory_args, **factory_kwargs):
        return wrist_train._wrist_dataset.SurfEmbWristOnlyCropDataset(
            *factory_args,
            min_wrist_pixels=int(args.wrist_min_visible_pixels),
            **factory_kwargs,
        )

    wrist_train._train._base_train.SurfEmbKeypointCropDataset = dataset_factory
    datasets = wrist_train._train.make_train_datasets(args)
    dataset = ConcatDataset(datasets)
    wrist_train._train.set_epoch_recursive(dataset, int(epoch))
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
        collate_fn=wrist_train._train.collate_fn_surfemb_keypoint_crop,
        persistent_workers=False,
    )
    return loader, datasets


def pose_from_batch(batch, index):
    action = batch["action"][index].detach().cpu().numpy().reshape(3)
    return {
        "rot": batch["wrist_quat"][index].detach().cpu().numpy(),
        "trans": batch["wrist_trans"][index].detach().cpu().numpy(),
        "alpha": float(action[0]),
        "theta_l": float(action[1]),
        "theta_r": float(action[2]),
    }


@torch.inference_mode()
def encode_key_image(model, coord_image, mask, device, chunk_size=8192):
    ys, xs = np.where(mask)
    dense = torch.zeros((*mask.shape, model.surfemb_emb_dim), device=device, dtype=torch.float32)
    if len(ys) == 0:
        return dense
    coords = torch.from_numpy(coord_image[ys, xs]).to(device=device, dtype=torch.float32)
    keys = []
    for start in range(0, len(coords), int(chunk_size)):
        keys.append(model.surface_key_mlp(coords[start : start + int(chunk_size)]).float())
    dense[torch.from_numpy(ys).to(device), torch.from_numpy(xs).to(device)] = torch.cat(keys, dim=0)
    return dense


def sampled_embedding_image(embedding_hwd, yx, mask_shape, demean):
    mask = torch.zeros(mask_shape, dtype=torch.bool, device=embedding_hwd.device)
    mask[yx[:, 0], yx[:, 1]] = True
    sparse = torch.zeros_like(embedding_hwd)
    sparse[mask] = embedding_hwd[mask]
    image = original_surfemb_embedding_rgb(sparse, mask=mask, demean=demean)
    yx_np = yx.detach().cpu().numpy()
    source = image[yx_np[:, 0], yx_np[:, 1]].copy()
    for offset_y in (-1, 0, 1):
        for offset_x in (-1, 0, 1):
            yy = np.clip(yx_np[:, 0] + offset_y, 0, image.shape[0] - 1)
            xx = np.clip(yx_np[:, 1] + offset_x, 0, image.shape[1] - 1)
            image[yy, xx] = source
    return image


def label_panel(image, text):
    image = np.asarray(image, dtype=np.uint8)
    bar = np.full((26, image.shape[1], 3), 22, dtype=np.uint8)
    cv2.putText(bar, text, (7, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (240, 240, 240), 1, cv2.LINE_AA)
    return np.concatenate((bar, image), axis=0)


def denormalize_imagenet_tensor(image_chw):
    mean = torch.tensor((0.485, 0.456, 0.406), device=image_chw.device).view(3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=image_chw.device).view(3, 1, 1)
    image = (image_chw.float() * std + mean).clamp(0.0, 1.0)
    return (image.permute(1, 2, 0) * 255.0).byte().cpu().numpy()


def make_sample_tile(input_rgb, pred_rgb, gt_rgb, sample_label, column_labels):
    panels = []
    for image, column in zip((input_rgb, pred_rgb, gt_rgb), column_labels):
        panels.append(label_panel(image, f"{sample_label} | {column}"))
    return np.concatenate(panels, axis=1)


def contact_sheet(tiles, samples_per_row, gap=8):
    rows = []
    tile_h, tile_w = tiles[0].shape[:2]
    columns = max(1, int(samples_per_row))
    for start in range(0, len(tiles), columns):
        row_tiles = list(tiles[start : start + columns])
        while len(row_tiles) < columns:
            row_tiles.append(np.full((tile_h, tile_w, 3), 22, dtype=np.uint8))
        row = row_tiles[0]
        for tile in row_tiles[1:]:
            row = np.concatenate((row, np.full((tile_h, gap, 3), 22, dtype=np.uint8), tile), axis=1)
        rows.append(row)
    sheet = rows[0]
    for row in rows[1:]:
        sheet = np.concatenate((sheet, np.full((gap, sheet.shape[1], 3), 22, dtype=np.uint8), row), axis=0)
    return sheet


def save_rgb(path, image, quality=94):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError(f"Failed to save {path}")


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    checkpoint_path = Path(args.checkpoint).resolve()
    model, train_args, checkpoint_iter, checkpoint_epoch = load_model_and_args(checkpoint_path, device)
    loader, datasets = build_training_loader(
        train_args,
        batch_size=args.batch_size,
        seed=args.seed,
        epoch=checkpoint_epoch,
    )
    x, batch = next(iter(loader))
    x = x.to(device=device, non_blocking=True)
    with torch.amp.autocast(device.type, enabled=device.type == "cuda" and bool(args.amp), dtype=torch.bfloat16):
        output = model(x, batch["K"].to(device=device, non_blocking=True))
    queries = output["surfemb_queries"].float().permute(0, 2, 3, 1)
    mask_probability = torch.sigmoid(output["inst_mask_logits"].float())

    renderer = InstrumentOpenGLDepthRenderer(int(train_args.crop_size), int(train_args.crop_size))
    dense_tiles = []
    sampled_tiles = []
    manifest = []
    for index in range(len(x)):
        input_rgb = denormalize_imagenet_tensor(x[index])
        gt_mask = batch["inst_mask"][index].cpu().numpy() > 0.5
        K_crop = batch["K"][index].cpu().numpy().astype(np.float32)
        coord_image, effective_part, depth, raster_valid = renderer.render_canonical_coordinates(
            pose_from_batch(batch, index),
            K_crop,
            gt_mask.shape,
        )
        dense_mask = raster_valid & (effective_part == 2) & (depth > 1e-4) & gt_mask
        gt_dense = encode_key_image(model, coord_image, dense_mask, device)
        dense_mask_t = torch.from_numpy(dense_mask).to(device=device)

        pred_mask = mask_probability[index] > 0.5
        pred_vis = original_surfemb_embedding_rgb(queries[index], mask=pred_mask, demean=False)
        gt_vis = original_surfemb_embedding_rgb(gt_dense, mask=dense_mask_t, demean=True)

        yx = batch["surfemb_mask_samples"][index].to(device=device, dtype=torch.long)
        sampled_query_vis = sampled_embedding_image(queries[index], yx, gt_mask.shape, demean=False)
        sampled_keys = model.surface_key_mlp(
            batch["surfemb_coords_pos"][index].to(device=device, dtype=torch.float32)
        ).float()
        sampled_key_dense = torch.zeros_like(queries[index])
        sampled_key_dense[yx[:, 0], yx[:, 1]] = sampled_keys
        sampled_key_vis = sampled_embedding_image(sampled_key_dense, yx, gt_mask.shape, demean=True)

        video = str(batch["video_name"][index])
        frame = str(batch["frame_id"][index])
        dataset_name = "LND" if "surgripe" in video.lower() else "RARP"
        sample_label = f"{index:02d} {dataset_name[0]} f{frame}"
        dense_tiles.append(
            make_sample_tile(
                input_rgb,
                pred_vis,
                gt_vis,
                sample_label,
                ("input", "pred query", "GT key"),
            )
        )
        sampled_tiles.append(
            make_sample_tile(
                input_rgb,
                sampled_query_vis,
                sampled_key_vis,
                sample_label,
                ("input", "pred 614px", "GT 614px"),
            )
        )
        manifest.append(
            {
                "batch_index": index,
                "dataset": dataset_name,
                "video_name": video,
                "frame_id": frame,
                "gt_wrist_pixels": int(gt_mask.sum()),
                "dense_visible_wrist_pixels": int(dense_mask.sum()),
                "predicted_wrist_pixels": int(pred_mask.sum().item()),
                "positive_samples": int(len(yx)),
            }
        )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dense_path = output_dir / "wristonly_batch_dense_correspondence_contact_sheet.jpg"
    sampled_path = output_dir / "wristonly_batch_sampled_pairs_contact_sheet.jpg"
    save_rgb(dense_path, contact_sheet(dense_tiles, args.samples_per_row))
    save_rgb(sampled_path, contact_sheet(sampled_tiles, args.samples_per_row))

    import json

    metadata = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_iter": checkpoint_iter,
        "checkpoint_epoch": checkpoint_epoch,
        "batch_size": len(x),
        "seed": int(args.seed),
        "dataset_reprs": [repr(dataset) for dataset in datasets],
        "embedding_visualization": (
            "Original SurfEmb get_emb_vis: reshape 12D to 3x4, mean each group, "
            "max-abs normalize, map to [0,1]; GT keys demeaned on visible/sample support."
        ),
        "samples": manifest,
    }
    (output_dir / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"checkpoint_iter={checkpoint_iter} batch={len(x)} output={output_dir}")
    print(f"dense_contact_sheet={dense_path}")
    print(f"sampled_contact_sheet={sampled_path}")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=56)
    parser.add_argument("--samples_per_row", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--amp", type=int, choices=(0, 1), default=1)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
