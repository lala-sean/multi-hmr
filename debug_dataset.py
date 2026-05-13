"""
Sanity-check all RARPInstanceDataset variants + SurgicalInstruments.
Loads a few batches from each dataset, validates output format, and
saves side-by-side visualisations.

Usage:
    conda run -n multiipr python debug_dataset.py [--n_batches 2] [--out_dir debug_vis]
"""
import os
import argparse
import traceback
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image

from datasets.surgical_instruments import SurgicalInstruments, collate_fn_instrument
from datasets.RarpInstanceDataset import RARPInstanceDataset
from datasets.endovis_dataset import EndoVisDataset
from utils import denormalize_rgb

# ── dataset configurations ─────────────────────────────────────────────────
NAS = '/mnt/nas/share/shuojue/data'

DATASET_CONFIGS = [
    dict(
        name='needlePuncture',
        dataset_root=f'{NAS}/needlePuncture_videos',
        pose_root=f'{NAS}/needlePuncture_results',
        v2_force=False,
    ),
    dict(
        name='needleGrasping',
        dataset_root=f'{NAS}/needleGrasping_videos',
        pose_root=f'{NAS}/needleGrasping_results',
        v2_force=False,
    ),
    dict(
        name='knotting',
        dataset_root=f'{NAS}/knotting_videos',
        pose_root=f'{NAS}/knotting_results',
        v2_force=False,
    ),
    dict(
        name='suturePulling',
        dataset_root=f'{NAS}/suturePulling_videos',
        pose_root=None,          # no filtering — all frames participate
        v2_force=True,           # uses frames_v2 and refined_masks_v2_*
    ),
]

# ── visualisation helpers ───────────────────────────────────────────────────
PART_COLORS = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255)]  # bg/gripper/wrist/shaft
ALPHA = 0.5


def overlay(img_np, inst_masks, part_masks, wrist_centers, valid):
    H, W = img_np.shape[:2]
    canvas = img_np.copy().astype(np.float32)
    for k in range(inst_masks.shape[0]):
        if valid[k] == 0:
            continue
        m = inst_masks[k]
        p = part_masks[k]
        m_full = np.array(Image.fromarray(m.astype(np.uint8)).resize((W, H), Image.NEAREST)) > 0
        p_full = np.array(Image.fromarray(p.astype(np.uint8)).resize((W, H), Image.NEAREST))
        if m_full.sum() == 0:
            continue
        color_map = np.zeros((H, W, 3), dtype=np.float32)
        for part_id, color in enumerate(PART_COLORS):
            region = m_full & (p_full == part_id)
            for c in range(3):
                color_map[:, :, c][region] = color[c]
        canvas[m_full] = canvas[m_full] * (1 - ALPHA) + color_map[m_full] * ALPHA
    canvas = canvas.clip(0, 255).astype(np.uint8)
    for k in range(wrist_centers.shape[0]):
        if valid[k] == 0:
            continue
        cx, cy = int(wrist_centers[k, 0].item()), int(wrist_centers[k, 1].item())
        r = 8
        for d in range(-r, r + 1):
            for t in [-1, 0, 1]:
                for px, py in [(cx + d, cy + t), (cx + t, cy + d)]:
                    if 0 <= px < W and 0 <= py < H:
                        canvas[py, px] = [255, 255, 255]
    return canvas


def batch_to_grid(x, y, tag, max_imgs=4):
    bs = min(x.shape[0], max_imgs)
    strips = []
    for i in range(bs):
        img_np = denormalize_rgb(x[i].numpy())
        H, W = img_np.shape[:2]
        valid  = y['valid_instruments'][i].numpy()
        inst_m = y['inst_masks'][i].numpy()
        part_m = y['part_masks'][i].numpy()
        wc     = y['wrist_centers'][i]
        n_valid = int(valid.sum())
        ov = overlay(img_np, inst_m, part_m, wc, valid)
        strip = np.concatenate([img_np, ov], axis=1)
        label = f"{tag}  img{i}  #inst={n_valid}"
        strip[:18, :len(label) * 7] = 40
        strips.append(strip)
    return np.concatenate(strips, axis=0)


def check_batch(x, y, tag, img_size):
    """Run format/value checks and print results. Returns True if all pass."""
    ok = True
    required_keys = ('valid_instruments', 'wrist_centers', 'inst_masks', 'part_masks')
    for k in required_keys:
        if k not in y:
            print(f"  [{tag}] MISSING key: {k}")
            ok = False

    if ok:
        # dtype checks
        for k in required_keys:
            print(f"  [{tag}] {k}: {tuple(y[k].shape)}  dtype={y[k].dtype}")

        # part label range
        vals = y['part_masks'].unique().tolist()
        bad = [v for v in vals if v not in (0, 1, 2, 3)]
        if bad:
            print(f"  [{tag}] BAD part labels: {bad}  ✗")
            ok = False
        else:
            print(f"  [{tag}] part labels {vals}  ✓")

        # wrist centres in bounds
        valid_mask = y['valid_instruments'].bool()
        if valid_mask.any():
            wc = y['wrist_centers'][valid_mask]
            in_bounds = ((wc >= 0) & (wc < img_size)).all()
            status = "✓" if in_bounds else "✗"
            print(f"  [{tag}] wrist_centers in [0,{img_size}): {bool(in_bounds)}  {status}")
            if not in_bounds:
                ok = False
        else:
            print(f"  [{tag}] no valid instruments in batch")

    return ok


# ── per-dataset test ────────────────────────────────────────────────────────
def test_dataset(cfg, args):
    name = cfg['name']
    print(f"\n{'='*70}")
    print(f"  Dataset: {name}")
    print(f"  dataset_root : {cfg['dataset_root']}")
    print(f"  pose_root    : {cfg.get('pose_root')}")
    print(f"  v2_force     : {cfg.get('v2_force', False)}")
    print(f"{'='*70}")

    try:
        ds = RARPInstanceDataset(
            split='train',
            training=False,
            img_size=args.img_size,
            dataset_root=cfg['dataset_root'],
            pose_root=cfg.get('pose_root'),
            v2_force=cfg.get('v2_force', False),
            subsample=args.subsample,
        )
    except Exception:
        print(f"  [ERROR] Failed to construct dataset:")
        traceback.print_exc()
        return False

    print(f"  Samples (after subsample={args.subsample}): {len(ds)}")
    if len(ds) == 0:
        print(f"  [WARN] Empty dataset — skipping loader test.")
        return False

    loader = DataLoader(ds, batch_size=args.batch_size,
                        collate_fn=collate_fn_instrument, shuffle=False,
                        num_workers=0)

    all_ok = True
    ds_out_dir = os.path.join(args.out_dir, name)
    os.makedirs(ds_out_dir, exist_ok=True)

    for batch_idx, (x, y) in enumerate(loader):
        if batch_idx >= args.n_batches:
            break

        print(f"\n  -- Batch {batch_idx}  x.shape={tuple(x.shape)}")
        ok = check_batch(x, y, name, args.img_size)
        all_ok = all_ok and ok

        # visualise
        grid = batch_to_grid(x, y, name)
        out_path = os.path.join(ds_out_dir, f"batch_{batch_idx:02d}.jpg")
        Image.fromarray(grid).save(out_path, quality=90)
        print(f"  Saved → {out_path}")

    return all_ok


ENDOVIS_CONFIGS = [
    dict(
        name='endovis17_train',
        root_dir='/mnt/nas/haofeng/data/Ref-Endovis17',
        split='train',
        expected_frames=2100,
    ),
    dict(
        name='endovis17_valid',
        root_dir='/mnt/nas/haofeng/data/Ref-Endovis17',
        split='valid',
        expected_frames=900,
    ),
    dict(
        name='endovis18_train',
        root_dir='/mnt/nas/haofeng/data/Ref-Endovis18',
        split='train',
        expected_frames=1639,
    ),
    dict(
        name='endovis18_valid',
        root_dir='/mnt/nas/haofeng/data/Ref-Endovis18',
        split='valid',
        expected_frames=596,
    ),
]


def test_endovis(cfg, args):
    name = cfg['name']
    expected = cfg['expected_frames']
    print(f"\n{'='*70}")
    print(f"  Dataset : {name}")
    print(f"  root    : {cfg['root_dir']}  split={cfg['split']}")
    print(f"  Expected frames: {expected}")
    print(f"{'='*70}")

    try:
        ds = EndoVisDataset(
            root_dir=cfg['root_dir'],
            split=cfg['split'],
            training=False,
            img_size=args.img_size,
            subsample=1,          # full count for frame-count check
        )
    except Exception:
        print(f"  [ERROR] Failed to construct dataset:")
        traceback.print_exc()
        return False

    print(f"  len(ds) = {len(ds)}  (expected {expected})")
    if len(ds) != expected:
        print(f"  [WARN] Frame count mismatch — expected {expected}, got {len(ds)}")

    # Build subsampled loader for batch checks
    ds_sub = EndoVisDataset(
        root_dir=cfg['root_dir'],
        split=cfg['split'],
        training=False,
        img_size=args.img_size,
        subsample=args.subsample,
    )
    print(f"  Subsampled to {len(ds_sub)} samples (subsample={args.subsample})")
    if len(ds_sub) == 0:
        print("  [WARN] Empty after subsampling — skipping.")
        return len(ds) == expected

    loader = DataLoader(ds_sub, batch_size=args.batch_size,
                        collate_fn=collate_fn_instrument, shuffle=False,
                        num_workers=0)

    all_ok = (len(ds) == expected)
    ds_out_dir = os.path.join(args.out_dir, name)
    os.makedirs(ds_out_dir, exist_ok=True)

    for batch_idx, (x, y) in enumerate(loader):
        if batch_idx >= args.n_batches:
            break
        print(f"\n  -- Batch {batch_idx}  x.shape={tuple(x.shape)}")
        ok = check_batch(x, y, name, args.img_size)
        all_ok = all_ok and ok

        grid = batch_to_grid(x, y, name)
        out_path = os.path.join(ds_out_dir, f"batch_{batch_idx:02d}.jpg")
        Image.fromarray(grid).save(out_path, quality=90)
        print(f"  Saved → {out_path}")

    return all_ok


# ── reference: SurgicalInstruments ─────────────────────────────────────────
def test_surgical_instruments(args):
    print(f"\n{'='*70}")
    print(f"  Dataset: SurgicalInstruments  (reference)")
    print(f"{'='*70}")
    ds = SurgicalInstruments(split='train', training=False,
                             img_size=args.img_size, subsample=args.subsample)
    print(f"  Samples: {len(ds)}")
    loader = DataLoader(ds, batch_size=args.batch_size,
                        collate_fn=collate_fn_instrument, shuffle=False,
                        num_workers=0)
    ds_out_dir = os.path.join(args.out_dir, 'SurgicalInstruments')
    os.makedirs(ds_out_dir, exist_ok=True)
    for batch_idx, (x, y) in enumerate(loader):
        if batch_idx >= args.n_batches:
            break
        print(f"\n  -- Batch {batch_idx}  x.shape={tuple(x.shape)}")
        check_batch(x, y, 'SurgicalInstruments', args.img_size)
        grid = batch_to_grid(x, y, 'SurgicalInstruments')
        out_path = os.path.join(ds_out_dir, f"batch_{batch_idx:02d}.jpg")
        Image.fromarray(grid).save(out_path, quality=90)
        print(f"  Saved → {out_path}")


# ── main ────────────────────────────────────────────────────────────────────
def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    test_surgical_instruments(args)

    results = {}
    for cfg in DATASET_CONFIGS:
        ok = test_dataset(cfg, args)
        results[cfg['name']] = ok

    for cfg in ENDOVIS_CONFIGS:
        ok = test_endovis(cfg, args)
        results[cfg['name']] = ok

    print(f"\n{'='*70}")
    print("  Summary:")
    for name, ok in results.items():
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"    {name:<25} {status}")
    print(f"{'='*70}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_batches',  type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--img_size',   type=int, default=448)
    parser.add_argument('--subsample',  type=int, default=100,
                        help='Keep every Nth sample to speed up the check')
    parser.add_argument('--out_dir',    type=str, default='debug_vis')
    args = parser.parse_args()
    main(args)
