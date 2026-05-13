import argparse
from collections import defaultdict

import torch
from torch.utils.data import ConcatDataset, DataLoader

from datasets.RarpInstanceDataset import RARPInstanceDataset
from datasets.collate_cse import collate_fn_instrument_cse


DATASETS = {
    "needle_puncture": (
        "/mnt/nas/share/shuojue/data/needlePuncture_videos",
        "/mnt/nas/share/shuojue/data/needlePuncture_results",
        False,
    ),
    "needle_grasping": (
        "/mnt/nas/share/shuojue/data/needleGrasping_videos",
        "/mnt/nas/share/shuojue/data/needleGrasping_results",
        False,
    ),
    "knotting": (
        "/mnt/nas/share/shuojue/data/knotting_videos",
        "/mnt/nas/share/shuojue/data/knotting_results",
        False,
    ),
}

PARTS = {
    1: "shaft",
    2: "wrist",
    3: "l_gripper",
    4: "r_gripper",
}


def build_dataset(args):
    names = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    datasets = []
    for name in names:
        root, pose_root, v2_force = DATASETS[name]
        datasets.append(
            RARPInstanceDataset(
                split=args.split,
                training=False,
                img_size=args.img_size,
                dataset_root=root,
                pose_root=pose_root,
                min_dice=[
                    args.min_dice_shaft,
                    args.min_dice_wrist,
                    args.min_dice_gripper,
                ],
                train_ratio=args.needle_train_ratio,
                subsample=args.subsample,
                n=args.n,
                v2_force=v2_force,
                cse_coord_root=args.cse_coord_root,
                render_on_the_fly=bool(args.render_on_the_fly),
            )
        )
    return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=[*DATASETS.keys(), "all"], default="all")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--img_size", type=int, default=630)
    parser.add_argument("--subsample", type=int, default=50)
    parser.add_argument("--n", type=int, default=-1)
    parser.add_argument("--max_batches", type=int, default=50)
    parser.add_argument("--cse_coord_root", default=None)
    parser.add_argument("--render_on_the_fly", type=int, default=1, choices=[0, 1])
    parser.add_argument("--needle_train_ratio", type=float, default=0.95)
    parser.add_argument("--min_dice_shaft", type=float, default=0.8)
    parser.add_argument("--min_dice_wrist", type=float, default=0.6)
    parser.add_argument("--min_dice_gripper", type=float, default=0.6)
    args = parser.parse_args()

    dataset = build_dataset(args)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn_instrument_cse,
    )

    totals = defaultdict(float)
    by_part = {
        pid: defaultdict(float)
        for pid in PARTS
    }

    for batch_idx, (_, y) in enumerate(loader):
        if batch_idx >= args.max_batches:
            break

        valid_inst = y["valid_instruments"].bool()
        has_cse = y["has_cse"].bool() & valid_inst
        totals["instances"] += float(valid_inst.sum().item())
        totals["cse_instances"] += float(has_cse.sum().item())

        coord_imgs = y["coord_imgs"]
        inst_masks = y["inst_masks"] > 0.5

        cse_indices = torch.where(has_cse)
        for bi, ii in zip(cse_indices[0], cse_indices[1]):
            coord = coord_imgs[bi, ii]
            part = coord[:, :, 3].long()
            coord_ok = part > 0
            inst_ok = inst_masks[bi, ii]
            supervised = coord_ok & inst_ok

            totals["coord_px"] += float(coord_ok.sum().item())
            totals["supervised_px"] += float(supervised.sum().item())

            for pid in PARTS:
                part_coord = part == pid
                part_supervised = supervised & part_coord
                by_part[pid]["coord_px"] += float(part_coord.sum().item())
                by_part[pid]["supervised_px"] += float(part_supervised.sum().item())

    print(f"dataset={args.dataset} split={args.split}")
    print(f"instances:      {totals['instances']:.0f}")
    print(f"cse_instances:  {totals['cse_instances']:.0f}")
    print(f"coord_px:       {totals['coord_px']:.0f}")
    print(f"supervised_px:  {totals['supervised_px']:.0f}")
    if totals["coord_px"] > 0:
        print(f"supervised/coord: {totals['supervised_px'] / totals['coord_px']:.4f}")

    print()
    for pid, name in PARTS.items():
        coord_px = by_part[pid]["coord_px"]
        sup_px = by_part[pid]["supervised_px"]
        ratio = sup_px / coord_px if coord_px > 0 else 0.0
        print(
            f"{name:10s} coord_px={coord_px:.0f} "
            f"supervised_px={sup_px:.0f} supervised/coord={ratio:.4f}"
        )


if __name__ == "__main__":
    main()
