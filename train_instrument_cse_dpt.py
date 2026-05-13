import os
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["EGL_DEVICE_ID"] = "0"

from argparse import ArgumentParser

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from datasets.RarpInstanceDataset import RARPInstanceDataset
from datasets.collate_cse import collate_fn_instrument_cse
from datasets.surgical_instruments import (
    RARP50_DIR,
    SurgicalInstruments,
    collate_fn_instrument,
)
from loss_instrument_cse_visible import LossInstrumentCSEVisibleMetrics
from multi_instrument.multi_instrument_cse_dpt import MultiInstrumentCSEDPT
from train_instrument_cse import TrainerCSE, is_main_process, setup_ddp


class TrainerCSEFixed(TrainerCSE):
    """
    TrainerCSE variant with a non-mutating prepare_gt implementation.

    The original parent path mutates y['valid_instruments'] while removing
    duplicate wrist-center patches. This version computes the same keep mask
    locally and uses it for masks, CSE flags, and coord images, keeping all
    per-instance tensors aligned.
    """

    def prepare_gt(self, y):
        valid = y["valid_instruments"].bool()
        idx_h = torch.where(valid)
        n_valid = int(valid.sum().item())
        if n_valid == 0:
            return None

        wrist_centers = y["wrist_centers"][idx_h[0], idx_h[1]]
        n_patch = self.args.img_size // self.raw_model.patch_size
        pk_coarse = (wrist_centers / self.raw_model.patch_size).int()
        pk_idx = torch.clamp(pk_coarse, 0, n_patch - 1)
        pk_offset = (
            wrist_centers - (pk_idx + 0.5) * self.raw_model.patch_size
        ) / self.raw_model.patch_size

        scores = torch.zeros((valid.shape[0], n_patch, n_patch), device=self.device)
        keep = torch.ones(n_valid, dtype=torch.bool, device=self.device)

        batch_all = idx_h[0].to(self.device)
        inst_all = idx_h[1].to(self.device)
        pk_idx_dev = pk_idx.to(self.device)

        for k in range(n_valid):
            bi = int(batch_all[k].item())
            px = pk_idx_dev[k, 1]
            py = pk_idx_dev[k, 0]
            if scores[bi, px, py] == 1:
                keep[k] = False
            else:
                scores[bi, px, py] = 1

        idx_vis = torch.where(keep)[0]
        if idx_vis.numel() == 0:
            return None

        target = {
            "scores": scores,
            "offset": pk_offset.to(self.device)[idx_vis],
            "loc": wrist_centers.to(self.device)[idx_vis],
            "inst_masks": y["inst_masks"][idx_h[0], idx_h[1]][idx_vis].to(self.device),
            "part_masks": y["part_masks"][idx_h[0], idx_h[1]][idx_vis].to(self.device),
            "idx": (
                batch_all[idx_vis],
                pk_idx_dev[idx_vis, 1],
                pk_idx_dev[idx_vis, 0],
                torch.zeros_like(batch_all[idx_vis]),
            ),
        }

        if "dataset_source" in y:
            target["det_source_mask"] = (y["dataset_source"] == 0).to(self.device)
            target["rarp50_inst_mask"] = target["det_source_mask"][target["idx"][0]]

        if "has_cse" in y and "coord_imgs" in y:
            has_cse_all = y["has_cse"][idx_h[0], idx_h[1]]
            coord_all = y["coord_imgs"][idx_h[0], idx_h[1]]
            target["has_cse_inst"] = has_cse_all[idx_vis].to(self.device)
            target["coord_imgs"] = coord_all[idx_vis].to(self.device)

        return target


def main(args):
    if args.log_freq <= 0:
        raise ValueError("--log_freq must be positive")
    if args.train_vis_freq <= 0:
        raise ValueError("--train_vis_freq must be positive")
    if args.train_vis_freq % args.log_freq != 0:
        raise ValueError("--train_vis_freq must be an integer multiple of --log_freq")

    device, rank, world_size = setup_ddp()
    use_ddp = world_size > 1

    model = MultiInstrumentCSEDPT(pretrained_backbone=True, **vars(args)).to(device)

    if args.pretrained is not None and os.path.isfile(args.pretrained):
        if is_main_process():
            print(f"Loading weights from {args.pretrained}")
        ckpt = torch.load(args.pretrained, map_location=device)
        log = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if is_main_process():
            print(f"{log}")

    if use_ddp:
        model = DDP(model, device_ids=[device], find_unused_parameters=True)

    l_val_data = []
    if args.use_rarp50:
        l_val_data.append(
            DataLoader(
                SurgicalInstruments(
                    split="test",
                    training=False,
                    img_size=args.img_size,
                    root_dir=args.data_dir,
                    subsample=args.val_subsample,
                ),
                batch_size=args.val_batch_size,
                num_workers=args.num_workers,
                shuffle=False,
                drop_last=False,
                collate_fn=collate_fn_instrument,
            )
        )

    l_val_data.append(
        DataLoader(
            RARPInstanceDataset(
                split="test",
                training=False,
                img_size=args.img_size,
                dataset_root=args.needle_puncture_data_dir,
                pose_root=args.needle_puncture_pose_dir,
                min_dice=[
                    args.min_dice_shaft,
                    args.min_dice_wrist,
                    args.min_dice_gripper,
                ],
                train_ratio=args.needle_train_ratio,
                subsample=args.val_subsample,
            ),
            batch_size=args.val_batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn_instrument,
        )
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss = LossInstrumentCSEVisibleMetrics(args)
    trainer = TrainerCSEFixed(
        model=model,
        loss=loss,
        optimizer=optimizer,
        device=device,
        args=args,
    )

    if is_main_process():
        print()
        print(f"ARGS: {trainer.args}")
        print(f"LOG_DIR: {trainer.args.log_dir}")
        print("MODEL: MultiInstrumentCSEDPT")
        print("LOSS:  LossInstrumentCSEVisibleMetrics")
        print()

    if args.eval_only:
        for vd in l_val_data:
            trainer.evaluate(vd)
    else:
        train_datasets = []
        if args.use_rarp50:
            ds_rarp50 = SurgicalInstruments(
                split="train",
                training=True,
                img_size=args.img_size,
                root_dir=args.data_dir,
                subsample=args.train_subsample,
            )
            train_datasets.append(ds_rarp50)
        else:
            ds_rarp50 = None

        cse_root = args.cse_coord_root
        otf = bool(args.render_on_the_fly)

        ds_needle_puncture = RARPInstanceDataset(
            split="train",
            training=True,
            img_size=args.img_size,
            dataset_root=args.needle_puncture_data_dir,
            pose_root=args.needle_puncture_pose_dir,
            min_dice=[
                args.min_dice_shaft,
                args.min_dice_wrist,
                args.min_dice_gripper,
            ],
            train_ratio=args.needle_train_ratio,
            subsample=1,
            cse_coord_root=cse_root,
            render_on_the_fly=otf,
        )
        ds_needle_grasping = RARPInstanceDataset(
            split="train",
            training=True,
            img_size=args.img_size,
            dataset_root=args.needle_grasping_data_dir,
            pose_root=args.needle_grasping_pose_dir,
            min_dice=[
                args.min_dice_shaft,
                args.min_dice_wrist,
                args.min_dice_gripper,
            ],
            train_ratio=args.needle_train_ratio,
            subsample=1,
            cse_coord_root=cse_root,
            render_on_the_fly=otf,
        )
        ds_knotting = RARPInstanceDataset(
            split="train",
            training=True,
            img_size=args.img_size,
            dataset_root=args.knotting_data_dir,
            pose_root=args.knotting_pose_dir,
            min_dice=[
                args.min_dice_shaft,
                args.min_dice_wrist,
                args.min_dice_gripper,
            ],
            train_ratio=args.needle_train_ratio,
            subsample=args.train_subsample,
            cse_coord_root=cse_root,
            render_on_the_fly=otf,
        )
        ds_suture_pulling = RARPInstanceDataset(
            split="train",
            training=True,
            img_size=args.img_size,
            dataset_root=args.suture_pulling_data_dir,
            pose_root=args.suture_pulling_pose_dir,
            min_dice=[
                args.min_dice_shaft,
                args.min_dice_wrist,
                args.min_dice_gripper,
            ],
            train_ratio=args.needle_train_ratio,
            subsample=2,
            v2_force=True,
            cse_coord_root=None,
            render_on_the_fly=False,
        )

        if is_main_process():
            if ds_rarp50 is not None:
                print(f"RARP50 train samples:  {len(ds_rarp50)}")
            else:
                print("RARP50 train samples:  disabled")
            print(f"NeedlePuncture train:  {len(ds_needle_puncture)}")
            print(f"NeedleGrasping train:  {len(ds_needle_grasping)}")
            print(f"Knotting train:        {len(ds_knotting)}")
            print(f"SuturePulling train:  {len(ds_suture_pulling)}")
            if otf:
                print("CSE coord source:      render_on_the_fly")
            elif cse_root:
                print(f"CSE coord source:      cached files from {cse_root}")
            else:
                print("CSE coord source:      none")

        train_datasets.extend(
            [
                ds_needle_puncture,
                ds_needle_grasping,
                ds_knotting,
                ds_suture_pulling,
            ]
        )
        train_dataset = ConcatDataset(train_datasets)
        train_sampler = DistributedSampler(train_dataset, shuffle=True) if use_ddp else None
        train_data = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=(train_sampler is None),
            drop_last=True,
            collate_fn=collate_fn_instrument_cse,
            sampler=train_sampler,
        )
        trainer.fit(train_data, l_val_data)

    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    PUNCTURE_DATASET_ROOT = "/mnt/nas/share/shuojue/data/needlePuncture_videos"
    PUNCTURE_POSE_ROOT = "/mnt/nas/share/shuojue/data/needlePuncture_results"
    GRASPING_DATASET_ROOT = "/mnt/nas/share/shuojue/data/needleGrasping_videos"
    GRASPING_POSE_ROOT = "/mnt/nas/share/shuojue/data/needleGrasping_results"
    KNOTTING_DATASET_ROOT = "/mnt/nas/share/shuojue/data/knotting_videos"
    KNOTTING_POSE_ROOT = "/mnt/nas/share/shuojue/data/knotting_results"
    PULLING_DATASET_ROOT = "/mnt/nas/share/shuojue/data/suturePulling_videos"
    PULLING_POSE_ROOT = None

    parser = ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="logs")
    parser.add_argument("--name", type=str, default="instrument_rarp_cse_dpt")
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=RARP50_DIR)
    parser.add_argument("--use_rarp50", type=int, default=1, choices=[0, 1])

    parser.add_argument("--needle_puncture_data_dir", type=str, default=PUNCTURE_DATASET_ROOT)
    parser.add_argument("--needle_puncture_pose_dir", type=str, default=PUNCTURE_POSE_ROOT)
    parser.add_argument("--needle_grasping_data_dir", type=str, default=GRASPING_DATASET_ROOT)
    parser.add_argument("--needle_grasping_pose_dir", type=str, default=GRASPING_POSE_ROOT)
    parser.add_argument("--knotting_data_dir", type=str, default=KNOTTING_DATASET_ROOT)
    parser.add_argument("--knotting_pose_dir", type=str, default=KNOTTING_POSE_ROOT)
    parser.add_argument("--suture_pulling_data_dir", type=str, default=PULLING_DATASET_ROOT)
    parser.add_argument("--suture_pulling_pose_dir", type=str, default=PULLING_POSE_ROOT)
    parser.add_argument("--min_dice_shaft", type=float, default=0.8)
    parser.add_argument("--min_dice_wrist", type=float, default=0.6)
    parser.add_argument("--min_dice_gripper", type=float, default=0.6)
    parser.add_argument("--needle_train_ratio", type=float, default=0.95)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", "-j", type=int, default=0)
    parser.add_argument("--img_size", type=int, default=630)
    parser.add_argument(
        "--backbone",
        type=str,
        default="dinov2_vits14",
        choices=["dinov2_vitl14", "dinov2_vitb14", "dinov2_vits14"],
    )
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--log_freq", type=int, default=100)
    parser.add_argument("--train_vis_freq", type=int, default=1000)
    parser.add_argument("--max_iter", type=int, default=60000)
    parser.add_argument("--nb_max_ckpt", type=int, default=10)
    parser.add_argument("--amp", type=int, default=1, choices=[0, 1])
    parser.add_argument("--learning_rate", "-lr", type=float, default=5e-5)
    parser.add_argument("--eval_only", type=int, default=0, choices=[0, 1])
    parser.add_argument("--det_thresh", type=float, default=0.3)
    parser.add_argument("--nms_kernel_size", type=int, default=3)
    parser.add_argument("--val_freq", type=int, default=1000)
    parser.add_argument("--val_batch_size", type=int, default=8)
    parser.add_argument("--train_subsample", type=int, default=1)
    parser.add_argument("--val_subsample", type=int, default=10)

    parser.add_argument("--xat_dim", type=int, default=512)
    parser.add_argument("--xat_depth", type=int, default=4)
    parser.add_argument("--xat_heads", type=int, default=16)
    parser.add_argument("--xat_dim_head", type=int, default=32)
    parser.add_argument("--xat_mlp_dim", type=int, default=2048)
    parser.add_argument("--mask_dim", type=int, default=256)
    parser.add_argument("--num_parts", type=int, default=4)
    parser.add_argument("--nocs_feat_dim", type=int, default=256)

    parser.add_argument("--cse_coord_root", type=str, default=None)
    parser.add_argument("--render_on_the_fly", type=int, default=1, choices=[0, 1])

    parser = LossInstrumentCSEVisibleMetrics.add_specific_args(parser)
    main(parser.parse_args())
