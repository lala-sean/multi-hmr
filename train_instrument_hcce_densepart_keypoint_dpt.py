import os
import sys

os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["EGL_DEVICE_ID"] = "0"

from argparse import ArgumentParser

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from datasets.collate_multitask import collate_fn_instrument_multitask
from datasets.surgical_instruments import RARP50_DIR, SurgicalInstruments
from datasets.surgpose_instruments import SURGPOSE_ROOT, SurgPoseInstruments
from datasets.vos_endovis_instruments import (
    VOS_ENDOVIS17_IMAGE_ROOT,
    VOS_ENDOVIS17_ROOT,
    VOS_ENDOVIS18_IMAGE_ROOT,
    VOS_ENDOVIS18_ROOT,
    VOSEndoVisInstruments,
)
from loss_instrument_hcce_densepart_keypoint_pose import (
    LossInstrumentHCCEDensePartKeypointPose,
)
from multi_instrument.multi_instrument_hcce_densepart_keypoint_dpt import (
    MultiInstrumentHCCEDensePartKeypointDPT,
)
from train_instrument_cse import is_main_process, setup_ddp
from train_instrument_hcce_densepart_dpt import TrainerHCCEDensePartPose
from train_instrument_hcce_pose_dpt import make_rarp_dataset


class TrainerHCCEDensePartKeypointPose(TrainerHCCEDensePartPose):
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

        b_sel = batch_all[idx_vis]
        i_sel = inst_all[idx_vis]
        target = {
            "scores": scores,
            "offset": pk_offset.to(self.device)[idx_vis],
            "loc": wrist_centers.to(self.device)[idx_vis],
            "inst_masks": y["inst_masks"][idx_h[0], idx_h[1]][idx_vis].to(self.device),
            "part_masks": y["part_masks"][idx_h[0], idx_h[1]][idx_vis].to(self.device),
            "idx": (
                b_sel,
                pk_idx_dev[idx_vis, 1],
                pk_idx_dev[idx_vis, 0],
                torch.zeros_like(b_sel),
            ),
        }

        if "part_masks_dense" in y:
            target["part_masks_dense"] = y["part_masks_dense"][b_sel, i_sel].to(self.device)
        if "inst_masks_dense" in y:
            target["inst_masks_dense"] = y["inst_masks_dense"][b_sel, i_sel].to(self.device)

        if "dataset_source" in y:
            target["det_source_mask"] = (y["dataset_source"] == 0).to(self.device)
            target["rarp50_inst_mask"] = target["det_source_mask"][target["idx"][0]]

        if "has_cse" in y and "coord_imgs" in y:
            target["has_cse_inst"] = y["has_cse"][idx_h[0], idx_h[1]][idx_vis].to(self.device)
            target["coord_imgs"] = y["coord_imgs"][idx_h[0], idx_h[1]][idx_vis].to(self.device)
        if "has_cse_dense" in y and "coord_imgs_dense" in y:
            target["has_cse_dense_inst"] = y["has_cse_dense"][b_sel, i_sel].to(self.device)
            target["coord_imgs_dense"] = y["coord_imgs_dense"][b_sel, i_sel].to(self.device)

        if "has_pose" in y:
            target["has_pose_inst"] = y["has_pose"][idx_h[0], idx_h[1]][idx_vis].to(self.device)
            target["action_gt_inst"] = y["action_gt"][idx_h[0], idx_h[1]][idx_vis].to(self.device)
            target["wrist_quat_gt_inst"] = y["wrist_quat_gt"][idx_h[0], idx_h[1]][idx_vis].to(self.device)
            target["wrist_trans_gt_inst"] = y["wrist_trans_gt"][idx_h[0], idx_h[1]][idx_vis].to(self.device)
        if "pose_sym_flipped" in y:
            target["pose_sym_flipped_inst"] = y["pose_sym_flipped"][idx_h[0], idx_h[1]][idx_vis].to(self.device)

        if "has_keypoints" in y and "keypoints" in y:
            target["has_keypoint_inst"] = y["has_keypoints"][b_sel, i_sel].to(self.device)
            target["keypoint_gt_inst"] = y["keypoints"][b_sel, i_sel].to(self.device)
            if "keypoints_valid" in y:
                target["keypoint_valid_inst"] = y["keypoints_valid"][b_sel, i_sel].to(self.device)

        if "has_depth" in y and "depth_maps_dense" in y and "depth_valid_dense" in y:
            target["has_depth_inst"] = y["has_depth"][b_sel, i_sel].to(self.device)
            target["depth_maps_dense_inst"] = y["depth_maps_dense"][b_sel, i_sel].to(self.device)
            target["depth_valid_dense_inst"] = y["depth_valid_dense"][b_sel, i_sel].to(self.device)

        return target


def add_vos_datasets(args, split, training, subsample):
    datasets = []
    if not args.use_vos_endovis:
        return datasets
    if os.path.isdir(args.vos_endovis17_root):
        datasets.append(
            VOSEndoVisInstruments(
                label_root=args.vos_endovis17_root,
                image_root=args.vos_endovis17_image_root,
                split=split if split != "test18" else "test",
                training=training,
                img_size=args.img_size,
                subsample=subsample,
                aug_random_crop_rotate=bool(args.aug_random_crop_rotate),
                aug_geom_prob=args.aug_geom_prob,
                aug_crop_scale=args.aug_crop_scale,
                aug_max_angle=args.aug_max_angle,
                aug_offset_scale=args.aug_offset_scale,
                aug_color_jitter=bool(args.aug_color_jitter),
                dense_output_stride=args.dense_output_stride,
            )
        )
    if os.path.isdir(args.vos_endovis18_root):
        split18 = split if split in ("train", "test") else "test"
        datasets.append(
            VOSEndoVisInstruments(
                label_root=args.vos_endovis18_root,
                image_root=args.vos_endovis18_image_root,
                split=split18,
                training=training,
                img_size=args.img_size,
                subsample=subsample,
                aug_random_crop_rotate=bool(args.aug_random_crop_rotate),
                aug_geom_prob=args.aug_geom_prob,
                aug_crop_scale=args.aug_crop_scale,
                aug_max_angle=args.aug_max_angle,
                aug_offset_scale=args.aug_offset_scale,
                aug_color_jitter=bool(args.aug_color_jitter),
                dense_output_stride=args.dense_output_stride,
            )
        )
    return datasets


def make_surgpose_dataset(args, split, training, subsample):
    return SurgPoseInstruments(
        root_dir=args.surgpose_root,
        split=split,
        training=training,
        img_size=args.img_size,
        subsample=subsample,
        train_episodes=args.surgpose_train_episodes,
        val_episodes=args.surgpose_val_episodes,
        aug_random_crop_rotate=bool(args.aug_random_crop_rotate),
        aug_geom_prob=args.aug_geom_prob,
        aug_crop_scale=args.aug_crop_scale,
        aug_max_angle=args.aug_max_angle,
        aug_offset_scale=args.aug_offset_scale,
        aug_color_jitter=bool(args.aug_color_jitter),
        dense_output_stride=args.dense_output_stride,
        load_depth=bool(args.use_surgpose_depth),
        num_keypoints=args.num_keypoints,
        depth_normalize_relative=bool(args.surgpose_depth_normalize_relative),
        depth_norm_low_pct=args.surgpose_depth_norm_low_pct,
        depth_norm_high_pct=args.surgpose_depth_norm_high_pct,
    )


def main(args):
    if args.dense_output_stride != 2:
        raise ValueError("This model expects --dense_output_stride 2 for H/2 targets.")

    device, rank, world_size = setup_ddp()
    use_ddp = world_size > 1

    model = MultiInstrumentHCCEDensePartKeypointDPT(pretrained_backbone=True, **vars(args)).to(device)
    load_path = args.resume if args.resume is not None else args.pretrained
    resume_ckpt = None
    resume_epoch = 0
    resume_iter = 0
    resume_best_val = 0.0
    if load_path is not None and os.path.isfile(load_path):
        if is_main_process():
            mode = "resume" if args.resume is not None else "pretrained"
            print(f"Loading {mode} checkpoint from {load_path}")
        ckpt = torch.load(load_path, map_location=device, weights_only=False)
        log = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if args.resume is not None:
            resume_ckpt = ckpt
            resume_epoch = int(ckpt.get("epoch", 0))
            resume_iter = int(ckpt.get("iter", 0))
            resume_best_val = float(ckpt.get("best_val", 0.0))
        if is_main_process():
            print(log)

    if use_ddp:
        model = DDP(model, device_ids=[device], find_unused_parameters=True)

    val_datasets = []
    if args.use_rarp50:
        val_datasets.append(
            SurgicalInstruments(
                split="test",
                training=False,
                img_size=args.img_size,
                root_dir=args.data_dir,
                subsample=args.val_subsample,
                dense_output_stride=args.dense_output_stride,
            )
        )
    val_datasets.append(
        make_rarp_dataset(
            args,
            split="test",
            training=False,
            dataset_root=args.needle_puncture_data_dir,
            pose_root=args.needle_puncture_pose_dir,
            subsample=args.val_subsample,
        )
    )
    if args.use_vos_endovis:
        val_datasets.extend(add_vos_datasets(args, "valid", False, args.val_subsample))
    if args.use_surgpose:
        val_datasets.append(make_surgpose_dataset(args, "valid", False, args.val_subsample))

    l_val_data = [
        DataLoader(
            ds,
            batch_size=args.val_batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn_instrument_multitask,
        )
        for ds in val_datasets
    ]

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss = LossInstrumentHCCEDensePartKeypointPose(args)
    trainer = TrainerHCCEDensePartKeypointPose(
        model=model,
        loss=loss,
        optimizer=optimizer,
        device=device,
        args=args,
        best_val=resume_best_val,
    )
    trainer.current_epoch = resume_epoch
    trainer.current_iter = resume_iter
    if resume_ckpt is not None and "optimizer_state_dict" in resume_ckpt:
        optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
    if resume_ckpt is not None and "scaler_state_dict" in resume_ckpt:
        trainer.scaler.load_state_dict(resume_ckpt["scaler_state_dict"])

    if is_main_process():
        print()
        print(f"ARGS: {trainer.args}")
        print(f"LOG_DIR: {trainer.args.log_dir}")
        print("MODEL: MultiInstrumentHCCEDensePartKeypointDPT")
        print("LOSS:  LossInstrumentHCCEDensePartKeypointPose")
        print("DENSE PART ORDER: [wrist, gripper, shaft]")
        print(f"DEPTH HEAD: {bool(args.use_depth_head)}")
        for ds in val_datasets:
            print(ds)
        print()

    if args.eval_only:
        for vd in l_val_data:
            trainer.evaluate(vd)
    else:
        train_datasets = []
        if args.use_rarp50:
            train_datasets.append(
                SurgicalInstruments(
                    split="train",
                    training=True,
                    img_size=args.img_size,
                    root_dir=args.data_dir,
                    subsample=args.train_subsample,
                    aug_random_crop_rotate=bool(args.aug_random_crop_rotate),
                    aug_geom_prob=args.aug_geom_prob,
                    aug_crop_scale=args.aug_crop_scale,
                    aug_max_angle=args.aug_max_angle,
                    aug_offset_scale=args.aug_offset_scale,
                    aug_color_jitter=bool(args.aug_color_jitter),
                    dense_output_stride=args.dense_output_stride,
                )
            )
        train_datasets.extend(
            [
                make_rarp_dataset(
                    args,
                    split="train",
                    training=True,
                    dataset_root=args.needle_puncture_data_dir,
                    pose_root=args.needle_puncture_pose_dir,
                    subsample=args.train_subsample,
                ),
                make_rarp_dataset(
                    args,
                    split="train",
                    training=True,
                    dataset_root=args.needle_grasping_data_dir,
                    pose_root=args.needle_grasping_pose_dir,
                    subsample=args.train_subsample,
                ),
                make_rarp_dataset(
                    args,
                    split="train",
                    training=True,
                    dataset_root=args.knotting_data_dir,
                    pose_root=args.knotting_pose_dir,
                    subsample=args.train_subsample,
                ),
            ]
        )
        if args.use_vos_endovis:
            train_datasets.extend(add_vos_datasets(args, "train", True, args.train_subsample))
        if args.use_surgpose:
            train_datasets.append(make_surgpose_dataset(args, "train", True, args.train_subsample))

        if is_main_process():
            for ds in train_datasets:
                print(ds)
            print(f"CSE coord source: {'render_on_the_fly' if args.render_on_the_fly else args.cse_coord_root}")

        train_dataset = ConcatDataset(train_datasets)
        train_sampler = DistributedSampler(train_dataset, shuffle=True) if use_ddp else None
        train_data = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=(train_sampler is None),
            drop_last=True,
            collate_fn=collate_fn_instrument_multitask,
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

    parser = ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="logs")
    parser.add_argument("--name", type=str, default="instrument_rarp_hcce_densepart_keypoint_dpt_h2")
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=RARP50_DIR)
    parser.add_argument("--use_rarp50", type=int, default=1, choices=[0, 1])
    parser.add_argument("--use_vos_endovis", type=int, default=1, choices=[0, 1])
    parser.add_argument("--use_surgpose", type=int, default=1, choices=[0, 1])
    parser.add_argument("--use_surgpose_depth", type=int, default=1, choices=[0, 1])

    parser.add_argument("--vos_endovis17_root", type=str, default=VOS_ENDOVIS17_ROOT)
    parser.add_argument("--vos_endovis18_root", type=str, default=VOS_ENDOVIS18_ROOT)
    parser.add_argument("--vos_endovis17_image_root", type=str, default=VOS_ENDOVIS17_IMAGE_ROOT)
    parser.add_argument("--vos_endovis18_image_root", type=str, default=VOS_ENDOVIS18_IMAGE_ROOT)
    parser.add_argument("--surgpose_root", type=str, default=SURGPOSE_ROOT)
    parser.add_argument("--surgpose_train_episodes", type=str, default="")
    parser.add_argument("--surgpose_val_episodes", type=str, default="")
    parser.add_argument("--num_keypoints", type=int, default=7)
    parser.add_argument("--surgpose_depth_normalize_relative", type=int, default=1, choices=[0, 1])
    parser.add_argument("--surgpose_depth_norm_low_pct", type=float, default=2.0)
    parser.add_argument("--surgpose_depth_norm_high_pct", type=float, default=98.0)

    parser.add_argument("--needle_puncture_data_dir", type=str, default=PUNCTURE_DATASET_ROOT)
    parser.add_argument("--needle_puncture_pose_dir", type=str, default=PUNCTURE_POSE_ROOT)
    parser.add_argument("--needle_grasping_data_dir", type=str, default=GRASPING_DATASET_ROOT)
    parser.add_argument("--needle_grasping_pose_dir", type=str, default=GRASPING_POSE_ROOT)
    parser.add_argument("--knotting_data_dir", type=str, default=KNOTTING_DATASET_ROOT)
    parser.add_argument("--knotting_pose_dir", type=str, default=KNOTTING_POSE_ROOT)
    parser.add_argument("--min_dice_shaft", type=float, default=0.8)
    parser.add_argument("--min_dice_wrist", type=float, default=0.6)
    parser.add_argument("--min_dice_gripper", type=float, default=0.6)
    parser.add_argument("--needle_train_ratio", type=float, default=0.95)

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--num_workers", "-j", type=int, default=0)
    parser.add_argument("--img_size", type=int, default=630)
    parser.add_argument("--backbone", type=str, default="dinov2_vits14", choices=["dinov2_vitl14", "dinov2_vitb14", "dinov2_vits14"])
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
    parser.add_argument("--val_batch_size", type=int, default=4)
    parser.add_argument("--train_subsample", type=int, default=1)
    parser.add_argument("--val_subsample", type=int, default=10)
    parser.add_argument("--dense_output_stride", type=int, default=2)

    parser.add_argument("--xat_dim", type=int, default=512)
    parser.add_argument("--xat_depth", type=int, default=4)
    parser.add_argument("--xat_heads", type=int, default=16)
    parser.add_argument("--xat_dim_head", type=int, default=32)
    parser.add_argument("--xat_mlp_dim", type=int, default=2048)
    parser.add_argument("--mask_dim", type=int, default=256)
    parser.add_argument("--num_parts", type=int, default=4)
    parser.add_argument("--hcce_feat_dim", type=int, default=256)
    parser.add_argument("--action_dim", type=int, default=3)
    parser.add_argument("--use_pose_heads", type=int, default=1, choices=[0, 1])
    parser.add_argument("--use_depth_head", type=int, default=0, choices=[0, 1])
    parser.add_argument("--pose_head_iter", type=int, default=4)
    parser.add_argument("--pose_head_dropout", type=float, default=0.3)

    parser.add_argument("--cse_coord_root", type=str, default=None)
    parser.add_argument("--render_on_the_fly", type=int, default=1, choices=[0, 1])
    parser.add_argument("--canonicalize_pose_symmetry", type=int, default=1, choices=[0, 1])
    parser.add_argument("--canonical_eps", type=float, default=0.08)
    parser.add_argument("--aug_random_crop_rotate", type=int, default=1, choices=[0, 1])
    parser.add_argument("--aug_crop_scale", type=float, default=1.2)
    parser.add_argument("--aug_geom_prob", type=float, default=0.3)
    parser.add_argument("--aug_max_angle", type=float, default=float(np.pi / 6.0))
    parser.add_argument("--aug_offset_scale", type=float, default=1.0)
    parser.add_argument("--aug_color_jitter", type=int, default=1, choices=[0, 1])

    parser = LossInstrumentHCCEDensePartKeypointPose.add_specific_args(parser)
    main(parser.parse_args())
