#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,2,3,4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

NAME="${NAME:-hcce_crop224_rarp_lnd_refinemem_bs56_gpu0234}"
LOG_PATH="logs/${NAME}_train.log"
mkdir -p logs

torchrun --standalone --nnodes=1 --nproc_per_node=4 train_hcce_crop_rarp_lnd_refinemem.py \
  --name "$NAME" \
  --include_lnd_train 1 \
  --include_lnd_val 1 \
  --lnd_root /mnt/iMVR/daiyun/Dataset/LND \
  --lnd_refine_memory /mnt/iMVR/daiyun/shuojue-temp/code/multi-hmr/submodules/gaussian-mesh-splatting/Results2/surgripe_lnd_refine_memory_train/TRAIN/refine_memory_pool.json \
  --lnd_sampling_rate 0.30 \
  --bbox_padding_frac 0.12 \
  --color_jitter 1 \
  --rgb_augmentation 1 \
  --occlusion_augmentation 1 \
  --occlusion_prob 0.5 \
  --batch_size 56 \
  --val_batch_size 56 \
  --num_workers 0 \
  --max_iter 60000 \
  --log_freq 10 \
  --val_freq 1000 \
  --val_max_batches 0 \
  --lr_backbone 1e-4 \
  --lr_dense 1e-4 \
  --lr_pose 1e-4 \
  --lr_keypoint 1e-4 \
  --alpha_action_l1 1.0 \
  --alpha_wrist_quat_l1 1.0 \
  --alpha_wrist_trans_l1 10.0 \
  --alpha_heatmap 1.0 \
  --alpha_keypoint_3d 0.0 \
  --alpha_keypoint_2d 0.0 \
  --alpha_hcce 1.0 \
  --heatmap_sigma 2.0 \
  --amp 1 \
  --pretrained_backbone 1 \
  --dist_timeout_sec 7200 \
  2>&1 | tee "$LOG_PATH"
