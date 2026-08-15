#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/iMVR/daiyun/shuojue-temp/code/multi-hmr"
cd "$ROOT"

RAW_RUN="surfemb_debug_current_lnd_augfix_rawdot_resume6k_to10k_b56x4_20260811"
COS_RUN="surfemb_debug_current_lnd_augfix_cosine_t01_resume6k_to10k_b56x4_20260811"
RAW_DIR="submodules/RoboPEPP/logs/${RAW_RUN}"
COS_DIR="submodules/RoboPEPP/logs/${COS_RUN}"

wait_tmux_done() {
  local session="$1"
  while tmux has-session -t "$session" 2>/dev/null; do
    sleep 60
  done
}

check_iter() {
  local ckpt="$1"
  local expected="$2"
  local iter
  iter=$(python - "$ckpt" <<'PY'
import sys
import torch
ckpt = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(int(ckpt.get("iter", -1)))
PY
)
  if [[ "$iter" != "$expected" ]]; then
    echo "CHECKPOINT_NOT_COMPLETE ckpt=$ckpt iter=$iter expected=$expected" >&2
    exit 1
  fi
}

wait_tmux_done "augfix_6k_to10k"
wait_tmux_done "augfix_cosine_6k_to10k"

check_iter "${RAW_DIR}/checkpoints/last.pt" "10000"
check_iter "${COS_DIR}/checkpoints/last.pt" "10000"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python submodules/RoboPEPP/eval_surfemb_wrist_lnd.py \
  --model raw10k_best="${RAW_DIR}/checkpoints/best_val_total.pt" \
  --model raw10k_last="${RAW_DIR}/checkpoints/last.pt" \
  --output_dir submodules/RoboPEPP/logs/augfix_raw10k_lnd_ours_topk50k_gtroi_exclude210_20260811 \
  --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
  --crop_mask_source wrist \
  --pose_estimator topk_ransac \
  --wrist_roi_source gt_wrist \
  --predicted_wrist_mask_mode binary_head \
  --surface_keys_per_part 50000 \
  --mask_keys_per_part 512 \
  --correspondence_similarity raw_dot \
  --correspondence_temperature 1.0 \
  --topk_max_correspondences 512 \
  --topk_min_correspondences 12 \
  --topk_ransac_iterations 2000 \
  --topk_ransac_reprojection_error 3.0 \
  --topk_ransac_confidence 0.999 \
  --topk_min_inliers 4 \
  --topk_min_inlier_fraction 0.0 \
  --topk_bfgs_refine 0 \
  --rotation_ensemble 0 \
  --amp 1 \
  --exclude_frame_ids 210 \
  --print_freq 20 \
  --fail_fast 1

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python submodules/RoboPEPP/eval_surfemb_wrist_lnd.py \
  --model cosine10k_best="${COS_DIR}/checkpoints/best_val_total.pt" \
  --model cosine10k_last="${COS_DIR}/checkpoints/last.pt" \
  --output_dir submodules/RoboPEPP/logs/augfix_cosine10k_lnd_ours_topk50k_gtroi_exclude210_20260811 \
  --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
  --crop_mask_source wrist \
  --pose_estimator topk_ransac \
  --wrist_roi_source gt_wrist \
  --predicted_wrist_mask_mode binary_head \
  --surface_keys_per_part 50000 \
  --mask_keys_per_part 512 \
  --correspondence_similarity cosine \
  --correspondence_temperature 0.1 \
  --topk_max_correspondences 512 \
  --topk_min_correspondences 12 \
  --topk_ransac_iterations 2000 \
  --topk_ransac_reprojection_error 3.0 \
  --topk_ransac_confidence 0.999 \
  --topk_min_inliers 4 \
  --topk_min_inlier_fraction 0.0 \
  --topk_bfgs_refine 0 \
  --rotation_ensemble 0 \
  --amp 1 \
  --exclude_frame_ids 210 \
  --print_freq 20 \
  --fail_fast 1
