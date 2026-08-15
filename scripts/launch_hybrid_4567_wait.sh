#!/usr/bin/env bash
set -u
set -o pipefail

cd /mnt/iMVR/daiyun/shuojue-temp/code/multi-hmr

WAIT_LOG="logs/hcce_crop_fixbf16_hybrid_4567_wait.log"
mkdir -p logs
exec > >(tee -a "$WAIT_LOG") 2>&1

echo "[$(date)] waiting for GPUs 4,5,6 to be free; current GPU7 single-card run continues until replacement starts"

while true; do
  busy=0
  for gpu in 4 5 6; do
    if nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits | grep -Eq "[0-9]"; then
      busy=1
    fi
  done

  if [ "$busy" = 0 ]; then
    break
  fi

  echo "[$(date)] GPUs 4-6 still busy:"
  nvidia-smi -i 4,5,6 --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits || true
  sleep 60
done

echo "[$(date)] GPUs 4-6 free; stopping temporary GPU7 single-card hybrid run"
tmux kill-session -t hcce_crop_fixbf16_hybrid_7_bs56 2>/dev/null || true
sleep 5

for bs in 56 32; do
  name="instrument_hcce_crop_surgpose_keypoint_hybrid_gpu4567_bs${bs}_bf16_fromscratch"
  train_log="logs/${name}_train.log"
  echo "[$(date)] starting 4-card hybrid from scratch: batch_size=${bs}, log=${train_log}"
  CUDA_VISIBLE_DEVICES=4,5,6,7 PYOPENGL_PLATFORM=egl \
    python -m torch.distributed.run --nproc_per_node=4 --master_port=29667 \
      train_instrument_hcce_crop_surgpose_keypoint_dpt.py \
      --save_dir logs \
      --name "$name" \
      --batch_size "$bs" \
      --val_batch_size 8 \
      --num_workers 0 \
      --log_freq 10 \
      --val_freq 1000 \
      --ckpt_freq 1000 \
      --max_iter 60000 \
      --amp 1 \
    2>&1 | tee "$train_log"
  status=${PIPESTATUS[0]}
  echo "[$(date)] batch_size=${bs} exited with status ${status}"
  if [ "$status" = 0 ]; then
    exit 0
  fi
  echo "[$(date)] trying next batch size if available"
done

exit 1
