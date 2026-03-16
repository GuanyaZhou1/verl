#!/bin/bash
set -e
source /mnt/data/home/zhengshurong/miniconda3/etc/profile.d/conda.sh
conda activate verl

cd /mnt/data/home/zhengshurong/project/verl

export NNODES=1
export N_GPUS=8
export RAY_ADDRESS=10.0.1.36:6380
export SKIP_VIDEO_CACHE=true
export RUN_EVAL=false
export NCCL_IB_DISABLE=0
export NCCL_CROSS_NIC=1
export NCCL_SOCKET_IFNAME=bond0
export NCCL_SOCKET_FAMILY=AF_INET
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_CUMEM_ENABLE=0
export TMPDIR=/tmp
export EXPERIMENT_NAME='Qwen3_8B_dapo_kl0.3_bbox0.3_topp0.7_lr1e-6__0314_GAE_perturn'
export PROJECT_NAME='video-reasoning-dapo'

bash examples/video_reasoning/run_video_reasoning_dapo_h200.sh \
    trainer.resume_mode=auto \
    trainer.experiment_name='Qwen3_8B_dapo_kl0.3_bbox0.3_topp0.7_lr1e-6__0314_GAE_perturn' \
    trainer.project_name='video-reasoning-dapo' \
    trainer.default_local_dir='/mnt/data/home/zhengshurong/project/verl/checkpoints/video-reasoning-dapo/Qwen3_8B_dapo_kl0.3_bbox0.3_topp0.7_lr1e-6__0314_GAE_perturn'
