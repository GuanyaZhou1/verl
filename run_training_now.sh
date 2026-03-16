#!/bin/bash
source /mnt/data/home/zhengshurong/miniconda3/etc/profile.d/conda.sh
conda activate verl
cd /mnt/data/home/zhengshurong/project/verl

export NNODES=2
export N_GPUS=8
export RAY_ADDRESS=10.1.3.15:6380
export SKIP_VIDEO_CACHE=true
export RUN_EVAL=false
export NCCL_IB_DISABLE=0
export NCCL_CROSS_NIC=1
export NCCL_SOCKET_IFNAME=bond0
export NCCL_SOCKET_FAMILY=AF_INET
export MASTER_ADDR=10.1.3.15
export MASTER_PORT=29500
export EXPERIMENT_NAME='Qwen3_8B_dapo_kl0.3_bbox0.3_topp0.7_lr1e-6__0314_GAE_perturn'
export PROJECT_NAME='video-reasoning-dapo'

LOG_FILE="/mnt/data/home/zhengshurong/project/verl/training_logs/training_$(date +%Y%m%d_%H%M%S).log"
echo "Starting training at $(date), log: $LOG_FILE"

bash examples/video_reasoning/run_video_reasoning_dapo_h200.sh \
    +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_IB_DISABLE=0 \
    +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_CROSS_NIC=1 \
    +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_SOCKET_IFNAME=bond0 \
    +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_SOCKET_FAMILY=AF_INET \
    +ray_kwargs.ray_init.runtime_env.env_vars.MASTER_ADDR=10.1.3.15 \
    +ray_kwargs.ray_init.runtime_env.env_vars.MASTER_PORT=29500 \
    trainer.resume_mode=auto \
    'trainer.experiment_name=Qwen3_8B_dapo_kl0.3_bbox0.3_topp0.7_lr1e-6__0314_GAE_perturn' \
    trainer.project_name=video-reasoning-dapo \
    'trainer.default_local_dir=/mnt/data/home/zhengshurong/project/verl/checkpoints/video-reasoning-dapo/Qwen3_8B_dapo_kl0.3_bbox0.3_topp0.7_lr1e-6__0314_GAE_perturn' \
    >> "$LOG_FILE" 2>&1
