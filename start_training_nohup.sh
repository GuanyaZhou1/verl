#!/bin/bash
# 使用 nohup 后台启动训练

set -e

HEAD_NODE="node15"
WORKER_NODES="node19"
HEAD_IP="10.1.3.15"
RAY_PORT=6380
GPUS=8
NNODES=2
PROJECT_DIR="/mnt/data/home/zhengshurong/project/verl"
LOG_DIR="$PROJECT_DIR/training_logs"
mkdir -p "$LOG_DIR"

EXPERIMENT_NAME="Qwen3_8B_dapo_kl0.3_bbox0.3_topp0.7_lr1e-6__0314_GAE_perturn"
PROJECT_NAME="video-reasoning-dapo"
CKPT_BASE="${PROJECT_DIR}/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/training_${TIMESTAMP}.log"

echo "===== 启动训练 =====" | tee "$LOG_FILE"
echo "日志文件: $LOG_FILE" | tee -a "$LOG_FILE"
echo "Checkpoint: $CKPT_BASE" | tee -a "$LOG_FILE"
echo "最新 Step: $(cat $CKPT_BASE/latest_checkpointed_iteration.txt 2>/dev/null || echo 'N/A')" | tee -a "$LOG_FILE"

# 启动 worker
echo "Starting worker on node19..." | tee -a "$LOG_FILE"
ssh node19 "source /mnt/data/home/zhengshurong/miniconda3/etc/profile.d/conda.sh && conda activate verl && \
    ray stop --force 2>/dev/null || true; sleep 2; \
    nohup ray start --address=$HEAD_IP:$RAY_PORT --num-gpus=$GPUS --disable-usage-stats > /tmp/ray_worker.log 2>&1 &"

# 启动 head
echo "Starting head on node15..." | tee -a "$LOG_FILE"
ssh node15 "source /mnt/data/home/zhengshurong/miniconda3/etc/profile.d/conda.sh && conda activate verl && \
    cd $PROJECT_DIR && \
    ray stop --force 2>/dev/null || true; sleep 2; \
    ray start --head --port=$RAY_PORT --num-gpus=$GPUS --disable-usage-stats --node-ip-address=$HEAD_IP && \
    sleep 15 && \
    export NNODES=$NNODES && \
    export N_GPUS=$GPUS && \
    export RAY_ADDRESS=$HEAD_IP:$RAY_PORT && \
    export SKIP_VIDEO_CACHE=true && \
    export RUN_EVAL=false && \
    export NCCL_IB_DISABLE=0 && \
    export NCCL_CROSS_NIC=1 && \
    export NCCL_SOCKET_IFNAME=bond0 && \
    export NCCL_SOCKET_FAMILY=AF_INET && \
    export MASTER_ADDR=$HEAD_IP && \
    export MASTER_PORT=29500 && \
    export EXPERIMENT_NAME='$EXPERIMENT_NAME' && \
    export PROJECT_NAME='$PROJECT_NAME' && \
    nohup bash examples/video_reasoning/run_video_reasoning_dapo_h200.sh \
        +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_IB_DISABLE=0 \
        +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_CROSS_NIC=1 \
        +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_SOCKET_IFNAME=bond0 \
        +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_SOCKET_FAMILY=AF_INET \
        +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_DEBUG=INFO \
        +ray_kwargs.ray_init.runtime_env.env_vars.MASTER_ADDR=$HEAD_IP \
        +ray_kwargs.ray_init.runtime_env.env_vars.MASTER_PORT=29500 \
        +ray_kwargs.ray_init.runtime_env.env_vars.TORCH_NCCL_AVOID_RECORD_STREAMS=1 \
        +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_CUMEM_ENABLE=0 \
        +ray_kwargs.ray_init.runtime_env.env_vars.TMPDIR=/tmp \
        trainer.resume_mode=auto \
        trainer.experiment_name='$EXPERIMENT_NAME' \
        trainer.project_name='$PROJECT_NAME' \
        trainer.default_local_dir='$CKPT_BASE' \
        >> $LOG_FILE 2>&1 &"

echo "训练已在后台启动，日志: $LOG_FILE" | tee -a "$LOG_FILE"
