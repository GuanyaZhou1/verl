#!/bin/bash
# =============================================================================
# 单节点恢复训练 - node36 (8x H200)
# 从 checkpoint 60 恢复
# =============================================================================
set -e

HEAD_NODE="node36"
GPUS=8
NNODES=1
PROJECT_DIR="/mnt/data/home/zhengshurong/project/verl"

EXPERIMENT_NAME="Qwen3_8B_dapo_kl0.3_bbox0.3_topp0.7_lr1e-6__0314_GAE_perturn"
PROJECT_NAME="video-reasoning-dapo"
CKPT_BASE="${PROJECT_DIR}/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}"

TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG_FILE="${PROJECT_DIR}/training_logs/training_${TIMESTAMP}.log"
mkdir -p "${PROJECT_DIR}/training_logs"

echo "===== 单节点恢复训练 ====="
echo "节点: $HEAD_NODE"
echo "实验: $EXPERIMENT_NAME"
echo "Checkpoint: $CKPT_BASE"
echo "最新 Step: $(cat $CKPT_BASE/latest_checkpointed_iteration.txt 2>/dev/null || echo 'N/A')"
echo "日志: $LOG_FILE"
echo "=========================="

# 在 node36 上启动
ssh $HEAD_NODE "bash -c '
set -e
source /mnt/data/home/zhengshurong/miniconda3/etc/profile.d/conda.sh
conda activate verl

# 清理旧进程
ray stop --force 2>/dev/null || true
pkill -f \"ray::\" 2>/dev/null || true
sleep 3

# 启动 Ray head
ray start --head --port=6380 --num-gpus=$GPUS --disable-usage-stats
sleep 5

cd $PROJECT_DIR

export NNODES=$NNODES
export N_GPUS=$GPUS
export RAY_ADDRESS=127.0.0.1:6380
export SKIP_VIDEO_CACHE=true
export RUN_EVAL=false

# NCCL 配置
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=\"^mlx5_bond,mlx5_6\"
export NCCL_CROSS_NIC=1
export NCCL_SOCKET_IFNAME=bond0
export NCCL_SOCKET_FAMILY=AF_INET
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_CUMEM_ENABLE=0
export TMPDIR=/tmp

export EXPERIMENT_NAME=\"$EXPERIMENT_NAME\"
export PROJECT_NAME=\"$PROJECT_NAME\"

bash examples/video_reasoning/run_video_reasoning_dapo_h200.sh \
    trainer.resume_mode=auto \
    trainer.experiment_name=\"$EXPERIMENT_NAME\" \
    trainer.project_name=\"$PROJECT_NAME\" \
    trainer.default_local_dir=\"$CKPT_BASE\"
'" 2>&1 | tee "$LOG_FILE" &

TRAIN_PID=$!
echo "训练进程 PID: $TRAIN_PID"
echo "$TRAIN_PID" > "${PROJECT_DIR}/.train_pid_node36"
echo "$LOG_FILE" > "${PROJECT_DIR}/.train_log_node36"

wait $TRAIN_PID
echo "===== 训练结束 ====="
