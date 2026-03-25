#!/bin/bash
# =============================================================================
# Multi-Node Launcher for LongVT Compatible Training
# =============================================================================
# 使用方式：
#   bash launch_longvt_multinode.sh
#   bash launch_longvt_multinode.sh -- trainer.total_epochs=10
# =============================================================================

set -eo pipefail

# =============================================================================
# 节点配置（修改这里）
# =============================================================================
NODES="10.0.1.33,10.0.1.34"  # 逗号分隔的节点 IP，第一个是 head
GPUS_PER_NODE=8
RAY_PORT=6380

# =============================================================================
# 路径配置
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONDA_ACTIVATE="source /mnt/data/home/zhengshurong/miniconda3/etc/profile.d/conda.sh && conda activate verl"

# =============================================================================
# 解析节点列表
# =============================================================================
IFS=',' read -ra NODE_ARRAY <<< "$NODES"
HEAD_NODE="${NODE_ARRAY[0]}"
NNODES="${#NODE_ARRAY[@]}"

echo "========================================"
echo "Multi-Node LongVT Training"
echo "========================================"
echo "Head node: $HEAD_NODE"
echo "Total nodes: $NNODES"
echo "GPUs per node: $GPUS_PER_NODE"
echo "========================================"

# =============================================================================
# 停止旧的 Ray 集群
# =============================================================================
echo "Stopping old Ray clusters..."
for node in "${NODE_ARRAY[@]}"; do
    echo "  Stopping Ray on $node..."
    ssh "$node" "$CONDA_ACTIVATE && ray stop --force" 2>/dev/null || true
done
sleep 2

# =============================================================================
# 启动 Ray Head
# =============================================================================
echo "Starting Ray head on $HEAD_NODE..."
ssh "$HEAD_NODE" "$CONDA_ACTIVATE && \
    ray start --head \
    --port=$RAY_PORT \
    --num-gpus=$GPUS_PER_NODE \
    --block &"
sleep 5

# =============================================================================
# 启动 Ray Workers
# =============================================================================
for node in "${NODE_ARRAY[@]:1}"; do
    echo "Starting Ray worker on $node..."
    ssh "$node" "$CONDA_ACTIVATE && \
        ray start \
        --address=$HEAD_NODE:$RAY_PORT \
        --num-gpus=$GPUS_PER_NODE \
        --block &"
    sleep 2
done

echo "Waiting for Ray cluster to be ready..."
sleep 10

# =============================================================================
# 启动训练
# =============================================================================
echo "Starting training on head node..."
ssh "$HEAD_NODE" "cd $PROJECT_DIR && \
    $CONDA_ACTIVATE && \
    export NNODES=$NNODES && \
    bash examples/video_reasoning/run_longvt_compat.sh $*"

# =============================================================================
# 清理
# =============================================================================
echo "Training finished. Stopping Ray cluster..."
for node in "${NODE_ARRAY[@]}"; do
    ssh "$node" "$CONDA_ACTIVATE && ray stop --force" 2>/dev/null || true
done

echo "Done!"
