#!/bin/bash
#SBATCH --job-name=test_multinode
#SBATCH --partition=debug
#SBATCH --nodes=2
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=logs/test_multinode_%j.log
#SBATCH --error=logs/test_multinode_%j.log

set -eo pipefail

# =============================================================================
# 多节点 Ray 集群连通性测试
# =============================================================================
CONDA_PATH="/mnt/data/home/zhengshurong/miniconda3"
source "$CONDA_PATH/etc/profile.d/conda.sh"
conda activate verl

GPUS_PER_NODE=8
RAY_PORT=6380

# Slurm 自动提供节点列表
NODELIST=$(scontrol show hostnames $SLURM_JOB_NODELIST)
HEAD_NODE=$(echo "$NODELIST" | head -1)
WORKER_NODES=$(echo "$NODELIST" | tail -n +2)
NNODES=$SLURM_NNODES

HEAD_IP=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname -I | awk '{print $1}')

echo "===== sbatch Multi-Node Test ====="
echo "Job ID:      $SLURM_JOB_ID"
echo "Head node:   $HEAD_NODE ($HEAD_IP)"
echo "All nodes:   $(echo $NODELIST | tr '\n' ' ')"
echo "Total nodes: $NNODES"
echo "=================================="

# --- 测试 1: 节点间端口连通性 ---
echo ""
echo "===== Test 1: 节点间网络连通性 ====="
for worker in $WORKER_NODES; do
    echo "从 $worker 测试到 $HEAD_IP:$RAY_PORT ..."
    srun --nodes=1 --ntasks=1 -w "$worker" bash -c \
        "timeout 3 bash -c 'echo > /dev/tcp/$HEAD_IP/$RAY_PORT' 2>/dev/null && echo '  $worker -> $HEAD_IP:$RAY_PORT OK' || echo '  $worker -> $HEAD_IP:$RAY_PORT BLOCKED'"
done

# --- 启动 Ray Head ---
echo ""
echo "===== Test 2: 启动 Ray 集群 ====="
echo "Starting Ray head on $HEAD_NODE..."
srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" bash -c "
    source $CONDA_PATH/etc/profile.d/conda.sh && conda activate verl
    ray stop --force 2>/dev/null || true
    sleep 2
    ray start --head --port=$RAY_PORT --num-gpus=$GPUS_PER_NODE --disable-usage-stats --node-ip-address=$HEAD_IP
" &
HEAD_PID=$!
wait $HEAD_PID

# --- 启动 Ray Workers ---
for worker in $WORKER_NODES; do
    echo "Starting Ray worker on $worker..."
    srun --nodes=1 --ntasks=1 -w "$worker" bash -c "
        source $CONDA_PATH/etc/profile.d/conda.sh && conda activate verl
        ray stop --force 2>/dev/null || true
        sleep 2
        # 等待 head 就绪
        for i in \$(seq 1 30); do
            if timeout 2 bash -c 'echo > /dev/tcp/$HEAD_IP/$RAY_PORT' 2>/dev/null; then
                echo '  [$worker] Head ready after \${i}s'
                break
            fi
            echo '  [$worker] Waiting for head... attempt \$i'
            sleep 2
        done
        ray start --address=$HEAD_IP:$RAY_PORT --num-gpus=$GPUS_PER_NODE --disable-usage-stats
    " &
done
wait

# --- 检查集群 GPU 数量 ---
echo ""
echo "===== Test 3: 检查集群 GPU 数量 ====="
EXPECTED=$((NNODES * GPUS_PER_NODE))
srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" bash -c "
    source $CONDA_PATH/etc/profile.d/conda.sh && conda activate verl
    for i in \$(seq 1 30); do
        NGPUS=\$(python3 -c \"import ray; ray.init(address='$HEAD_IP:$RAY_PORT'); print(int(ray.cluster_resources().get('GPU',0))); ray.shutdown()\" 2>/dev/null || echo 0)
        echo \"  [\$i] GPUs: \$NGPUS / $EXPECTED\"
        if [ \"\$NGPUS\" -ge \"$EXPECTED\" ]; then
            echo '===== SUCCESS: 所有 GPU 已就绪 ====='
            break
        fi
        sleep 5
    done
"

# 清理
echo ""
echo "===== 清理 Ray ====="
for node in $HEAD_NODE $WORKER_NODES; do
    srun --nodes=1 --ntasks=1 -w "$node" bash -c "
        source $CONDA_PATH/etc/profile.d/conda.sh && conda activate verl
        ray stop --force 2>/dev/null || true
    " &
done
wait

echo "===== 测试完成 ====="
