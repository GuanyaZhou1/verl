#!/bin/bash
# =============================================================================
# Slurm Multi-Node Launcher for LongVT Compatible Training
# =============================================================================
# 使用方式：
#   bash launch_longvt_slurm.sh --jobid 22322 --nodes "node33,node34"
# =============================================================================

set -eo pipefail

# =============================================================================
# 默认参数
# =============================================================================
JOBID=""
NODES=""
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GPUS_PER_NODE=8
RAY_PORT=6380
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONDA_ENV="verl"
CONDA_PATH="/mnt/data/home/zhengshurong/miniconda3"

# =============================================================================
# 参数解析
# =============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --jobid)
            JOBID="$2"
            shift 2
            ;;
        --nodes)
            NODES="$2"
            shift 2
            ;;
        --gpus-per-node)
            GPUS_PER_NODE="$2"
            shift 2
            ;;
        --)
            shift
            EXTRA_ARGS="$*"
            break
            ;;
        -h|--help)
            echo "Usage: bash launch_longvt_slurm.sh --jobid JOB_ID --nodes node1,node2 [-- extra_args]"
            exit 0
            ;;
        *)
            echo "ERROR: Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ -z "$JOBID" ]; then
    echo "ERROR: --jobid is required"
    exit 1
fi

if [ -z "$NODES" ]; then
    echo "ERROR: --nodes is required (e.g., --nodes node33,node34)"
    exit 1
fi

# =============================================================================
# 解析节点列表
# =============================================================================
IFS=',' read -ra NODE_LIST <<< "$NODES"
HEAD_NODE="${NODE_LIST[0]}"
NNODES=${#NODE_LIST[@]}

echo "===== LongVT Slurm Launcher ====="
echo "Job ID:        $JOBID"
echo "Head node:     $HEAD_NODE"
echo "Total nodes:   $NNODES (${NODE_LIST[*]})"
echo "GPUs per node: $GPUS_PER_NODE"
echo "================================="

# 获取 head IP（优先使用训练网络接口）
HEAD_IP=$(srun --jobid=$JOBID --overlap -w "$HEAD_NODE" -N1 -n1 bash -lc '
for nic in bond0.1573 bond0 bond1 br0 eth0; do
    ip=$(ip -o -4 addr show dev "$nic" up scope global 2>/dev/null | awk "NR==1 {print \$4}" | cut -d/ -f1)
    if [ -n "$ip" ]; then
        echo "$ip"
        exit 0
    fi
done
ip -o -4 addr show up scope global | awk "NR==1 {print \$4}" | cut -d/ -f1
')
echo "Head IP: $HEAD_IP"

# =============================================================================
# 生成启动脚本
# =============================================================================
LAUNCH_SCRIPT="$PROJECT_DIR/.ray_launch_longvt_$$.sh"
cat > "$LAUNCH_SCRIPT" << 'SCRIPT_EOF'
#!/bin/bash
set -e
source CONDA_PATH_PLACEHOLDER/etc/profile.d/conda.sh
conda activate CONDA_ENV_PLACEHOLDER

NODE_ROLE=$1
HEAD_IP=$2
RAY_PORT=$3
GPUS=$4
NNODES=$5
PROJECT_DIR=$6
shift 6
EXTRA_ARGS="$*"

if [ "$NODE_ROLE" = "head" ]; then
    unset ROCR_VISIBLE_DEVICES
    unset HIP_VISIBLE_DEVICES

    # NCCL 配置
    export NCCL_IB_DISABLE=0
    export NCCL_IB_HCA="^mlx5_bond,mlx5_6,mlx5_9"
    export NCCL_CROSS_NIC=1
    export NCCL_SOCKET_IFNAME=bond0
    export NCCL_SOCKET_FAMILY=AF_INET

    echo "[$(hostname)] Cleaning up old Ray..."
    ray stop --force 2>/dev/null || true
    pkill -9 -f "ray::" 2>/dev/null || true
    sleep 3

    echo "[$(hostname)] Starting Ray head..."
    ray start --head --port=$RAY_PORT --num-gpus=$GPUS --disable-usage-stats --node-ip-address=$HEAD_IP

    echo "[$(hostname)] Waiting for cluster ($NNODES nodes, $((NNODES * GPUS)) GPUs)..."
    EXPECTED=$((NNODES * GPUS))
    for i in $(seq 1 60); do
        NGPUS=$(python3 -c "import ray; ray.init(address='$HEAD_IP:$RAY_PORT'); print(int(ray.cluster_resources().get('GPU',0))); ray.shutdown()" 2>/dev/null || echo 0)
        echo "  [$i] GPUs: $NGPUS / $EXPECTED"
        [ "$NGPUS" -ge "$EXPECTED" ] && break
        sleep 5
    done

    echo "[$(hostname)] Starting training..."
    cd "$PROJECT_DIR"
    export NNODES=$NNODES
    export MASTER_ADDR=$HEAD_IP
    export MASTER_PORT=29500
    export RAY_ADDRESS=$HEAD_IP:$RAY_PORT
    bash examples/video_reasoning/run_longvt_compat.sh $EXTRA_ARGS

    echo "[$(hostname)] Training done, stopping Ray..."
    ray stop --force
else
    unset ROCR_VISIBLE_DEVICES
    unset HIP_VISIBLE_DEVICES

    # NCCL 配置
    export NCCL_IB_DISABLE=0
    export NCCL_IB_HCA="^mlx5_bond,mlx5_6,mlx5_9"
    export NCCL_CROSS_NIC=1
    export NCCL_SOCKET_IFNAME=bond0
    export NCCL_SOCKET_FAMILY=AF_INET

    echo "[$(hostname)] Cleaning up old Ray..."
    ray stop --force 2>/dev/null || true
    pkill -9 -f "ray::" 2>/dev/null || true
    sleep 3

    # 等待 head 端口就绪
    echo "[$(hostname)] Waiting for Ray head at $HEAD_IP:$RAY_PORT..."
    for i in $(seq 1 60); do
        if timeout 2 bash -c "echo > /dev/tcp/$HEAD_IP/$RAY_PORT" 2>/dev/null; then
            echo "[$(hostname)] Head is ready after $((i*2))s"
            break
        fi
        echo "[$(hostname)] Attempt $i: head not ready yet..."
        sleep 2
    done

    echo "[$(hostname)] Starting Ray worker..."
    ray start --address=$HEAD_IP:$RAY_PORT --num-gpus=$GPUS --disable-usage-stats

    echo "[$(hostname)] Worker started, keeping alive..."
    # 保持运行
    while true; do sleep 3600; done
fi
SCRIPT_EOF

sed -i "s|CONDA_PATH_PLACEHOLDER|$CONDA_PATH|g" "$LAUNCH_SCRIPT"
sed -i "s|CONDA_ENV_PLACEHOLDER|$CONDA_ENV|g" "$LAUNCH_SCRIPT"
chmod +x "$LAUNCH_SCRIPT"

echo "Launch script: $LAUNCH_SCRIPT"
echo ""

# =============================================================================
# 启动 worker 节点（后台）
# =============================================================================
WORKER_PIDS=()
for i in $(seq 1 $((NNODES - 1))); do
    worker="${NODE_LIST[$i]}"
    echo "Starting worker on $worker..."
    WORKER_LOG="$PROJECT_DIR/logs/worker_${worker}.log"
    mkdir -p "$PROJECT_DIR/logs"
    srun --jobid=$JOBID --overlap -w "$worker" -N1 -n1 bash "$LAUNCH_SCRIPT" worker "$HEAD_IP" "$RAY_PORT" "$GPUS_PER_NODE" "$NNODES" "$PROJECT_DIR" > "$WORKER_LOG" 2>&1 &
    WORKER_PIDS+=($!)
done

# =============================================================================
# 启动 head 节点（前台）
# =============================================================================
echo ""
echo "Starting head on $HEAD_NODE (foreground)..."
echo "============================================"
srun --jobid=$JOBID --overlap -w "$HEAD_NODE" -N1 -n1 bash "$LAUNCH_SCRIPT" head "$HEAD_IP" "$RAY_PORT" "$GPUS_PER_NODE" "$NNODES" "$PROJECT_DIR" $EXTRA_ARGS

# 清理
echo ""
echo "Cleaning up workers..."
for pid in "${WORKER_PIDS[@]}"; do
    kill $pid 2>/dev/null || true
done
rm -f "$LAUNCH_SCRIPT"
echo "Done!"
