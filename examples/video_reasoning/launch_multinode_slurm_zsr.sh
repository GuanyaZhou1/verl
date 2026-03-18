#!/bin/bash
# =============================================================================
# Multi-Node Launcher for Slurm (using srun --overlap)
# =============================================================================
# 在已经 salloc 占用的节点上启动 Ray 集群并运行训练
#
# 使用方式：
#   bash launch_multinode_slurm.sh --jobid 21690
#   bash launch_multinode_slurm.sh --jobid 21690 --nodes "node4,node35"
#
# =============================================================================

set -eo pipefail

# =============================================================================
# 默认参数
# =============================================================================
JOBID=""
NODES=""
NODE_FILE=""
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEFAULT_NODE_FILE="$SCRIPT_DIR/nodes.txt"
GPUS_PER_NODE=8
RAY_PORT=6380
EXTRA_ARGS=""
PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
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
        --node-file)
            NODE_FILE="$2"
            shift 2
            ;;
        --gpus-per-node)
            GPUS_PER_NODE="$2"
            shift 2
            ;;
        --ray-port)
            RAY_PORT="$2"
            shift 2
            ;;
        --)
            shift
            EXTRA_ARGS="$*"
            break
            ;;
        -h|--help)
            echo "Usage: bash launch_multinode_slurm.sh --jobid JOB_ID [options] [-- extra_args]"
            echo ""
            echo "Options:"
            echo "  --jobid JOB_ID         Slurm job ID from salloc (required)"
            echo "  --nodes node1,node2    Comma-separated list of nodes (first is head)"
            echo "  --node-file FILE       File with one node per line (default: nodes.txt)"
            echo "  --gpus-per-node N      Number of GPUs per node (default: 8)"
            echo "  --ray-port PORT        Ray head port (default: 6380)"
            echo "  -- ARGS                Extra args passed to training script"
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

# =============================================================================
# 解析节点列表
# =============================================================================
if [ -n "$NODE_FILE" ]; then
    [ ! -f "$NODE_FILE" ] && echo "ERROR: Node file not found: $NODE_FILE" && exit 1
    NODE_LIST=($(grep -v '^#' "$NODE_FILE" | grep -v '^$' | xargs))
elif [ -n "$NODES" ]; then
    IFS=',' read -ra NODE_LIST <<< "$NODES"
elif [ -f "$DEFAULT_NODE_FILE" ]; then
    echo "Using default node file: $DEFAULT_NODE_FILE"
    NODE_LIST=($(grep -v '^#' "$DEFAULT_NODE_FILE" | grep -v '^$' | xargs))
else
    echo "ERROR: No nodes specified"
    exit 1
fi

HEAD_NODE="${NODE_LIST[0]}"
NNODES=${#NODE_LIST[@]}

echo "===== Slurm Multi-Node Launcher ====="
echo "Job ID:        $JOBID"
echo "Head node:     $HEAD_NODE"
echo "Total nodes:   $NNODES (${NODE_LIST[*]})"
echo "GPUs per node: $GPUS_PER_NODE"
echo "======================================"
echo ""

# 获取 head IP
HEAD_IP=$(srun --jobid=$JOBID --overlap -w "$HEAD_NODE" -n1 hostname -I | awk '{print $1}')
echo "Head IP: $HEAD_IP"

# =============================================================================
# 生成启动脚本（放在共享目录）
# =============================================================================
LAUNCH_SCRIPT="$PROJECT_DIR/.ray_launch_$$.sh"
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
    # 清除可能冲突的环境变量
    unset ROCR_VISIBLE_DEVICES
    unset HIP_VISIBLE_DEVICES

    echo "[$(hostname)] Starting Ray head..."
    ray stop --force 2>/dev/null || true
    sleep 2
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
    export N_GPUS=$GPUS
    export SKIP_VIDEO_CACHE=true
    export RAY_ADDRESS=$HEAD_IP:$RAY_PORT

    # =============================================================================
    # NCCL IB 配置 - 强制使用 InfiniBand
    # =============================================================================
    export NCCL_IB_DISABLE=0
    export NCCL_IB_HCA="^mlx5_bond,mlx5_6"      # 排除 bond 和有问题的 mlx5_6
    export NCCL_CROSS_NIC=1                      # 允许跨 NIC 通信
    export NCCL_SOCKET_IFNAME=bond0              # fallback 用的以太网接口
    export NCCL_SOCKET_FAMILY=AF_INET            # 强制 IPv4

    # Gloo 配置
    export MASTER_ADDR=$HEAD_IP
    export MASTER_PORT=29500

    # 通过 Hydra 参数注入环境变量到 Ray runtime_env
    # 确保所有 Ray worker 都能获取这些 NCCL 配置
    # 注意：NCCL_IB_HCA 和 NCCL_DEBUG_SUBSYS 含特殊字符，通过 shell export 设置，不走 Hydra
    HYDRA_ENV_ARGS="+ray_kwargs.ray_init.runtime_env.env_vars.NCCL_IB_DISABLE=0"
    HYDRA_ENV_ARGS="$HYDRA_ENV_ARGS +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_CROSS_NIC=1"
    HYDRA_ENV_ARGS="$HYDRA_ENV_ARGS +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_SOCKET_IFNAME=bond0"
    HYDRA_ENV_ARGS="$HYDRA_ENV_ARGS +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_SOCKET_FAMILY=AF_INET"
    HYDRA_ENV_ARGS="$HYDRA_ENV_ARGS +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_DEBUG=INFO"
    HYDRA_ENV_ARGS="$HYDRA_ENV_ARGS +ray_kwargs.ray_init.runtime_env.env_vars.MASTER_ADDR=$HEAD_IP"
    HYDRA_ENV_ARGS="$HYDRA_ENV_ARGS +ray_kwargs.ray_init.runtime_env.env_vars.MASTER_PORT=29500"
    HYDRA_ENV_ARGS="$HYDRA_ENV_ARGS +ray_kwargs.ray_init.runtime_env.env_vars.TORCH_NCCL_AVOID_RECORD_STREAMS=1"
    HYDRA_ENV_ARGS="$HYDRA_ENV_ARGS +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_CUMEM_ENABLE=0"
    HYDRA_ENV_ARGS="$HYDRA_ENV_ARGS +ray_kwargs.ray_init.runtime_env.env_vars.TMPDIR=/tmp"

    bash examples/video_reasoning/run_video_reasoning_dapo_h200_zsr_gdpo.sh $HYDRA_ENV_ARGS $EXTRA_ARGS

    echo "[$(hostname)] Training complete!"
else
    # 清除可能冲突的环境变量
    unset ROCR_VISIBLE_DEVICES
    unset HIP_VISIBLE_DEVICES

    echo "[$(hostname)] Starting Ray worker, connecting to $HEAD_IP:$RAY_PORT..."
    ray stop --force 2>/dev/null || true
    sleep 2
    # 等待 head 启动
    sleep 10
    ray start --address=$HEAD_IP:$RAY_PORT --num-gpus=$GPUS --disable-usage-stats
    echo "[$(hostname)] Worker started, keeping alive..."
    # 保持运行
    while true; do sleep 3600; done
fi
SCRIPT_EOF

# 替换占位符
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
    srun --jobid=$JOBID --overlap -w "$worker" -n1 bash "$LAUNCH_SCRIPT" worker "$HEAD_IP" "$RAY_PORT" "$GPUS_PER_NODE" "$NNODES" "$PROJECT_DIR" &
    WORKER_PIDS+=($!)
done

# =============================================================================
# 启动 head 节点（前台）
# =============================================================================
echo ""
echo "Starting head on $HEAD_NODE (foreground)..."
echo "============================================"
srun --jobid=$JOBID --overlap -w "$HEAD_NODE" -n1 bash "$LAUNCH_SCRIPT" head "$HEAD_IP" "$RAY_PORT" "$GPUS_PER_NODE" "$NNODES" "$PROJECT_DIR" $EXTRA_ARGS

# 清理
echo ""
echo "Cleaning up workers..."
for pid in "${WORKER_PIDS[@]}"; do
    kill $pid 2>/dev/null || true
done
rm -f "$LAUNCH_SCRIPT"
echo "Done!"
