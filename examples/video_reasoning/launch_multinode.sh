#!/bin/bash
# =============================================================================
# Multi-Node Launcher for Video Reasoning DAPO Training
# =============================================================================
# 通过 SSH 在多台机器上启动 Ray 集群，然后在 head 节点上运行训练脚本。
#
# 前提条件：
#   - 所有节点之间 SSH 免密登录
#   - 共享文件系统（所有节点可访问同一项目目录）
#   - InfiniBand 互联（mlx5_2/5/6/7）
#
# 使用方式：
#   bash launch_multinode.sh --nodes "10.0.0.1,10.0.0.2" [--gpus-per-node 8]
#   bash launch_multinode.sh --node-file nodes.txt [--gpus-per-node 8]
#   bash launch_multinode.sh --nodes "10.0.0.1,10.0.0.2" -- trainer.total_epochs=2
#
# 说明：
#   - 第一个 IP 作为 Ray head 节点
#   - `--` 后的参数会透传给训练脚本 run_video_reasoning_dapo.sh
#   - 视频缓存步骤在共享文件系统下只需提前执行一次，launcher 会自动跳过
# =============================================================================

set -eo pipefail

# =============================================================================
# 默认参数
# =============================================================================
NODES=""
NODE_FILE=""
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEFAULT_NODE_FILE="$SCRIPT_DIR/nodes.txt"
GPUS_PER_NODE=8
SSH_USER=""
NCCL_IFNAME="bond0"
RAY_PORT=6380   # 6379 被系统 Redis 占用，改用 6380
EXTRA_ARGS=""
PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
# 远程节点上激活 conda 再执行 ray/python（SSH 非交互 shell 不会自动激活 conda）
CONDA_ACTIVATE="${CONDA_ACTIVATE:-source /share_data/gyzhou/anaconda3/etc/profile.d/conda.sh && conda activate verl_clone}"
REMOTE_PREFIX="$CONDA_ACTIVATE && "

# =============================================================================
# 参数解析
# =============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
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
        --ssh-user)
            SSH_USER="$2"
            shift 2
            ;;
        --nccl-ifname)
            NCCL_IFNAME="$2"
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
            echo "Usage: bash launch_multinode.sh [options] [-- extra_training_args]"
            echo ""
            echo "If no --nodes or --node-file is given, defaults to nodes.txt in the same directory."
            echo ""
            echo "Options:"
            echo "  --nodes IP1,IP2,...    Comma-separated list of node IPs (first is head)"
            echo "  --node-file FILE       File with one IP per line (first is head, default: nodes.txt)"
            echo "  --gpus-per-node N      Number of GPUs per node (default: 8)"
            echo "  --ssh-user USER        SSH username (default: current user)"
            echo "  --nccl-ifname IFACE    NCCL socket interface (default: bond0)"
            echo "  --ray-port PORT        Ray head port (default: 6379)"
            echo "  -- ARGS                Extra args passed to training script"
            echo ""
            echo "Environment (for conda users, set before running):"
            echo "  CONDA_ACTIVATE          Activate conda on remote nodes so 'ray' is in PATH."
            echo "                          Example: export CONDA_ACTIVATE=\"source ~/miniconda3/etc/profile.d/conda.sh && conda activate verl_clone\""
            echo ""
            echo "Examples:"
            echo "  bash launch_multinode.sh --nodes \"10.96.11.1,10.96.11.2\" --gpus-per-node 8"
            echo "  bash launch_multinode.sh --node-file nodes.txt -- trainer.total_epochs=2"
            exit 0
            ;;
        *)
            echo "ERROR: Unknown option: $1"
            echo "Use --help for usage information."
            exit 1
            ;;
    esac
done

# =============================================================================
# 解析节点列表
# =============================================================================
if [ -n "$NODE_FILE" ]; then
    if [ ! -f "$NODE_FILE" ]; then
        echo "ERROR: Node file not found: $NODE_FILE"
        exit 1
    fi
    # 读取文件，过滤注释和空行
    NODE_LIST=()
    while IFS= read -r line || [ -n "$line" ]; do
        line=$(echo "$line" | sed 's/#.*//' | xargs)  # 去注释、去首尾空格
        [ -n "$line" ] && NODE_LIST+=("$line")
    done < "$NODE_FILE"
elif [ -n "$NODES" ]; then
    IFS=',' read -ra NODE_LIST <<< "$NODES"
elif [ -f "$DEFAULT_NODE_FILE" ]; then
    echo "No --nodes or --node-file specified, using default: $DEFAULT_NODE_FILE"
    NODE_LIST=()
    while IFS= read -r line || [ -n "$line" ]; do
        line=$(echo "$line" | sed 's/#.*//' | xargs)
        [ -n "$line" ] && NODE_LIST+=("$line")
    done < "$DEFAULT_NODE_FILE"
else
    echo "ERROR: Must specify --nodes or --node-file, or create $DEFAULT_NODE_FILE"
    echo "Use --help for usage information."
    exit 1
fi

if [ ${#NODE_LIST[@]} -lt 1 ]; then
    echo "ERROR: No nodes specified"
    exit 1
fi

HEAD_NODE="${NODE_LIST[0]}"
NNODES=${#NODE_LIST[@]}

# SSH 前缀
ssh_cmd() {
    local node="$1"
    shift
    if [ -n "$SSH_USER" ]; then
        ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "${SSH_USER}@${node}" "$@"
    else
        ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "${node}" "$@"
    fi
}

# =============================================================================
# 打印配置
# =============================================================================
echo "===== Multi-Node Launcher ====="
echo "Project dir:   $PROJECT_DIR"
echo "Head node:     $HEAD_NODE"
echo "Total nodes:   $NNODES"
echo "GPUs per node: $GPUS_PER_NODE"
echo "Ray port:      $RAY_PORT"
echo "NCCL ifname:   $NCCL_IFNAME"
[ -n "$CONDA_ACTIVATE" ] && echo "Conda activate: set (will run on remote before ray/python)"
echo "Nodes:"
for i in "${!NODE_LIST[@]}"; do
    if [ "$i" -eq 0 ]; then
        echo "  [$i] ${NODE_LIST[$i]} (head)"
    else
        echo "  [$i] ${NODE_LIST[$i]} (worker)"
    fi
done
if [ -n "$EXTRA_ARGS" ]; then
    echo "Extra args:    $EXTRA_ARGS"
fi
echo "==============================="
echo ""

# =============================================================================
# Step 1: 验证 SSH 连通性
# =============================================================================
echo "===== Step 1: Verifying SSH connectivity ====="
FAILED_NODES=()
for node in "${NODE_LIST[@]}"; do
    if ssh_cmd "$node" "echo ok" &>/dev/null; then
        echo "  [OK] $node"
    else
        echo "  [FAIL] $node"
        FAILED_NODES+=("$node")
    fi
done

if [ ${#FAILED_NODES[@]} -gt 0 ]; then
    echo ""
    echo "ERROR: Cannot reach the following nodes via SSH:"
    for node in "${FAILED_NODES[@]}"; do
        echo "  - $node"
    done
    echo "Please check SSH key configuration and network connectivity."
    exit 1
fi
echo ""

# =============================================================================
# Step 2: Cleanup trap
# =============================================================================
cleanup() {
    echo ""
    echo "===== Cleaning up: Stopping Ray on all nodes ====="
    for node in "${NODE_LIST[@]}"; do
        echo "  Stopping Ray on $node ..."
        ssh_cmd "$node" "${REMOTE_PREFIX}ray stop 2>/dev/null" &>/dev/null || true
    done
    echo "===== Cleanup complete ====="
}
trap cleanup EXIT INT TERM

# =============================================================================
# Step 3: 停止旧 Ray 进程
# =============================================================================
echo "===== Step 2: Stopping existing Ray processes ====="
for node in "${NODE_LIST[@]}"; do
    echo "  Stopping Ray on $node ..."
    ssh_cmd "$node" "${REMOTE_PREFIX}ray stop 2>/dev/null" || true
done
sleep 2
echo ""

# =============================================================================
# NCCL 环境变量（在 ray start 之前设置，使所有 worker 进程继承）
# =============================================================================
NCCL_ENVS="export LD_LIBRARY_PATH=/usr/local/cuda-13.1/compat:\$LD_LIBRARY_PATH && \
export NCCL_SOCKET_IFNAME=$NCCL_IFNAME && \
export NCCL_IB_DISABLE=0 && \
export NCCL_DEBUG=WARN && \
export TORCH_NCCL_AVOID_RECORD_STREAMS=1 && \
export NCCL_CUMEM_ENABLE=0"

# =============================================================================
# Step 4: 启动 Ray head 节点
# =============================================================================
echo "===== Step 3: Starting Ray head on $HEAD_NODE ====="
ssh_cmd "$HEAD_NODE" "${REMOTE_PREFIX}$NCCL_ENVS && cd $PROJECT_DIR && ray start --head --port=$RAY_PORT \
    --dashboard-host=0.0.0.0 --num-gpus=$GPUS_PER_NODE --disable-usage-stats"
echo ""

# =============================================================================
# Step 5: 启动 Ray worker 节点
# =============================================================================
if [ $NNODES -gt 1 ]; then
    echo "===== Step 4: Starting Ray workers ====="
    for i in $(seq 1 $((NNODES - 1))); do
        worker="${NODE_LIST[$i]}"
        echo "  Starting Ray worker on $worker (connecting to $HEAD_NODE:$RAY_PORT) ..."
        ssh_cmd "$worker" "${REMOTE_PREFIX}$NCCL_ENVS && cd $PROJECT_DIR && ray start --address=$HEAD_NODE:$RAY_PORT \
            --num-gpus=$GPUS_PER_NODE --disable-usage-stats"
    done
    echo ""
fi

# =============================================================================
# Step 6: 等待集群就绪
# =============================================================================
echo "===== Step 5: Waiting for Ray cluster to be ready ====="
EXPECTED_GPUS=$((NNODES * GPUS_PER_NODE))

# 在 head 节点上检查集群状态
ssh_cmd "$HEAD_NODE" "${REMOTE_PREFIX}cd $PROJECT_DIR && python3 -u -c \"
import ray
import time
import sys

ray.init(address='$HEAD_NODE:$RAY_PORT')

timeout = 300
start = time.time()
while True:
    resources = ray.cluster_resources()
    num_gpus = int(resources.get('GPU', 0))
    num_nodes = len(ray.nodes())
    alive_nodes = sum(1 for n in ray.nodes() if n['Alive'])

    elapsed = time.time() - start
    print(f'  Cluster: {alive_nodes}/{$NNODES} nodes, {num_gpus}/{$EXPECTED_GPUS} GPUs ({elapsed:.0f}s)')

    if num_gpus >= $EXPECTED_GPUS and alive_nodes >= $NNODES:
        print(f'  Cluster ready!')
        break
    if elapsed > timeout:
        print(f'ERROR: Timeout waiting for cluster. Expected {$NNODES} nodes with {$EXPECTED_GPUS} GPUs.')
        print(f'  Got {alive_nodes} nodes with {num_gpus} GPUs.')
        ray.shutdown()
        sys.exit(1)
    time.sleep(5)

ray.shutdown()
\""
echo ""

# =============================================================================
# Step 7: 启动训练
# =============================================================================
echo "===== Step 6: Launching training on head node ($HEAD_NODE) ====="
echo "  NNODES=$NNODES, N_GPUS=$GPUS_PER_NODE, SKIP_VIDEO_CACHE=true"
echo "  Ray Dashboard: http://$HEAD_NODE:8265"
echo ""

# 通过 Hydra config 注入 NCCL 变量到 ray.init(runtime_env)，确保所有节点 worker 都能获取
NCCL_HYDRA_ARGS="+ray_kwargs.ray_init.runtime_env.env_vars.NCCL_SOCKET_IFNAME=$NCCL_IFNAME"
NCCL_HYDRA_ARGS="$NCCL_HYDRA_ARGS +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_IB_DISABLE=0"
NCCL_HYDRA_ARGS="$NCCL_HYDRA_ARGS +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_DEBUG=WARN"
NCCL_HYDRA_ARGS="$NCCL_HYDRA_ARGS +ray_kwargs.ray_init.runtime_env.env_vars.TORCH_NCCL_AVOID_RECORD_STREAMS=1"
NCCL_HYDRA_ARGS="$NCCL_HYDRA_ARGS +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_CUMEM_ENABLE=0"
NCCL_HYDRA_ARGS="$NCCL_HYDRA_ARGS +ray_kwargs.ray_init.runtime_env.env_vars.TIKTOKEN_CACHE_DIR=/data_gpu/gyzhou/tmp/tiktoken_cache"
NCCL_HYDRA_ARGS="$NCCL_HYDRA_ARGS +ray_kwargs.ray_init.runtime_env.env_vars.TMPDIR=/tmp"

ssh_cmd "$HEAD_NODE" "${REMOTE_PREFIX}cd $PROJECT_DIR && \
    $NCCL_ENVS && \
    export NNODES=$NNODES && \
    export N_GPUS=$GPUS_PER_NODE && \
    export SKIP_VIDEO_CACHE=true && \
    export RAY_ADDRESS=$HEAD_NODE:$RAY_PORT && \
    bash examples/video_reasoning/run_video_reasoning_dapo.sh $NCCL_HYDRA_ARGS $EXTRA_ARGS"
