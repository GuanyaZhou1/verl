#!/bin/bash
# =============================================================================
# Multi-Node Launcher for dapo_0328_wg.sh
# =============================================================================
# 适配 wg 机器集群（gpu005 等），通过 SSH 启动 Ray 集群后运行训练。
#
# 与 launch_multinode.sh 的区别：
#   - conda 路径适配 wg 机器: /share_data/gyzhou/anaconda3, env=verl_clone
#   - NCCL IB HCA 排除列表适配 wg 机器网络拓扑
#   - TMPDIR/RAY_TEMPDIR 使用 /share_data/gyzhou/.tmp
#   - 训练脚本指向 dapo_0328_wg.sh
#
# 使用方式：
#   bash launch_multinode_0328_wg.sh --nodes "10.96.11.5,10.96.11.6"
#   bash launch_multinode_0328_wg.sh --node-file nodes_wg.txt
#   bash launch_multinode_0328_wg.sh --nodes "10.96.11.5,10.96.11.6" -- trainer.total_epochs=2
# =============================================================================

set -eo pipefail

# =============================================================================
# 默认参数
# =============================================================================
NODES=""
NODE_FILE=""
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEFAULT_NODE_FILE="$SCRIPT_DIR/nodes_wg.txt"
GPUS_PER_NODE=8
SSH_USER=""
NCCL_IFNAME="bond0"
RAY_PORT=6380
RAY_DASHBOARD_PORT=""
RAY_CLIENT_SERVER_PORT=""
RAY_DASHBOARD_AGENT_LISTEN_PORT=""
EXTRA_ARGS=""
PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
TRAINING_SCRIPT="examples/video_reasoning/dapo_0328_wg.sh"

# wg 机器的 conda 环境
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
        --dashboard-port)
            RAY_DASHBOARD_PORT="$2"
            shift 2
            ;;
        --ray-client-server-port)
            RAY_CLIENT_SERVER_PORT="$2"
            shift 2
            ;;
        --dashboard-agent-listen-port)
            RAY_DASHBOARD_AGENT_LISTEN_PORT="$2"
            shift 2
            ;;
        --script)
            TRAINING_SCRIPT="$2"
            shift 2
            ;;
        --)
            shift
            EXTRA_ARGS="$*"
            break
            ;;
        -h|--help)
            echo "Usage: bash launch_multinode_0328_wg.sh [options] [-- extra_training_args]"
            echo ""
            echo "Options:"
            echo "  --nodes IP1,IP2,...    Comma-separated node IPs (first is head)"
            echo "  --node-file FILE       File with one IP per line (default: nodes_wg.txt)"
            echo "  --gpus-per-node N      GPUs per node (default: 8)"
            echo "  --ssh-user USER        SSH username (default: current user)"
            echo "  --nccl-ifname IFACE    NCCL socket interface (default: bond0)"
            echo "  --ray-port PORT        Ray head port (default: 6380)"
            echo "  --dashboard-port PORT  Ray dashboard port (default: ray-port + 1000)"
            echo "  --ray-client-server-port PORT"
            echo "                         Ray client server port (default: ray-port + 2000)"
            echo "  --dashboard-agent-listen-port PORT"
            echo "                         Dashboard agent HTTP port (default: ray-port + 3000)"
            echo "  --script PATH          Training script relative to project root"
            echo "                         (default: examples/video_reasoning/dapo_0328_wg.sh)"
            echo "  -- ARGS                Extra args passed to training script"
            echo ""
            echo "Examples:"
            echo "  bash launch_multinode_0328_wg.sh --nodes \"10.96.11.5,10.96.11.6\""
            echo "  bash launch_multinode_0328_wg.sh --nodes \"10.96.11.5\" --gpus-per-node 8"
            echo "  bash launch_multinode_0328_wg.sh --node-file nodes_wg.txt -- trainer.total_epochs=2"
            exit 0
            ;;
        *)
            echo "ERROR: Unknown option: $1"
            echo "Use --help for usage."
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
    NODE_LIST=()
    while IFS= read -r line || [ -n "$line" ]; do
        line=$(echo "$line" | sed 's/#.*//' | xargs)
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
    echo "Use --help for usage."
    exit 1
fi

if [ ${#NODE_LIST[@]} -lt 1 ]; then
    echo "ERROR: No nodes specified"
    exit 1
fi

HEAD_NODE="${NODE_LIST[0]}"
NNODES=${#NODE_LIST[@]}
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-$((RAY_PORT + 1000))}"
RAY_CLIENT_SERVER_PORT="${RAY_CLIENT_SERVER_PORT:-$((RAY_PORT + 2000))}"
RAY_DASHBOARD_AGENT_LISTEN_PORT="${RAY_DASHBOARD_AGENT_LISTEN_PORT:-$((RAY_PORT + 3000))}"

# SSH helper
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
echo "===== Multi-Node Launcher (wg) ====="
echo "Project dir:     $PROJECT_DIR"
echo "Training script: $TRAINING_SCRIPT"
echo "Head node:       $HEAD_NODE"
echo "Total nodes:     $NNODES"
echo "GPUs per node:   $GPUS_PER_NODE"
echo "Ray port:        $RAY_PORT"
echo "Dashboard port:  $RAY_DASHBOARD_PORT"
echo "Client port:     $RAY_CLIENT_SERVER_PORT"
echo "Agent HTTP port: $RAY_DASHBOARD_AGENT_LISTEN_PORT"
echo "NCCL ifname:     $NCCL_IFNAME"
echo "Nodes:"
for i in "${!NODE_LIST[@]}"; do
    if [ "$i" -eq 0 ]; then
        echo "  [$i] ${NODE_LIST[$i]} (head)"
    else
        echo "  [$i] ${NODE_LIST[$i]} (worker)"
    fi
done
[ -n "$EXTRA_ARGS" ] && echo "Extra args:      $EXTRA_ARGS"
echo "======================================"
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
# Step 4: 检查目标节点上是否仍有其他 Ray 集群残留
# =============================================================================
echo "===== Step 3: Checking for remaining Ray daemons ====="
for node in "${NODE_LIST[@]}"; do
    REMAINING_RAY_DAEMONS="$(ssh_cmd "$node" "ps -ef | grep -E 'ray/core/src/ray/(raylet|gcs/gcs_server)' | grep -v grep || true")"
    if [ -n "$REMAINING_RAY_DAEMONS" ]; then
        echo "ERROR: Found existing Ray daemons on $node after 'ray stop':"
        echo "$REMAINING_RAY_DAEMONS"
        echo ""
        echo "Another user's Ray cluster may still be occupying this node and its GPUs."
        echo "Use different nodes or stop the other Ray cluster first."
        exit 1
    fi
    echo "  [OK] No remaining Ray daemons on $node"
done
echo ""

# =============================================================================
# Step 5: 检查关键端口是否被占用
# =============================================================================
echo "===== Step 4: Checking Ray ports on head node ====="
PORT_CHECK_OUTPUT="$(ssh_cmd "$HEAD_NODE" "ss -lnt | awk 'NR>1 {print \$4}' | grep -E ':(($RAY_PORT)|($RAY_DASHBOARD_PORT)|($RAY_CLIENT_SERVER_PORT)|($RAY_DASHBOARD_AGENT_LISTEN_PORT))\$' || true")"
if [ -n "$PORT_CHECK_OUTPUT" ]; then
    echo "ERROR: The following ports are already in use on $HEAD_NODE:"
    echo "$PORT_CHECK_OUTPUT"
    echo ""
    echo "This usually means another Ray cluster is still running on the head node."
    echo "Choose different ports or stop the other cluster first."
    exit 1
fi
echo "  [OK] Ports $RAY_PORT / $RAY_DASHBOARD_PORT / $RAY_CLIENT_SERVER_PORT / $RAY_DASHBOARD_AGENT_LISTEN_PORT are free on $HEAD_NODE"
echo ""

# =============================================================================
# NCCL 环境变量
# =============================================================================
# wg 机器网络拓扑：排除 bond 和有问题的 mlx5_6
# 注意：不设置 RAY_TEMPDIR，让 Ray 使用默认的 /tmp/ray，避免 socket path 不匹配问题
NCCL_ENVS="export LD_LIBRARY_PATH=/usr/local/cuda-13.1/compat:\$LD_LIBRARY_PATH && \
export NCCL_SOCKET_IFNAME=$NCCL_IFNAME && \
export NCCL_SOCKET_FAMILY=AF_INET && \
export NCCL_IB_DISABLE=0 && \
export NCCL_IB_HCA='^mlx5_bond,mlx5_6' && \
export NCCL_CROSS_NIC=1 && \
export NCCL_DEBUG=WARN && \
export TORCH_NCCL_AVOID_RECORD_STREAMS=1 && \
export NCCL_CUMEM_ENABLE=0"

# =============================================================================
# Step 6: 启动 Ray head 节点
# =============================================================================
echo "===== Step 5: Starting Ray head on $HEAD_NODE ====="
ssh_cmd "$HEAD_NODE" "${REMOTE_PREFIX}$NCCL_ENVS && cd $PROJECT_DIR && ray start --head --port=$RAY_PORT \
    --dashboard-host=0.0.0.0 --dashboard-port=$RAY_DASHBOARD_PORT \
    --dashboard-agent-listen-port=$RAY_DASHBOARD_AGENT_LISTEN_PORT \
    --ray-client-server-port=$RAY_CLIENT_SERVER_PORT \
    --num-gpus=$GPUS_PER_NODE --disable-usage-stats"
echo ""

# =============================================================================
# Step 7: 启动 Ray worker 节点
# =============================================================================
if [ $NNODES -gt 1 ]; then
    echo "===== Step 6: Starting Ray workers ====="
    for i in $(seq 1 $((NNODES - 1))); do
        worker="${NODE_LIST[$i]}"
        echo "  Starting Ray worker on $worker (connecting to $HEAD_NODE:$RAY_PORT) ..."
        ssh_cmd "$worker" "${REMOTE_PREFIX}$NCCL_ENVS && cd $PROJECT_DIR && ray start --address=$HEAD_NODE:$RAY_PORT \
            --dashboard-agent-listen-port=$RAY_DASHBOARD_AGENT_LISTEN_PORT \
            --num-gpus=$GPUS_PER_NODE --disable-usage-stats"
    done
    echo ""
fi

# =============================================================================
# Step 8: 等待集群就绪
# =============================================================================
echo "===== Step 7: Waiting for Ray cluster to be ready ====="
EXPECTED_GPUS=$((NNODES * GPUS_PER_NODE))

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
# Step 9: 启动训练
# =============================================================================
echo "===== Step 8: Launching training on head node ($HEAD_NODE) ====="
echo "  Script:  $TRAINING_SCRIPT"
echo "  NNODES=$NNODES, N_GPUS=$GPUS_PER_NODE, SKIP_VIDEO_CACHE=true"
echo "  Ray Dashboard: http://$HEAD_NODE:$RAY_DASHBOARD_PORT"
echo ""

# 通过 Hydra config 注入 NCCL 变量到 ray.init(runtime_env)
NCCL_HYDRA_ARGS="+ray_kwargs.ray_init.runtime_env.env_vars.NCCL_SOCKET_IFNAME=$NCCL_IFNAME"
NCCL_HYDRA_ARGS="$NCCL_HYDRA_ARGS +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_SOCKET_FAMILY=AF_INET"
NCCL_HYDRA_ARGS="$NCCL_HYDRA_ARGS +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_IB_DISABLE=0"
NCCL_HYDRA_ARGS="$NCCL_HYDRA_ARGS +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_DEBUG=WARN"
NCCL_HYDRA_ARGS="$NCCL_HYDRA_ARGS +ray_kwargs.ray_init.runtime_env.env_vars.TORCH_NCCL_AVOID_RECORD_STREAMS=1"
NCCL_HYDRA_ARGS="$NCCL_HYDRA_ARGS +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_CUMEM_ENABLE=0"
NCCL_HYDRA_ARGS="$NCCL_HYDRA_ARGS +ray_kwargs.ray_init.runtime_env.env_vars.TIKTOKEN_RS_CACHE_DIR=/data_gpu/gyzhou/harmony_cache"

ssh_cmd "$HEAD_NODE" "${REMOTE_PREFIX}cd $PROJECT_DIR && \
    $NCCL_ENVS && \
    export NNODES=$NNODES && \
    export N_GPUS=$GPUS_PER_NODE && \
    export SKIP_VIDEO_CACHE=true && \
    export RAY_ADDRESS=$HEAD_NODE:$RAY_PORT && \
    bash $TRAINING_SCRIPT $NCCL_HYDRA_ARGS $EXTRA_ARGS"
