#!/bin/bash
# =============================================================================
# Multi-Node Launcher for LongVT Compatible Training
# =============================================================================
# 通过 SSH 在多台机器上启动 Ray 集群，然后在 head 节点上运行 LongVT 训练脚本。
#
# 前提条件：
#   - 所有节点之间 SSH 免密登录
#   - 共享文件系统（所有节点可访问同一项目目录）
#   - 所有节点上的项目路径、conda 环境、网络接口命名保持一致
#
# 使用方式：
#   bash launch_longvt_multinode.sh --nodes "10.96.11.5,10.96.11.6"
#   bash launch_longvt_multinode.sh --node-file nodes.txt -- trainer.total_epochs=2
#
# 说明：
#   - 第一个节点作为 Ray head
#   - `--` 后的参数会透传给 run_longvt_compat.sh
#   - launcher 会显式把 Ray/NCCL/Gloo 网络配置注入到远端和 Ray runtime_env
# =============================================================================

set -eo pipefail

# =============================================================================
# 默认参数
# =============================================================================
NODES="${NODES:-}"
NODE_FILE="${NODE_FILE:-}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEFAULT_NODE_FILE="$SCRIPT_DIR/nodes.txt"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
SSH_USER="${SSH_USER:-}"
NCCL_IFNAME="${NCCL_IFNAME:-bond0}"
NCCL_IB_HCA="${NCCL_IB_HCA:-^mlx5_bond,mlx5_6}"
RAY_PORT="${RAY_PORT:-6380}"
MASTER_PORT="${MASTER_PORT:-29500}"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONDA_ACTIVATE="${CONDA_ACTIVATE:-source /share_data/gyzhou/anaconda3/etc/profile.d/conda.sh && conda activate verl_clone}"
REMOTE_PREFIX="${CONDA_ACTIVATE:+$CONDA_ACTIVATE && }"
EXTRA_ARGS=()

shell_join() {
    local quoted=()
    local arg
    for arg in "$@"; do
        quoted+=("$(printf '%q' "$arg")")
    done
    printf '%s' "${quoted[*]}"
}

ssh_cmd() {
    local node="$1"
    shift
    if [ -n "$SSH_USER" ]; then
        ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "${SSH_USER}@${node}" "$@"
    else
        ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "${node}" "$@"
    fi
}

ssh_bash() {
    local node="$1"
    shift
    local cmd="$1"
    ssh_cmd "$node" "bash -lc $(printf '%q' "$cmd")"
}

usage() {
    echo "Usage: bash launch_longvt_multinode.sh [options] [-- extra_training_args]"
    echo ""
    echo "If no --nodes or --node-file is given, defaults to nodes.txt in the same directory."
    echo ""
    echo "Options:"
    echo "  --nodes IP1,IP2,...    Comma-separated list of node IPs/hostnames (first is head)"
    echo "  --node-file FILE       File with one node per line (first is head, default: nodes.txt)"
    echo "  --gpus-per-node N      Number of GPUs per node (default: 8)"
    echo "  --ssh-user USER        SSH username (default: current user)"
    echo "  --nccl-ifname IFACE    Network interface used by NCCL/Gloo/Ray (default: bond0)"
    echo "  --ray-port PORT        Ray head port (default: 6380)"
    echo "  --master-port PORT     Torch distributed master port (default: 29500)"
    echo "  -- ARGS                Extra args passed to run_longvt_compat.sh"
    echo ""
    echo "Environment:"
    echo "  CONDA_ACTIVATE         Command used on remote nodes before ray/python."
    echo "                         Example: export CONDA_ACTIVATE=\"source /share_data/gyzhou/anaconda3/etc/profile.d/conda.sh && conda activate verl_clone\""
}

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
        --master-port)
            MASTER_PORT="$2"
            shift 2
            ;;
        --)
            shift
            EXTRA_ARGS=("$@")
            break
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# =============================================================================
# 解析节点列表
# =============================================================================
NODE_LIST=()
if [ -n "$NODE_FILE" ]; then
    if [ ! -f "$NODE_FILE" ]; then
        echo "ERROR: Node file not found: $NODE_FILE"
        exit 1
    fi
    while IFS= read -r line || [ -n "$line" ]; do
        line=$(echo "$line" | sed 's/#.*//' | xargs)
        [ -n "$line" ] && NODE_LIST+=("$line")
    done < "$NODE_FILE"
elif [ -n "$NODES" ]; then
    IFS=',' read -ra NODE_LIST <<< "$NODES"
elif [ -f "$DEFAULT_NODE_FILE" ]; then
    echo "No --nodes or --node-file specified, using default: $DEFAULT_NODE_FILE"
    while IFS= read -r line || [ -n "$line" ]; do
        line=$(echo "$line" | sed 's/#.*//' | xargs)
        [ -n "$line" ] && NODE_LIST+=("$line")
    done < "$DEFAULT_NODE_FILE"
else
    echo "ERROR: Must specify --nodes or --node-file, or create $DEFAULT_NODE_FILE"
    usage
    exit 1
fi

if [ ${#NODE_LIST[@]} -lt 1 ]; then
    echo "ERROR: No nodes specified"
    exit 1
fi

HEAD_NODE="${NODE_LIST[0]}"
NNODES=${#NODE_LIST[@]}
declare -A NODE_IPS

# =============================================================================
# 打印配置
# =============================================================================
echo "===== LongVT Multi-Node Launcher ====="
echo "Project dir:   $PROJECT_DIR"
echo "Head node:     $HEAD_NODE"
echo "Total nodes:   $NNODES"
echo "GPUs per node: $GPUS_PER_NODE"
echo "Ray port:      $RAY_PORT"
echo "Master port:   $MASTER_PORT"
echo "NCCL ifname:   $NCCL_IFNAME"
echo "NCCL IB HCA:   $NCCL_IB_HCA"
[ -n "$CONDA_ACTIVATE" ] && echo "Conda activate: set"
echo "Nodes:"
for i in "${!NODE_LIST[@]}"; do
    if [ "$i" -eq 0 ]; then
        echo "  [$i] ${NODE_LIST[$i]} (head)"
    else
        echo "  [$i] ${NODE_LIST[$i]} (worker)"
    fi
done
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    echo "Extra args:    $(shell_join "${EXTRA_ARGS[@]}")"
fi
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
    echo "Please verify SSH keys and network connectivity."
    exit 1
fi
echo ""

# =============================================================================
# Step 2: 验证远端运行环境
# =============================================================================
echo "===== Step 2: Verifying remote runtime environment ====="
FAILED_ENV_NODES=()
for node in "${NODE_LIST[@]}"; do
    if ssh_bash "$node" "${REMOTE_PREFIX}command -v ray >/dev/null && command -v python3 >/dev/null && echo ok" >/dev/null 2>&1; then
        echo "  [OK] $node"
    else
        echo "  [FAIL] $node"
        FAILED_ENV_NODES+=("$node")
    fi
done

if [ ${#FAILED_ENV_NODES[@]} -gt 0 ]; then
    echo ""
    echo "ERROR: Remote runtime bootstrap failed on:"
    for node in "${FAILED_ENV_NODES[@]}"; do
        echo "  - $node"
    done
    echo "Current CONDA_ACTIVATE:"
    echo "  $CONDA_ACTIVATE"
    echo "Set CONDA_ACTIVATE explicitly if these nodes use a different conda path or env."
    exit 1
fi
echo ""

# =============================================================================
# Step 3: 验证网络接口并收集每台机器的训练 IP
# =============================================================================
echo "===== Step 3: Resolving node IPs on $NCCL_IFNAME ====="
for node in "${NODE_LIST[@]}"; do
    node_ip=$(ssh_bash "$node" "ip -o -4 addr show dev $NCCL_IFNAME up scope global 2>/dev/null | awk 'NR==1 {print \$4}' | cut -d/ -f1")
    if [ -z "$node_ip" ]; then
        echo "ERROR: Interface $NCCL_IFNAME has no global IPv4 on $node"
        echo "Use --nccl-ifname to specify the correct training interface."
        exit 1
    fi
    NODE_IPS["$node"]="$node_ip"
    echo "  $node -> $node_ip"
done
HEAD_IP="${NODE_IPS[$HEAD_NODE]}"
echo ""

cleanup() {
    echo ""
    echo "===== Cleaning up: Stopping Ray on all nodes ====="
    for node in "${NODE_LIST[@]}"; do
        echo "  Stopping Ray on $node ..."
        ssh_bash "$node" "${REMOTE_PREFIX}ray stop --force 2>/dev/null || true" >/dev/null 2>&1 || true
    done
    echo "===== Cleanup complete ====="
}
trap cleanup EXIT INT TERM

COMMON_REMOTE_ENV="export NCCL_SOCKET_IFNAME=$NCCL_IFNAME && \
export GLOO_SOCKET_IFNAME=$NCCL_IFNAME && \
export TP_SOCKET_IFNAME=$NCCL_IFNAME && \
export NCCL_IB_DISABLE=0 && \
export NCCL_IB_HCA=$NCCL_IB_HCA && \
export NCCL_CROSS_NIC=1 && \
export NCCL_SOCKET_FAMILY=AF_INET && \
export MASTER_ADDR=$HEAD_IP && \
export MASTER_PORT=$MASTER_PORT && \
export TMPDIR=/tmp && \
export TRITON_CACHE_DIR=/tmp/.triton_cache_\$(whoami)"

# =============================================================================
# Step 4: 停止旧的 Ray 集群
# =============================================================================
echo "===== Step 4: Stopping existing Ray processes ====="
for node in "${NODE_LIST[@]}"; do
    echo "  Stopping Ray on $node ..."
    ssh_bash "$node" "${REMOTE_PREFIX}ray stop --force 2>/dev/null || true" >/dev/null 2>&1 || true
done
sleep 2
echo ""

# =============================================================================
# Step 5: 启动 Ray head
# =============================================================================
echo "===== Step 5: Starting Ray head on $HEAD_NODE ($HEAD_IP) ====="
ssh_bash "$HEAD_NODE" "${REMOTE_PREFIX}$COMMON_REMOTE_ENV && cd $PROJECT_DIR && ray start --head --port=$RAY_PORT --node-ip-address=$HEAD_IP --dashboard-host=0.0.0.0 --num-gpus=$GPUS_PER_NODE --disable-usage-stats"
echo ""

# =============================================================================
# Step 6: 启动 Ray workers
# =============================================================================
if [ $NNODES -gt 1 ]; then
    echo "===== Step 6: Starting Ray workers ====="
    for i in $(seq 1 $((NNODES - 1))); do
        worker="${NODE_LIST[$i]}"
        worker_ip="${NODE_IPS[$worker]}"
        echo "  Starting Ray worker on $worker ($worker_ip) ..."
        ssh_bash "$worker" "${REMOTE_PREFIX}$COMMON_REMOTE_ENV && cd $PROJECT_DIR && ray start --address=$HEAD_IP:$RAY_PORT --node-ip-address=$worker_ip --num-gpus=$GPUS_PER_NODE --disable-usage-stats"
    done
    echo ""
fi

# =============================================================================
# Step 7: 等待集群就绪
# =============================================================================
echo "===== Step 7: Waiting for Ray cluster to be ready ====="
EXPECTED_GPUS=$((NNODES * GPUS_PER_NODE))
WAIT_CMD=$(cat <<EOF
$REMOTE_PREFIX
cd $PROJECT_DIR
python3 -u - <<'PY'
import ray
import sys
import time

ray.init(address="${HEAD_IP}:${RAY_PORT}")

timeout = 300
start = time.time()
while True:
    resources = ray.cluster_resources()
    num_gpus = int(resources.get("GPU", 0))
    alive_nodes = sum(1 for node in ray.nodes() if node["Alive"])
    elapsed = time.time() - start
    print(f"  Cluster: {alive_nodes}/${NNODES} nodes, {num_gpus}/${EXPECTED_GPUS} GPUs ({elapsed:.0f}s)")

    if num_gpus >= ${EXPECTED_GPUS} and alive_nodes >= ${NNODES}:
        print("  Cluster ready!")
        break
    if elapsed > timeout:
        print(
            f"ERROR: Timeout waiting for cluster. Expected ${NNODES} nodes with ${EXPECTED_GPUS} GPUs, "
            f"got {alive_nodes} nodes with {num_gpus} GPUs."
        )
        ray.shutdown()
        sys.exit(1)
    time.sleep(5)

ray.shutdown()
PY
EOF
)
ssh_bash "$HEAD_NODE" "$WAIT_CMD"
echo ""

# =============================================================================
# Step 8: 启动训练
# =============================================================================
echo "===== Step 8: Launching LongVT training on $HEAD_NODE ====="
echo "  Head IP:       $HEAD_IP"
echo "  Ray dashboard: http://$HEAD_IP:8265"
echo ""

TRAIN_ARGS=(
    "+ray_kwargs.ray_init.runtime_env.env_vars.NCCL_SOCKET_IFNAME='${NCCL_IFNAME}'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.GLOO_SOCKET_IFNAME='${NCCL_IFNAME}'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.TP_SOCKET_IFNAME='${NCCL_IFNAME}'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.NCCL_IB_DISABLE='0'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.NCCL_CROSS_NIC='1'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.NCCL_SOCKET_FAMILY='AF_INET'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.MASTER_ADDR='${HEAD_IP}'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.MASTER_PORT='${MASTER_PORT}'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.TMPDIR='/tmp'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.TRITON_CACHE_DIR='/tmp/.triton_cache_$(whoami)'"
)
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    TRAIN_ARGS+=("${EXTRA_ARGS[@]}")
fi

TRAIN_CMD="${REMOTE_PREFIX}cd $PROJECT_DIR && \
export NNODES=$NNODES && \
export N_GPUS=$GPUS_PER_NODE && \
export RAY_ADDRESS=$HEAD_IP:$RAY_PORT && \
export MASTER_ADDR=$HEAD_IP && \
export MASTER_PORT=$MASTER_PORT && \
export NCCL_SOCKET_IFNAME=$NCCL_IFNAME && \
export GLOO_SOCKET_IFNAME=$NCCL_IFNAME && \
export TP_SOCKET_IFNAME=$NCCL_IFNAME && \
export NCCL_IB_DISABLE=0 && \
export NCCL_IB_HCA=$NCCL_IB_HCA && \
export NCCL_CROSS_NIC=1 && \
export NCCL_SOCKET_FAMILY=AF_INET && \
bash examples/video_reasoning/run_longvt_compat.sh $(shell_join "${TRAIN_ARGS[@]}")"

ssh_bash "$HEAD_NODE" "$TRAIN_CMD"
