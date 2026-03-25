#!/bin/bash
set -e
source /mnt/data/home/zhengshurong/miniconda3/etc/profile.d/conda.sh
conda activate verl

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
