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

    bash examples/video_reasoning/run_video_reasoning_dapo_h200_zsr.sh $HYDRA_ENV_ARGS $EXTRA_ARGS

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
