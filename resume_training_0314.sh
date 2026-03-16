#!/bin/bash
# =============================================================================
# 恢复训练脚本 - 从 global_step_30 恢复
# 实验: Qwen3_8B_dapo_kl0.3_bbox0.3_topp0.7_lr1e-6__0314_GAE_perturn
# 节点: node15 (head), node19 (worker)
# =============================================================================

set -e

# 配置
HEAD_NODE="node15"
WORKER_NODES="node19"
HEAD_IP="10.1.3.15"
RAY_PORT=6380
GPUS=8
NNODES=2
PROJECT_DIR="/mnt/data/home/zhengshurong/project/verl"

# 实验配置 - 与原实验一致
EXPERIMENT_NAME="Qwen3_8B_dapo_kl0.3_bbox0.3_topp0.7_lr1e-6__0314_GAE_perturn"
PROJECT_NAME="video-reasoning-dapo"
# 使用绝对路径
CKPT_BASE="${PROJECT_DIR}/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}"

echo "===== 恢复训练配置 ====="
echo "实验名: $EXPERIMENT_NAME"
echo "Checkpoint: $CKPT_BASE"
echo "最新 Step: $(cat $CKPT_BASE/latest_checkpointed_iteration.txt 2>/dev/null || echo 'N/A')"
echo "节点: $HEAD_NODE (head), $WORKER_NODES (worker)"
echo "========================="

# 生成 ray launch 脚本
LAUNCH_SCRIPT="$PROJECT_DIR/.ray_launch_resume_$$.sh"
cat > "$LAUNCH_SCRIPT" << 'LAUNCH_EOF'
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

    echo "[$(hostname)] Starting Ray head..."
    ray stop --force 2>/dev/null || true
    sleep 2
    ray start --head --port=$RAY_PORT --num-gpus=$GPUS --disable-usage-stats --node-ip-address=$HEAD_IP

    echo "[$(hostname)] Waiting for cluster ($NNODES nodes, $((NNODES * GPUS)) GPUs)..."
    EXPECTED=$((NNODES * GPUS))
    for i in $(seq 1 120); do
        NGPUS=$(python3 -c "import ray; ray.init(address='$HEAD_IP:$RAY_PORT'); print(int(ray.cluster_resources().get('GPU',0))); ray.shutdown()" 2>/dev/null || echo 0)
        echo "  [$i] GPUs: $NGPUS / $EXPECTED"
        [ "$NGPUS" -ge "$EXPECTED" ] && break
        sleep 5
    done

    echo "[$(hostname)] Starting training..."
    cd "$PROJECT_DIR"
    export NNODES=$NNODES
    export N_GPUS=$GPUS
    export RAY_ADDRESS=$HEAD_IP:$RAY_PORT
    export SKIP_VIDEO_CACHE=true
    export RUN_EVAL=false

    # NCCL 配置
    export NCCL_IB_DISABLE=0
    export NCCL_IB_HCA="^mlx5_bond,mlx5_6"
    export NCCL_CROSS_NIC=1
    export NCCL_SOCKET_IFNAME=bond0
    export NCCL_SOCKET_FAMILY=AF_INET
    export MASTER_ADDR=$HEAD_IP
    export MASTER_PORT=29500

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

    bash examples/video_reasoning/run_video_reasoning_dapo_h200.sh $HYDRA_ENV_ARGS $EXTRA_ARGS

    echo "[$(hostname)] Training complete!"
else
    unset ROCR_VISIBLE_DEVICES
    unset HIP_VISIBLE_DEVICES

    echo "[$(hostname)] Starting Ray worker, connecting to $HEAD_IP:$RAY_PORT..."
    ray stop --force 2>/dev/null || true
    sleep 2
    sleep 10
    ray start --address=$HEAD_IP:$RAY_PORT --num-gpus=$GPUS --disable-usage-stats
    echo "[$(hostname)] Ray worker started, waiting..."
    while true; do sleep 3600; done
fi
LAUNCH_EOF
chmod +x "$LAUNCH_SCRIPT"

echo ""
echo "===== 清理旧进程 ====="
for node in $HEAD_NODE $WORKER_NODES; do
    echo "Stopping ray on $node..."
    ssh $node "ray stop --force 2>/dev/null; pkill -f 'ray::' 2>/dev/null || true" &
done
wait
sleep 3

echo ""
echo "===== 启动训练 ====="

# 启动 worker 节点
for node in $WORKER_NODES; do
    echo "Starting worker on $node..."
    ssh $node "nohup bash $LAUNCH_SCRIPT worker $HEAD_IP $RAY_PORT $GPUS $NNODES $PROJECT_DIR > /tmp/ray_worker_$$.log 2>&1 &"
done

sleep 5

# 启动 head 节点 (前台运行，输出到终端)
echo "Starting head on $HEAD_NODE..."
# 直接通过 hydra 参数传递实验名和恢复模式，避免脚本内部覆盖
# 使用绝对路径确保 checkpoint 恢复正常
ssh $HEAD_NODE "export EXPERIMENT_NAME='$EXPERIMENT_NAME'; \
    export PROJECT_NAME='$PROJECT_NAME'; \
    bash $LAUNCH_SCRIPT head $HEAD_IP $RAY_PORT $GPUS $NNODES $PROJECT_DIR \
    trainer.resume_mode=auto \
    trainer.experiment_name='$EXPERIMENT_NAME' \
    trainer.project_name='$PROJECT_NAME' \
    trainer.default_local_dir='$CKPT_BASE'"

echo ""
echo "===== 训练完成 ====="
