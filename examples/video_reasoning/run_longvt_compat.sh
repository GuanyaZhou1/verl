#!/bin/bash
# =============================================================================
# LongVT Compatible Training Script for verl
# =============================================================================
# 完全对齐 LongVT 原始训练脚本的参数配置
# 原始脚本: /mnt/data/home/zhengshurong/project/LongVT/examples/video_tools/longvt_7b_rl_train.sh
# =============================================================================

set -x
set -eo pipefail
export HYDRA_FULL_ERROR=1
export LD_LIBRARY_PATH="/usr/local/cuda-13.1/compat:${LD_LIBRARY_PATH:-}"
# Triton 缓存放到本地磁盘，避免 NFS 上多进程并发编译的竞争条件
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/.triton_cache_$USER}"
# =============================================================================
# 环境配置（与 LongVT 一致）
# =============================================================================
ulimit -n 65535

detect_socket_ifname() {
    local candidate
    local ifname

    for candidate in bond0.1573 bond0 bond1 br0 eth0; do
        if ip -o -4 addr show dev "$candidate" up scope global >/dev/null 2>&1; then
            echo "$candidate"
            return 0
        fi
    done

    ifname=$(ip -o route get 1.1.1.1 2>/dev/null | awk '{for (i = 1; i <= NF; ++i) if ($i == "dev") {print $(i + 1); exit}}')
    if [ -n "$ifname" ] && ip -o -4 addr show dev "$ifname" up scope global >/dev/null 2>&1; then
        echo "$ifname"
        return 0
    fi

    ifname=$(ip -o -4 addr show up scope global | awk '{print $2}' | sort -u | head -n 1)
    if [ -n "$ifname" ]; then
        echo "$ifname"
        return 0
    fi

    return 1
}

detect_ipv4_for_ifname() {
    local ifname="$1"
    ip -o -4 addr show dev "$ifname" up scope global 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -n 1
}

# NCCL 配置
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA="${NCCL_IB_HCA:-^mlx5_bond,mlx5_6}"
export NCCL_CROSS_NIC=${NCCL_CROSS_NIC:-1}
DEFAULT_SOCKET_IFNAME="${DEFAULT_SOCKET_IFNAME:-$(detect_socket_ifname || true)}"
if [ -z "${NCCL_SOCKET_IFNAME:-}" ] && [ -n "$DEFAULT_SOCKET_IFNAME" ]; then
    export NCCL_SOCKET_IFNAME="$DEFAULT_SOCKET_IFNAME"
fi
if [ -z "${GLOO_SOCKET_IFNAME:-}" ] && [ -n "${NCCL_SOCKET_IFNAME:-}" ]; then
    export GLOO_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME"
fi
if [ -z "${TP_SOCKET_IFNAME:-}" ] && [ -n "${GLOO_SOCKET_IFNAME:-}" ]; then
    export TP_SOCKET_IFNAME="$GLOO_SOCKET_IFNAME"
fi
export NCCL_SOCKET_FAMILY=${NCCL_SOCKET_FAMILY:-AF_INET}
if [ -z "${MASTER_ADDR:-}" ] && [ -n "${GLOO_SOCKET_IFNAME:-}" ]; then
    MASTER_ADDR="$(detect_ipv4_for_ifname "$GLOO_SOCKET_IFNAME")"
fi
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}

# LLM-as-judge 配置（必需）
export LLM_AS_A_JUDGE_BASE="http://10.0.1.35:8081/v1"
export LLM_AS_A_JUDGE_KEY="${LLM_AS_A_JUDGE_KEY:-123456}"

# 视频读取器配置
export FORCE_QWENVL_VIDEO_READER=decord

# Ray 配置
export RAY_DEBUG=0
export PYDEVD_DISABLE_FILE_VALIDATION=1
export TMPDIR=/tmp

# =============================================================================
# 路径配置（与 LongVT 一致）
# =============================================================================
PROJECT_DIR="/data_gpu/gyzhou/prj/verl"
MODEL_PATH='/data_gpu/zhengshurong/data/hf_cache/LongVT/LongVT-SFT'
DATA_PATH="${DATA_PATH:-$PROJECT_DIR/long_video_data_ablate/longvt}"
CONFIG_PATH="$PROJECT_DIR/examples/video_reasoning/config"
CKPT_DIR="$PROJECT_DIR/checkpoints"

# =============================================================================
# 实验配置
# =============================================================================
PROJECT_NAME="LongVT"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-longvt-verl-compat-$(date +%Y%m%d_%H%M%S)}"
NNODES=${NNODES:-2}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.7}
ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-65536}
ROLLOUT_LOGPROB_MAX_TOKEN_LEN_PER_GPU=${ROLLOUT_LOGPROB_MAX_TOKEN_LEN_PER_GPU:-65536}
REF_LOGPROB_MAX_TOKEN_LEN_PER_GPU=${REF_LOGPROB_MAX_TOKEN_LEN_PER_GPU:-65536}

# =============================================================================
# 日志配置
# =============================================================================
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/rl_train_${TIMESTAMP}.log"

echo "========================================"
echo "LongVT Compatible Training (verl)"
echo "========================================"
echo "Model: $MODEL_PATH"
echo "Data: $DATA_PATH"
echo "LLM Judge: $LLM_AS_A_JUDGE_BASE"
echo "Experiment: $EXPERIMENT_NAME"
echo "Logging to: $LOG_FILE"
echo "========================================"

# =============================================================================
# 启动训练（参数完全对齐 LongVT）
# =============================================================================
python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='longvt_compat' \
    \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    \
    data.train_files=$DATA_PATH/longvt_rl_selfqa_1k6.parquet \
    data.val_files=$DATA_PATH/longvt_rl_val_114.parquet \
    data.train_batch_size=16 \
    data.max_prompt_length=36000 \
    data.max_response_length=16384 \
    data.filter_overlong_prompts=False \
    data.truncation='left' \
    data.return_raw_chat=True \
    +data.sglang_video=True \
    data.dataloader_num_workers=2 \
    data.return_multi_modal_inputs=False \
    \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=4 \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTILIZATION \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.max_model_len=$ROLLOUT_MAX_MODEL_LEN \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$ROLLOUT_LOGPROB_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=5 \
    actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode=disable \
    actor_rollout_ref.rollout.agent.default_agent_loop=longvt \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$REF_LOGPROB_MAX_TOKEN_LEN_PER_GPU \
    \
    custom_reward_function.path=pkg://verl.utils.reward_score.longvt_reward \
    custom_reward_function.name=compute_score \
    custom_reward_function.reward_kwargs.use_iou_reward=True \
    \
    trainer.total_epochs=5 \
    trainer.critic_warmup=0 \
    trainer.logger='[console,tensorboard]' \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=$NNODES \
    trainer.default_local_dir=$CKPT_DIR/$PROJECT_NAME/$EXPERIMENT_NAME \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.val_before_train=False \
    trainer.rollout_data_dir=$CKPT_DIR/$PROJECT_NAME/$EXPERIMENT_NAME/rollout \
    \
    "$@" \
    2>&1 | tee "$LOG_FILE"
