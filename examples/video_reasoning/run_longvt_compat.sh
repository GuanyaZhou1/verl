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

# =============================================================================
# 环境配置（与 LongVT 一致）
# =============================================================================
ulimit -n 65535

# NCCL 配置
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA="${NCCL_IB_HCA:-^mlx5_bond,mlx5_6}"
export NCCL_CROSS_NIC=${NCCL_CROSS_NIC:-1}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-bond0.1573}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-bond0.1573}
export NCCL_SOCKET_FAMILY=${NCCL_SOCKET_FAMILY:-AF_INET}
export MASTER_ADDR=${MASTER_ADDR:-10.0.1.33}
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
export RAY_TMPDIR=/tmp/ray_$USER

# =============================================================================
# 路径配置（与 LongVT 一致）
# =============================================================================
PROJECT_DIR="/mnt/data/home/zhengshurong/project/verl"
MODEL_PATH='/mnt/data/home/zhengshurong/model/LongVT-SFT'
DATA_PATH='/mnt/data/home/zhengshurong/dataset/LongVT-Parquet'
CONFIG_PATH="$PROJECT_DIR/examples/video_reasoning/config"
CKPT_DIR="$PROJECT_DIR/checkpoints"

# =============================================================================
# 实验配置
# =============================================================================
PROJECT_NAME="LongVT"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-longvt-verl-compat-$(date +%Y%m%d_%H%M%S)}"
NNODES=${NNODES:-2}

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
    data.train_files=$DATA_PATH/longvt_rl_selfqa_1k6_fixed.parquet \
    data.val_files=$DATA_PATH/longvt_rl_val_114_fixed.parquet \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=5 \
    actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode=disable \
    actor_rollout_ref.rollout.agent.default_agent_loop=longvt \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
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
