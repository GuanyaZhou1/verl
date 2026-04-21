#!/bin/bash
# =============================================================================
# Parameterized Video Reasoning DAPO Training Script
# =============================================================================
# 支持外部传参，便于自动化实验
#
# 使用方式：
#   # 默认配置
#   bash run_experiment.sh
#
#   # 自定义参数
#   bash run_experiment.sh \
#       --kl-coef 0.3 \
#       --bbox-weight 0.3 \
#       --top-p 0.7 \
#       --exp-name "my_experiment"
#
# =============================================================================

set -eo pipefail
export HYDRA_FULL_ERROR=1

# =============================================================================
# 默认参数（可被命令行覆盖）
# =============================================================================
KL_LOSS_COEF=${KL_LOSS_COEF:-0.3}
ENTROPY_COEFF=${ENTROPY_COEFF:-0.0}
BBOX_WEIGHT=${BBOX_WEIGHT:-0.6}
ANSWER_WEIGHT=${ANSWER_WEIGHT:-1.0}
TOP_P=${TOP_P:-1.0}
TOP_K=${TOP_K:--1}
TEMPERATURE=${TEMPERATURE:-1.0}
LEARNING_RATE=${LEARNING_RATE:-1e-6}
CLIP_RATIO_LOW=${CLIP_RATIO_LOW:-0.2}
CLIP_RATIO_HIGH=${CLIP_RATIO_HIGH:-0.25}
NORM_ADV_BY_STD=${NORM_ADV_BY_STD:-False}
N_ROLLOUTS=${N_ROLLOUTS:-8}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-3}

# Token placement
TOKEN_PLACEMENT_METHOD=${TOKEN_PLACEMENT_METHOD:-broadcast}
BBOX_CONTEXT_WINDOW=${BBOX_CONTEXT_WINDOW:-16}
SEGMENT_CONTEXT_WINDOW=${SEGMENT_CONTEXT_WINDOW:-16}

# 实验标识
EXPERIMENT_NAME=${EXPERIMENT_NAME:-""}
EXPERIMENT_ID=${EXPERIMENT_ID:-""}

# 集群配置
N_GPUS=${N_GPUS:-8}
NNODES=${NNODES:-1}

# =============================================================================
# 解析命令行参数
# =============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --kl-coef) KL_LOSS_COEF="$2"; shift 2 ;;
        --entropy-coef) ENTROPY_COEFF="$2"; shift 2 ;;
        --bbox-weight) BBOX_WEIGHT="$2"; shift 2 ;;
        --answer-weight) ANSWER_WEIGHT="$2"; shift 2 ;;
        --top-p) TOP_P="$2"; shift 2 ;;
        --top-k) TOP_K="$2"; shift 2 ;;
        --temperature) TEMPERATURE="$2"; shift 2 ;;
        --lr) LEARNING_RATE="$2"; shift 2 ;;
        --clip-low) CLIP_RATIO_LOW="$2"; shift 2 ;;
        --clip-high) CLIP_RATIO_HIGH="$2"; shift 2 ;;
        --norm-adv) NORM_ADV_BY_STD="$2"; shift 2 ;;
        --n-rollouts) N_ROLLOUTS="$2"; shift 2 ;;
        --batch-size) TRAIN_BATCH_SIZE="$2"; shift 2 ;;
        --epochs) TOTAL_EPOCHS="$2"; shift 2 ;;
        --exp-name) EXPERIMENT_NAME="$2"; shift 2 ;;
        --exp-id) EXPERIMENT_ID="$2"; shift 2 ;;
        --n-gpus) N_GPUS="$2"; shift 2 ;;
        --nnodes) NNODES="$2"; shift 2 ;;
        --) shift; break ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# =============================================================================
# 生成实验名称（如果未指定）
# =============================================================================
TIMESTAMP=$(date '+%Y%m%d-%H%M%S')
if [ -z "$EXPERIMENT_NAME" ]; then
    # 自动生成名称，包含关键参数
    EXPERIMENT_NAME="exp_kl${KL_LOSS_COEF}_bbox${BBOX_WEIGHT}_topp${TOP_P}_${TIMESTAMP}"
fi

if [ -n "$EXPERIMENT_ID" ]; then
    EXPERIMENT_NAME="exp${EXPERIMENT_ID}_${EXPERIMENT_NAME}"
fi

# =============================================================================
# 环境配置
# =============================================================================
ulimit -n 65535
export VLLM_USE_V1=1
export TMPDIR=/tmp
export RAY_TMPDIR=/tmp/ray_$USER
export TIKTOKEN_RS_CACHE_DIR=/mnt/data/home/zhengshurong/harmony_cache

# NCCL 配置
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export NCCL_IB_HCA=${NCCL_IB_HCA:-^mlx5_bond,mlx5_6}
export NCCL_CROSS_NIC=${NCCL_CROSS_NIC:-1}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-bond0}
export NCCL_SOCKET_FAMILY=${NCCL_SOCKET_FAMILY:-AF_INET}
export TORCH_NCCL_AVOID_RECORD_STREAMS=${TORCH_NCCL_AVOID_RECORD_STREAMS:-1}
export NCCL_CUMEM_ENABLE=${NCCL_CUMEM_ENABLE:-0}

# =============================================================================
# 路径配置
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODEL_PATH="/mnt/data/home/zhengshurong/project/Qwen3-VL/qwen-vl-finetune/checkpoints/video/Qwen3-VL-8B-Instruct-longvt_tvg-openo3video_stgr-selfconstructdata-sft-lr1e-5-bs64-ep1"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/long_video_data/video_holmes}"
CACHE_DIR="${CACHE_DIR:-$PROJECT_DIR/.cache}"
CONFIG_PATH="$PROJECT_DIR/examples/video_reasoning/config"

# 实验输出目录
PROJECT_NAME="video-reasoning-dapo"
CKPT_BASE="$PROJECT_DIR/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}"
REWARD_LOG_DIR="${CKPT_BASE}/reward_logs"
TENSORBOARD_DIR="${CKPT_BASE}/tensorboard_log"
LOG_DIR="$PROJECT_DIR/logs"
LOG_FILE="$LOG_DIR/${EXPERIMENT_NAME}.log"

# 自动化实验状态文件
AUTO_EXP_DIR="$PROJECT_DIR/scripts/auto_experiment"
STATE_FILE="$AUTO_EXP_DIR/experiment_state.json"

# =============================================================================
# VLM 配置
# =============================================================================
VLM_ENDPOINT="10.0.1.31:8081"
VLM_MODEL_NAME="/mnt/data/home/zhengshurong/model/Qwen/Qwen3-VL-235B-A22B-Instruct"
VLM_API_KEY="123456"
USE_VLM_SCORING=true
USE_BBOX_VERIFICATION=true

# 视频缓存配置
CACHE_FPS=1
CACHE_MAX_FRAMES=512
CACHE_MAX_FRAMES_PER_SEGMENT=32
USE_CACHED_INITIAL_VIDEO=True

# Checkpoint 配置
SAVE_FREQ=30
TEST_FREQ=20
VAL_BEFORE_TRAIN=True
RESUME_MODE=disable

# =============================================================================
# 打印配置
# =============================================================================
echo "=========================================="
echo "Experiment: $EXPERIMENT_NAME"
echo "=========================================="
echo "Parameters:"
echo "  KL_LOSS_COEF:    $KL_LOSS_COEF"
echo "  ENTROPY_COEFF:   $ENTROPY_COEFF"
echo "  BBOX_WEIGHT:     $BBOX_WEIGHT"
echo "  ANSWER_WEIGHT:   $ANSWER_WEIGHT"
echo "  TOP_P:           $TOP_P"
echo "  LEARNING_RATE:   $LEARNING_RATE"
echo "  CLIP_RATIO:      [$CLIP_RATIO_LOW, $CLIP_RATIO_HIGH]"
echo "  NORM_ADV_BY_STD: $NORM_ADV_BY_STD"
echo "  N_ROLLOUTS:      $N_ROLLOUTS"
echo "  BATCH_SIZE:      $TRAIN_BATCH_SIZE"
echo "  EPOCHS:          $TOTAL_EPOCHS"
echo ""
echo "Cluster:"
echo "  N_GPUS:          $N_GPUS"
echo "  NNODES:          $NNODES"
echo ""
echo "Paths:"
echo "  Log:             $LOG_FILE"
echo "  Checkpoint:      $CKPT_BASE"
echo "=========================================="

# =============================================================================
# 预检查
# =============================================================================
if [ ! -f "$DATA_DIR/train.parquet" ] || [ ! -f "$DATA_DIR/val.parquet" ]; then
    echo "ERROR: Training/validation data not found at $DATA_DIR"
    exit 1
fi

# =============================================================================
# 创建目录
# =============================================================================
mkdir -p "$LOG_DIR"
mkdir -p "$REWARD_LOG_DIR"
mkdir -p "$TENSORBOARD_DIR"

# =============================================================================
# 缓存视频帧
# =============================================================================
if [ "${SKIP_VIDEO_CACHE:-false}" != "true" ]; then
    echo "===== Caching video frames ====="
    python "$PROJECT_DIR/examples/video_reasoning/cache_video_frames.py" \
        --input_parquet "$DATA_DIR/train.parquet" \
        --cache_dir "$CACHE_DIR" \
        --fps "$CACHE_FPS" \
        --max_frames "$CACHE_MAX_FRAMES"
fi

# =============================================================================
# 启动训练
# =============================================================================
echo "===== Starting Training ====="
export TENSORBOARD_DIR="$TENSORBOARD_DIR"

cd "$PROJECT_DIR"
python3 -m recipe.dapo.main_dapo \
    --config-path="$CONFIG_PATH" \
    --config-name='base' \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/val.parquet" \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.gen_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=36000 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.clip_ratio_low=$CLIP_RATIO_LOW \
    actor_rollout_ref.actor.clip_ratio_high=$CLIP_RATIO_HIGH \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFF \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.top_p=$TOP_P \
    actor_rollout_ref.rollout.top_k=$TOP_K \
    actor_rollout_ref.rollout.temperature=$TEMPERATURE \
    +actor_rollout_ref.rollout.enable_sleep_mode=False \
    +actor_rollout_ref.rollout.repetition_penalty=1.1 \
    +actor_rollout_ref.rollout.max_tokens_per_turn=2048 \
    actor_rollout_ref.rollout.n=$N_ROLLOUTS \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.max_model_len=128000 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=104768 \
    actor_rollout_ref.rollout.calculate_log_probs=true \
    actor_rollout_ref.rollout.over_sample_rate=0.1 \
    actor_rollout_ref.rollout.update_weights_bucket_megabytes=512 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=5 \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=5 \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.multi_turn.cache_config.cache_dir=$CACHE_DIR \
    actor_rollout_ref.rollout.multi_turn.cache_config.fps=$CACHE_FPS \
    actor_rollout_ref.rollout.multi_turn.cache_config.max_frames=$CACHE_MAX_FRAMES \
    actor_rollout_ref.rollout.multi_turn.cache_config.max_frames_per_segment=$CACHE_MAX_FRAMES_PER_SEGMENT \
    actor_rollout_ref.rollout.multi_turn.cache_config.use_cached_initial_video=$USE_CACHED_INITIAL_VIDEO \
    actor_rollout_ref.rollout.agent.default_agent_loop=video_reasoning \
    actor_rollout_ref.rollout.agent.num_workers=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=104768 \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.adv_estimator=grpo \
    algorithm.rollout_correction.bypass_mode=False \
    algorithm.norm_adv_by_std_in_grpo=$NORM_ADV_BY_STD \
    algorithm.use_kl_in_reward=False \
    algorithm.filter_groups.enable=False \
    algorithm.filter_groups.metric=score \
    algorithm.filter_groups.max_num_gen_batches=5 \
    algorithm.token_placement.method=$TOKEN_PLACEMENT_METHOD \
    algorithm.token_placement.bbox_context_window=$BBOX_CONTEXT_WINDOW \
    algorithm.token_placement.segment_context_window=$SEGMENT_CONTEXT_WINDOW \
    reward_model.enable=False \
    reward_model.reward_manager=dapo \
    reward_model.overlong_buffer.enable=True \
    reward_model.overlong_buffer.len=2048 \
    reward_model.overlong_buffer.penalty_factor=2.0 \
    custom_reward_function.path=pkg://verl.utils.reward_score.video_reasoning_async \
    custom_reward_function.name=compute_score \
    custom_reward_function.reward_kwargs.vlm_endpoint="$VLM_ENDPOINT" \
    custom_reward_function.reward_kwargs.vlm_model_name="$VLM_MODEL_NAME" \
    custom_reward_function.reward_kwargs.vlm_api_key="$VLM_API_KEY" \
    custom_reward_function.reward_kwargs.use_vlm_scoring=$USE_VLM_SCORING \
    custom_reward_function.reward_kwargs.use_bbox_verification=$USE_BBOX_VERIFICATION \
    custom_reward_function.reward_kwargs.answer_weight=$ANSWER_WEIGHT \
    custom_reward_function.reward_kwargs.bbox_weight=$BBOX_WEIGHT \
    custom_reward_function.reward_kwargs.bbox_coord_range=1.0 \
    custom_reward_function.reward_kwargs.cache_dir="$CACHE_DIR" \
    custom_reward_function.reward_kwargs.cache_fps=$CACHE_FPS \
    custom_reward_function.reward_kwargs.cache_max_frames=$CACHE_MAX_FRAMES \
    custom_reward_function.reward_kwargs.save_bbox_visualization=true \
    custom_reward_function.reward_kwargs.bbox_vis_sample_rate=0.001 \
    custom_reward_function.reward_kwargs.enable_logging=true \
    custom_reward_function.reward_kwargs.save_samples=true \
    custom_reward_function.reward_kwargs.save_every_n=10 \
    custom_reward_function.reward_kwargs.log_dir="$REWARD_LOG_DIR" \
    custom_reward_function.reward_kwargs.log_every_n=10 \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=$NNODES \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.val_before_train=$VAL_BEFORE_TRAIN \
    trainer.critic_warmup=0 \
    trainer.resume_mode=$RESUME_MODE \
    trainer.logger='["console", "tensorboard"]' \
    +ray_kwargs.ray_init.runtime_env.env_vars.TENSORBOARD_DIR="$TENSORBOARD_DIR" \
    "$@" 2>&1 | tee -a "$LOG_FILE"

echo "===== Training Complete ====="
