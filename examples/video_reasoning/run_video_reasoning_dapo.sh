#!/bin/bash
# =============================================================================
# Video Reasoning DAPO Training Script
# =============================================================================
# 所有配置集中在此文件，YAML 只做最小继承
#
# 使用方式：
#   bash run_video_reasoning_dapo.sh                    # 默认运行
#   bash run_video_reasoning_dapo.sh trainer.total_epochs=10  # 临时覆盖参数
#
# 特性：
#   - 使用 DAPO recipe，支持 filter_groups（过滤全对/全错组）
#   - 支持 Clip-Higher（非对称 clip ratio）
#   - 支持 Dr.GRPO（可选不除以 std）
#
# Hydra 配置加载优先级（从低到高）：
#   1. verl/trainer/config/ppo_trainer.yaml (base defaults via hydra searchpath)
#   2. examples/video_reasoning/config/base.yaml (--config-name 指定)
#   3. 命令行参数 key=value (本脚本中的配置)
#   4. "$@" 传入的额外参数 (运行时覆盖)
#
# 配置加载流程：
#   --config-path="$CONFIG_PATH"  指定配置目录
#   --config-name='base'          指定加载 base.yaml
#   base.yaml 中 defaults: [ppo_trainer] 会从 searchpath 加载 ppo_trainer.yaml
# =============================================================================

set -eo pipefail  # 遇错退出，管道中任一命令失败即退出

# =============================================================================
# 环境配置
# =============================================================================
ulimit -n 65535
export VLLM_USE_V1=1
export LD_LIBRARY_PATH=/usr/local/cuda-13.1/compat:$LD_LIBRARY_PATH
# export RAY_ADDRESS=local
# export CUDA_VISIBLE_DEVICES=0,1,2,3

# # Ray 在此机器上无法自动检测 GPU，需手动指定
# ray stop 2>/dev/null || true
# ray start --head --num-gpus=8

# =============================================================================
# 路径配置
# =============================================================================
#MODEL_PATH="/data_gpu/songlin/rl/verl/checkpoints/video-reasoning-dapo/video_reasoning_dapo_20260205-063449/merged_model"
MODEL_PATH="/data_gpu/zhengshurong/data/project/Qwen2.5-VL/qwen-vl-finetune/checkpoints/video/Qwen2.5-VL-7B-Instruct-self_holmes_caption_233-self_longvideoreason_caption_930-openo3video_stgr_singleturn_7k-self_holmes_multiturn_1k5-self_longvideoreason_multiturn_5k3-sft-lr5e-5-b24"
DATA_DIR="./long_video_data/longvt_selfqa"
CACHE_DIR="./.cache"
CONFIG_PATH="$(pwd)/examples/video_reasoning/config"
LOG_DIR="./logs"

# =============================================================================
# 训练参数
# =============================================================================
TRAIN_BATCH_SIZE=16
GEN_BATCH_SIZE=32                        # DAPO: 生成批次 2x 训练批次，给过滤留足余量，默认似乎需要是4x
MAX_PROMPT_LENGTH=36000
MAX_RESPONSE_LENGTH=16384

LEARNING_RATE=1e-6
TOTAL_EPOCHS=1

N_ROLLOUTS=8                             # 每个 prompt 生成的 response 数
AGENT_NUM_WORKERS=8                      # AgentLoopWorker 数量

N_GPUS=8
NNODES=1

# =============================================================================
# DAPO 算法参数
# =============================================================================
ENABLE_FILTER_GROUPS=True                # 过滤组内全对/全错的样本
FILTER_GROUPS_METRIC=acc                 # acc / score / seq_reward / seq_final_reward
MAX_NUM_GEN_BATCHES=5                    # 最多重采样轮数，0=无限制

CLIP_RATIO_LOW=0.2                       # Clip-Higher: 非对称 clip ratio
CLIP_RATIO_HIGH=0.28                     # > low，鼓励正向更新

NORM_ADV_BY_STD=False                     # False = Dr.GRPO 模式

USE_KL_IN_REWARD=False
USE_KL_LOSS=True
KL_LOSS_COEF=0.001
KL_LOSS_TYPE=low_var_kl

# =============================================================================
# 视频缓存参数
# =============================================================================
CACHE_FPS=1
CACHE_MAX_FRAMES=512
CACHE_MAX_FRAMES_PER_SEGMENT=32
USE_CACHED_INITIAL_VIDEO=True            # 使用缓存帧而非原始视频，减少 CPU 内存

# 初始视频分辨率（低分辨率概览）
INITIAL_VIDEO_FPS=1
INITIAL_VIDEO_MAX_FRAMES=512
INITIAL_VIDEO_MIN_PIXELS=784             # 28*28
INITIAL_VIDEO_MAX_PIXELS=12544           # ~112x112

# Segment 视频分辨率（高分辨率细节）
SEGMENT_VIDEO_FPS=1
SEGMENT_VIDEO_MAX_FRAMES=32
SEGMENT_VIDEO_MIN_PIXELS=784             # 28*28
SEGMENT_VIDEO_MAX_PIXELS=50176           # ~224x224

# =============================================================================
# 时间戳水印参数（可选功能）
# =============================================================================
# 启用后，rollout 时帧上会显示时间戳（如 "12s"），帮助模型理解时序
# logp 计算时使用原始帧（无水印），避免模型只学会从水印获取时序
USE_TIMESTAMP_WATERMARK=True            # 是否启用时间戳水印
WATERMARK_POSITION="top_left"            # 水印位置: top_left, top_right, bottom_left, bottom_right
WATERMARK_FONT_SIZE=0                    # 字体大小 (0=根据图片高度自适应)
WATERMARK_RATIO=0.5                     # 水印采样比例: 1.0=全部使用, 0.0=全部不使用, 0.5=50%采样

# =============================================================================
# 奖励函数参数
# =============================================================================
VLM_ENDPOINT="10.96.11.3:8081"
VLM_MODEL_NAME="Qwen3-VL-30B-A3B-Instruct"
VLM_API_KEY="123456"

USE_VLM_SCORING=true
USE_BBOX_VERIFICATION=true
ANSWER_WEIGHT=1.0
BBOX_WEIGHT=0.3
BBOX_COORD_RANGE=1.0                     # bbox 坐标范围 [0, 1]

SAVE_BBOX_VISUALIZATION=true
BBOX_VIS_SAMPLE_RATE=0.01
REWARD_LOG_DIR="./reward_logs"
SAVE_SAMPLES=true
SAVE_EVERY_N=10
LOG_EVERY_N=10

# =============================================================================
# Checkpoint 配置
# =============================================================================
SAVE_FREQ=30
TEST_FREQ=20
VAL_BEFORE_TRAIN=True
RESUME_MODE=disable                      # disable / resume_path / auto

# =============================================================================
# 实验名称
# =============================================================================
TIMESTAMP=$(date '+%Y%m%d-%H%M%S')
PROJECT_NAME="video-reasoning-dapo"
EXPERIMENT_NAME="video_reasoning_dapo_longvt_selfqa_watermark_0_5_genbs32_ep1_lr1e_6_${TIMESTAMP}"

# =============================================================================
# 预检查
# =============================================================================
echo "===== Pre-flight Checks ====="

if [ ! -f "$DATA_DIR/train.parquet" ] || [ ! -f "$DATA_DIR/val.parquet" ]; then
    echo "ERROR: Training/validation data not found at $DATA_DIR"
    echo "Please run preprocessing first:"
    echo "  bash examples/video_reasoning/preprocess_video_reasoning_data.sh"
    exit 1
fi

python3 -c "
import pandas as pd
import sys
df = pd.read_parquet('$DATA_DIR/train.parquet')
required = ['extra_info', 'video_path', 'videos', 'prompt']
missing = [c for c in required if c not in df.columns]
if missing:
    print(f'ERROR: Parquet missing required columns: {missing}')
    sys.exit(1)
sample_extra = df['extra_info'].iloc[0]
if isinstance(sample_extra, str):
    print('ERROR: extra_info should be dict, got string')
    sys.exit(1)
print(f'Data check passed: {len(df)} samples')
" || exit 1

# =============================================================================
# 打印配置摘要
# =============================================================================
echo ""
echo "===== Configuration Summary ====="
echo "Model:           $MODEL_PATH"
echo "Data:            $DATA_DIR"
echo "Train/Gen batch: $TRAIN_BATCH_SIZE / $GEN_BATCH_SIZE"
echo "Rollouts:        $N_ROLLOUTS"
echo "GPUs:            $N_GPUS x $NNODES nodes"
echo ""
echo "DAPO Settings:"
echo "  filter_groups: $ENABLE_FILTER_GROUPS (metric=$FILTER_GROUPS_METRIC)"
echo "  clip_ratio:    [$CLIP_RATIO_LOW, $CLIP_RATIO_HIGH]"
echo "  norm_by_std:   $NORM_ADV_BY_STD"
echo ""
echo "Reward:"
echo "  VLM scoring:   $USE_VLM_SCORING ($VLM_ENDPOINT)"
echo "  BBox verify:   $USE_BBOX_VERIFICATION"
echo "  Weights:       answer=$ANSWER_WEIGHT, bbox=$BBOX_WEIGHT"
echo "================================="
echo ""

# =============================================================================
# Step 1: 缓存视频帧
# =============================================================================
echo "===== Step 1: Caching video frames ====="
python examples/video_reasoning/cache_video_frames.py \
    --input_parquet "$DATA_DIR/train.parquet" \
    --cache_dir "$CACHE_DIR" \
    --fps "$CACHE_FPS" \
    --max_frames "$CACHE_MAX_FRAMES"

# set -eo pipefail 已确保上述命令失败时脚本自动退出
echo "===== Step 1 Complete ====="
echo ""

# =============================================================================
# Step 2: 启动训练
# =============================================================================
echo "===== Step 2: Starting DAPO Training ====="
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${EXPERIMENT_NAME}.log"
echo "Log file: $LOG_FILE"

python3 -m recipe.dapo.main_dapo \
    --config-path="$CONFIG_PATH" \
    --config-name='base' \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/val.parquet" \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.gen_batch_size=$GEN_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.clip_ratio_low=$CLIP_RATIO_LOW \
    actor_rollout_ref.actor.clip_ratio_high=$CLIP_RATIO_HIGH \
    actor_rollout_ref.actor.use_kl_loss=$USE_KL_LOSS \
    actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
    actor_rollout_ref.actor.kl_loss_type=$KL_LOSS_TYPE \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=$N_ROLLOUTS \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
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
    actor_rollout_ref.rollout.multi_turn.initial_video_config.fps=$INITIAL_VIDEO_FPS \
    actor_rollout_ref.rollout.multi_turn.initial_video_config.max_frames=$INITIAL_VIDEO_MAX_FRAMES \
    actor_rollout_ref.rollout.multi_turn.initial_video_config.min_pixels=$INITIAL_VIDEO_MIN_PIXELS \
    actor_rollout_ref.rollout.multi_turn.initial_video_config.max_pixels=$INITIAL_VIDEO_MAX_PIXELS \
    actor_rollout_ref.rollout.multi_turn.segment_video_config.fps=$SEGMENT_VIDEO_FPS \
    actor_rollout_ref.rollout.multi_turn.segment_video_config.max_frames=$SEGMENT_VIDEO_MAX_FRAMES \
    actor_rollout_ref.rollout.multi_turn.segment_video_config.min_pixels=$SEGMENT_VIDEO_MIN_PIXELS \
    actor_rollout_ref.rollout.multi_turn.segment_video_config.max_pixels=$SEGMENT_VIDEO_MAX_PIXELS \
    actor_rollout_ref.rollout.multi_turn.watermark_config.enable=$USE_TIMESTAMP_WATERMARK \
    actor_rollout_ref.rollout.multi_turn.watermark_config.position=$WATERMARK_POSITION \
    actor_rollout_ref.rollout.multi_turn.watermark_config.font_size=$WATERMARK_FONT_SIZE \
    actor_rollout_ref.rollout.multi_turn.watermark_config.ratio=$WATERMARK_RATIO \
    actor_rollout_ref.rollout.agent.default_agent_loop=video_reasoning \
    actor_rollout_ref.rollout.agent.num_workers=$AGENT_NUM_WORKERS \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=$NORM_ADV_BY_STD \
    algorithm.use_kl_in_reward=$USE_KL_IN_REWARD \
    algorithm.filter_groups.enable=$ENABLE_FILTER_GROUPS \
    algorithm.filter_groups.metric=$FILTER_GROUPS_METRIC \
    algorithm.filter_groups.max_num_gen_batches=$MAX_NUM_GEN_BATCHES \
    reward_model.enable=False \
    reward_model.reward_manager=dapo \
    reward_model.overlong_buffer.enable=True \
    reward_model.overlong_buffer.len=256 \
    reward_model.overlong_buffer.penalty_factor=1.0 \
    custom_reward_function.path=pkg://verl.utils.reward_score.video_reasoning_async \
    custom_reward_function.name=compute_score \
    custom_reward_function.reward_kwargs.vlm_endpoint="$VLM_ENDPOINT" \
    custom_reward_function.reward_kwargs.vlm_model_name="$VLM_MODEL_NAME" \
    custom_reward_function.reward_kwargs.vlm_api_key="$VLM_API_KEY" \
    custom_reward_function.reward_kwargs.use_vlm_scoring=$USE_VLM_SCORING \
    custom_reward_function.reward_kwargs.use_bbox_verification=$USE_BBOX_VERIFICATION \
    custom_reward_function.reward_kwargs.answer_weight=$ANSWER_WEIGHT \
    custom_reward_function.reward_kwargs.bbox_weight=$BBOX_WEIGHT \
    custom_reward_function.reward_kwargs.bbox_coord_range=$BBOX_COORD_RANGE \
    custom_reward_function.reward_kwargs.cache_dir="$CACHE_DIR" \
    custom_reward_function.reward_kwargs.cache_fps=$CACHE_FPS \
    custom_reward_function.reward_kwargs.cache_max_frames=$CACHE_MAX_FRAMES \
    custom_reward_function.reward_kwargs.save_bbox_visualization=$SAVE_BBOX_VISUALIZATION \
    custom_reward_function.reward_kwargs.bbox_vis_sample_rate=$BBOX_VIS_SAMPLE_RATE \
    custom_reward_function.reward_kwargs.enable_logging=true \
    custom_reward_function.reward_kwargs.save_samples=$SAVE_SAMPLES \
    custom_reward_function.reward_kwargs.save_every_n=$SAVE_EVERY_N \
    custom_reward_function.reward_kwargs.log_dir="$REWARD_LOG_DIR" \
    custom_reward_function.reward_kwargs.log_every_n=$LOG_EVERY_N \
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
    "$@" 2>&1 | tee -a "$LOG_FILE"

# trainer.resume_from_path=/data_gpu/songlin/rl/verl/checkpoints/video-reasoning-grpo/video_reasoning_grpo_20260131-085501/global_step_200

echo ""
echo "===== Step 2 Complete: Training Finished ====="

# =============================================================================
# Step 3: 自动合并模型 (Merge FSDP checkpoints to HuggingFace format)
# =============================================================================
CKPT_BASE="./checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}"

if [ -f "$CKPT_BASE/latest_checkpointed_iteration.txt" ]; then
    LATEST_STEP=$(cat "$CKPT_BASE/latest_checkpointed_iteration.txt")
    echo ""
    echo "===== Step 3: Merging checkpoint global_step_${LATEST_STEP} ====="
    echo "Source: $CKPT_BASE/global_step_${LATEST_STEP}/actor"
    echo "Target: $CKPT_BASE/merged_model"

    python -m verl.model_merger merge \
        --backend fsdp \
        --local_dir "$CKPT_BASE/global_step_${LATEST_STEP}/actor" \
        --target_dir "$CKPT_BASE/merged_model" \
        --trust-remote-code

    if [ $? -eq 0 ]; then
        echo ""
        echo "===== Step 3 Complete: Model Merged Successfully ====="
        echo "Merged model saved to: $CKPT_BASE/merged_model"
        echo ""
        echo "You can load the model with:"
        echo "  from transformers import AutoModelForVision2Seq, AutoProcessor"
        echo "  model = AutoModelForVision2Seq.from_pretrained('$CKPT_BASE/merged_model', trust_remote_code=True)"
    else
        echo "ERROR: Failed to merge model"
        exit 1
    fi
else
    echo ""
    echo "WARNING: No checkpoint found at $CKPT_BASE"
    echo "Skipping model merge step."
fi

echo ""
echo "===== All Steps Complete ====="
echo "Log file: $LOG_FILE"
echo "TensorBoard: tensorboard --logdir=./tensorboard_log"
