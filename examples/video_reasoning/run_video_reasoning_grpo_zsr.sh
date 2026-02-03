#!/bin/bash
# Video Reasoning GRPO Training Script
# 运行前确保在 verl 项目根目录

# set -x
ulimit -n 65535

# ===== GPU 配置（使用8张卡）=====
# export CUDA_VISIBLE_DEVICES=4,5,6,7
export VLLM_USE_V1=1
export LD_LIBRARY_PATH=/usr/local/cuda-13.1/compat:$LD_LIBRARY_PATH

# ===== 模型路径（需要修改）=====
MODEL_PATH="/data_gpu/zhengshurong/data/project/Qwen2.5-VL/qwen-vl-finetune/checkpoints/video/Qwen2.5-VL-7B-Instruct-self_holmes_multiturn_1k5-self_longvideoreason_multiturn_5k3-sft-lr5e-5-bs32"

# ===== 数据路径（固定）=====
DATA_DIR="./long_video_data"
VIDEO_BASE_PATH="/data_gpu/zhengshurong/data/dataset/LongVideo-Reason_videos/longvila_videos"
CACHE_DIR="./.cache"
CONFIG_PATH="$(pwd)/examples/video_reasoning/config"

# ===== 缓存参数 =====
CACHE_FPS=1
CACHE_MAX_FRAMES=512
CACHE_MAX_FRAMES_PER_SEGMENT=32
USE_CACHED_INITIAL_VIDEO=True         # 第一轮使用缓存帧而非原始视频，减少 CPU 内存占用

# ===== 初始视频分辨率参数（对齐 eval 脚本）=====
INITIAL_VIDEO_FPS=1
INITIAL_VIDEO_MAX_FRAMES=512
INITIAL_VIDEO_MIN_PIXELS=784          # 28*28, 对齐 eval 脚本 processor min_pixels
INITIAL_VIDEO_MAX_PIXELS=12544        # ~112x112, 低分辨率概览

# ===== Segment 视频分辨率参数（对齐 eval 脚本）=====
SEGMENT_VIDEO_FPS=1
SEGMENT_VIDEO_MAX_FRAMES=32
SEGMENT_VIDEO_MIN_PIXELS=784          # 28*28, 对齐 eval 脚本
SEGMENT_VIDEO_MAX_PIXELS=50176        # ~224x224, 高分辨率看细节

# ===== 训练参数（8卡配置）=====
TRAIN_BATCH_SIZE=16
MAX_PROMPT_LENGTH=36000
MAX_RESPONSE_LENGTH=16384
N_ROLLOUTS=8
LEARNING_RATE=1e-6
N_GPUS=8
AGENT_NUM_WORKERS=8                   # AgentLoopWorker 数量，减少可降低 CPU 内存占用

# ===== 实验名称 =====
TIMESTAMP=$(date '+%Y%m%d-%H%M%S')
EXPERIMENT_NAME="video_reasoning_grpo_${TIMESTAMP}"

# ===== 日志配置 =====
LOG_DIR="./logs_zsr"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${EXPERIMENT_NAME}.log"
echo "Log file: $LOG_FILE"

# ===== PRE-CHECK =====
# Check if training data exists
if [ ! -f "$DATA_DIR/train.parquet" ] || [ ! -f "$DATA_DIR/val.parquet" ]; then
    echo "ERROR: Training/validation data not found"
    echo "Please run preprocessing first:"
    echo "  bash examples/video_reasoning/preprocess_video_reasoning_data.sh"
    exit 1
fi

# Check required columns in parquet
python3 -c "
import pandas as pd
import sys
df = pd.read_parquet('$DATA_DIR/train.parquet')
required = ['extra_info', 'video_path', 'videos', 'prompt']
missing = [c for c in required if c not in df.columns]
if missing:
    print(f'ERROR: Parquet missing required columns: {missing}')
    print('Please regenerate parquet with:')
    print('  bash examples/video_reasoning/preprocess_video_reasoning_data.sh')
    sys.exit(1)
# Check extra_info is dict (not string)
sample_extra = df['extra_info'].iloc[0]
if isinstance(sample_extra, str):
    print('ERROR: extra_info should be dict, got string')
    print('Please regenerate parquet with:')
    print('  bash examples/video_reasoning/preprocess_video_reasoning_data.sh')
    sys.exit(1)
" || exit 1

echo "===== Configuration ====="
echo "MODEL_PATH: $MODEL_PATH"
echo "DATA_DIR: $DATA_DIR"
echo "VIDEO_BASE_PATH: $VIDEO_BASE_PATH"
echo "CACHE_DIR: $CACHE_DIR"
echo "CACHE_FPS: $CACHE_FPS"
echo "CACHE_MAX_FRAMES: $CACHE_MAX_FRAMES"
echo "TRAIN_BATCH_SIZE: $TRAIN_BATCH_SIZE"
echo "N_ROLLOUTS: $N_ROLLOUTS"
echo "N_GPUS: $N_GPUS"
echo "AGENT_NUM_WORKERS: $AGENT_NUM_WORKERS"
echo "========================="

# ===== STEP 1: Cache video frames =====
echo "===== Step 1: Caching video frames (fps=$CACHE_FPS, max_frames=$CACHE_MAX_FRAMES) ====="
# 直接从 parquet 的 video_path 列读取路径
python examples/video_reasoning/cache_video_frames.py \
    --input_parquet "$DATA_DIR/train.parquet" \
    --cache_dir "$CACHE_DIR" \
    --fps "$CACHE_FPS" \
    --max_frames "$CACHE_MAX_FRAMES"

# Check if caching was successful
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to cache video frames"
    exit 1
fi

echo "===== Step 1 Complete: Video frames cached ====="

# ===== STEP 2: Run training =====
echo "===== Step 2: Starting GRPO training ====="
echo "Training log: $LOG_FILE"
nohup python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='video_reasoning_grpo' \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=$N_ROLLOUTS \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.rollout.over_sample_rate=0.1 \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=5 \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=5 \
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
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.agent.default_agent_loop=video_reasoning \
    actor_rollout_ref.rollout.agent.num_workers=$AGENT_NUM_WORKERS \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name='video-reasoning-grpo' \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=30 \
    trainer.test_freq=20 \
    trainer.val_before_train=True \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/val.parquet" \
    reward_model.enable=False \
    custom_reward_function.path=pkg://verl.utils.reward_score.video_reasoning_async \
    custom_reward_function.name=compute_score \
    custom_reward_function.reward_kwargs.log_dir="./reward_logs_zsr" \
    custom_reward_function.reward_kwargs.use_vlm_scoring=true \
    custom_reward_function.reward_kwargs.use_bbox_verification=true \
    custom_reward_function.reward_kwargs.vlm_endpoint="10.96.11.3:8081" \
    custom_reward_function.reward_kwargs.vlm_model_name="Qwen3-VL-30B-A3B-Instruct" \
    custom_reward_function.reward_kwargs.vlm_api_key="123456" \
    custom_reward_function.reward_kwargs.cache_dir="$CACHE_DIR" \
    custom_reward_function.reward_kwargs.cache_fps=$CACHE_FPS \
    custom_reward_function.reward_kwargs.cache_max_frames=$CACHE_MAX_FRAMES \
    trainer.total_epochs=5 \
    trainer.rollout_data_dir="$LOG_DIR/${EXPERIMENT_NAME}/rollout" \
    trainer.validation_data_dir="$LOG_DIR/${EXPERIMENT_NAME}/validation" \
    actor_rollout_ref.rollout.update_weights_bucket_megabytes=512 "$@" 2>&1 | tee -a "$LOG_FILE" &

echo "Training started in background. PID: $!"
echo "View logs: tail -f $LOG_FILE"

#   custom_reward_function.reward_kwargs.use_bbox_verification=true \                                                                                                
#   custom_reward_function.reward_kwargs.answer_weight=0.4 \                                                                                                         
#   custom_reward_function.reward_kwargs.bbox_weight=0.3 \                                                                                                           
#   custom_reward_function.reward_kwargs.vlm_weight=0.3 \                                                                                                            
#   custom_reward_function.reward_kwargs.vlm_endpoint="10.96.11.3:8081" \                                                                                             
#   custom_reward_function.reward_kwargs.vlm_model_name="Qwen3-VL-30B-A3B-Instruct" \                                                                                
#   custom_reward_function.reward_kwargs.vlm_api_key="123456"  