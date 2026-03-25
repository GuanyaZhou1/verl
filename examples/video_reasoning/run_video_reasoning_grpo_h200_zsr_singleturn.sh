#!/bin/bash
# =============================================================================
# Video Reasoning GRPO Training Script
# =============================================================================
# 所有配置集中在此文件，YAML 只做最小继承
#
# 使用方式：
#   bash run_video_reasoning_dapo.sh                    # 默认运行
#   bash run_video_reasoning_dapo.sh trainer.total_epochs=10  # 临时覆盖参数
#
# 特性：
#   - 使用 GRPO algorithm，支持 filter_groups（过滤全对/全错组）
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
export HYDRA_FULL_ERROR=1  # 报错时打印完整 stack trace

# =============================================================================
# 环境配置
# =============================================================================
ulimit -n 65535
export VLLM_USE_V1=1
# export TIKTOKEN_CACHE_DIR=${TIKTOKEN_CACHE_DIR:-/data_gpu/gyzhou/tmp/tiktoken_cache}
# export LD_LIBRARY_PATH=/usr/local/cuda-13.1/compat:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=$HOME/cuda-13.1/compat:$LD_LIBRARY_PATH  # 已删除 cuda-13.1 时注释，用系统默认
export TMPDIR=/tmp  # multiprocessing 临时文件放本地磁盘，避免 NFS 上 EBUSY 错误
export RAY_TMPDIR=/tmp/ray_$USER  # 隔离 Ray 临时目录，避免多用户权限冲突
# export RAY_ADDRESS=local
# export CUDA_VISIBLE_DEVICES=0,1,2,3
export TIKTOKEN_RS_CACHE_DIR=/mnt/data/home/zhengshurong/harmony_cache
# =============================================================================
# NCCL 环境变量（IB 优先，单机多机通用）
# =============================================================================
# 使用 ${VAR:-default} 语法，允许 launcher 脚本覆盖
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}               # 启用 IB
export NCCL_IB_HCA=${NCCL_IB_HCA:-^mlx5_bond,mlx5_6}       # 排除 bond 和有问题的 mlx5_6
export NCCL_CROSS_NIC=${NCCL_CROSS_NIC:-1}                 # 允许跨 NIC 通信
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-bond0}     # fallback 以太网接口
export NCCL_SOCKET_FAMILY=${NCCL_SOCKET_FAMILY:-AF_INET}   # 强制 IPv4
export TORCH_NCCL_AVOID_RECORD_STREAMS=${TORCH_NCCL_AVOID_RECORD_STREAMS:-1}
export NCCL_CUMEM_ENABLE=${NCCL_CUMEM_ENABLE:-0}

# Ray 在此机器上无法自动检测 GPU（/dev/vfio 被误判为 TPU），需手动指定
# 放在 Step 1 缓存视频之前启动，利用缓存时间完成集群初始化
# ray stop 2>/dev/null || true
# ray start --head --num-gpus=${N_GPUS:-8}

# =============================================================================
# 路径配置
# =============================================================================
MODEL_PATH="${MODEL_PATH:-/mnt/data/home/zhengshurong/hf_cache/Qwen/Qwen3-VL-8B-Instruct}"
# MODEL_PATH="${MODEL_PATH:-/mnt/data/home/zhengshurong/hf_cache/Qwen/Qwen2.5-VL-7B-Instruct}"
#MODEL_PATH="/data_gpu/songlin/rl/verl/checkpoints/video-reasoning-grpo/video_reasoning_grpo_20260205-063449/merged_model"
#MODEL_PATH="/data_gpu/zhengshurong/data/project/Qwen2.5-VL/qwen-vl-finetune/checkpoints/video/Qwen2.5-VL-7B-Instruct-self_holmes_caption_233-self_longvideoreason_caption_930-openo3video_stgr_singleturn_7k-self_holmes_multiturn_1k5-self_longvideoreason_multiturn_5k3-sft-lr5e-5-b24"
# MODEL_PATH="${MODEL_PATH:-/data_gpu/zhengshurong/data/project/Qwen2.5-VL/qwen-vl-finetune/checkpoints/video/Qwen2.5-VL-7B-Instruct-stgr-turn_llm_freeze25_freeze_mlp-lr1e-5-epo5}"
# MODEL_PATH="/mnt/data/home/zhengshurong/project/Qwen3-VL/qwen-vl-finetune/checkpoints/video/Qwen3-VL-8B-Instruct-longvtdata-stgrdata-selfconstructdata-sft-lr1e-5-bs128-ep1/checkpoint-3003"
# MODEL_PATH="/mnt/data/home/zhengshurong/project/Qwen3-VL/qwen-vl-finetune/checkpoints/video/Qwen3-VL-8B-Instruct-longvt_tvg-openo3video_stgr-selfconstructdata-sft-lr1e-5-bs64-ep1"
DATA_DIR="${DATA_DIR:-./long_video_data_singleturn/longvt_selfqa}"
CACHE_DIR="${CACHE_DIR:-./.cache_new}"
CONFIG_PATH="$(pwd)/examples/video_reasoning/config"
LOG_DIR="./logs_zsr"

# =============================================================================
# 训练参数 (支持环境变量覆盖)
# =============================================================================
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
GEN_BATCH_SIZE=${GEN_BATCH_SIZE:-16}     # GRPO: 生成批次，开启 filter 时需要增大
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-16384}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}

LEARNING_RATE=${LEARNING_RATE:-1e-6}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-15}

N_ROLLOUTS=${N_ROLLOUTS:-8}              # 每个 prompt 生成的 response 数
AGENT_NUM_WORKERS=${AGENT_NUM_WORKERS:-4}

N_GPUS=${N_GPUS:-8}
NNODES=${NNODES:-1}

# =============================================================================
# GRPO 算法参数 (支持环境变量覆盖)
# =============================================================================
NORM_ADV_BY_STD=${NORM_ADV_BY_STD:-True}  # 归一化 advantage by std

USE_KL_IN_REWARD=False
USE_KL_LOSS=True
KL_LOSS_COEF=${KL_LOSS_COEF:-0.01}
KL_LOSS_TYPE=low_var_kl

ENTROPY_COEFF=${ENTROPY_COEFF:-0.0}                   # Entropy 系数
TOP_P=${TOP_P:-1.0}                                   # Top-p 采样
TEMPERATURE=${TEMPERATURE:-1.0}                       # 采样温度

# Ulysses 序列并行 (设为1禁用，可能导致KL loss异常)
ULYSSES_SP_SIZE=${ULYSSES_SP_SIZE:-1}

# =============================================================================
# 视频缓存参数
# =============================================================================
CACHE_FPS=1
CACHE_MAX_FRAMES=0
CACHE_MAX_FRAMES_PER_SEGMENT=32
USE_CACHED_INITIAL_VIDEO=True            # 使用缓存帧而非原始视频，减少 CPU 内存
CACHE_NUM_WORKERS=${CACHE_NUM_WORKERS:-64}  # 视频缓存并行数

# 初始视频分辨率（低分辨率概览）
INITIAL_VIDEO_FPS=1
INITIAL_VIDEO_MAX_FRAMES=32
INITIAL_VIDEO_MIN_PIXELS=784             # 28*28
# INITIAL_VIDEO_MAX_PIXELS=12544           # ~112x112
INITIAL_VIDEO_MAX_PIXELS=50176 # ~448x448

# Segment 视频分辨率（高分辨率细节）
SEGMENT_VIDEO_FPS=1
SEGMENT_VIDEO_MAX_FRAMES=32
SEGMENT_VIDEO_MIN_PIXELS=784             # 28*28
# SEGMENT_VIDEO_MAX_PIXELS=50176           # ~224x224
SEGMENT_VIDEO_MAX_PIXELS=327680 # ~640x640

# =============================================================================
# 时间戳水印参数（可选功能）
# =============================================================================
# 启用后，rollout 时帧上会显示时间戳（如 "12s"），帮助模型理解时序
# logp 计算时使用原始帧（无水印），避免模型只学会从水印获取时序
USE_TIMESTAMP_WATERMARK=False            # 是否启用时间戳水印
WATERMARK_POSITION="top_left"            # 水印位置: top_left, top_right, bottom_left, bottom_right
WATERMARK_FONT_SIZE=0                    # 字体大小 (0=根据图片高度自适应)
WATERMARK_RATIO=1.0                     # 水印采样比例: 1.0=全部使用, 0.0=全部不使用, 0.5=50%采样

# =============================================================================
# 奖励函数参数
# =============================================================================
VLM_ENDPOINT="10.0.1.35:8081"
VLM_MODEL_NAME="Qwen3-VL-235B-A22B-Instruct"
VLM_API_KEY="123456"

USE_VLM_SCORING=true
USE_BBOX_VERIFICATION=true
ANSWER_WEIGHT=${ANSWER_WEIGHT:-1.0}
BBOX_WEIGHT=${BBOX_WEIGHT:-0.0}
FORMAT_WEIGHT=${FORMAT_WEIGHT:-1.0}          # 格式奖励权重 (0 = 不使用)
SEGMENT_WEIGHT=${SEGMENT_WEIGHT:-0.0}        # segment 时间段匹配奖励权重 (0 = 不使用)
USE_STRICT_FORMAT=${USE_STRICT_FORMAT:-true}  # 是否使用严格的 segment 格式检查
BBOX_COORD_RANGE=1.0                     # bbox 坐标范围 [0, 1]，不影响，内部后续改为根据模型的输出动态调整

SAVE_BBOX_VISUALIZATION=true
BBOX_VIS_SAMPLE_RATE=0.001
# REWARD_LOG_DIR 在 EXPERIMENT_NAME 后设置
SAVE_SAMPLES=true
SAVE_EVERY_N=1
LOG_EVERY_N=10

# =============================================================================
# Checkpoint 配置
# =============================================================================
SAVE_FREQ=20
TEST_FREQ=5
VAL_BEFORE_TRAIN=True
RESUME_MODE=disable                      # disable / resume_path / auto

# =============================================================================
# Agent Loop 类型
# =============================================================================
AGENT_LOOP_TYPE=${AGENT_LOOP_TYPE:-single_turn_agent}

# =============================================================================
# 实验名称
# =============================================================================
TIMESTAMP=$(date '+%Y%m%d-%H%M%S')
PROJECT_NAME="video-reasoning-grpo"
EXPERIMENT_NAME="Qwen3-VL-8B-Instruct_grpo_longvtrl_singleturn_agent${AGENT_LOOP_TYPE}_genbs${GEN_BATCH_SIZE}_trainbs${TRAIN_BATCH_SIZE}_ep${TOTAL_EPOCHS}_lr${LEARNING_RATE}_ans${ANSWER_WEIGHT}_bbox${BBOX_WEIGHT}_fmt${FORMAT_WEIGHT}_seg${SEGMENT_WEIGHT}_strictfmt${USE_STRICT_FORMAT}_klcoef${KL_LOSS_COEF}_resp${MAX_RESPONSE_LENGTH}_topp${TOP_P}_temp${TEMPERATURE}_usp${ULYSSES_SP_SIZE}"
# EXPERIMENT_NAME="Qwen3_8B_longvt_tvg-openo3video_stgr-selfconstructdata_grpo_long_video_data_genbs32_ep1_lr1e_6_bbox0_0_normadvbystdfalse_${TIMESTAMP}"

# 将 reward_logs 和 tensorboard_log 放到 checkpoint 目录下
CKPT_BASE="./checkpoints_zsr/${PROJECT_NAME}/${EXPERIMENT_NAME}"
REWARD_LOG_DIR="${CKPT_BASE}/reward_logs"
ROLLOUT_DATA_DIR="${CKPT_BASE}/rollout_data"
TENSORBOARD_DIR="${CKPT_BASE}/tensorboard_log"
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
echo "Cache workers:   $CACHE_NUM_WORKERS"
echo ""
echo "GRPO Settings:"
echo "  norm_by_std:   $NORM_ADV_BY_STD"
echo "  kl_loss_coef:  $KL_LOSS_COEF"
echo "  top_p:         $TOP_P"
echo "  temperature:   $TEMPERATURE"
echo ""
echo "Reward:"
echo "  VLM scoring:   $USE_VLM_SCORING ($VLM_ENDPOINT)"
echo "  BBox verify:   $USE_BBOX_VERIFICATION"
echo "  Weights:       answer=$ANSWER_WEIGHT, bbox=$BBOX_WEIGHT, format=$FORMAT_WEIGHT, segment=$SEGMENT_WEIGHT"
echo "  Strict format: $USE_STRICT_FORMAT"
echo "================================="
echo ""

# =============================================================================
# Step 1: 缓存视频帧
# =============================================================================
if [ "${SKIP_VIDEO_CACHE:-false}" != "true" ]; then
    echo "===== Step 1: Caching video frames ====="
    python examples/video_reasoning/cache_video_frames.py \
        --input_parquet "$DATA_DIR/train.parquet" \
        --cache_dir "$CACHE_DIR" \
        --fps "$CACHE_FPS" \
        --max_frames "$CACHE_MAX_FRAMES" \
        --num_workers "$CACHE_NUM_WORKERS"

    # set -eo pipefail 已确保上述命令失败时脚本自动退出
    echo "===== Step 1 Complete ====="
    echo ""
else
    echo "===== Step 1: Skipping cache (handled by launcher) ====="
    echo ""
fi
    # +actor_rollout_ref.model.override_config.attn_implementation=eager \
    # +critic.model.override_config.attn_implementation=eager \
# =============================================================================
# Step 2: 启动训练
# =============================================================================
echo "===== Step 2: Starting DAPO Training ====="
mkdir -p "$LOG_DIR"
mkdir -p "$REWARD_LOG_DIR"
mkdir -p "$ROLLOUT_DATA_DIR"
mkdir -p "$TENSORBOARD_DIR"
LOG_FILE="$LOG_DIR/${EXPERIMENT_NAME}/${TIMESTAMP}.log"
mkdir -p "$(dirname "$LOG_FILE")"
echo "Log file: $LOG_FILE"
echo "Reward logs: $REWARD_LOG_DIR"
echo "Rollout data: $ROLLOUT_DATA_DIR"
echo "TensorBoard: $TENSORBOARD_DIR"

# 设置 TensorBoard 目录环境变量
export TENSORBOARD_DIR="$TENSORBOARD_DIR"

python3 -m verl.trainer.main_ppo \
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
    data.image_key=images \
    +data.video_fps=1 \
    +data.video_max_frames=32 \
    +data.video_min_frames=4 \
    +data.max_pixels=50176 \
    +data.min_pixels=3136 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
    actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=$USE_KL_LOSS \
    actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
    actor_rollout_ref.actor.kl_loss_type=$KL_LOSS_TYPE \
    actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFF \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$ULYSSES_SP_SIZE \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.top_p=$TOP_P \
    actor_rollout_ref.rollout.temperature=$TEMPERATURE \
    +actor_rollout_ref.rollout.enable_sleep_mode=False \
    +actor_rollout_ref.rollout.repetition_penalty=1.0 \
    +actor_rollout_ref.rollout.max_tokens_per_turn=2048 \
    actor_rollout_ref.rollout.n=$N_ROLLOUTS \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.max_model_len=128000 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=104768 \
    actor_rollout_ref.rollout.calculate_log_probs=true \
    actor_rollout_ref.rollout.over_sample_rate=0.1 \
    actor_rollout_ref.rollout.update_weights_bucket_megabytes=4096 \
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
    actor_rollout_ref.rollout.agent.default_agent_loop=${AGENT_LOOP_TYPE} \
    actor_rollout_ref.rollout.agent.num_workers=$AGENT_NUM_WORKERS \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=104768 \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=$ULYSSES_SP_SIZE \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=$NORM_ADV_BY_STD \
    algorithm.use_kl_in_reward=$USE_KL_IN_REWARD \
    reward_model.enable=False \
    custom_reward_function.path=pkg://verl.utils.reward_score.video_reasoning_async \
    custom_reward_function.name=compute_score \
    custom_reward_function.reward_kwargs.vlm_endpoint="$VLM_ENDPOINT" \
    custom_reward_function.reward_kwargs.vlm_model_name="$VLM_MODEL_NAME" \
    custom_reward_function.reward_kwargs.vlm_api_key="$VLM_API_KEY" \
    custom_reward_function.reward_kwargs.use_vlm_scoring=$USE_VLM_SCORING \
    custom_reward_function.reward_kwargs.use_bbox_verification=$USE_BBOX_VERIFICATION \
    custom_reward_function.reward_kwargs.answer_weight=$ANSWER_WEIGHT \
    custom_reward_function.reward_kwargs.bbox_weight=$BBOX_WEIGHT \
    custom_reward_function.reward_kwargs.format_weight=$FORMAT_WEIGHT \
    custom_reward_function.reward_kwargs.segment_weight=$SEGMENT_WEIGHT \
    custom_reward_function.reward_kwargs.use_strict_format=$USE_STRICT_FORMAT \
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
    trainer.default_local_dir="$CKPT_BASE" \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=$NNODES \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.log_val_generations=10 \
    trainer.val_before_train=$VAL_BEFORE_TRAIN \
    trainer.critic_warmup=0 \
    trainer.resume_mode=$RESUME_MODE \
    trainer.logger='["console", "tensorboard"]' \
    +ray_kwargs.ray_init.runtime_env.env_vars.VLLM_USE_V1='"1"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.TENSORBOARD_DIR="$TENSORBOARD_DIR" \
    "$@" 2>&1 | tee -a "$LOG_FILE"

# trainer.resume_from_path=/data_gpu/songlin/rl/verl/checkpoints/video-reasoning-grpo/video_reasoning_grpo_20260131-085501/global_step_200

echo ""
echo "===== Step 2 Complete: Training Finished ====="

# =============================================================================
# Step 3: 自动合并模型 (Merge FSDP checkpoints to HuggingFace format)
# =============================================================================
CKPT_BASE="./checkpoints_zsr/${PROJECT_NAME}/${EXPERIMENT_NAME}"

if [ -f "$CKPT_BASE/latest_checkpointed_iteration.txt" ]; then
    LATEST_STEP=$(cat "$CKPT_BASE/latest_checkpointed_iteration.txt")
    echo ""
    echo "===== Step 3: Merging checkpoint global_step_${LATEST_STEP} ====="
    echo "Source: $CKPT_BASE/global_step_${LATEST_STEP}/actor"
    echo "Target: $CKPT_BASE"

    python -m verl.model_merger merge \
        --backend fsdp \
        --local_dir "$CKPT_BASE/global_step_${LATEST_STEP}/actor" \
        --target_dir "$CKPT_BASE" \
        --trust-remote-code

    if [ $? -eq 0 ]; then
        echo ""
        echo "===== Step 3 Complete: Model Merged Successfully ====="
        echo "Merged model saved to: $CKPT_BASE"
        echo ""
        echo "You can load the model with:"
        echo "  from transformers import AutoModelForVision2Seq, AutoProcessor"
        echo "  model = AutoModelForVision2Seq.from_pretrained('$CKPT_BASE', trust_remote_code=True)"
    else
        echo "ERROR: Failed to merge model"
        exit 1
    fi
else
    echo ""
    echo "WARNING: No checkpoint found at $CKPT_BASE"
    echo "Skipping model merge step."
fi

# =============================================================================
# Step 4: 评测 (Evaluation on head node)
# =============================================================================
# 评测配置
EVAL_WORKDIR="/mnt/data/home/zhengshurong/project/lmms-eval"
EVAL_SCRIPT="${EVAL_WORKDIR}/examples/models/vidvllm_task_parallel_multiturn_zsr.sh"
EVAL_TASKS="video_holmes_multiturn"

# 评测 conda 环境
EVAL_CONDA_BASE="/mnt/data/home/zhengshurong/miniconda3"
EVAL_CONDA_ENV="lmms"

# 是否运行评测（可通过环境变量控制）
RUN_EVAL=${RUN_EVAL:-true}

if [ "$RUN_EVAL" = "true" ] && [ -f "$CKPT_BASE/config.json" ]; then
    echo ""
    echo "===== Step 4: Starting Evaluation ====="

    # 4a: 停止 Ray 集群，释放 GPU 资源
    echo "[$(date)] Stopping Ray cluster to release GPUs..."
    ray stop --force 2>/dev/null || true
    sleep 5

    # 4b: 等待 GPU 进程完全退出
    echo "[$(date)] Waiting for GPU processes to exit..."
    for i in $(seq 1 60); do
        GPU_PROCS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -v "^$" | wc -l)
        if [ "$GPU_PROCS" -eq 0 ]; then
            echo "[$(date)] All GPU processes exited."
            break
        fi
        echo "[$(date)] Still $GPU_PROCS GPU process(es) running, waiting... ($i/60)"
        sleep 5
    done
    sleep 5

    # 4c: 切换 conda 环境
    if [ -n "${EVAL_CONDA_BASE:-}" ] && [ -n "${EVAL_CONDA_ENV:-}" ]; then
        echo "[$(date)] Switching to eval conda environment: ${EVAL_CONDA_ENV}"
        eval "$(${EVAL_CONDA_BASE}/bin/conda shell.bash hook)"
        conda activate ${EVAL_CONDA_ENV}
    fi

    # 4d: 准备评测模型路径和日志
    EVAL_MODEL_PATH=$(realpath "$CKPT_BASE")
    EVAL_LOG="${CKPT_BASE}/logs/eval_${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}.log"
    mkdir -p "$(dirname $EVAL_LOG)"

    echo "[$(date)] Starting evaluation..."
    echo "Model path: ${EVAL_MODEL_PATH}"
    echo "Eval script: ${EVAL_SCRIPT}"
    echo "Eval tasks: ${EVAL_TASKS}"
    echo "Eval log: ${EVAL_LOG}"

    # 4e: 运行评测
    if [ -f "${EVAL_SCRIPT}" ]; then
        pushd "${EVAL_WORKDIR}" > /dev/null
        bash "${EVAL_SCRIPT}" \
            "${EVAL_MODEL_PATH}" \
            "${EVAL_TASKS}" \
            "${EXPERIMENT_NAME}" 2>&1 | tee "${EVAL_LOG}"
        EVAL_EXIT_CODE=${PIPESTATUS[0]}
        popd > /dev/null

        if [ ${EVAL_EXIT_CODE} -ne 0 ]; then
            echo "[$(date)] Evaluation failed with exit code ${EVAL_EXIT_CODE}."
        else
            echo "[$(date)] Evaluation completed successfully."
            echo "Results saved to: ${EVAL_WORKDIR}/logs_zsr/${EXPERIMENT_NAME}/"
        fi
    else
        echo "[$(date)] Eval script not found at ${EVAL_SCRIPT}, skipping evaluation."
    fi

    echo ""
    echo "===== Step 4 Complete: Evaluation Finished ====="
else
    if [ "$RUN_EVAL" != "true" ]; then
        echo ""
        echo "===== Step 4: Evaluation skipped (RUN_EVAL=${RUN_EVAL}) ====="
    elif [ ! -f "$CKPT_BASE/config.json" ]; then
        echo ""
        echo "===== Step 4: Evaluation skipped (no merged model found at $CKPT_BASE) ====="
    fi
fi

echo ""
echo "===== All Steps Complete ====="
echo "Log file: $LOG_FILE"
echo "TensorBoard: tensorboard --logdir=$TENSORBOARD_DIR"
if [ "$RUN_EVAL" = "true" ] && [ -f "$CKPT_BASE/config.json" ]; then
    echo "Eval results: ${EVAL_WORKDIR}/logs_zsr/${EXPERIMENT_NAME}/"
fi
