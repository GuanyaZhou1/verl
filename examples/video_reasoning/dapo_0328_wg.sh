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
export HYDRA_FULL_ERROR=1  # 报错时打印完整 stack trace

# =============================================================================
# 环境配置
# =============================================================================
ulimit -n 65535
export VLLM_USE_V1=1
#禁用 Python 断点，只在当前有效
export RAY_DEBUG=0                                                                       
export PYDEVD_DISABLE_FILE_VALIDATION=1  
# export NCCL_DEBUG=INFO

# export TIKTOKEN_CACHE_DIR=${TIKTOKEN_CACHE_DIR:-/data_gpu/gyzhou/tmp/tiktoken_cache}
export LD_LIBRARY_PATH="/usr/local/cuda-13.1/compat:${LD_LIBRARY_PATH:-}"
export TMPDIR=/tmp  # multiprocessing 临时文件放本地磁盘，避免 NFS 上 EBUSY 错误
# 多机 launcher 已经提前启动好 Ray 集群时，不再覆盖其 temp dir，避免 driver/worker 侧路径不一致。
if [ -z "${RAY_ADDRESS:-}" ]; then
    export RAY_TEMPDIR=/tmp/ray_$USER
else
    unset RAY_TEMPDIR
fi
# export RAY_ADDRESS=local
# export CUDA_VISIBLE_DEVICES=0,1,2,3
export TIKTOKEN_RS_CACHE_DIR=/data_gpu/gyzhou/harmony_cache
# =============================================================================
# NCCL 环境变量（IB 优先，单机多机通用）
# =============================================================================
# 使用 ${VAR:-default} 语法，允许 launcher 脚本覆盖
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}                        # 隐藏 NCCL INFO 日志，只显示 WARN 及以上
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
#MODEL_PATH="/data_gpu/songlin/rl/verl/checkpoints/video-reasoning-dapo/video_reasoning_dapo_20260205-063449/merged_model"
#MODEL_PATH="/data_gpu/zhengshurong/data/project/Qwen2.5-VL/qwen-vl-finetune/checkpoints/video/Qwen2.5-VL-7B-Instruct-self_holmes_caption_233-self_longvideoreason_caption_930-openo3video_stgr_singleturn_7k-self_holmes_multiturn_1k5-self_longvideoreason_multiturn_5k3-sft-lr5e-5-b24"
# MODEL_PATH="${MODEL_PATH:-/data_gpu/zhengshurong/data/project/Qwen2.5-VL/qwen-vl-finetune/checkpoints/video/Qwen2.5-VL-7B-Instruct-stgr-turn_llm_freeze25_freeze_mlp-lr1e-5-epo5}"
MODEL_PATH="/data_gpu/zhengshurong/data/project/Qwen3-VL/qwen-vl-finetune/checkpoints/video/ablate_reasoning_format/Qwen3-VL-8B-Instruct-openo3video_stgr_singleturn_7k-self_holmes_multiturn_1k5-self_longvideoreason_multiturn_5k3_abscoord-sft-lr5e-5-bs32"
DATA_DIR="${DATA_DIR:-/data_gpu/gyzhou/prj/verl/long_video_data/video_holmes}"
CACHE_DIR="${CACHE_DIR:-./.cache_new}"
CONFIG_PATH="$(pwd)/examples/video_reasoning/config"
LOG_DIR="./logs"
# =============================================================================
# 训练参数
# =============================================================================
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
GEN_BATCH_SIZE=${GEN_BATCH_SIZE:-32}                        # DAPO: 生成批次
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-24000}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-16384}

LEARNING_RATE=${LEARNING_RATE:-1e-6}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-3}

WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-5.0}

N_ROLLOUTS=${N_ROLLOUTS:-8}                             # 每个 prompt 生成的 response 数
AGENT_NUM_WORKERS=${AGENT_NUM_WORKERS:-1}                      # AgentLoopWorker 数量

N_GPUS=${N_GPUS:-8}
NNODES=${NNODES:-1}

# =============================================================================
# DAPO 算法参数（支持环境变量覆盖，便于自动化实验）
# =============================================================================
ENABLE_FILTER_GROUPS=${ENABLE_FILTER_GROUPS:-true}   # 过滤组内全对/全错的样本
FILTER_GROUPS_METRIC=${FILTER_GROUPS_METRIC:-acc}   # 用总分做组过滤
FILTER_GROUPS_STD_THRESHOLD=${FILTER_GROUPS_STD_THRESHOLD:-0.0}  # 最小std阈值
MAX_NUM_GEN_BATCHES=${MAX_NUM_GEN_BATCHES:-5}         # 最多重采样轮数

CLIP_RATIO_LOW=${CLIP_RATIO_LOW:-0.2}                 # Clip-Higher: 非对称 clip ratio
CLIP_RATIO_HIGH=${CLIP_RATIO_HIGH:-0.28}              # > low，鼓励正向更新

NORM_ADV_BY_STD=${NORM_ADV_BY_STD:-false}             # 归一化 advantage

USE_KL_IN_REWARD=true
USE_KL_LOSS=false
KL_LOSS_COEF=${KL_LOSS_COEF:-0.01}                     # KL 约束系数
KL_LOSS_TYPE=low_var_kl

BY_PASS_ROLLOUT_CORRECTION=false
ENTROPY_COEFF=${ENTROPY_COEFF:-0}                   # Entropy 系数
TOP_P=${TOP_P:-1.0}                                   # Top-p 采样
TEMPERATURE=${TEMPERATURE:-1.0}                       # 采样温度

# =============================================================================
# Corrected Rollout SFT 配置（混合 RL + SFT Loss）
# =============================================================================
# 用 VLM 验证得到的 GT bbox 替换模型预测的不准确 bbox，对替换后的样本计算 SFT loss
# 仅对 answer_score=1 且有 bbox_details 的样本生效（coupled 设计）
# 注意：启用 SFT loss 需要关闭 use_remove_padding 和 use_fused_kernels
SFT_LOSS_ENABLED=${SFT_LOSS_ENABLED:-false}    # 是否启用 Corrected Rollout SFT
SFT_LOSS_WEIGHT=${SFT_LOSS_WEIGHT:-0.3}        # SFT loss 权重 (推荐 0.1-0.5)
MAX_SFT_SAMPLES=${MAX_SFT_SAMPLES:-32}         # 每步最多 SFT 样本数，从不同 prompt 均匀采样

# =============================================================================
# GDPO & Token Placement 配置（多轮 CoT 细粒度奖励分配）
# =============================================================================
# Advantage 估计器选择:
#   - grpo: 原始 GRPO
#   - gdpo: Group reward-Decoupled normalization Policy Optimization
#           每个 reward component 独立归一化，避免 advantage collapse
# 使用 bbox/segment 的 turn 级别 group norm 时，必须设为 gdpo。
ADV_ESTIMATOR=${ADV_ESTIMATOR:-gdpo}

# =============================================================================
# 统一的 Reward 权重配置（同时用于 grpo 和 gdpo）
# =============================================================================
# 设为 0 可排除该 component。grpo 和 gdpo 模式都使用这套权重。
REWARD_WEIGHT_ANSWER=${REWARD_WEIGHT_ANSWER:-1.0}     # 答案正确性权重
REWARD_WEIGHT_FORMAT=${REWARD_WEIGHT_FORMAT:-0.2}     # 格式正确性权重
REWARD_WEIGHT_BBOX=${REWARD_WEIGHT_BBOX:-0.5}         # BBox 验证权重
REWARD_WEIGHT_SEGMENT=${REWARD_WEIGHT_SEGMENT:-0.5}   # Segment 定位权重

# GDPO batch norm（仅 adv_estimator=gdpo 时生效）
GDPO_ENABLE_BATCH_NORM=${GDPO_ENABLE_BATCH_NORM:-true} 

# Token Placement 方法（控制奖励如何分配到 token，仅 adv_estimator=gdpo 时生效）
# bbox/segment 的 turn 级别 group norm 由 gdpo 计算，在此通过 method 决定如何落到 token：
#   - broadcast: 所有 token 共享同一 advantage，不区分轮；bbox/segment 的 turn 级信号未用到
#   - per_turn: 轮内广播，bbox/segment 使用 turn 级别 group norm 后的 advantage 分配到对应轮
#   - per_turn_gae: 轮内 GAE 衰减传播，同样使用 turn 级别 group norm 的 bbox/segment
# 启用 bbox+segment turn 级 group norm 时，请设为 per_turn 或 per_turn_gae。
TOKEN_PLACEMENT_METHOD=${TOKEN_PLACEMENT_METHOD:-per_turn}

# 全局奖励（acc, format）的传播方式
#   - broadcast: 平均分配到所有 token（推荐，稳定）
#   - gae: 从 EOS 向前 GAE 衰减传播（可能导致前面 token 信号过弱）
TOKEN_PLACEMENT_GLOBAL_MODE=${TOKEN_PLACEMENT_GLOBAL_MODE:-broadcast}

# 最终 batch norm（推荐开启以稳定训练）
TP_ENABLE_BATCH_NORM=${TP_ENABLE_BATCH_NORM:-true}


# GAE 参数（per_turn_gae 或 global_mode=gae 时使用）
TP_GAMMA=${TP_GAMMA:-0.99}                            # 衰减因子
TP_LAMBDA=${TP_LAMBDA:-0.95}                          # GAE lambda
SINGLE_TURN_NO_OBSERVATION=${SINGLE_TURN_NO_OBSERVATION:-false}  # 单轮消融: 不注入 observation

# =============================================================================
# 视频缓存参数
# =============================================================================
CACHE_FPS=1
CACHE_MAX_FRAMES=0  # 0表示不限制，全部缓存（每秒1帧）
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
USE_TIMESTAMP_WATERMARK=False            # 是否启用时间戳水印
WATERMARK_POSITION="top_left"            # 水印位置: top_left, top_right, bottom_left, bottom_right
WATERMARK_FONT_SIZE=0                    # 字体大小 (0=根据图片高度自适应)
WATERMARK_RATIO=1.0                     # 水印采样比例: 1.0=全部使用, 0.0=全部不使用, 0.5=50%采样

# =============================================================================
# 奖励函数参数
# =============================================================================
VLM_ENDPOINT="10.96.11.3:8081"
# 必须与 vLLM 服务 /v1/models 返回的 id 一致（用完整路径，否则会 404）
VLM_MODEL_NAME="Qwen3-VL-30B-A3B-Instruct"
VLM_API_KEY="123456"

USE_VLM_SCORING=${USE_VLM_SCORING:-false}   # 关闭 VLM 打分: 设 false，改用规则化 EM+ROUGE
USE_BBOX_VERIFICATION=true

# BBox 评分指标选择
BBOX_METRIC=${BBOX_METRIC:-iou}          # "iou" 原始指标, "adaptive_iou" 小目标宽松 (别名 "nwd")
TEMPORAL_TOLERANCE=${TEMPORAL_TOLERANCE:-1}  # 相邻帧容忍度 (0=禁用, 1=±1帧)，对应 Qwen3-VL temporal_patch_size=2
BBOX_PER_TURN=${BBOX_PER_TURN:-2}  # 每个 think turn 期望输出的 bbox 数量

# Accuracy Gate: 答案分数低于此阈值时，bbox/segment reward 清零
# 0   = 关闭 gate（始终给 grounding reward）
# 0.5 = 开放题 ROUGE>=0.5 或选择题答对才给 grounding reward（推荐）
# 1.0 = 只有完全答对才给（最严格，仅适合纯选择题）
ACCURACY_GATE_THRESHOLD=${ACCURACY_GATE_THRESHOLD:-0.5}

SAVE_BBOX_VISUALIZATION=false
BBOX_VIS_SAMPLE_RATE=0.001
# REWARD_LOG_DIR 在 EXPERIMENT_NAME 后设置
SAVE_SAMPLES=true
SAVE_EVERY_N=1
LOG_EVERY_N=10

# =============================================================================
# Checkpoint 配置
# =============================================================================
SAVE_FREQ=30
TEST_FREQ=20
VAL_BEFORE_TRAIN=True
RESUME_MODE=disable                      # disable / resume_path / auto

# =============================================================================
# 实验名称（支持环境变量传入，便于自动化实验）
# =============================================================================
TIMESTAMP=$(date '+%Y%m%d-%H%M%S')
PROJECT_NAME="${PROJECT_NAME:-video-reasoning-dapo}"
BBOX_COORD_RANGE=1.0
# 如果未设置 EXPERIMENT_NAME，则根据当前参数自动生成
if [ -z "$EXPERIMENT_NAME" ]; then
    # 自动生成实验名：包含关键超参数
    EXPERIMENT_NAME="Qwen3_8B_${ADV_ESTIMATOR}_acc${REWARD_WEIGHT_ANSWER}_fmt${REWARD_WEIGHT_FORMAT}_bbox${REWARD_WEIGHT_BBOX}_seg${REWARD_WEIGHT_SEGMENT}_kl${KL_LOSS_COEF}_lr${LEARNING_RATE}_$(date +%m%d)_bs${TRAIN_BATCH_SIZE}_GATE${ACCURACY_GATE_THRESHOLD}_warmup10_gspo_minibs8"
fi
# 示例手动设置：
# export EXPERIMENT_NAME="my_custom_exp_name"
# bash run_video_reasoning_dapo_h200.sh

# 将 reward_logs 和 tensorboard_log 放到 checkpoint 目录下
CKPT_BASE="./checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}"
REWARD_LOG_DIR="${CKPT_BASE}/reward_logs"
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
echo "GPUs:            $N_GPUS x $NNOD[ES nodes"
echo ""
echo "DAPO Settings:"
echo "  filter_groups: $ENABLE_FILTER_GROUPS (metric=$FILTER_GROUPS_METRIC, std_threshold=$FILTER_GROUPS_STD_THRESHOLD)"
echo "  clip_ratio:    [$CLIP_RATIO_LOW, $CLIP_RATIO_HIGH]"
echo "  norm_by_std:   $NORM_ADV_BY_STD"
echo ""
echo "GDPO/Token Placement:"
echo "  adv_estimator: $ADV_ESTIMATOR"
echo "  reward_weights: ans=$REWARD_WEIGHT_ANSWER, fmt=$REWARD_WEIGHT_FORMAT, bbox=$REWARD_WEIGHT_BBOX, seg=$REWARD_WEIGHT_SEGMENT"
if [ "$ADV_ESTIMATOR" = "gdpo" ]; then
    echo "  batch_norm:    $GDPO_ENABLE_BATCH_NORM"
    echo "  token_placement:"
    echo "    method:      $TOKEN_PLACEMENT_METHOD"
    echo "    global_mode: $TOKEN_PLACEMENT_GLOBAL_MODE"
    echo "    single_turn_no_observation: $SINGLE_TURN_NO_OBSERVATION"
    if [ "$TOKEN_PLACEMENT_METHOD" != "broadcast" ]; then
        echo "    gae:         gamma=$TP_GAMMA, lambda=$TP_LAMBDA"
    fi
fi
echo ""
echo "Reward:"
echo "  VLM scoring:   $USE_VLM_SCORING ($VLM_ENDPOINT)"
echo "  BBox verify:   $USE_BBOX_VERIFICATION"
echo "  BBox metric:   $BBOX_METRIC (temporal_tolerance=$TEMPORAL_TOLERANCE, bbox_per_turn=$BBOX_PER_TURN)"
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
        --num_workers 64

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
mkdir -p "$TENSORBOARD_DIR"
LOG_FILE="$LOG_DIR/${EXPERIMENT_NAME}.log"
echo "Log file: $LOG_FILE"
echo "Reward logs: $REWARD_LOG_DIR"
echo "TensorBoard: $TENSORBOARD_DIR"
#    ++actor_rollout_ref.actor.optim.lr_scheduler_type=cosine \
# 设置 TensorBoard 目录环境变量
export TENSORBOARD_DIR="$TENSORBOARD_DIR"

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
    actor_rollout_ref.actor.optim.weight_decay=$WEIGHT_DECAY \
    ++actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    ++actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
    actor_rollout_ref.actor.grad_clip=$MAX_GRAD_NORM \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.clip_ratio_low=$CLIP_RATIO_LOW \
    actor_rollout_ref.actor.clip_ratio_high=$CLIP_RATIO_HIGH \
    actor_rollout_ref.actor.use_kl_loss=$USE_KL_LOSS \
    actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
    actor_rollout_ref.actor.kl_loss_type=$KL_LOSS_TYPE \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFF \
    +actor_rollout_ref.actor.sft_loss_enabled=$SFT_LOSS_ENABLED \
    +actor_rollout_ref.actor.sft_loss_weight=$SFT_LOSS_WEIGHT \
    +actor_rollout_ref.actor.max_sft_samples=$MAX_SFT_SAMPLES \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.top_p=$TOP_P \
    actor_rollout_ref.rollout.temperature=$TEMPERATURE \
    +actor_rollout_ref.rollout.enable_sleep_mode=False \
    +actor_rollout_ref.rollout.max_tokens_per_turn=4096 \
    actor_rollout_ref.rollout.n=$N_ROLLOUTS \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.max_model_len=128000 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=65536 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
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
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=104768 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.adv_estimator=$ADV_ESTIMATOR \
    algorithm.rollout_correction.bypass_mode=$BY_PASS_ROLLOUT_CORRECTION \
    algorithm.norm_adv_by_std_in_grpo=$NORM_ADV_BY_STD \
    algorithm.use_kl_in_reward=$USE_KL_IN_REWARD \
    algorithm.filter_groups.enable=$ENABLE_FILTER_GROUPS \
    algorithm.filter_groups.metric=$FILTER_GROUPS_METRIC \
    +algorithm.filter_groups.std_threshold=$FILTER_GROUPS_STD_THRESHOLD \
    algorithm.filter_groups.max_num_gen_batches=$MAX_NUM_GEN_BATCHES \
    algorithm.gdpo.enable_batch_norm=$GDPO_ENABLE_BATCH_NORM \
    algorithm.reward_weights.answer_score=$REWARD_WEIGHT_ANSWER \
    algorithm.reward_weights.format_score=$REWARD_WEIGHT_FORMAT \
    algorithm.reward_weights.bbox_score=$REWARD_WEIGHT_BBOX \
    algorithm.reward_weights.segment_score=$REWARD_WEIGHT_SEGMENT \
    algorithm.token_placement.method=$TOKEN_PLACEMENT_METHOD \
    algorithm.token_placement.global_reward_mode=$TOKEN_PLACEMENT_GLOBAL_MODE \
    algorithm.token_placement.enable_batch_norm=$TP_ENABLE_BATCH_NORM \
    algorithm.token_placement.gamma=$TP_GAMMA \
    algorithm.token_placement.lambda=$TP_LAMBDA \
    reward_model.enable=False \
    reward_model.reward_manager=dapo \
    reward_model.overlong_buffer.enable=False \
    reward_model.overlong_buffer.len=0 \
    reward_model.overlong_buffer.penalty_factor=0.0 \
    custom_reward_function.path=pkg://verl.utils.reward_score.video_reasoning_async \
    custom_reward_function.name=compute_score \
    custom_reward_function.reward_kwargs.vlm_endpoint="$VLM_ENDPOINT" \
    custom_reward_function.reward_kwargs.vlm_model_name="$VLM_MODEL_NAME" \
    custom_reward_function.reward_kwargs.vlm_api_key="$VLM_API_KEY" \
    custom_reward_function.reward_kwargs.use_vlm_scoring=$USE_VLM_SCORING \
    custom_reward_function.reward_kwargs.use_bbox_verification=$USE_BBOX_VERIFICATION \
    custom_reward_function.reward_kwargs.answer_weight=$REWARD_WEIGHT_ANSWER \
    custom_reward_function.reward_kwargs.format_weight=$REWARD_WEIGHT_FORMAT \
    custom_reward_function.reward_kwargs.bbox_weight=$REWARD_WEIGHT_BBOX \
    custom_reward_function.reward_kwargs.segment_weight=$REWARD_WEIGHT_SEGMENT \
    custom_reward_function.reward_kwargs.bbox_coord_range=$BBOX_COORD_RANGE \
    +custom_reward_function.reward_kwargs.bbox_metric=$BBOX_METRIC \
    +custom_reward_function.reward_kwargs.temporal_tolerance=$TEMPORAL_TOLERANCE \
    +custom_reward_function.reward_kwargs.bbox_per_turn=$BBOX_PER_TURN \
    +custom_reward_function.reward_kwargs.accuracy_gate_threshold=$ACCURACY_GATE_THRESHOLD \
    +custom_reward_function.reward_kwargs.allow_missing_observation_between_turns=$SINGLE_TURN_NO_OBSERVATION \
    custom_reward_function.reward_kwargs.cache_dir="$CACHE_DIR" \
    custom_reward_function.reward_kwargs.cache_fps=$CACHE_FPS \
    custom_reward_function.reward_kwargs.cache_max_frames=$CACHE_MAX_FRAMES \
    custom_reward_function.reward_kwargs.save_bbox_visualization=$SAVE_BBOX_VISUALIZATION \
    custom_reward_function.reward_kwargs.bbox_vis_sample_rate=$BBOX_VIS_SAMPLE_RATE \
    custom_reward_function.reward_kwargs.enable_logging=True \
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
    +ray_kwargs.ray_init.runtime_env.env_vars.TENSORBOARD_DIR="$TENSORBOARD_DIR" \
    "$@" 2>&1 | tee -a "$LOG_FILE"
#     +actor_rollout_ref.rollout.repetition_penalty=1.1 \
# trainer.resume_from_path=/data_gpu/songlin/rl/verl/checkpoints/video-reasoning-grpo/video_reasoning_grpo_20260131-085501/global_step_200
#    actor_rollout_ref.actor.ppo_epochs=2 \

echo ""
echo "===== Step 2 Complete: Training Finished ====="
#     actor_rollout_ref.actor.ulysses_sequence_parallel_size=4 \
#     actor_rollout_ref.ref.ulysses_sequence_parallel_size=4 \
# =============================================================================
# Step 3: 自动合并模型 (Merge FSDP checkpoints to HuggingFace format)
# =============================================================================
CKPT_BASE="./checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}"

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
# Step 4: 评测 (Holmes + VideoMME, task-parallel)
# =============================================================================
# 评测脚本：vidvllm_qwen3_holmes_videomme.sh
#   接受参数: $1=model_path  $2=experiment_name
#   自动在 8 卡上 task-parallel: Holmes(GPU 0-3) + VideoMME(GPU 4-7)
EVAL_WORKDIR="/data_gpu/gyzhou/prj/lmms-eval"
EVAL_SCRIPT="${EVAL_WORKDIR}/examples/models/vidvllm_qwen3_holmes_videomme.sh"

# 评测 conda 环境（eval 脚本需要 lmms-eval 环境）
EVAL_CONDA_BASE="/share_data/gyzhou/anaconda3"
EVAL_CONDA_ENV="al"

# 是否运行评测（可通过环境变量控制）
RUN_EVAL=${RUN_EVAL:-true}

if [ "$RUN_EVAL" = "true" ] && [ -f "$CKPT_BASE/config.json" ]; then
    echo ""
    echo "===== Step 4: Starting Evaluation (Holmes + VideoMME) ====="

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
    echo "Eval log: ${EVAL_LOG}"

    # 4e: 运行评测
    if [ -f "${EVAL_SCRIPT}" ]; then
        pushd "${EVAL_WORKDIR}" > /dev/null
        bash "${EVAL_SCRIPT}" \
            "${EVAL_MODEL_PATH}" \
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
