#!/bin/bash
# =============================================================================
# Video Reasoning SFT Model Sampling & Difficulty Classification
# =============================================================================
# 使用 SFT 模型在 RL 训练数据上进行 n_rollout=8 的采样，
# 计算奖励并根据 accuracy 进行难度分级。
#
# 使用方式：
#   bash examples/video_reasoning/run_sample_and_classify.sh
#   bash examples/video_reasoning/run_sample_and_classify.sh --max_samples 10  # 测试
#
# 配置说明：
#   - 模型路径、数据路径、缓存目录等配置与训练脚本保持一致
#   - 采样参数 (temperature, top_p, n_rollouts) 与训练配置一致
#   - 视频帧缓存参数与训练配置一致
#   - 奖励权重与训练配置一致 (answer=0.0, format=0.5, segment=1.0)
# =============================================================================

set -eo pipefail

# =============================================================================
# Conda 环境
# =============================================================================
CONDA_ENV="${CONDA_ENV:-/share_data/gyzhou/anaconda3/envs/verl_clone}"
export PATH="${CONDA_ENV}/bin:$PATH"
echo "Using Python: $(which python)"

# =============================================================================
# 环境配置 (与训练脚本保持一致)
# =============================================================================
ulimit -n 65535
export VLLM_USE_V1=1
export TMPDIR=/tmp
export RAY_TMPDIR=/tmp/ray_$USER
export TIKTOKEN_RS_CACHE_DIR=/mnt/data/home/zhengshurong/harmony_cache

# NCCL
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export NCCL_IB_HCA=${NCCL_IB_HCA:-^mlx5_bond,mlx5_6}
export NCCL_CROSS_NIC=${NCCL_CROSS_NIC:-1}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-bond0}
export NCCL_SOCKET_FAMILY=${NCCL_SOCKET_FAMILY:-AF_INET}
export TORCH_NCCL_AVOID_RECORD_STREAMS=${TORCH_NCCL_AVOID_RECORD_STREAMS:-1}
export NCCL_CUMEM_ENABLE=${NCCL_CUMEM_ENABLE:-0}

# =============================================================================
# 路径配置 (与训练脚本 run_video_reasoning_dapo_h200_zsr_videor1.sh 一致)
# =============================================================================
MODEL_PATH="/mnt/data/home/zhengshurong/project/Qwen3-VL/qwen-vl-finetune/checkpoints/video/Qwen3-VL-8B-Instruct-longvt_tvg-openo3video_stgr-selfconstructdata-sft-lr1e-5-bs64-ep1"
DATA_DIR="${DATA_DIR:-./long_video_data/longvt_selfqa}"
CACHE_DIR="${CACHE_DIR:-./.cache}"
OUTPUT_DIR="${OUTPUT_DIR:-./sampling_results}"

# =============================================================================
# 采样参数 (与训练配置一致)
# =============================================================================
N_ROLLOUTS=${N_ROLLOUTS:-8}
MAX_ASSISTANT_TURNS=5
MAX_TOKENS_PER_TURN=2048
TEMPERATURE=0.7
TOP_P=0.7
REPETITION_PENALTY=1.1

# =============================================================================
# 视频缓存参数 (与训练配置一致)
# =============================================================================
CACHE_FPS=1
CACHE_MAX_FRAMES=512
MAX_FRAMES_PER_SEGMENT=32
USE_CACHED_INITIAL_VIDEO=True

# 初始视频分辨率
INITIAL_FPS=1
INITIAL_MAX_FRAMES=512
INITIAL_MIN_PIXELS=784
INITIAL_MAX_PIXELS=12544

# Segment 视频分辨率
SEGMENT_FPS=1
SEGMENT_MAX_FRAMES=32
SEGMENT_MIN_PIXELS=784
SEGMENT_MAX_PIXELS=50176

# =============================================================================
# 奖励权重 (与训练配置一致)
# =============================================================================
ANSWER_WEIGHT=0.0
FORMAT_WEIGHT=0.5
SEGMENT_WEIGHT=1.0
USE_STRICT_FORMAT=True

# =============================================================================
# vLLM 配置
# =============================================================================
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.85}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-128000}
CHUNK_SIZE=${CHUNK_SIZE:-64}

# =============================================================================
# 预检查
# =============================================================================
echo "===== Pre-flight Checks ====="

if [ ! -f "$DATA_DIR/train.parquet" ]; then
    echo "ERROR: Training data not found at $DATA_DIR/train.parquet"
    exit 1
fi

echo "Model:    $MODEL_PATH"
echo "Data:     $DATA_DIR/train.parquet"
echo "Cache:    $CACHE_DIR"
echo "Output:   $OUTPUT_DIR"
echo "Rollouts: $N_ROLLOUTS"
echo "TP size:  $TENSOR_PARALLEL_SIZE"
echo ""

# =============================================================================
# Step 1: 确保视频缓存存在
# =============================================================================
if [ "${SKIP_VIDEO_CACHE:-false}" != "true" ]; then
    echo "===== Step 1: Checking/Caching video frames ====="
    python examples/video_reasoning/cache_video_frames.py \
        --input_parquet "$DATA_DIR/train.parquet" \
        --cache_dir "$CACHE_DIR" \
        --fps "$CACHE_FPS" \
        --max_frames "$CACHE_MAX_FRAMES" \
        --num_workers "${CACHE_NUM_WORKERS:-64}"
    echo "===== Step 1 Complete ====="
    echo ""
else
    echo "===== Step 1: Skipping cache (SKIP_VIDEO_CACHE=true) ====="
    echo ""
fi

# =============================================================================
# Step 2: 运行采样与难度分级
# =============================================================================
echo "===== Step 2: Sampling & Difficulty Classification ====="

python examples/video_reasoning/sample_and_classify.py \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_DIR/train.parquet" \
    --cache_dir "$CACHE_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --n_rollouts "$N_ROLLOUTS" \
    --max_assistant_turns "$MAX_ASSISTANT_TURNS" \
    --max_tokens_per_turn "$MAX_TOKENS_PER_TURN" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --repetition_penalty "$REPETITION_PENALTY" \
    --cache_fps "$CACHE_FPS" \
    --cache_max_frames "$CACHE_MAX_FRAMES" \
    --max_frames_per_segment "$MAX_FRAMES_PER_SEGMENT" \
    --initial_fps "$INITIAL_FPS" \
    --initial_max_frames "$INITIAL_MAX_FRAMES" \
    --initial_min_pixels "$INITIAL_MIN_PIXELS" \
    --initial_max_pixels "$INITIAL_MAX_PIXELS" \
    --segment_fps "$SEGMENT_FPS" \
    --segment_max_frames "$SEGMENT_MAX_FRAMES" \
    --segment_min_pixels "$SEGMENT_MIN_PIXELS" \
    --segment_max_pixels "$SEGMENT_MAX_PIXELS" \
    --answer_weight "$ANSWER_WEIGHT" \
    --format_weight "$FORMAT_WEIGHT" \
    --segment_weight "$SEGMENT_WEIGHT" \
    --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
    --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
    --max_model_len "$MAX_MODEL_LEN" \
    --chunk_size "$CHUNK_SIZE" \
    "$@"

echo ""
echo "===== Step 2 Complete ====="
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo "  - sampling_results.jsonl      : 所有样本的详细采样结果"
echo "  - difficulty_summary.json     : 难度分级统计总结"
echo "  - difficulty_easy.json        : 简单样本列表"
echo "  - difficulty_medium_easy.json : 中等偏简单样本列表"
echo "  - difficulty_medium.json      : 中等样本列表"
echo "  - difficulty_medium_hard.json : 中等偏难样本列表"
echo "  - difficulty_hard.json        : 困难样本列表"
