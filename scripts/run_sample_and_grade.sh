#!/bin/bash
# =============================================================================
# SFT 模型采样与难度分级运行脚本 (支持多轮推理 + 多GPU并行)
# =============================================================================
# 使用方式：
#   # 方式1: 单GPU启动 vLLM 服务
#   bash scripts/run_sample_and_grade.sh --start_server
#
#   # 方式2: 多GPU并行 (每张卡启动一个vLLM服务)
#   bash scripts/run_sample_and_grade.sh --start_server --num_gpus 4
#
#   # 方式3: 使用已有的 vLLM 服务
#   bash scripts/run_sample_and_grade.sh --api_base localhost:8000
#
#   # 方式4: 使用多个已有的 vLLM 服务 (逗号分隔)
#   bash scripts/run_sample_and_grade.sh --api_base "localhost:8000,localhost:8001,localhost:8002"
#
#   # 方式5: 从中断处继续
#   bash scripts/run_sample_and_grade.sh --api_base localhost:8000 --resume
# =============================================================================

set -eo pipefail

# =============================================================================
# 配置 (从 run_video_reasoning_dapo_h200_zsr_videor1.sh 继承)
# =============================================================================

# 模型路径 (SFT 模型)
MODEL_PATH="${MODEL_PATH:-/mnt/data/home/zhengshurong/project/Qwen3-VL/qwen-vl-finetune/checkpoints/video/Qwen3-VL-8B-Instruct-longvt_tvg-openo3video_stgr-selfconstructdata-sft-lr1e-5-bs64-ep1}"

# 数据路径
DATA_DIR="${DATA_DIR:-./long_video_data/longvt_selfqa}"
DATA_PATH="${DATA_DIR}/train.parquet"

# 输出目录
OUTPUT_DIR="${OUTPUT_DIR:-./difficulty_analysis/longvt_selfqa}"

# 采样参数
N_ROLLOUTS=${N_ROLLOUTS:-8}
MAX_TURNS=${MAX_TURNS:-5}  # 多轮推理最大轮数
MAX_TOKENS=${MAX_TOKENS:-16384}
TEMPERATURE=${TEMPERATURE:-0.7}
TOP_P=${TOP_P:-0.7}
BATCH_SIZE=${BATCH_SIZE:-1}
NUM_SAMPLES=${NUM_SAMPLES:-}  # 空表示全部

# vLLM 服务参数
VLLM_PORT_BASE=${VLLM_PORT_BASE:-8000}  # 多GPU时的起始端口
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.85}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-65536}

# 多GPU并行参数
NUM_GPUS=${NUM_GPUS:-8}  # 使用的GPU数量

# 视频缓存参数
CACHE_DIR="${CACHE_DIR:-./.cache}"
CACHE_FPS=${CACHE_FPS:-1}
CACHE_MAX_FRAMES=${CACHE_MAX_FRAMES:-512}
MAX_FRAMES_PER_SEGMENT=${MAX_FRAMES_PER_SEGMENT:-32}

# 视频参数配置 (与 video_reasoning_agent_loop.py 一致)
# Initial video config (for first frame load)
INITIAL_FPS=${INITIAL_FPS:-1}
INITIAL_MAX_FRAMES=${INITIAL_MAX_FRAMES:-512}
INITIAL_MIN_PIXELS=${INITIAL_MIN_PIXELS:-784}      # 28*28
INITIAL_MAX_PIXELS=${INITIAL_MAX_PIXELS:-12544}    # ~112x112
# Segment video config (for segment frames)
SEGMENT_FPS=${SEGMENT_FPS:-1}
SEGMENT_MAX_FRAMES=${SEGMENT_MAX_FRAMES:-32}
SEGMENT_MIN_PIXELS=${SEGMENT_MIN_PIXELS:-784}      # 28*28
SEGMENT_MAX_PIXELS=${SEGMENT_MAX_PIXELS:-50176}    # ~224x224

# VLM 奖励评分服务
VLM_ENDPOINT="${VLM_ENDPOINT:-10.0.1.35:8081}"
VLM_MODEL_NAME="${VLM_MODEL_NAME:-Qwen3-VL-235B-A22B-Instruct}"
VLM_API_KEY="${VLM_API_KEY:-123456}"

# 奖励权重 (与训练脚本一致)
ANSWER_WEIGHT=${ANSWER_WEIGHT:-1.0}
BBOX_WEIGHT=${BBOX_WEIGHT:-0.5}
FORMAT_WEIGHT=${FORMAT_WEIGHT:-0.5}
SEGMENT_WEIGHT=${SEGMENT_WEIGHT:-1.0}
USE_STRICT_FORMAT=${USE_STRICT_FORMAT:-true}

# =============================================================================
# 解析命令行参数
# =============================================================================
START_SERVER=false
API_BASE=""
RESUME=false
USE_OFFLINE=false  # 使用离线 vLLM 模式
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --start_server)
            START_SERVER=true
            shift
            ;;
        --api_base)
            API_BASE="$2"
            shift 2
            ;;
        --num_samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --max_turns)
            MAX_TURNS="$2"
            shift 2
            ;;
        --resume)
            RESUME=true
            shift
            ;;
        --num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --offline)
            USE_OFFLINE=true
            shift
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# =============================================================================
# 环境配置
# =============================================================================
export VLLM_USE_V1=1
export TMPDIR=/tmp
export TIKTOKEN_RS_CACHE_DIR=/mnt/data/home/zhengshurong/harmony_cache

# 离线模式需要禁用 V1 多进程，避免 CUDA fork 问题
if [ "$USE_OFFLINE" = true ]; then
    export VLLM_ENABLE_V1_MULTIPROCESSING=0
fi

# =============================================================================
# 预检查
# =============================================================================
echo "===== Pre-flight Checks ====="

if [ ! -f "$DATA_PATH" ]; then
    echo "ERROR: Data file not found at $DATA_PATH"
    exit 1
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    exit 1
fi

echo "Model: $MODEL_PATH"
echo "Data: $DATA_PATH"
echo "Output: $OUTPUT_DIR"
echo "N_ROLLOUTS: $N_ROLLOUTS"
echo "MAX_TURNS: $MAX_TURNS"
echo "NUM_GPUS: $NUM_GPUS"
echo "VLM Endpoint: $VLM_ENDPOINT"
echo ""

# =============================================================================
# Step 1: 缓存视频帧 (如果需要)
# =============================================================================
if [ "${SKIP_VIDEO_CACHE:-false}" != "true" ]; then
    echo "===== Step 1: Caching video frames ====="
    CACHE_NUM_WORKERS=${CACHE_NUM_WORKERS:-64}

    python examples/video_reasoning/cache_video_frames.py \
        --input_parquet "$DATA_PATH" \
        --cache_dir "$CACHE_DIR" \
        --fps "$CACHE_FPS" \
        --max_frames "$CACHE_MAX_FRAMES" \
        --num_workers "$CACHE_NUM_WORKERS"

    echo "===== Step 1 Complete ====="
    echo ""
fi

# =============================================================================
# Step 2: 启动 vLLM 服务 (如果需要)
# =============================================================================
VLLM_PIDS=()

# 获取模型名称 (从路径中提取，用于 API 调用)
MODEL_NAME=$(basename "$MODEL_PATH")

if [ "$START_SERVER" = true ]; then
    echo "===== Step 2: Starting vLLM server(s) ====="
    mkdir -p "$OUTPUT_DIR"

    if [ "$NUM_GPUS" -gt 1 ]; then
        # 多GPU模式：每张卡启动一个vLLM服务
        echo "Starting $NUM_GPUS vLLM servers (one per GPU)..."
        API_ENDPOINTS=""

        for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
            port=$((VLLM_PORT_BASE + gpu_id))
            log_file="$OUTPUT_DIR/vllm_server_gpu${gpu_id}.log"

            echo "Starting vLLM server on GPU $gpu_id, port $port..."
            CUDA_VISIBLE_DEVICES=$gpu_id python -m vllm.entrypoints.openai.api_server \
                --model "$MODEL_PATH" \
                --served-model-name "$MODEL_NAME" \
                --port $port \
                --tensor-parallel-size 1 \
                --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
                --max-model-len $MAX_MODEL_LEN \
                --trust-remote-code \
                --dtype auto \
                --generation-config vllm \
                --limit-mm-per-prompt '{"image": 1000, "video": 10}' \
                > "$log_file" 2>&1 &

            VLLM_PIDS+=($!)

            if [ -z "$API_ENDPOINTS" ]; then
                API_ENDPOINTS="localhost:$port"
            else
                API_ENDPOINTS="$API_ENDPOINTS,localhost:$port"
            fi
        done

        API_BASE="$API_ENDPOINTS"
        echo "vLLM servers started with PIDs: ${VLLM_PIDS[*]}"
        echo "API endpoints: $API_BASE"

        # 等待所有服务启动
        echo "Waiting for all servers to be ready..."
        for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
            port=$((VLLM_PORT_BASE + gpu_id))
            for i in $(seq 1 360); do
                if curl -s "http://localhost:$port/health" > /dev/null 2>&1; then
                    echo "Server on port $port is ready!"
                    break
                fi
                if [ $i -eq 360 ]; then
                    echo "ERROR: vLLM server on port $port failed to start"
                    # 清理已启动的服务
                    for pid in "${VLLM_PIDS[@]}"; do
                        kill $pid 2>/dev/null || true
                    done
                    exit 1
                fi
                sleep 1
            done
        done

    else
        # 单GPU模式
        VLLM_LOG_FILE="$OUTPUT_DIR/vllm_server.log"
        python -m vllm.entrypoints.openai.api_server \
            --model "$MODEL_PATH" \
            --served-model-name "$MODEL_NAME" \
            --port $VLLM_PORT_BASE \
            --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
            --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
            --max-model-len $MAX_MODEL_LEN \
            --trust-remote-code \
            --dtype auto \
            --generation-config vllm \
            --limit-mm-per-prompt '{"image": 1000, "video": 10}' \
            > "$VLLM_LOG_FILE" 2>&1 &

        VLLM_PIDS+=($!)
        API_BASE="localhost:$VLLM_PORT_BASE"

        echo "vLLM server started with PID: ${VLLM_PIDS[0]}"
        echo "Log file: $VLLM_LOG_FILE"
        echo "Waiting for server to be ready..."

        for i in $(seq 1 360); do
            if curl -s "http://$API_BASE/health" > /dev/null 2>&1; then
                echo "vLLM server is ready!"
                break
            fi
            if [ $i -eq 360 ]; then
                echo "ERROR: vLLM server failed to start within 360 seconds"
                kill ${VLLM_PIDS[0]} 2>/dev/null || true
                exit 1
            fi
            echo "Waiting... ($i/360)"
            sleep 1
        done
    fi

    echo "===== Step 2 Complete ====="
    echo ""
fi

# 如果没有指定 API_BASE，使用默认值
if [ -z "$API_BASE" ]; then
    API_BASE="localhost:$VLLM_PORT_BASE"
fi

# =============================================================================
# Step 3: 运行采样与难度分级
# =============================================================================
echo "===== Step 3: Running sampling and difficulty grading ====="

# 构建命令
if [ "$USE_OFFLINE" = true ]; then
    # 离线 vLLM 模式
    echo "Using offline vLLM mode"
    CMD="python scripts/sample_and_grade_difficulty.py \
        --model_path $MODEL_PATH \
        --data_path $DATA_PATH \
        --output_dir $OUTPUT_DIR \
        --n_rollouts $N_ROLLOUTS \
        --max_turns $MAX_TURNS \
        --max_tokens $MAX_TOKENS \
        --temperature $TEMPERATURE \
        --top_p $TOP_P \
        --batch_size $BATCH_SIZE \
        --tensor_parallel_size $TENSOR_PARALLEL_SIZE \
        --gpu_memory_utilization $GPU_MEMORY_UTILIZATION \
        --max_model_len $MAX_MODEL_LEN \
        --vlm_endpoint $VLM_ENDPOINT \
        --vlm_model_name $VLM_MODEL_NAME \
        --vlm_api_key $VLM_API_KEY \
        --cache_dir $CACHE_DIR \
        --cache_fps $CACHE_FPS \
        --cache_max_frames $CACHE_MAX_FRAMES \
        --max_frames_per_segment $MAX_FRAMES_PER_SEGMENT \
        --initial_fps $INITIAL_FPS \
        --initial_max_frames $INITIAL_MAX_FRAMES \
        --initial_min_pixels $INITIAL_MIN_PIXELS \
        --initial_max_pixels $INITIAL_MAX_PIXELS \
        --segment_fps $SEGMENT_FPS \
        --segment_max_frames $SEGMENT_MAX_FRAMES \
        --segment_min_pixels $SEGMENT_MIN_PIXELS \
        --segment_max_pixels $SEGMENT_MAX_PIXELS \
        --answer_weight $ANSWER_WEIGHT \
        --bbox_weight $BBOX_WEIGHT \
        --format_weight $FORMAT_WEIGHT \
        --segment_weight $SEGMENT_WEIGHT"
else
    # API 模式
    CMD="python scripts/sample_and_grade_difficulty.py \
        --use_api \
        --api_base $API_BASE \
        --api_model_name $MODEL_NAME \
        --data_path $DATA_PATH \
        --output_dir $OUTPUT_DIR \
        --n_rollouts $N_ROLLOUTS \
        --max_turns $MAX_TURNS \
        --max_tokens $MAX_TOKENS \
        --temperature $TEMPERATURE \
        --top_p $TOP_P \
        --batch_size $BATCH_SIZE \
        --vlm_endpoint $VLM_ENDPOINT \
        --vlm_model_name $VLM_MODEL_NAME \
        --vlm_api_key $VLM_API_KEY \
        --cache_dir $CACHE_DIR \
        --cache_fps $CACHE_FPS \
        --cache_max_frames $CACHE_MAX_FRAMES \
        --max_frames_per_segment $MAX_FRAMES_PER_SEGMENT \
        --initial_fps $INITIAL_FPS \
        --initial_max_frames $INITIAL_MAX_FRAMES \
        --initial_min_pixels $INITIAL_MIN_PIXELS \
        --initial_max_pixels $INITIAL_MAX_PIXELS \
        --segment_fps $SEGMENT_FPS \
        --segment_max_frames $SEGMENT_MAX_FRAMES \
        --segment_min_pixels $SEGMENT_MIN_PIXELS \
        --segment_max_pixels $SEGMENT_MAX_PIXELS \
        --answer_weight $ANSWER_WEIGHT \
        --bbox_weight $BBOX_WEIGHT \
        --format_weight $FORMAT_WEIGHT \
        --segment_weight $SEGMENT_WEIGHT"
fi

# 添加可选参数
if [ -n "$NUM_SAMPLES" ]; then
    CMD="$CMD --num_samples $NUM_SAMPLES"
fi

if [ "$USE_STRICT_FORMAT" = "true" ]; then
    CMD="$CMD --use_strict_format"
fi

if [ "$RESUME" = "true" ]; then
    CMD="$CMD --resume"
fi

# 执行
echo "Running: $CMD"
eval $CMD $EXTRA_ARGS

# 清理
if [ ${#VLLM_PIDS[@]} -gt 0 ]; then
    echo ""
    echo "Stopping vLLM server(s)..."
    for pid in "${VLLM_PIDS[@]}"; do
        echo "Stopping PID: $pid"
        kill $pid 2>/dev/null || true
        wait $pid 2>/dev/null || true
    done
fi

echo ""
echo "===== Step 3 Complete ====="
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo "Check the following files:"
echo "  - realtime_results.jsonl: 实时保存的结果"
echo "  - detailed_results_*.jsonl: 详细结果"
echo "  - difficulty_summary_*.json: 难度分级摘要"
echo "  - easy_questions_*.txt: 简单问题列表"
echo "  - medium_questions_*.txt: 中等问题列表"
echo "  - hard_questions_*.txt: 困难问题列表"
echo "  - difficulty_analysis_*.parquet: 分析结果 (parquet 格式)"
