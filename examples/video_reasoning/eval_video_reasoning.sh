#!/bin/bash
# =============================================================================
# Video Reasoning 独立评测脚本
# =============================================================================
# 只需提供模型路径即可运行评测和可视化，无需训练
#
# 使用方式：
#   bash eval_video_reasoning.sh /path/to/model
#   bash eval_video_reasoning.sh /path/to/model "my_experiment"
#
# 参数：
#   $1 - 模型路径（必须，HuggingFace 格式，需含 config.json）
#   $2 - 实验名称（可选，默认使用模型目录名）
#
# 环境变量：
#   SKIP_NORMAL_EVAL=true   - 跳过普通评测
#   SKIP_MULTITURN_EVAL=true - 跳过多轮评测
#   SKIP_VIS=true           - 跳过可视化
#   NORMAL_TASKS="..."      - 自定义普通评测任务
#   MULTITURN_TASKS="..."   - 自定义多轮评测任务
#
# =============================================================================

set -e

# =============================================================================
# 参数解析
# =============================================================================
MODEL_PATH="${1:-}"
EXPERIMENT_NAME="${2:-}"
SKIP_NORMAL_EVAL=true
if [ -z "$MODEL_PATH" ]; then
    echo "ERROR: 必须提供模型路径"
    echo ""
    echo "Usage: bash eval_video_reasoning.sh /path/to/model [experiment_name]"
    echo ""
    echo "Examples:"
    echo "  bash eval_video_reasoning.sh ./checkpoints/video-reasoning-dapo/my_exp"
    echo "  bash eval_video_reasoning.sh ./checkpoints/my_exp \"my_experiment_name\""
    echo ""
    echo "Environment variables:"
    echo "  SKIP_NORMAL_EVAL=true    - Skip normal eval"
    echo "  SKIP_MULTITURN_EVAL=true - Skip multiturn eval"
    echo "  SKIP_VIS=true            - Skip visualization"
    echo "  NORMAL_TASKS=\"...\"       - Custom normal eval tasks"
    echo "  MULTITURN_TASKS=\"...\"    - Custom multiturn eval tasks"
    exit 1
fi

# 转为绝对路径
MODEL_PATH=$(realpath "$MODEL_PATH")

# 检查模型路径
if [ ! -f "$MODEL_PATH/config.json" ]; then
    echo "ERROR: 模型路径无效，未找到 config.json"
    echo "路径: $MODEL_PATH"
    exit 1
fi

# 默认实验名称为模型目录名
if [ -z "$EXPERIMENT_NAME" ]; then
    EXPERIMENT_NAME=$(basename "$MODEL_PATH")
fi

# =============================================================================
# 评测配置
# =============================================================================
CONDA_BASE="/mnt/data/home/zhengshurong/miniconda3"
CONDA_ENV="lmms"

EVAL_WORKDIR="/mnt/data/home/zhengshurong/project/lmms-eval"

# Normal eval: 通用能力评测
EVAL_SCRIPT_NORMAL="${EVAL_WORKDIR}/examples/models/vidvllm_task_parallel.sh"
NORMAL_TASKS="${NORMAL_TASKS:-video_holmes,lvbench,videomme_format,vsibench_format}"

# Multiturn eval: 多轮能力评测
EVAL_SCRIPT_MULTITURN="${EVAL_WORKDIR}/examples/models/vidvllm_task_parallel_multiturn.sh"
MULTITURN_TASKS="${MULTITURN_TASKS:-video_holmes_multiturn,video_mmmu_multiturn}"

# 输出目录
OUTPUT_DIR="${MODEL_PATH}/eval_results"
mkdir -p "${OUTPUT_DIR}/logs"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# =============================================================================
# 打印配置
# =============================================================================
echo "=========================================="
echo "Video Reasoning 评测"
echo "=========================================="
echo "模型路径:       ${MODEL_PATH}"
echo "实验名称:       ${EXPERIMENT_NAME}"
echo "输出目录:       ${OUTPUT_DIR}"
echo ""
echo "Normal eval:"
echo "  脚本:         ${EVAL_SCRIPT_NORMAL}"
echo "  任务:         ${NORMAL_TASKS}"
echo "  跳过:         ${SKIP_NORMAL_EVAL:-false}"
echo ""
echo "Multiturn eval:"
echo "  脚本:         ${EVAL_SCRIPT_MULTITURN}"
echo "  任务:         ${MULTITURN_TASKS}"
echo "  跳过:         ${SKIP_MULTITURN_EVAL:-false}"
echo ""
echo "Conda 环境:     ${CONDA_ENV}"
echo "=========================================="
echo ""

# =============================================================================
# 激活 Conda 环境
# =============================================================================
echo "[$(date)] Activating conda environment: ${CONDA_ENV}"
eval "$(${CONDA_BASE}/bin/conda shell.bash hook)"
conda activate ${CONDA_ENV}

# 环境变量
export LD_LIBRARY_PATH=/usr/local/cuda-13.1/compat:$LD_LIBRARY_PATH 2>/dev/null || true
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export MKL_THREADING_LAYER="GNU"
export TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor_${USER}_$(id -u)"
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="/mnt/data/home/zhengshurong/.cache/huggingface/"

# =============================================================================
# Step 1a: Normal Evaluation (通用能力评测)
# =============================================================================
SKIP_NORMAL_EVAL=${SKIP_NORMAL_EVAL:-false}

if [ "$SKIP_NORMAL_EVAL" != "true" ]; then
    echo ""
    echo "===== Step 1a: Running Normal Evaluation ====="

    EVAL_LOG_NORMAL="${OUTPUT_DIR}/logs/eval_normal_${TIMESTAMP}.log"

    if [ -f "${EVAL_SCRIPT_NORMAL}" ]; then
        pushd "${EVAL_WORKDIR}" > /dev/null

        echo "[$(date)] Starting normal evaluation..."
        echo "Tasks: ${NORMAL_TASKS}"
        echo "Log: ${EVAL_LOG_NORMAL}"

        bash "${EVAL_SCRIPT_NORMAL}" \
            "${MODEL_PATH}" \
            "${NORMAL_TASKS}" \
            "${EXPERIMENT_NAME}" 2>&1 | tee "${EVAL_LOG_NORMAL}"

        EVAL_NORMAL_EXIT=${PIPESTATUS[0]}
        popd > /dev/null

        if [ ${EVAL_NORMAL_EXIT} -ne 0 ]; then
            echo "[$(date)] Normal evaluation failed with exit code ${EVAL_NORMAL_EXIT}."
        else
            echo "[$(date)] Normal evaluation completed successfully."
        fi
    else
        echo "[$(date)] Normal eval script not found at ${EVAL_SCRIPT_NORMAL}, skipping."
    fi

    echo ""
    echo "===== Step 1a Complete ====="
else
    echo ""
    echo "===== Step 1a: Normal evaluation skipped (SKIP_NORMAL_EVAL=true) ====="
fi

# =============================================================================
# 等待 GPU 进程退出（两次评测之间）
# =============================================================================
echo ""
echo "[$(date)] Waiting for GPU processes from normal eval to exit..."
for i in $(seq 1 30); do
    GPU_PROCS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -v "^$" | wc -l)
    if [ "$GPU_PROCS" -eq 0 ]; then
        echo "[$(date)] All GPU processes exited."
        break
    fi
    echo "[$(date)] Still $GPU_PROCS GPU process(es) running, waiting... ($i/30)"
    sleep 5
done
sleep 5

# =============================================================================
# Step 1b: Multiturn Evaluation (多轮能力评测)
# =============================================================================
SKIP_MULTITURN_EVAL=${SKIP_MULTITURN_EVAL:-false}

if [ "$SKIP_MULTITURN_EVAL" != "true" ]; then
    echo ""
    echo "===== Step 1b: Running Multiturn Evaluation ====="

    EVAL_LOG_MULTITURN="${OUTPUT_DIR}/logs/eval_multiturn_${TIMESTAMP}.log"

    if [ -f "${EVAL_SCRIPT_MULTITURN}" ]; then
        pushd "${EVAL_WORKDIR}" > /dev/null

        echo "[$(date)] Starting multiturn evaluation..."
        echo "Tasks: ${MULTITURN_TASKS}"
        echo "Log: ${EVAL_LOG_MULTITURN}"

        bash "${EVAL_SCRIPT_MULTITURN}" \
            "${MODEL_PATH}" \
            "${MULTITURN_TASKS}" \
            "${EXPERIMENT_NAME}_multiturn" 2>&1 | tee "${EVAL_LOG_MULTITURN}"

        EVAL_MT_EXIT=${PIPESTATUS[0]}
        popd > /dev/null

        if [ ${EVAL_MT_EXIT} -ne 0 ]; then
            echo "[$(date)] Multiturn evaluation failed with exit code ${EVAL_MT_EXIT}."
        else
            echo "[$(date)] Multiturn evaluation completed successfully."
        fi
    else
        echo "[$(date)] Multiturn eval script not found at ${EVAL_SCRIPT_MULTITURN}, skipping."
    fi

    echo ""
    echo "===== Step 1b Complete ====="
else
    echo ""
    echo "===== Step 1b: Multiturn evaluation skipped (SKIP_MULTITURN_EVAL=true) ====="
fi

# =============================================================================
# Step 2: 等待 GPU 进程退出（评测和可视化之间）
# =============================================================================
echo ""
echo "[$(date)] Waiting for GPU processes to exit before visualization..."
for i in $(seq 1 30); do
    GPU_PROCS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -v "^$" | wc -l)
    if [ "$GPU_PROCS" -eq 0 ]; then
        echo "[$(date)] All GPU processes exited."
        break
    fi
    echo "[$(date)] Still $GPU_PROCS GPU process(es) running, waiting... ($i/30)"
    sleep 5
done
sleep 3

# =============================================================================
# Step 3: Bbox 可视化 + 量化指标
# =============================================================================
SKIP_VIS=${SKIP_VIS:-false}

if [ "$SKIP_VIS" != "true" ]; then
    echo ""
    echo "===== Step 2: Running Visualization & Metrics ====="

    # 评测结果目录
    LOGS_DIR="${EVAL_WORKDIR}/logs_zsr"
    MT_SAMPLES_DIR="${LOGS_DIR}/${EXPERIMENT_NAME}_multiturn"

    # --- 2a: Bbox 可视化 ---
    VIZ_OUTPUT="${OUTPUT_DIR}/visualizations_multiturn"
    VIZ_LOG="${OUTPUT_DIR}/logs/visualize_${TIMESTAMP}.log"

    if [ -d "${MT_SAMPLES_DIR}" ]; then
        echo "[$(date)] Running bbox visualization (multiturn results)..."
        echo "Samples dir: ${MT_SAMPLES_DIR}"
        echo "Output: ${VIZ_OUTPUT}"

        pushd "${EVAL_WORKDIR}" > /dev/null
        python -m lmms_eval.visualize_bbox \
            --samples_dir "${MT_SAMPLES_DIR}" \
            --output_dir "${VIZ_OUTPUT}" \
            --max_samples 50 2>&1 | tee "${VIZ_LOG}"
        popd > /dev/null

        echo "[$(date)] Visualization complete."
    else
        echo "[$(date)] No multiturn samples found at ${MT_SAMPLES_DIR}, skipping visualization."
    fi

    # --- 2b: 量化 Bbox 指标 ---
    METRICS_OUTPUT="${OUTPUT_DIR}/bbox_metrics"
    METRICS_LOG="${OUTPUT_DIR}/logs/metrics_${TIMESTAMP}.log"
    mkdir -p "${METRICS_OUTPUT}"

    # 检查是否有 video_holmes_multiturn 任务
    if [[ "${MULTITURN_TASKS}" == *"video_holmes_multiturn"* ]] && [ -d "${MT_SAMPLES_DIR}" ]; then
        echo ""
        echo "[$(date)] Running bbox metrics analysis..."

        pushd "${EVAL_WORKDIR}" > /dev/null
        python -m lmms_eval.analyze_bbox_metrics \
            --logs_dir "${LOGS_DIR}" \
            --experiments "${EXPERIMENT_NAME}_multiturn" \
            --task video_holmes_multiturn \
            --output_dir "${METRICS_OUTPUT}" \
            --num_vis_samples 50 \
            --gt_endpoint 10.0.1.36:8081 \
            --gt_model_name Qwen3-VL-30B-A3B-Instruct \
            --gt_api_key 123456 \
            --num_gt_samples 200 \
            --gt_concurrency 16 \
            --seed 42 2>&1 | tee "${METRICS_LOG}"
        METRICS_EXIT=${PIPESTATUS[0]}
        popd > /dev/null

        if [ ${METRICS_EXIT} -ne 0 ]; then
            echo "[$(date)] Metrics analysis failed (exit ${METRICS_EXIT})."
        else
            echo "[$(date)] Metrics saved to ${METRICS_OUTPUT}"
        fi
    else
        echo "[$(date)] Skipping bbox metrics (task not in list or no samples)."
    fi

    echo ""
    echo "===== Step 2 Complete ====="
else
    echo ""
    echo "===== Step 2: Visualization skipped (SKIP_VIS=true) ====="
fi

# =============================================================================
# 完成
# =============================================================================
echo ""
echo "=========================================="
echo "All Done!"
echo "=========================================="
echo "模型:           ${MODEL_PATH}"
echo "实验名称:       ${EXPERIMENT_NAME}"
echo ""
echo "Normal eval:    ${EVAL_WORKDIR}/logs_zsr/${EXPERIMENT_NAME}/"
echo "Multiturn eval: ${EVAL_WORKDIR}/logs_zsr/${EXPERIMENT_NAME}_multiturn/"
echo "可视化:         ${OUTPUT_DIR}/visualizations_multiturn/"
echo "Bbox 指标:      ${OUTPUT_DIR}/bbox_metrics/"
echo "日志:           ${OUTPUT_DIR}/logs/"
echo "=========================================="
