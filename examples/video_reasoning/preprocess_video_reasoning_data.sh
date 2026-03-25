#!/bin/bash
# Video Reasoning Data Preprocessing Script
# 生成训练数据 parquet（包含 video_path）
# 视频帧缓存由训练脚本根据参数（fps, max_frames）自动处理
#
# 使用方法:
#   cd /path/to/verl
#   # 处理 longvideo_reason 数据:
#   bash examples/video_reasoning/preprocess_video_reasoning_data.sh
#
#   # 处理 video-r1 数据 (随机采样 10000 个视频样本):
#   bash examples/video_reasoning/preprocess_video_reasoning_data.sh --videor1 --max_samples 10000
#
#   # 只处理 video-r1:
#   bash examples/video_reasoning/preprocess_video_reasoning_data.sh --videor1-only --max_samples 5000

set -e

# ===== 解析命令行参数 =====
PROCESS_LONGVIDEO=true
PROCESS_VIDEOR1=false
VIDEOR1_MAX_SAMPLES=-1  # -1 表示全部
PROMPT_VERSION="default"  # default/multiturn or singleturn

while [[ $# -gt 0 ]]; do
    case $1 in
        --videor1)
            PROCESS_VIDEOR1=true
            shift
            ;;
        --videor1-only)
            PROCESS_VIDEOR1=true
            PROCESS_LONGVIDEO=false
            shift
            ;;
        --max_samples)
            VIDEOR1_MAX_SAMPLES="$2"
            shift 2
            ;;
        --no-longvideo)
            PROCESS_LONGVIDEO=false
            shift
            ;;
        --singleturn)
            PROMPT_VERSION="singleturn"
            shift
            ;;
        --prompt_version)
            PROMPT_VERSION="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--videor1] [--videor1-only] [--max_samples N] [--no-longvideo] [--singleturn] [--prompt_version VERSION]"
            exit 1
            ;;
    esac
done

# ===== 加载服务器路径配置 =====
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ ! -f "$SCRIPT_DIR/env.sh" ]; then
    echo "ERROR: $SCRIPT_DIR/env.sh not found"
    echo "Please run: cp $SCRIPT_DIR/env.sh.example $SCRIPT_DIR/env.sh"
    echo "Then edit env.sh with your server paths"
    exit 1
fi
source "$SCRIPT_DIR/env.sh"

# ===== 路径配置 =====
# 原始数据 JSON 文件
INPUT_JSON="$LONGVIDEO_REASON_JSON"
# 视频文件目录
VIDEO_BASE_PATH="$LONGVIDEO_REASON_DIR"
# 输出目录 (根据 prompt 版本调整)
# 注意：这里强制覆盖 env.sh 中的设置，确保 singleturn 数据保存到正确目录
if [ "$PROMPT_VERSION" = "singleturn" ]; then
    OUTPUT_DIR="./long_video_data_singleturn/longvideo_reason"
    VIDEOR1_OUTPUT_DIR="./long_video_data_singleturn/videor1"
else
    OUTPUT_DIR="./long_video_data/longvideo_reason"
    VIDEOR1_OUTPUT_DIR="${VIDEOR1_OUTPUT_DIR:-./long_video_data/videor1}"
fi

# ===== Video-R1 配置 (从 env.sh 读取) =====
VIDEOR1_PARQUET="${VIDEOR1_PARQUET:-/mnt/data/home/zhengshurong/dataset/LongVT-Parquet/longvt_sft_videor1_165k5.parquet}"
VIDEOR1_VIDEO_DIR="${VIDEOR1_VIDEO_DIR:-/mnt/data/home/zhengshurong/dataset/LongVT-Source/videor1}"

# ===== 数据划分参数 =====
VAL_RATIO=0.05  # 5% 作为验证集
SEED=42

# ===== 检查输入文件 =====
if [ "$PROCESS_LONGVIDEO" = true ]; then
    if [ ! -f "$INPUT_JSON" ]; then
        echo "ERROR: Input JSON not found: $INPUT_JSON"
        echo "Please ensure the results.json file exists"
        exit 1
    fi

    # ===== 检查视频目录 =====
    if [ ! -d "$VIDEO_BASE_PATH" ]; then
        echo "ERROR: Video directory not found: $VIDEO_BASE_PATH"
        exit 1
    fi

    # 检查视频目录是否有 mp4 文件
    VIDEO_COUNT=$(find "$VIDEO_BASE_PATH" -maxdepth 1 -name "*.mp4" 2>/dev/null | wc -l)
    if [ "$VIDEO_COUNT" -eq 0 ]; then
        echo "ERROR: No mp4 files found in $VIDEO_BASE_PATH"
        exit 1
    fi
    echo "Found $VIDEO_COUNT videos in $VIDEO_BASE_PATH"
fi

# ===== 检查 Video-R1 输入 =====
if [ "$PROCESS_VIDEOR1" = true ]; then
    if [ ! -f "$VIDEOR1_PARQUET" ]; then
        echo "ERROR: Video-R1 parquet not found: $VIDEOR1_PARQUET"
        exit 1
    fi
    if [ ! -d "$VIDEOR1_VIDEO_DIR" ]; then
        echo "ERROR: Video-R1 video directory not found: $VIDEOR1_VIDEO_DIR"
        exit 1
    fi
    echo "Video-R1 parquet: $VIDEOR1_PARQUET"
    echo "Video-R1 max_samples: $VIDEOR1_MAX_SAMPLES"
fi

# ===== 生成 longvideo_reason parquet 数据 =====
if [ "$PROCESS_LONGVIDEO" = true ]; then
    echo ""
    echo "=========================================="
    echo "Generating longvideo_reason parquet data"
    echo "=========================================="
    python examples/data_preprocess/video_reasoning_multiturn.py \
        --input_json "$INPUT_JSON" \
        --video_base_path "$VIDEO_BASE_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --val_ratio "$VAL_RATIO" \
        --seed "$SEED" \
        --prompt_version "$PROMPT_VERSION"

    # 检查 parquet 是否生成成功
    if [ ! -f "$OUTPUT_DIR/train.parquet" ] || [ ! -f "$OUTPUT_DIR/val.parquet" ]; then
        echo "ERROR: Failed to generate parquet files"
        exit 1
    fi
    echo "Longvideo_reason data generated successfully"
fi

# ===== 生成 Video-R1 parquet 数据 =====
if [ "$PROCESS_VIDEOR1" = true ]; then
    echo ""
    echo "=========================================="
    echo "Generating Video-R1 parquet data"
    echo "=========================================="

    VIDEOR1_ARGS=(
        --input_parquet "$VIDEOR1_PARQUET"
        --video_base_path "$VIDEOR1_VIDEO_DIR"
        --output_dir "$VIDEOR1_OUTPUT_DIR"
        --val_ratio "$VAL_RATIO"
        --seed "$SEED"
        --prompt_version "$PROMPT_VERSION"
    )

    if [ "$VIDEOR1_MAX_SAMPLES" != "-1" ]; then
        VIDEOR1_ARGS+=(--max_samples "$VIDEOR1_MAX_SAMPLES")
    fi

    python examples/data_preprocess/convert_videor1.py "${VIDEOR1_ARGS[@]}"

    # 检查 parquet 是否生成成功
    if [ ! -f "$VIDEOR1_OUTPUT_DIR/train.parquet" ] || [ ! -f "$VIDEOR1_OUTPUT_DIR/val.parquet" ]; then
        echo "ERROR: Failed to generate Video-R1 parquet files"
        exit 1
    fi
    echo "Video-R1 data generated successfully"
fi

# ===== 验证生成的数据 =====
verify_parquet() {
    local parquet_dir="$1"
    local name="$2"

    echo ""
    echo "Verifying $name parquet..."
    python3 -c "
import pandas as pd
import os

parquet_dir = '$parquet_dir'
df = pd.read_parquet(os.path.join(parquet_dir, 'train.parquet'))
print('Columns:', list(df.columns))
print('Train samples:', len(df))

df_val = pd.read_parquet(os.path.join(parquet_dir, 'val.parquet'))
print('Val samples:', len(df_val))

# Check required fields
required = ['extra_info', 'video_path', 'videos', 'prompt']
missing = [c for c in required if c not in df.columns]
if missing:
    print('ERROR: Missing required columns:', missing)
    exit(1)

# Check prompt format
sample_prompt = df['prompt'].iloc[0]
try:
    if isinstance(sample_prompt, str):
        print('ERROR: prompt should be list of messages, got string')
        exit(1)
    if 'content' not in sample_prompt[0]:
        print('ERROR: prompt[0] missing content key')
        exit(1)
    print('Prompt format: OK')
except (TypeError, KeyError, IndexError) as e:
    print('ERROR: prompt format invalid:', e)
    exit(1)

# Check videos format
sample_videos = df['videos'].iloc[0]
try:
    if isinstance(sample_videos, str):
        print('ERROR: videos should be list of video dicts, got string')
        exit(1)
    if 'video' not in sample_videos[0]:
        print('ERROR: videos[0] missing video key')
        exit(1)
    print('Videos format: OK')
except (TypeError, KeyError, IndexError) as e:
    print('ERROR: videos format invalid:', e)
    exit(1)

# Check extra_info format
sample_extra = df['extra_info'].iloc[0]
try:
    if isinstance(sample_extra, str):
        print('ERROR: extra_info should be dict, got string')
        exit(1)
    if 'index' not in sample_extra:
        print('ERROR: extra_info missing index key')
        exit(1)
    print('Extra_info format: OK')
except (TypeError, KeyError) as e:
    print('ERROR: extra_info format invalid:', e)
    exit(1)

# Check video path
sample_path = df['video_path'].iloc[0]
print('Sample video_path:', sample_path)

if not os.path.exists(sample_path):
    print('WARNING: Sample video path does not exist!')
else:
    print('Video path verified OK')
"
}

if [ "$PROCESS_LONGVIDEO" = true ]; then
    verify_parquet "$OUTPUT_DIR" "longvideo_reason"
fi

if [ "$PROCESS_VIDEOR1" = true ]; then
    verify_parquet "$VIDEOR1_OUTPUT_DIR" "video-r1"
fi

# ===== 完成 =====
echo ""
echo "=========================================="
echo "Preprocessing Complete!"
echo "=========================================="

if [ "$PROCESS_LONGVIDEO" = true ]; then
    echo "Longvideo_reason data:"
    echo "  Train: $OUTPUT_DIR/train.parquet"
    echo "  Val:   $OUTPUT_DIR/val.parquet"
fi

if [ "$PROCESS_VIDEOR1" = true ]; then
    echo "Video-R1 data:"
    echo "  Train: $VIDEOR1_OUTPUT_DIR/train.parquet"
    echo "  Val:   $VIDEOR1_OUTPUT_DIR/val.parquet"
fi

echo ""
echo "Next step: Run training with:"
echo "  bash examples/video_reasoning/run_video_reasoning_grpo.sh"
echo ""
echo "Note: Video frame caching will be done automatically by the training script"
echo "      based on cache parameters (fps, max_frames)"
echo ""
echo "Usage examples:"
echo "  # Process longvideo_reason only (default):"
echo "  bash examples/video_reasoning/preprocess_video_reasoning_data.sh"
echo ""
echo "  # Process video-r1 with 10000 random samples:"
echo "  bash examples/video_reasoning/preprocess_video_reasoning_data.sh --videor1 --max_samples 10000"
echo ""
echo "  # Process video-r1 only (skip longvideo_reason):"
echo "  bash examples/video_reasoning/preprocess_video_reasoning_data.sh --videor1-only --max_samples 5000"

echo "  # Use singleturn prompt (simple think+answer, no segment/bbox):"
echo "  bash examples/video_reasoning/preprocess_video_reasoning_data.sh --singleturn"
echo ""
