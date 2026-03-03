#!/bin/bash
# Video Reasoning Data Preprocessing Script
# 生成训练数据 parquet（包含 video_path）
# 视频帧缓存由训练脚本根据参数（fps, max_frames）自动处理
#
# 使用方法:
#   cd /data_gpu/songlin/rl/verl
#   bash examples/video_reasoning/preprocess_video_reasoning_data.sh

set -e

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
# 输出目录
OUTPUT_DIR="./long_video_data"

# ===== 数据划分参数 =====
VAL_RATIO=0.05  # 5% 作为验证集
SEED=42

# ===== 检查输入文件 =====
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

# ===== 生成 parquet 训练数据 =====
echo ""
echo "=========================================="
echo "Generating parquet training data"
echo "=========================================="
python examples/data_preprocess/video_reasoning_multiturn.py \
    --input_json "$INPUT_JSON" \
    --video_base_path "$VIDEO_BASE_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --val_ratio "$VAL_RATIO" \
    --seed "$SEED"

# 检查 parquet 是否生成成功
if [ ! -f "$OUTPUT_DIR/train.parquet" ] || [ ! -f "$OUTPUT_DIR/val.parquet" ]; then
    echo "ERROR: Failed to generate parquet files"
    exit 1
fi

echo ""
echo "Verifying generated parquet..."
python3 -c "
import pandas as pd
df = pd.read_parquet('$OUTPUT_DIR/train.parquet')
print('Columns:', list(df.columns))
print('Train samples:', len(df))

df_val = pd.read_parquet('$OUTPUT_DIR/val.parquet')
print('Val samples:', len(df_val))

# Check required fields (tools_kwargs is now inside extra_info)
required = ['extra_info', 'video_path', 'videos', 'prompt']
missing = [c for c in required if c not in df.columns]
if missing:
    print('ERROR: Missing required columns:', missing)
    exit(1)

# Check prompt format (should be list/array of messages, not JSON string)
sample_prompt = df['prompt'].iloc[0]
# pandas may convert list to numpy array, so check if it's indexable with dict elements
try:
    if isinstance(sample_prompt, str):
        print('ERROR: prompt should be list of messages, got string')
        exit(1)
    if 'content' not in sample_prompt[0]:
        print('ERROR: prompt[0] missing content key')
        exit(1)
    print('Prompt format: OK (messages list with <video> placeholder)')
except (TypeError, KeyError, IndexError) as e:
    print('ERROR: prompt format invalid:', e)
    exit(1)

# Check videos format (should be list/array of video dicts, not JSON string)
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

# Check extra_info format (should be dict with index and tools_kwargs)
sample_extra = df['extra_info'].iloc[0]
try:
    if isinstance(sample_extra, str):
        print('ERROR: extra_info should be dict, got string')
        exit(1)
    if 'index' not in sample_extra:
        print('ERROR: extra_info missing index key')
        exit(1)
    if 'tools_kwargs' not in sample_extra:
        print('WARNING: extra_info missing tools_kwargs (OK for video_reasoning agent loop)')
    print('Extra_info format: OK')
except (TypeError, KeyError) as e:
    print('ERROR: extra_info format invalid:', e)
    exit(1)

# Check video path
sample_path = df['video_path'].iloc[0]
print('Sample video_path:', sample_path)

import os
if not os.path.exists(sample_path):
    print('WARNING: Sample video path does not exist!')
else:
    print('Video path verified OK')
"

# ===== 完成 =====
echo ""
echo "=========================================="
echo "Preprocessing Complete!"
echo "=========================================="
echo "Training data: $OUTPUT_DIR/train.parquet"
echo "Validation data: $OUTPUT_DIR/val.parquet"
echo ""
echo "Next step: Run training with:"
echo "  bash examples/video_reasoning/run_video_reasoning_grpo.sh"
echo ""
echo "Note: Video frame caching will be done automatically by the training script"
echo "      based on cache parameters (fps, max_frames)"
