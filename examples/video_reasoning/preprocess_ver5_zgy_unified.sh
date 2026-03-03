#!/bin/bash
# Video-Holmes Data Preprocessing Script (ver4)
# 将 Video-Holmes 数据集转换为 veRL 训练格式
#
# 使用方法:
#   cd /data_gpu/songlin/rl/verl
#   bash examples/video_reasoning/preprocess_ver5_zgy_unified.sh

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
INPUT_JSON="$HOLMES_ANNOTATION_JSON"
# 视频文件目录
VIDEO_BASE_PATH="$VIDEO_HOLMES_DIR"
# 输出目录
OUTPUT_DIR="./long_video_data/video_holmes"

# ===== 数据划分参数 =====
VAL_RATIO=0.05  # 5% 作为验证集
SEED=42

# ===== 检查输入文件 =====
if [ ! -f "$INPUT_JSON" ]; then
    echo "ERROR: Input JSON not found: $INPUT_JSON"
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
echo "Converting Video-Holmes (ver4) to veRL format"
echo "=========================================="
echo "Input:  $INPUT_JSON"
echo "Videos: $VIDEO_BASE_PATH"
echo "Output: $OUTPUT_DIR"
echo ""

python examples/data_preprocess/video_reasoning_multiturn_holmes.py \
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
import pyarrow.parquet as pq

# 使用 pyarrow 读取
pf = pq.ParquetFile('$OUTPUT_DIR/train.parquet')
print('Columns:', pf.schema_arrow.names)
print('Train samples:', pf.metadata.num_rows)

pf_val = pq.ParquetFile('$OUTPUT_DIR/val.parquet')
print('Val samples:', pf_val.metadata.num_rows)

# 读取第一行验证格式
t = pf.read_row_group(0, columns=['prompt', 'videos', 'video_path', 'extra_info', 'correct_answer', 'is_openended'])

# Check prompt format
sample_prompt = t.column('prompt')[0].as_py()
if isinstance(sample_prompt, list) and len(sample_prompt) > 0 and 'content' in sample_prompt[0]:
    print('Prompt format: OK (messages list with <video> placeholder)')
else:
    print('ERROR: prompt format invalid')
    exit(1)

# Check videos format
sample_videos = t.column('videos')[0].as_py()
if isinstance(sample_videos, list) and len(sample_videos) > 0 and 'video' in sample_videos[0]:
    print('Videos format: OK')
    print('Sample video path:', sample_videos[0]['video'])
else:
    print('ERROR: videos format invalid')
    exit(1)

# Check video exists
import os
video_path = t.column('video_path')[0].as_py()
if os.path.exists(video_path):
    print('Video file verified OK')
else:
    print('WARNING: Sample video path does not exist:', video_path)

# Check openended distribution
df_train = pf.read().to_pandas()
openended_count = df_train['is_openended'].sum()
mc_count = len(df_train) - openended_count
print(f'Open-ended: {openended_count}, Multiple-choice: {mc_count}')
"

# ===== 完成 =====
echo ""
echo "=========================================="
echo "Preprocessing Complete!"
echo "=========================================="
echo "Training data:   $OUTPUT_DIR/train.parquet"
echo "Validation data: $OUTPUT_DIR/val.parquet"
echo ""
echo "Dataset: Video-Holmes Unified (1551 samples, 233 videos)"
echo ""
echo "To train with this data, modify run_video_reasoning_grpo.sh:"
echo "  DATA_DIR=\"$OUTPUT_DIR\""
echo ""
