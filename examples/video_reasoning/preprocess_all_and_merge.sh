#!/bin/bash
# =============================================================================
# 统一预处理 + 合并脚本
# =============================================================================
# 将三个数据集分别预处理到各自子文件夹，然后合并到顶层 train/val.parquet
#
# 目录结构：
#   long_video_data/
#   ├── longvideo_reason/    # 数据集1: LongVideo-Reason (多选+开放, ~2586条)
#   │   ├── train.parquet
#   │   └── val.parquet
#   ├── video_holmes/        # 数据集2: Video-Holmes (多选题, ~1551条)
#   │   ├── train.parquet
#   │   └── val.parquet
#   ├── longvt_selfqa/       # 数据集3: LongVT Self-QA (开放题, ~1668条)
#   │   ├── train.parquet
#   │   └── val.parquet
#   ├── train.parquet         # 合并后 (~5805条)
#   └── val.parquet           # 合并后
#
# 使用方法:
#   cd /data_gpu/gyzhou/prj/verl
#   bash examples/video_reasoning/preprocess_all_and_merge.sh
#
# 可选参数:
#   --skip-preprocess   跳过预处理，只做合并（子文件夹已有 parquet 时用）
#   --only <name>       只预处理指定数据集 (longvideo_reason|video_holmes|longvt_selfqa)
# =============================================================================

set -e

# =============================================================================
# 加载服务器路径配置
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ ! -f "$SCRIPT_DIR/env.sh" ]; then
    echo "ERROR: $SCRIPT_DIR/env.sh not found"
    echo "Please run: cp $SCRIPT_DIR/env.sh.example $SCRIPT_DIR/env.sh"
    echo "Then edit env.sh with your server paths"
    exit 1
fi
source "$SCRIPT_DIR/env.sh"

# =============================================================================
# 路径配置
# =============================================================================
OUTPUT_BASE="./long_video_data"
VAL_RATIO=0.05
SEED=42

# 数据集1: LongVideo-Reason
DS1_NAME="longvideo_reason"
DS1_INPUT_JSON="$LONGVIDEO_REASON_JSON"
DS1_VIDEO_BASE="$LONGVIDEO_REASON_DIR"
DS1_OUTPUT="$OUTPUT_BASE/$DS1_NAME"

# 数据集2: Video-Holmes
DS2_NAME="video_holmes"
DS2_INPUT_JSON="$HOLMES_ANNOTATION_JSON"
DS2_VIDEO_BASE="$VIDEO_HOLMES_DIR"
DS2_OUTPUT="$OUTPUT_BASE/$DS2_NAME"

# 数据集3: LongVT Self-QA
DS3_NAME="longvt_selfqa"
DS3_INPUT_JSON="$LONGVT_SELFQA_JSON"
DS3_VIDEO_BASE="$LONGVT_SELFQA_DIR"
DS3_OUTPUT="$OUTPUT_BASE/$DS3_NAME"

# =============================================================================
# 参数解析
# =============================================================================
SKIP_PREPROCESS=false
ONLY_DATASET=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-preprocess)
            SKIP_PREPROCESS=true
            shift
            ;;
        --only)
            ONLY_DATASET="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--skip-preprocess] [--only <dataset_name>]"
            exit 1
            ;;
    esac
done

# =============================================================================
# 预处理函数
# =============================================================================
preprocess_longvideo_reason() {
    echo ""
    echo "=========================================="
    echo "[1/3] Preprocessing LongVideo-Reason"
    echo "=========================================="
    echo "Input:  $DS1_INPUT_JSON"
    echo "Videos: $DS1_VIDEO_BASE"
    echo "Output: $DS1_OUTPUT"

    if [ ! -f "$DS1_INPUT_JSON" ]; then
        echo "ERROR: Input JSON not found: $DS1_INPUT_JSON"
        return 1
    fi

    python examples/data_preprocess/video_reasoning_multiturn.py \
        --input_json "$DS1_INPUT_JSON" \
        --video_base_path "$DS1_VIDEO_BASE" \
        --output_dir "$DS1_OUTPUT" \
        --val_ratio "$VAL_RATIO" \
        --seed "$SEED"

    echo "[1/3] Done: $(python3 -c "import pandas as pd; print(len(pd.read_parquet('$DS1_OUTPUT/train.parquet')))"
) train samples"
}

preprocess_video_holmes() {
    echo ""
    echo "=========================================="
    echo "[2/3] Preprocessing Video-Holmes"
    echo "=========================================="
    echo "Input:  $DS2_INPUT_JSON"
    echo "Videos: $DS2_VIDEO_BASE"
    echo "Output: $DS2_OUTPUT"

    if [ ! -f "$DS2_INPUT_JSON" ]; then
        echo "ERROR: Input JSON not found: $DS2_INPUT_JSON"
        return 1
    fi

    python examples/data_preprocess/video_reasoning_multiturn_holmes.py \
        --input_json "$DS2_INPUT_JSON" \
        --video_base_path "$DS2_VIDEO_BASE" \
        --output_dir "$DS2_OUTPUT" \
        --val_ratio "$VAL_RATIO" \
        --seed "$SEED"

    echo "[2/3] Done: $(python3 -c "import pandas as pd; print(len(pd.read_parquet('$DS2_OUTPUT/train.parquet')))"
) train samples"
}

preprocess_longvt_selfqa() {
    echo ""
    echo "=========================================="
    echo "[3/3] Preprocessing LongVT Self-QA"
    echo "=========================================="
    echo "Input:  $DS3_INPUT_JSON"
    echo "Videos: $DS3_VIDEO_BASE"
    echo "Output: $DS3_OUTPUT"

    if [ ! -f "$DS3_INPUT_JSON" ]; then
        echo "ERROR: Input JSON not found: $DS3_INPUT_JSON"
        return 1
    fi

    python examples/data_preprocess/convert_longvt_selfqa.py \
        --input_json "$DS3_INPUT_JSON" \
        --video_base_path "$DS3_VIDEO_BASE" \
        --output_dir "$DS3_OUTPUT" \
        --val_ratio "$VAL_RATIO" \
        --seed "$SEED" \
        --skip_missing_videos

    echo "[3/3] Done: $(python3 -c "import pandas as pd; print(len(pd.read_parquet('$DS3_OUTPUT/train.parquet')))"
) train samples"
}

# =============================================================================
# Step 1: 预处理各数据集
# =============================================================================
if [ "$SKIP_PREPROCESS" = false ]; then
    echo "===== Step 1: Preprocessing datasets ====="

    if [ -z "$ONLY_DATASET" ] || [ "$ONLY_DATASET" = "$DS1_NAME" ]; then
        preprocess_longvideo_reason
    fi

    if [ -z "$ONLY_DATASET" ] || [ "$ONLY_DATASET" = "$DS2_NAME" ]; then
        preprocess_video_holmes
    fi

    if [ -z "$ONLY_DATASET" ] || [ "$ONLY_DATASET" = "$DS3_NAME" ]; then
        preprocess_longvt_selfqa
    fi
else
    echo "===== Skipping preprocessing (--skip-preprocess) ====="
fi

# =============================================================================
# Step 2: 合并所有数据集
# =============================================================================
echo ""
echo "=========================================="
echo "Merging all datasets"
echo "=========================================="

python3 -c "
import pandas as pd
import os

output_base = '$OUTPUT_BASE'
datasets = {
    '$DS1_NAME': '$DS1_OUTPUT',
    '$DS2_NAME': '$DS2_OUTPUT',
    '$DS3_NAME': '$DS3_OUTPUT',
}

train_dfs = []
val_dfs = []

for name, path in datasets.items():
    train_path = os.path.join(path, 'train.parquet')
    val_path = os.path.join(path, 'val.parquet')

    if not os.path.exists(train_path):
        print(f'WARNING: {train_path} not found, skipping {name}')
        continue

    df_train = pd.read_parquet(train_path)
    df_val = pd.read_parquet(val_path)

    print(f'  {name}: {len(df_train)} train + {len(df_val)} val = {len(df_train) + len(df_val)} total')
    train_dfs.append(df_train)
    val_dfs.append(df_val)

if not train_dfs:
    print('ERROR: No datasets found to merge!')
    exit(1)

# Concat
merged_train = pd.concat(train_dfs, ignore_index=True)
merged_val = pd.concat(val_dfs, ignore_index=True)

# Shuffle merged data (keep val deterministic)
merged_train = merged_train.sample(frac=1, random_state=42).reset_index(drop=True)

# Save
merged_train.to_parquet(os.path.join(output_base, 'train.parquet'), index=False)
merged_val.to_parquet(os.path.join(output_base, 'val.parquet'), index=False)

print()
print(f'Merged train: {len(merged_train)} samples')
print(f'Merged val:   {len(merged_val)} samples')
print(f'Total:        {len(merged_train) + len(merged_val)} samples')
print()

# Stats
print('Per-dataset breakdown (train):')
if 'data_source' in merged_train.columns:
    print(merged_train['data_source'].value_counts().to_string())
print()
print(f'Open-ended: {merged_train[\"is_openended\"].sum()}, Multiple-choice: {(~merged_train[\"is_openended\"]).sum()}')
print()
print(f'Saved to:')
print(f'  {os.path.join(output_base, \"train.parquet\")}')
print(f'  {os.path.join(output_base, \"val.parquet\")}')
"

echo ""
echo "=========================================="
echo "All done!"
echo "=========================================="
echo ""
echo "目录结构:"
echo "  $OUTPUT_BASE/"
echo "  ├── $DS1_NAME/     (LongVideo-Reason)"
echo "  ├── $DS2_NAME/      (Video-Holmes)"
echo "  ├── $DS3_NAME/    (LongVT Self-QA)"
echo "  ├── train.parquet    (合并)"
echo "  └── val.parquet      (合并)"
