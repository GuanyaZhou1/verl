#!/bin/bash
# =============================================================================
# Preprocess refined SFT annotation data and merge into veRL parquet format
# =============================================================================
#
# Converts refined SFT conversation-format annotations to veRL parquet format.
# Processes Holmes MC and LongVideoReason open-ended datasets, then merges.
#
# Directory structure:
#   <output_base>/
#   ├── video_holmes/
#   │   ├── train.parquet
#   │   └── val.parquet
#   ├── longvideo_reason/
#   │   ├── train.parquet
#   │   └── val.parquet
#   ├── train.parquet      (merged)
#   └── val.parquet        (merged)
#
# Usage:
#   cd /data_gpu/gyzhou/prj/verl
#   bash examples/video_reasoning/preprocess_refined_and_merge.sh
#
# Options:
#   --annotation-dir <path>  SFT annotation directory (default: /data_gpu/gyzhou/annotations/refined_vlm_detect)
#   --output-base <path>     Output base directory (default: ./long_video_data_new)
#   --skip-preprocess        Skip preprocessing, only merge
#   --only <name>            Only preprocess one dataset (video_holmes|longvideo_reason)
# =============================================================================

set -e

# =============================================================================
# Load video path config
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
# Default configuration
# =============================================================================
ANNOTATION_DIR="/data_gpu/gyzhou/annotations/refined_vlm_detect"
OUTPUT_BASE="./long_video_data_new"
VAL_RATIO=0.05
SEED=42
SKIP_PREPROCESS=false
ONLY_DATASET=""

# Dataset file names (within annotation dir)
HOLMES_JSON="holmes_sft_twoturn_v4_validtimestamp_noduplicate_checkanswer.json"
LONGVIDEO_JSON="longvideoreason_multiturn_5k3_validtimestamp_noduplicate_checkanswer.json"

# =============================================================================
# Parse arguments
# =============================================================================
while [[ $# -gt 0 ]]; do
    case "$1" in
        --annotation-dir)
            ANNOTATION_DIR="$2"
            shift 2
            ;;
        --output-base)
            OUTPUT_BASE="$2"
            shift 2
            ;;
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
            echo "Usage: $0 [--annotation-dir <path>] [--output-base <path>] [--skip-preprocess] [--only <name>]"
            exit 1
            ;;
    esac
done

echo "===== Configuration ====="
echo "Annotation dir: $ANNOTATION_DIR"
echo "Output base:    $OUTPUT_BASE"
echo "Video Holmes:   $VIDEO_HOLMES_DIR"
echo "LongVideo:      $LONGVIDEO_REASON_DIR"
echo ""

# =============================================================================
# Preprocessing functions
# =============================================================================
preprocess_video_holmes() {
    local input_json="$ANNOTATION_DIR/$HOLMES_JSON"
    local output_dir="$OUTPUT_BASE/video_holmes"

    echo ""
    echo "=========================================="
    echo "[1/2] Preprocessing Video-Holmes (MC)"
    echo "=========================================="
    echo "Input:  $input_json"
    echo "Videos: $VIDEO_HOLMES_DIR"
    echo "Output: $output_dir"

    if [ ! -f "$input_json" ]; then
        echo "ERROR: Input JSON not found: $input_json"
        return 1
    fi

    python examples/data_preprocess/convert_sft_to_verl.py \
        --input_json "$input_json" \
        --video_base_path "$VIDEO_HOLMES_DIR" \
        --output_dir "$output_dir" \
        --dataset_name video_holmes \
        --val_ratio "$VAL_RATIO" \
        --seed "$SEED"

    echo "[1/2] Done: $(python3 -c "import pandas as pd; print(len(pd.read_parquet('$output_dir/train.parquet')))"
) train samples"
}

preprocess_longvideo_reason() {
    local input_json="$ANNOTATION_DIR/$LONGVIDEO_JSON"
    local output_dir="$OUTPUT_BASE/longvideo_reason"

    echo ""
    echo "=========================================="
    echo "[2/2] Preprocessing LongVideo-Reason (OE)"
    echo "=========================================="
    echo "Input:  $input_json"
    echo "Videos: $LONGVIDEO_REASON_DIR"
    echo "Output: $output_dir"

    if [ ! -f "$input_json" ]; then
        echo "ERROR: Input JSON not found: $input_json"
        return 1
    fi

    python examples/data_preprocess/convert_sft_to_verl.py \
        --input_json "$input_json" \
        --video_base_path "$LONGVIDEO_REASON_DIR" \
        --output_dir "$output_dir" \
        --dataset_name longvideo_reason \
        --val_ratio "$VAL_RATIO" \
        --seed "$SEED"

    echo "[2/2] Done: $(python3 -c "import pandas as pd; print(len(pd.read_parquet('$output_dir/train.parquet')))"
) train samples"
}

# =============================================================================
# Step 1: Preprocess
# =============================================================================
if [ "$SKIP_PREPROCESS" = false ]; then
    echo "===== Step 1: Preprocessing datasets ====="

    if [ -z "$ONLY_DATASET" ] || [ "$ONLY_DATASET" = "video_holmes" ]; then
        preprocess_video_holmes
    fi

    if [ -z "$ONLY_DATASET" ] || [ "$ONLY_DATASET" = "longvideo_reason" ]; then
        preprocess_longvideo_reason
    fi
else
    echo "===== Skipping preprocessing (--skip-preprocess) ====="
fi

# =============================================================================
# Step 2: Merge
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
    'video_holmes': os.path.join(output_base, 'video_holmes'),
    'longvideo_reason': os.path.join(output_base, 'longvideo_reason'),
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

# Concat and shuffle
merged_train = pd.concat(train_dfs, ignore_index=True)
merged_val = pd.concat(val_dfs, ignore_index=True)
merged_train = merged_train.sample(frac=1, random_state=42).reset_index(drop=True)

# Save
merged_train.to_parquet(os.path.join(output_base, 'train.parquet'), index=False)
merged_val.to_parquet(os.path.join(output_base, 'val.parquet'), index=False)

print()
print(f'Merged train: {len(merged_train)} samples')
print(f'Merged val:   {len(merged_val)} samples')
print(f'Total:        {len(merged_train) + len(merged_val)} samples')
print()
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
echo "Directory structure:"
echo "  $OUTPUT_BASE/"
echo "  ├── video_holmes/      (Holmes MC)"
echo "  ├── longvideo_reason/  (LongVideo-Reason OE)"
echo "  ├── train.parquet      (merged)"
echo "  └── val.parquet        (merged)"
