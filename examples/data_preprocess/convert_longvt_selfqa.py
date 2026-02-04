#!/usr/bin/env python3
"""
Convert longvt_rl_selfqa_1k6.json to veRL training format.

This script converts open-ended video QA data to the same format as video_reasoning_multiturn.py,
but without options (for open-ended questions).

Usage:
    python convert_longvt_selfqa.py \
        --input_json /data_gpu/zhengshurong/data/dataset/LongVT-Parquet/longvt_rl_selfqa_1k6.json \
        --video_base_path /data_gpu/zhengshurong/data/dataset/LongVT-Source/selfqa \
        --output_dir /data_gpu/songlin/rl/verl/long_video_data/longvt_selfqa
"""

import json
import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
from tqdm import tqdm
import cv2


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds using OpenCV."""
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()

        if fps > 0:
            return frame_count / fps
        return 0.0
    except Exception as e:
        print(f"Warning: Could not get duration for {video_path}: {e}")
        return 0.0


# System prompt matching the current training format (same as video_reasoning_multiturn.py)
MULTITURN_SYSTEM_PROMPT = """You should reason step by step and, in EACH step, FIRST analyze and THEN focus on specific video segments. Place the grounded time segments at the END of the step.

Each reasoning step must be enclosed within '<think>' tags and reference specific time segments.

<think>
{Single reasoning step — analyze the question; summarize relevant findings from the currently available sampled input and any previously inspected segments; brainstorm hypotheses; verify whether current evidence is sufficient; refine errors; revisit prior steps if needed; if insufficient to answer, decide the NEXT most informative segments to inspect based on question intent and previously seen content}
</think>

When identifying relevant segments, use '<segment>' tags with time ranges in seconds:

<segment>
[(start1, end1), (start2, end2), ...]
</segment>

Your reasoning should be grounded in visual spatiotemporal evidence from the video. When mentioning any objects related to the evidence, strictly follow this format:
<obj>object_name</obj><box>[x1,y1,x2,y2]</box>at<t>time_in_seconds</t>

When ready to provide the final answer, enclose it within '<answer>' tags:

<answer> {final answer} </answer>
"""

# Output template for open-ended questions (different from multiple-choice)
OUTPUT_TEMPLATE_OPENENDED = "Please provide your answer within the <answer> </answer> tags."


def parse_args():
    parser = argparse.ArgumentParser(description="Convert longvt selfqa data to veRL format")
    parser.add_argument(
        "--input_json",
        type=str,
        default="/data_gpu/zhengshurong/data/dataset/LongVT-Parquet/longvt_rl_selfqa_1k6.json",
        help="Path to the input longvt json file"
    )
    parser.add_argument(
        "--video_base_path",
        type=str,
        default="/data_gpu/zhengshurong/data/dataset/LongVT-Source/selfqa",
        help="Base path to video files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data_gpu/songlin/rl/verl/long_video_data/longvt_selfqa",
        help="Output directory for parquet files"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.05,
        help="Ratio of data to use for validation (default: 0.05)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val split"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=-1,
        help="Maximum number of samples to process (-1 for all)"
    )
    parser.add_argument(
        "--skip_missing_videos",
        action="store_true",
        help="Skip samples with missing video files instead of raising error"
    )
    return parser.parse_args()


def extract_video_filename(videos_field: List[Dict]) -> str:
    """
    Extract video filename from the videos field.

    Input format: [{"video": "file:///softhome/.../xxx.mp4", ...}]
    Output: "xxx.mp4"
    """
    video_url = videos_field[0]["video"]
    # Remove file:// prefix and get filename
    if video_url.startswith("file://"):
        video_url = video_url[7:]  # Remove "file://"
    return os.path.basename(video_url)


def create_prompt_messages(
    question: str,
    duration: float = None
) -> List[Dict[str, Any]]:
    """
    Create the initial prompt as a messages list for veRL (open-ended version).

    Args:
        question: The question to answer
        duration: Duration of the video in seconds (optional)

    Returns:
        List of message dictionaries in chat format
    """
    user_content_parts = []

    # Add video placeholder
    user_content_parts.append("<video>\n")

    # Add duration if available
    if duration:
        user_content_parts.append(f"This is a video with duration {duration:.1f} seconds.\n")

    # Add system prompt
    user_content_parts.append(MULTITURN_SYSTEM_PROMPT)

    # Add question (no options for open-ended)
    user_content_parts.append(f"\nQuestion:\n{question}\n")

    # Add output template for open-ended questions
    user_content_parts.append(OUTPUT_TEMPLATE_OPENENDED)

    user_content = "".join(user_content_parts)

    return [{"role": "user", "content": user_content}]


def process_sample(sample: Dict[str, Any], video_base_path: str, skip_missing: bool = False) -> Dict[str, Any]:
    """
    Process a single sample from longvt format to veRL format.

    Args:
        sample: A single data sample from longvt json
        video_base_path: Base path to video files
        skip_missing: If True, return None for missing videos instead of raising error

    Returns:
        Processed sample in veRL format, or None if video missing and skip_missing=True
    """
    # Extract from longvt format
    question = sample["extra_info"]["question"]
    answer = sample["extra_info"]["answer"]
    data_source = sample.get("data_source", "longvt_selfqa")
    index = sample["extra_info"].get("index", 0)
    video_segment = sample["extra_info"].get("video_segment", [])

    # Extract video filename and construct full path
    video_filename = extract_video_filename(sample["videos"])
    video_path = os.path.join(video_base_path, video_filename)

    # Check if video exists
    if not os.path.exists(video_path):
        if skip_missing:
            print(f"Warning: Video not found, skipping: {video_path}")
            return None
        else:
            raise FileNotFoundError(f"Video not found: {video_path}")

    # Get video duration
    duration = get_video_duration(video_path)

    # Extract video_id from filename (without extension)
    video_id = os.path.splitext(video_filename)[0]

    # Create prompt messages (open-ended format, no options)
    prompt_messages = create_prompt_messages(question, duration=duration)

    # Videos field - only path, no resolution params (injected at training time)
    videos = [{"video": video_path}]

    # tools_kwargs for ToolAgentLoop compatibility
    tools_kwargs = {
        "fetch_frames": {
            "create_kwargs": {
                "video_path": video_path,
                "video_duration": duration,
            }
        }
    }

    # extra_info structure (和原 video_reasoning_multiturn.py 保持一致)
    extra_info = {
        "split": "train",
        "index": index,
        "video_path": video_path,
        "video_duration": duration,
        "video_id": video_id,
        "question": question,
        "correct_answer": answer,
        # video_segment 存在顶层 reference_segments 字段，不在 extra_info 重复存储
        "need_tools_kwargs": True,
        "tools_kwargs": tools_kwargs,
    }

    processed = {
        # Core fields
        "prompt": prompt_messages,
        "videos": videos,
        "video_path": video_path,
        "video_id": video_id,
        "question_id": index,
        "question": question,
        "options": "",  # Empty for open-ended
        "correct_answer": answer,
        # reward_model field for NaiveRewardManager
        "reward_model": {"style": "rule", "ground_truth": answer},
        # Metadata
        "question_type": "open_ended",
        "is_openended": True,
        "source": data_source,
        "reference_reasoning": "",
        "reference_segments": json.dumps([video_segment]) if video_segment else "[]",  # 包装成 [[start, end]] 格式
        "data_source": f"longvt_{data_source}",
        # extra_info
        "extra_info": extra_info,
    }

    return processed


def main():
    args = parse_args()

    # Load input JSON
    print(f"Loading data from {args.input_json}...")
    with open(args.input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} samples")

    # Limit samples if specified
    if args.max_samples > 0:
        data = data[:args.max_samples]
        print(f"Processing first {len(data)} samples")

    # Process each sample
    processed_data = []
    skipped_count = 0
    for sample in tqdm(data, desc="Processing samples"):
        try:
            processed = process_sample(sample, args.video_base_path, skip_missing=args.skip_missing_videos)
            if processed is not None:
                processed_data.append(processed)
            else:
                skipped_count += 1
        except Exception as e:
            print(f"Error processing sample {sample.get('extra_info', {}).get('index', 'unknown')}: {e}")
            if not args.skip_missing_videos:
                raise
            skipped_count += 1

    print(f"Successfully processed {len(processed_data)} samples")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} samples due to missing videos")

    if len(processed_data) == 0:
        print("No samples processed, exiting.")
        return

    # Convert to DataFrame
    df = pd.DataFrame(processed_data)

    # Split into train and val
    import random
    random.seed(args.seed)

    indices = list(range(len(df)))
    random.shuffle(indices)

    val_size = int(len(df) * args.val_ratio)
    train_size = len(df) - val_size

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_df = df.iloc[train_indices].reset_index(drop=True)
    val_df = df.iloc[val_indices].reset_index(drop=True)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save train parquet
    train_file = output_dir / "train.parquet"
    train_df.to_parquet(train_file, index=False)
    print(f"\nSaved training data to {train_file}")
    print(f"  Train samples: {len(train_df)}")

    # Save val parquet
    val_file = output_dir / "val.parquet"
    val_df.to_parquet(val_file, index=False)
    print(f"Saved validation data to {val_file}")
    print(f"  Val samples: {len(val_df)}")

    print(f"\nDataFrame columns: {list(df.columns)}")

    # Print statistics
    print(f"\nStatistics:")
    print(f"  - Total samples: {len(df)}")
    print(f"  - Unique videos: {df['video_id'].nunique()}")
    print(f"  - Data sources: {df['source'].value_counts().to_dict()}")
    print(f"  - All open-ended: {df['is_openended'].all()}")

    # Show sample prompt
    print(f"\n{'='*80}")
    print("Sample prompt (first 800 chars):")
    print(f"{'='*80}")
    sample_prompt = df['prompt'].iloc[0]
    if isinstance(sample_prompt, list) and len(sample_prompt) > 0:
        sample_content = sample_prompt[0].get('content', '')[:800]
    else:
        sample_content = str(sample_prompt)[:800]
    print(sample_content)
    print("...")
    print(f"\nSample answer: {df['correct_answer'].iloc[0]}")
    print(f"Sample videos field: {df['videos'].iloc[0]}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
