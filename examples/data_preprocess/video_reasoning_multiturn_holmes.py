#!/usr/bin/env python3
"""
Data preprocessing script for Video-Holmes dataset (ver4).
Converts results.json to veRL's parquet format.

This script is adapted for datasets that may lack 'source' and 'is_openended' fields.
It processes all samples with options as multiple-choice questions.

Usage:
    python video_reasoning_multiturn_holmes.py \
        --input_json /path/to/results.json \
        --video_base_path /path/to/videos \
        --output_dir /path/to/output
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
from tqdm import tqdm
import cv2


def get_video_duration(video_path: str) -> float:
    """
    Get video duration in seconds using OpenCV.
    Aligned with SFT data processing script.
    """
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


# System prompt matching the eval script format
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

OUTPUT_TEMPLATE = "Please provide only the single option (e.g., A, B, C, D, etc.) within the <answer> </answer> tags."


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess video reasoning data for veRL")
    parser.add_argument(
        "--input_json",
        type=str,
        required=True,
        help="Path to the input results.json file"
    )
    parser.add_argument(
        "--video_base_path",
        type=str,
        required=True,
        help="Base path to video files (e.g., /path/to/Video-Holmes_/video)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
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
    return parser.parse_args()


def format_options(options: Dict[str, str]) -> str:
    """Format options dictionary into text."""
    if not options:
        return ""
    return '\n'.join([f"{k}. {v}" for k, v in sorted(options.items())])


def create_prompt_messages(
    question: str,
    options: Dict[str, str],
    duration: float = None
) -> List[Dict[str, Any]]:
    """
    Create the initial prompt as a messages list for veRL.

    The prompt contains a <video> placeholder which will be replaced
    by the actual video content during data loading.

    Args:
        question: The question to answer
        options: Dictionary of answer options (e.g., {"A": "...", "B": "..."})
        duration: Duration of the video in seconds (optional)

    Returns:
        List of message dictionaries in chat format
    """
    user_content_parts = []

    # Add video placeholder (will be replaced by veRL's _build_messages)
    user_content_parts.append("<video>\n")

    # Add duration if available
    if duration:
        user_content_parts.append(f"This is a video with duration {duration:.1f} seconds.\n")

    # Add system prompt
    user_content_parts.append(MULTITURN_SYSTEM_PROMPT)

    # Add question and options (format aligned with eval script)
    options_text = format_options(options) if options else ""
    user_content_parts.append(f"\nQuestion:\n{question}\n\nOptions:\n{options_text}\n")

    # Add output template
    user_content_parts.append(OUTPUT_TEMPLATE)

    user_content = "".join(user_content_parts)

    # Return as messages list (veRL expected format)
    return [{"role": "user", "content": user_content}]


def process_sample(sample: Dict[str, Any], video_base_path: str) -> Dict[str, Any]:
    """
    Process a single sample from results.json into veRL format.

    Args:
        sample: A single data sample from results.json
        video_base_path: Base path to video files

    Returns:
        Processed sample in veRL format
    """
    video_id = sample["video_id"]
    question = sample["question"]
    correct_answer = sample["correct_answer"]
    options = sample.get("options", {})

    # Construct video path
    video_path = str(Path(video_base_path) / f"{video_id}.mp4")

    # Get video duration (aligned with SFT data format)
    duration = get_video_duration(video_path)

    # Create the prompt as messages list (veRL expected format)
    # Contains <video> placeholder that will be replaced during data loading
    prompt_messages = create_prompt_messages(question, options, duration=duration)

    # Videos field - list of video info dicts
    # veRL's _build_messages will replace <video> placeholder with this
    # NOTE: Resolution params (fps, max_frames, min_pixels, max_pixels) are NOT stored here
    # They will be injected at training time by VideoReasoningAgentLoop from config
    # This allows using the same data with different resolution settings
    videos = [{
        "video": video_path,
    }]

    # tools_kwargs for ToolAgentLoop (方案二) - nested inside extra_info
    tools_kwargs = {
        "fetch_frames": {
            "create_kwargs": {
                "video_path": video_path,
                "video_duration": duration,
            }
        }
    }

    # extra_info structure following veRL's expected format (直接存 dict，不是 JSON 字符串)
    # 参考 gsm8k_multiturn_w_tool.py
    extra_info = {
        "split": "train",  # Default to train for this dataset
        "index": sample["question_id"],  # Required by rl_dataset.py
        "video_path": video_path,
        "video_duration": duration,  # Video duration in seconds
        "video_id": video_id,
        "question": question,
        "correct_answer": correct_answer,
        # Ground truth bboxes for reward function comparison (if available)
        "gt_bboxes": sample.get("ground_truth_bboxes", []),
        # For tool-based approach (方案二)
        "need_tools_kwargs": True,
        "tools_kwargs": tools_kwargs,
    }

    processed = {
        # prompt as messages list with <video> placeholder (直接存列表，不是 JSON 字符串)
        "prompt": prompt_messages,
        # videos field for veRL to replace <video> placeholder (直接存列表)
        "videos": videos,
        "video_path": video_path,
        "video_id": video_id,
        "question_id": sample["question_id"],
        "question": question,
        "options": json.dumps(options) if options else "",  # Store as JSON string
        "correct_answer": correct_answer,
        # reward_model field for NaiveRewardManager to read ground_truth
        "reward_model": {"style": "rule", "ground_truth": correct_answer},
        "question_type": sample.get("question_type", "general"),
        "is_openended": False,  # All samples with options are multiple-choice
        "source": "train",  # Default source for this dataset
        # Store the reference reasoning for potential reward shaping
        "reference_reasoning": sample.get("reasoning", ""),
        "reference_segments": json.dumps(sample.get("segments", [])),  # Store as JSON string
        # For reward function to check correctness
        "data_source": "video_reasoning",
        # extra_info - 直接存 dict（参考 veRL 官方示例）
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

    # Count by type
    mc_count = sum(1 for s in data if s.get("options"))
    oe_count = len(data) - mc_count
    print(f"  Multiple-choice (with options): {mc_count}, Open-ended: {oe_count}")

    # Limit samples if specified
    if args.max_samples > 0:
        data = data[:args.max_samples]
        print(f"Processing first {len(data)} samples")

    # Process each sample
    processed_data = []
    for sample in tqdm(data, desc="Processing samples"):
        try:
            processed = process_sample(sample, args.video_base_path)
            processed_data.append(processed)
        except Exception as e:
            print(f"Error processing sample {sample.get('video_id', 'unknown')}: {e}")
            continue

    print(f"Successfully processed {len(processed_data)} samples")

    # Convert to DataFrame
    df = pd.DataFrame(processed_data)

    # Split into train and val
    import random
    random.seed(args.seed)

    # Shuffle indices
    indices = list(range(len(df)))
    random.shuffle(indices)

    # Calculate split point
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
    print(f"\nStatistics (total):")
    print(f"  - Unique videos: {df['video_id'].nunique()}")
    print(f"  - Question types: {df['question_type'].value_counts().to_dict()}")
    print(f"  - Multiple-choice questions: {(~df['is_openended']).sum()}")

    # Show sample prompt
    print(f"\n{'='*80}")
    print("Sample prompt:")
    print(f"{'='*80}")
    sample_prompt = df['prompt'].iloc[0]
    if isinstance(sample_prompt, list) and len(sample_prompt) > 0:
        sample_content = sample_prompt[0].get('content', '')[:800]
    else:
        sample_content = str(sample_prompt)[:800]
    print(sample_content)
    print("...")
    print(f"\nSample videos field:")
    print(df['videos'].iloc[0])
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
