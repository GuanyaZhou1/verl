#!/usr/bin/env python3
"""
Convert video-r1 parquet data to veRL training format.

This script:
1. Loads video-r1 SFT data from parquet
2. Filters for video samples only (excludes images)
3. Randomly samples specified number of samples
4. Converts to veRL's parquet format with proper prompt structure

Usage:
    python convert_videor1.py \
        --input_parquet /path/to/longvt_sft_videor1_165k5.parquet \
        --video_base_path /path/to/videor1 \
        --output_dir ./long_video_data/videor1 \
        --max_samples 10000 \
        --val_ratio 0.05
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "video_reasoning"))
from prompts import get_prompts, MULTITURN_SYSTEM_PROMPT, OUTPUT_TEMPLATE, OUTPUT_TEMPLATE_OPENENDED


def parse_args():
    parser = argparse.ArgumentParser(description="Convert video-r1 data to veRL format")
    parser.add_argument(
        "--input_parquet",
        type=str,
        required=True,
        help="Path to video-r1 parquet file"
    )
    parser.add_argument(
        "--video_base_path",
        type=str,
        required=True,
        help="Base path to video files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for converted parquet files"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=-1,
        help="Maximum number of samples to use (-1 for all)"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.05,
        help="Ratio of data to use for validation"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling and splitting"
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default=None,
        help="Path to custom system prompt file (optional)"
    )
    return parser.parse_args()


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


def extract_video_url(messages: List[Dict]) -> Optional[str]:
    """Extract video URL from messages, return None if only images."""
    for msg in messages:
        content = msg.get('content', [])
        if content is None:
            continue
        for item in content:
            if isinstance(item, dict):
                video_url = item.get('video_url')
                if video_url and isinstance(video_url, dict):
                    return video_url.get('url')
    return None


def extract_question_and_answer(messages: List[Dict]) -> tuple:
    """Extract question text and answer from messages."""
    question = ""
    answer = ""

    for msg in messages:
        role = msg.get('role', '')
        content = msg.get('content', [])
        if content is None:
            continue

        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get('text'):
                text_parts.append(item['text'])

        full_text = ' '.join(text_parts)

        if role == 'user':
            question = full_text
        elif role == 'assistant':
            answer = full_text

    return question, answer


def extract_answer_content(answer_text: str) -> str:
    """Extract content from <answer>...</answer> tags."""
    match = re.search(r'<answer>(.*?)</answer>', answer_text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return answer_text.strip()


def is_multiple_choice(question: str) -> bool:
    """Check if question is multiple choice (has Options: section)."""
    return 'Options:' in question or 'options:' in question


def create_prompt_messages(
    question: str,
    duration: float = None,
    system_prompt: str = MULTITURN_SYSTEM_PROMPT,
) -> List[Dict[str, Any]]:
    """Create prompt messages in veRL format."""
    user_content_parts = []

    # Add video placeholder
    user_content_parts.append("<video>\n")

    # Add duration if available
    if duration and duration > 0:
        user_content_parts.append(f"This is a video with duration {duration:.1f} seconds.\n")

    # Add system prompt (multi-turn reasoning format)
    user_content_parts.append(system_prompt)

    # Add question
    user_content_parts.append(f"\nQuestion:\n{question}\n")

    # Add output template based on question type
    if is_multiple_choice(question):
        user_content_parts.append(OUTPUT_TEMPLATE)
    else:
        user_content_parts.append(OUTPUT_TEMPLATE_OPENENDED)

    user_content = "".join(user_content_parts)

    return [{"role": "user", "content": user_content}]


def process_sample(
    row: pd.Series,
    video_base_path: str,
    idx: int,
    system_prompt: str,
) -> Optional[Dict[str, Any]]:
    """Process a single video-r1 sample to veRL format."""
    messages = row['messages']
    sample_id = row.get('id', f'videor1_{idx}')

    # Extract video URL
    video_url = extract_video_url(messages)
    if not video_url:
        return None  # Skip non-video samples

    # Construct full video path
    video_path = os.path.join(video_base_path, video_url)

    # Check if video exists
    if not os.path.exists(video_path):
        print(f"Warning: Video not found: {video_path}")
        return None

    # Get video duration
    duration = get_video_duration(video_path)

    # Extract question and answer
    question, answer_text = extract_question_and_answer(messages)
    if not question:
        return None

    # Extract answer content from <answer> tags
    correct_answer = extract_answer_content(answer_text)

    # Determine if multiple choice
    is_mc = is_multiple_choice(question)

    # Create prompt messages
    prompt_messages = create_prompt_messages(question, duration, system_prompt)

    # Videos field
    videos = [{"video": video_path}]

    # Extract video_id from path
    video_id = Path(video_url).stem

    # Extra info
    extra_info = {
        "split": "train",
        "index": idx,
        "video_path": video_path,
        "video_duration": duration,
        "video_id": video_id,
        "question": question,
        "correct_answer": correct_answer,
        "original_id": sample_id,
        "need_tools_kwargs": True,
        "tools_kwargs": {
            "fetch_frames": {
                "create_kwargs": {
                    "video_path": video_path,
                    "video_duration": duration,
                }
            }
        },
    }

    # Build processed sample
    processed = {
        "prompt": prompt_messages,
        "videos": videos,
        "video_path": video_path,
        "video_id": video_id,
        "question_id": idx,
        "question": question,
        "options": "",  # Options are embedded in question text
        "correct_answer": correct_answer,
        "reward_model": {"style": "rule", "ground_truth": correct_answer},
        "question_type": "multiple_choice" if is_mc else "open_ended",
        "is_openended": not is_mc,
        "source": "videor1",
        "reference_reasoning": answer_text,  # Store full reasoning
        "reference_segments": "[]",
        "data_source": "videor1",
        "extra_info": extra_info,
    }

    return processed


def main():
    args = parse_args()

    # Load custom prompts if specified
    system_prompt = MULTITURN_SYSTEM_PROMPT
    if args.prompt_file:
        system_prompt, _, _ = get_prompts(prompt_file=args.prompt_file)
        print(f"Loaded custom system prompt from: {args.prompt_file}")

    # Load input parquet
    print(f"Loading data from {args.input_parquet}...")
    df = pd.read_parquet(args.input_parquet)
    print(f"Loaded {len(df)} total samples")

    # Filter for video samples only
    print("Filtering for video samples...")
    video_indices = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Scanning"):
        if extract_video_url(row['messages']):
            video_indices.append(idx)

    df_video = df.iloc[video_indices].reset_index(drop=True)
    print(f"Found {len(df_video)} video samples (excluded {len(df) - len(df_video)} image samples)")

    # Random sampling if max_samples specified
    np.random.seed(args.seed)
    if args.max_samples > 0 and args.max_samples < len(df_video):
        indices = np.random.choice(len(df_video), size=args.max_samples, replace=False)
        df_video = df_video.iloc[indices].reset_index(drop=True)
        print(f"Randomly sampled {args.max_samples} samples")

    # Process samples
    print(f"Processing {len(df_video)} samples...")
    processed_data = []
    for idx, row in tqdm(df_video.iterrows(), total=len(df_video), desc="Converting"):
        try:
            processed = process_sample(row, args.video_base_path, idx, system_prompt)
            if processed:
                processed_data.append(processed)
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue

    print(f"Successfully processed {len(processed_data)} samples")

    if len(processed_data) == 0:
        print("ERROR: No samples processed!")
        sys.exit(1)

    # Convert to DataFrame
    df_out = pd.DataFrame(processed_data)

    # Split into train and val
    np.random.seed(args.seed)
    n_val = int(len(df_out) * args.val_ratio)
    indices = np.random.permutation(len(df_out))
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    df_train = df_out.iloc[train_indices].reset_index(drop=True)
    df_val = df_out.iloc[val_indices].reset_index(drop=True)

    print(f"Train samples: {len(df_train)}, Val samples: {len(df_val)}")

    # Save to parquet
    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "train.parquet")
    val_path = os.path.join(args.output_dir, "val.parquet")

    df_train.to_parquet(train_path, index=False)
    df_val.to_parquet(val_path, index=False)

    print(f"\nSaved to:")
    print(f"  Train: {train_path}")
    print(f"  Val: {val_path}")

    # Print sample info
    print(f"\nSample columns: {list(df_train.columns)}")
    print(f"Sample correct_answer: {df_train['correct_answer'].iloc[0][:100]}...")


if __name__ == "__main__":
    main()
