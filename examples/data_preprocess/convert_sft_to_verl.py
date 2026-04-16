#!/usr/bin/env python3
"""
Convert refined SFT annotation data (conversation format) to veRL's parquet format.

SFT format: [{id, video, conversations: [{from, value}, ...], metadata?}, ...]
veRL format: parquet with columns: prompt, videos, video_path, video_id, question_id,
             question, options, correct_answer, reward_model, is_openended, data_source,
             reference_reasoning, reference_segments, extra_info, etc.

Key design: the prompt text is reused directly from conversations[0]['value'],
ensuring SFT/RL prompt consistency without depending on prompts.py.

Supports both:
  - Holmes MC (has metadata.gt_answer, options in prompt)
  - LongVideoReason open-ended (answer in last gpt turn's <answer> tag)

Usage:
    python convert_sft_to_verl.py \
        --input_json /path/to/sft.json \
        --video_base_path /path/to/videos \
        --output_dir /path/to/output \
        --dataset_name video_holmes
"""

import json
import argparse
import random
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert refined SFT annotation data to veRL parquet format"
    )
    parser.add_argument("--input_json", type=str, required=True,
                        help="Path to the SFT JSON file")
    parser.add_argument("--video_base_path", type=str, required=True,
                        help="Base directory containing video files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for parquet files")
    parser.add_argument("--dataset_name", type=str, default="video_reasoning",
                        choices=["video_holmes", "longvideo_reason", "video_reasoning"],
                        help="Dataset name for data_source tagging")
    parser.add_argument("--val_ratio", type=float, default=0.05,
                        help="Ratio of data for validation (default: 0.05)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for train/val split")
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="Max samples to process (-1 for all)")
    return parser.parse_args()


def extract_question(prompt_text: str) -> str:
    """Extract the question text from the prompt."""
    match = re.search(r'Question:\n(.+?)(?:\n\n[A-F]\.|\n\n|\Z)', prompt_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def extract_options(prompt_text: str) -> Dict[str, str]:
    """Extract MC options (A. ... B. ...) from the prompt text."""
    options = {}
    # Match option lines like "A. some text" up to next option or end
    for m in re.finditer(r'([A-F])\.\s+(.+?)(?=\n[A-F]\.\s|\n\n|\Z)', prompt_text, re.DOTALL):
        options[m.group(1)] = m.group(2).strip()
    return options


def extract_duration(prompt_text: str) -> float:
    """Extract duration from prompt text like 'duration 184.2 seconds'."""
    match = re.search(r'duration\s+([\d.]+)\s+seconds', prompt_text)
    if match:
        return float(match.group(1))
    return 0.0


def extract_answer(sample: Dict[str, Any]) -> str:
    """Extract the correct answer from the sample.

    Holmes: metadata.gt_answer contains '<answer>F</answer>'
    LongVideoReason: last gpt turn contains '<answer>...</answer>'
    """
    # Try metadata first (Holmes)
    metadata = sample.get("metadata", {})
    if metadata and "gt_answer" in metadata:
        gt = metadata["gt_answer"]
        match = re.search(r'<answer>\s*(.*?)\s*</answer>', gt, re.DOTALL)
        return match.group(1).strip() if match else gt.strip()

    # Fall back to last gpt turn (LongVideoReason)
    gpt_turns = [c for c in sample["conversations"] if c["from"] == "gpt"]
    if gpt_turns:
        last_gpt = gpt_turns[-1]["value"]
        match = re.search(r'<answer>\s*(.*?)\s*</answer>', last_gpt, re.DOTALL)
        if match:
            return match.group(1).strip()

    return ""


def extract_segments(sample: Dict[str, Any]) -> List:
    """Extract <segment> tags from all gpt turns and parse into float pairs.

    Each <segment> tag may contain a JSON list of [start, end] pairs, e.g.
    '<segment>[[28.0, 60.0], [60.0, 86.0]]</segment>'.
    Returns a flat list of [start, end] pairs.
    """
    segments = []
    for conv in sample["conversations"]:
        if conv["from"] == "gpt":
            for m in re.finditer(r'<segment>\s*(.*?)\s*</segment>', conv["value"], re.DOTALL):
                raw = m.group(1).strip()
                try:
                    parsed = json.loads(raw)
                    if isinstance(parsed, list):
                        # Could be [[start, end], ...] or [start, end]
                        if parsed and isinstance(parsed[0], list):
                            segments.extend(parsed)
                        else:
                            segments.append(parsed)
                    else:
                        segments.append(raw)
                except (json.JSONDecodeError, TypeError):
                    segments.append(raw)
    return segments


def extract_reference_reasoning(sample: Dict[str, Any]) -> str:
    """Concatenate all gpt turns as the reference reasoning."""
    gpt_turns = [c["value"] for c in sample["conversations"] if c["from"] == "gpt"]
    return "\n\n".join(gpt_turns)


def process_sample(sample: Dict[str, Any], video_base_path: str,
                   dataset_name: str, index: int) -> Optional[Dict[str, Any]]:
    """Process a single SFT sample into veRL format."""
    video_filename = sample["video"]  # e.g. "oZ4pa_5R0nY.mp4"
    video_id = video_filename.replace(".mp4", "")
    video_path = str(Path(video_base_path) / video_filename)

    # Use prompt directly from conversations[0]
    prompt_text = sample["conversations"][0]["value"]

    # Parse fields from prompt
    question = extract_question(prompt_text)
    options = extract_options(prompt_text)
    is_openended = len(options) == 0
    duration = extract_duration(prompt_text)

    # If duration not in prompt, try reading from video file
    if duration == 0.0:
        duration = get_video_duration(video_path)

    # Extract answer
    correct_answer = extract_answer(sample)

    # Extract reference info from gpt turns
    reference_reasoning = extract_reference_reasoning(sample)
    reference_segments = extract_segments(sample)

    # Question ID: try to parse numeric suffix from id, fallback to index
    sample_id = sample.get("id", "")
    id_parts = sample_id.rsplit("_", 1)
    try:
        question_id = int(id_parts[-1]) if len(id_parts) > 1 else index
    except ValueError:
        question_id = index

    # Build prompt as messages list (reuse SFT prompt as-is)
    prompt_messages = [{"role": "user", "content": prompt_text}]

    # Videos field
    videos = [{"video": video_path}]

    # tools_kwargs for ToolAgentLoop
    tools_kwargs = {
        "fetch_frames": {
            "create_kwargs": {
                "video_path": video_path,
                "video_duration": duration,
            }
        }
    }

    # extra_info
    extra_info = {
        "split": "train",
        "index": question_id,
        "video_path": video_path,
        "video_duration": duration,
        "video_id": video_id,
        "question": question,
        "correct_answer": correct_answer,
        "gt_bboxes": [],
        "need_tools_kwargs": True,
        "tools_kwargs": tools_kwargs,
    }

    processed = {
        "prompt": prompt_messages,
        "videos": videos,
        "video_path": video_path,
        "video_id": video_id,
        "question_id": question_id,
        "question": question,
        "options": json.dumps(options) if options else "",
        "correct_answer": correct_answer,
        "reward_model": {"style": "rule", "ground_truth": correct_answer},
        "question_type": "general",
        "is_openended": is_openended,
        "source": "train",
        "reference_reasoning": reference_reasoning,
        "reference_segments": reference_segments,
        "data_source": dataset_name,
        "extra_info": extra_info,
    }

    return processed


def main():
    args = parse_args()

    print(f"Loading data from {args.input_json}...")
    with open(args.input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples")

    if args.max_samples > 0:
        data = data[:args.max_samples]
        print(f"Processing first {len(data)} samples")

    # Process each sample
    processed_data = []
    for i, sample in enumerate(tqdm(data, desc="Processing samples")):
        try:
            processed = process_sample(sample, args.video_base_path,
                                       args.dataset_name, i)
            if processed:
                processed_data.append(processed)
        except Exception as e:
            print(f"Error processing sample {sample.get('id', 'unknown')}: {e}")
            continue

    print(f"Successfully processed {len(processed_data)} samples")

    if not processed_data:
        print("ERROR: No samples processed!")
        return

    # Convert to DataFrame
    df = pd.DataFrame(processed_data)

    # Train/val split
    random.seed(args.seed)
    indices = list(range(len(df)))
    random.shuffle(indices)

    val_size = int(len(df) * args.val_ratio)
    train_indices = indices[:len(df) - val_size]
    val_indices = indices[len(df) - val_size:]

    train_df = df.iloc[train_indices].reset_index(drop=True)
    val_df = df.iloc[val_indices].reset_index(drop=True)

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_file = output_dir / "train.parquet"
    train_df.to_parquet(train_file, index=False)
    print(f"\nSaved training data to {train_file}")
    print(f"  Train samples: {len(train_df)}")

    val_file = output_dir / "val.parquet"
    val_df.to_parquet(val_file, index=False)
    print(f"Saved validation data to {val_file}")
    print(f"  Val samples: {len(val_df)}")

    print(f"\nDataFrame columns: {list(df.columns)}")

    # Statistics
    print(f"\nStatistics:")
    print(f"  - Unique videos: {df['video_id'].nunique()}")
    print(f"  - Multiple-choice: {(~df['is_openended']).sum()}, Open-ended: {df['is_openended'].sum()}")
    print(f"  - Samples with answer: {(df['correct_answer'] != '').sum()}")

    # Show sample prompt
    print(f"\n{'='*80}")
    print("Sample prompt (first 500 chars):")
    print(f"{'='*80}")
    sample_prompt = df['prompt'].iloc[0]
    if isinstance(sample_prompt, list) and len(sample_prompt) > 0:
        print(sample_prompt[0].get('content', '')[:500])
    else:
        print(str(sample_prompt)[:500])
    print("...")
    print(f"\nSample correct_answer: {df['correct_answer'].iloc[0][:100]}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
