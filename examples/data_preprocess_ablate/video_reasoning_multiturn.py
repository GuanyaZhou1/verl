#!/usr/bin/env python3
"""
Data preprocessing script for video reasoning multi-turn RL training (ablation version).
Supports multiple prompt formats for ablation study via --prompt_version.

Prompt versions (aligned with SFT ablation formats in ablate_reasoning_format/):
  - direct_answer:            No CoT, question + direct answer template
  - singleturn_nospatial_cot: Single-turn CoT (think tags, no grounding)
  - singleturn_spatial_cot:   Single-turn CoT + spatial grounding (bbox)
  - multiturn_nospatial:      Multi-turn with segment grounding, no spatial
  - multiturn_spatial:         Multi-turn with segment + spatial grounding (original default)

Usage:
    python video_reasoning_multiturn.py \
        --input_json /path/to/results.json \
        --video_base_path /path/to/videos \
        --output_dir /path/to/output \
        --prompt_version singleturn_nospatial_cot
"""

import json
import argparse
import sys
import random
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd
from tqdm import tqdm
import cv2


# ============================================================================
# Prompt templates for each ablation version
# (extracted from ablate_reasoning_format/ SFT data)
# ============================================================================

# --- direct_answer: No CoT ---
# MC output template
DIRECT_ANSWER_OUTPUT_MC = "Please provide only the single option (e.g., A, B, C, D, etc.) within the <answer> </answer> tags."
# Open-ended output template
DIRECT_ANSWER_OUTPUT_OE = "Please provide a clear, concise response within <answer> </answer> tags that directly addresses the question."

# --- singleturn_nospatial_cot: Think tags only ---
SINGLETURN_NOSPATIAL_COT_PROMPT = """You should reason step by step. Each reasoning step must be enclosed within '<think>' tags.

<think>
{reasoning step — analyze the question; summarize relevant findings from the video; brainstorm hypotheses; verify whether current evidence is sufficient}
</think>

When ready to provide the final answer, enclose it within '<answer>' tags:

<answer> {final answer} </answer>"""

# --- singleturn_spatial_cot: Think tags + spatial grounding ---
SINGLETURN_SPATIAL_COT_PROMPT = """You should reason step by step. Each reasoning step must be enclosed within '<think>' tags.

<think>
{reasoning step — analyze the question; summarize relevant findings from the video; brainstorm hypotheses; verify whether current evidence is sufficient}
</think>

Your reasoning should be grounded in visual spatiotemporal evidence from the video. When mentioning any objects related to the evidence, strictly follow this format:
<obj>object_name</obj><box>[x1,y1,x2,y2]</box>at<t>time_in_seconds</t>

When ready to provide the final answer, enclose it within '<answer>' tags:

<answer> {final answer} </answer>"""

# --- multiturn_nospatial: Segment grounding, no spatial ---
MULTITURN_NOSPATIAL_PROMPT = """You should reason step by step and, in EACH step, FIRST analyze and THEN focus on specific video segments. Place the grounded time segments at the END of the step.

Each reasoning step must be enclosed within '<think>' tags and reference specific time segments.

<think>
{Single reasoning step — analyze the question; summarize relevant findings from the currently available sampled input and any previously inspected segments; brainstorm hypotheses; verify whether current evidence is sufficient; refine errors; revisit prior steps if needed; if insufficient to answer, decide the NEXT most informative segments to inspect based on question intent and previously seen content}
</think>

When identifying relevant segments, use '<segment>' tags with time ranges in seconds:

<segment>
[(start1, end1), (start2, end2), ...]
</segment>

When ready to provide the final answer, enclose it within '<answer>' tags:

<answer> {final answer} </answer>"""

# --- multiturn_spatial: Segment + spatial grounding (original default) ---
MULTITURN_SPATIAL_PROMPT = """You should reason step by step and, in EACH step, FIRST analyze and THEN focus on specific video segments. Place the grounded time segments at the END of the step.

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

<answer> {final answer} </answer>"""

# MC / open-ended output templates (shared by CoT versions)
OUTPUT_TEMPLATE_MC = "Please provide only the single option (e.g., A, B, C, D, etc.) within the <answer> </answer> tags."
OUTPUT_TEMPLATE_OE = "Please provide your answer within the <answer> </answer> tags."

PROMPT_VERSION_CHOICES = [
    "direct_answer",
    "singleturn_nospatial_cot",
    "singleturn_spatial_cot",
    "multiturn_nospatial",
    "multiturn_spatial",
]


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
        description="Preprocess video reasoning data for veRL (ablation prompt versions)"
    )
    parser.add_argument(
        "--input_json", type=str, required=True,
        help="Path to the input results.json file"
    )
    parser.add_argument(
        "--video_base_path", type=str, required=True,
        help="Base path to video files"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory for parquet files"
    )
    parser.add_argument(
        "--prompt_version", type=str, required=True,
        choices=PROMPT_VERSION_CHOICES,
        help="Prompt format version for ablation study"
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.05,
        help="Ratio of data to use for validation (default: 0.05)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for train/val split"
    )
    parser.add_argument(
        "--max_samples", type=int, default=-1,
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
    duration: float,
    is_openended: bool,
    prompt_version: str,
) -> List[Dict[str, Any]]:
    """
    Create the initial prompt as a messages list for veRL.

    Prompt layout varies by prompt_version:
      - direct_answer:            <video> + Question + Options + OutputTemplate
      - singleturn_nospatial_cot: <video> + Duration + SystemPrompt + Question + Options
      - singleturn_spatial_cot:   <video> + Duration + SystemPrompt + Question + Options
      - multiturn_nospatial:      <video> + Duration + SystemPrompt + Question + Options
      - multiturn_spatial:        <video> + Duration + SystemPrompt + Question + Options
    """
    parts = []

    # Video placeholder
    parts.append("<video>\n")

    if prompt_version == "direct_answer":
        # direct_answer: no duration, no CoT instructions
        parts.append(f"Question:\n{question}\n")
        if options and not is_openended:
            parts.append(f"\n{format_options(options)}\n")
        # Output template
        if options and not is_openended:
            parts.append(DIRECT_ANSWER_OUTPUT_MC)
        else:
            parts.append(DIRECT_ANSWER_OUTPUT_OE)
        parts.append("\n")

    else:
        # All CoT versions: add duration + system prompt + question + options
        if duration:
            parts.append(f"This is a video with duration {duration:.1f} seconds.\n\n")

        # Select system prompt
        if prompt_version == "singleturn_nospatial_cot":
            system_prompt = SINGLETURN_NOSPATIAL_COT_PROMPT
        elif prompt_version == "singleturn_spatial_cot":
            system_prompt = SINGLETURN_SPATIAL_COT_PROMPT
        elif prompt_version == "multiturn_nospatial":
            system_prompt = MULTITURN_NOSPATIAL_PROMPT
        elif prompt_version == "multiturn_spatial":
            system_prompt = MULTITURN_SPATIAL_PROMPT
        else:
            raise ValueError(f"Unknown prompt_version: {prompt_version}")

        parts.append(system_prompt)
        parts.append(f"\n\nQuestion:\n{question}\n")

        if options and not is_openended:
            parts.append(f"\n{format_options(options)}\n")

    user_content = "".join(parts)
    return [{"role": "user", "content": user_content}]


def process_sample(
    sample: Dict[str, Any],
    video_base_path: str,
    prompt_version: str,
) -> Dict[str, Any]:
    """Process a single sample from results.json into veRL format."""
    video_id = sample["video_id"]
    question = sample["question"]
    correct_answer = sample["correct_answer"]
    options = sample.get("options", {})
    is_openended = sample.get("is_openended", False)

    # Construct video path
    video_path = str(Path(video_base_path) / f"{video_id}.mp4")

    # Get video duration
    duration = get_video_duration(video_path)

    # Create prompt messages
    prompt_messages = create_prompt_messages(
        question, options, duration=duration,
        is_openended=is_openended, prompt_version=prompt_version,
    )

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
        "split": sample.get("source", "train"),
        "index": sample["question_id"],
        "video_path": video_path,
        "video_duration": duration,
        "video_id": video_id,
        "question": question,
        "correct_answer": correct_answer,
        "gt_bboxes": sample.get("ground_truth_bboxes", []),
        "need_tools_kwargs": True,
        "tools_kwargs": tools_kwargs,
    }

    processed = {
        "prompt": prompt_messages,
        "videos": videos,
        "video_path": video_path,
        "video_id": video_id,
        "question_id": sample["question_id"],
        "question": question,
        "options": json.dumps(options) if options else "",
        "correct_answer": correct_answer,
        "reward_model": {"style": "rule", "ground_truth": correct_answer},
        "question_type": sample.get("question_type", "general"),
        "is_openended": sample.get("is_openended", False),
        "source": sample.get("source", "unknown"),
        "reference_reasoning": sample.get("reasoning", ""),
        "reference_segments": json.dumps(sample.get("segments", [])),
        "data_source": "video_reasoning",
        "extra_info": extra_info,
    }

    return processed


def main():
    args = parse_args()

    print(f"Prompt version: {args.prompt_version}")

    # Load input JSON
    print(f"Loading data from {args.input_json}...")
    with open(args.input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} samples")

    mc_count = sum(1 for s in data if not s.get("is_openended", False))
    oe_count = sum(1 for s in data if s.get("is_openended", False))
    print(f"  Multiple-choice: {mc_count}, Open-ended: {oe_count}")

    if args.max_samples > 0:
        data = data[:args.max_samples]
        print(f"Processing first {len(data)} samples")

    # Process each sample
    processed_data = []
    for sample in tqdm(data, desc="Processing samples"):
        try:
            processed = process_sample(sample, args.video_base_path, args.prompt_version)
            processed_data.append(processed)
        except Exception as e:
            print(f"Error processing sample {sample.get('video_id', 'unknown')}: {e}")
            continue

    print(f"Successfully processed {len(processed_data)} samples")

    # Convert to DataFrame
    df = pd.DataFrame(processed_data)

    # Split into train and val
    random.seed(args.seed)
    indices = list(range(len(df)))
    random.shuffle(indices)

    val_size = int(len(df) * args.val_ratio)
    train_indices = indices[:len(df) - val_size]
    val_indices = indices[len(df) - val_size:]

    train_df = df.iloc[train_indices].reset_index(drop=True)
    val_df = df.iloc[val_indices].reset_index(drop=True)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save parquet files
    train_file = output_dir / "train.parquet"
    train_df.to_parquet(train_file, index=False)
    print(f"\nSaved training data to {train_file}")
    print(f"  Train samples: {len(train_df)}")

    val_file = output_dir / "val.parquet"
    val_df.to_parquet(val_file, index=False)
    print(f"Saved validation data to {val_file}")
    print(f"  Val samples: {len(val_df)}")

    print(f"\nDataFrame columns: {list(df.columns)}")

    # Print statistics
    if len(df) > 0:
        print(f"\nStatistics (total):")
        print(f"  - Unique videos: {df['video_id'].nunique()}")
        print(f"  - Question types: {df['question_type'].value_counts().to_dict()}")
        print(f"  - Multiple-choice: {(~df['is_openended']).sum()}, Open-ended: {df['is_openended'].sum()}")

        print(f"\n{'='*80}")
        print(f"Sample prompt (prompt_version={args.prompt_version}):")
        print(f"{'='*80}")
        sample_prompt = df['prompt'].iloc[0]
        if isinstance(sample_prompt, list) and len(sample_prompt) > 0:
            sample_content = sample_prompt[0].get('content', '')[:1000]
        else:
            sample_content = str(sample_prompt)[:1000]
        print(sample_content)
        print("...")
        print(f"\nSample videos field:")
        print(df['videos'].iloc[0])
        print(f"{'='*80}")
    else:
        print("\nWARNING: No samples were processed successfully!")


if __name__ == "__main__":
    main()
