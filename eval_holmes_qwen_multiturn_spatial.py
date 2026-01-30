#!/usr/bin/env python3
"""
Multi-turn Evaluation script for Video-Holmes dataset
Supports iterative reasoning with video segment grounding
"""

import os
import json
import argparse
import torch
import re
import itertools
import subprocess
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from tqdm import tqdm
from functools import partial
from datetime import timedelta
from torch.utils.data import Dataset

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from vision_process import process_vision_info


# System prompts
SYSTEM_PROMPT = """
You are a helpful assistant.
Please think about this question as if you were a human pondering deeply. Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions. It's encouraged to include self reflection or verification in the reasoning process. Provide your detailed reasoning between the <think> and </think> tags, and then give your final answer between the <answer> and </answer> tags.
"""

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

def extract_option_letter(answer: str) -> str:
    """
    Extract the option letter from an answer.
    Handles formats like:
    - "B"
    - "B."
    - "B. Some explanation text"
    - "Option B"
    """
    answer = answer.strip()

    # Try to find a single letter option (A, B, C, D, etc.)
    # Match pattern: optional "Option" + letter + optional period/text
    match = re.match(r'^(?:Option\s+)?([A-Z])(?:\.|:|$|\s)', answer, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # If no pattern matched, check if the answer is just a single letter
    if len(answer) == 1 and answer.isalpha():
        return answer.upper()

    # Return the original answer if no option letter found
    return answer.strip().upper()


def extract_answer(text: str) -> Optional[str]:
    """
    Extract answer from text in format <answer>...</answer>

    Args:
        text: Model output text

    Returns:
        Extracted answer content (stripped) or None
    """
    # Match <answer>...</answer> pattern and extract everything inside
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    return None


def extract_segments(text: str) -> List[Tuple[float, float]]:
    """
    Extract time segments from text in format <segment>[(start, end), ...]</segment>

    Args:
        text: Model output text

    Returns:
        List of (start, end) tuples
    """
    # Match <segment>[(...)...]</segment> pattern
    match = re.search(r'<segment>\s*\[(.*?)\]\s*</segment>', text, re.DOTALL | re.IGNORECASE)
    if not match:
        return []

    segment_str = match.group(1)

    # Extract all (float, float) pairs
    segments = []
    # pattern = r'\(\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\)'
    pattern = r'[\(\[]\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*[\)\]]'
    for match in re.finditer(pattern, segment_str):
        start = float(match.group(1))
        end = float(match.group(2))
        segments.append((start, end))

    return segments


def get_video_duration(video_path: str) -> Optional[float]:
    """
    Get video duration using ffprobe

    Args:
        video_path: Path to video file

    Returns:
        Duration in seconds, or None if failed
    """
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
        duration = float(result.stdout.decode().strip())
        return duration
    except Exception as e:
        print(f"[WARNING] Failed to get video duration for {video_path}: {e}")
        return None


class VideoHolmesDataset(Dataset):
    """Dataset for loading Video-Holmes data"""

    def __init__(self, data_path: str, video_base_path: str, processor,
                 nframes: int = 512, max_pixels: int = 50176,
                 segment_nframes: int = 64, segment_max_pixels: int = 50176,
                 video_duration_path: Optional[str] = None):
        super(VideoHolmesDataset, self).__init__()
        self.processor = processor
        self.video_base_path = video_base_path
        self.nframes = nframes
        self.max_pixels = max_pixels
        self.segment_nframes = segment_nframes
        self.segment_max_pixels = segment_max_pixels

        # Load JSON data
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)[1000:]

        # Load video durations if provided
        self.video_durations = {}
        if video_duration_path and os.path.exists(video_duration_path):
            with open(video_duration_path, 'r', encoding='utf-8') as f:
                duration_data = json.load(f)
                for item in duration_data:
                    self.video_durations[item['video_id']] = item['duration']

        print(f"Loaded {len(self.data)} samples from {data_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        entry = self.data[index]

        video_id = entry['video ID']
        question_id = entry['Question ID']
        question_type = entry['Question Type']
        question = entry['Question']
        options_dict = entry['Options']
        correct_answer = entry['Answer']

        # Resolve video path
        video_path = os.path.join(self.video_base_path, f"{video_id}.mp4")

        # Get video duration
        duration = self.video_durations.get(video_id, None)

        # If duration not in cache, read from video file
        if duration is None:
            duration = get_video_duration(video_path)
            # Cache the result
            if duration is not None:
                self.video_durations[video_id] = duration

        # Format options as text
        options_text = '\n'.join([f"{k}. {v}" for k, v in sorted(options_dict.items())])

        # Build initial user message
        user_content_parts = []

        # Add duration if available
        if duration:
            user_content_parts.append(f"This is a video with duration {duration:.1f} seconds.\n")

        # Add system prompt
        user_content_parts.append(MULTITURN_SYSTEM_PROMPT)

        # Add question
        user_content_parts.append(f"\nQuestion:\n{question}\n\nOptions:\n{options_text}\n")
        # Add post prompt
        user_content_parts.append(OUTPUT_TEMPLATE)

        user_content = "".join(user_content_parts)

        return {
            "video_id": video_id,
            "question_id": question_id,
            "question_type": question_type,
            "gt_answer": correct_answer,
            "video_path": video_path,
            "question": question,
            "user_content": user_content,
            "duration": duration,
        }


class InferenceSampler(torch.utils.data.sampler.Sampler):
    """Sampler for distributed inference"""

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


def collate_fn(batches):
    """Collate function for DataLoader - just return batches as-is for multi-turn"""
    return batches


def generate_response(model, processor, messages: List[Dict], max_new_tokens: int = 1024,
                     continue_final_message: bool = False, debug: bool = False) -> str:
    """
    Generate response for given messages

    Args:
        model: The model
        processor: The processor
        messages: List of message dicts
        max_new_tokens: Max tokens to generate
        continue_final_message: If True, continue from the last assistant message
        debug: If True, print debug information

    Returns:
        Generated text
    """
    if debug:
        print("\n" + "="*80)
        print("DEBUG: Messages before apply_chat_template")
        print("="*80)
        for i, msg in enumerate(messages):
            print(f"\nMessage {i}:")
            print(f"  Role: {msg['role']}")
            if isinstance(msg.get('content'), list):
                print(f"  Content (list with {len(msg['content'])} items):")
                for j, item in enumerate(msg['content']):
                    if item.get('type') == 'text':
                        print(f"    [{j}] text: {item['text'][:100]}...")
                    elif item.get('type') == 'video':
                        print(f"    [{j}] video: {item.get('video', 'N/A')}")
                        if 'video_start' in item:
                            print(f"         start: {item['video_start']}, end: {item['video_end']}")
                    elif item.get('type') == 'image':
                        print(f"    [{j}] image")
            else:
                content_str = str(msg.get('content', ''))
                print(f"  Content: {content_str[:200]}...")

    # Process messages
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=not continue_final_message,
        continue_final_message=continue_final_message
    )

    if debug:
        print("\n" + "="*80)
        print("DEBUG: Text after apply_chat_template")
        print("="*80)
        print(text)
        print("="*80 + "\n")

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    # Move to device
    inputs = inputs.to(model.device)

    # Generate
    generated_ids = model.generate(
        **inputs,
        use_cache=True,
        do_sample=False,
        max_new_tokens=max_new_tokens
    )

    # Decode output
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    return output_text


def multiturn_inference(
    model,
    processor,
    video_path: str,
    user_content: str,
    duration: float,
    max_steps: int,
    nframes: int,
    max_pixels: int,
    segment_nframes: int,
    segment_max_pixels: int,
    max_new_tokens: int = 1024,
    debug: bool = False,
) -> Tuple[str, List[Dict]]:
    """
    Perform multi-turn inference with video segment grounding

    Args:
        model: The model
        processor: The processor
        video_path: Path to video file
        user_content: Initial user message content
        max_steps: Maximum number of reasoning steps
        nframes: Number of frames for full video
        max_pixels: Max pixels for full video
        segment_nframes: Number of frames for segments
        segment_max_pixels: Max pixels for segments
        max_new_tokens: Max tokens per generation
        debug: If True, print debug information

    Returns:
        Tuple of (final_answer, conversation_history)
    """
    conversation_history = []
    final_answer = None

    # Step 1: Initial reasoning with full video
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "fps": 1,
                    "max_frames": nframes,
                    "max_pixels": max_pixels,
                },
                {"type": "text", "text": user_content},
            ],
        }
    ]
    conversation_history.append({
        "step": 0,
        "role": "user",
        "content": user_content,
    })

    if debug:
        print(f"\n{'='*80}")
        print("DEBUG: Initial message constructed")
        print(f"{'='*80}")
        print(f"Video path: {video_path}")
        print(f"Video nframes: {nframes}, max_pixels: {max_pixels}")
        print(f"User content (first 200 chars): {user_content[:200]}...")
        print(f"{'='*80}\n")

    for step in range(max_steps):
        if debug:
            print(f"\n{'='*80}")
            print(f"STEP {step + 1}/{max_steps}")
            print(f"{'='*80}")

        # Check if this is the last step (force answer)
        is_last_step = (step == max_steps - 1)

        # If last step and no answer in response, force answer tag
        if is_last_step:
            # Insert answer tag at the end
            response = "<think>Based on all the information I've gathered, I'll now provide my final answer.</think>\n<answer>"

            # Continue generation to get the actual answer
            messages_with_response = messages + [{"role": "assistant", "content": response}]
            forced_answer = generate_response(
                model, processor, messages_with_response,
                max_new_tokens=100,
                continue_final_message=True,
                debug=debug
            )

            # Combine
            response = response + forced_answer
        else:
            # Generate response
            response = generate_response(model, processor, messages, max_new_tokens, debug=debug)

        if debug:
            print(f"\nDEBUG: Model response (first 500 chars):")
            print(response[:500])
            if len(response) > 500:
                print("...")
            print()

        # Add assistant response to messages
        messages.append({"role": "assistant", "content": response})

        # Record in history
        conversation_history.append({
            "step": step + 1,
            "role": "assistant",
            "content": response,
        })

        # Check if answer is provided
        answer = extract_answer(response)
        if answer:
            final_answer = extract_option_letter(answer)
            break

        # If not last step, check for segments
        if not is_last_step:
            segments = extract_segments(response)

            if segments:
                if debug:
                    print(f"\nDEBUG: Extracted {len(segments)} segments: {segments}")

                # Build observation message with segment videos
                content_list = []

                # Add opening text
                content_list.append({
                    "type": "text",
                    "text": "<observation>Here are the cropped video segments."
                })

                # Add each segment as text + video pair
                for start, end in segments:
                    if start >= end:
                        end = start + 2.0  # Ensure non-zero duration
                    if end >= duration:
                        start = min(start, duration-2.0)
                        end = duration
                    content_list.append({
                        "type": "text",
                        "text": f"\nFrom {start}s to {end}s:"
                    })
                    content_list.append({
                        "type": "video",
                        "video": video_path,
                        "video_start": start,
                        "video_end": end,
                        "fps": 1,
                        "max_frames": segment_nframes,
                        "max_pixels": segment_max_pixels,
                    })

                # Add closing text
                content_list.append({
                    "type": "text",
                    "text": "\n</observation>"
                })

                # Add observation message
                observation_message = {
                    "role": "user",
                    "content": content_list,
                }

                if debug:
                    print(f"\nDEBUG: Adding observation message with {len(content_list)} content items")

                messages.append(observation_message)

                # Build observation text for history (without video tags)
                observation_text = "Here are the cropped video segments."
                for start, end in segments:
                    observation_text += f"\nFrom {start}s to {end}s"

                conversation_history.append({
                    "step": step + 1,
                    "role": "user",
                    "content": observation_text,
                    "segments": segments,
                })
            else:
                # No segments provided, stop
                break
        else:
            # Last step reached
            break

    return final_answer, conversation_history


def evaluate_predictions(predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluate predictions and compute accuracy by Question Type

    Args:
        predictions: List of prediction dicts

    Returns:
        Dict with evaluation metrics
    """
    total = len(predictions)
    correct = 0

    # Statistics by Question Type
    question_type_stats = {}

    for pred in predictions:
        question_type = pred.get('question_type', 'Unknown')
        gt_answer = pred['gt_answer']
        pred_answer = pred['pred_answer']

        # Initialize stats
        if question_type not in question_type_stats:
            question_type_stats[question_type] = {'total': 0, 'correct': 0}

        question_type_stats[question_type]['total'] += 1

        # Check correctness
        is_correct = pred_answer and pred_answer == gt_answer
        if is_correct:
            correct += 1
            question_type_stats[question_type]['correct'] += 1

    # Compute overall accuracy
    accuracy = correct / total if total > 0 else 0.0

    # Compute per-Question Type accuracy
    question_type_accuracies = {}
    for qtype, stats in question_type_stats.items():
        question_type_accuracies[qtype] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0

    return {
        'total': total,
        'correct': correct,
        'accuracy': accuracy,
        'question_type_stats': question_type_stats,
        'question_type_accuracies': question_type_accuracies,
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-turn Evaluation for Video-Holmes dataset")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model")
    parser.add_argument("--data-path", type=str,
                       default="/home/zhuyousong/zhengsr/dataset/Video-Holmes_/test_Video-Holmes.json",
                       help="Path to test_Video-Holmes.json file")
    parser.add_argument("--video-base-path", type=str,
                       default="/home/zhuyousong/zhengsr/dataset/Video-Holmes_/video",
                       help="Base path for video files")
    parser.add_argument("--video-duration-path", type=str, default=None,
                       help="Path to video duration JSON file")
    parser.add_argument("--output-path", type=str, default="./eval_output_holmes_multiturn",
                       help="Output directory")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (only 1 supported for multi-turn)")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of workers")
    parser.add_argument("--nframes", type=int, default=512, help="Number of frames for full video")
    parser.add_argument("--max-pixels", type=int, default=12544, help="Max pixels for full video")
    parser.add_argument("--segment-nframes", type=int, default=32, help="Number of frames for segments")
    parser.add_argument("--segment-max-pixels", type=int, default=50176, help="Max pixels for segments")
    parser.add_argument("--init", type=str, default="tcp://127.0.0.1:12457", help="Distributed init method")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="Max new tokens to generate")
    parser.add_argument("--max-steps", type=int, default=2, help="Maximum number of reasoning steps")
    parser.add_argument("--debug", action="store_true", help="Enable debug output for messages and templates")

    args = parser.parse_args()

    # Force batch size to 1 for multi-turn
    if args.batch_size != 1:
        print(f"Warning: batch_size must be 1 for multi-turn inference, setting to 1")
        args.batch_size = 1

    # Environment initialization
    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
        init_method=str(args.init),
        timeout=timedelta(minutes=600)
    )
    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))

    # Load model
    print(f"Loading model from: {args.model_path}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        use_fast=True,
        min_pixels=28*28,
        max_pixels=12800*28*28
    )

    # Create dataset
    dataset = VideoHolmesDataset(
        data_path=args.data_path,
        video_base_path=args.video_base_path,
        processor=processor,
        nframes=args.nframes,
        max_pixels=args.max_pixels,
        segment_nframes=args.segment_nframes,
        segment_max_pixels=args.segment_max_pixels,
        video_duration_path=args.video_duration_path,
    )

    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        sampler=InferenceSampler(len(dataset)),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False,
        collate_fn=collate_fn
    )

    # Output directory
    model_name = os.path.basename(args.model_path)
    output_dir = os.path.join(args.output_path, model_name)
    os.makedirs(output_dir, exist_ok=True)

    # Check if predictions file already exists
    pred_path = os.path.join(output_dir, f"predictions_step{args.max_steps}_1000_.json")
    stats_path = os.path.join(output_dir, f"inference_stats_step{args.max_steps}.json")

    if os.path.exists(pred_path):
        # Load existing predictions
        print(f"Found existing predictions at: {pred_path}")
        print("Skipping inference, loading predictions...")

        with open(pred_path, 'r', encoding='utf-8') as f:
            merged_predictions = json.load(f)

        if os.path.exists(stats_path):
            with open(stats_path, 'r', encoding='utf-8') as f:
                merged_stats = json.load(f)
        else:
            merged_stats = []

        print(f"Loaded {len(merged_predictions)} predictions from file")
    else:
        # Run inference
        predictions = []
        inference_stats = []

        print(f"Starting multi-turn inference (max_steps={args.max_steps})...")
        with torch.inference_mode():
            for batch in tqdm(dataloader, desc="Evaluating"):
                # Batch size is 1
                sample = batch[0]

                import time
                torch.cuda.reset_peak_memory_stats()
                start_time = time.time()

                # Multi-turn inference
                final_answer, conversation_history = multiturn_inference(
                    model=model,
                    processor=processor,
                    video_path=sample["video_path"],
                    user_content=sample["user_content"],
                    duration=sample["duration"],
                    max_steps=args.max_steps,
                    nframes=args.nframes,
                    max_pixels=args.max_pixels,
                    segment_nframes=args.segment_nframes,
                    segment_max_pixels=args.segment_max_pixels,
                    max_new_tokens=args.max_new_tokens,
                    debug=args.debug,
                )

                # Record inference time and memory
                torch.cuda.synchronize()
                end_time = time.time()
                infer_time = end_time - start_time
                cur_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB

                predictions.append({
                    "video_id": sample["video_id"],
                    "question_id": sample["question_id"],
                    "question_type": sample["question_type"],
                    "question": sample["question"],
                    "gt_answer": sample["gt_answer"],
                    "pred_answer": final_answer,
                    "conversation_history": conversation_history,
                    "video_path": sample["video_path"],
                    "num_turns": len(conversation_history),
                })

                inference_stats.append({
                    "video_id": sample["video_id"],
                    "question_id": sample["question_id"],
                    "infer_time_sec": round(infer_time, 4),
                    "max_memory_MB": round(cur_memory, 2),
                    "num_turns": len(conversation_history),
                })

        # Gather results from all processes
        torch.distributed.barrier()
        world_size = torch.distributed.get_world_size()

        merged_predictions = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(merged_predictions, predictions)
        merged_predictions = list(itertools.chain.from_iterable(merged_predictions))

        merged_stats = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(merged_stats, inference_stats)
        merged_stats = list(itertools.chain.from_iterable(merged_stats))

    # Evaluate on rank 0
    if torch.distributed.get_rank() == 0:
        print("\n" + "="*80)
        print(f"EVALUATION RESULTS - Video-Holmes (Multi-turn, max_steps={args.max_steps})")
        print("="*80)

        # Save predictions and stats
        if not os.path.exists(pred_path) or 'predictions' in locals():
            # Save predictions
            with open(pred_path, 'w', encoding='utf-8') as f:
                json.dump(merged_predictions, f, indent=2, ensure_ascii=False)
            print(f"\nSaved predictions to: {pred_path}")

            # Save inference stats
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(merged_stats, f, indent=4, ensure_ascii=False)
            print(f"Saved inference stats to: {stats_path}")
        else:
            print(f"\nUsing existing predictions from: {pred_path}")

        # Evaluate
        results = evaluate_predictions(merged_predictions)

        # Print overall results
        print(f"\nOverall Accuracy: {results['accuracy']:.4f} ({results['correct']}/{results['total']})")

        # Print per-Question Type results
        print("\n" + "-"*80)
        print("Accuracy by Question Type:")
        print("-"*80)
        for qtype in sorted(results['question_type_accuracies'].keys()):
            acc = results['question_type_accuracies'][qtype]
            stats = results['question_type_stats'][qtype]
            print(f"  {qtype:20s}: {acc:.4f} ({stats['correct']:4d}/{stats['total']:4d})")

        # Print inference statistics
        total_samples = len(merged_stats)
        total_infer_time = sum([s["infer_time_sec"] for s in merged_stats])
        avg_infer_time = total_infer_time / total_samples if total_samples > 0 else 0
        throughput = total_samples / total_infer_time if total_infer_time > 0 else 0
        max_mem_MB = max([s["max_memory_MB"] for s in merged_stats], default=0)
        avg_turns = sum([s["num_turns"] for s in merged_stats]) / total_samples if total_samples > 0 else 0

        print("\n" + "-"*80)
        print("Inference Statistics:")
        print("-"*80)
        print(f"  Total samples: {total_samples}")
        print(f"  Max steps: {args.max_steps}")
        print(f"  Average turns: {avg_turns:.2f}")
        print(f"  Total time: {total_infer_time:.2f} sec")
        print(f"  Average time per sample: {avg_infer_time:.4f} sec")
        print(f"  Throughput: {throughput:.2f} samples/sec")
        print(f"  Max GPU memory: {max_mem_MB:.2f} MB")

        # Save evaluation results
        results_path = os.path.join(output_dir, f"eval_results_step{args.max_steps}_1000_.json")
        eval_summary = {
            "max_steps": args.max_steps,
            "accuracy": results['accuracy'],
            "total": results['total'],
            "correct": results['correct'],
            "question_type_accuracies": results['question_type_accuracies'],
            "question_type_stats": results['question_type_stats'],
            "inference_stats": {
                "total_samples": total_samples,
                "average_turns": avg_turns,
                "total_time_sec": total_infer_time,
                "avg_time_per_sample_sec": avg_infer_time,
                "throughput_samples_per_sec": throughput,
                "max_memory_MB": max_mem_MB,
            }
        }
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(eval_summary, f, indent=2, ensure_ascii=False)
        print(f"\nSaved evaluation results to: {results_path}")

        print("\n" + "="*80)
        print("Done!")
        print("="*80)

    torch.distributed.barrier()


if __name__ == "__main__":
    main()
