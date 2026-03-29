#!/usr/bin/env python3
"""
Video Reasoning SFT Model Sampling & Difficulty Classification

对 SFT 模型在 RL 训练数据上进行 n_rollout=8 的多轮视频推理采样，
计算各项奖励 (accuracy, format, segment)，并根据 accuracy 进行难度分级。

Rollout 逻辑与 video_reasoning_agent_loop.py 保持一致：
  - 多轮对话：模型生成 -> 提取 <segment> -> 加载帧 -> 构建 observation -> 继续生成
  - 直到找到 <answer> 或达到最大轮数

奖励计算与 video_reasoning_async.py 保持一致：
  - acc_score: 规则匹配答案选项字母 (A/B/C/D)
  - format_score: 多轮格式验证
  - segment_score: 时间段匹配分数 (IoU/IoP/IoG)

Usage:
    # 使用 shell 脚本 (推荐)
    bash examples/video_reasoning/run_sample_and_classify.sh

    # 或直接运行
    python examples/video_reasoning/sample_and_classify.py \\
        --model_path /path/to/sft_model \\
        --data_path ./long_video_data/longvt_selfqa/train.parquet \\
        --output_dir ./sampling_results
"""

import os
import sys
import copy
import json
import re
import argparse
import logging
import time
import importlib.util
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any
from collections import defaultdict

import pandas as pd
import numpy as np
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ==============================================================================
# Direct file imports to avoid verl package init dependency chain
# ==============================================================================
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))


def _import_from_file(module_name: str, file_path: str):
    """Import a module directly from file path, bypassing package __init__."""
    abs_path = os.path.join(_REPO_ROOT, file_path) if not os.path.isabs(file_path) else file_path
    spec = importlib.util.spec_from_file_location(module_name, abs_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Import reward functions directly from file
_reward_mod = _import_from_file(
    "video_reasoning_async",
    "verl/utils/reward_score/video_reasoning_async.py",
)
_format_reward = _reward_mod.format_reward
_compute_segment_score = _reward_mod.compute_segment_score

# Import VideoFrameCache directly from file
_cache_mod = _import_from_file(
    "video_frame_cache",
    "verl/utils/video_frame_cache.py",
)
VideoFrameCache = _cache_mod.VideoFrameCache


# ==============================================================================
# Utility functions (aligned with agent loop & reward module)
# ==============================================================================

def extract_segments(text: str) -> List[Tuple[float, float]]:
    """Extract time segments from <segment>[(start, end), ...]</segment> tags."""
    match = re.search(r"<segment>\s*\[(.*?)\]\s*</segment>", text, re.DOTALL | re.IGNORECASE)
    if not match:
        return []
    segment_str = match.group(1)
    segments = []
    pattern = r"[\(\[]\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*[\)\]]"
    for m in re.finditer(pattern, segment_str):
        start = float(m.group(1))
        end = float(m.group(2))
        segments.append((start, end))
    return segments


def extract_answer(text: str) -> Optional[str]:
    """Extract answer from <answer>...</answer> tags."""
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def extract_option_letter(text: str) -> Optional[str]:
    """Extract option letter (A/B/C/D) from text."""
    if not text:
        return None
    text_upper = text.strip().upper()
    if len(text_upper) == 1 and text_upper in "ABCD":
        return text_upper
    match = re.search(r"\b([A-D])\b", text_upper)
    return match.group(1) if match else None


# ==============================================================================
# Rollout state tracker
# ==============================================================================

class RolloutState:
    """Track state for a single multi-turn rollout."""

    def __init__(self, sample_idx: int, rollout_idx: int, messages: list,
                 video_path: str, video_duration: float):
        self.sample_idx = sample_idx
        self.rollout_idx = rollout_idx
        self.messages = copy.deepcopy(messages)
        self.video_path = video_path
        self.video_duration = video_duration
        self.responses: List[str] = []
        self.all_segments: List[List[Tuple[float, float]]] = []
        self.done = False
        self.answer: Optional[str] = None
        self.num_turns = 0
        self.full_response = ""
        self.error: Optional[str] = None


# ==============================================================================
# Message preparation (aligned with agent loop)
# ==============================================================================

def inject_video_params(messages: list, fps: int, max_frames: int,
                        min_pixels: int, max_pixels: int) -> list:
    """Inject video parameters into messages containing video content."""
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "video":
                    item.setdefault("fps", fps)
                    item.setdefault("max_frames", max_frames)
                    item.setdefault("min_pixels", min_pixels)
                    item.setdefault("max_pixels", max_pixels)
    return messages


def replace_video_with_cached_frames(messages: list, frame_paths: list) -> list:
    """Replace video path in messages with cached frame paths."""
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "video":
                    item["video"] = frame_paths
    return messages


def prepare_initial_messages(row, frame_cache, config: dict) -> list:
    """Prepare initial messages for a sample (aligned with agent loop __init__ + run)."""
    prompt = row["prompt"]
    if isinstance(prompt, np.ndarray):
        prompt = prompt.tolist()
    messages = copy.deepcopy(list(prompt))

    extra_info = row.get("extra_info", {})
    if isinstance(extra_info, str):
        extra_info = json.loads(extra_info)
    video_path = row.get("video_path") or extra_info.get("video_path")

    # Inject initial video params (aligned with agent loop _inject_video_params)
    inject_video_params(
        messages,
        fps=config["initial_fps"],
        max_frames=config["initial_max_frames"],
        min_pixels=config["initial_min_pixels"],
        max_pixels=config["initial_max_pixels"],
    )

    # Replace video with cached frames if enabled (aligned with agent loop use_cached_initial_video)
    if config["use_cached_initial_video"] and video_path and frame_cache is not None:
        try:
            frame_paths = frame_cache.load_frame_paths(
                video_path,
                segments=None,
                max_frames_per_segment=config["initial_max_frames"],
            )
            if frame_paths:
                replace_video_with_cached_frames(messages, frame_paths)
                logger.debug(f"Using {len(frame_paths)} cached frames for {video_path}")
        except Exception as e:
            logger.debug(f"Cache miss for {video_path}: {e}")

    return messages


def build_observation_message(frame_cache, video_path: str,
                              segments: List[Tuple[float, float]],
                              video_duration: float, config: dict) -> Tuple[dict, int]:
    """
    Build observation message with segment frames.
    Aligned with agent loop _run_impl observation construction.
    """
    content_list = [
        {"type": "text", "text": "<observation>Here are the cropped video segments."}
    ]

    total_frames = 0
    for start, end in segments:
        # Boundary check (aligned with agent loop)
        if start >= end:
            end = start + 2.0
        if video_duration and end >= video_duration:
            start = min(start, video_duration - 2.0)
            end = video_duration

        try:
            seg_frames = frame_cache.load_frame_paths(
                video_path,
                [(start, end)],
                max_frames_per_segment=config["max_frames_per_segment"],
            )
        except Exception:
            seg_frames = []

        if not seg_frames:
            continue

        total_frames += len(seg_frames)
        content_list.append({"type": "text", "text": f"\nFrom {start}s to {end}s:"})
        content_list.append({
            "type": "video",
            "video": seg_frames,
            "fps": config["segment_fps"],
            "max_frames": config["segment_max_frames"],
            "min_pixels": config["segment_min_pixels"],
            "max_pixels": config["segment_max_pixels"],
        })

    content_list.append({"type": "text", "text": "\n</observation>"})

    return {"role": "user", "content": content_list}, total_frames


# ==============================================================================
# vLLM inference helpers
# ==============================================================================

def prepare_vllm_input(messages: list, processor):
    """
    Convert chat messages to vLLM-compatible input.

    Uses qwen_vl_utils.process_vision_info to extract multi-modal data,
    then processor.apply_chat_template for text, letting vLLM handle
    tokenization and multi-modal processing consistently.
    """
    from qwen_vl_utils import process_vision_info

    # Extract vision data
    # Returns: (images, videos, video_kwargs) - 3 values
    result = process_vision_info(messages)
    image_inputs = result[0]
    video_inputs = result[1]
    # video_kwargs (result[2]) contains extra kwargs for processor, not needed here

    # Get text prompt with chat template
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Build multi_modal_data for vLLM
    mm_data = {}
    if video_inputs:
        mm_data["video"] = video_inputs
    if image_inputs:
        mm_data["image"] = image_inputs

    return {
        "prompt": text,
        "multi_modal_data": mm_data if mm_data else None,
    }


# ==============================================================================
# Reward computation (aligned with video_reasoning_async.py)
# ==============================================================================

def compute_rewards(state: RolloutState, correct_answer: str,
                    gt_segments: list, config: dict) -> dict:
    """Compute all reward components for a completed rollout."""

    # 1. Accuracy: rule-based option letter matching
    predicted = extract_option_letter(state.answer) if state.answer else None
    correct = extract_option_letter(correct_answer)
    acc_score = 1.0 if (predicted and correct and predicted == correct) else 0.0

    # 2. Format reward (from video_reasoning_async.py, imported at module level)
    fmt_score = 0.0
    try:
        fmt_score = _format_reward(
            state.full_response,
            strict_segment=config.get("use_strict_format", True),
        )
    except Exception:
        # Fallback: basic check
        fmt_score = 1.0 if state.answer is not None else 0.0

    # 3. Segment score (from video_reasoning_async.py, imported at module level)
    seg_score = seg_iou = seg_iop = seg_iog = 0.0
    if gt_segments and state.all_segments:
        try:
            seg_score, seg_iou, seg_iop, seg_iog = _compute_segment_score(
                state.all_segments, gt_segments
            )
        except Exception:
            pass

    # 4. Weighted total (matching training config weights)
    w_ans = config["answer_weight"]
    w_fmt = config["format_weight"]
    w_seg = config["segment_weight"]
    total_w = w_ans + w_fmt + w_seg
    total_score = (
        (w_ans * acc_score + w_fmt * fmt_score + w_seg * seg_score) / total_w
        if total_w > 0
        else 0.0
    )

    return {
        "acc_score": acc_score,
        "predicted_letter": predicted,
        "correct_letter": correct,
        "format_score": fmt_score,
        "segment_score": seg_score,
        "segment_iou": seg_iou,
        "segment_iop": seg_iop,
        "segment_iog": seg_iog,
        "total_score": total_score,
    }


# ==============================================================================
# Difficulty classification
# ==============================================================================

def classify_difficulty(n_correct: int, n_rollouts: int) -> str:
    """
    Classify question difficulty based on accuracy across rollouts.

    Levels:
      - easy:        全对 (n_correct == n_rollouts)
      - medium_easy: 大部分对 (>= 62.5%)
      - medium:      一半左右 (>= 25%)
      - medium_hard: 少数对 (> 0)
      - hard:        全错 (n_correct == 0)
    """
    if n_correct == n_rollouts:
        return "easy"
    ratio = n_correct / n_rollouts
    if ratio >= 0.625:
        return "medium_easy"
    if ratio >= 0.25:
        return "medium"
    if n_correct > 0:
        return "medium_hard"
    return "hard"


# ==============================================================================
# Saving results
# ==============================================================================

def save_results(results: list, output_dir: str):
    """Save all results to JSONL."""
    path = os.path.join(output_dir, "sampling_results.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")


def save_difficulty_summary(results: list, output_dir: str, config: dict):
    """Save difficulty classification summary and per-difficulty sample lists."""
    difficulty_counts = defaultdict(int)
    difficulty_samples = defaultdict(list)

    for r in results:
        d = r["difficulty"]
        difficulty_counts[d] += 1
        difficulty_samples[d].append({
            "sample_idx": r["sample_idx"],
            "video_id": r.get("video_id", ""),
            "question_id": r.get("question_id", ""),
            "question": r.get("question", ""),
            "correct_answer": r.get("correct_answer", ""),
            "acc_mean": r["acc_mean"],
            "n_correct": r["n_correct"],
            "format_mean": r["format_mean"],
            "segment_mean": r["segment_mean"],
            "total_mean": r["total_mean"],
        })

    n_total = len(results)
    summary = {
        "timestamp": datetime.now().isoformat(),
        "model_path": config["model_path"],
        "data_path": config["data_path"],
        "total_samples": n_total,
        "n_rollouts": config["n_rollouts"],
        "difficulty_distribution": {
            d: {"count": difficulty_counts.get(d, 0),
                "percentage": difficulty_counts.get(d, 0) / n_total * 100 if n_total else 0}
            for d in ["easy", "medium_easy", "medium", "medium_hard", "hard"]
        },
        "overall_stats": {
            "acc_mean": float(np.mean([r["acc_mean"] for r in results])),
            "acc_std": float(np.std([r["acc_mean"] for r in results])),
            "format_mean": float(np.mean([r["format_mean"] for r in results])),
            "segment_mean": float(np.mean([r["segment_mean"] for r in results])),
            "total_mean": float(np.mean([r["total_mean"] for r in results])),
            "avg_turns": float(np.mean([
                np.mean([ro["num_turns"] for ro in r["rollouts"]]) for r in results
            ])),
        },
    }

    # Save summary
    with open(os.path.join(output_dir, "difficulty_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    # Save per-difficulty lists
    for difficulty, samples in difficulty_samples.items():
        path = os.path.join(output_dir, f"difficulty_{difficulty}.json")
        with open(path, "w") as f:
            json.dump(samples, f, indent=2, ensure_ascii=False, default=str)

    # Print summary
    logger.info("")
    logger.info("=" * 65)
    logger.info("  DIFFICULTY CLASSIFICATION SUMMARY")
    logger.info("=" * 65)
    for d in ["easy", "medium_easy", "medium", "medium_hard", "hard"]:
        cnt = difficulty_counts.get(d, 0)
        pct = cnt / n_total * 100 if n_total else 0
        logger.info(f"  {d:15s}: {cnt:5d}  ({pct:5.1f}%)")
    logger.info(f"  {'TOTAL':15s}: {n_total:5d}")
    logger.info("-" * 65)
    logger.info(f"  Avg Accuracy : {summary['overall_stats']['acc_mean']:.4f}")
    logger.info(f"  Avg Format   : {summary['overall_stats']['format_mean']:.4f}")
    logger.info(f"  Avg Segment  : {summary['overall_stats']['segment_mean']:.4f}")
    logger.info(f"  Avg Total    : {summary['overall_stats']['total_mean']:.4f}")
    logger.info(f"  Avg Turns    : {summary['overall_stats']['avg_turns']:.2f}")
    logger.info("=" * 65)


# ==============================================================================
# Main sampling logic
# ==============================================================================

def run_sampling(config: dict):
    """Main entry: load data, model; sample; compute rewards; classify."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config["output_dir"], timestamp)
    os.makedirs(output_dir, exist_ok=True)

    # Save config
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    logger.info(f"Output: {output_dir}")

    # ---- Load data ----
    logger.info(f"Loading data from {config['data_path']}")
    df = pd.read_parquet(config["data_path"])
    if config.get("max_samples"):
        df = df.head(config["max_samples"])
    logger.info(f"Loaded {len(df)} samples")

    # ---- Init frame cache ----
    frame_cache = VideoFrameCache(
        cache_dir=config["cache_dir"],
        fps=config["cache_fps"],
        max_frames=config["cache_max_frames"],
    )

    # ---- Init vLLM ----
    from vllm import LLM, SamplingParams
    from transformers import AutoProcessor

    logger.info(f"Loading model: {config['model_path']}")
    processor = AutoProcessor.from_pretrained(
        config["model_path"], trust_remote_code=True
    )
    llm = LLM(
        model=config["model_path"],
        tensor_parallel_size=config["tensor_parallel_size"],
        gpu_memory_utilization=config["gpu_memory_utilization"],
        max_model_len=config["max_model_len"],
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 100, "video": 30},
    )

    sampling_params = SamplingParams(
        temperature=config["temperature"],
        top_p=config["top_p"],
        max_tokens=config["max_tokens_per_turn"],
        repetition_penalty=config["repetition_penalty"],
    )

    # ---- Prepare all rollout states ----
    logger.info("Preparing initial messages ...")
    all_states: List[RolloutState] = []
    sample_meta: Dict[int, dict] = {}

    for sample_idx in tqdm(range(len(df)), desc="Preparing prompts"):
        row = df.iloc[sample_idx]
        extra_info = row.get("extra_info", {})
        if isinstance(extra_info, str):
            extra_info = json.loads(extra_info)

        video_path = row.get("video_path") or extra_info.get("video_path")
        video_duration = extra_info.get("video_duration")
        correct_answer = row.get("correct_answer") or extra_info.get("correct_answer", "")

        gt_segments_raw = row.get("reference_segments")
        gt_segments = []
        if gt_segments_raw is not None:
            if isinstance(gt_segments_raw, str):
                try:
                    gt_segments = json.loads(gt_segments_raw)
                except Exception:
                    gt_segments = []
            elif isinstance(gt_segments_raw, (list, np.ndarray)):
                gt_segments = [tuple(s) for s in gt_segments_raw]

        sample_meta[sample_idx] = {
            "video_path": video_path,
            "video_id": row.get("video_id", ""),
            "question_id": row.get("question_id", ""),
            "question": row.get("question", ""),
            "correct_answer": correct_answer,
            "gt_segments": gt_segments,
        }

        initial_messages = prepare_initial_messages(row, frame_cache, config)

        for rollout_idx in range(config["n_rollouts"]):
            state = RolloutState(
                sample_idx=sample_idx,
                rollout_idx=rollout_idx,
                messages=initial_messages,
                video_path=video_path,
                video_duration=video_duration,
            )
            all_states.append(state)

    logger.info(f"Total rollouts: {len(all_states)}")

    # ---- Multi-turn generation loop ----
    chunk_size = config.get("chunk_size", 64)
    max_turns = config["max_assistant_turns"]

    for turn in range(max_turns):
        active = [s for s in all_states if not s.done]
        if not active:
            break
        logger.info(f"Turn {turn + 1}/{max_turns}: {len(active)} active rollouts")

        # Process in chunks to manage memory
        for chunk_start in range(0, len(active), chunk_size):
            chunk = active[chunk_start : chunk_start + chunk_size]

            # Prepare vLLM inputs
            vllm_inputs = []
            valid_indices = []
            for i, state in enumerate(chunk):
                try:
                    inp = prepare_vllm_input(state.messages, processor)
                    vllm_inputs.append(inp)
                    valid_indices.append(i)
                except Exception as e:
                    state.error = str(e)
                    state.done = True
                    logger.warning(
                        f"Input prep error (sample={state.sample_idx}, "
                        f"rollout={state.rollout_idx}): {e}"
                    )

            if not vllm_inputs:
                continue

            # Generate
            try:
                outputs = llm.generate(vllm_inputs, sampling_params=sampling_params)
            except Exception as e:
                logger.error(f"Generation error: {e}")
                for idx in valid_indices:
                    chunk[idx].error = str(e)
                    chunk[idx].done = True
                continue

            # Process outputs
            for out_idx, vi in enumerate(valid_indices):
                state = chunk[vi]
                try:
                    response_text = outputs[out_idx].outputs[0].text
                except Exception as e:
                    state.error = str(e)
                    state.done = True
                    continue

                state.responses.append(response_text)
                state.full_response += response_text
                state.num_turns += 1

                # Add assistant response to conversation
                state.messages.append({"role": "assistant", "content": response_text})

                # Check for answer
                answer = extract_answer(response_text)
                if answer:
                    state.answer = answer
                    state.done = True
                    continue

                # Check for segments
                segments = extract_segments(response_text)
                if not segments or not state.video_path:
                    state.done = True
                    continue

                state.all_segments.append(segments)

                # Build observation (aligned with agent loop)
                obs_msg, total_frames = build_observation_message(
                    frame_cache, state.video_path, segments,
                    state.video_duration, config,
                )
                if total_frames == 0:
                    state.done = True
                    continue

                state.messages.append(obs_msg)

        logger.info(
            f"  Turn {turn + 1} done. "
            f"Active: {sum(1 for s in all_states if not s.done)}, "
            f"Done: {sum(1 for s in all_states if s.done)}"
        )

    # Mark remaining as done
    for s in all_states:
        if not s.done:
            # Didn't find answer within max turns, extract last answer attempt
            s.answer = extract_answer(s.full_response)
            s.done = True

    # ---- Compute rewards & aggregate ----
    logger.info("Computing rewards ...")
    all_results = []

    # Group states by sample
    states_by_sample = defaultdict(list)
    for state in all_states:
        states_by_sample[state.sample_idx].append(state)

    for sample_idx in tqdm(sorted(states_by_sample.keys()), desc="Computing rewards"):
        meta = sample_meta[sample_idx]
        states = states_by_sample[sample_idx]

        rollout_results = []
        for state in states:
            rewards = compute_rewards(
                state, meta["correct_answer"], meta["gt_segments"], config
            )
            rollout_results.append({
                "rollout_idx": state.rollout_idx,
                "answer": state.answer,
                "num_turns": state.num_turns,
                "num_segments_turns": len(state.all_segments),
                "all_segments": state.all_segments,
                "responses": state.responses,
                "error": state.error,
                **rewards,
            })

        # Sample-level aggregation
        acc_scores = [r["acc_score"] for r in rollout_results]
        fmt_scores = [r["format_score"] for r in rollout_results]
        seg_scores = [r["segment_score"] for r in rollout_results]
        total_scores = [r["total_score"] for r in rollout_results]
        n_correct = sum(1 for s in acc_scores if s > 0.5)

        difficulty = classify_difficulty(n_correct, config["n_rollouts"])

        sample_result = {
            "sample_idx": sample_idx,
            **{k: meta[k] for k in ["video_path", "video_id", "question_id",
                                     "question", "correct_answer"]},
            "gt_segments": meta["gt_segments"],
            "n_correct": n_correct,
            "acc_mean": float(np.mean(acc_scores)),
            "format_mean": float(np.mean(fmt_scores)),
            "segment_mean": float(np.mean(seg_scores)),
            "total_mean": float(np.mean(total_scores)),
            "difficulty": difficulty,
            "rollouts": rollout_results,
        }
        all_results.append(sample_result)

        # Periodic save & log
        if (sample_idx + 1) % 50 == 0:
            save_results(all_results, output_dir)
            logger.info(
                f"[{sample_idx + 1}/{len(df)}] "
                f"acc={sample_result['acc_mean']:.2f} "
                f"fmt={sample_result['format_mean']:.2f} "
                f"seg={sample_result['segment_mean']:.2f} "
                f"diff={difficulty}"
            )

    # ---- Save final results ----
    save_results(all_results, output_dir)
    save_difficulty_summary(all_results, output_dir, config)
    logger.info(f"All results saved to {output_dir}")


# ==============================================================================
# CLI
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Video Reasoning SFT Model Sampling & Difficulty Classification"
    )

    # Model & data
    parser.add_argument(
        "--model_path", type=str,
        default="/mnt/data/home/zhengshurong/project/Qwen3-VL/qwen-vl-finetune/checkpoints/"
                "video/Qwen3-VL-8B-Instruct-longvt_tvg-openo3video_stgr-selfconstructdata-"
                "sft-lr1e-5-bs64-ep1",
        help="Path to the SFT model",
    )
    parser.add_argument(
        "--data_path", type=str,
        default="./long_video_data/longvt_selfqa/train.parquet",
        help="Path to training parquet file",
    )
    parser.add_argument("--cache_dir", type=str, default="./.cache")
    parser.add_argument("--output_dir", type=str, default="./sampling_results")

    # Sampling
    parser.add_argument("--n_rollouts", type=int, default=8)
    parser.add_argument("--max_assistant_turns", type=int, default=5)
    parser.add_argument("--max_tokens_per_turn", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.7)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)

    # Video config
    parser.add_argument("--cache_fps", type=int, default=1)
    parser.add_argument("--cache_max_frames", type=int, default=512)
    parser.add_argument("--max_frames_per_segment", type=int, default=32)
    parser.add_argument("--use_cached_initial_video", type=bool, default=True)
    parser.add_argument("--initial_fps", type=int, default=1)
    parser.add_argument("--initial_max_frames", type=int, default=512)
    parser.add_argument("--initial_min_pixels", type=int, default=784)
    parser.add_argument("--initial_max_pixels", type=int, default=12544)
    parser.add_argument("--segment_fps", type=int, default=1)
    parser.add_argument("--segment_max_frames", type=int, default=32)
    parser.add_argument("--segment_min_pixels", type=int, default=784)
    parser.add_argument("--segment_max_pixels", type=int, default=50176)

    # Reward weights (match training config)
    parser.add_argument("--answer_weight", type=float, default=0.0)
    parser.add_argument("--format_weight", type=float, default=0.5)
    parser.add_argument("--segment_weight", type=float, default=1.0)
    parser.add_argument("--use_strict_format", type=bool, default=True)

    # vLLM config
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--max_model_len", type=int, default=128000)
    parser.add_argument("--chunk_size", type=int, default=64,
                        help="Batch size for vLLM generation per turn")

    # Debug
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of samples (for testing)")

    return vars(parser.parse_args())


if __name__ == "__main__":
    config = parse_args()
    run_sampling(config)
