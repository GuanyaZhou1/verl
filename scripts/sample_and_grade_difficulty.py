#!/usr/bin/env python3
"""
SFT 模型采样与难度分级脚本 (支持多轮推理)

功能：
1. 加载 SFT 模型，在 RL 训练数据上进行采样 (n_rollout=8)
2. 支持多轮推理：解析 <segment> 标签，加载视频帧，继续对话直到 <answer>
3. 使用 video_reasoning_async.py 中的奖励函数计算各项奖励
4. 根据奖励对不同问题样本进行难度分级

多轮推理逻辑 (与 video_reasoning_agent_loop.py 一致)：
- 生成响应后检查 <answer> 标签，找到则终止
- 检查 <segment> 标签，找到则加载对应视频帧
- 构建 observation 消息，继续对话
- 循环直到找到 <answer> 或达到最大轮数

使用方式：
    # 使用 OpenAI 兼容 API (推荐，需要先启动 vLLM 服务)
    python scripts/sample_and_grade_difficulty.py \
        --use_api \
        --api_base localhost:8000 \
        --api_model_name Qwen3-VL-8B \
        --data_path ./long_video_data/longvt_selfqa/train.parquet \
        --output_dir ./difficulty_analysis \
        --n_rollouts 8 \
        --max_turns 5

    # 使用本地 vLLM 引擎
    python scripts/sample_and_grade_difficulty.py \
        --model_path /path/to/sft/model \
        --data_path ./long_video_data/longvt_selfqa/train.parquet \
        --output_dir ./difficulty_analysis \
        --n_rollouts 8 \
        --max_turns 5
"""

import os
import sys
import json
import re
import asyncio
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, field

import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import video frame cache for multi-turn reasoning
from verl.utils.video_frame_cache import VideoFrameCache, CacheNotFoundError


def extract_segments(text: str) -> List[Tuple[float, float]]:
    """
    Extract time segments from text in format <segment>[(start, end), ...]</segment>

    Aligned with video_reasoning_agent_loop.py

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
    # Support both (start, end) and [start, end] formats
    pattern = r'[\(\[]\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*[\)\]]'
    for match in re.finditer(pattern, segment_str):
        start = float(match.group(1))
        end = float(match.group(2))
        segments.append((start, end))

    return segments


def extract_answer(text: str) -> Optional[str]:
    """
    Extract answer from text in format <answer>...</answer>

    Aligned with video_reasoning_agent_loop.py

    Args:
        text: Model output text

    Returns:
        Extracted answer content or None
    """
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


@dataclass
class QuestionStats:
    """问题级别的统计信息"""
    question_id: str
    video_id: str
    video_path: str
    question: str
    correct_answer: str
    reference_segments: List = field(default_factory=list)

    # 采样结果
    n_samples: int = 0
    responses: List[str] = field(default_factory=list)  # 完整的多轮对话响应
    num_turns: List[int] = field(default_factory=list)  # 每个 rollout 的轮数

    # 各项奖励统计
    answer_scores: List[float] = field(default_factory=list)
    format_scores: List[float] = field(default_factory=list)
    bbox_scores: List[float] = field(default_factory=list)
    segment_scores: List[float] = field(default_factory=list)
    final_scores: List[float] = field(default_factory=list)

    # 难度指标
    accuracy: float = 0.0
    avg_score: float = 0.0
    score_std: float = 0.0
    all_correct: bool = False
    all_wrong: bool = False
    difficulty_level: str = ""

    def compute_stats(self):
        """计算统计指标"""
        if not self.final_scores:
            return

        self.n_samples = len(self.final_scores)
        self.accuracy = sum(1 for s in self.answer_scores if s == 1.0) / self.n_samples if self.n_samples > 0 else 0
        self.avg_score = float(np.mean(self.final_scores))
        self.score_std = float(np.std(self.final_scores))
        self.all_correct = all(s == 1.0 for s in self.answer_scores)
        self.all_wrong = all(s == 0.0 for s in self.answer_scores)

        # 难度分级
        if self.accuracy >= 0.875:  # 7/8 或更高
            self.difficulty_level = "easy"
        elif self.accuracy <= 0.125:  # 1/8 或更低
            self.difficulty_level = "hard"
        else:
            self.difficulty_level = "medium"


def get_unique_sample_id(row, idx) -> str:
    """生成唯一的样本 ID，用于 resume 机制"""
    # 优先使用 video_id + question 组合作为唯一标识
    video_id = str(row.get('video_id', ''))
    question = str(row.get('question', ''))[:100]  # 截取前100字符
    if video_id and question:
        return f"{video_id}_{hash(question) % 10000000}"
    # 否则使用 DataFrame 索引
    return f"idx_{idx}"


def extract_text_from_prompt(prompt) -> str:
    """从 prompt 中提取纯文本内容"""
    if isinstance(prompt, str):
        return prompt
    elif isinstance(prompt, list):
        # 消息列表格式
        texts = []
        for msg in prompt:
            if isinstance(msg, dict):
                content = msg.get('content', '')
                if isinstance(content, str):
                    texts.append(content)
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get('type') == 'text':
                            texts.append(item.get('text', ''))
        return '\n'.join(texts)
    return str(prompt)


def extract_video_path_from_row(row) -> str:
    """从数据行中提取视频路径"""
    # 优先使用 video_path 字段
    if 'video_path' in row and row['video_path']:
        return row['video_path']

    # 从 videos 字段提取
    videos = row.get('videos')
    if videos is not None:
        if isinstance(videos, np.ndarray):
            videos = videos.tolist()
        if isinstance(videos, list) and len(videos) > 0:
            first_video = videos[0]
            if isinstance(first_video, dict):
                return first_video.get('video', '')
            elif isinstance(first_video, str):
                return first_video

    # 从 extra_info 提取
    extra_info = row.get('extra_info', {})
    if isinstance(extra_info, dict):
        return extra_info.get('video_path', '')

    return ''


class OpenAIAPIInferenceEngine:
    """使用 OpenAI 兼容 API 进行推理（适用于已部署的 vLLM 服务）- 支持多轮推理和多endpoint并行"""

    def __init__(
        self,
        api_base: str,  # 支持逗号分隔的多个endpoint，如 "localhost:8000,localhost:8001"
        model_name: str,
        api_key: str = "EMPTY",
        cache_dir: str = ".cache",
        cache_fps: int = 1,
        cache_max_frames: int = 512,
        max_frames_per_segment: int = 16,
    ):
        try:
            import openai
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("请安装 openai: pip install openai")

        # 解析多个 endpoint
        self.endpoints = [ep.strip() for ep in api_base.split(",")]
        self.model_name = model_name
        self.api_key = api_key

        # 创建同步客户端（用于单个请求）
        self.client = openai.OpenAI(
            base_url=f"http://{self.endpoints[0]}/v1",
            api_key=api_key,
        )

        # 创建异步客户端列表（用于并行请求）
        self.async_clients = [
            AsyncOpenAI(
                base_url=f"http://{ep}/v1",
                api_key=api_key,
            )
            for ep in self.endpoints
        ]

        # Initialize frame cache for multi-turn reasoning
        self.frame_cache = VideoFrameCache(
            cache_dir=cache_dir,
            fps=cache_fps,
            max_frames=cache_max_frames,
        )
        self.max_frames_per_segment = max_frames_per_segment

        logger.info(f"Initialized OpenAI API client with {len(self.endpoints)} endpoint(s): {self.endpoints}")

    def _load_image_as_base64(self, image_path: str) -> str:
        """Load image and convert to base64 for API"""
        import base64
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _build_observation_message(
        self,
        segments: List[Tuple[float, float]],
        video_path: str,
        video_duration: Optional[float] = None,
    ) -> Optional[Dict]:
        """
        Build observation message with video frames for segments.

        Aligned with video_reasoning_agent_loop.py
        """
        content_list = [
            {"type": "text", "text": "<observation>Here are the cropped video segments."}
        ]

        total_frames = 0
        for start, end in segments:
            # Boundary check (like eval script)
            if start >= end:
                end = start + 2.0
            if video_duration and end >= video_duration:
                start = min(start, video_duration - 2.0)
                end = video_duration

            # Load frame paths for this segment
            try:
                frame_paths = self.frame_cache.load_frame_paths(
                    video_path,
                    [(start, end)],
                    max_frames_per_segment=self.max_frames_per_segment,
                )
            except CacheNotFoundError:
                logger.warning(f"Cache not found for {video_path}")
                continue

            if not frame_paths:
                continue

            total_frames += len(frame_paths)

            # Add segment label
            content_list.append({"type": "text", "text": f"\nFrom {start}s to {end}s:"})

            # Add frames as images (OpenAI API format)
            for frame_path in frame_paths:
                base64_image = self._load_image_as_base64(frame_path)
                content_list.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                })

        if total_frames == 0:
            return None

        content_list.append({"type": "text", "text": "\n</observation>"})

        return {"role": "user", "content": content_list}

    def generate_single_multiturn(
        self,
        messages: List[Dict],
        video_path: Optional[str],
        video_duration: Optional[float],
        max_turns: int = 5,
        max_tokens: int = 8192,
        temperature: float = 0.7,
        top_p: float = 0.7,
    ) -> Tuple[str, int]:
        """
        Generate response with multi-turn reasoning.

        Returns:
            Tuple of (full_response, num_turns)
        """
        current_messages = list(messages)
        full_response_parts = []
        num_turns = 0

        for turn in range(max_turns):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=current_messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                response_text = response.choices[0].message.content or ""
            except Exception as e:
                logger.warning(f"API call failed: {e}")
                break

            num_turns += 1
            full_response_parts.append(response_text)

            # Check for answer - if found, we're done
            answer = extract_answer(response_text)
            if answer:
                break

            # Check for segments
            segments = extract_segments(response_text)
            if not segments or not video_path:
                # No segments found and no answer, stop
                break

            # Add assistant response to messages
            current_messages.append({"role": "assistant", "content": response_text})

            # Build observation message with video frames
            observation_message = self._build_observation_message(
                segments, video_path, video_duration
            )

            if observation_message is None:
                # No frames loaded, stop
                break

            current_messages.append(observation_message)

        # Combine all response parts
        full_response = "\n".join(full_response_parts)
        return full_response, num_turns

    def generate_single(
        self,
        messages: List[Dict],
        max_tokens: int = 8192,
        temperature: float = 0.7,
        top_p: float = 0.7,
        **kwargs,
    ) -> str:
        """生成单个响应 (单轮，保留向后兼容)"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.warning(f"API call failed: {e}")
            return ""

    async def _async_generate_single_multiturn(
        self,
        client_idx: int,
        messages: List[Dict],
        video_path: Optional[str],
        video_duration: Optional[float],
        max_turns: int = 5,
        max_tokens: int = 8192,
        temperature: float = 0.7,
        top_p: float = 0.7,
        seed: Optional[int] = None,
    ) -> Tuple[str, int]:
        """
        Async version of generate_single_multiturn using specified client.

        Note: vLLM V1 defaults seed=0, which causes same outputs even with temperature>0.
        Must pass different seed per request to get diverse outputs.
        """
        client = self.async_clients[client_idx]
        current_messages = list(messages)
        full_response_parts = []
        num_turns = 0

        for turn in range(max_turns):
            try:
                # 每轮使用不同的 seed 确保多样性
                request_seed = seed + turn if seed is not None else None
                response = await client.chat.completions.create(
                    model=self.model_name,
                    messages=current_messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    seed=request_seed,
                )
                response_text = response.choices[0].message.content or ""
            except Exception as e:
                logger.warning(f"API call failed on endpoint {self.endpoints[client_idx]}: {e}")
                break

            num_turns += 1
            full_response_parts.append(response_text)

            # Check for answer
            answer = extract_answer(response_text)
            if answer:
                break

            # Check for segments
            segments = extract_segments(response_text)
            if not segments or not video_path:
                break

            # Add assistant response
            current_messages.append({"role": "assistant", "content": response_text})

            # Build observation message
            observation_message = self._build_observation_message(
                segments, video_path, video_duration
            )

            if observation_message is None:
                break

            current_messages.append(observation_message)

        full_response = "\n".join(full_response_parts)
        return full_response, num_turns

    async def generate(
        self,
        prompts: List[Any],
        video_paths: Optional[List[str]] = None,
        video_durations: Optional[List[float]] = None,
        n_rollouts: int = 8,
        max_turns: int = 5,
        max_tokens: int = 8192,
        temperature: float = 0.7,
        top_p: float = 0.7,
        **kwargs,
    ) -> List[List[Tuple[str, int]]]:
        """
        Generate multiple responses with multi-turn reasoning.
        Supports parallel requests across multiple endpoints.

        Returns:
            List of lists of (response, num_turns) tuples
        """
        if video_paths is None:
            video_paths = [None] * len(prompts)
        if video_durations is None:
            video_durations = [None] * len(prompts)

        # 直接调用异步方法
        return await self._async_generate_all(
            prompts, video_paths, video_durations,
            n_rollouts, max_turns, max_tokens, temperature, top_p
        )

    async def _async_generate_all(
        self,
        prompts: List[Any],
        video_paths: List[Optional[str]],
        video_durations: List[Optional[float]],
        n_rollouts: int,
        max_turns: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> List[List[Tuple[str, int]]]:
        """
        Async implementation for parallel generation across multiple endpoints.
        """
        import random as _random
        num_endpoints = len(self.async_clients)
        results = []

        # 创建所有任务
        all_tasks = []
        task_info = []  # (prompt_idx, rollout_idx)

        for prompt_idx, (prompt, video_path, video_duration) in enumerate(
            zip(prompts, video_paths, video_durations)
        ):
            # 构建消息
            if isinstance(prompt, list):
                messages = prompt
            elif isinstance(prompt, str):
                messages = [{"role": "user", "content": prompt}]
            else:
                messages = [{"role": "user", "content": str(prompt)}]

            for rollout_idx in range(n_rollouts):
                # 轮询分配到不同的 endpoint
                client_idx = (prompt_idx * n_rollouts + rollout_idx) % num_endpoints
                # 为每个 rollout 生成不同的随机种子
                seed = _random.randint(0, 2**31 - 1)
                task = self._async_generate_single_multiturn(
                    client_idx=client_idx,
                    messages=messages,
                    video_path=video_path,
                    video_duration=video_duration,
                    max_turns=max_turns,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    seed=seed,
                )
                all_tasks.append(task)
                task_info.append((prompt_idx, rollout_idx))

        # 并行执行所有任务，带进度条
        logger.info(f"Running {len(all_tasks)} tasks across {num_endpoints} endpoint(s)...")

        # 使用 semaphore 限制并发数，避免过载
        max_concurrent = num_endpoints * 4  # 每个 endpoint 最多 4 个并发
        semaphore = asyncio.Semaphore(max_concurrent)

        async def run_with_semaphore(task):
            async with semaphore:
                return await task

        # 执行所有任务
        task_results = await asyncio.gather(
            *[run_with_semaphore(task) for task in all_tasks],
            return_exceptions=True
        )

        # 整理结果
        # 初始化结果列表
        for _ in prompts:
            results.append([None] * n_rollouts)

        for (prompt_idx, rollout_idx), result in zip(task_info, task_results):
            if isinstance(result, Exception):
                logger.warning(f"Task failed: {result}")
                results[prompt_idx][rollout_idx] = ("", 0)
            else:
                results[prompt_idx][rollout_idx] = result

        return results


class VLLMInferenceEngine:
    """vLLM 推理引擎 - 支持多轮推理"""

    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.8,
        max_model_len: int = 32768,
        trust_remote_code: bool = True,
        cache_dir: str = ".cache",
        cache_fps: int = 1,
        cache_max_frames: int = 512,
        max_frames_per_segment: int = 16,
    ):
        try:
            from vllm import LLM
        except ImportError:
            raise ImportError("请安装 vllm: pip install vllm")

        logger.info(f"Initializing vLLM engine with model: {model_path}")

        engine_kwargs = {
            "model": model_path,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "trust_remote_code": trust_remote_code,
            "max_model_len": max_model_len,
            "dtype": "auto",
        }

        self.llm = LLM(**engine_kwargs)
        self.tokenizer = self.llm.get_tokenizer()

        # Initialize frame cache for multi-turn reasoning
        self.frame_cache = VideoFrameCache(
            cache_dir=cache_dir,
            fps=cache_fps,
            max_frames=cache_max_frames,
        )
        self.max_frames_per_segment = max_frames_per_segment

        logger.info("vLLM engine initialized successfully")

    def _build_observation_message(
        self,
        segments: List[Tuple[float, float]],
        video_path: str,
        video_duration: Optional[float] = None,
    ) -> Optional[Dict]:
        """
        Build observation message with video frames for segments.

        Uses frame paths directly for vLLM (not base64).
        """
        content_list = [
            {"type": "text", "text": "<observation>Here are the cropped video segments."}
        ]

        total_frames = 0
        for start, end in segments:
            # Boundary check
            if start >= end:
                end = start + 2.0
            if video_duration and end >= video_duration:
                start = min(start, video_duration - 2.0)
                end = video_duration

            # Load frame paths for this segment
            try:
                frame_paths = self.frame_cache.load_frame_paths(
                    video_path,
                    [(start, end)],
                    max_frames_per_segment=self.max_frames_per_segment,
                )
            except CacheNotFoundError:
                logger.warning(f"Cache not found for {video_path}")
                continue

            if not frame_paths:
                continue

            total_frames += len(frame_paths)

            # Add segment label
            content_list.append({"type": "text", "text": f"\nFrom {start}s to {end}s:"})

            # Add frames as video (vLLM format)
            content_list.append({
                "type": "video",
                "video": frame_paths,
            })

        if total_frames == 0:
            return None

        content_list.append({"type": "text", "text": "\n</observation>"})

        return {"role": "user", "content": content_list}

    def generate_single_multiturn(
        self,
        messages: List[Dict],
        video_path: Optional[str],
        video_duration: Optional[float],
        max_turns: int = 5,
        max_tokens: int = 8192,
        temperature: float = 0.7,
        top_p: float = 0.7,
        repetition_penalty: float = 1.1,
        seed: Optional[int] = None,
    ) -> Tuple[str, int]:
        """
        Generate response with multi-turn reasoning using vLLM.

        Returns:
            Tuple of (full_response, num_turns)
        """
        import random as _random
        from vllm import SamplingParams

        # 为每次生成使用不同的随机种子
        if seed is None:
            seed = _random.randint(0, 2**31 - 1)

        sampling_params = SamplingParams(
            n=1,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            repetition_penalty=repetition_penalty,
            seed=seed,
        )

        current_messages = list(messages)
        full_response_parts = []
        num_turns = 0

        for turn in range(max_turns):
            # Apply chat template
            try:
                text_prompt = self.tokenizer.apply_chat_template(
                    current_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception as e:
                logger.warning(f"Failed to apply chat template: {e}")
                break

            # Generate
            outputs = self.llm.generate([text_prompt], sampling_params)
            if not outputs or not outputs[0].outputs:
                break

            response_text = outputs[0].outputs[0].text
            num_turns += 1
            full_response_parts.append(response_text)

            # Check for answer
            answer = extract_answer(response_text)
            if answer:
                break

            # Check for segments
            segments = extract_segments(response_text)
            if not segments or not video_path:
                break

            # Add assistant response
            current_messages.append({"role": "assistant", "content": response_text})

            # Build observation message
            observation_message = self._build_observation_message(
                segments, video_path, video_duration
            )

            if observation_message is None:
                break

            current_messages.append(observation_message)

        full_response = "\n".join(full_response_parts)
        return full_response, num_turns

    def generate(
        self,
        prompts: List[Any],
        video_paths: Optional[List[str]] = None,
        video_durations: Optional[List[float]] = None,
        n_rollouts: int = 8,
        max_turns: int = 5,
        max_tokens: int = 8192,
        temperature: float = 0.7,
        top_p: float = 0.7,
        repetition_penalty: float = 1.1,
        **kwargs,
    ) -> List[List[Tuple[str, int]]]:
        """
        Generate multiple responses with multi-turn reasoning.

        Returns:
            List of lists of (response, num_turns) tuples
        """
        results = []

        if video_paths is None:
            video_paths = [None] * len(prompts)
        if video_durations is None:
            video_durations = [None] * len(prompts)

        for prompt, video_path, video_duration in tqdm(
            zip(prompts, video_paths, video_durations),
            desc="Generating via vLLM",
            total=len(prompts)
        ):
            # 处理 prompt 格式
            if isinstance(prompt, list):
                messages = prompt
            elif isinstance(prompt, str):
                messages = [{"role": "user", "content": prompt}]
            else:
                messages = [{"role": "user", "content": str(prompt)}]

            responses = []
            for _ in range(n_rollouts):
                resp, num_turns = self.generate_single_multiturn(
                    messages=messages,
                    video_path=video_path,
                    video_duration=video_duration,
                    max_turns=max_turns,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                )
                responses.append((resp, num_turns))
            results.append(responses)

        return results

    async def generate_async(
        self,
        prompts: List[Any],
        video_paths: Optional[List[str]] = None,
        video_durations: Optional[List[float]] = None,
        n_rollouts: int = 8,
        max_turns: int = 5,
        max_tokens: int = 8192,
        temperature: float = 0.7,
        top_p: float = 0.7,
        repetition_penalty: float = 1.1,
        **kwargs,
    ) -> List[List[Tuple[str, int]]]:
        """
        Async wrapper for generate (vLLM is synchronous, but we need async interface).
        """
        return self.generate(
            prompts=prompts,
            video_paths=video_paths,
            video_durations=video_durations,
            n_rollouts=n_rollouts,
            max_turns=max_turns,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            **kwargs,
        )


async def compute_reward_for_sample(
    response: str,
    ground_truth: str,
    extra_info: Dict,
    reward_kwargs: Dict,
) -> Dict[str, float]:
    """计算单个样本的奖励"""
    try:
        from verl.utils.reward_score.video_reasoning_async import compute_score
    except ImportError as e:
        logger.error(f"Failed to import compute_score: {e}")
        return {
            'score': 0.0,
            'answer_score': 0.0,
            'format_score': 0.0,
            'bbox_score': 0.0,
            'segment_score': 0.0,
        }

    try:
        result = await compute_score(
            data_source="difficulty_analysis",
            solution_str=response,
            ground_truth=ground_truth,
            extra_info=extra_info,
            **reward_kwargs,
        )
        return result
    except Exception as e:
        logger.warning(f"Error computing reward: {e}")
        return {
            'score': 0.0,
            'answer_score': 0.0,
            'format_score': 0.0,
            'bbox_score': 0.0,
            'segment_score': 0.0,
        }


def parse_reference_segments(segments) -> List:
    """解析 reference_segments，支持字符串和列表格式"""
    if segments is None:
        return []
    if isinstance(segments, str):
        try:
            parsed = json.loads(segments)
            # 转换为 [(start, end), ...] 格式
            if isinstance(parsed, list):
                result = []
                for seg in parsed:
                    if isinstance(seg, (list, tuple)) and len(seg) >= 2:
                        result.append((float(seg[0]), float(seg[1])))
                return result
        except (json.JSONDecodeError, ValueError):
            return []
    if isinstance(segments, list):
        result = []
        for seg in segments:
            if isinstance(seg, (list, tuple)) and len(seg) >= 2:
                result.append((float(seg[0]), float(seg[1])))
        return result
    return []


async def process_question(
    row: pd.Series,
    row_idx: int,
    responses: List[Tuple[str, int]],
    reward_kwargs: Dict,
) -> QuestionStats:
    """处理单个问题的所有采样结果"""

    video_path = extract_video_path_from_row(row)

    # 解析 reference_segments
    ref_segments = parse_reference_segments(row.get('reference_segments'))

    # 构建 extra_info
    extra_info = dict(row.get('extra_info', {}) or {})
    extra_info.update({
        'video_path': video_path,
        'video_id': row.get('video_id', ''),
        'question': row.get('question', ''),
        'reference_segments': ref_segments,
    })

    ground_truth = row.get('correct_answer', '')

    # 使用唯一 ID
    unique_id = get_unique_sample_id(row, row_idx)

    stats = QuestionStats(
        question_id=unique_id,
        video_id=str(row.get('video_id', '')),
        video_path=video_path,
        question=row.get('question', ''),
        correct_answer=ground_truth,
        reference_segments=ref_segments,
    )

    # 计算每个响应的奖励
    tasks = []
    for response, num_turns in responses:
        tasks.append(compute_reward_for_sample(
            response=response,
            ground_truth=ground_truth,
            extra_info=extra_info,
            reward_kwargs=reward_kwargs,
        ))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.warning(f"Error computing reward: {result}")
            continue

        response, num_turns = responses[i]
        stats.responses.append(response)
        stats.num_turns.append(num_turns)
        stats.answer_scores.append(result.get('answer_score', 0.0))
        stats.format_scores.append(result.get('format_score', 0.0))
        stats.bbox_scores.append(result.get('bbox_score', 0.0))
        stats.segment_scores.append(result.get('segment_score', 0.0))
        stats.final_scores.append(result.get('score', 0.0))

    stats.compute_stats()
    return stats


def grade_difficulty(all_stats: List[QuestionStats]) -> Dict[str, List[QuestionStats]]:
    """根据统计结果进行难度分级"""
    graded = {
        'easy': [],
        'medium': [],
        'hard': [],
    }

    for stats in all_stats:
        if stats.difficulty_level:
            graded[stats.difficulty_level].append(stats)

    return graded


def save_results(
    all_stats: List[QuestionStats],
    graded: Dict[str, List[QuestionStats]],
    output_dir: str,
) -> Dict:
    """保存结果"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. 保存详细结果 (JSONL)
    detailed_path = os.path.join(output_dir, f"detailed_results_{timestamp}.jsonl")
    with open(detailed_path, 'w', encoding='utf-8') as f:
        for stats in all_stats:
            data = {
                'question_id': stats.question_id,
                'video_id': stats.video_id,
                'video_path': stats.video_path,
                'question': stats.question[:500],
                'correct_answer': stats.correct_answer,
                'reference_segments': stats.reference_segments,
                'n_samples': stats.n_samples,
                'accuracy': stats.accuracy,
                'avg_score': stats.avg_score,
                'score_std': stats.score_std,
                'all_correct': stats.all_correct,
                'all_wrong': stats.all_wrong,
                'difficulty_level': stats.difficulty_level,
                'answer_scores': stats.answer_scores,
                'format_scores': stats.format_scores,
                'bbox_scores': stats.bbox_scores,
                'segment_scores': stats.segment_scores,
                'final_scores': stats.final_scores,
                'num_turns': stats.num_turns,
                'avg_turns': float(np.mean(stats.num_turns)) if stats.num_turns else 0,
                'responses': [r[:10000] for r in stats.responses],
            }
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    logger.info(f"Saved detailed results to {detailed_path}")

    # 2. 保存难度分级摘要
    summary_path = os.path.join(output_dir, f"difficulty_summary_{timestamp}.json")
    total_questions = len(all_stats)
    summary = {
        'timestamp': timestamp,
        'total_questions': total_questions,
        'difficulty_distribution': {
            level: len(stats_list) for level, stats_list in graded.items()
        },
        'statistics': {
            'overall': {
                'avg_accuracy': float(np.mean([s.accuracy for s in all_stats])) if all_stats else 0,
                'avg_score': float(np.mean([s.avg_score for s in all_stats])) if all_stats else 0,
                'all_correct_count': sum(1 for s in all_stats if s.all_correct),
                'all_wrong_count': sum(1 for s in all_stats if s.all_wrong),
                'avg_turns': float(np.mean([np.mean(s.num_turns) for s in all_stats if s.num_turns])) if any(s.num_turns for s in all_stats) else 0,
            },
        },
    }

    for level, stats_list in graded.items():
        if stats_list:
            summary['statistics'][level] = {
                'count': len(stats_list),
                'avg_accuracy': float(np.mean([s.accuracy for s in stats_list])),
                'avg_score': float(np.mean([s.avg_score for s in stats_list])),
                'avg_format_score': float(np.mean([np.mean(s.format_scores) for s in stats_list if s.format_scores])) if any(s.format_scores for s in stats_list) else 0,
                'avg_bbox_score': float(np.mean([np.mean(s.bbox_scores) for s in stats_list if s.bbox_scores])) if any(s.bbox_scores for s in stats_list) else 0,
                'avg_segment_score': float(np.mean([np.mean(s.segment_scores) for s in stats_list if s.segment_scores])) if any(s.segment_scores for s in stats_list) else 0,
                'avg_turns': float(np.mean([np.mean(s.num_turns) for s in stats_list if s.num_turns])) if any(s.num_turns for s in stats_list) else 0,
            }

    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved summary to {summary_path}")

    # 3. 保存各难度级别的问题 ID 列表
    for level, stats_list in graded.items():
        level_path = os.path.join(output_dir, f"{level}_questions_{timestamp}.txt")
        with open(level_path, 'w') as f:
            for stats in stats_list:
                f.write(f"{stats.question_id}\t{stats.accuracy:.3f}\t{stats.avg_score:.3f}\n")
        logger.info(f"Saved {level} questions ({len(stats_list)}) to {level_path}")

    # 4. 保存为 parquet 格式
    df_data = []
    for stats in all_stats:
        df_data.append({
            'question_id': stats.question_id,
            'video_id': stats.video_id,
            'video_path': stats.video_path,
            'question': stats.question,
            'correct_answer': stats.correct_answer,
            'accuracy': stats.accuracy,
            'avg_score': stats.avg_score,
            'score_std': stats.score_std,
            'difficulty_level': stats.difficulty_level,
            'all_correct': stats.all_correct,
            'all_wrong': stats.all_wrong,
            'avg_answer_score': float(np.mean(stats.answer_scores)) if stats.answer_scores else 0,
            'avg_format_score': float(np.mean(stats.format_scores)) if stats.format_scores else 0,
            'avg_bbox_score': float(np.mean(stats.bbox_scores)) if stats.bbox_scores else 0,
            'avg_segment_score': float(np.mean(stats.segment_scores)) if stats.segment_scores else 0,
            'avg_turns': float(np.mean(stats.num_turns)) if stats.num_turns else 0,
        })

    df = pd.DataFrame(df_data)
    parquet_path = os.path.join(output_dir, f"difficulty_analysis_{timestamp}.parquet")
    df.to_parquet(parquet_path)
    logger.info(f"Saved parquet to {parquet_path}")

    return summary


def print_summary(summary: Dict):
    """打印摘要信息"""
    print("\n" + "=" * 60)
    print("难度分级结果摘要")
    print("=" * 60)
    print(f"总问题数: {summary['total_questions']}")
    print(f"\n难度分布:")
    for level, count in summary['difficulty_distribution'].items():
        pct = count / summary['total_questions'] * 100 if summary['total_questions'] > 0 else 0
        print(f"  {level}: {count} ({pct:.1f}%)")

    print(f"\n整体统计:")
    overall = summary['statistics']['overall']
    print(f"  平均准确率: {overall['avg_accuracy']:.3f}")
    print(f"  平均分数: {overall['avg_score']:.3f}")
    print(f"  全对样本数: {overall['all_correct_count']}")
    print(f"  全错样本数: {overall['all_wrong_count']}")
    print(f"  平均轮数: {overall.get('avg_turns', 0):.2f}")

    print(f"\n各难度级别统计:")
    for level in ['easy', 'medium', 'hard']:
        if level in summary['statistics']:
            stats = summary['statistics'][level]
            print(f"\n  [{level.upper()}] (n={stats['count']})")
            print(f"    平均准确率: {stats['avg_accuracy']:.3f}")
            print(f"    平均分数: {stats['avg_score']:.3f}")
            print(f"    平均格式分: {stats.get('avg_format_score', 0):.3f}")
            print(f"    平均bbox分: {stats.get('avg_bbox_score', 0):.3f}")
            print(f"    平均segment分: {stats.get('avg_segment_score', 0):.3f}")
            print(f"    平均轮数: {stats.get('avg_turns', 0):.2f}")

    print("=" * 60)


async def main_async(args):
    """异步主函数"""
    # 加载数据
    logger.info(f"Loading data from {args.data_path}")
    df = pd.read_parquet(args.data_path)
    logger.info(f"Loaded {len(df)} samples")

    if args.num_samples:
        df = df.head(args.num_samples)
        logger.info(f"Limited to {len(df)} samples")

    # 提取视频路径和时长
    video_paths = []
    video_durations = []
    for _, row in df.iterrows():
        video_path = extract_video_path_from_row(row)
        video_paths.append(video_path)
        # 尝试从 extra_info 获取视频时长
        extra_info = row.get('extra_info', {}) or {}
        video_duration = extra_info.get('video_duration')
        video_durations.append(video_duration)

    # 初始化推理引擎
    if args.use_api:
        engine = OpenAIAPIInferenceEngine(
            api_base=args.api_base,
            model_name=args.api_model_name,
            api_key=args.api_key,
            cache_dir=args.cache_dir,
            cache_fps=args.cache_fps,
            cache_max_frames=args.cache_max_frames,
            max_frames_per_segment=args.max_frames_per_segment,
        )
    else:
        engine = VLLMInferenceEngine(
            model_path=args.model_path,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            cache_dir=args.cache_dir,
            cache_fps=args.cache_fps,
            cache_max_frames=args.cache_max_frames,
            max_frames_per_segment=args.max_frames_per_segment,
        )

    # 准备奖励函数参数
    reward_kwargs = {
        'vlm_endpoint': args.vlm_endpoint,
        'vlm_model_name': args.vlm_model_name,
        'vlm_api_key': args.vlm_api_key,
        'cache_dir': args.cache_dir,
        'use_vlm_scoring': args.use_vlm_scoring,
        'use_bbox_verification': args.use_bbox_verification,
        'answer_weight': args.answer_weight,
        'bbox_weight': args.bbox_weight,
        'format_weight': args.format_weight,
        'segment_weight': args.segment_weight,
        'use_strict_format': args.use_strict_format,
        'enable_logging': False,
        'save_samples': False,
    }

    # 批量处理
    all_stats = []
    prompts = df['prompt'].tolist()

    logger.info(f"Generating responses with n_rollouts={args.n_rollouts}, max_turns={args.max_turns}")

    # 实时保存文件路径
    os.makedirs(args.output_dir, exist_ok=True)
    realtime_jsonl_path = os.path.join(args.output_dir, "realtime_results.jsonl")

    # Resume 机制：检查已处理的样本
    processed_question_ids = set()
    if args.resume and os.path.exists(realtime_jsonl_path):
        logger.info(f"Resume mode: loading existing results from {realtime_jsonl_path}")
        with open(realtime_jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    processed_question_ids.add(data['question_id'])
                    # 重建 stats 对象
                    stats = QuestionStats(
                        question_id=data['question_id'],
                        video_id=data.get('video_id', ''),
                        video_path=data.get('video_path', ''),
                        question=data.get('question', ''),
                        correct_answer=data.get('correct_answer', ''),
                    )
                    stats.n_samples = data.get('n_samples', 0)
                    stats.accuracy = data.get('accuracy', 0.0)
                    stats.avg_score = data.get('avg_score', 0.0)
                    stats.score_std = data.get('score_std', 0.0)
                    stats.difficulty_level = data.get('difficulty_level', '')
                    stats.answer_scores = data.get('answer_scores', [])
                    stats.format_scores = data.get('format_scores', [])
                    stats.bbox_scores = data.get('bbox_scores', [])
                    stats.segment_scores = data.get('segment_scores', [])
                    stats.final_scores = data.get('final_scores', [])
                    stats.num_turns = data.get('num_turns', [])
                    stats.responses = data.get('responses', [])
                    all_stats.append(stats)
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Failed to parse line: {e}")
                    continue
        logger.info(f"Loaded {len(processed_question_ids)} already processed samples")
    else:
        # 非 resume 模式，清空已有文件
        if os.path.exists(realtime_jsonl_path):
            os.remove(realtime_jsonl_path)

    logger.info(f"Realtime results will be saved to: {realtime_jsonl_path}")

    # 分批生成
    for batch_start in tqdm(range(0, len(prompts), args.batch_size), desc="Processing batches"):
        batch_end = min(batch_start + args.batch_size, len(prompts))
        batch_prompts = prompts[batch_start:batch_end]
        batch_video_paths = video_paths[batch_start:batch_end]
        batch_video_durations = video_durations[batch_start:batch_end]
        batch_df = df.iloc[batch_start:batch_end]

        # 检查这个 batch 中哪些需要处理
        batch_indices_to_process = []
        for i, (idx, row) in enumerate(batch_df.iterrows()):
            sample_id = get_unique_sample_id(row, idx)
            if sample_id not in processed_question_ids:
                batch_indices_to_process.append(i)

        if not batch_indices_to_process:
            logger.info(f"Batch {batch_start//args.batch_size + 1} already processed, skipping")
            continue

        # 只处理未完成的样本
        filtered_prompts = [batch_prompts[i] for i in batch_indices_to_process]
        filtered_video_paths = [batch_video_paths[i] for i in batch_indices_to_process]
        filtered_video_durations = [batch_video_durations[i] for i in batch_indices_to_process]
        filtered_rows = [(batch_df.iloc[i], batch_df.index[i]) for i in batch_indices_to_process]  # (row, idx)

        # 生成响应 (多轮推理)
        logger.info(f"Generating batch {batch_start//args.batch_size + 1} ({len(filtered_prompts)} samples)")

        # 根据引擎类型选择调用方式
        if isinstance(engine, VLLMInferenceEngine):
            batch_responses = await engine.generate_async(
                filtered_prompts,
                video_paths=filtered_video_paths,
                video_durations=filtered_video_durations,
                n_rollouts=args.n_rollouts,
                max_turns=args.max_turns,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
        else:
            batch_responses = await engine.generate(
                filtered_prompts,
                video_paths=filtered_video_paths,
                video_durations=filtered_video_durations,
                n_rollouts=args.n_rollouts,
                max_turns=args.max_turns,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )

        # 计算奖励
        logger.info(f"Computing rewards for batch {batch_start//args.batch_size + 1}")
        batch_stats = []
        for i, (row, row_idx) in enumerate(filtered_rows):
            responses = batch_responses[i]  # List of (response, num_turns) tuples
            stats = await process_question(row, row_idx, responses, reward_kwargs)
            all_stats.append(stats)
            batch_stats.append(stats)
            processed_question_ids.add(stats.question_id)

        logger.info(f"Processed {len(all_stats)}/{len(df)} questions")

        # 实时保存当前 batch 的结果
        with open(realtime_jsonl_path, 'a', encoding='utf-8') as f:
            for stats in batch_stats:
                data = {
                    'question_id': stats.question_id,
                    'video_id': stats.video_id,
                    'video_path': stats.video_path,
                    'question': stats.question[:500],
                    'correct_answer': stats.correct_answer,
                    'n_samples': stats.n_samples,
                    'accuracy': stats.accuracy,
                    'avg_score': stats.avg_score,
                    'score_std': stats.score_std,
                    'difficulty_level': stats.difficulty_level,
                    'answer_scores': stats.answer_scores,
                    'format_scores': stats.format_scores,
                    'bbox_scores': stats.bbox_scores,
                    'segment_scores': stats.segment_scores,
                    'final_scores': stats.final_scores,
                    'num_turns': stats.num_turns,
                    'responses': [r[:10000] for r in stats.responses],
                }
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
        logger.info(f"Saved batch {batch_start//args.batch_size + 1} to {realtime_jsonl_path}")

    # 难度分级
    graded = grade_difficulty(all_stats)

    # 保存结果
    summary = save_results(all_stats, graded, args.output_dir)

    # 打印摘要
    print_summary(summary)

    return summary


def main():
    parser = argparse.ArgumentParser(description="SFT 模型采样与难度分级")

    # 模型参数
    parser.add_argument("--model_path", type=str, help="SFT 模型路径 (本地 vLLM 模式)")
    parser.add_argument("--use_api", action="store_true", help="使用 OpenAI 兼容 API 模式")
    parser.add_argument("--api_base", type=str, default="localhost:8000", help="API 服务地址")
    parser.add_argument("--api_model_name", type=str, default="default", help="API 模型名称")
    parser.add_argument("--api_key", type=str, default="EMPTY", help="API Key")

    # 数据参数
    parser.add_argument("--data_path", type=str, required=True, help="训练数据路径 (parquet)")
    parser.add_argument("--output_dir", type=str, default="./difficulty_analysis", help="输出目录")
    parser.add_argument("--num_samples", type=int, default=None, help="限制采样的问题数量")
    parser.add_argument("--resume", action="store_true", help="从上次中断处继续，跳过已处理的样本")

    # 生成参数
    parser.add_argument("--n_rollouts", type=int, default=8, help="每个问题的采样数")
    parser.add_argument("--max_turns", type=int, default=5, help="多轮推理的最大轮数")
    parser.add_argument("--max_tokens", type=int, default=8192, help="最大生成 token 数")
    parser.add_argument("--temperature", type=float, default=0.7, help="采样温度")
    parser.add_argument("--top_p", type=float, default=0.7, help="Top-p 采样")
    parser.add_argument("--batch_size", type=int, default=4, help="批处理大小")

    # vLLM 参数
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="vLLM tensor parallel size")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8, help="GPU 内存利用率")
    parser.add_argument("--max_model_len", type=int, default=32768, help="最大模型长度")

    # 视频帧缓存参数 (多轮推理需要)
    parser.add_argument("--cache_fps", type=int, default=1, help="帧缓存的 FPS")
    parser.add_argument("--cache_max_frames", type=int, default=512, help="每个视频最大缓存帧数")
    parser.add_argument("--max_frames_per_segment", type=int, default=16, help="每个 segment 最大帧数")

    # 奖励函数参数
    parser.add_argument("--vlm_endpoint", type=str, default="10.0.1.35:8081", help="VLM 服务地址")
    parser.add_argument("--vlm_model_name", type=str, default="Qwen3-VL-235B-A22B-Instruct", help="VLM 模型名称")
    parser.add_argument("--vlm_api_key", type=str, default="123456", help="VLM API Key")
    parser.add_argument("--cache_dir", type=str, default=".cache", help="视频帧缓存目录")
    parser.add_argument("--use_vlm_scoring", action="store_true", default=True, help="使用 VLM 评分")
    parser.add_argument("--no_vlm_scoring", action="store_false", dest="use_vlm_scoring", help="不使用 VLM 评分")
    parser.add_argument("--use_bbox_verification", action="store_true", default=True, help="使用 bbox 验证")
    parser.add_argument("--no_bbox_verification", action="store_false", dest="use_bbox_verification", help="不使用 bbox 验证")
    parser.add_argument("--answer_weight", type=float, default=0.0, help="答案分数权重")
    parser.add_argument("--bbox_weight", type=float, default=0.0, help="bbox 分数权重")
    parser.add_argument("--format_weight", type=float, default=0.5, help="格式分数权重")
    parser.add_argument("--segment_weight", type=float, default=1.0, help="segment 分数权重")
    parser.add_argument("--use_strict_format", action="store_true", default=True, help="使用严格格式检查")

    args = parser.parse_args()

    # 验证参数
    if not args.use_api and not args.model_path:
        parser.error("必须指定 --model_path 或使用 --use_api 模式")

    # 运行
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
