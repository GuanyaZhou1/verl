#!/usr/bin/env python3
# Copyright 2025 Individual Contributor: Sudong Wang, Zuhao Yang, Kaichen Zhang
# Copyright 2024 Bytedance Ltd. and/or its affiliates (verl integration)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
LongVT Agent Loop for multi-turn video reasoning RL training.

This AgentLoop uses LongVT's output format:
1. Parses <tool_call>{"name":"crop_video","arguments":{...}}</tool_call> tags
2. Loads frames from cache as jpg files
3. Uses <tool_response>...</tool_response> format for observations
4. Continues until <answer> is found or max turns reached

Expected output format:
    <think>分析...</think>
    <tool_call>{"name":"crop_video","arguments":{"video_path":"...","start_time":10,"end_time":20}}</tool_call>
    <tool_response>视频帧已加载</tool_response>
    <think>继续分析...</think>
    <answer>答案</answer>

This differs from video_reasoning_agent_loop.py which uses:
    <segment>[(start, end)]</segment> and <observation>...</observation>
"""

import ast
import logging
import os
import random
import re
from typing import Any, List, Optional, Tuple
from uuid import uuid4

import torch
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopMetrics,
    AgentLoopOutput,
    AsyncLLMServerManager,
    DictConfigWrap,
    _merge_multi_modal_inputs,
    register,
)
from verl.utils.profiler import simple_timer
from verl.utils.video_frame_cache import CacheNotFoundError, VideoFrameCache, add_timestamp_watermark

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def extract_tool_calls(text: str) -> List[dict]:
    """
    Extract tool calls from text in format <tool_call>{"name":"crop_video",...}</tool_call>

    Args:
        text: Model output text

    Returns:
        List of tool call dictionaries with start_time, end_time, video_path
    """
    pattern = r"<tool_call>(.*?)</tool_call>"
    matches = re.findall(pattern, text, re.DOTALL)
    tool_calls = []

    for match in matches:
        try:
            data = ast.literal_eval(match.strip())
            if isinstance(data, dict) and data.get("name") == "crop_video":
                arguments = data.get("arguments", {})
                if "start_time" in arguments and "end_time" in arguments:
                    tool_calls.append(
                        {
                            "start_time": float(arguments["start_time"]),
                            "end_time": float(arguments["end_time"]),
                            "video_path": arguments.get("video_path"),
                        }
                    )
        except (ValueError, SyntaxError, KeyError):
            continue

    return tool_calls


def extract_answer(text: str) -> Optional[str]:
    """
    Extract answer from text in format <answer>...</answer>

    Args:
        text: Model output text

    Returns:
        Extracted answer content or None
    """
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


@register("longvt")
class LongVTAgentLoop(AgentLoopBase):
    """
    Agent loop for LongVT-style video reasoning with <tool_call> format.

    Unlike VideoReasoningAgentLoop which uses <segment> tags,
    this loop parses <tool_call> tags for crop_video function calls
    and returns <tool_response> messages.

    Multi-turn format:
    - Round 1: [video + prompt] -> model generates <think>...<tool_call>...
    - Round 2: [video + prompt + assistant + tool_response] -> model generates ...
    - Final: <think>...<answer>...</answer>
    """

    def __init__(
        self,
        trainer_config: DictConfigWrap,
        server_manager: AsyncLLMServerManager,
        tokenizer: AutoTokenizer,
        processor: AutoProcessor,
        **kwargs,
    ):
        super().__init__(trainer_config, server_manager, tokenizer, processor, **kwargs)
        config = trainer_config.config

        # Multi-turn configuration
        self.max_user_turns = config.actor_rollout_ref.rollout.multi_turn.get("max_user_turns", 5)
        self.max_assistant_turns = config.actor_rollout_ref.rollout.multi_turn.get("max_assistant_turns", 5)
        self.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        self.response_length = config.actor_rollout_ref.rollout.response_length

        # Frame cache configuration
        cache_config = config.actor_rollout_ref.rollout.multi_turn.get("cache_config", {})
        self.cache_dir = cache_config.get("cache_dir", ".cache")
        self.cache_fps = cache_config.get("fps", 1)
        self.cache_max_frames = cache_config.get("max_frames", 512)
        self.max_frames_per_segment = cache_config.get("max_frames_per_segment", 16)

        # Initial video configuration
        initial_video_config = config.actor_rollout_ref.rollout.multi_turn.get("initial_video_config", {})
        self.initial_fps = initial_video_config.get("fps", 1)
        self.initial_max_frames = initial_video_config.get("max_frames", 512)
        self.initial_min_pixels = initial_video_config.get("min_pixels", 784)
        self.initial_max_pixels = initial_video_config.get("max_pixels", 12544)

        # Segment video configuration
        segment_video_config = config.actor_rollout_ref.rollout.multi_turn.get("segment_video_config", {})
        self.segment_fps = segment_video_config.get("fps", 1)
        self.segment_max_frames = segment_video_config.get("max_frames", 32)
        self.segment_min_pixels = segment_video_config.get("min_pixels", 784)
        self.segment_max_pixels = segment_video_config.get("max_pixels", 50176)

        # Whether to use cached frames for initial video
        self.use_cached_initial_video = cache_config.get("use_cached_initial_video", False)

        # Timestamp watermark configuration
        watermark_config = config.actor_rollout_ref.rollout.multi_turn.get("watermark_config", {})
        self.use_timestamp_watermark = watermark_config.get("enable", False)
        self.watermark_position = watermark_config.get("position", "top_left")
        self.watermark_font_size = watermark_config.get("font_size", 0)
        self.watermark_ratio = watermark_config.get("ratio", 1.0)

        # Initialize frame cache
        self.frame_cache = VideoFrameCache(
            cache_dir=self.cache_dir,
            fps=self.cache_fps,
            max_frames=self.cache_max_frames,
        )

    @staticmethod
    def _fix_video_metadata_timestamps(
        obs_videos: list,
        per_video_timestamps: List[List[float]],
        cache_fps: int,
    ) -> list:
        """Fix video_metadata.frames_indices for correct absolute timestamps."""
        for (video_tensor, metadata), timestamps in zip(obs_videos, per_video_timestamps):
            n_frames = len(metadata.get("frames_indices", []))
            abs_indices = [int(round(ts * cache_fps)) for ts in timestamps]
            while len(abs_indices) < n_frames:
                abs_indices.append(abs_indices[-1] if abs_indices else 0)
            metadata["frames_indices"] = abs_indices[:n_frames]
            metadata["fps"] = cache_fps
        return obs_videos

    def _get_frame_paths_with_timestamps_for_segments(
        self,
        video_path: str,
        segments: List[Tuple[float, float]],
    ) -> List[Tuple[str, float]]:
        """Load frame jpg paths with timestamps from cache for given segments."""
        try:
            frame_paths_with_ts = self.frame_cache.load_frame_paths_with_timestamps(
                video_path,
                segments,
                max_frames_per_segment=self.max_frames_per_segment,
            )
            return frame_paths_with_ts
        except CacheNotFoundError:
            logger.warning(f"Cache not found for {video_path}")
            return []

    def _get_initial_frame_paths_with_timestamps(self, video_path: str) -> List[Tuple[str, float]]:
        """Load all cached frame paths with timestamps for initial video."""
        try:
            frame_paths_with_ts = self.frame_cache.load_frame_paths_with_timestamps(
                video_path,
                segments=None,
                max_frames_per_segment=self.initial_max_frames,
            )
            return frame_paths_with_ts
        except CacheNotFoundError:
            logger.warning(f"Cache not found for {video_path}, falling back to original video")
            return []

    def _add_watermarks_to_frames(
        self,
        frame_paths_with_ts: List[Tuple[str, float]],
    ) -> List[Image.Image]:
        """Add timestamp watermarks to frames."""
        watermarked_frames = []
        for frame_path, timestamp in frame_paths_with_ts:
            img = Image.open(frame_path)
            add_timestamp_watermark(
                img,
                timestamp,
                position=self.watermark_position,
                font_size=self.watermark_font_size,
            )
            watermarked_frames.append(img)
        return watermarked_frames

    def _replace_video_with_cached_frames(
        self,
        messages: List[dict],
        frame_paths: List[str],
    ) -> List[dict]:
        """Replace video path in messages with cached frame paths."""
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "video":
                        item["video"] = frame_paths
        return messages

    def _replace_video_with_pil_images(
        self,
        messages: List[dict],
        pil_images: List[Image.Image],
    ) -> List[dict]:
        """Replace video path in messages with PIL Images."""
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "video":
                        item["video"] = pil_images
        return messages

    def _inject_video_params(
        self,
        messages: List[dict],
        fps: int,
        max_frames: int,
        min_pixels: int,
        max_pixels: int,
    ) -> List[dict]:
        """Inject video parameters into messages containing video content."""
        for message in messages:
            content = message.get("content")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "video":
                        if "fps" not in item:
                            item["fps"] = fps
                        if "max_frames" not in item:
                            item["max_frames"] = max_frames
                        if "min_pixels" not in item:
                            item["min_pixels"] = min_pixels
                        if "max_pixels" not in item:
                            item["max_pixels"] = max_pixels
        return messages

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """Main loop for LongVT video reasoning rollout."""
        try:
            return await self._run_impl(sampling_params, **kwargs)
        except Exception as e:
            logger.warning(f"Error in LongVT agent loop, skipping sample: {e}")
            return AgentLoopOutput(
                prompt_ids=[],
                response_ids=[],
                response_mask=[],
                multi_modal_data={},
                accumulated_multi_modal_inputs={},
                accumulated_multi_modal_inputs_no_watermark=None,
                num_turns=0,
                metrics=AgentLoopMetrics(),
            )

    async def _run_impl(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """Actual run implementation."""
        raw_prompt = kwargs.get("raw_prompt", [])
        extra_info = kwargs.get("extra_info", {})

        video_path = extra_info.get("video_path")
        video_duration = extra_info.get("video_duration")

        # Per-sample watermark decision
        use_watermark = self.use_timestamp_watermark and random.random() < self.watermark_ratio

        messages = list(raw_prompt)
        messages_no_watermark = list(raw_prompt) if use_watermark else None

        # Inject initial video params
        self._inject_video_params(
            messages,
            fps=self.initial_fps,
            max_frames=self.initial_max_frames,
            min_pixels=self.initial_min_pixels,
            max_pixels=self.initial_max_pixels,
        )
        if messages_no_watermark:
            self._inject_video_params(
                messages_no_watermark,
                fps=self.initial_fps,
                max_frames=self.initial_max_frames,
                min_pixels=self.initial_min_pixels,
                max_pixels=self.initial_max_pixels,
            )

        # Handle cached initial video
        initial_timestamps = None
        if self.use_cached_initial_video and video_path:
            if use_watermark:
                initial_frame_paths_with_ts = self._get_initial_frame_paths_with_timestamps(video_path)
                if initial_frame_paths_with_ts:
                    watermarked_frames = self._add_watermarks_to_frames(initial_frame_paths_with_ts)
                    self._replace_video_with_pil_images(messages, watermarked_frames)
                    original_frame_paths = [fp for fp, _ in initial_frame_paths_with_ts]
                    self._replace_video_with_cached_frames(messages_no_watermark, original_frame_paths)
                    initial_timestamps = [ts for _, ts in initial_frame_paths_with_ts]
            else:
                initial_frame_paths_with_ts = self._get_initial_frame_paths_with_timestamps(video_path)
                if initial_frame_paths_with_ts:
                    initial_frame_paths = [fp for fp, _ in initial_frame_paths_with_ts]
                    self._replace_video_with_cached_frames(messages, initial_frame_paths)
                    initial_timestamps = [ts for _, ts in initial_frame_paths_with_ts]

        # Process initial vision info
        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images", [])
        videos = multi_modal_data.get("videos", [])

        # Fix video metadata timestamps
        if initial_timestamps and videos:
            self._fix_video_metadata_timestamps(videos, [initial_timestamps], self.cache_fps)

        # Process non-watermark version if needed
        multi_modal_data_no_watermark = None
        images_no_watermark = None
        videos_no_watermark = None
        if use_watermark and messages_no_watermark:
            multi_modal_data_no_watermark = await self.process_vision_info(messages_no_watermark)
            images_no_watermark = multi_modal_data_no_watermark.get("images", [])
            videos_no_watermark = multi_modal_data_no_watermark.get("videos", [])
            if initial_timestamps and videos_no_watermark:
                self._fix_video_metadata_timestamps(videos_no_watermark, [initial_timestamps], self.cache_fps)

        metrics = {}
        request_id = uuid4().hex

        # State variables
        prompt_ids = []
        response_mask = []
        accumulated_mm_inputs = {}
        accumulated_mm_inputs_no_watermark = {}

        user_turns = 0
        assistant_turns = 0

        # Tokenize initial prompt
        prompt_ids, initial_mm_inputs = await self.apply_chat_template(
            messages,
            images=images if images else None,
            videos=videos if videos else None,
        )
        accumulated_mm_inputs = _merge_multi_modal_inputs(accumulated_mm_inputs, initial_mm_inputs)

        # Process non-watermark version
        if use_watermark and messages_no_watermark:
            _, initial_mm_inputs_no_watermark = await self.apply_chat_template(
                messages_no_watermark,
                images=images_no_watermark if images_no_watermark else None,
                videos=videos_no_watermark if videos_no_watermark else None,
            )
            accumulated_mm_inputs_no_watermark = _merge_multi_modal_inputs(
                accumulated_mm_inputs_no_watermark, initial_mm_inputs_no_watermark
            )

        # Main reasoning loop
        for turn in range(self.max_assistant_turns):
            if user_turns >= self.max_user_turns:
                break

            if len(response_mask) >= self.response_length:
                break

            # Generate response
            with simple_timer("generate_sequences", metrics):
                generate_result = await self.server_manager.generate(
                    request_id=request_id,
                    prompt_ids=prompt_ids,
                    sampling_params=sampling_params,
                    image_data=images if images else None,
                    video_data=videos if videos else None,
                )

            response_ids = generate_result.token_ids
            assistant_turns += 1

            prompt_ids = prompt_ids + response_ids
            response_mask = response_mask + [1] * len(response_ids)

            # Decode response
            response_text = await self.loop.run_in_executor(
                None, lambda rid=response_ids: self.tokenizer.decode(rid, skip_special_tokens=True)
            )

            metrics[f"turn_{turn}_response_length"] = len(response_ids)

            # Check for answer
            answer = extract_answer(response_text)
            if answer:
                metrics["found_answer"] = True
                metrics["answer"] = answer
                break

            # Check for tool calls
            tool_calls = extract_tool_calls(response_text)
            if not tool_calls or not video_path:
                break

            metrics["found_tool_calls"] = True
            metrics["num_tool_calls"] = len(tool_calls)

            # Add assistant response to messages
            messages.append({"role": "assistant", "content": response_text})
            if messages_no_watermark:
                messages_no_watermark.append({"role": "assistant", "content": response_text})

            # Process tool calls and load frames
            with simple_timer("tool_calls", metrics):
                per_segment_frames = []
                total_frames = 0
                segments = [(tc["start_time"], tc["end_time"]) for tc in tool_calls]

                for start, end in segments:
                    seg_frames = self._get_frame_paths_with_timestamps_for_segments(video_path, [(start, end)])
                    per_segment_frames.append(seg_frames)
                    total_frames += len(seg_frames)

            metrics["num_frames"] = total_frames

            if total_frames == 0:
                break

            # Build tool_response message with per-segment video entries
            # Using "tool" role to match LongVT's chat template format
            content_list = [{"type": "text", "text": ""}]
            content_list_no_watermark = [{"type": "text", "text": ""}] if use_watermark else None

            all_absolute_timestamps = []

            for (start, end), seg_frames_with_ts in zip(segments, per_segment_frames):
                if not seg_frames_with_ts:
                    continue

                # Boundary check
                if start >= end:
                    end = start + 2.0
                if video_duration and end >= video_duration:
                    start = min(start, video_duration - 2.0)
                    end = video_duration

                seg_paths = [fp for fp, _ in seg_frames_with_ts]
                seg_timestamps = [ts for _, ts in seg_frames_with_ts]
                all_absolute_timestamps.append(seg_timestamps)

                content_list.append({"type": "text", "text": f"\nFrom {start}s to {end}s:"})
                if content_list_no_watermark:
                    content_list_no_watermark.append({"type": "text", "text": f"\nFrom {start}s to {end}s:"})

                video_params = {
                    "fps": self.segment_fps,
                    "max_frames": self.segment_max_frames,
                    "min_pixels": self.segment_min_pixels,
                    "max_pixels": self.segment_max_pixels,
                }

                if use_watermark:
                    watermarked_frames = self._add_watermarks_to_frames(seg_frames_with_ts)
                    content_list.append(
                        {
                            "type": "video",
                            "video": watermarked_frames,
                            **video_params,
                        }
                    )
                    content_list_no_watermark.append(
                        {
                            "type": "video",
                            "video": seg_paths,
                            **video_params,
                        }
                    )
                else:
                    content_list.append(
                        {
                            "type": "video",
                            "video": seg_paths,
                            **video_params,
                        }
                    )

            # Use "tool" role to trigger <tool_response> format in chat template
            tool_response_message = {"role": "tool", "content": content_list}
            messages.append(tool_response_message)

            tool_response_message_no_watermark = None
            if content_list_no_watermark:
                tool_response_message_no_watermark = {"role": "tool", "content": content_list_no_watermark}
                messages_no_watermark.append(tool_response_message_no_watermark)

            # Process tool response to extract videos
            obs_multi_modal = await self.process_vision_info([tool_response_message])
            obs_videos = obs_multi_modal.get("videos", [])

            # Fix video metadata timestamps
            self._fix_video_metadata_timestamps(obs_videos, all_absolute_timestamps, self.cache_fps)

            # Accumulate videos
            if videos is None:
                videos = []
            videos = videos + obs_videos

            # Tokenize tool response
            obs_ids, obs_mm_inputs = await self.apply_chat_template(
                [tool_response_message],
                images=None,
                videos=obs_videos if obs_videos else None,
                remove_system_prompt=True,
            )

            # Check if adding observation would exceed response length
            if len(response_mask) + len(obs_ids) >= self.response_length:
                break

            # Accumulate multi_modal_inputs
            accumulated_mm_inputs = _merge_multi_modal_inputs(accumulated_mm_inputs, obs_mm_inputs)

            # Process non-watermark tool response
            if use_watermark and tool_response_message_no_watermark:
                obs_multi_modal_no_watermark = await self.process_vision_info([tool_response_message_no_watermark])
                obs_videos_no_watermark = obs_multi_modal_no_watermark.get("videos", [])
                self._fix_video_metadata_timestamps(obs_videos_no_watermark, all_absolute_timestamps, self.cache_fps)

                if videos_no_watermark is None:
                    videos_no_watermark = []
                videos_no_watermark = videos_no_watermark + obs_videos_no_watermark

                _, obs_mm_inputs_no_watermark = await self.apply_chat_template(
                    [tool_response_message_no_watermark],
                    images=None,
                    videos=obs_videos_no_watermark if obs_videos_no_watermark else None,
                    remove_system_prompt=True,
                )
                accumulated_mm_inputs_no_watermark = _merge_multi_modal_inputs(
                    accumulated_mm_inputs_no_watermark, obs_mm_inputs_no_watermark
                )

            # Add tool response to prompt_ids
            prompt_ids = prompt_ids + obs_ids
            response_mask = response_mask + [0] * len(obs_ids)
            user_turns += 1

        # Prepare output
        initial_prompt_len = len(prompt_ids) - len(response_mask)
        final_prompt_ids = prompt_ids[:initial_prompt_len]
        final_response_ids = prompt_ids[initial_prompt_len:]

        output_multi_modal_data = {}
        if images:
            output_multi_modal_data["images"] = images
        if videos:
            output_multi_modal_data["videos"] = videos

        return AgentLoopOutput(
            prompt_ids=final_prompt_ids,
            response_ids=final_response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            multi_modal_data=output_multi_modal_data,
            accumulated_multi_modal_inputs=accumulated_mm_inputs,
            accumulated_multi_modal_inputs_no_watermark=accumulated_mm_inputs_no_watermark if use_watermark else None,
            num_turns=user_turns + assistant_turns + 1,
            metrics=AgentLoopMetrics(**metrics) if isinstance(metrics, dict) else metrics,
        )
