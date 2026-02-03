#!/usr/bin/env python3
"""
Video Reasoning Agent Loop for multi-turn video reasoning RL training.

This AgentLoop:
1. Parses <segment>[(start, end)]</segment> tags from model output (NOT OpenAI function calling)
2. Loads frames from cache as jpg files
3. Uses {"type": "video", "video": [jpg_paths]} format to generate <|video_pad|> token
   (aligned with SFT training)
4. Continues until <answer> is found or max turns reached

Format aligned with eval_holmes_qwen_multiturn_spatial.py
"""

import logging
import os
import re
from typing import Any, Optional, List, Tuple
from uuid import uuid4

import torch
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopOutput,
    AgentLoopMetrics,
    AsyncLLMServerManager,
    DictConfigWrap,
    register,
    _merge_multi_modal_inputs,
)
from verl.utils.video_frame_cache import VideoFrameCache, CacheNotFoundError
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


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

    Args:
        text: Model output text

    Returns:
        Extracted answer content or None
    """
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


@register("video_reasoning")
class VideoReasoningAgentLoop(AgentLoopBase):
    """
    Custom AgentLoop for video reasoning task.

    Unlike ToolAgentLoop which uses OpenAI function calling,
    this loop parses <segment> tags directly from the text.

    Uses {"type": "video", "video": [jpg_paths]} format to generate <|video_pad|> token
    (aligned with SFT training which uses video format).

    Multi-turn format (aligned with eval script):
    - Round 1: [video + prompt] -> model generates -> <segment>...</segment>
    - Round 2: [video + prompt + assistant_response + observation] -> model generates -> ...
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

        # Initial video configuration (for first frame load)
        # Default values aligned with eval script: --nframes=512, --max-pixels=12544
        initial_video_config = config.actor_rollout_ref.rollout.multi_turn.get("initial_video_config", {})
        self.initial_fps = initial_video_config.get("fps", 1)
        self.initial_max_frames = initial_video_config.get("max_frames", 512)
        self.initial_min_pixels = initial_video_config.get("min_pixels", 784)  # 28*28
        self.initial_max_pixels = initial_video_config.get("max_pixels", 12544)  # ~112x112

        # Segment video configuration (for segment frames)
        # Default values aligned with eval script: --segment-nframes=32, --segment-max-pixels=50176
        segment_video_config = config.actor_rollout_ref.rollout.multi_turn.get("segment_video_config", {})
        self.segment_fps = segment_video_config.get("fps", 1)
        self.segment_max_frames = segment_video_config.get("max_frames", 32)
        self.segment_min_pixels = segment_video_config.get("min_pixels", 784)  # 28*28
        self.segment_max_pixels = segment_video_config.get("max_pixels", 50176)  # ~224x224

        # Whether to use cached frames for initial video (saves CPU memory by skipping video decoding)
        self.use_cached_initial_video = cache_config.get("use_cached_initial_video", False)

        # Initialize frame cache (shared across all instances)
        self.frame_cache = VideoFrameCache(
            cache_dir=self.cache_dir,
            fps=self.cache_fps,
            max_frames=self.cache_max_frames,
        )

    def _get_frame_paths_for_segments(
        self,
        video_path: str,
        segments: List[Tuple[float, float]],
    ) -> List[str]:
        """
        Load frame jpg paths from cache for given segments.

        Returns paths that can be used with {"type": "video", "video": paths} format.
        """
        try:
            frame_paths = self.frame_cache.load_frame_paths(
                video_path,
                segments,
                max_frames_per_segment=self.max_frames_per_segment,
            )
            return frame_paths
        except CacheNotFoundError:
            logger.warning(f"Cache not found for {video_path}")
            return []

    def _get_initial_frame_paths(self, video_path: str) -> List[str]:
        """
        Load all cached frame paths for initial video (no segment filtering).

        This is used when use_cached_initial_video=True to avoid video decoding.
        Returns paths that can be used with {"type": "video", "video": paths} format.
        """
        try:
            # Load all frames without segment filtering (segments=None)
            frame_paths = self.frame_cache.load_frame_paths(
                video_path,
                segments=None,  # Load all cached frames
                max_frames_per_segment=self.initial_max_frames,  # Use initial_max_frames as limit
            )
            return frame_paths
        except CacheNotFoundError:
            logger.warning(f"Cache not found for {video_path}, falling back to original video")
            return []

    def _replace_video_with_cached_frames(
        self,
        messages: List[dict],
        frame_paths: List[str],
    ) -> List[dict]:
        """
        Replace video path in messages with cached frame paths.

        This modifies the messages in-place to use cached frames instead of original video.
        """
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "video":
                        # Replace video path with frame paths
                        item["video"] = frame_paths
        return messages

    def _inject_video_params(
        self,
        messages: List[dict],
        fps: int,
        max_frames: int,
        min_pixels: int,
        max_pixels: int,
    ) -> List[dict]:
        """
        Inject video parameters into messages containing video content.

        This allows parquet data to only store video paths, with resolution
        params injected at training time from config.

        Args:
            messages: List of message dicts
            fps: Frames per second for video sampling
            max_frames: Maximum number of frames to extract
            min_pixels: Minimum pixels for video frame
            max_pixels: Maximum pixels for video frame

        Returns:
            Modified messages with video params injected
        """
        for message in messages:
            content = message.get("content")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "video":
                        # Inject params if not already present
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
        """
        Main loop for video reasoning rollout.

        Flow (aligned with eval script):
        1. Load initial video and prompt
        2. Generate model response
        3. Add assistant response to prompt_ids and messages
        4. Check for <segment> tags, if found:
           - Load frame jpg paths from cache
           - Build observation message using {"type": "video", "video": [paths]} format
           - Add observation to prompt_ids
        5. Check for <answer> tags, if found terminate
        6. Continue until max turns or answer found
        """
        raw_prompt = kwargs.get("raw_prompt", [])
        extra_info = kwargs.get("extra_info", {})

        # Extract video info from extra_info
        video_path = extra_info.get("video_path")
        video_duration = extra_info.get("video_duration")

        messages = list(raw_prompt)

        # Inject initial video params from config (allows parquet to only store video path)
        self._inject_video_params(
            messages,
            fps=self.initial_fps,
            max_frames=self.initial_max_frames,
            min_pixels=self.initial_min_pixels,
            max_pixels=self.initial_max_pixels,
        )

        # If use_cached_initial_video is enabled, replace video path with cached frame paths
        # This avoids video decoding and reduces CPU memory usage
        if self.use_cached_initial_video and video_path:
            initial_frame_paths = self._get_initial_frame_paths(video_path)
            if initial_frame_paths:
                self._replace_video_with_cached_frames(messages, initial_frame_paths)
                logger.info(f"Using {len(initial_frame_paths)} cached frames for initial video")

        # Process initial vision info using parent class method
        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images", [])
        videos = multi_modal_data.get("videos", [])

        metrics = {}
        request_id = uuid4().hex

        # State variables (accumulated across turns, like ToolAgentLoop)
        prompt_ids = []
        response_mask = []
        response_logprobs = []
        accumulated_mm_inputs = {}  # Accumulated multi_modal_inputs from processor

        user_turns = 0
        assistant_turns = 0

        # Tokenize initial prompt and get multi_modal_inputs
        prompt_ids, initial_mm_inputs = await self.apply_chat_template(
            messages,
            images=images if images else None,
            videos=videos if videos else None,
        )
        accumulated_mm_inputs = _merge_multi_modal_inputs(accumulated_mm_inputs, initial_mm_inputs)

        # Main reasoning loop
        for turn in range(self.max_assistant_turns):
            if user_turns >= self.max_user_turns:
                break

            # Check if we've exceeded response length
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

            # Add response to prompt_ids (like ToolAgentLoop line 245)
            prompt_ids = prompt_ids + response_ids
            response_mask = response_mask + [1] * len(response_ids)

            # Decode response
            response_text = await self.loop.run_in_executor(
                None,
                lambda rid=response_ids: self.tokenizer.decode(rid, skip_special_tokens=True)
            )

            metrics[f"turn_{turn}_response_length"] = len(response_ids)

            # Check for answer - if found, we're done
            answer = extract_answer(response_text)
            if answer:
                metrics["found_answer"] = True
                metrics["answer"] = answer
                break

            # Check for segments
            segments = extract_segments(response_text)
            if not segments or not video_path:
                # No segments found and no answer, stop
                break

            metrics["found_segments"] = True
            metrics["segments"] = segments

            # Add assistant response to messages (like ToolAgentLoop line 269-270)
            messages.append({"role": "assistant", "content": response_text})

            # Load frame jpg paths from cache
            with simple_timer("tool_calls", metrics):
                frame_paths = self._get_frame_paths_for_segments(video_path, segments)
            metrics["num_frames"] = len(frame_paths)

            if not frame_paths:
                # No frames loaded, stop
                break

            # Build observation message using video format (generates <|video_pad|> token)
            # This aligns with SFT training which uses video format
            #
            # Format aligned with eval script:
            #   <observation>Here are the cropped video segments.
            #   From Xs to Ys: [video frames]
            #   </observation>
            content_list = [
                {"type": "text", "text": "<observation>Here are the cropped video segments."}
            ]

            # Add segment info and video for each segment
            for start, end in segments:
                # Boundary check (like eval script lines 497-501)
                if start >= end:
                    end = start + 2.0
                if video_duration and end >= video_duration:
                    start = min(start, video_duration - 2.0)
                    end = video_duration

                content_list.append({"type": "text", "text": f"\nFrom {start}s to {end}s:"})

            # Add video using jpg paths - this generates <|video_pad|> token
            # All segment frames as one video entry
            # Use segment config params (default aligned with eval script)
            content_list.append({
                "type": "video",
                "video": frame_paths,
                "fps": self.segment_fps,
                "max_frames": self.segment_max_frames,
                "min_pixels": self.segment_min_pixels,
                "max_pixels": self.segment_max_pixels,
            })
            content_list.append({"type": "text", "text": "\n</observation>"})

            observation_message = {"role": "user", "content": content_list}
            messages.append(observation_message)

            # Process observation message to extract videos in correct format
            # process_vision_info will load jpg files and return (tensor, metadata) tuples
            obs_multi_modal = await self.process_vision_info([observation_message])
            obs_videos = obs_multi_modal.get("videos", [])

            # Accumulate videos for output
            if videos is None:
                videos = []
            videos = videos + obs_videos

            # Tokenize observation message with processed videos
            obs_ids, obs_mm_inputs = await self.apply_chat_template(
                [observation_message],
                images=None,
                videos=obs_videos if obs_videos else None,
                remove_system_prompt=True,
            )

            # Accumulate multi_modal_inputs from observation
            accumulated_mm_inputs = _merge_multi_modal_inputs(accumulated_mm_inputs, obs_mm_inputs)

            # Check if adding observation would exceed response length
            if len(response_mask) + len(obs_ids) >= self.response_length:
                break

            # Add observation to prompt_ids (like ToolAgentLoop line 372-373)
            prompt_ids = prompt_ids + obs_ids
            response_mask = response_mask + [0] * len(obs_ids)
            user_turns += 1

        # Prepare output (like ToolAgentLoop line 189-190)
        # prompt_ids contains: [initial_prompt] + [response1] + [obs1] + [response2] + ...
        # response_mask marks which parts are LLM generated (1) vs observation (0)
        # Split: prompt_ids[:initial_len] is prompt, rest is response
        initial_prompt_len = len(prompt_ids) - len(response_mask)
        final_prompt_ids = prompt_ids[:initial_prompt_len]
        final_response_ids = prompt_ids[initial_prompt_len:]

        # Prepare multi_modal_data for output
        output_multi_modal_data = {}
        if images:
            output_multi_modal_data["images"] = images
        if videos:
            output_multi_modal_data["videos"] = videos

        return AgentLoopOutput(
            prompt_ids=final_prompt_ids,
            response_ids=final_response_ids[:self.response_length],
            response_mask=response_mask[:self.response_length],
            multi_modal_data=output_multi_modal_data,
            accumulated_multi_modal_inputs=accumulated_mm_inputs,
            num_turns=user_turns + assistant_turns + 1,
            metrics=AgentLoopMetrics(**metrics) if isinstance(metrics, dict) else metrics,
        )
