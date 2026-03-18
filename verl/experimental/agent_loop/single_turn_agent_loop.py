# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
import logging
import os
from typing import Any, List, Tuple
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.profiler import simple_timer
from verl.utils.video_frame_cache import VideoFrameCache, CacheNotFoundError

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("single_turn_agent")
class SingleTurnAgentLoop(AgentLoopBase):
    """Naive agent loop that only do single turn chat completion with video frame cache support."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.config.actor_rollout_ref.rollout.prompt_length
        self.response_length = self.config.actor_rollout_ref.rollout.response_length

        tool_config_path = self.config.data.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        self.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]

        # Video resolution configuration (read from multi_turn config for consistency)
        # These control the resolution of video frames to prevent OOM during training
        video_config = self.config.actor_rollout_ref.rollout.multi_turn.get("initial_video_config", {})
        self.video_fps = video_config.get("fps", 1)
        self.video_max_frames = video_config.get("max_frames", 512)
        self.video_min_pixels = video_config.get("min_pixels", 784)  # 28*28
        self.video_max_pixels = video_config.get("max_pixels", 12544)  # ~112x112

        # Frame cache configuration - enables loading pre-decoded jpg frames instead of video decoding
        cache_config = self.config.actor_rollout_ref.rollout.multi_turn.get("cache_config", {})
        self.cache_dir = cache_config.get("cache_dir", ".cache")
        self.cache_fps = cache_config.get("fps", 1)
        self.cache_max_frames = cache_config.get("max_frames", 512)
        self.use_cached_initial_video = cache_config.get("use_cached_initial_video", False)

        # Initialize frame cache if enabled
        self.frame_cache = None
        if self.use_cached_initial_video:
            self.frame_cache = VideoFrameCache(
                cache_dir=self.cache_dir,
                fps=self.cache_fps,
                max_frames=self.cache_max_frames,
            )
            logger.info(f"VideoFrameCache enabled: cache_dir={self.cache_dir}, fps={self.cache_fps}, max_frames={self.cache_max_frames}")

    def _get_cached_frame_paths_with_timestamps(self, video_path: str) -> List[Tuple[str, float]]:
        """
        Load cached frame paths with timestamps for the video.

        Returns:
            List of (frame_path, timestamp) tuples, or empty list if cache not found.
        """
        if not self.frame_cache:
            return []
        try:
            frame_paths_with_ts = self.frame_cache.load_frame_paths_with_timestamps(
                video_path,
                segments=None,  # Load all cached frames
                max_frames_per_segment=self.video_max_frames,
            )
            return frame_paths_with_ts
        except CacheNotFoundError:
            logger.warning(f"Cache not found for {video_path}, falling back to video decoding")
            return []

    def _replace_video_with_cached_frames(self, messages: List[dict], frame_paths: List[str]) -> None:
        """
        Replace video path in messages with cached frame paths (in-place).
        """
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "video":
                        item["video"] = frame_paths

    @staticmethod
    def _fix_video_metadata_timestamps(
        videos: list,
        timestamps: List[float],
        cache_fps: int,
    ) -> None:
        """
        Fix video_metadata.frames_indices in-place for correct absolute timestamps.

        When loading from cached jpg frames, fetch_video produces frames_indices=[0,1,2,...],
        causing the processor to emit relative timestamps. This method replaces them with
        indices derived from the known absolute timestamps of the cached frames.
        """
        if not videos or not timestamps:
            return
        for video_tensor, metadata in videos:
            # Convert timestamps to frame indices based on cache fps
            abs_indices = [int(round(ts * cache_fps)) for ts in timestamps]
            if "frames_indices" in metadata:
                metadata["frames_indices"] = abs_indices
            # Also update fps to match cache fps for correct timestamp generation
            metadata["fps"] = cache_fps

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
        params injected at training time from config to prevent OOM.

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
        messages = list(kwargs["raw_prompt"])

        # 0. inject video params to control resolution and prevent OOM
        self._inject_video_params(
            messages,
            fps=self.video_fps,
            max_frames=self.video_max_frames,
            min_pixels=self.video_min_pixels,
            max_pixels=self.video_max_pixels,
        )

        # 0.5. If video frame cache is enabled, replace video path with cached frame paths
        # This avoids expensive video decoding and significantly speeds up training
        initial_timestamps = None
        if self.use_cached_initial_video:
            # Get video_path from extra_info (same as video_reasoning_agent_loop)
            extra_info = kwargs.get("extra_info", {})
            video_path = extra_info.get("video_path")
            if video_path:
                frame_paths_with_ts = self._get_cached_frame_paths_with_timestamps(video_path)
                if frame_paths_with_ts:
                    frame_paths = [fp for fp, _ in frame_paths_with_ts]
                    initial_timestamps = [ts for _, ts in frame_paths_with_ts]
                    self._replace_video_with_cached_frames(messages, frame_paths)
                    logger.debug(f"Using {len(frame_paths)} cached frames for video: {video_path}")

        # 1. extract images and videos from messages
        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")

        # 1.5. Fix video_metadata.frames_indices for correct absolute timestamps
        # When loading from cached jpg frames, frames_indices=[0,1,2,...] causes
        # relative timestamps. This fix ensures correct absolute timestamps.
        if initial_timestamps and videos:
            self._fix_video_metadata_timestamps(videos, initial_timestamps, self.cache_fps)

        # 2. apply chat template and tokenize
        prompt_ids, _ = await self.apply_chat_template(
            messages,
            tools=self.tool_schemas,
            images=images,
            videos=videos,
        )

        # 3. generate sequences
        metrics = {}
        with simple_timer("generate_sequences", metrics):
            output = await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=images,
                video_data=videos,
            )
        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
        response_mask = [1] * len(output.token_ids)

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=output.token_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=output.log_probs[: self.response_length] if output.log_probs else None,
            routed_experts=(
                output.routed_experts[: len(prompt_ids) + self.response_length]
                if output.routed_experts is not None
                else None
            ),
            multi_modal_data=multi_modal_data,
            num_turns=2,
            metrics=metrics,
        )
        return output
