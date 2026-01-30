# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
FetchFramesTool for video reasoning - 方案二

This tool fetches video frames from cache based on time segments.
Used with ToolAgentLoop for video reasoning RL training.

IMPORTANT: This tool returns video paths (jpg files) in the video field,
which allows ToolAgentLoop to use {"type": "video", "video": paths} format
to generate <|video_pad|> token (aligned with SFT training).

Usage in tool_config.yaml:
    tools:
      - class_name: verl.tools.fetch_frames_tool.FetchFramesTool
        config:
          type: native
          cache_dir: .cache
          fps: 1
          max_frames: 512
          max_frames_per_segment: 16
        tool_schema:
          type: function
          function:
            name: fetch_frames
            description: "Fetch video frames for specified time segments"
            parameters:
              type: object
              properties:
                segments:
                  type: array
                  description: "List of [start, end] time pairs in seconds"
                  items:
                    type: array
              required:
                - segments
"""

import logging
import os
from typing import Any, Optional, List, Tuple
from uuid import uuid4

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse
from verl.utils.video_frame_cache import VideoFrameCache, CacheNotFoundError

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class FetchFramesTool(BaseTool):
    """
    Tool for fetching video frames from cache based on time segments.

    This tool is designed to work with ToolAgentLoop for video reasoning tasks.
    It loads pre-cached video frames (as jpg files) and returns them as video paths.

    Returns video paths in ToolResponse.video field, which allows ToolAgentLoop
    to use {"type": "video", "video": paths} format to generate <|video_pad|> token.

    Format aligned with eval_holmes_qwen_multiturn_spatial.py observation format.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        Initialize FetchFramesTool.

        Args:
            config: Tool configuration containing:
                - cache_dir: Directory for frame cache (default: .cache)
                - fps: Frames per second for sampling (default: 1)
                - max_frames: Maximum frames per video (default: 512)
                - max_frames_per_segment: Maximum frames per segment (default: 16)
            tool_schema: OpenAI format tool schema
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}

        # Cache configuration
        self.cache_dir = config.get("cache_dir", ".cache")
        self.fps = config.get("fps", 1)
        self.max_frames = config.get("max_frames", 512)
        self.max_frames_per_segment = config.get("max_frames_per_segment", 16)

        # Initialize frame cache
        self.frame_cache = VideoFrameCache(
            cache_dir=self.cache_dir,
            fps=self.fps,
            max_frames=self.max_frames,
        )

        logger.info(f"Initialized FetchFramesTool with config: {config}")

    async def create(
        self,
        instance_id: Optional[str] = None,
        create_kwargs: Optional[dict] = None,
        **kwargs
    ) -> tuple[str, ToolResponse]:
        """
        Create a tool instance for a trajectory.

        Args:
            instance_id: Optional unique identifier
            create_kwargs: Should contain:
                - video_path: Path to the video file
                - video_duration: Duration of the video in seconds (optional)

        Returns:
            Tuple of (instance_id, ToolResponse)
        """
        if instance_id is None:
            instance_id = str(uuid4())

        # Get video info from create_kwargs
        create_kwargs = create_kwargs or {}
        video_path = create_kwargs.get("video_path")
        video_duration = create_kwargs.get("video_duration")

        if video_path is None:
            logger.warning("Missing video_path in create_kwargs")

        self._instance_dict[instance_id] = {
            "video_path": video_path,
            "video_duration": video_duration,
        }

        return instance_id, ToolResponse()

    def _load_frame_paths_from_cache(
        self,
        video_path: str,
        segments: List[Tuple[float, float]],
        video_duration: Optional[float] = None,
    ) -> List[str]:
        """
        Load frame jpg paths from cache for given segments.

        Args:
            video_path: Path to video file
            segments: List of (start, end) time tuples
            video_duration: Video duration for boundary checking

        Returns:
            List of jpg file paths
        """
        # Validate and adjust segments
        validated_segments = []
        for start, end in segments:
            # Boundary check (like eval script)
            if start >= end:
                end = start + 2.0
            if video_duration and end >= video_duration:
                start = min(start, video_duration - 2.0)
                end = video_duration
            validated_segments.append((start, end))

        try:
            frame_paths = self.frame_cache.load_frame_paths(
                video_path,
                validated_segments,
                max_frames_per_segment=self.max_frames_per_segment,
            )
            return frame_paths
        except CacheNotFoundError:
            logger.warning(f"Cache not found for {video_path}")
            return []

    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        """
        Execute the tool to fetch frames for specified segments.

        Returns frame paths in ToolResponse.video field for video format processing.

        Args:
            instance_id: Tool instance ID
            parameters: Should contain:
                - segments: List of [start, end] time pairs

        Returns:
            Tuple of (ToolResponse, reward, metrics)
        """
        segments_raw = parameters.get("segments", [])

        # Validate segments parameter
        if not segments_raw:
            return (
                ToolResponse(text="Error: segments parameter is missing or empty."),
                -0.05,
                {"success": False},
            )

        # Convert to list of tuples
        try:
            segments = [(float(s[0]), float(s[1])) for s in segments_raw]
        except (TypeError, IndexError, ValueError) as e:
            return (
                ToolResponse(text=f"Error: Invalid segments format. Expected [[start, end], ...]. Error: {e}"),
                -0.05,
                {"success": False},
            )

        # Get instance data
        instance_data = self._instance_dict.get(instance_id, {})
        video_path = instance_data.get("video_path")
        video_duration = instance_data.get("video_duration")

        if not video_path:
            return (
                ToolResponse(text="Error: video_path not set. Make sure to call create() with video_path."),
                -0.05,
                {"success": False},
            )

        # Load frame paths from cache
        try:
            frame_paths = self._load_frame_paths_from_cache(video_path, segments, video_duration)
        except Exception as e:
            logger.error(f"Error loading frames: {e}")
            return (
                ToolResponse(text=f"Error loading frames: {e}"),
                -0.05,
                {"success": False},
            )

        if not frame_paths:
            return (
                ToolResponse(text="No frames could be loaded for the specified segments."),
                -0.05,
                {"success": False},
            )

        # Build response text (aligned with eval script observation format)
        # Format: Here are the cropped video segments.\nFrom Xs to Ys:
        response_parts = ["Here are the cropped video segments."]
        for start, end in segments:
            response_parts.append(f"\nFrom {start}s to {end}s:")
        response_text = "".join(response_parts)

        # Return frame paths in video field
        # This allows ToolAgentLoop to use {"type": "video", "video": paths} format
        # which generates <|video_pad|> token (aligned with SFT training)
        return (
            ToolResponse(
                video=[frame_paths],  # List of frame path lists (each list is one "video")
                text=response_text,
            ),
            0.0,  # Neutral reward for tool execution
            {"success": True, "num_frames": len(frame_paths), "num_segments": len(segments)},
        )

    async def release(self, instance_id: str, **kwargs) -> None:
        """Release the tool instance."""
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
