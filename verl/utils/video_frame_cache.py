#!/usr/bin/env python3
"""
Video frame cache for video reasoning RL training.

Caches video frames as jpg files to disk to avoid repeated decoding during training.
Using jpg files allows using {"type": "video", "video": [paths]} format which
generates <|video_pad|> token (aligned with SFT training).
"""

import os
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import numpy as np


def add_timestamp_watermark(
    image: Image.Image,
    timestamp: float,
    position: str = "top_left",
    font_size: int = 0,
) -> Image.Image:
    """
    Add a timestamp watermark to an image.

    Args:
        image: PIL Image to add watermark to
        timestamp: Timestamp in seconds
        position: Watermark position, one of "top_left", "top_right", "bottom_left", "bottom_right"
        font_size: Font size for the timestamp text. If 0, auto-scales based on image height.

    Returns:
        PIL Image with watermark added (modifies in-place)
    """
    draw = ImageDraw.Draw(image)
    img_width, img_height = image.size

    # Format timestamp with 1 decimal (e.g., "12.5s")
    text = f"{timestamp:.1f}s"

    # Auto-scale font size based on image height if not specified
    if font_size <= 0:
        font_size = max(12, int(img_height * 0.06))

    # Try to load a TrueType font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except (IOError, OSError):
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        except (IOError, OSError):
            font = ImageFont.load_default()

    # Get text dimensions
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    # Calculate position
    margin = 12
    pad = 6

    if position == "top_right":
        x = img_width - tw - margin - pad
        y = margin
    elif position == "bottom_left":
        x = margin
        y = img_height - th - margin - pad
    elif position == "bottom_right":
        x = img_width - tw - margin - pad
        y = img_height - th - margin - pad
    else:
        # Default: top_left
        x = margin
        y = margin

    # Draw dark background rectangle and yellow text
    draw.rectangle([x - pad, y - pad, x + tw + pad, y + th + pad], fill=(0, 0, 0, 220))
    draw.text((x, y), text, fill=(255, 255, 0), font=font)

    return image


class VideoFrameCache:
    """
    Cache for video frames saved as jpg files.

    This allows using the frames with {"type": "video", "video": [jpg_paths]} format
    which generates <|video_pad|> token (aligned with SFT training).
    """

    def __init__(
        self,
        cache_dir: str = ".cache",
        fps: int = 1,
        max_frames: int = 512,
    ):
        """
        Initialize the video frame cache.

        Args:
            cache_dir: Directory to store cached frames
            fps: Sampling frequency (frames per second)
            max_frames: Maximum number of frames to cache per video
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.max_frames = max_frames

    def _get_cache_dir_for_video(self, video_path: str) -> Path:
        """Get the cache directory for a video."""
        video_name = Path(video_path).stem
        cache_name = f"{video_name}_fps{self.fps}_max{self.max_frames}"
        return self.cache_dir / cache_name

    def _get_metadata_path(self, video_path: str) -> Path:
        """Get the metadata file path for a video."""
        return self._get_cache_dir_for_video(video_path) / "metadata.json"

    def is_cached(self, video_path: str) -> bool:
        """Check if a video is already cached."""
        metadata_path = self._get_metadata_path(video_path)
        return metadata_path.exists()

    def load_frame_paths(
        self,
        video_path: str,
        segments: Optional[List[Tuple[float, float]]] = None,
        auto_cache: bool = True,
        max_frames_per_segment: int = 16,
    ) -> List[str]:
        """
        Load frame jpg file paths from cache.

        Returns paths that can be used with {"type": "video", "video": paths} format
        to generate <|video_pad|> token.

        Args:
            video_path: Path to the video file
            segments: Optional list of time segments to load. If None, loads all frames.
            auto_cache: If True, automatically cache the video if cache doesn't exist.
            max_frames_per_segment: Maximum number of frames to return per segment.

        Returns:
            List of jpg file paths
        """
        cache_dir = self._get_cache_dir_for_video(video_path)
        metadata_path = self._get_metadata_path(video_path)

        # Try to load from cache
        if not metadata_path.exists():
            if auto_cache:
                print(f"[VideoFrameCache] Cache miss for {video_path}")
                print(f"[VideoFrameCache] Parameters: fps={self.fps}, max_frames={self.max_frames}")
                print(f"[VideoFrameCache] Auto-caching...")
                self.cache_video(video_path)
            else:
                raise CacheNotFoundError(
                    f"Cache not found for {video_path} with fps={self.fps}, max_frames={self.max_frames}. "
                    f"Expected cache dir: {cache_dir}"
                )

        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        all_frame_info = metadata['frames']  # List of {"path": str, "timestamp": float}

        # Filter by segments if provided
        if segments is not None:
            filtered_paths = []
            for start, end in segments:
                segment_paths = []
                for frame_info in all_frame_info:
                    ts = frame_info['timestamp']
                    if start <= ts <= end:
                        segment_paths.append(str(cache_dir / frame_info['path']))
                        if len(segment_paths) >= max_frames_per_segment:
                            break
                filtered_paths.extend(segment_paths)
            return filtered_paths

        # Return all frame paths
        return [str(cache_dir / f['path']) for f in all_frame_info]

    def load_frame_paths_with_timestamps(
        self,
        video_path: str,
        segments: Optional[List[Tuple[float, float]]] = None,
        auto_cache: bool = True,
        max_frames_per_segment: int = 16,
    ) -> List[Tuple[str, float]]:
        """
        Load frame jpg file paths along with their timestamps from cache.

        This is useful for adding timestamp watermarks to frames.

        Args:
            video_path: Path to the video file
            segments: Optional list of time segments to load. If None, loads all frames.
            auto_cache: If True, automatically cache the video if cache doesn't exist.
            max_frames_per_segment: Maximum number of frames to return per segment.

        Returns:
            List of (path, timestamp) tuples
        """
        cache_dir = self._get_cache_dir_for_video(video_path)
        metadata_path = self._get_metadata_path(video_path)

        # Try to load from cache
        if not metadata_path.exists():
            if auto_cache:
                print(f"[VideoFrameCache] Cache miss for {video_path}")
                print(f"[VideoFrameCache] Parameters: fps={self.fps}, max_frames={self.max_frames}")
                print(f"[VideoFrameCache] Auto-caching...")
                self.cache_video(video_path)
            else:
                raise CacheNotFoundError(
                    f"Cache not found for {video_path} with fps={self.fps}, max_frames={self.max_frames}. "
                    f"Expected cache dir: {cache_dir}"
                )

        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        all_frame_info = metadata['frames']  # List of {"path": str, "timestamp": float}

        # Filter by segments if provided
        if segments is not None:
            filtered_results = []
            for start, end in segments:
                segment_results = []
                for frame_info in all_frame_info:
                    ts = frame_info['timestamp']
                    if start <= ts <= end:
                        segment_results.append((str(cache_dir / frame_info['path']), ts))
                        if len(segment_results) >= max_frames_per_segment:
                            break
                filtered_results.extend(segment_results)
            return filtered_results

        # Return all frame paths with timestamps
        return [(str(cache_dir / f['path']), f['timestamp']) for f in all_frame_info]

    def load_frames(
        self,
        video_path: str,
        segments: Optional[List[Tuple[float, float]]] = None,
        auto_cache: bool = True,
        max_frames_per_segment: int = 16,
    ) -> List[Tuple[float, Image.Image]]:
        """
        Load frames from cache as PIL Images (for backward compatibility).

        Args:
            video_path: Path to the video file
            segments: Optional list of time segments to load. If None, loads all frames.
            auto_cache: If True, automatically cache the video if cache doesn't exist.
            max_frames_per_segment: Maximum number of frames to return per segment.

        Returns:
            List of (timestamp, PIL.Image) tuples
        """
        cache_dir = self._get_cache_dir_for_video(video_path)
        metadata_path = self._get_metadata_path(video_path)

        # Try to load from cache
        if not metadata_path.exists():
            if auto_cache:
                print(f"[VideoFrameCache] Cache miss for {video_path}")
                self.cache_video(video_path)
            else:
                raise CacheNotFoundError(
                    f"Cache not found for {video_path}"
                )

        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        all_frame_info = metadata['frames']

        # Filter by segments if provided
        if segments is not None:
            filtered_frames = []
            for start, end in segments:
                count = 0
                for frame_info in all_frame_info:
                    ts = frame_info['timestamp']
                    if start <= ts <= end:
                        frame_path = cache_dir / frame_info['path']
                        img = Image.open(frame_path)
                        filtered_frames.append((ts, img))
                        count += 1
                        if count >= max_frames_per_segment:
                            break
            return filtered_frames

        # Return all frames
        result = []
        for frame_info in all_frame_info:
            frame_path = cache_dir / frame_info['path']
            img = Image.open(frame_path)
            result.append((frame_info['timestamp'], img))
        return result

    def cache_video(self, video_path: str) -> Dict:
        """
        Cache all frames from a video file as jpg files.

        Args:
            video_path: Path to the video file

        Returns:
            Dictionary with 'duration' and 'num_frames'

        Raises:
            FileNotFoundError: If video file does not exist
            RuntimeError: If video file cannot be opened
        """
        cache_dir = self._get_cache_dir_for_video(video_path)
        metadata_path = self._get_metadata_path(video_path)

        # Check if already cached
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            return {'duration': metadata['duration'], 'num_frames': len(metadata['frames'])}

        # Check if video file exists BEFORE attempting to open
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Create cache directory
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Decode and cache frames
        import cv2
        cap = cv2.VideoCapture(video_path)

        # Check if video was opened successfully
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {video_path}")

        # Get video properties
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / video_fps if video_fps > 0 else 0

        # Sample frames by timestamp (ensures clean integer timestamps)
        # Calculate target timestamps: 0, 1, 2, 3, ... seconds
        num_target_frames = min(int(duration * self.fps) + 1, self.max_frames)
        target_timestamps = [i / self.fps for i in range(num_target_frames)]

        frames_info = []
        for saved_count, target_ts in enumerate(target_timestamps):
            # Seek to target timestamp
            target_frame_idx = int(target_ts * video_fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)

            ret, frame = cap.read()
            if not ret:
                break

            # Use clean integer timestamp
            timestamp = int(target_ts) if target_ts == int(target_ts) else target_ts

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            # Save as jpg with integer timestamp
            frame_filename = f"frame_{saved_count:04d}_{int(target_ts)}s.jpg"
            frame_path = cache_dir / frame_filename
            pil_image.save(frame_path, "JPEG", quality=95)

            frames_info.append({
                'path': frame_filename,
                'timestamp': float(int(target_ts)),  # Clean integer timestamp
                'index': saved_count,
            })

        cap.release()

        # Save metadata
        metadata = {
            'video_path': video_path,
            'duration': duration,
            'fps': self.fps,
            'max_frames': self.max_frames,
            'frames': frames_info,
        }

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        return {'duration': duration, 'num_frames': len(frames_info)}

    def cache_videos_from_list(
        self,
        video_paths: List[str],
    ) -> Dict[str, Dict]:
        """
        Cache multiple videos.

        Args:
            video_paths: List of video file paths

        Returns:
            Dictionary mapping video paths to cache info
        """
        results = {}

        for video_path in tqdm(video_paths, desc="Caching videos"):
            try:
                info = self.cache_video(video_path)
                results[video_path] = info
            except Exception as e:
                print(f"Error caching {video_path}: {e}")
                results[video_path] = {'error': str(e)}

        return results

    def get_video_info(self, video_path: str) -> Optional[Dict]:
        """Get cached video info without loading frames."""
        metadata_path = self._get_metadata_path(video_path)

        if not metadata_path.exists():
            return None

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        return {
            'duration': metadata['duration'],
            'num_frames': len(metadata['frames']),
            'fps': self.fps,
        }

    def clear_cache(self, video_path: Optional[str] = None):
        """
        Clear cache for a video or all videos.

        Args:
            video_path: If provided, clears only this video's cache.
                       Otherwise clears all cached directories.
        """
        import shutil

        if video_path is None:
            # Clear all cache directories
            for cache_subdir in self.cache_dir.iterdir():
                if cache_subdir.is_dir():
                    shutil.rmtree(cache_subdir)
        else:
            cache_dir = self._get_cache_dir_for_video(video_path)
            if cache_dir.exists():
                shutil.rmtree(cache_dir)


class CacheNotFoundError(Exception):
    """Raised when cached data is not found."""
    pass


def create_frame_cache_for_dataset(
    video_paths: List[str],
    cache_dir: str = ".cache",
    fps: int = 1,
    max_frames: int = 512,
) -> VideoFrameCache:
    """
    Create a frame cache and cache all videos in a dataset.

    Args:
        video_paths: List of video file paths
        cache_dir: Directory to store cached frames
        fps: Sampling frequency
        max_frames: Maximum frames per video

    Returns:
        VideoFrameCache instance
    """
    cache = VideoFrameCache(cache_dir=cache_dir, fps=fps, max_frames=max_frames)
    cache.cache_videos_from_list(video_paths)
    return cache
