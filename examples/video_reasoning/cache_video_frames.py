#!/usr/bin/env python3
"""
Cache video frames as jpg files for video reasoning RL training.

The cached jpg files can be used with {"type": "video", "video": [paths]} format
to generate <|video_pad|> token (aligned with SFT training).

Usage:
    # From JSON file
    python cache_video_frames.py \
        --input_json ./long_video_data/results.json \
        --video_base_path /path/to/videos \
        --cache_dir .cache

    # From parquet file
    python cache_video_frames.py \
        --input_parquet ./long_video_data/train.parquet \
        --video_base_path /path/to/videos \
        --cache_dir .cache
"""

import argparse
import json
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from multiprocessing import Pool, cpu_count
from functools import partial

from verl.utils.video_frame_cache import VideoFrameCache


def cache_single_video(video_path: str, cache_dir: str, fps: int, max_frames: int) -> tuple:
    """Cache a single video. Used for multiprocessing."""
    try:
        cache = VideoFrameCache(cache_dir=cache_dir, fps=fps, max_frames=max_frames)
        info = cache.cache_video(video_path)
        return (video_path, info)
    except Exception as e:
        return (video_path, {'error': str(e)})


def parse_args():
    parser = argparse.ArgumentParser(description="Cache video frames as jpg files for training")
    parser.add_argument(
        "--input_json",
        type=str,
        help="Path to results.json file"
    )
    parser.add_argument(
        "--input_parquet",
        type=str,
        help="Path to parquet file (alternative to json)"
    )
    parser.add_argument(
        "--video_base_path",
        type=str,
        default=None,
        help="Base path to video files (optional if parquet has video_path column)"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=".cache",
        help="Directory to store cached frames"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=1,
        help="Sampling frequency (frames per second)"
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=512,
        help="Maximum frames to cache per video"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of parallel workers for caching (default: 8)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.input_json and not args.input_parquet:
        raise ValueError("Must provide either --input_json or --input_parquet")

    # Load data
    unique_videos = set()
    video_base_path = Path(args.video_base_path) if args.video_base_path else None

    if args.input_parquet:
        print(f"Loading data from {args.input_parquet}...")
        df = pd.read_parquet(args.input_parquet)
        # Prefer video_path column if exists, otherwise use video_id
        if "video_path" in df.columns:
            for video_path in df["video_path"].unique():
                unique_videos.add(video_path)
        else:
            if video_base_path is None:
                raise ValueError("--video_base_path required when parquet doesn't have video_path column")
            for video_id in df["video_id"].unique():
                video_path = str(video_base_path / f"{video_id}.mp4")
                unique_videos.add(video_path)
    else:
        print(f"Loading data from {args.input_json}...")
        if video_base_path is None:
            raise ValueError("--video_base_path required when using --input_json")
        with open(args.input_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for sample in data:
            video_id = sample["video_id"]
            video_path = str(video_base_path / f"{video_id}.mp4")
            unique_videos.add(video_path)

    video_list = sorted(list(unique_videos))
    print(f"Found {len(video_list)} unique videos")

    # Cache all videos with multiprocessing
    print(f"Caching frames to {args.cache_dir}/")
    print(f"Parameters: fps={args.fps}, max_frames={args.max_frames}, num_workers={args.num_workers}")
    print(f"Format: jpg files (for video_pad token alignment)")

    # Use multiprocessing for faster caching
    cache_fn = partial(
        cache_single_video,
        cache_dir=args.cache_dir,
        fps=args.fps,
        max_frames=args.max_frames
    )

    results = {}
    with Pool(processes=args.num_workers) as pool:
        for video_path, info in tqdm(
            pool.imap_unordered(cache_fn, video_list),
            total=len(video_list),
            desc="Caching videos"
        ):
            results[video_path] = info

    # Print summary
    print("\n" + "="*60)
    print("Caching Summary")
    print("="*60)
    success = sum(1 for r in results.values() if 'error' not in r)
    failed = len(results) - success

    print(f"Success: {success}")
    print(f"Failed: {failed}")

    if failed > 0:
        print("\nFailed videos:")
        for video, result in results.items():
            if 'error' in result:
                print(f"  {video}: {result['error']}")

    # Calculate cache size (now using directories with jpg files)
    cache_path = Path(args.cache_dir)
    total_size = 0
    for cache_subdir in cache_path.iterdir():
        if cache_subdir.is_dir():
            for f in cache_subdir.glob("*.jpg"):
                total_size += f.stat().st_size

    print("\nCache directory:", args.cache_dir)
    print(f"Cache size: {total_size / (1024**3):.2f} GB")
    print(f"Cache format: jpg files in {args.cache_dir}/{{video_name}}_fps{{fps}}_max{{max}}/ directories")


if __name__ == "__main__":
    main()
