#!/usr/bin/env python3
"""Benchmark: cv2 seek vs decord get_batch for video frame caching."""

import time
import tempfile
import os
import json
import numpy as np
from PIL import Image
from pathlib import Path

VIDEO_PATH = "/data_gpu/zhengshurong/data/dataset/Video-Holmes/videos_cropped/033fKtGdpPc.mp4"
TARGET_FPS = 4
JPEG_QUALITY = 95


def cache_with_cv2(video_path, fps, output_dir):
    """Current implementation: cv2 seek per frame."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / video_fps if video_fps > 0 else 0
    effective_fps = min(fps, video_fps) if video_fps > 0 else fps
    total_frames = int(duration * effective_fps) + 1
    target_timestamps = [i / effective_fps for i in range(total_frames)]

    frames_info = []
    t0 = time.time()
    for saved_count, target_ts in enumerate(target_timestamps):
        target_frame_idx = int(target_ts * video_fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        frame_filename = f"frame_{saved_count:04d}.jpg"
        pil_image.save(os.path.join(output_dir, frame_filename), "JPEG", quality=JPEG_QUALITY)
        frames_info.append({
            'path': frame_filename,
            'timestamp': round(target_ts, 3),
            'index': saved_count,
        })
    cap.release()
    elapsed = time.time() - t0
    return elapsed, len(frames_info), duration, effective_fps


def cache_with_decord(video_path, fps, output_dir):
    """Optimized: decord batch read."""
    import decord
    decord.bridge.set_bridge("native")
    vr = decord.VideoReader(video_path, num_threads=1)
    video_fps = vr.get_avg_fps()
    total_video_frames = len(vr)
    duration = total_video_frames / video_fps if video_fps > 0 else 0
    effective_fps = min(fps, video_fps) if video_fps > 0 else fps
    total_frames = int(duration * effective_fps) + 1
    target_timestamps = [i / effective_fps for i in range(total_frames)]

    # Convert timestamps to frame indices
    frame_indices = [min(int(ts * video_fps), total_video_frames - 1) for ts in target_timestamps]

    t0 = time.time()
    # Batch decode all frames at once
    batch = vr.get_batch(frame_indices).asnumpy()  # (N, H, W, C) RGB

    frames_info = []
    for saved_count, (target_ts, frame_rgb) in enumerate(zip(target_timestamps, batch)):
        pil_image = Image.fromarray(frame_rgb)
        frame_filename = f"frame_{saved_count:04d}.jpg"
        pil_image.save(os.path.join(output_dir, frame_filename), "JPEG", quality=JPEG_QUALITY)
        frames_info.append({
            'path': frame_filename,
            'timestamp': round(target_ts, 3),
            'index': saved_count,
        })
    elapsed = time.time() - t0
    return elapsed, len(frames_info), duration, effective_fps


def verify_frames_match(dir1, dir2, num_check=5):
    """Spot-check that frames from both methods are similar."""
    files1 = sorted(os.listdir(dir1))
    files2 = sorted(os.listdir(dir2))
    if len(files1) != len(files2):
        print(f"  Frame count mismatch: cv2={len(files1)}, decord={len(files2)}")
        return False

    # Check a few frames for similarity
    check_indices = np.linspace(0, len(files1) - 1, min(num_check, len(files1)), dtype=int)
    for idx in check_indices:
        img1 = np.array(Image.open(os.path.join(dir1, files1[idx])))
        img2 = np.array(Image.open(os.path.join(dir2, files2[idx])))
        if img1.shape != img2.shape:
            print(f"  Shape mismatch at frame {idx}: {img1.shape} vs {img2.shape}")
            return False
        # Allow small differences due to different decoders
        diff = np.abs(img1.astype(float) - img2.astype(float)).mean()
        print(f"  Frame {idx} ({files1[idx]}): mean pixel diff = {diff:.2f}")
        if diff > 30:  # very loose threshold
            print(f"  WARNING: large difference at frame {idx}")
    return True


if __name__ == "__main__":
    print(f"Video: {VIDEO_PATH}")
    print(f"Target FPS: {TARGET_FPS}")
    print()

    with tempfile.TemporaryDirectory() as tmpdir:
        cv2_dir = os.path.join(tmpdir, "cv2")
        decord_dir = os.path.join(tmpdir, "decord")
        os.makedirs(cv2_dir)
        os.makedirs(decord_dir)

        # Benchmark cv2
        print("Running cv2 (current implementation)...")
        cv2_time, cv2_count, duration, eff_fps = cache_with_cv2(VIDEO_PATH, TARGET_FPS, cv2_dir)
        print(f"  Duration: {duration:.1f}s, effective_fps: {eff_fps}, frames: {cv2_count}")
        print(f"  Time: {cv2_time:.2f}s ({cv2_count / cv2_time:.1f} frames/sec)")
        print()

        # Benchmark decord
        print("Running decord (optimized)...")
        decord_time, decord_count, _, _ = cache_with_decord(VIDEO_PATH, TARGET_FPS, decord_dir)
        print(f"  Frames: {decord_count}")
        print(f"  Time: {decord_time:.2f}s ({decord_count / decord_time:.1f} frames/sec)")
        print()

        # Speedup
        print(f"Speedup: {cv2_time / decord_time:.1f}x faster with decord")
        print()

        # Verify correctness
        print("Verifying frame similarity...")
        verify_frames_match(cv2_dir, decord_dir)
