#!/usr/bin/env python3
"""
BBox Reward 评分方法对比测试

对比两种评分方法的合理性：
- 方法 1 (可视化评分): 在帧图片上画出预测的 bbox → VLM 打分 (0-10)
- 方法 2 (IOU 评分): 让 VLM 输出 GT bbox → 计算 IOU

Usage:
    python verl/utils/reward_score/test_bbox_comparison.py \
        --sample_file reward_logs/samples/reward_samples_20260203.jsonl \
        --num_samples 30 \
        --vlm_endpoint "10.96.11.3:8081" \
        --output_file bbox_comparison_results.json
"""

import re
import os
import io
import json
import base64
import random
import asyncio
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

import aiohttp
from PIL import Image

# Direct import to avoid heavy dependencies (ray, etc.)
import sys
_utils_dir = str(Path(__file__).resolve().parent.parent)  # verl/utils
if _utils_dir not in sys.path:
    sys.path.insert(0, _utils_dir)
from video_frame_cache import VideoFrameCache


# ============== Constants ==============

# GT bbox prompt (Qwen3-VL-30B, 输出 0-1000 范围)
BBOX_DETECT_PROMPT = """Look at this image carefully.

Task: Find and locate "{object_name}" in this image.
Context from video analysis: {context}

Instructions:
1. If "{object_name}" is NOT visible anywhere in this image, respond: None
2. If "{object_name}" IS visible, provide its bounding box in normalized [0,1] coordinates.

Output format (ONLY ONE of these):
- None
- [x1, y1, x2, y2]

where (x1,y1) = top-left corner, (x2,y2) = bottom-right corner, all values between 0 and 1.

Output ONLY the result, no explanation."""


# SFT 模型 prompt (Qwen2.5-VL-7B-sft, 输出归一化 [0,1] 坐标)
SFT_BBOX_DETECT_PROMPT = """Look at this image carefully.

Task: Find and locate "{object_name}" in this image.
Context: {context}

If the object is visible, provide its bounding box in normalized [0,1] coordinates.
If the object is NOT visible, respond: None

Output format:
- None
- [x1, y1, x2, y2]

where (x1,y1) = top-left corner, (x2,y2) = bottom-right corner, all values between 0 and 1.

Output ONLY the coordinates or None, no explanation."""

# Base 模型 prompt (Qwen2.5-VL-7B-base)
# Qwen2.5-VL 输出基于 resized image 的绝对像素坐标
BASE_BBOX_DETECT_PROMPT = """Look at this image carefully.

Task: Find and locate "{object_name}" in this image.
Context: {context}

If the object is visible, provide its bounding box coordinates.
If the object is NOT visible, respond: None

Output format:
- None
- [x1, y1, x2, y2]

where (x1,y1) = top-left corner, (x2,y2) = bottom-right corner.

Output ONLY the coordinates or None, no explanation."""

# Base 模型 prompt - 归一化版本
BASE_BBOX_DETECT_PROMPT_NORMALIZED = """Look at this image carefully.

Task: Find and locate "{object_name}" in this image.
Context: {context}

If the object is visible, provide its bounding box in NORMALIZED coordinates (values between 0 and 1).
- 0 means the left/top edge of the image
- 1 means the right/bottom edge of the image

If the object is NOT visible, respond: None

Output format:
- None
- [x1, y1, x2, y2]

where (x1,y1) = top-left corner, (x2,y2) = bottom-right corner, all values between 0.0 and 1.0.

Output ONLY the coordinates or None, no explanation."""

# 方法 1 使用的 prompt (与 video_reasoning_async.py 一致)
BBOX_VERIFY_PROMPT = """A red bounding box labeled "{object_name}" is drawn on the image.

Step 1: Is "{object_name}" visible ANYWHERE in this image frame?
- If "{object_name}" is NOT visible in the image at all, score 0 immediately.

Step 2: If "{object_name}" IS visible, evaluate the red bounding box:
- Does the box overlap with the object?
- Does the box tightly fit the object without too much extra background?
- Is the object fully contained in the box, or partially cut off?

Scoring guide:
0: "{object_name}" is NOT visible anywhere in this image frame
5: "{object_name}" is visible in the frame, but the box does NOT overlap with it (wrong location)
6: Box partially overlaps, but major offset or very loose (mostly background)
7: Box covers the object but too large (significant extra background) or partially cuts it off
8: Box mostly accurate, slight offset or slightly too large/small
9: Box accurately covers the object with minor imperfection
10: Box tightly and precisely fits the object, minimal extra background

Output only a single integer (0-10)."""


# ============== Utility Functions ==============

def image_to_base64(image: Image.Image, format: str = "JPEG") -> str:
    """将 PIL Image 转换为 base64 编码的 data URL"""
    buffer = io.BytesIO()
    image.save(buffer, format=format, quality=85)
    buffer.seek(0)
    img_bytes = buffer.read()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    mime_type = f"image/{format.lower()}"
    return f"data:{mime_type};base64,{img_base64}"


def extract_bbox_context(solution_str: str, object_name: str) -> str:
    """
    从 solution_str 提取 bbox 周围的上下文描述

    1. 找到 <obj>object_name</obj> 的位置
    2. 提取前后约 100 字符的文本
    3. 去掉 bbox 相关的标签
    """
    # 找到包含该 object 的句子
    # 注意：object_name 可能包含特殊字符，需要转义
    escaped_name = re.escape(object_name)
    pattern = rf'[^.]*<obj>{escaped_name}</obj><box>\[[^\]]+\]</box>at<t>[^<]+</t>[^.]*\.?'
    match = re.search(pattern, solution_str, re.IGNORECASE)
    if match:
        context = match.group(0)
        # 清理标签，保留纯文本
        context = re.sub(r'<obj>([^<]+)</obj>', r'\1', context)
        context = re.sub(r'<box>\[[^\]]+\]</box>at<t>[^<]+</t>', '', context)
        return context.strip()

    # 尝试宽松匹配
    pattern = rf'[^.]*{escaped_name}[^.]*\.?'
    match = re.search(pattern, solution_str, re.IGNORECASE)
    if match:
        return match.group(0).strip()

    return object_name  # 降级为只返回 object_name


def compute_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """计算两个 bbox 的 IOU (Intersection over Union)"""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union_area = bbox1_area + bbox2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def save_bbox_visualization(
    frame_path: str,
    pred_bbox: List[float],
    gt_bbox: Optional[List[float]],
    object_name: str,
    score1: float,
    score2: float,
    output_path: str,
    bbox_coord_range: float = 1.0,
    sft_bbox: Optional[List[float]] = None,
    base_bbox_pixel: Optional[List[float]] = None,  # Base (绝对坐标) 用像素坐标
    base_bbox_norm: Optional[List[float]] = None,   # Base (归一化) 用归一化坐标
    iou_sft: float = 0.0,
    iou_base_abs: float = 0.0,
    iou_base_norm: float = 0.0,
):
    """
    保存 bbox 可视化对比图 (支持 5 个 bbox)

    Args:
        frame_path: 原始帧图片路径
        pred_bbox: RL 模型预测的 bbox [x1, y1, x2, y2] (归一化坐标)
        gt_bbox: GT bbox from Qwen3-VL-30B (归一化坐标) 或 None
        object_name: 物体名称
        score1: 方法1分数 (可视化评分)
        score2: 方法2分数 (IOU with GT)
        output_path: 输出图片路径
        bbox_coord_range: bbox 坐标范围
        sft_bbox: SFT 模型输出的 bbox (归一化坐标) 或 None
        base_bbox_pixel: Base 模型 (绝对坐标 prompt) 输出的 bbox (像素坐标) 或 None
        base_bbox_norm: Base 模型 (归一化 prompt) 输出的 bbox (归一化坐标) 或 None
        iou_sft: SFT bbox 与 GT 的 IOU
        iou_base_abs: Base (绝对坐标) bbox 与 GT 的 IOU
        iou_base_norm: Base (归一化) bbox 与 GT 的 IOU
    """
    from PIL import ImageDraw, ImageFont

    img = Image.open(frame_path).convert("RGB")
    img_width, img_height = img.size
    draw = ImageDraw.Draw(img)

    # 加载字体
    try:
        font_size = max(14, min(img_width, img_height) // 35)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except Exception:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", font_size)
            except Exception:
                font = ImageFont.load_default()
        small_font_size = max(11, font_size - 3)
        try:
            small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", small_font_size)
        except Exception:
            small_font = font
    except Exception:
        font = ImageFont.load_default()
        small_font = font

    line_width = max(2, min(img_width, img_height) // 250)

    def draw_bbox_normalized(bbox, color, line_w=line_width):
        """绘制归一化坐标的 bbox"""
        if bbox is None:
            return
        x1, y1, x2, y2 = bbox
        px1 = int(x1 * img_width)
        py1 = int(y1 * img_height)
        px2 = int(x2 * img_width)
        py2 = int(y2 * img_height)
        draw.rectangle([px1, py1, px2, py2], outline=color, width=line_w)

    def draw_bbox_pixel(bbox, color, line_w=line_width):
        """绘制像素坐标的 bbox"""
        if bbox is None:
            return
        x1, y1, x2, y2 = bbox
        # 直接使用像素坐标
        px1 = int(x1)
        py1 = int(y1)
        px2 = int(x2)
        py2 = int(y2)
        # 裁剪到图片范围内
        px1 = max(0, min(px1, img_width - 1))
        py1 = max(0, min(py1, img_height - 1))
        px2 = max(0, min(px2, img_width - 1))
        py2 = max(0, min(py2, img_height - 1))
        if px1 < px2 and py1 < py2:
            draw.rectangle([px1, py1, px2, py2], outline=color, width=line_w)

    # 绘制 5 个 bbox (不同颜色)
    # 1. RL Pred bbox (红色) - 归一化坐标
    x1, y1, x2, y2 = pred_bbox
    x1_norm = x1 / bbox_coord_range
    y1_norm = y1 / bbox_coord_range
    x2_norm = x2 / bbox_coord_range
    y2_norm = y2 / bbox_coord_range
    draw_bbox_normalized([x1_norm, y1_norm, x2_norm, y2_norm], "red", line_width + 2)

    # 2. GT bbox (绿色/lime) - Qwen3-VL-30B，归一化坐标
    draw_bbox_normalized(gt_bbox, "lime", line_width + 1)

    # 3. SFT bbox (蓝色/cyan) - 归一化坐标
    draw_bbox_normalized(sft_bbox, "cyan", line_width)

    # 4. Base bbox 绝对坐标 (黄色) - 像素坐标，直接绘制
    draw_bbox_pixel(base_bbox_pixel, "yellow", line_width)

    # 5. Base bbox 归一化 (橙色) - 归一化坐标
    draw_bbox_normalized(base_bbox_norm, "orange", line_width)

    # 绘制图例和分数信息 (左上角)
    padding = 5

    # 构建信息文本
    info_text = [
        f"Object: {object_name[:25]}{'...' if len(object_name) > 25 else ''}",
        f"",
        f"--- IOU with GT ---",
        f"RL Pred:    {score2:.2f}",
        f"SFT:        {iou_sft:.2f}" + (" (None)" if sft_bbox is None else ""),
        f"Base(abs):  {iou_base_abs:.2f}" + (" (None)" if base_bbox_pixel is None else ""),
        f"Base(norm): {iou_base_norm:.2f}" + (" (None)" if base_bbox_norm is None else ""),
        f"",
        f"--- Legend ---",
        f"RED    = RL Pred",
        f"GREEN  = GT (30B)",
        f"CYAN   = SFT",
        f"YELLOW = Base(abs)",
        f"ORANGE = Base(norm)",
    ]

    # 计算文本区域大小
    max_text_width = 0
    total_height = 0
    for text in info_text:
        bbox = draw.textbbox((0, 0), text, font=small_font)
        max_text_width = max(max_text_width, bbox[2] - bbox[0])
        total_height += bbox[3] - bbox[1] + 3

    # 绘制半透明背景
    bg_x1, bg_y1 = 5, 5
    bg_x2, bg_y2 = bg_x1 + max_text_width + 2 * padding, bg_y1 + total_height + 2 * padding

    # 创建半透明覆盖层
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=(0, 0, 0, 200))
    img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
    draw = ImageDraw.Draw(img)

    # 绘制文本
    current_y = bg_y1 + padding
    for text in info_text:
        color = "white"
        if "RED" in text:
            color = "red"
        elif "GREEN" in text:
            color = "lime"
        elif "CYAN" in text:
            color = "cyan"
        elif "YELLOW" in text:
            color = "yellow"
        elif "ORANGE" in text:
            color = "orange"
        draw.text((bg_x1 + padding, current_y), text, fill=color, font=small_font)
        bbox = draw.textbbox((0, 0), text, font=small_font)
        current_y += bbox[3] - bbox[1] + 3

    # 保存图片
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path, "JPEG", quality=95)


def get_frame_for_timestamp(
    video_path: str,
    timestamp: float,
    cache_dir: str = ".cache",
    fps: int = 1,
    max_frames: int = 512,
) -> Optional[str]:
    """获取指定时间戳对应的帧图片路径"""
    cache = VideoFrameCache(cache_dir=cache_dir, fps=fps, max_frames=max_frames)

    # 获取缓存目录
    cache_video_dir = cache._get_cache_dir_for_video(video_path)
    metadata_path = cache._get_metadata_path(video_path)

    # 确保缓存存在
    if not metadata_path.exists():
        try:
            cache.cache_video(video_path)
        except Exception as e:
            print(f"Error caching video {video_path}: {e}")
            return None

    # 加载 metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # 找到最接近的帧
    frames_info = metadata['frames']
    closest_frame = None
    min_diff = float('inf')

    for frame_info in frames_info:
        diff = abs(frame_info['timestamp'] - timestamp)
        if diff < min_diff:
            min_diff = diff
            closest_frame = frame_info

    if closest_frame:
        return str(cache_video_dir / closest_frame['path'])
    return None


# ============== Method 1: Visual Scoring ==============

async def verify_single_bbox_with_vlm(
    frame_path: str,
    bbox: List[float],
    object_name: str,
    vlm_endpoint: str,
    vlm_model_name: str,
    vlm_api_key: str = "",
    bbox_coord_range: float = 1.0,
) -> Tuple[float, str]:
    """
    使用 VLM 验证单个 bbox 的准确性 (方法 1 - 可视化评分)

    Returns:
        (score, explanation): 分数(0-1) 和解释
    """
    score = 0.5
    response_text = ""

    try:
        # 1. 加载图片并绘制 bbox
        from PIL import ImageDraw, ImageFont

        img = Image.open(frame_path).convert("RGB")
        img_width, img_height = img.size

        # 将坐标归一化到 [0, 1] 再转为像素坐标
        x1, y1, x2, y2 = bbox
        x1_norm = x1 / bbox_coord_range
        y1_norm = y1 / bbox_coord_range
        x2_norm = x2 / bbox_coord_range
        y2_norm = y2 / bbox_coord_range

        x1_px = int(x1_norm * img_width)
        y1_px = int(y1_norm * img_height)
        x2_px = int(x2_norm * img_width)
        y2_px = int(y2_norm * img_height)

        draw = ImageDraw.Draw(img)

        # 绘制更醒目的 bbox：粗红框
        line_width = max(4, min(img_width, img_height) // 150)
        draw.rectangle([x1_px, y1_px, x2_px, y2_px], outline="red", width=line_width)

        # 绘制物体名称标签
        try:
            font_size = max(20, min(img_width, img_height) // 25)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
            except Exception:
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", font_size)
                except Exception:
                    font = ImageFont.load_default()

            label_text = f" {object_name} "
            label_y = max(0, y1_px - font_size - 8)

            text_bbox = draw.textbbox((x1_px, label_y), label_text, font=font)
            draw.rectangle(
                [text_bbox[0] - 2, text_bbox[1] - 2, text_bbox[2] + 2, text_bbox[3] + 2],
                fill="red"
            )
            draw.text((x1_px, label_y), label_text, fill="white", font=font)
        except Exception:
            draw.text((x1_px, max(0, y1_px - 15)), f"{object_name}", fill="red")

        # 2. 转换为 base64
        img_base64_url = image_to_base64(img, format="JPEG")

        # 3. 构建请求
        prompt = BBOX_VERIFY_PROMPT.format(object_name=object_name)

        payload = {
            "model": vlm_model_name,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": img_base64_url}},
                    {"type": "text", "text": prompt}
                ]
            }],
            "temperature": 0.3,
            "max_tokens": 16,
        }

        # 4. 发送请求
        headers = {"Content-Type": "application/json"}
        if vlm_api_key:
            headers["Authorization"] = f"Bearer {vlm_api_key}"

        timeout = aiohttp.ClientTimeout(total=60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            url = f"http://{vlm_endpoint}/v1/chat/completions"
            async with session.post(url, json=payload, headers=headers) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    response_text = result["choices"][0]["message"]["content"].strip()

                    # 解析分数
                    score_match = re.search(r'(\d+(?:\.\d+)?)', response_text)
                    if score_match:
                        score = float(score_match.group(1)) / 10.0  # 归一化到 0-1
                        score = min(1.0, max(0.0, score))
                else:
                    error_text = await resp.text()
                    response_text = f"HTTP error {resp.status}: {error_text[:200]}"

        return score, response_text

    except Exception as e:
        return 0.5, f"Error: {str(e)}"


# ============== Method 2: IOU Scoring ==============

async def get_gt_bbox_from_vlm(
    frame_path: str,
    object_name: str,
    context: str,
    vlm_endpoint: str,
    vlm_model_name: str,
    vlm_api_key: str = "",
    debug: bool = False,
) -> Tuple[Optional[List[float]], str]:
    """
    让 VLM 输出物体的 GT bbox (方法 2)

    Returns:
        (GT bbox [x1, y1, x2, y2] 或 None, raw_response)
    """
    raw_response = ""
    try:
        # 1. 加载图片并转 base64
        img = Image.open(frame_path).convert("RGB")
        img_base64_url = image_to_base64(img, format="JPEG")

        # 2. 构建请求
        prompt = BBOX_DETECT_PROMPT.format(object_name=object_name, context=context)

        payload = {
            "model": vlm_model_name,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": img_base64_url}},
                    {"type": "text", "text": prompt}
                ]
            }],
            "temperature": 0.1,  # 低 temperature 以获得更确定性的输出
            "max_tokens": 128,
        }

        # 3. 发送请求
        headers = {"Content-Type": "application/json"}
        if vlm_api_key:
            headers["Authorization"] = f"Bearer {vlm_api_key}"

        timeout = aiohttp.ClientTimeout(total=60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            url = f"http://{vlm_endpoint}/v1/chat/completions"
            async with session.post(url, json=payload, headers=headers) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    raw_response = result["choices"][0]["message"]["content"].strip()

                    if debug:
                        print(f"    [DEBUG] VLM raw response: {raw_response[:200]}")

                    # 解析返回 - 更宽松的 None 检测
                    if "none" in raw_response.lower() or "not visible" in raw_response.lower() or "cannot" in raw_response.lower():
                        return None, raw_response

                    # 尝试解析 bbox - 支持多种格式
                    # 格式1: [x1, y1, x2, y2]
                    bbox_match = re.search(r'\[([0-9.]+),\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)\]', raw_response)
                    if bbox_match:
                        bbox = [float(bbox_match.group(i)) for i in range(1, 5)]

                        # 检测并归一化坐标范围
                        max_val = max(bbox)
                        if max_val > 1:
                            # 假设是 0-1000 范围，归一化到 0-1
                            scale = 1000.0 if max_val <= 1000 else max_val
                            bbox = [v / scale for v in bbox]

                        # 验证 bbox 有效性
                        if all(0 <= v <= 1 for v in bbox) and bbox[0] < bbox[2] and bbox[1] < bbox[3]:
                            return bbox, raw_response

                    # 格式2: (x1, y1, x2, y2) 或带空格
                    bbox_match = re.search(r'\(([0-9.]+),\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)\)', raw_response)
                    if bbox_match:
                        bbox = [float(bbox_match.group(i)) for i in range(1, 5)]
                        max_val = max(bbox)
                        if max_val > 1:
                            scale = 1000.0 if max_val <= 1000 else max_val
                            bbox = [v / scale for v in bbox]
                        if all(0 <= v <= 1 for v in bbox) and bbox[0] < bbox[2] and bbox[1] < bbox[3]:
                            return bbox, raw_response

                    # 格式3: 四个独立的数字
                    numbers = re.findall(r'([0-9]+\.?[0-9]*)', raw_response)
                    if len(numbers) >= 4:
                        try:
                            bbox = [float(numbers[i]) for i in range(4)]
                            max_val = max(bbox)
                            if max_val > 1:
                                scale = 1000.0 if max_val <= 1000 else max_val
                                bbox = [v / scale for v in bbox]
                            if all(0 <= v <= 1 for v in bbox) and bbox[0] < bbox[2] and bbox[1] < bbox[3]:
                                return bbox, raw_response
                        except ValueError:
                            pass

                    return None, raw_response
                else:
                    error_text = await resp.text()
                    return None, f"HTTP error {resp.status}: {error_text[:100]}"

    except Exception as e:
        return None, f"Error: {str(e)}"


async def get_bbox_from_sft_model(
    frame_path: str,
    object_name: str,
    context: str,
    vlm_endpoint: str = "10.96.11.8:8081",
    vlm_model_name: str = "Qwen2.5-VL-7B-sft",
    vlm_api_key: str = "123456",
    debug: bool = False,
) -> Tuple[Optional[List[float]], str]:
    """
    从 SFT 模型获取 bbox (输出归一化 [0,1] 坐标)
    """
    raw_response = ""
    try:
        img = Image.open(frame_path).convert("RGB")
        img_base64_url = image_to_base64(img, format="JPEG")

        prompt = SFT_BBOX_DETECT_PROMPT.format(object_name=object_name, context=context)

        payload = {
            "model": vlm_model_name,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": img_base64_url}},
                    {"type": "text", "text": prompt}
                ]
            }],
            "temperature": 0.1,
            "max_tokens": 128,
        }

        headers = {"Content-Type": "application/json"}
        if vlm_api_key:
            headers["Authorization"] = f"Bearer {vlm_api_key}"

        timeout = aiohttp.ClientTimeout(total=60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            url = f"http://{vlm_endpoint}/v1/chat/completions"
            async with session.post(url, json=payload, headers=headers) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    raw_response = result["choices"][0]["message"]["content"].strip()

                    if debug:
                        print(f"    [DEBUG] SFT raw response: {raw_response[:200]}")

                    if "none" in raw_response.lower() or "not visible" in raw_response.lower():
                        return None, raw_response

                    # 解析 bbox - SFT 模型输出归一化坐标 [0,1]
                    bbox_match = re.search(r'\[([0-9.]+),\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)\]', raw_response)
                    if bbox_match:
                        bbox = [float(bbox_match.group(i)) for i in range(1, 5)]
                        # 如果值大于1，可能是 0-1000 范围，归一化
                        max_val = max(bbox)
                        if max_val > 1:
                            scale = 1000.0 if max_val <= 1000 else max_val
                            bbox = [v / scale for v in bbox]
                        if all(0 <= v <= 1 for v in bbox) and bbox[0] < bbox[2] and bbox[1] < bbox[3]:
                            return bbox, raw_response

                    return None, raw_response
                else:
                    error_text = await resp.text()
                    return None, f"HTTP error {resp.status}: {error_text[:100]}"

    except Exception as e:
        return None, f"Error: {str(e)}"


async def get_bbox_from_base_model(
    frame_path: str,
    object_name: str,
    context: str,
    vlm_endpoint: str = "10.96.11.8:8082",
    vlm_model_name: str = "Qwen2.5-VL-7B-base",
    vlm_api_key: str = "123456",
    debug: bool = False,
) -> Tuple[Optional[List[float]], Optional[List[float]], str]:
    """
    从 Base 模型获取 bbox (绝对像素坐标 prompt)
    Qwen2.5-VL 输出绝对像素坐标 (基于 resized image)

    Returns:
        (normalized_bbox, pixel_bbox, raw_response)
        - normalized_bbox: 归一化后的 bbox [0,1]，用于计算 IOU
        - pixel_bbox: 原始像素坐标，用于直接绘制
        - raw_response: 原始响应文本
    """
    raw_response = ""
    try:
        img = Image.open(frame_path).convert("RGB")
        img_width, img_height = img.size
        img_base64_url = image_to_base64(img, format="JPEG")

        prompt = BASE_BBOX_DETECT_PROMPT.format(
            object_name=object_name,
            context=context
        )

        payload = {
            "model": vlm_model_name,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": img_base64_url}},
                    {"type": "text", "text": prompt}
                ]
            }],
            "temperature": 0.1,
            "max_tokens": 128,
        }

        headers = {"Content-Type": "application/json"}
        if vlm_api_key:
            headers["Authorization"] = f"Bearer {vlm_api_key}"

        timeout = aiohttp.ClientTimeout(total=60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            url = f"http://{vlm_endpoint}/v1/chat/completions"
            async with session.post(url, json=payload, headers=headers) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    raw_response = result["choices"][0]["message"]["content"].strip()

                    if debug:
                        print(f"    [DEBUG] Base (abs) raw response: {raw_response[:200]}")

                    if "none" in raw_response.lower() or "not visible" in raw_response.lower():
                        return None, None, raw_response

                    # 解析 bbox
                    bbox_match = re.search(r'\[([0-9.]+),\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)\]', raw_response)
                    if bbox_match:
                        pixel_bbox = [float(bbox_match.group(i)) for i in range(1, 5)]

                        # 检查是否是有效的 bbox
                        if pixel_bbox[0] >= pixel_bbox[2] or pixel_bbox[1] >= pixel_bbox[3]:
                            return None, None, raw_response

                        # 归一化用于 IOU 计算
                        # Qwen2.5-VL 输出的是基于 resized image 的坐标
                        # 我们用原图尺寸归一化（假设模型按原图比例处理）
                        max_val = max(pixel_bbox)
                        if max_val <= 1:
                            # 已经是归一化坐标
                            normalized_bbox = pixel_bbox
                            # 转成像素坐标用于绘制
                            pixel_bbox = [
                                pixel_bbox[0] * img_width,
                                pixel_bbox[1] * img_height,
                                pixel_bbox[2] * img_width,
                                pixel_bbox[3] * img_height
                            ]
                        else:
                            # 是像素坐标，归一化
                            normalized_bbox = [
                                pixel_bbox[0] / img_width,
                                pixel_bbox[1] / img_height,
                                pixel_bbox[2] / img_width,
                                pixel_bbox[3] / img_height
                            ]
                            # 裁剪到 [0,1]
                            normalized_bbox = [min(1.0, max(0.0, v)) for v in normalized_bbox]

                        return normalized_bbox, pixel_bbox, raw_response

                    return None, None, raw_response
                else:
                    error_text = await resp.text()
                    return None, None, f"HTTP error {resp.status}: {error_text[:100]}"

    except Exception as e:
        return None, None, f"Error: {str(e)}"


async def get_bbox_from_base_model_normalized(
    frame_path: str,
    object_name: str,
    context: str,
    vlm_endpoint: str = "10.96.11.8:8082",
    vlm_model_name: str = "Qwen2.5-VL-7B-base",
    vlm_api_key: str = "123456",
    debug: bool = False,
) -> Tuple[Optional[List[float]], str]:
    """
    从 Base 模型获取 bbox (归一化坐标 prompt)
    让模型输出 [0,1] 归一化坐标

    Returns:
        (normalized_bbox, raw_response)
        - normalized_bbox: 归一化 bbox [0,1]
        - raw_response: 原始响应文本
    """
    raw_response = ""
    try:
        img = Image.open(frame_path).convert("RGB")
        img_base64_url = image_to_base64(img, format="JPEG")

        prompt = BASE_BBOX_DETECT_PROMPT_NORMALIZED.format(
            object_name=object_name,
            context=context
        )

        payload = {
            "model": vlm_model_name,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": img_base64_url}},
                    {"type": "text", "text": prompt}
                ]
            }],
            "temperature": 0.1,
            "max_tokens": 128,
        }

        headers = {"Content-Type": "application/json"}
        if vlm_api_key:
            headers["Authorization"] = f"Bearer {vlm_api_key}"

        timeout = aiohttp.ClientTimeout(total=60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            url = f"http://{vlm_endpoint}/v1/chat/completions"
            async with session.post(url, json=payload, headers=headers) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    raw_response = result["choices"][0]["message"]["content"].strip()

                    if debug:
                        print(f"    [DEBUG] Base (norm) raw response: {raw_response[:200]}")

                    if "none" in raw_response.lower() or "not visible" in raw_response.lower():
                        return None, raw_response

                    # 解析 bbox
                    bbox_match = re.search(r'\[([0-9.]+),\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)\]', raw_response)
                    if bbox_match:
                        bbox = [float(bbox_match.group(i)) for i in range(1, 5)]

                        # 检查是否是有效的 bbox
                        if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
                            return None, raw_response

                        # 如果值大于1，可能模型没有遵循指令，尝试归一化
                        max_val = max(bbox)
                        if max_val > 1:
                            # 可能是 0-1000 范围或像素坐标
                            if max_val <= 1000:
                                bbox = [v / 1000.0 for v in bbox]
                            else:
                                # 可能是像素坐标，但我们没有图片尺寸信息
                                # 这种情况下返回 None
                                return None, raw_response

                        # 验证归一化后的 bbox
                        if all(0 <= v <= 1 for v in bbox) and bbox[0] < bbox[2] and bbox[1] < bbox[3]:
                            return bbox, raw_response

                    return None, raw_response
                else:
                    error_text = await resp.text()
                    return None, f"HTTP error {resp.status}: {error_text[:100]}"

    except Exception as e:
        return None, f"Error: {str(e)}"


# ============== Data Loading ==============

def load_samples_with_bboxes(sample_file: str, num_samples: int = 30) -> List[Dict]:
    """
    从样本文件加载包含 bbox 的样本

    Args:
        sample_file: JSONL 文件路径
        num_samples: 要抽取的样本数量

    Returns:
        样本列表
    """
    samples = []

    with open(sample_file, 'r') as f:
        for line in f:
            try:
                sample = json.loads(line.strip())
                # 只选择包含 bbox 的样本
                if sample.get('bboxes') and len(sample['bboxes']) > 0:
                    samples.append(sample)
            except json.JSONDecodeError:
                continue

    print(f"Loaded {len(samples)} samples with bboxes")

    # 随机抽取
    if len(samples) > num_samples:
        samples = random.sample(samples, num_samples)

    return samples


# ============== Main Comparison Test ==============

async def run_comparison_test(
    sample_file: str,
    num_samples: int = 30,
    vlm_endpoint: str = "10.96.11.3:8081",
    vlm_model_name: str = "Qwen3-VL-30B-A3B-Instruct",
    vlm_api_key: str = "123456",
    cache_dir: str = ".cache",
    output_file: str = "bbox_comparison_results.json",
    vis_dir: str = "./bbox_vis_comparison",
    save_vis: bool = False,
):
    """
    主测试流程:
    1. 随机抽取 num_samples 个含 bbox 的样本
    2. 对每个 bbox:
       - 方法1: 调用 verify_single_bbox_with_vlm()
       - 方法2: get_gt_bbox_from_vlm() → compute_iou()
    3. 记录并输出对比结果
    4. (可选) 保存可视化图片
    """
    print(f"Loading samples from {sample_file}...")
    samples = load_samples_with_bboxes(sample_file, num_samples)
    print(f"Selected {len(samples)} samples for comparison")

    results = []
    total_bboxes = 0
    processed = 0

    for sample_idx, sample in enumerate(samples):
        video_path = sample.get('video_path')
        solution_str = sample.get('solution_str_full', '')
        video_id = sample.get('video_id', f'sample_{sample_idx}')

        if not video_path or not os.path.exists(video_path):
            print(f"Skip sample {video_id}: video not found at {video_path}")
            continue

        bboxes = sample.get('bboxes', [])
        total_bboxes += len(bboxes)

        for bbox_info in bboxes:
            obj_name = bbox_info.get('object', '')
            pred_bbox = bbox_info.get('bbox', [])
            timestamp = bbox_info.get('time', 0.0)

            if not obj_name or len(pred_bbox) != 4:
                continue

            # 获取帧图片路径
            frame_path = get_frame_for_timestamp(video_path, timestamp, cache_dir)
            if not frame_path or not os.path.exists(frame_path):
                print(f"  Skip bbox: frame not found for timestamp {timestamp}")
                continue

            # 提取 context
            context = extract_bbox_context(solution_str, obj_name)

            processed += 1
            print(f"\n[{processed}] Processing: {video_id}, object='{obj_name}', time={timestamp}s")
            print(f"    Context: {context[:80]}...")
            print(f"    Pred bbox: {pred_bbox}")

            # 方法 1: 可视化评分
            print("    Running Method 1 (visual scoring)...")
            score1, response1 = await verify_single_bbox_with_vlm(
                frame_path, pred_bbox, obj_name,
                vlm_endpoint, vlm_model_name, vlm_api_key
            )
            print(f"    Method 1 score: {score1:.2f} (response: {response1})")

            # 方法 2: IOU 评分
            print("    Running Method 2 (GT from Qwen3-VL-30B)...")
            gt_bbox, gt_response = await get_gt_bbox_from_vlm(
                frame_path, obj_name, context,
                vlm_endpoint, vlm_model_name, vlm_api_key,
                debug=True
            )

            if gt_bbox:
                score2 = compute_iou(pred_bbox, gt_bbox)
                print(f"    GT bbox: {gt_bbox}")
                print(f"    RL Pred IOU with GT: {score2:.2f}")
            else:
                score2 = 0.0  # VLM 认为物体不存在
                print(f"    GT bbox: None (VLM response: {gt_response[:150] if gt_response else 'empty'})")

            # 调用 SFT 模型
            print("    Running SFT model (Qwen2.5-VL-7B-sft)...")
            sft_bbox, sft_response = await get_bbox_from_sft_model(
                frame_path, obj_name, context,
                vlm_endpoint="10.96.11.8:8081",
                vlm_model_name="Qwen2.5-VL-7B-sft",
                debug=True
            )
            iou_sft = compute_iou(sft_bbox, gt_bbox) if (sft_bbox and gt_bbox) else 0.0
            if sft_bbox:
                print(f"    SFT bbox: {sft_bbox}, IOU with GT: {iou_sft:.2f}")
            else:
                print(f"    SFT bbox: None")

            # 调用 Base 模型 (绝对坐标 prompt)
            print("    Running Base model (abs coords prompt)...")
            base_bbox_abs, base_bbox_pixel, base_response_abs = await get_bbox_from_base_model(
                frame_path, obj_name, context,
                vlm_endpoint="10.96.11.8:8082",
                vlm_model_name="Qwen2.5-VL-7B-base",
                debug=True
            )
            iou_base_abs = compute_iou(base_bbox_abs, gt_bbox) if (base_bbox_abs and gt_bbox) else 0.0
            if base_bbox_abs:
                print(f"    Base (abs) bbox (norm): {[f'{v:.3f}' for v in base_bbox_abs]}, pixel: {base_bbox_pixel}, IOU: {iou_base_abs:.2f}")
            else:
                print(f"    Base (abs) bbox: None")

            # 调用 Base 模型 (归一化坐标 prompt)
            print("    Running Base model (norm coords prompt)...")
            base_bbox_norm, base_response_norm = await get_bbox_from_base_model_normalized(
                frame_path, obj_name, context,
                vlm_endpoint="10.96.11.8:8082",
                vlm_model_name="Qwen2.5-VL-7B-base",
                debug=True
            )
            iou_base_norm = compute_iou(base_bbox_norm, gt_bbox) if (base_bbox_norm and gt_bbox) else 0.0
            if base_bbox_norm:
                print(f"    Base (norm) bbox: {[f'{v:.3f}' for v in base_bbox_norm]}, IOU: {iou_base_norm:.2f}")
            else:
                print(f"    Base (norm) bbox: None")

            result = {
                'video_id': video_id,
                'object': obj_name,
                'timestamp': timestamp,
                'pred_bbox': pred_bbox,
                'gt_bbox': gt_bbox,
                'sft_bbox': sft_bbox,
                'base_bbox_abs': base_bbox_abs,
                'base_bbox_pixel': base_bbox_pixel,
                'base_bbox_norm': base_bbox_norm,
                'gt_response': gt_response,
                'sft_response': sft_response,
                'base_response_abs': base_response_abs,
                'base_response_norm': base_response_norm,
                'context': context,
                'frame_path': frame_path,
                'score_method1': score1,
                'iou_rl_gt': score2,
                'iou_sft_gt': iou_sft,
                'iou_base_abs_gt': iou_base_abs,
                'iou_base_norm_gt': iou_base_norm,
                'response_method1': response1,
            }

            # 保存可视化图片
            if save_vis:
                # 生成文件名: video_id_object_timestamp.jpg
                safe_obj_name = re.sub(r'[^\w\-]', '_', obj_name)[:30]
                vis_filename = f"{video_id}_{safe_obj_name}_{int(timestamp)}s_rl{score2:.0%}_sft{iou_sft:.0%}_babs{iou_base_abs:.0%}_bnorm{iou_base_norm:.0%}.jpg"
                vis_path = os.path.join(vis_dir, vis_filename)
                save_bbox_visualization(
                    frame_path=frame_path,
                    pred_bbox=pred_bbox,
                    gt_bbox=gt_bbox,
                    object_name=obj_name,
                    score1=score1,
                    score2=score2,
                    output_path=vis_path,
                    sft_bbox=sft_bbox,
                    base_bbox_pixel=base_bbox_pixel,
                    base_bbox_norm=base_bbox_norm,
                    iou_sft=iou_sft,
                    iou_base_abs=iou_base_abs,
                    iou_base_norm=iou_base_norm,
                )
                result['vis_path'] = vis_path
                print(f"    Saved visualization: {vis_path}")

            results.append(result)

    # 保存结果
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'sample_file': sample_file,
            'num_samples': num_samples,
            'vlm_endpoint': vlm_endpoint,
            'vlm_model_name': vlm_model_name,
            'sft_endpoint': '10.96.11.8:8081',
            'base_endpoint': '10.96.11.8:8082',
        },
        'summary': {},
        'results': results,
    }

    # 计算统计信息
    if results:
        iou_rl = [r['iou_rl_gt'] for r in results]
        iou_sft = [r['iou_sft_gt'] for r in results]
        iou_base_abs = [r['iou_base_abs_gt'] for r in results]
        iou_base_norm = [r['iou_base_norm_gt'] for r in results]

        def calc_stats(scores):
            if not scores:
                return {'mean': 0, 'std': 0}
            mean = sum(scores) / len(scores)
            std = (sum((s - mean)**2 for s in scores) / len(scores)) ** 0.5
            return {'mean': mean, 'std': std}

        output_data['summary'] = {
            'total_bboxes': len(results),
            'iou_rl_gt': calc_stats(iou_rl),
            'iou_sft_gt': calc_stats(iou_sft),
            'iou_base_abs_gt': calc_stats(iou_base_abs),
            'iou_base_norm_gt': calc_stats(iou_base_norm),
            'gt_detected_count': sum(1 for r in results if r['gt_bbox']),
            'sft_detected_count': sum(1 for r in results if r['sft_bbox']),
            'base_abs_detected_count': sum(1 for r in results if r['base_bbox_abs']),
            'base_norm_detected_count': sum(1 for r in results if r['base_bbox_norm']),
        }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print("MULTI-MODEL BBOX COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"Total bboxes evaluated: {len(results)}")
    if results:
        summary = output_data['summary']
        print(f"\n--- Detection Rate ---")
        print(f"  GT (30B) detected:   {summary['gt_detected_count']}/{len(results)}")
        print(f"  SFT detected:        {summary['sft_detected_count']}/{len(results)}")
        print(f"  Base (abs) detected: {summary['base_abs_detected_count']}/{len(results)}")
        print(f"  Base (norm) detected:{summary['base_norm_detected_count']}/{len(results)}")

        print(f"\n--- IOU with GT (Qwen3-VL-30B) ---")
        print(f"  RL Pred:     Mean={summary['iou_rl_gt']['mean']:.3f}, Std={summary['iou_rl_gt']['std']:.3f}")
        print(f"  SFT:         Mean={summary['iou_sft_gt']['mean']:.3f}, Std={summary['iou_sft_gt']['std']:.3f}")
        print(f"  Base (abs):  Mean={summary['iou_base_abs_gt']['mean']:.3f}, Std={summary['iou_base_abs_gt']['std']:.3f}")
        print(f"  Base (norm): Mean={summary['iou_base_norm_gt']['mean']:.3f}, Std={summary['iou_base_norm_gt']['std']:.3f}")

        # 打印 IOU 最低的 RL 样本 (问题最严重的)
        print(f"\n{'='*70}")
        print("TOP 5 WORST RL PREDICTIONS (Lowest IOU with GT)")
        print(f"{'='*70}")
        sorted_results = sorted(results, key=lambda x: x['iou_rl_gt'])[:5]
        for i, r in enumerate(sorted_results, 1):
            print(f"\n{i}. {r['video_id']} - '{r['object']}' @ {r['timestamp']}s")
            print(f"   IOU with GT:  RL={r['iou_rl_gt']:.2f}, SFT={r['iou_sft_gt']:.2f}, Base(abs)={r['iou_base_abs_gt']:.2f}, Base(norm)={r['iou_base_norm_gt']:.2f}")
            print(f"   RL Pred:      {r['pred_bbox']}")
            print(f"   GT:           {r['gt_bbox']}")
            print(f"   SFT:          {r['sft_bbox']}")
            print(f"   Base (abs):   {r['base_bbox_abs']}")
            print(f"   Base (norm):  {r['base_bbox_norm']}")

    print(f"\nResults saved to: {output_file}")
    if save_vis:
        print(f"Visualizations saved to: {vis_dir}/")
    return output_data


def compute_correlation(x: List[float], y: List[float]) -> float:
    """计算 Pearson 相关系数"""
    if len(x) != len(y) or len(x) == 0:
        return 0.0

    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n

    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y)) / n
    std_x = (sum((xi - mean_x)**2 for xi in x) / n) ** 0.5
    std_y = (sum((yi - mean_y)**2 for yi in y) / n) ** 0.5

    if std_x == 0 or std_y == 0:
        return 0.0

    return cov / (std_x * std_y)


def main():
    parser = argparse.ArgumentParser(description="Compare BBox scoring methods")
    parser.add_argument("--sample_file", type=str, required=True,
                        help="Path to JSONL sample file")
    parser.add_argument("--num_samples", type=int, default=30,
                        help="Number of samples to evaluate")
    parser.add_argument("--vlm_endpoint", type=str, default="10.96.11.3:8081",
                        help="VLM endpoint address")
    parser.add_argument("--vlm_model_name", type=str, default="Qwen3-VL-30B-A3B-Instruct",
                        help="VLM model name")
    parser.add_argument("--vlm_api_key", type=str, default="123456",
                        help="VLM API key")
    parser.add_argument("--cache_dir", type=str, default=".cache",
                        help="Frame cache directory")
    parser.add_argument("--output_file", type=str, default="bbox_comparison_results.json",
                        help="Output file for results")
    parser.add_argument("--vis_dir", type=str, default="./bbox_vis_comparison",
                        help="Directory to save visualization images")
    parser.add_argument("--save_vis", action="store_true",
                        help="Save visualization images")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling")

    args = parser.parse_args()

    # 设置随机种子
    random.seed(args.seed)

    # 运行测试
    asyncio.run(run_comparison_test(
        sample_file=args.sample_file,
        num_samples=args.num_samples,
        vlm_endpoint=args.vlm_endpoint,
        vlm_model_name=args.vlm_model_name,
        vlm_api_key=args.vlm_api_key,
        cache_dir=args.cache_dir,
        output_file=args.output_file,
        vis_dir=args.vis_dir,
        save_vis=args.save_vis,
    ))


if __name__ == "__main__":
    main()
