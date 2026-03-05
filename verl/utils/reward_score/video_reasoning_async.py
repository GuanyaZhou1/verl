#!/usr/bin/env python3
"""
异步奖励函数 - 支持 VLM bbox 验证和答案打分

This module provides an async reward function that evaluates:
1. Answer correctness (rule-based)
2. BBox accuracy (via VLM verification)
3. Answer quality (via VLM scoring)
"""
import re
import os
import io
import json
import base64
import uuid
import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

import aiohttp
from PIL import Image, ImageDraw


# ============== 默认配置常量 ==============
# 修改这里的值会影响所有使用默认参数的调用

DEFAULT_IOU_THRESHOLD = 0.0          # IOU 阈值，设为 0 不截断梯度
DEFAULT_TEMPORAL_WEIGHT = 0.5        # 时序奖励权重 (物体是否存在)
DEFAULT_SPATIAL_WEIGHT = 0.5         # 空间奖励权重 (IOU 分数)
DEFAULT_BBOX_COORD_RANGE = 1000.0    # bbox 坐标范围 (1.0 = [0,1], 1000.0 = [0,1000])
DEFAULT_ANSWER_WEIGHT = 0.4          # 答案分数权重
DEFAULT_BBOX_WEIGHT = 0.3            # bbox 分数权重
DEFAULT_VLM_WEIGHT = 0.3             # VLM 打分权重


# ============== 日志和样本保存 ==============

# 全局计数器和统计
_reward_stats = defaultdict(lambda: {
    "total_calls": 0,
    "bbox_found": 0,
    "bbox_verified": 0,
    "vlm_scored": 0,
    "answer_correct": 0,
    "total_score": 0.0,
})

_sample_counter = 0
_log_file_handle = None


def setup_reward_logging(log_dir: str = "./reward_logs"):
    """设置奖励函数日志"""
    global _log_file_handle

    os.makedirs(log_dir, exist_ok=True)

    # 设置 logger
    logger = logging.getLogger("video_reward")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # Don't propagate to root logger (avoids console output)

    # 避免重复添加 handler
    if not logger.handlers:
        # 文件 handler (所有日志写入文件)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"reward_{timestamp}.log")
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(fh)

        # 不添加控制台 handler，避免训练时大量输出
        logger.info(f"Reward logging initialized. Log file: {log_file}")

    return logger


def get_reward_logger():
    """获取奖励函数 logger"""
    logger = logging.getLogger("video_reward")
    if not logger.handlers:
        setup_reward_logging()
    return logger


def save_reward_sample(
    sample_data: Dict[str, Any],
    output_dir: str = "./reward_logs/samples",
):
    """
    保存单个奖励计算样本到 JSONL 文件

    Args:
        sample_data: 样本数据字典
        output_dir: 输出目录
    """
    global _sample_counter

    os.makedirs(output_dir, exist_ok=True)

    # 使用日期分文件
    date_str = datetime.now().strftime("%Y%m%d")
    output_file = os.path.join(output_dir, f"reward_samples_{date_str}.jsonl")

    _sample_counter += 1
    sample_data["sample_id"] = _sample_counter
    sample_data["timestamp"] = datetime.now().isoformat()

    with open(output_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(sample_data, ensure_ascii=False) + "\n")


def print_reward_stats():
    """打印奖励统计信息"""
    logger = get_reward_logger()

    for key, stats in _reward_stats.items():
        if stats["total_calls"] > 0:
            avg_score = stats["total_score"] / stats["total_calls"]
            acc = stats["answer_correct"] / stats["total_calls"]
            bbox_rate = stats["bbox_found"] / stats["total_calls"]

            logger.info(f"\n=== Reward Stats ({key}) ===")
            logger.info(f"  Total calls: {stats['total_calls']}")
            logger.info(f"  Avg score: {avg_score:.4f}")
            logger.info(f"  Answer accuracy: {acc:.4f}")
            logger.info(f"  BBox found rate: {bbox_rate:.4f}")
            logger.info(f"  BBox verified: {stats['bbox_verified']}")
            logger.info(f"  VLM scored: {stats['vlm_scored']}")


def image_to_base64(image: Image.Image, format: str = "JPEG") -> str:
    """
    将 PIL Image 转换为 base64 编码的 data URL

    Args:
        image: PIL Image 对象
        format: 图片格式 (JPEG, PNG 等)

    Returns:
        base64 data URL string (e.g., "data:image/jpeg;base64,...")
    """
    buffer = io.BytesIO()
    image.save(buffer, format=format, quality=85)
    buffer.seek(0)
    img_bytes = buffer.read()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    mime_type = f"image/{format.lower()}"
    return f"data:{mime_type};base64,{img_base64}"


# ============== BBox 提取 ==============

def extract_bboxes(text: str) -> List[Dict[str, Any]]:
    """
    从文本中提取 bbox 信息
    格式: <obj>object_name</obj><box>[x1,y1,x2,y2]</box>at<t>time_in_seconds</t>

    Returns:
        List of dicts: [{"object": str, "bbox": [x1,y1,x2,y2], "time": float}, ...]
    """
    pattern = r'<obj>(.*?)</obj><box>\[([\d.,\s]+)\]</box>at<t>([\d.]+)</t>'
    matches = re.findall(pattern, text, re.IGNORECASE)

    bboxes = []
    for obj_name, bbox_str, time_str in matches:
        try:
            coords = [float(x.strip()) for x in bbox_str.split(',')]
            if len(coords) == 4:
                bboxes.append({
                    'object': obj_name.strip(),
                    'bbox': coords,  # [x1, y1, x2, y2]
                    'time': float(time_str)
                })
        except ValueError:
            continue
    return bboxes


# ============== 帧加载和绘制 ==============

def get_frame_path_for_timestamp(
    video_path: str,
    timestamp: float,
    cache_dir: str = ".cache",
    fps: int = 1,
    max_frames: int = 512,
) -> Optional[str]:
    """
    从缓存中获取最接近指定时间戳的帧路径

    Args:
        video_path: 视频文件路径
        timestamp: 目标时间戳(秒)
        cache_dir: 帧缓存目录
        fps: 缓存帧的fps
        max_frames: 缓存的最大帧数

    Returns:
        帧文件路径，如果不存在则返回None
    """
    from verl.utils.video_frame_cache import VideoFrameCache

    cache = VideoFrameCache(cache_dir=cache_dir, fps=fps, max_frames=max_frames)

    # 尝试加载帧
    try:
        frame_paths = cache.load_frame_paths(video_path, segments=None, auto_cache=False)
        if not frame_paths:
            return None

        # 找到最接近指定时间戳的帧
        # 帧文件名格式: frame_0010_10s.jpg
        best_path = None
        best_diff = float('inf')

        for path in frame_paths:
            # 从文件名解析时间戳
            filename = os.path.basename(path)
            match = re.search(r'_(\d+)s\.jpg$', filename)
            if match:
                frame_ts = float(match.group(1))
                diff = abs(frame_ts - timestamp)
                if diff < best_diff:
                    best_diff = diff
                    best_path = path

        return best_path
    except Exception:
        return None


def draw_bbox_on_image(
    image_path: str,
    bbox: List[float],
    output_path: str,
    color: str = "red",
    width: int = 3,
) -> str:
    """
    在图片上绘制 bbox 并保存到新文件

    Args:
        image_path: 原始图片路径
        bbox: [x1, y1, x2, y2] 归一化坐标 (0-1)
        output_path: 输出图片路径
        color: 框的颜色
        width: 框的线宽

    Returns:
        输出图片路径
    """
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)

    # 获取图片尺寸
    img_width, img_height = img.size

    # 将归一化坐标转换为像素坐标
    x1, y1, x2, y2 = bbox
    x1_px = x1 * img_width
    y1_px = y1 * img_height
    x2_px = x2 * img_width
    y2_px = y2 * img_height

    # 绘制矩形框
    draw.rectangle([x1_px, y1_px, x2_px, y2_px], outline=color, width=width)

    # 保存图片
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path)

    return output_path


# ============== VLM bbox 验证 ==============

# Prompt: 让 VLM 先简短推理再输出 bbox（参考 long_ver5_zgy.py，[0,1000] 坐标范围更稳定）
BBOX_DETECT_PROMPT = """Find and locate "{object_name}" in this image.
{context_section}
Briefly analyze in <reasoning></reasoning>, then output bbox in <bbox></bbox>.

Rules:
- Coordinates in [0, 1000] range (0=top-left, 1000=bottom-right)
- If NOT visible: <bbox>None</bbox>
- Keep reasoning SHORT (1-2 sentences)

Example:
<reasoning>
The red car is in the lower-right area, ~650-850 horizontally, ~550-780 vertically.
</reasoning>
<bbox>[650, 550, 850, 780]</bbox>"""


def _format_bbox_detect_prompt(object_name: str, context: str = "") -> str:
    """格式化 bbox 检测 prompt"""
    context_section = ""
    if context:
        context_section = f"\nContext from video analysis:\n\"{context}\"\n"
    return BBOX_DETECT_PROMPT.format(
        object_name=object_name,
        context_section=context_section,
    )


def compute_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """
    计算两个 bbox 的 IOU (Intersection over Union)

    Args:
        bbox1: [x1, y1, x2, y2] 归一化坐标
        bbox2: [x1, y1, x2, y2] 归一化坐标

    Returns:
        IOU 值 (0-1)
    """
    # 计算交集区域
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    # 计算各自面积
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    # 计算并集面积
    union_area = bbox1_area + bbox2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def extract_bbox_context(solution_str: str, object_name: str) -> str:
    """
    从 solution_str 中提取与 bbox 相关的上下文信息

    Args:
        solution_str: 模型的完整输出
        object_name: 目标物体名称

    Returns:
        上下文字符串，帮助 VLM 理解要找什么物体
    """
    # 尝试提取 <think> 标签中的内容
    think_matches = re.findall(r'<think>(.*?)</think>', solution_str, re.DOTALL | re.IGNORECASE)

    context_parts = []
    for think_content in think_matches[-2:]:  # 取最后两个 think 块
        # 查找与物体相关的句子
        sentences = think_content.split('.')
        for sentence in sentences:
            if object_name.lower() in sentence.lower():
                context_parts.append(sentence.strip())

    if context_parts:
        return ". ".join(context_parts[:3])  # 最多3句

    return f"Looking for {object_name} in the video frame."


async def get_gt_bbox_from_vlm(
    frame_path: str,
    object_name: str,
    context: str,
    vlm_endpoint: str,
    vlm_model_name: str,
    vlm_api_key: str = "",
) -> Tuple[Optional[List[float]], str]:
    """
    调用 VLM 获取 GT bbox（使用 <reasoning>/<bbox> 格式，[0,1000] 坐标范围）

    Args:
        frame_path: 帧图片路径
        object_name: 目标物体名称
        context: 上下文信息
        vlm_endpoint: VLM 服务地址
        vlm_model_name: VLM 模型名称
        vlm_api_key: VLM API Key

    Returns:
        (gt_bbox, raw_response): GT bbox [x1,y1,x2,y2] 归一化到 [0,1]，以及原始响应
    """
    logger = get_reward_logger()

    try:
        # 加载图片并转为 base64
        img = Image.open(frame_path).convert("RGB")
        img_base64_url = image_to_base64(img, format="JPEG")

        # 使用 chain-of-thought prompt（参考 long_ver5_zgy.py）
        prompt = _format_bbox_detect_prompt(object_name, context)

        payload = {
            "model": vlm_model_name,
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": img_base64_url}
                    },
                    {"type": "text", "text": prompt}
                ]
            }],
            "temperature": 0.2,    # 略高温度允许更好的推理
            "max_tokens": 256,     # 简短推理 + bbox 足够
        }

        headers = {"Content-Type": "application/json"}
        if vlm_api_key:
            headers["Authorization"] = f"Bearer {vlm_api_key}"

        timeout = aiohttp.ClientTimeout(total=120)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            url = f"http://{vlm_endpoint}/v1/chat/completions"
            async with session.post(url, json=payload, headers=headers) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    response_text = result["choices"][0]["message"]["content"].strip()

                    # 使用多层解析器解析响应
                    gt_bbox = _parse_vlm_bbox_response(response_text)

                    if gt_bbox is not None:
                        # VLM 输出是 [0,1000] 范围，归一化到 [0,1]
                        gt_bbox = [c / 1000.0 for c in gt_bbox]
                        # 裁剪到 [0,1]
                        gt_bbox = [min(1.0, max(0.0, c)) for c in gt_bbox]
                        # 验证有效性
                        if gt_bbox[0] < gt_bbox[2] and gt_bbox[1] < gt_bbox[3]:
                            return gt_bbox, response_text
                        else:
                            logger.debug(f"Invalid GT bbox after normalization: {gt_bbox}")
                            return None, response_text

                    return None, response_text
                else:
                    error_text = await resp.text()
                    logger.warning(f"VLM detect HTTP error {resp.status}: {error_text[:100]}")
                    return None, f"HTTP error {resp.status}"

    except Exception as e:
        logger.warning(f"VLM detect exception: {str(e)}")
        return None, f"Error: {str(e)}"


def _parse_vlm_bbox_response(response: str) -> Optional[List[float]]:
    """
    多层解析 VLM bbox 响应（参考 long_ver5_zgy.py 的解析逻辑）

    支持格式:
    1. <bbox>[x1, y1, x2, y2]</bbox>  (推荐格式)
    2. <answer>[x1, y1, x2, y2]</answer>
    3. JSON: {"found": true, "bbox": [x1, y1, x2, y2]}
    4. 裸坐标: [x1, y1, x2, y2]

    返回 [0,1000] 范围的坐标，或 None
    """
    if not response:
        return None

    # 检查 None / not visible
    for tag in ['bbox', 'answer']:
        tag_match = re.search(rf'<{tag}>\s*(.*?)\s*</{tag}>', response, re.DOTALL)
        if tag_match:
            content = tag_match.group(1).strip()
            if content.lower() == 'none' or 'not visible' in content.lower():
                return None

    # 方法 1: 解析 <bbox>[x1, y1, x2, y2]</bbox> 或 <answer>[...]</answer>
    for tag in ['bbox', 'answer']:
        tag_match = re.search(rf'<{tag}>\s*\[([^\]]+)\]\s*</{tag}>', response, re.DOTALL)
        if tag_match:
            try:
                coords = [float(x.strip()) for x in tag_match.group(1).split(',')]
                if len(coords) == 4:
                    coords = [max(0, min(1000, c)) for c in coords]
                    if coords[0] < coords[2] and coords[1] < coords[3]:
                        return coords
            except (ValueError, TypeError):
                pass

    # 方法 1b: <bbox> 内有数字但格式不规范
    for tag in ['bbox', 'answer']:
        tag_match = re.search(rf'<{tag}>(.*?)</{tag}>', response, re.DOTALL)
        if tag_match:
            content = tag_match.group(1).strip()
            if content.lower() == 'none':
                return None
            numbers = re.findall(r'[\d.]+', content)
            if len(numbers) >= 4:
                try:
                    coords = [float(numbers[i]) for i in range(4)]
                    coords = [max(0, min(1000, c)) for c in coords]
                    if coords[0] < coords[2] and coords[1] < coords[3]:
                        return coords
                except (ValueError, TypeError):
                    pass

    # 方法 2: JSON 格式 {"found": true, "bbox": [...]}
    try:
        json_match = re.search(r'\{[^{}]*\}', response)
        if json_match:
            parsed = json.loads(json_match.group(0))
            if not parsed.get("found", False):
                return None
            bbox = parsed.get("bbox")
            if bbox and len(bbox) == 4:
                coords = [float(c) for c in bbox]
                coords = [max(0, min(1000, c)) for c in coords]
                if coords[0] < coords[2] and coords[1] < coords[3]:
                    return coords
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # 方法 3: 裸 [x1, y1, x2, y2]
    bracket_match = re.search(
        r'\[(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\]',
        response
    )
    if bracket_match:
        try:
            coords = [float(bracket_match.group(i)) for i in range(1, 5)]
            coords = [max(0, min(1000, c)) for c in coords]
            if coords[0] < coords[2] and coords[1] < coords[3]:
                return coords
        except (ValueError, TypeError):
            pass

    # 检查全局 None
    if 'none' in response.lower() and 'not visible' in response.lower():
        return None

    return None


def _get_neighboring_frame_paths(
    frame_path: str,
    video_path: str,
    bbox_timestamp: float,
    cache_dir: str,
    cache_fps: int,
    cache_max_frames: int,
    num_neighbors: int = 2,
) -> List[Tuple[str, float, bool]]:
    """
    Get neighboring frame paths around the target timestamp.

    Returns a list of (path, timestamp, is_target) tuples, sorted by timestamp.
    Always returns up to (2 * num_neighbors + 1) frames, handling boundary conditions.
    E.g. for timestamp=3 with num_neighbors=2: frames at t=1,2,3,4,5
         for timestamp=1 with num_neighbors=2: frames at t=0,1,2,3 (or fewer if 0 doesn't exist)
    """
    from verl.utils.video_frame_cache import VideoFrameCache

    cache = VideoFrameCache(cache_dir=cache_dir, fps=cache_fps, max_frames=cache_max_frames)
    try:
        all_frames_with_ts = cache.load_frame_paths_with_timestamps(
            video_path, segments=None, auto_cache=False
        )
    except Exception:
        return [(frame_path, bbox_timestamp, True)]

    if not all_frames_with_ts:
        return [(frame_path, bbox_timestamp, True)]

    # Find the index of the target frame (closest to bbox_timestamp)
    target_idx = min(
        range(len(all_frames_with_ts)),
        key=lambda i: abs(all_frames_with_ts[i][1] - bbox_timestamp)
    )

    # Collect neighbors: expand window to always get ~(2*num_neighbors+1) frames
    start = max(0, target_idx - num_neighbors)
    end = min(len(all_frames_with_ts), target_idx + num_neighbors + 1)
    # If near boundaries, expand the other side
    desired = 2 * num_neighbors + 1
    if end - start < desired:
        if start == 0:
            end = min(len(all_frames_with_ts), start + desired)
        elif end == len(all_frames_with_ts):
            start = max(0, end - desired)

    result = []
    for i in range(start, end):
        path, ts = all_frames_with_ts[i]
        is_target = (i == target_idx)
        result.append((path, ts, is_target))

    return result


async def verify_single_bbox_with_vlm(
    frame_path: str,
    bbox: List[float],
    object_name: str,
    context: str,
    vlm_endpoint: str,
    vlm_model_name: str,
    vlm_api_key: str = "",
    bbox_coord_range: float = DEFAULT_BBOX_COORD_RANGE,
    temporal_weight: float = DEFAULT_TEMPORAL_WEIGHT,
    spatial_weight: float = DEFAULT_SPATIAL_WEIGHT,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    save_visualization: bool = False,
    visualization_dir: str = "./reward_logs/bbox_vis",
    video_path: Optional[str] = None,
    bbox_timestamp: Optional[float] = None,
    cache_dir: str = ".cache",
    cache_fps: int = 1,
    cache_max_frames: int = 512,
) -> Tuple[float, float, float, Optional[List[float]], str, str, str, Optional[str]]:
    """
    使用 VLM 验证单个 bbox 的准确性（基于 IOU 的双维度奖励）

    Args:
        frame_path: 帧图片路径
        bbox: [x1, y1, x2, y2] 坐标
        object_name: 目标物体名称
        context: 上下文信息，帮助 VLM 理解要找什么物体
        vlm_endpoint: VLM 服务地址
        vlm_model_name: VLM 模型名称
        vlm_api_key: VLM API Key
        bbox_coord_range: bbox 坐标的范围 (1.0 = [0,1], 1000.0 = [0,1000])
        temporal_weight: 时序奖励权重
        spatial_weight: 空间奖励权重
        iou_threshold: IOU 阈值，低于此值 spatial_score=0
        save_visualization: 是否保存可视化图片
        visualization_dir: 可视化图片保存目录

    Returns:
        (total_score, temporal_score, spatial_score, gt_bbox, explanation, vlm_prompt, vlm_response, vis_path)
        vis_path: 可视化图片的绝对路径，如果未保存则为 None
    """
    logger = get_reward_logger()

    # 构建 VLM prompt（用于保存到 JSONL 调试）
    vlm_prompt = _format_bbox_detect_prompt(object_name, context)

    try:
        # 1. 调用 VLM 获取 GT bbox
        gt_bbox, raw_response = await get_gt_bbox_from_vlm(
            frame_path=frame_path,
            object_name=object_name,
            context=context,
            vlm_endpoint=vlm_endpoint,
            vlm_model_name=vlm_model_name,
            vlm_api_key=vlm_api_key,
        )

        # 智能检测 bbox 坐标范围
        # - 如果所有值都 <= 1，则是 [0,1] 归一化范围
        # - 否则是 [0,1000] 范围，需要先除以 1000 归一化
        if all(c <= 1.0 for c in bbox):
            effective_coord_range = 1.0
        else:
            effective_coord_range = 1000.0
            logger.debug(f"Auto-detected bbox in [0,1000] range: {bbox}")

        # 直接用 IOU 作为 bbox 分数（移除 temporal/spatial 拆分）
        # - IOU 已隐含时序准确性（时序错→物体不在帧中→GT 找不到→IOU=0）
        # - GT 返回 None 时 IOU=0（直接给 0 分，不再给免费 0.5）
        iou = 0.0
        if gt_bbox is not None:
            pred_normalized = [c / effective_coord_range for c in bbox]
            iou = compute_iou(pred_normalized, gt_bbox)

        total_score = iou
        temporal_score = 1.0 if gt_bbox is not None else 0.0  # 仅用于日志
        spatial_score = iou  # 仅用于日志

        # 5. 保存可视化图片（5帧横向拼接：目标帧±2邻近帧，目标帧加亮色边框）
        saved_vis_path = None
        if save_visualization:
            os.makedirs(visualization_dir, exist_ok=True)
            from PIL import ImageFont

            # 获取邻近帧路径
            if video_path and bbox_timestamp is not None:
                neighbor_frames = _get_neighboring_frame_paths(
                    frame_path, video_path, bbox_timestamp,
                    cache_dir, cache_fps, cache_max_frames, num_neighbors=2,
                )
            else:
                neighbor_frames = [(frame_path, bbox_timestamp or 0, True)]

            pred_normalized = [c / effective_coord_range for c in bbox]

            # 加载字体
            try:
                label_font_size = 16
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", label_font_size)
            except Exception:
                font = ImageFont.load_default()

            # 为每帧绘制 bbox 并标注时间戳
            annotated_frames = []
            border_width = 6
            for fpath, fts, is_target in neighbor_frames:
                fimg = Image.open(fpath).convert("RGB")
                fw, fh = fimg.size
                fdraw = ImageDraw.Draw(fimg)
                line_w = max(2, min(fw, fh) // 200)

                # 绘制预测 bbox（红色）
                ppx = [
                    int(pred_normalized[0] * fw), int(pred_normalized[1] * fh),
                    int(pred_normalized[2] * fw), int(pred_normalized[3] * fh),
                ]
                fdraw.rectangle(ppx, outline="red", width=line_w)

                # 绘制 GT bbox（绿色）
                if gt_bbox is not None:
                    gpx = [
                        int(gt_bbox[0] * fw), int(gt_bbox[1] * fh),
                        int(gt_bbox[2] * fw), int(gt_bbox[3] * fh),
                    ]
                    fdraw.rectangle(gpx, outline="green", width=line_w)

                # 顶部时间戳标签
                ts_text = f"{fts:.0f}s"
                ts_bbox = fdraw.textbbox((0, 0), ts_text, font=font)
                ts_w = ts_bbox[2] - ts_bbox[0]
                ts_x = (fw - ts_w) // 2
                fdraw.rectangle([ts_x - 4, 2, ts_x + ts_w + 4, ts_bbox[3] - ts_bbox[1] + 6], fill="black")
                fdraw.text((ts_x, 3), ts_text, fill="yellow", font=font)

                # 目标帧：加亮色（cyan）边框 + "TARGET" 标记
                if is_target:
                    for offset in range(border_width):
                        fdraw.rectangle(
                            [offset, offset, fw - 1 - offset, fh - 1 - offset],
                            outline="cyan",
                        )
                    tag = "TARGET"
                    tag_bbox = fdraw.textbbox((0, 0), tag, font=font)
                    tag_w = tag_bbox[2] - tag_bbox[0]
                    tag_x = (fw - tag_w) // 2
                    tag_y = fh - (tag_bbox[3] - tag_bbox[1]) - 8
                    fdraw.rectangle([tag_x - 4, tag_y - 2, tag_x + tag_w + 4, tag_y + (tag_bbox[3] - tag_bbox[1]) + 4], fill="cyan")
                    fdraw.text((tag_x, tag_y), tag, fill="black", font=font)

                annotated_frames.append(fimg)

            # 横向拼接所有帧
            total_w = sum(f.width for f in annotated_frames) + 2 * (len(annotated_frames) - 1)
            max_h = max(f.height for f in annotated_frames)
            # 顶部留空间给总评分信息
            info_height = 50
            strip = Image.new("RGB", (total_w, max_h + info_height), color=(40, 40, 40))

            # 在顶部绘制评分信息
            strip_draw = ImageDraw.Draw(strip)
            try:
                info_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
            except Exception:
                info_font = font
            info_text = (
                f"Object: {object_name[:40]}  |  IOU: {iou:.2f}  |  "
                f"Red=Pred  Green=GT  Cyan border=TARGET frame"
            )
            strip_draw.text((8, 8), info_text, fill="white", font=info_font)

            # 粘贴帧
            x_offset = 0
            for i, fimg in enumerate(annotated_frames):
                strip.paste(fimg, (x_offset, info_height))
                x_offset += fimg.width + 2  # 2px gap

            vis_filename = f"bbox_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}_iou{iou:.2f}.jpg"
            saved_vis_path = os.path.abspath(os.path.join(visualization_dir, vis_filename))
            strip.save(saved_vis_path, "JPEG", quality=90)
            logger.debug(f"Saved bbox vis strip: {saved_vis_path}")
        else:
            saved_vis_path = None

        explanation = f"temporal={temporal_score:.1f}, spatial={spatial_score:.2f}, iou={iou:.2f}, gt_bbox={gt_bbox}, vlm_response={raw_response[:100]}"
        logger.debug(f"BBox verify: obj={object_name}, pred={bbox}, {explanation}")

        return total_score, temporal_score, spatial_score, gt_bbox, explanation, vlm_prompt, raw_response, saved_vis_path

    except Exception as e:
        logger.warning(f"BBox verify exception: {str(e)}")
        return 0.0, 0.0, 0.0, None, f"Error: {str(e)}", vlm_prompt, "", None


async def verify_bboxes_with_vlm(
    bboxes: List[Dict],
    video_path: str,
    solution_str: str,
    vlm_endpoint: str,
    vlm_model_name: str,
    vlm_api_key: str = "",
    cache_dir: str = ".cache",
    cache_fps: int = 1,
    cache_max_frames: int = 512,
    save_bbox_visualization: bool = False,
    bbox_vis_sample_rate: float = 0.1,
    visualization_dir: str = "./reward_logs/bbox_vis",
    bbox_coord_range: float = DEFAULT_BBOX_COORD_RANGE,
    temporal_weight: float = DEFAULT_TEMPORAL_WEIGHT,
    spatial_weight: float = DEFAULT_SPATIAL_WEIGHT,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> Tuple[float, float, float, List[Dict]]:
    """
    验证所有 bbox 并返回平均分数（基于 IOU 的双维度奖励）

    Args:
        bboxes: 从模型输出提取的 bbox 列表
        video_path: 视频路径（用于加载帧）
        solution_str: 模型的完整输出（用于提取上下文）
        vlm_endpoint: VLM 服务地址
        vlm_model_name: VLM 模型名称
        vlm_api_key: VLM API Key
        cache_dir: 帧缓存目录
        cache_fps: 缓存帧的fps
        cache_max_frames: 缓存的最大帧数
        save_bbox_visualization: 是否保存 bbox 可视化图片
        bbox_vis_sample_rate: 可视化采样率 (0.1 = 10%)
        visualization_dir: 可视化图片保存目录
        bbox_coord_range: bbox 坐标范围
        temporal_weight: 时序奖励权重
        spatial_weight: 空间奖励权重
        iou_threshold: IOU 阈值

    Returns:
        (avg_total_score, avg_temporal_score, avg_spatial_score, details)
    """
    logger = get_reward_logger()

    if not bboxes:
        return 0.0, 0.0, 0.0, []  # 没有 bbox = 0 分（不再给免费 0.5）

    details = []
    total_scores = []
    temporal_scores = []
    spatial_scores = []

    # 并行验证所有 bbox
    tasks = []
    valid_bbox_indices = []

    import random
    for i, bbox_info in enumerate(bboxes):
        frame_path = get_frame_path_for_timestamp(
            video_path,
            bbox_info['time'],
            cache_dir,
            fps=cache_fps,
            max_frames=cache_max_frames,
        )
        if frame_path and os.path.exists(frame_path):
            # 提取上下文
            context = extract_bbox_context(solution_str, bbox_info['object'])
            # 按采样率决定是否保存可视化
            should_save_vis = save_bbox_visualization and (random.random() < bbox_vis_sample_rate)
            tasks.append(verify_single_bbox_with_vlm(
                frame_path=frame_path,
                bbox=bbox_info['bbox'],
                object_name=bbox_info['object'],
                context=context,
                vlm_endpoint=vlm_endpoint,
                vlm_model_name=vlm_model_name,
                vlm_api_key=vlm_api_key,
                bbox_coord_range=bbox_coord_range,
                temporal_weight=temporal_weight,
                spatial_weight=spatial_weight,
                iou_threshold=iou_threshold,
                save_visualization=should_save_vis,
                visualization_dir=visualization_dir,
                video_path=video_path,
                bbox_timestamp=bbox_info['time'],
                cache_dir=cache_dir,
                cache_fps=cache_fps,
                cache_max_frames=cache_max_frames,
            ))
            valid_bbox_indices.append((i, frame_path))  # 同时保存 frame_path
        else:
            logger.debug(f"Frame not found for timestamp {bbox_info['time']}, video={video_path}")

    if tasks:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for (idx, frame_path), result in zip(valid_bbox_indices, results):
            bbox_info = bboxes[idx]
            if isinstance(result, Exception):
                total_score, temporal_score, spatial_score = 0.0, 0.0, 0.0
                gt_bbox, explanation, vlm_prompt, vlm_response, vis_path = None, str(result), "", "", None
            else:
                total_score, temporal_score, spatial_score, gt_bbox, explanation, vlm_prompt, vlm_response, vis_path = result

            total_scores.append(total_score)
            temporal_scores.append(temporal_score)
            spatial_scores.append(spatial_score)
            details.append({
                "bbox_info": bbox_info,
                "total_score": total_score,
                "temporal_score": temporal_score,
                "spatial_score": spatial_score,
                "gt_bbox": gt_bbox,
                "explanation": explanation[:200] if explanation else "",
                "vlm_prompt": vlm_prompt,
                "vlm_response": vlm_response,
                "frame_path": os.path.abspath(frame_path),  # 原始帧图片的绝对路径
                "vis_path": vis_path,  # 可视化图片的绝对路径 (如果保存了)
            })

    avg_total = sum(total_scores) / len(total_scores) if total_scores else 0.0
    avg_temporal = sum(temporal_scores) / len(temporal_scores) if temporal_scores else 0.0
    avg_spatial = sum(spatial_scores) / len(spatial_scores) if spatial_scores else 0.0

    logger.debug(f"BBox verification: {len(tasks)} tasks, avg_total={avg_total:.4f}, avg_temporal={avg_temporal:.4f}, avg_spatial={avg_spatial:.4f}")
    return avg_total, avg_temporal, avg_spatial, details


# ============== 答案评分 (VLM) ==============

# English prompt - just check if answer is correct (for open-ended questions)
ANSWER_SCORE_PROMPT = """Compare the predicted answer with the correct answer.

Question: {question}
Predicted answer: {predicted_answer}
Correct answer: {ground_truth}

Are they semantically equivalent or both correct answers to the question?
Score: 10 if correct/equivalent, 0 if wrong.
Output only a number (0-10)."""


async def score_answer_with_vlm(
    question: str,
    predicted_answer: str,
    ground_truth: str,
    vlm_endpoint: str,
    vlm_model_name: str,
    vlm_api_key: str = "",
) -> Tuple[float, str]:
    """
    使用 VLM 判断答案是否正确（二元分类：正确=1.0，错误=0.0）
    """
    logger = get_reward_logger()

    prompt = ANSWER_SCORE_PROMPT.format(
        question=question,
        predicted_answer=predicted_answer,
        ground_truth=ground_truth,
    )

    payload = {
        "model": vlm_model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 16,  # 只需要数字
    }

    headers = {"Content-Type": "application/json"}
    if vlm_api_key:
        headers["Authorization"] = f"Bearer {vlm_api_key}"

    try:
        timeout = aiohttp.ClientTimeout(total=60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            url = f"http://{vlm_endpoint}/v1/chat/completions"
            async with session.post(url, json=payload, headers=headers) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    response_text = result["choices"][0]["message"]["content"]

                    score_match = re.search(r'(\d+(?:\.\d+)?)', response_text)
                    if score_match:
                        raw_score = float(score_match.group(1))
                        # 二元分类：>=5 视为正确(1.0)，<5 视为错误(0.0)
                        score = 1.0 if raw_score >= 5 else 0.0
                        logger.debug(f"VLM answer score: {score:.0f} (raw={raw_score}), response: {response_text[:50]}")
                        return score, response_text
                    # 解析失败返回 0（错误）
                    return 0.0, response_text
                else:
                    error_text = await resp.text()
                    logger.warning(f"VLM score HTTP error {resp.status}: {error_text[:100]}")
                    return 0.0, f"HTTP error {resp.status}: {error_text[:200]}"
    except Exception as e:
        logger.warning(f"VLM score exception: {str(e)}")
        return 0.0, f"Error: {str(e)}"


# ============== 主函数 ==============

def extract_answer(text: str) -> str:
    """提取 <answer>...</answer> 中的内容"""
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def extract_option_letter(answer: str) -> str:
    """提取答案选项字母"""
    answer = answer.strip()
    match = re.match(r'^(?:Option\s+)?([A-Z])(?:\.|:|$|\s)', answer, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    if len(answer) == 1 and answer.isalpha():
        return answer.upper()
    return answer.strip().upper()


def extract_segments(text: str) -> List[Tuple[float, float]]:
    """提取 <segment>[(start, end), ...]</segment> 中的时间段"""
    match = re.search(r'<segment>\s*\[(.*?)\]\s*</segment>', text, re.DOTALL | re.IGNORECASE)
    if not match:
        return []

    segments = []
    # 匹配 (start, end) 格式
    segment_pattern = r'\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)'
    for m in re.finditer(segment_pattern, match.group(1)):
        try:
            start = float(m.group(1))
            end = float(m.group(2))
            segments.append((start, end))
        except ValueError:
            continue
    return segments


def count_turns(text: str) -> Dict[str, int]:
    """
    统计多轮对话中各标签的出现次数

    Returns:
        Dict with counts: {think, segment, observation, answer}
    """
    return {
        "think": len(re.findall(r'<think>', text, re.IGNORECASE)),
        "segment": len(re.findall(r'<segment>', text, re.IGNORECASE)),
        "observation": len(re.findall(r'<observation>', text, re.IGNORECASE)),
        "answer": len(re.findall(r'<answer>', text, re.IGNORECASE)),
    }


def extract_all_segments(text: str) -> List[List[Tuple[float, float]]]:
    """
    提取所有 <segment> 标签中的时间段（多轮）

    Returns:
        List of segment lists, one per <segment> tag
    """
    all_segments = []
    pattern = r'<segment>\s*\[(.*?)\]\s*</segment>'
    segment_pattern = r'\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)'

    for match in re.finditer(pattern, text, re.DOTALL | re.IGNORECASE):
        segments = []
        for m in re.finditer(segment_pattern, match.group(1)):
            try:
                start = float(m.group(1))
                end = float(m.group(2))
                segments.append((start, end))
            except ValueError:
                continue
        all_segments.append(segments)

    return all_segments


async def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict = None,
    # 通过 reward_kwargs 传入
    vlm_endpoint: str = None,
    vlm_model_name: str = "Qwen3-VL-30B-A3B-Instruct",
    vlm_api_key: str = "123456",
    cache_dir: str = ".cache",
    cache_fps: int = 1,
    cache_max_frames: int = 512,
    use_vlm_scoring: bool = True,
    use_bbox_verification: bool = True,
    answer_weight: float = DEFAULT_ANSWER_WEIGHT,
    bbox_weight: float = DEFAULT_BBOX_WEIGHT,
    vlm_weight: float = DEFAULT_VLM_WEIGHT,
    # BBox 参数
    bbox_coord_range: float = DEFAULT_BBOX_COORD_RANGE,  # bbox 坐标范围 (1000 = [0,1000], 1 = [0,1])
    save_bbox_visualization: bool = False,
    bbox_vis_sample_rate: float = 0.1,  # 采样率：0.1 = 10% 的 bbox 保存可视化
    # IOU-based bbox scoring parameters
    temporal_weight: float = DEFAULT_TEMPORAL_WEIGHT,  # 时序奖励权重
    spatial_weight: float = DEFAULT_SPATIAL_WEIGHT,   # 空间奖励权重
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,    # IOU 阈值，低于此值 spatial_score=0
    # 日志相关参数
    enable_logging: bool = True,
    save_samples: bool = True,
    save_every_n: int = 1,  # 每 N 个样本保存一次 (1=全部保存, 10=每10个保存1个)
    log_dir: str = "./reward_logs",
    log_every_n: int = 10,  # 每 N 个样本打印一次统计
    **kwargs,
) -> float:
    """
    异步计算多维度奖励

    Args:
        data_source: 数据集标识
        solution_str: 模型输出字符串
        ground_truth: 正确答案
        extra_info: 额外信息 (包含 video_path, question 等)
        vlm_endpoint: VLM 服务地址 (如 localhost:8081)
        vlm_model_name: VLM 模型名称
        vlm_api_key: VLM API Key
        cache_dir: 帧缓存目录
        cache_fps: 缓存帧的fps
        cache_max_frames: 缓存的最大帧数
        use_vlm_scoring: 是否使用 VLM 对答案打分
        use_bbox_verification: 是否验证 bbox
        answer_weight: 答案分数权重
        bbox_weight: bbox 分数权重
        vlm_weight: VLM 打分权重
        temporal_weight: 时序奖励权重 (bbox 内部)
        spatial_weight: 空间奖励权重 (bbox 内部)
        iou_threshold: IOU 阈值，低于此值 spatial_score=0
        enable_logging: 是否启用日志
        save_samples: 是否保存样本
        save_every_n: 每 N 个样本保存一次 (1=全部保存, 10=每10个保存1个)
        log_dir: 日志目录
        log_every_n: 每 N 个样本打印一次统计

    Returns:
        float: 最终奖励分数
    """
    global _reward_stats, _sample_counter

    # 初始化日志
    if enable_logging:
        logger = setup_reward_logging(log_dir)
    else:
        logger = logging.getLogger("video_reward")

    start_time = time.time()

    extra_info = extra_info or {}
    video_path = extra_info.get("video_path", "")
    question = extra_info.get("question", "")
    video_id = extra_info.get("video_id", "")

    # 1. 提取预测答案
    predicted_answer = extract_answer(solution_str)

    # 2. 提取 segments (所有轮次)
    all_segments = extract_all_segments(solution_str)
    segments = extract_segments(solution_str)  # 最后一个 segment

    # 3. 提取 bboxes
    bboxes = extract_bboxes(solution_str)

    # 4. 统计多轮信息
    turn_counts = count_turns(solution_str)

    # 预先确定这个样本是否会被保存到 JSONL
    # 这样可以确保：只有保存到 JSONL 的样本才保存可视化，且保存所有 bbox 的可视化
    stats_key = data_source or "default"
    current_count = _reward_stats[stats_key]["total_calls"] + 1  # 预测下一个计数
    should_save_sample = save_samples and (current_count % save_every_n == 0)

    # 5. BBox 验证分数 (VLM, 异步) - 直接用 IOU 作为 bbox_score
    bbox_score = 0.0
    bbox_temporal_score = 0.0
    bbox_spatial_score = 0.0
    bbox_details = []
    bbox_verified = False

    if use_bbox_verification and vlm_endpoint and bboxes and video_path:
        # 如果这个样本会被保存到 JSONL，则保存所有 bbox 的可视化 (sample_rate=1.0)
        # 否则不保存可视化
        effective_save_vis = save_bbox_visualization and should_save_sample
        effective_sample_rate = 1.0 if should_save_sample else bbox_vis_sample_rate

        bbox_score, bbox_temporal_score, bbox_spatial_score, bbox_details = await verify_bboxes_with_vlm(
            bboxes=bboxes,
            video_path=video_path,
            solution_str=solution_str,
            vlm_endpoint=vlm_endpoint,
            vlm_model_name=vlm_model_name,
            vlm_api_key=vlm_api_key,
            cache_dir=cache_dir,
            cache_fps=cache_fps,
            cache_max_frames=cache_max_frames,
            save_bbox_visualization=effective_save_vis,
            bbox_vis_sample_rate=effective_sample_rate,
            visualization_dir=os.path.join(log_dir, "bbox_vis"),
            bbox_coord_range=bbox_coord_range,
            temporal_weight=temporal_weight,
            spatial_weight=spatial_weight,
            iou_threshold=iou_threshold,
        )
        bbox_verified = len(bbox_details) > 0

    # 6. 答案正确性评分
    # 优先使用 VLM 判断（支持开放题和选择题）
    # 如果没有 VLM，则回退到规则匹配
    answer_score = 0.0
    vlm_explanation = ""
    use_vlm_for_answer = use_vlm_scoring and vlm_endpoint and predicted_answer

    if use_vlm_for_answer:
        # 使用 VLM 判断答案是否正确
        answer_score, vlm_explanation = await score_answer_with_vlm(
            question=question,
            predicted_answer=predicted_answer,
            ground_truth=ground_truth,
            vlm_endpoint=vlm_endpoint,
            vlm_model_name=vlm_model_name,
            vlm_api_key=vlm_api_key,
        )
    elif predicted_answer:
        # 回退：规则匹配（仅适用于选择题）
        predicted = extract_option_letter(predicted_answer)
        correct = extract_option_letter(ground_truth)
        answer_score = 1.0 if predicted == correct else 0.0

    # 7. 计算最终分数
    if use_bbox_verification:
        # 答案分数 + BBox 分数
        final_score = answer_weight * answer_score + bbox_weight * bbox_score
        total_weight = answer_weight + bbox_weight
        final_score = final_score / total_weight if total_weight > 0 else 0.0
    else:
        # 只有答案分数
        final_score = answer_score

    elapsed_time = time.time() - start_time

    # 更新统计 (stats_key 已在前面定义)
    _reward_stats[stats_key]["total_calls"] += 1
    _reward_stats[stats_key]["total_score"] += final_score
    if answer_score == 1.0:  # 二元分类：答案正确
        _reward_stats[stats_key]["answer_correct"] += 1
    if bboxes:
        _reward_stats[stats_key]["bbox_found"] += 1
    if bbox_verified:
        _reward_stats[stats_key]["bbox_verified"] += 1
    if use_vlm_for_answer:
        _reward_stats[stats_key]["vlm_scored"] += 1

    # 日志输出
    if enable_logging:
        score_method = "VLM" if use_vlm_for_answer else "rule"
        logger.info(
            f"[Sample {_sample_counter+1}] video={video_id}, "
            f"pred={predicted_answer[:20] if predicted_answer else 'N/A'}..., "
            f"gt={ground_truth}, method={score_method}, "
            f"turns=(think={turn_counts['think']}, seg={turn_counts['segment']}, obs={turn_counts['observation']}), "
            f"num_bboxes={len(bboxes)}, "
            f"scores=(ans={answer_score:.2f}, bbox={bbox_score:.2f}, temporal={bbox_temporal_score:.2f}, spatial={bbox_spatial_score:.2f}), "
            f"final={final_score:.4f}, time={elapsed_time:.2f}s"
        )

        # 每 N 个样本打印统计
        if _reward_stats[stats_key]["total_calls"] % log_every_n == 0:
            print_reward_stats()

    # 保存样本 (使用预先计算的 should_save_sample，确保与可视化保存同步)
    if should_save_sample:
        sample_data = {
            "video_id": video_id,
            "video_path": video_path,
            "question": question[:500] if question else "",
            "ground_truth": ground_truth,
            "predicted_answer": predicted_answer,
            "answer_correct": answer_score == 1.0,  # 二元分类
            "score_method": "VLM" if use_vlm_for_answer else "rule",
            # 多轮统计
            "turn_counts": turn_counts,
            "num_turns": turn_counts["think"],
            # 完整的 solution_str (保存更多内容以便调试)
            "solution_str_preview": solution_str[:1000],  # 前1000字符预览
            "solution_str_full": solution_str,  # 完整内容
            "solution_str_length": len(solution_str),
            # 所有轮次的 segments
            "all_segments": all_segments,
            "last_segment": segments,
            # bboxes
            "bboxes": bboxes,
            "bbox_details": bbox_details,
            # 分数
            "scores": {
                "answer_score": answer_score,
                "bbox_score": bbox_score,
                "bbox_temporal_score": bbox_temporal_score,
                "bbox_spatial_score": bbox_spatial_score,
                "final_score": final_score,
            },
            "vlm_explanation": vlm_explanation[:200] if vlm_explanation else "",
            "config": {
                "use_vlm_scoring": use_vlm_scoring,
                "use_bbox_verification": use_bbox_verification,
                "vlm_endpoint": vlm_endpoint,
                "weights": {
                    "answer": answer_weight,
                    "bbox": bbox_weight,
                },
                "bbox_iou_config": {
                    "temporal_weight": temporal_weight,
                    "spatial_weight": spatial_weight,
                    "iou_threshold": iou_threshold,
                }
            },
            "elapsed_time": elapsed_time,
        }
        save_reward_sample(sample_data, output_dir=os.path.join(log_dir, "samples"))

    # 返回字典格式，支持 FILTER_GROUPS_METRIC=acc 或 score
    # - score: 最终分数 (float)
    # - acc: 答案是否正确 (bool，用于 filter_groups)
    return {
        "score": final_score,
        "acc": answer_score == 1.0,  # 二元分类：答案正确为 True
    }
