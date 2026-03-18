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
DEFAULT_BBOX_METRIC = "iou"          # bbox 评分指标: "iou" (原始) 或 "adaptive_iou" (小目标宽松)
DEFAULT_NWD_CONSTANT = 2.0           # NWD 归一化常数，控制衰减速度 (越大越宽松)
DEFAULT_TEMPORAL_TOLERANCE = 1       # 相邻帧容忍度 (0=禁用, 1=±1帧)，对应 Qwen3-VL temporal_patch_size=2
DEFAULT_ANSWER_WEIGHT = 0.4          # 答案分数权重
DEFAULT_BBOX_WEIGHT = 0.3            # bbox 分数权重
DEFAULT_VLM_WEIGHT = 0.3             # VLM 打分权重
DEFAULT_FORMAT_WEIGHT = 0.0          # 格式分数权重 (0 = 不使用, 设为正值启用)
DEFAULT_SEGMENT_WEIGHT = 0.0         # segment 分数权重 (0 = 不使用, 设为正值启用)
DEFAULT_MIN_COVERAGE_FACTOR = 0.5    # coverage 惩罚下限 (0.5 = 不输出bbox时保留50%分数, 1.0 = 无惩罚) [已废弃，保留兼容]
DEFAULT_BBOX_PER_TURN = 2            # 每个 think turn 期望输出的 bbox 数量（用于计算期望总数）

# VLM 重试配置
DEFAULT_VLM_MAX_RETRIES = 3          # VLM 调用最大重试次数
DEFAULT_VLM_RETRY_DELAY = 1.0        # 初始重试延迟（秒）
DEFAULT_VLM_BACKOFF_FACTOR = 2.0     # 指数退避因子


# ============== VLM 重试辅助函数 ==============

async def _call_vlm_with_retry(
    url: str,
    payload: dict,
    headers: dict,
    timeout: aiohttp.ClientTimeout,
    max_retries: int = DEFAULT_VLM_MAX_RETRIES,
    retry_delay: float = DEFAULT_VLM_RETRY_DELAY,
    backoff_factor: float = DEFAULT_VLM_BACKOFF_FACTOR,
) -> Tuple[Optional[dict], Optional[str]]:
    """
    带重试的 VLM 调用

    Args:
        url: VLM API URL
        payload: 请求 payload
        headers: 请求 headers
        timeout: aiohttp 超时配置
        max_retries: 最大重试次数
        retry_delay: 初始重试延迟（秒）
        backoff_factor: 指数退避因子

    Returns:
        (response_json, error_message) - 成功时 error_message 为 None
    """
    logger = logging.getLogger("video_reward")
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload, headers=headers) as resp:
                    if resp.status == 200:
                        return await resp.json(), None
                    elif resp.status in (429, 503, 502, 504):  # 可重试的 HTTP 错误
                        last_error = f"HTTP {resp.status}"
                    else:
                        error_text = await resp.text()
                        # 非可重试错误，直接返回
                        return None, f"HTTP error {resp.status}: {error_text[:100]}"
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
            last_error = str(e)

        # 如果还有重试机会，等待后重试
        if attempt < max_retries:
            delay = retry_delay * (backoff_factor ** attempt)
            logger.debug(f"VLM call failed (attempt {attempt+1}/{max_retries+1}), retrying in {delay:.1f}s: {last_error}")
            await asyncio.sleep(delay)

    return None, f"Error after {max_retries+1} attempts: {last_error}"


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
    training_step: int = None,
    sample_uid: str = None,
):
    """
    保存单个奖励计算样本到 JSONL 文件

    Args:
        sample_data: 样本数据字典
        output_dir: 输出目录
        training_step: 训练步数 (用于分目录保存)
        sample_uid: 样本唯一标识 (用于分组)
    """
    global _sample_counter

    # 按 step 分目录
    if training_step is not None:
        output_dir = os.path.join(output_dir, f"step_{training_step}")

    os.makedirs(output_dir, exist_ok=True)

    # 使用日期分文件
    date_str = datetime.now().strftime("%Y%m%d")
    output_file = os.path.join(output_dir, f"reward_samples_{date_str}.jsonl")

    _sample_counter += 1
    sample_data["sample_id"] = _sample_counter
    sample_data["timestamp"] = datetime.now().isoformat()
    if training_step is not None:
        sample_data["training_step"] = training_step
    if sample_uid is not None:
        sample_data["sample_uid"] = sample_uid

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

def extract_bboxes(text: str, include_positions: bool = False) -> List[Dict[str, Any]]:
    """
    从文本中提取 bbox 信息
    格式: <obj>object_name</obj><box>[x1,y1,x2,y2]</box>at<t>time_in_seconds</t>

    Args:
        text: 模型输出文本
        include_positions: 是否包含字符位置和轮次信息 (用于 token placement)

    Returns:
        List of dicts: [{"object": str, "bbox": [x1,y1,x2,y2], "time": float, "char_pos": int, "turn_idx": int}, ...]
        char_pos 是 </t> 结束位置，turn_idx 是所属轮次 (仅当 include_positions=True 时)
    """
    pattern = r'<obj>(.*?)</obj><box>\[([\d.,\s]+)\]</box>at<t>([\d.]+)</t>'

    bboxes = []

    if include_positions:
        # 先找出所有 <think> 的位置用于确定轮次
        think_starts = [m.start() for m in re.finditer(r'<think>', text, re.IGNORECASE)]
        think_ends = [m.end() for m in re.finditer(r'</think>', text, re.IGNORECASE)]

        def find_turn_for_position(pos: int) -> int:
            """确定字符位置属于哪个轮次的 think"""
            for i in range(len(think_starts)):
                start = think_starts[i]
                end = think_ends[i] if i < len(think_ends) else len(text)
                if start <= pos <= end:
                    return i
            return -1  # 不在任何 think 中

        for m in re.finditer(pattern, text, re.IGNORECASE):
            obj_name = m.group(1)
            bbox_str = m.group(2)
            time_str = m.group(3)
            char_pos = m.end()  # </t> 结束位置

            try:
                coords = [float(x.strip()) for x in bbox_str.split(',')]
                if len(coords) == 4:
                    turn_idx = find_turn_for_position(char_pos)
                    bboxes.append({
                        'object': obj_name.strip(),
                        'bbox': coords,  # [x1, y1, x2, y2]
                        'time': float(time_str),
                        'char_pos': char_pos,
                        'turn_idx': turn_idx,
                    })
            except ValueError:
                continue
    else:
        # 原始逻辑，不包含位置信息
        matches = re.findall(pattern, text, re.IGNORECASE)
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


def count_partial_bbox_attempts(text: str) -> int:
    """
    统计模型输出中的"部分 bbox"尝试次数

    部分 bbox = <obj>name</obj>at<t>time</t>，缺少 <box>[coords]</box>

    这用于检测模型是否在"作弊"：输出 obj 和 time 但跳过 box

    Returns:
        不完整 bbox 的数量 (有 <obj> 和 at<t> 但没有 <box>)
    """
    # 匹配完整的 bbox 模式
    full_pattern = r'<obj>(.*?)</obj><box>\[([\d.,\s]+)\]</box>at<t>([\d.]+)</t>'
    full_matches = set(re.findall(full_pattern, text, re.IGNORECASE))

    # 匹配部分模式: <obj>...</obj>at<t>...</t> (中间没有 <box>)
    # 注意：这个模式会匹配完整的和不完整的，我们需要排除完整的
    partial_pattern = r'<obj>(.*?)</obj>at<t>([\d.]+)</t>'
    partial_matches = re.findall(partial_pattern, text, re.IGNORECASE)

    # 统计部分匹配中有多少是真正不完整的（没有 box）
    incomplete_count = 0
    for obj_name, time_str in partial_matches:
        # 检查是否存在完整匹配
        is_complete = any(
            obj_name.strip() == full_obj.strip() and time_str == full_time
            for full_obj, _, full_time in full_matches
        )
        if not is_complete:
            incomplete_count += 1

    return incomplete_count


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

# 简洁版 prompt (效果更好)
BBOX_DETECT_PROMPT = """Detect and locate "{object_name}" in this image.
Output the bounding box as [x1, y1, x2, y2] where coordinates are in 0-1000 range.
If not found, output "not found"."""


def _format_bbox_detect_prompt(object_name: str, context: str = "") -> str:
    """格式化 bbox 检测 prompt"""
    return BBOX_DETECT_PROMPT.format(object_name=object_name)


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


def compute_nwd(bbox1: List[float], bbox2: List[float], constant: float = DEFAULT_NWD_CONSTANT) -> float:
    """
    计算两个 bbox 的 NWD (Normalized Wasserstein Distance) 相似度（改进版：自适应尺度感知）

    基于论文: "A Normalized Gaussian Wasserstein Distance for Tiny Object Detection"
    https://arxiv.org/abs/2110.13389

    改进：自适应尺度感知
    - 大目标使用更小的 constant（更严格），因为大目标即使有较大绝对偏移，
      原始 NWD 用 GT 对角线归一化后也能获得高分
    - 小目标使用较大的 constant（更宽松），保持原有对小目标友好的特性

    NWD 相比 IOU 的优势:
    - 对小目标的位置偏移更宽松 (小偏移不会剧烈惩罚)
    - 即使无重叠也能提供有意义的相似度
    - 具有尺度不变性，对不同大小目标公平

    Args:
        bbox1: [x1, y1, x2, y2] 归一化坐标 (预测框)
        bbox2: [x1, y1, x2, y2] 归一化坐标 (GT框)
        constant: 基础归一化常数，控制衰减速度 (越大越宽松，默认 2.0)

    Returns:
        NWD 相似度 (0-1)，越高越相似
    """
    import math

    # 中心点
    pred_cx = (bbox1[0] + bbox1[2]) / 2
    pred_cy = (bbox1[1] + bbox1[3]) / 2
    gt_cx = (bbox2[0] + bbox2[2]) / 2
    gt_cy = (bbox2[1] + bbox2[3]) / 2

    # 宽高
    pred_w = max(bbox1[2] - bbox1[0], 1e-6)
    pred_h = max(bbox1[3] - bbox1[1], 1e-6)
    gt_w = max(bbox2[2] - bbox2[0], 1e-6)
    gt_h = max(bbox2[3] - bbox2[1], 1e-6)

    # 2D Gaussian Wasserstein Distance (闭式解)
    # 将 bbox 建模为 2D 高斯分布: μ=(cx, cy), σ=(w/2, h/2)
    # W² = ||μ1 - μ2||² + ||σ1 - σ2||²_F
    center_dist_sq = (pred_cx - gt_cx) ** 2 + (pred_cy - gt_cy) ** 2
    size_dist_sq = ((pred_w - gt_w) ** 2 + (pred_h - gt_h) ** 2) / 4

    wasserstein_sq = center_dist_sq + size_dist_sq
    wasserstein = math.sqrt(wasserstein_sq)

    # 用 GT 目标尺寸归一化
    gt_diag = math.sqrt(gt_w ** 2 + gt_h ** 2)
    gt_diag = max(gt_diag, 1e-6)  # 防止除零

    normalized_dist = wasserstein / gt_diag

    # 【改进】自适应常数：大目标更严格
    # 假设图像对角线 = sqrt(2) ≈ 1.414（归一化坐标 [0,1]）
    image_diagonal = 1.414
    relative_size = gt_diag / image_diagonal  # 0 ~ 1

    # 自适应常数: 小目标 constant 大（宽松），大目标 constant 小（严格）
    # relative_size=0.1 → adaptive=constant*0.95
    # relative_size=0.5 → adaptive=constant*0.75
    adaptive_constant = constant * (1.0 - 0.5 * relative_size)
    adaptive_constant = max(0.5, adaptive_constant)  # 下限保护

    # 指数映射到 [0, 1]: exp(-normalized_dist * adaptive_constant)
    nwd = math.exp(-normalized_dist * adaptive_constant)

    return nwd


def compute_adaptive_iou(
    iou: float,
    gt_area: float,
    small_threshold: float = 0.05,
    large_threshold: float = 0.20,
) -> float:
    """
    尺度自适应 IOU 评分

    - 小目标 (面积 < 5%):  √IOU (宽松，因为小目标的小偏移会导致 IOU 大幅下降)
    - 大目标 (面积 > 20%): IOU² (严格，大目标应该更容易精确定位)
    - 中等目标: 线性插值

    Args:
        iou: 原始 IOU 值
        gt_area: GT 框面积 (相对于整个图像，范围 0-1)
        small_threshold: 小目标面积阈值 (默认 5%)
        large_threshold: 大目标面积阈值 (默认 20%)

    Returns:
        自适应调整后的分数 (0-1)
    """
    import math

    if iou <= 0:
        return 0.0

    sqrt_iou = math.sqrt(iou)
    iou_sq = iou ** 2

    if gt_area < small_threshold:
        # 小目标：宽松，用 √IOU
        return sqrt_iou
    elif gt_area > large_threshold:
        # 大目标：严格，用 IOU²
        return iou_sq
    else:
        # 中等目标：线性插值 √IOU → IOU²
        # gt_area 从 small_threshold 到 large_threshold 时，t 从 0 到 1
        t = (gt_area - small_threshold) / (large_threshold - small_threshold)
        return sqrt_iou * (1 - t) + iou_sq * t


def compute_bbox_similarity(
    bbox1: List[float],
    bbox2: List[float],
    metric: str = "iou",
    **kwargs,
) -> float:
    """
    统一接口：计算两个 bbox 的相似度

    Args:
        bbox1: [x1, y1, x2, y2] 归一化坐标 (预测框)
        bbox2: [x1, y1, x2, y2] 归一化坐标 (GT框)
        metric: 评分指标:
            - "iou": 原始 IOU
            - "adaptive_iou": 尺度自适应 IOU (小目标 √IOU 宽松, 大目标 IOU² 严格)
            - "nwd": 别名，等同于 "adaptive_iou" (历史兼容)

    Returns:
        相似度分数 (0-1)
    """
    iou = compute_iou(bbox1, bbox2)

    if metric in ("nwd", "adaptive_iou"):
        # 尺度自适应 IOU 评分
        gt_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        return compute_adaptive_iou(iou, gt_area)
    else:
        return iou


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
    max_retries: int = DEFAULT_VLM_MAX_RETRIES,
    retry_delay: float = DEFAULT_VLM_RETRY_DELAY,
    backoff_factor: float = DEFAULT_VLM_BACKOFF_FACTOR,
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
        max_retries: 最大重试次数
        retry_delay: 初始重试延迟（秒）
        backoff_factor: 指数退避因子

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
        url = f"http://{vlm_endpoint}/v1/chat/completions"

        # 使用带重试的 VLM 调用
        result, error = await _call_vlm_with_retry(
            url=url,
            payload=payload,
            headers=headers,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
            backoff_factor=backoff_factor,
        )

        if error:
            logger.warning(f"VLM detect failed: {error}")
            return None, error

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

    except Exception as e:
        logger.warning(f"VLM detect exception: {str(e)}")
        return None, f"Error: {str(e)}"


def _parse_vlm_bbox_response(response: str) -> Optional[List[float]]:
    """
    多层解析 VLM bbox 响应（参考 long_ver5_zgy.py 的解析逻辑）

    支持格式:
    1. <exists>YES/NO</exists> + <bbox>[x1, y1, x2, y2]</bbox>  (V2 格式)
    2. <bbox>[x1, y1, x2, y2]</bbox>  (推荐格式)
    3. <answer>[x1, y1, x2, y2]</answer>
    4. JSON: {"found": true, "bbox": [x1, y1, x2, y2]}
    5. 裸坐标: [x1, y1, x2, y2]

    返回 [0,1000] 范围的坐标，或 None
    """
    if not response:
        return None

    # V2 格式: 先检查 <exists> 标签
    exists_match = re.search(r'<exists>\s*(YES|NO)\s*</exists>', response, re.IGNORECASE)
    if exists_match:
        if exists_match.group(1).upper() == 'NO':
            return None  # 明确说不存在，直接返回 None

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
    bbox_metric: str = DEFAULT_BBOX_METRIC,
    temporal_tolerance: int = DEFAULT_TEMPORAL_TOLERANCE,
    save_visualization: bool = False,
    visualization_dir: str = "./reward_logs/bbox_vis",
    video_path: Optional[str] = None,
    bbox_timestamp: Optional[float] = None,
    cache_dir: str = ".cache",
    cache_fps: int = 1,
    cache_max_frames: int = 512,
) -> Tuple[float, float, float, Optional[List[float]], str, str, str, Optional[str]]:
    """
    使用 VLM 验证单个 bbox 的准确性

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
        bbox_metric: 评分指标 "iou" 或 "adaptive_iou" (小目标宽松)
        temporal_tolerance: 相邻帧容忍度 (0=禁用, 1=±1帧)，用于处理 Qwen3-VL 的时序融合
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
        # 1. 调用 VLM 获取当前帧的 GT bbox
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

        pred_normalized = [c / effective_coord_range for c in bbox]

        # 使用选定的指标计算 bbox 分数
        # - bbox_metric="iou": 原始 IOU
        # - bbox_metric="adaptive_iou" (或 "nwd"): 尺度自适应 IOU (小目标宽松, 大目标严格)
        # GT 返回 None 时分数为 0（物体不存在）
        current_similarity = 0.0
        if gt_bbox is not None:
            current_similarity = compute_bbox_similarity(
                pred_normalized, gt_bbox,
                metric=bbox_metric,
            )

        # 2. 相邻帧融合：检查相邻帧的 GT bbox，取最大相似度
        # 这是为了处理 Qwen3-VL 的 temporal_patch_size=2 导致的时序融合问题
        # 模型预测的 bbox 可能对应融合帧中的任一帧
        best_similarity = current_similarity
        best_gt_bbox = gt_bbox
        best_frame_info = "current"

        if temporal_tolerance > 0 and video_path and bbox_timestamp is not None:
            # 获取相邻帧路径
            neighbor_frames = _get_neighboring_frame_paths(
                frame_path, video_path, bbox_timestamp,
                cache_dir, cache_fps, cache_max_frames,
                num_neighbors=temporal_tolerance,
            )

            # 并行获取所有相邻帧的 GT bbox
            neighbor_tasks = []
            neighbor_info = []  # (path, ts, is_target)
            for neighbor_path, neighbor_ts, is_target in neighbor_frames:
                if is_target:
                    continue  # 跳过当前帧，已经计算过
                neighbor_tasks.append(get_gt_bbox_from_vlm(
                    frame_path=neighbor_path,
                    object_name=object_name,
                    context=context,
                    vlm_endpoint=vlm_endpoint,
                    vlm_model_name=vlm_model_name,
                    vlm_api_key=vlm_api_key,
                ))
                neighbor_info.append((neighbor_path, neighbor_ts))

            if neighbor_tasks:
                neighbor_results = await asyncio.gather(*neighbor_tasks, return_exceptions=True)
                for (neighbor_path, neighbor_ts), result in zip(neighbor_info, neighbor_results):
                    if isinstance(result, Exception):
                        logger.debug(f"Neighbor frame GT bbox failed: {neighbor_ts}s, {result}")
                        continue
                    neighbor_gt, _ = result
                    if neighbor_gt is not None:
                        neighbor_similarity = compute_bbox_similarity(
                            pred_normalized, neighbor_gt,
                            metric=bbox_metric,
                        )
                        if neighbor_similarity > best_similarity:
                            best_similarity = neighbor_similarity
                            best_gt_bbox = neighbor_gt
                            best_frame_info = f"neighbor@{neighbor_ts:.0f}s"
                            logger.debug(
                                f"Better match in neighbor frame: {neighbor_ts}s, "
                                f"{bbox_metric}={neighbor_similarity:.3f} > {current_similarity:.3f}"
                            )

        similarity = best_similarity
        temporal_score = 1.0 if best_gt_bbox is not None else 0.0  # 物体是否存在
        spatial_score = similarity  # IOU 或 NWD

        # 使用类似 segment_score 的公式: (2*IoU + IoP + IoG) / 4
        # 对于 bbox: (2*spatial + spatial + temporal) / 4 = (3*spatial + temporal) / 4
        # 这样 temporal_score 作为"召回"惩罚：如果物体不存在则扣分
        # 同时 spatial_score 权重更高，确保定位精度仍是主要目标
        total_score = (3 * spatial_score + temporal_score) / 4

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

                # 绘制 GT bbox（绿色）- 使用 best_gt_bbox
                if best_gt_bbox is not None:
                    gpx = [
                        int(best_gt_bbox[0] * fw), int(best_gt_bbox[1] * fh),
                        int(best_gt_bbox[2] * fw), int(best_gt_bbox[3] * fh),
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
                f"Object: {object_name[:40]}  |  {bbox_metric.upper()}: {similarity:.2f} ({best_frame_info})  |  "
                f"Red=Pred  Green=GT  Cyan border=TARGET frame"
            )
            strip_draw.text((8, 8), info_text, fill="white", font=info_font)

            # 粘贴帧
            x_offset = 0
            for i, fimg in enumerate(annotated_frames):
                strip.paste(fimg, (x_offset, info_height))
                x_offset += fimg.width + 2  # 2px gap

            vis_filename = f"bbox_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}_{bbox_metric}{similarity:.2f}.jpg"
            saved_vis_path = os.path.abspath(os.path.join(visualization_dir, vis_filename))
            strip.save(saved_vis_path, "JPEG", quality=90)
            logger.debug(f"Saved bbox vis strip: {saved_vis_path}")
        else:
            saved_vis_path = None

        explanation = f"metric={bbox_metric}, score={similarity:.2f} ({best_frame_info}), temporal={temporal_score:.1f}, gt_bbox={best_gt_bbox}, vlm_response={raw_response[:100]}"
        logger.debug(f"BBox verify: obj={object_name}, pred={bbox}, {explanation}")

        return total_score, temporal_score, spatial_score, best_gt_bbox, explanation, vlm_prompt, raw_response, saved_vis_path

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
    bbox_metric: str = DEFAULT_BBOX_METRIC,
    temporal_tolerance: int = DEFAULT_TEMPORAL_TOLERANCE,
) -> Tuple[float, float, float, List[Dict]]:
    """
    验证所有 bbox 并返回平均分数

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
        bbox_metric: 评分指标 "iou" 或 "adaptive_iou"
        temporal_tolerance: 相邻帧容忍度 (0=禁用, 1=±1帧)

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
                bbox_metric=bbox_metric,
                temporal_tolerance=temporal_tolerance,
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
    max_retries: int = DEFAULT_VLM_MAX_RETRIES,
    retry_delay: float = DEFAULT_VLM_RETRY_DELAY,
    backoff_factor: float = DEFAULT_VLM_BACKOFF_FACTOR,
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
        url = f"http://{vlm_endpoint}/v1/chat/completions"

        # 使用带重试的 VLM 调用
        result, error = await _call_vlm_with_retry(
            url=url,
            payload=payload,
            headers=headers,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
            backoff_factor=backoff_factor,
        )

        if error:
            logger.warning(f"VLM score failed: {error}")
            return 0.0, error

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

    except Exception as e:
        logger.warning(f"VLM score exception: {str(e)}")
        return 0.0, f"Error: {str(e)}"


# ============== Format Reward ==============

# Regex to extract structural tags for format validation
_FORMAT_TAG_RE = re.compile(r"<(/?)(think|segment|observation|answer)>", re.IGNORECASE)

# Regex to validate segment content: [(start, end), ...] or [[start, end], ...]
_SEGMENT_CONTENT_RE = re.compile(r'<segment>\s*\[(.*?)\]\s*</segment>', re.DOTALL | re.IGNORECASE)
# Match both (start, end) and [start, end] formats
_SEGMENT_TUPLE_RE = re.compile(r'[\(\[]\s*([\d.]+)\s*,\s*([\d.]+)\s*[\)\]]')

# Regex to validate object grounding format: <obj>...</obj><box>[...]</box>at<t>...</t>
_OBJ_BOX_RE = re.compile(r'<obj>([^<]+)</obj>\s*<box>\s*\[([^\]]+)\]\s*</box>\s*at\s*<t>\s*([\d.]+)\s*</t>', re.IGNORECASE)


def format_reward(predict_str: str, strict_segment: bool = False) -> float:
    """
    Validate the format of video reasoning output.

    Valid format:
    - Must end with </answer>
    - Sequence: <think>...</think> followed by:
        - <segment>...</segment><observation>...</observation><think>...</think> (can repeat)
        - OR <answer>...</answer> (terminates)
    - <segment> must contain valid list of (start, end) tuples
    - Object grounding (optional): <obj>name</obj><box>[x1,y1,x2,y2]</box>at<t>time</t>

    Args:
        predict_str: Model output string
        strict_segment: If True, require valid segment content format

    Returns:
        1.0 if format is valid, 0.0 otherwise
    """
    s = predict_str.strip()

    # 1) Must finish with </answer> (optionally followed by EOS token)
    if not re.search(r"</answer>\s*(<\|im_end\|>)?\s*$", s, re.DOTALL):
        return 0.0

    # 2) Walk through tag sequence to enforce grammar
    tags_iter = _FORMAT_TAG_RE.finditer(s)
    state = "think_open"  # expected next state

    for m in tags_iter:
        is_close = m.group(1) == "/"
        tag_name = m.group(2).lower()

        if state == "think_open":
            # Expect <think>
            if tag_name != "think" or is_close:
                return 0.0
            state = "think_close"

        elif state == "think_close":
            # Expect </think>
            if tag_name != "think" or not is_close:
                return 0.0
            state = "post_think"

        elif state == "post_think":
            # After </think>, expect <segment> or <answer>
            if is_close:
                return 0.0
            if tag_name == "segment":
                state = "segment_close"
            elif tag_name == "answer":
                state = "answer_close"
            else:
                return 0.0

        elif state == "segment_close":
            # Expect </segment>
            if tag_name != "segment" or not is_close:
                return 0.0
            state = "obs_open"

        elif state == "obs_open":
            # Expect <observation>
            if tag_name != "observation" or is_close:
                return 0.0
            state = "obs_close"

        elif state == "obs_close":
            # Expect </observation>
            if tag_name != "observation" or not is_close:
                return 0.0
            state = "think_open"  # Loop back for next turn

        elif state == "answer_close":
            # Expect </answer>
            if tag_name != "answer" or not is_close:
                return 0.0
            state = "end"

        elif state == "end":
            # No structural tags allowed after </answer>
            return 0.0

    # Must have reached end state
    if state != "end":
        return 0.0

    # 3) Validate segment content format (optional strict check)
    if strict_segment:
        for match in _SEGMENT_CONTENT_RE.finditer(s):
            segment_content = match.group(1)
            # Must have at least one valid (start, end) tuple
            tuples = _SEGMENT_TUPLE_RE.findall(segment_content)
            if not tuples:
                return 0.0
            # Validate each tuple has valid numbers
            for start_str, end_str in tuples:
                try:
                    start = float(start_str)
                    end = float(end_str)
                    if start < 0 or end < 0 or start > end:
                        return 0.0
                except ValueError:
                    return 0.0

    return 1.0


def format_reward_lenient(predict_str: str) -> float:
    """
    Lenient format check - only verifies basic structure.

    Checks:
    - Has <think>...</think>
    - Has <answer>...</answer>
    - Ends with </answer>

    Returns:
        1.0 if basic format is valid, 0.0 otherwise
    """
    s = predict_str.strip()

    # Must have think tags
    if not re.search(r'<think>.*?</think>', s, re.DOTALL | re.IGNORECASE):
        return 0.0

    # Must have answer tags
    if not re.search(r'<answer>.*?</answer>', s, re.DOTALL | re.IGNORECASE):
        return 0.0

    # Must end with </answer>
    if not re.search(r"</answer>\s*(<\|im_end\|>)?\s*$", s, re.DOTALL):
        return 0.0

    return 1.0


# ============== 主函数 ==============

def extract_answer(text: str) -> str:
    """提取 <answer>...</answer> 中的内容"""
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
    # return match.group(1).strip() if match else ""
    return match.group(1).strip() if match else text.strip()


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
    """提取 <segment>[(start, end), ...]</segment> 或 <segment>[[start, end], ...]</segment> 中的时间段"""
    match = re.search(r'<segment>\s*\[(.*?)\]\s*</segment>', text, re.DOTALL | re.IGNORECASE)
    if not match:
        return []

    segments = []
    # 匹配 (start, end) 或 [start, end] 格式
    segment_pattern = r'[\(\[]\s*([\d.]+)\s*,\s*([\d.]+)\s*[\)\]]'
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
    # 匹配 (start, end) 或 [start, end] 格式
    segment_pattern = r'[\(\[]\s*([\d.]+)\s*,\s*([\d.]+)\s*[\)\]]'

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


def compute_segment_score(
    pred_segments: List[List[Tuple[float, float]]],
    gt_segments: List[Tuple[float, float]],
) -> Tuple[float, float, float, float]:
    """
    计算预测 segments 与 GT segments 的匹配分数

    使用公式: (2*IoU + IoP + IoG) / 4
    其中:
    - IoU = Intersection over Union
    - IoP = Intersection over Prediction (预测的总长度)
    - IoG = Intersection over Ground Truth (GT 的总长度)

    Args:
        pred_segments: 模型预测的所有轮次的 segments，格式为 [[(start, end), ...], ...]
        gt_segments: GT 的 reference_segments，格式为 [(start, end), ...]

    Returns:
        (final_score, iou, iop, iog): 最终分数和三个组件分数
    """
    if not gt_segments:
        # 没有 GT segments，无法计算
        return 0.0, 0.0, 0.0, 0.0

    # 将所有预测的 segments 展平为一个列表
    all_pred_segments = []
    for turn_segments in pred_segments:
        all_pred_segments.extend(turn_segments)

    if not all_pred_segments:
        # 没有预测的 segments
        return 0.0, 0.0, 0.0, 0.0

    # 将 segments 转换为时间轴上的区间集合，用于计算交集和并集
    def segments_to_intervals(segments: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """将 segments 合并为不重叠的区间"""
        if not segments:
            return []
        # 按起始时间排序
        sorted_segs = sorted(segments, key=lambda x: x[0])
        merged = [sorted_segs[0]]
        for start, end in sorted_segs[1:]:
            if start <= merged[-1][1]:
                # 重叠，合并
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        return merged

    def total_length(intervals: List[Tuple[float, float]]) -> float:
        """计算区间总长度"""
        return sum(end - start for start, end in intervals)

    def compute_intersection(intervals1: List[Tuple[float, float]],
                            intervals2: List[Tuple[float, float]]) -> float:
        """计算两个区间集合的交集长度"""
        if not intervals1 or not intervals2:
            return 0.0

        intersection = 0.0
        i, j = 0, 0
        while i < len(intervals1) and j < len(intervals2):
            start1, end1 = intervals1[i]
            start2, end2 = intervals2[j]

            # 计算交集
            inter_start = max(start1, start2)
            inter_end = min(end1, end2)
            if inter_start < inter_end:
                intersection += inter_end - inter_start

            # 移动指针
            if end1 < end2:
                i += 1
            else:
                j += 1

        return intersection

    # 合并区间
    pred_intervals = segments_to_intervals(all_pred_segments)
    gt_intervals = segments_to_intervals(gt_segments)

    # 计算各项指标
    pred_length = total_length(pred_intervals)
    gt_length = total_length(gt_intervals)
    intersection = compute_intersection(pred_intervals, gt_intervals)
    union = pred_length + gt_length - intersection

    # 计算 IoU, IoP, IoG
    iou = intersection / union if union > 0 else 0.0
    iop = intersection / pred_length if pred_length > 0 else 0.0
    iog = intersection / gt_length if gt_length > 0 else 0.0

    # 最终分数: (2*IoU + IoP + IoG) / 4
    final_score = (2 * iou + iop + iog) / 4

    return final_score, iou, iop, iog


def compute_per_turn_segment_score(
    pred_segments: List[List[Tuple[float, float]]],
    gt_segments: List[Tuple[float, float]],
) -> List[float]:
    """
    计算每轮预测 segments 与 GT segments 的匹配分数

    对每个 turn 独立计算其 segments 与 GT 的重叠程度。
    使用公式: (2*IoU + IoP + IoG) / 4
    其中:
    - IoU = turn 的 segments 与 GT 的交集 / 并集
    - IoP = 交集 / turn 预测长度
    - IoG = 交集 / GT 长度

    Args:
        pred_segments: 模型预测的所有轮次的 segments，格式为 [[(start, end), ...], ...]
        gt_segments: GT 的 reference_segments，格式为 [(start, end), ...]

    Returns:
        per_turn_scores: 每轮的分数列表
    """
    if not gt_segments:
        return [0.0] * len(pred_segments)

    def segments_to_intervals(segments):
        if not segments:
            return []
        sorted_segs = sorted(segments, key=lambda x: x[0])
        merged = [sorted_segs[0]]
        for start, end in sorted_segs[1:]:
            if start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        return merged

    def total_length(intervals):
        return sum(end - start for start, end in intervals)

    def compute_intersection(intervals1, intervals2):
        if not intervals1 or not intervals2:
            return 0.0
        intersection = 0.0
        i, j = 0, 0
        while i < len(intervals1) and j < len(intervals2):
            start1, end1 = intervals1[i]
            start2, end2 = intervals2[j]
            inter_start = max(start1, start2)
            inter_end = min(end1, end2)
            if inter_start < inter_end:
                intersection += inter_end - inter_start
            if end1 < end2:
                i += 1
            else:
                j += 1
        return intersection

    gt_intervals = segments_to_intervals(gt_segments)
    gt_length = total_length(gt_intervals)

    per_turn_scores = []
    for turn_segments in pred_segments:
        if not turn_segments:
            per_turn_scores.append(0.0)
            continue

        turn_intervals = segments_to_intervals(turn_segments)
        turn_length = total_length(turn_intervals)
        intersection = compute_intersection(turn_intervals, gt_intervals)
        union = turn_length + gt_length - intersection

        iou = intersection / union if union > 0 else 0.0
        iop = intersection / turn_length if turn_length > 0 else 0.0
        iog = intersection / gt_length if gt_length > 0 else 0.0

        score = (2 * iou + iop + iog) / 4
        per_turn_scores.append(score)

    return per_turn_scores


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
    format_weight: float = DEFAULT_FORMAT_WEIGHT,
    segment_weight: float = DEFAULT_SEGMENT_WEIGHT,  # segment 分数权重
    use_strict_format: bool = False,  # 是否使用严格的 segment 格式检查
    # BBox 参数
    bbox_coord_range: float = DEFAULT_BBOX_COORD_RANGE,  # bbox 坐标范围 (1000 = [0,1000], 1 = [0,1])
    save_bbox_visualization: bool = False,
    bbox_vis_sample_rate: float = 0.1,  # 采样率：0.1 = 10% 的 bbox 保存可视化
    # IOU-based bbox scoring parameters
    temporal_weight: float = DEFAULT_TEMPORAL_WEIGHT,  # 时序奖励权重
    spatial_weight: float = DEFAULT_SPATIAL_WEIGHT,   # 空间奖励权重
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,    # IOU 阈值，低于此值 spatial_score=0
    # BBox metric selection
    bbox_metric: str = DEFAULT_BBOX_METRIC,          # "iou" 原始指标，"adaptive_iou" 小目标宽松 (别名 "nwd")
    temporal_tolerance: int = DEFAULT_TEMPORAL_TOLERANCE,  # 相邻帧容忍度 (0=禁用, 1=±1帧)
    # BBox 期望数量参数
    bbox_per_turn: int = DEFAULT_BBOX_PER_TURN,      # 每个 think turn 期望输出的 bbox 数量
    # Coverage 惩罚参数 (GRPO) [已废弃，保留兼容]
    min_coverage_factor: float = DEFAULT_MIN_COVERAGE_FACTOR,  # 不输出bbox时保留的最低分数比例 (已废弃)
    # 日志相关参数
    enable_logging: bool = True,
    save_samples: bool = True,
    save_every_n: int = 1,  # 每 N 个样本保存一次 (1=全部保存, 10=每10个保存1个)
    log_dir: str = "./reward_logs",
    log_every_n: int = 10,  # 每 N 个样本打印一次统计
    # 训练上下文参数 (用于按 step/uid 分层保存)
    training_step: int = None,  # 当前训练步数
    sample_uid: str = None,  # 样本唯一标识 (group id)
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
        format_weight: 格式分数权重 (0 = 不参与最终分数计算)
        use_strict_format: 是否使用严格的 segment 格式检查
        temporal_weight: 时序奖励权重 (bbox 内部)
        spatial_weight: 空间奖励权重 (bbox 内部)
        iou_threshold: IOU 阈值，低于此值 spatial_score=0
        enable_logging: 是否启用日志
        save_samples: 是否保存样本
        save_every_n: 每 N 个样本保存一次 (1=全部保存, 10=每10个保存1个)
        log_dir: 日志目录
        log_every_n: 每 N 个样本打印一次统计

    Returns:
        dict: 包含 score, acc, format 的奖励分数字典
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

    # 从 extra_info 获取训练上下文 (如果通过 kwargs 传入则优先使用 kwargs)
    if training_step is None:
        training_step = extra_info.get("training_step", None)
    if sample_uid is None:
        sample_uid = extra_info.get("uid", extra_info.get("index", None))
        if sample_uid is not None:
            sample_uid = str(sample_uid)

    # 1. 提取预测答案
    predicted_answer = extract_answer(solution_str)

    # 2. 提取 segments (所有轮次)
    all_segments = extract_all_segments(solution_str)
    segments = extract_segments(solution_str)  # 最后一个 segment

    # 3. 提取 bboxes (包含位置信息用于 token placement)
    bboxes = extract_bboxes(solution_str, include_positions=True)

    # 4. 统计多轮信息
    turn_counts = count_turns(solution_str)

    # 4.5 格式检查分数
    format_score = format_reward(solution_str, strict_segment=use_strict_format)

    # 预先确定这个样本是否会被保存到 JSONL
    # 这样可以确保：只有保存到 JSONL 的样本才保存可视化，且保存所有 bbox 的可视化
    stats_key = data_source or "default"
    current_count = _reward_stats[stats_key]["total_calls"] + 1  # 预测下一个计数
    should_save_sample = save_samples and (current_count % save_every_n == 0)

    # 5. BBox 验证分数 (VLM, 异步)
    # bbox_score 融合了质量和数量：未输出或格式错误的 bbox 视为 0 分参与平均
    bbox_score = 0.0
    bbox_details = []
    bbox_verified = False
    bbox_coverage = 1.0  # 保留用于日志，但不再用于惩罚 answer_score

    # 计算期望的 bbox 数量
    # 期望数 = max(基于 think turn 的期望, 模型尝试输出的数量)
    num_think_turns = turn_counts.get("think", 0)
    base_expected = max(num_think_turns - 1, 0) * bbox_per_turn  # 每个 turn 期望 bbox_per_turn 个

    # 统计模型尝试输出的 bbox 数量（包括完整和不完整的）
    complete_bbox_count = len(bboxes)  # 完整格式的 bbox
    partial_bbox_count = count_partial_bbox_attempts(solution_str)  # 不完整格式（<obj>at<t> 缺少 <box>）
    total_attempts = complete_bbox_count + partial_bbox_count

    # 期望数量 = max(基于 turn 的期望, 模型尝试的数量)
    # 这样可以惩罚"作弊"行为：输出很多 <obj>at<t> 但没有 <box>
    expected_bbox_count = max(base_expected, total_attempts)

    if use_bbox_verification and vlm_endpoint and bboxes and video_path:
        # 如果这个样本会被保存到 JSONL，则保存所有 bbox 的可视化 (sample_rate=1.0)
        # 否则不保存可视化
        effective_save_vis = save_bbox_visualization and should_save_sample
        effective_sample_rate = 1.0 if should_save_sample else bbox_vis_sample_rate

        raw_bbox_score, _, _, bbox_details = await verify_bboxes_with_vlm(
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
            bbox_metric=bbox_metric,
            temporal_tolerance=temporal_tolerance,
        )
        bbox_verified = len(bbox_details) > 0

        # 计算融合了数量惩罚的 bbox_score
        # 每个验证通过的 bbox 的分数 + 未输出/格式错误的 bbox 视为 0 分
        if expected_bbox_count > 0:
            # 获取每个 bbox 的 total_score
            actual_scores = [d.get("total_score", 0.0) for d in bbox_details]
            # 补零：未输出或格式错误的 bbox 视为 0 分
            num_missing = expected_bbox_count - len(actual_scores)
            if num_missing > 0:
                actual_scores.extend([0.0] * num_missing)
            # 平均分数
            bbox_score = sum(actual_scores) / expected_bbox_count
            # 计算 coverage 用于日志
            bbox_coverage = min(len(bbox_details) / expected_bbox_count, 1.0)
        else:
            # 不期望输出 bbox
            bbox_score = 0.0
            bbox_coverage = 1.0
    elif expected_bbox_count > 0:
        # 期望输出 bbox 但没有启用验证，或没有输出任何完整 bbox
        # 所有期望的 bbox 都视为 0 分
        bbox_score = 0.0
        bbox_coverage = 0.0

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

    # 6.5. Segment 分数计算
    segment_score = 0.0
    segment_iou = 0.0
    segment_iop = 0.0
    segment_iog = 0.0
    gt_segments = extra_info.get("reference_segments", [])
    if gt_segments and all_segments:
        segment_score, segment_iou, segment_iop, segment_iog = compute_segment_score(all_segments, gt_segments)

    # 7. 计算最终分数
    # bbox_score 已经融合了数量惩罚（未输出的 bbox 视为 0 分），不再需要 coverage_factor
    if use_bbox_verification:
        # 答案分数 + BBox 分数 + 格式分数 + Segment 分数
        final_score = (answer_weight * answer_score + bbox_weight * bbox_score +
                       format_weight * format_score + segment_weight * segment_score)
        total_weight = answer_weight + bbox_weight + format_weight + segment_weight
        final_score = final_score / total_weight if total_weight > 0 else 0.0
    else:
        # 答案分数 + 格式分数 + Segment 分数
        final_score = (answer_weight * answer_score + format_weight * format_score +
                       segment_weight * segment_score)
        total_weight = answer_weight + format_weight + segment_weight
        final_score = final_score / total_weight if total_weight > 0 else 0.0

    elapsed_time = time.time() - start_time

    # 更新统计 (stats_key 已在前面定义)
    _reward_stats[stats_key]["total_calls"] += 1
    _reward_stats[stats_key]["total_score"] += final_score
    if answer_score == 1.0:  # 二元分类：答案正确
        _reward_stats[stats_key]["answer_correct"] += 1
    if format_score == 1.0:  # 格式正确
        _reward_stats[stats_key]["format_valid"] = _reward_stats[stats_key].get("format_valid", 0) + 1
    if bboxes:
        _reward_stats[stats_key]["bbox_found"] += 1
    if bbox_verified:
        _reward_stats[stats_key]["bbox_verified"] += 1
    if use_vlm_for_answer:
        _reward_stats[stats_key]["vlm_scored"] += 1

    # 日志输出
    if enable_logging:
        score_method = "VLM" if use_vlm_for_answer else "rule"
        # Format segments for logging
        gt_segs_str = str(gt_segments) if gt_segments else "[]"
        pred_segs_str = str(all_segments) if all_segments else "[]"
        logger.info(
            f"[Sample {_sample_counter+1}] video={video_id}, "
            f"pred={predicted_answer[:20] if predicted_answer else 'N/A'}..., "
            f"gt={ground_truth}, method={score_method}, "
            f"turns=(think={turn_counts['think']}, seg={turn_counts['segment']}, obs={turn_counts['observation']}), "
            f"num_bboxes={len(bboxes)}, "
            f"scores=(ans={answer_score:.2f}, bbox={bbox_score:.2f}, fmt={format_score:.2f}, seg={segment_score:.2f}), "
            f"gt_segs={gt_segs_str}, pred_segs={pred_segs_str}, "
            f"final={final_score:.4f}, time={elapsed_time:.2f}s"
        )

        # 每 N 个样本打印统计
        if _reward_stats[stats_key]["total_calls"] % log_every_n == 0:
            print_reward_stats()

    # 保存样本 (使用预先计算的 should_save_sample，确保与可视化保存同步)
    if should_save_sample:
        # 获取原始输入文本 (prompt_str)
        prompt_str = extra_info.get("prompt_str", "")

        sample_data = {
            "video_id": video_id,
            "video_path": video_path,
            "question": question[:500] if question else "",
            "input_text": prompt_str,  # 原始输入的完整文本
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
            "gt_segments": gt_segments,
            # bboxes
            "bboxes": bboxes,
            "bbox_details": bbox_details,
            # 分数
            "scores": {
                "answer_score": answer_score,
                "format_score": format_score,
                "bbox_score": bbox_score,
                "segment_score": segment_score,
                "segment_iou": segment_iou,
                "segment_iop": segment_iop,
                "segment_iog": segment_iog,
                "bbox_coverage": bbox_coverage,
                "expected_bbox_count": expected_bbox_count,
                "actual_bbox_count": len(bbox_details),
                "partial_bbox_count": partial_bbox_count,
                "final_score": final_score,
            },
            "vlm_explanation": vlm_explanation[:200] if vlm_explanation else "",
            "config": {
                "use_vlm_scoring": use_vlm_scoring,
                "use_bbox_verification": use_bbox_verification,
                "vlm_endpoint": vlm_endpoint,
                "weights": {
                    "answer": answer_weight,
                    "format": format_weight,
                    "bbox": bbox_weight,
                    "segment": segment_weight,
                },
                "bbox_iou_config": {
                    "temporal_weight": temporal_weight,
                    "spatial_weight": spatial_weight,
                    "iou_threshold": iou_threshold,
                    "bbox_metric": bbox_metric,
                }
            },
            "elapsed_time": elapsed_time,
        }
        save_reward_sample(
            sample_data,
            output_dir=os.path.join(log_dir, "samples"),
            training_step=training_step,
            sample_uid=sample_uid,
        )

    # 返回字典格式，支持 FILTER_GROUPS_METRIC=acc 或 score
    # 所有返回的 key 都会被记录到 TensorBoard
    # - score: 最终分数 (float)
    # - acc: 答案是否正确 (bool，用于 filter_groups)
    # - format: 格式是否正确 (bool)
    # - answer_score/format_score/bbox_score/segment_score: 各组件分数 (float，用于 GDPO)
    # - bbox_details: 每个 bbox 的详细信息和分数 (用于 token placement)
    # - segment_details: 每轮的 segment 分数 (用于 token placement)

    # 构建 segment_details (每轮的 segment 分数)
    # 使用 per-turn 独立评分：每个 turn 的 segments 独立与 GT 比较
    segment_details = []
    gt_segments = extra_info.get("reference_segments", [])
    if all_segments:
        if gt_segments:
            # 计算每轮独立的 segment 分数
            per_turn_scores = compute_per_turn_segment_score(all_segments, gt_segments)
            for turn_idx, (turn_segments, turn_score) in enumerate(zip(all_segments, per_turn_scores)):
                segment_details.append({
                    "turn_idx": turn_idx,
                    "segments": turn_segments,
                    "score": turn_score,  # 每轮独立评分
                })
        else:
            # 没有 GT，所有 turn 的 segment 分数都是 0
            for turn_idx, turn_segments in enumerate(all_segments):
                segment_details.append({
                    "turn_idx": turn_idx,
                    "segments": turn_segments,
                    "score": 0.0,
                })

    # 为 bbox_details 添加归一化的 score 字段 (用于 token placement)
    # bbox_details 中已经有 total_score，直接使用
    for bd in bbox_details:
        # 添加统一的 score 字段 (token placement 会读取这个)
        bd["score"] = bd.get("total_score", 0.0)
        # 添加位置信息 (从 bbox_info 中获取)
        bbox_info = bd.get("bbox_info", {})
        bd["char_pos"] = bbox_info.get("char_pos", -1)
        bd["turn_idx"] = bbox_info.get("turn_idx", -1)

    return {
        "score": final_score,
        "acc": answer_score == 1.0,  # 二元分类：答案正确为 True
        "format": format_score == 1.0,  # 格式正确为 True
        # 各组件的原始分数 (用于 TensorBoard 分析和 GDPO)
        "answer_score": answer_score,
        "format_score": format_score,
        "bbox_score": bbox_score,  # 已融合数量惩罚：未输出的 bbox 视为 0 分参与平均
        "segment_score": segment_score,
        # 日志信息（不参与 GDPO 计算）
        "bbox_coverage": bbox_coverage,  # 覆盖率：实际bbox数/期望bbox数 (仅用于日志)
        # Token placement 详细信息
        "bbox_details": bbox_details,  # 每个 bbox 的详细信息 (含 score, char_pos, turn_idx)
        "segment_details": segment_details,  # 每轮的 segment 信息 (含 score)
    }
