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

# Detailed prompt encouraging gradient scoring, stricter evaluation
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


async def verify_single_bbox_with_vlm(
    frame_path: str,
    bbox: List[float],
    object_name: str,
    vlm_endpoint: str,
    vlm_model_name: str,
    vlm_api_key: str = "",
    temp_image_dir: str = "/tmp/bbox_verify",
    save_visualization: bool = False,
    visualization_dir: str = "./reward_logs/bbox_vis",
    bbox_coord_range: float = 1.0,  # bbox 坐标范围 (1.0 = [0,1], 1000.0 = [0,1000])
) -> Tuple[float, str]:
    """
    使用 VLM 验证单个 bbox 的准确性

    Args:
        frame_path: 帧图片路径
        bbox: [x1, y1, x2, y2] 坐标
        object_name: 目标物体名称
        vlm_endpoint: VLM 服务地址
        vlm_model_name: VLM 模型名称
        vlm_api_key: VLM API Key
        temp_image_dir: 临时图片目录 (unused, kept for API compatibility)
        save_visualization: 是否保存可视化图片
        visualization_dir: 可视化图片保存目录
        bbox_coord_range: bbox 坐标的范围 (1.0 = [0,1], 1000.0 = [0,1000])

    Returns:
        (score, explanation): 分数(0-1) 和解释
    """
    logger = get_reward_logger()
    saved_vis_path = None
    score = 0.5
    response_text = ""

    try:
        # 1. 加载图片并绘制 bbox
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

        # 绘制更醒目的 bbox：粗红框 + 半透明填充
        line_width = max(4, min(img_width, img_height) // 150)  # 自适应线宽
        draw.rectangle([x1_px, y1_px, x2_px, y2_px], outline="red", width=line_width)

        # 绘制物体名称标签（大字体 + 背景色）
        try:
            # 尝试加载大字体
            from PIL import ImageFont
            font_size = max(20, min(img_width, img_height) // 25)  # 自适应字体大小
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
            except Exception:
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", font_size)
                except Exception:
                    font = ImageFont.load_default()

            label_text = f" {object_name} "
            label_y = max(0, y1_px - font_size - 8)

            # 绘制标签背景
            text_bbox = draw.textbbox((x1_px, label_y), label_text, font=font)
            draw.rectangle(
                [text_bbox[0] - 2, text_bbox[1] - 2, text_bbox[2] + 2, text_bbox[3] + 2],
                fill="red"
            )
            draw.text((x1_px, label_y), label_text, fill="white", font=font)
        except Exception:
            # 回退：默认字体
            draw.text((x1_px, max(0, y1_px - 15)), f"{object_name}", fill="red")

        # 2. 转换为 base64 (用于 VLM 调用)
        img_base64_url = image_to_base64(img, format="JPEG")

        # 3. 构建请求 (OpenAI兼容格式，使用 base64)
        prompt = BBOX_VERIFY_PROMPT.format(object_name=object_name)

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
            "temperature": 0.3,  # 稍高 temperature 增加评分差异
            "max_tokens": 16,
        }

        # 4. 发送请求 (带 API Key)
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
                    logger.warning(f"BBox verify HTTP error {resp.status}: {error_text[:100]}")
                    response_text = f"HTTP error {resp.status}: {error_text[:200]}"

        # 5. 保存可视化图片 (在 VLM 调用之后，包含评分)
        if save_visualization:
            os.makedirs(visualization_dir, exist_ok=True)

            # 在图片上添加 VLM 评分
            try:
                from PIL import ImageFont
                score_font_size = max(24, min(img_width, img_height) // 20)
                try:
                    score_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", score_font_size)
                except Exception:
                    try:
                        score_font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", score_font_size)
                    except Exception:
                        score_font = ImageFont.load_default()

                score_text = f"Score: {score*10:.0f}/10"
                score_color = "green" if score >= 0.7 else "orange" if score >= 0.4 else "red"

                # 绘制评分背景
                text_bbox = draw.textbbox((5, 5), score_text, font=score_font)
                draw.rectangle(
                    [text_bbox[0] - 4, text_bbox[1] - 4, text_bbox[2] + 4, text_bbox[3] + 4],
                    fill="white", outline=score_color, width=2
                )
                draw.text((5, 5), score_text, fill=score_color, font=score_font)
            except Exception:
                draw.text((5, 5), f"Score: {score*10:.0f}/10", fill="green" if score >= 0.7 else "red")

            vis_filename = f"bbox_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}_score{score*10:.0f}.jpg"
            saved_vis_path = os.path.join(visualization_dir, vis_filename)
            img.save(saved_vis_path, "JPEG", quality=90)
            logger.debug(f"Saved bbox vis: {saved_vis_path}")

        logger.debug(f"BBox verify: obj={object_name}, bbox={bbox}, score={score:.2f}, vis={saved_vis_path}")
        return score, response_text

    except Exception as e:
        logger.warning(f"BBox verify exception: {str(e)}")
        return 0.5, f"Error: {str(e)}"


async def verify_bboxes_with_vlm(
    bboxes: List[Dict],
    video_path: str,
    vlm_endpoint: str,
    vlm_model_name: str,
    vlm_api_key: str = "",
    cache_dir: str = ".cache",
    cache_fps: int = 1,
    cache_max_frames: int = 512,
    temp_image_dir: str = "/tmp/bbox_verify",
    save_bbox_visualization: bool = False,
    bbox_vis_sample_rate: float = 0.1,
    visualization_dir: str = "./reward_logs/bbox_vis",
    bbox_coord_range: float = 1.0,
) -> Tuple[float, List[Dict]]:
    """
    验证所有 bbox 并返回平均分数

    Args:
        bboxes: 从模型输出提取的 bbox 列表
        video_path: 视频路径（用于加载帧）
        vlm_endpoint: VLM 服务地址
        vlm_model_name: VLM 模型名称
        vlm_api_key: VLM API Key
        cache_dir: 帧缓存目录
        cache_fps: 缓存帧的fps
        cache_max_frames: 缓存的最大帧数
        temp_image_dir: 临时图片目录
        save_bbox_visualization: 是否保存 bbox 可视化图片
        bbox_vis_sample_rate: 可视化采样率 (0.1 = 10%)
        visualization_dir: 可视化图片保存目录

    Returns:
        (average_score, details): 平均分数和详细信息
    """
    logger = get_reward_logger()

    if not bboxes:
        return 0.5, []  # 没有 bbox 返回中性分数

    details = []
    scores = []

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
            # 按采样率决定是否保存可视化
            should_save_vis = save_bbox_visualization and (random.random() < bbox_vis_sample_rate)
            tasks.append(verify_single_bbox_with_vlm(
                frame_path=frame_path,
                bbox=bbox_info['bbox'],
                object_name=bbox_info['object'],
                vlm_endpoint=vlm_endpoint,
                vlm_model_name=vlm_model_name,
                vlm_api_key=vlm_api_key,
                temp_image_dir=temp_image_dir,
                save_visualization=should_save_vis,
                visualization_dir=visualization_dir,
                bbox_coord_range=bbox_coord_range,
            ))
            valid_bbox_indices.append(i)
        else:
            logger.debug(f"Frame not found for timestamp {bbox_info['time']}, video={video_path}")

    if tasks:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for idx, result in zip(valid_bbox_indices, results):
            bbox_info = bboxes[idx]
            if isinstance(result, Exception):
                score, explanation = 0.5, str(result)
            else:
                score, explanation = result

            scores.append(score)
            details.append({
                "bbox_info": bbox_info,
                "score": score,
                "explanation": explanation[:200] if explanation else "",
            })

    avg_score = sum(scores) / len(scores) if scores else 0.5
    logger.debug(f"BBox verification: {len(tasks)} tasks, avg_score={avg_score:.4f}")
    return avg_score, details


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
    使用 VLM 判断答案是否正确（用于开放题）
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
                        score = float(score_match.group(1)) / 10.0
                        logger.debug(f"VLM answer score: {score:.2f}, response: {response_text[:50]}")
                        return min(1.0, max(0.0, score)), response_text
                    return 0.5, response_text
                else:
                    error_text = await resp.text()
                    logger.warning(f"VLM score HTTP error {resp.status}: {error_text[:100]}")
                    return 0.5, f"HTTP error {resp.status}: {error_text[:200]}"
    except Exception as e:
        logger.warning(f"VLM score exception: {str(e)}")
        return 0.5, f"Error: {str(e)}"


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
    temp_image_dir: str = "/tmp/bbox_verify",
    use_vlm_scoring: bool = True,
    use_bbox_verification: bool = True,
    answer_weight: float = 0.4,
    bbox_weight: float = 0.3,
    vlm_weight: float = 0.3,
    # BBox 参数
    bbox_coord_range: float = 1.0,  # bbox 坐标范围 (1000 = [0,1000], 1 = [0,1])
    save_bbox_visualization: bool = False,
    bbox_vis_sample_rate: float = 0.1,  # 采样率：0.1 = 10% 的 bbox 保存可视化
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
        temp_image_dir: 临时图片目录
        use_vlm_scoring: 是否使用 VLM 对答案打分
        use_bbox_verification: 是否验证 bbox
        answer_weight: 答案分数权重
        bbox_weight: bbox 分数权重
        vlm_weight: VLM 打分权重
        enable_logging: 是否启用日志
        save_samples: 是否保存样本
        save_every_n: 每 N 个样本保存一次 (1=全部保存, 10=每10个保存1个)
        log_dir: 日志目录
        log_every_n: 每 N 个样本打印一次统计

    Returns:
        dict: {score, acc, answer_score, bbox_score, vlm_score, ...}
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

    # 5. BBox 验证分数 (VLM, 异步)
    bbox_score = 0.5
    bbox_details = []
    bbox_verified = False

    if use_bbox_verification and vlm_endpoint and bboxes and video_path:
        bbox_score, bbox_details = await verify_bboxes_with_vlm(
            bboxes=bboxes,
            video_path=video_path,
            vlm_endpoint=vlm_endpoint,
            vlm_model_name=vlm_model_name,
            vlm_api_key=vlm_api_key,
            cache_dir=cache_dir,
            cache_fps=cache_fps,
            cache_max_frames=cache_max_frames,
            temp_image_dir=temp_image_dir,
            save_bbox_visualization=save_bbox_visualization,
            bbox_vis_sample_rate=bbox_vis_sample_rate,
            visualization_dir=os.path.join(log_dir, "bbox_vis"),
            bbox_coord_range=bbox_coord_range,
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

    # 更新统计
    stats_key = data_source or "default"
    _reward_stats[stats_key]["total_calls"] += 1
    _reward_stats[stats_key]["total_score"] += final_score
    if answer_score > 0.5:  # VLM 返回 > 0.5 视为正确
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
            f"scores=(ans={answer_score:.2f}, bbox={bbox_score:.2f}), "
            f"final={final_score:.4f}, time={elapsed_time:.2f}s"
        )

        # 每 N 个样本打印统计
        if _reward_stats[stats_key]["total_calls"] % log_every_n == 0:
            print_reward_stats()

    # 保存样本 (按间隔)
    if save_samples and (_reward_stats[stats_key]["total_calls"] % save_every_n == 0):
        sample_data = {
            "video_id": video_id,
            "video_path": video_path,
            "question": question[:500] if question else "",
            "ground_truth": ground_truth,
            "predicted_answer": predicted_answer,
            "answer_correct": answer_score > 0.5,
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
                }
            },
            "elapsed_time": elapsed_time,
        }
        save_reward_sample(sample_data, output_dir=os.path.join(log_dir, "samples"))

    # 返回 float 分数 (veRL metrics 处理期望 float，不是 dict)
    return final_score
