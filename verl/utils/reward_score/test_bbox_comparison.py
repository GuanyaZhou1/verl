#!/usr/bin/env python3
"""
SFT vs RL BBox 能力对比测试 (以 Qwen3-VL-30B 为 GT)

从 reward_samples JSONL 中读取 bboxes（含 object, timestamp），
抽取对应视频帧，分别发送给 GT(30B)、SFT、RL 三个模型预测 bbox，
以 GT 为基准计算 IOU，可视化对比 SFT 和 RL 的 bbox 能力。

Usage:
    python verl/utils/reward_score/test_bbox_comparison.py \
        --sample_file reward_logs/samples/reward_samples_20260210.jsonl \
        --num_samples 200 \
        --gt_endpoint "10.96.11.3:8081" \
        --sft_endpoint "10.96.11.4:8081" \
        --rl_endpoint "10.96.11.4:8082" \
        --save_vis
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
from typing import List, Dict, Optional, Tuple
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


# ============== Utility Functions ==============

def image_to_base64(image: Image.Image, fmt: str = "JPEG") -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt, quality=85)
    buffer.seek(0)
    img_bytes = buffer.read()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    mime_type = f"image/{fmt.lower()}"
    return f"data:{mime_type};base64,{img_base64}"


def extract_bbox_context(solution_str: str, object_name: str) -> str:
    escaped_name = re.escape(object_name)
    pattern = rf'[^.]*<obj>{escaped_name}</obj><box>\[[^\]]+\]</box>at<t>[^<]+</t>[^.]*\.?'
    match = re.search(pattern, solution_str, re.IGNORECASE)
    if match:
        context = match.group(0)
        context = re.sub(r'<obj>([^<]+)</obj>', r'\1', context)
        context = re.sub(r'<box>\[[^\]]+\]</box>at<t>[^<]+</t>', '', context)
        return context.strip()
    pattern = rf'[^.]*{escaped_name}[^.]*\.?'
    match = re.search(pattern, solution_str, re.IGNORECASE)
    if match:
        return match.group(0).strip()
    return object_name


def compute_iou(bbox1: Optional[List[float]], bbox2: Optional[List[float]]) -> float:
    if bbox1 is None or bbox2 is None:
        return 0.0
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union_area = bbox1_area + bbox2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def get_frame_for_timestamp(
    video_path: str, timestamp: float, cache_dir: str = ".cache",
    fps: int = 1, max_frames: int = 512,
) -> Optional[str]:
    cache = VideoFrameCache(cache_dir=cache_dir, fps=fps, max_frames=max_frames)
    cache_video_dir = cache._get_cache_dir_for_video(video_path)
    metadata_path = cache._get_metadata_path(video_path)
    if not metadata_path.exists():
        try:
            cache.cache_video(video_path)
        except Exception as e:
            print(f"Error caching video {video_path}: {e}")
            return None
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    closest_frame, min_diff = None, float('inf')
    for fi in metadata['frames']:
        diff = abs(fi['timestamp'] - timestamp)
        if diff < min_diff:
            min_diff = diff
            closest_frame = fi
    if closest_frame:
        return str(cache_video_dir / closest_frame['path'])
    return None


def save_bbox_visualization(
    frame_path: str,
    object_name: str,
    gt_bbox: Optional[List[float]],
    rl_bbox: Optional[List[float]],
    sft_bbox: Optional[List[float]],
    iou_rl_gt: float,
    iou_sft_gt: float,
    iou_rl_sft: float,
    output_path: str,
):
    """
    可视化 3 个 bbox:
    - 绿色 (粗): GT (Qwen3-VL-30B)
    - 红色: RL 模型
    - 蓝色: SFT 模型
    """
    from PIL import ImageDraw, ImageFont

    img = Image.open(frame_path).convert("RGB")
    img_width, img_height = img.size
    draw = ImageDraw.Draw(img)

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

    def draw_bbox_norm(bbox, color, line_w=line_width):
        if bbox is None:
            return
        x1, y1, x2, y2 = bbox
        px1, py1 = int(x1 * img_width), int(y1 * img_height)
        px2, py2 = int(x2 * img_width), int(y2 * img_height)
        if px1 > px2: px1, px2 = px2, px1
        if py1 > py2: py1, py2 = py2, py1
        if px1 == px2 or py1 == py2:
            return
        draw.rectangle([px1, py1, px2, py2], outline=color, width=line_w)

    # GT 绿色最粗, RL 红色, SFT 蓝色
    draw_bbox_norm(gt_bbox, "lime", line_width + 3)
    draw_bbox_norm(rl_bbox, "red", line_width + 1)
    draw_bbox_norm(sft_bbox, "cyan", line_width + 1)

    info_text = [
        f"Object: {object_name[:35]}{'...' if len(object_name) > 35 else ''}",
        f"",
        f"--- IOU vs GT(30B) ---",
        f"RL  vs GT: {iou_rl_gt:.2f}",
        f"SFT vs GT: {iou_sft_gt:.2f}",
        f"RL vs SFT: {iou_rl_sft:.2f}",
        f"",
        f"--- Legend ---",
        f"GREEN = GT (30B)",
        f"RED   = RL",
        f"CYAN  = SFT",
    ]

    padding = 5
    max_text_width = 0
    total_height = 0
    for text in info_text:
        bt = draw.textbbox((0, 0), text, font=small_font)
        max_text_width = max(max_text_width, bt[2] - bt[0])
        total_height += bt[3] - bt[1] + 3

    bg_x1, bg_y1 = 5, 5
    bg_x2 = bg_x1 + max_text_width + 2 * padding
    bg_y2 = bg_y1 + total_height + 2 * padding

    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    ImageDraw.Draw(overlay).rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=(0, 0, 0, 200))
    img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
    draw = ImageDraw.Draw(img)

    cy = bg_y1 + padding
    for text in info_text:
        color = "white"
        if "GREEN" in text: color = "lime"
        elif "RED" in text: color = "red"
        elif "CYAN" in text: color = "cyan"
        draw.text((bg_x1 + padding, cy), text, fill=color, font=small_font)
        bt = draw.textbbox((0, 0), text, font=small_font)
        cy += bt[3] - bt[1] + 3

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path, "JPEG", quality=95)


# ============== Model Query ==============

async def query_model_bbox(
    frame_path: str, object_name: str, context: str,
    endpoint: str, model_name: str,
    api_key: str = "123456", debug: bool = False,
    img_base64_url: Optional[str] = None,
) -> Tuple[Optional[List[float]], str]:
    """向 VLM 模型查询 bbox。可复用 img_base64_url 避免重复编码。"""
    raw_response = ""
    try:
        if img_base64_url is None:
            img = Image.open(frame_path).convert("RGB")
            img_base64_url = image_to_base64(img, fmt="JPEG")

        prompt = BBOX_DETECT_PROMPT.format(object_name=object_name, context=context)

        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": img_base64_url}},
                {"type": "text", "text": prompt}
            ]}],
            "temperature": 0.1,
            "max_tokens": 128,
        }

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        timeout = aiohttp.ClientTimeout(total=120)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            url = f"http://{endpoint}/v1/chat/completions"
            async with session.post(url, json=payload, headers=headers) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    raw_response = result["choices"][0]["message"]["content"].strip()
                    if debug:
                        print(f"      [{model_name}] raw: {raw_response[:200]}")
                    if "none" in raw_response.lower() or "not visible" in raw_response.lower() or "cannot" in raw_response.lower():
                        return None, raw_response

                    # 解析 bbox - 多种格式
                    for pat in [
                        r'\[([0-9.]+),\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)\]',
                        r'\(([0-9.]+),\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)\)',
                    ]:
                        m = re.search(pat, raw_response)
                        if m:
                            bbox = [float(m.group(i)) for i in range(1, 5)]
                            mx = max(bbox)
                            if mx > 1:
                                scale = 1000.0 if mx <= 1000 else mx
                                bbox = [v / scale for v in bbox]
                            if all(0 <= v <= 1 for v in bbox) and bbox[0] < bbox[2] and bbox[1] < bbox[3]:
                                return bbox, raw_response

                    numbers = re.findall(r'([0-9]+\.?[0-9]*)', raw_response)
                    if len(numbers) >= 4:
                        bbox = [float(numbers[i]) for i in range(4)]
                        mx = max(bbox)
                        if mx > 1:
                            scale = 1000.0 if mx <= 1000 else mx
                            bbox = [v / scale for v in bbox]
                        if all(0 <= v <= 1 for v in bbox) and bbox[0] < bbox[2] and bbox[1] < bbox[3]:
                            return bbox, raw_response

                    return None, raw_response
                else:
                    error_text = await resp.text()
                    return None, f"HTTP error {resp.status}: {error_text[:200]}"
    except Exception as e:
        return None, f"Error: {str(e)}"


# ============== Data Loading ==============

def load_samples_with_bboxes(sample_file: str, num_samples: int = 30) -> List[Dict]:
    samples = []
    with open(sample_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
                if sample.get('bboxes') and len(sample['bboxes']) > 0:
                    samples.append(sample)
            except json.JSONDecodeError:
                continue
    print(f"Loaded {len(samples)} samples with bboxes (total in file)")
    if len(samples) > num_samples:
        samples = random.sample(samples, num_samples)
    return samples


# ============== Main ==============

async def run_comparison(
    sample_file: str,
    num_samples: int = 200,
    gt_endpoint: str = "10.96.11.3:8081",
    gt_model_name: str = "Qwen3-VL-30B-A3B-Instruct",
    sft_endpoint: str = "10.96.11.4:8081",
    sft_model_name: str = "Qwen2.5-VL-7B-sft",
    rl_endpoint: str = "10.96.11.4:8082",
    rl_model_name: str = "Qwen2.5-VL-7B-RL",
    api_key: str = "123456",
    cache_dir: str = ".cache",
    output_file: str = "bbox_gt_sft_rl_results.json",
    vis_dir: str = "./bbox_vis_gt_sft_rl",
    save_vis: bool = False,
    max_bboxes_per_sample: int = 3,
    debug: bool = False,
):
    print(f"Loading samples from {sample_file}...")
    samples = load_samples_with_bboxes(sample_file, num_samples)
    print(f"Selected {len(samples)} samples for comparison")
    print(f"GT  endpoint: {gt_endpoint} ({gt_model_name})")
    print(f"SFT endpoint: {sft_endpoint} ({sft_model_name})")
    print(f"RL  endpoint: {rl_endpoint} ({rl_model_name})")

    results = []
    processed = 0

    for sample_idx, sample in enumerate(samples):
        video_path = sample.get('video_path')
        solution_str = sample.get('solution_str_full', '')
        video_id = sample.get('video_id', f'sample_{sample_idx}')

        if not video_path or not os.path.exists(video_path):
            print(f"Skip sample {video_id}: video not found at {video_path}")
            continue

        bboxes = sample.get('bboxes', [])
        if len(bboxes) > max_bboxes_per_sample:
            bboxes = random.sample(bboxes, max_bboxes_per_sample)

        for bbox_info in bboxes:
            obj_name = bbox_info.get('object', '')
            timestamp = bbox_info.get('time', 0.0)
            rollout_bbox = bbox_info.get('bbox', [])

            if not obj_name or len(rollout_bbox) != 4:
                continue

            frame_path = get_frame_for_timestamp(video_path, timestamp, cache_dir)
            if not frame_path or not os.path.exists(frame_path):
                print(f"  Skip bbox: frame not found for t={timestamp}")
                continue

            context = extract_bbox_context(solution_str, obj_name)

            # 预编码图片，3 个模型共享
            img = Image.open(frame_path).convert("RGB")
            img_b64 = image_to_base64(img, fmt="JPEG")

            processed += 1
            print(f"\n[{processed}] {video_id} | '{obj_name}' | t={timestamp}s")

            # 并发查询 3 个模型
            gt_task = query_model_bbox(
                frame_path, obj_name, context,
                gt_endpoint, gt_model_name, api_key, debug, img_b64
            )
            sft_task = query_model_bbox(
                frame_path, obj_name, context,
                sft_endpoint, sft_model_name, api_key, debug, img_b64
            )
            rl_task = query_model_bbox(
                frame_path, obj_name, context,
                rl_endpoint, rl_model_name, api_key, debug, img_b64
            )

            (gt_bbox, gt_resp), (sft_bbox, sft_resp), (rl_bbox, rl_resp) = \
                await asyncio.gather(gt_task, sft_task, rl_task)

            iou_rl_gt = compute_iou(rl_bbox, gt_bbox)
            iou_sft_gt = compute_iou(sft_bbox, gt_bbox)
            iou_rl_sft = compute_iou(rl_bbox, sft_bbox)

            gt_str = f"{[round(v,3) for v in gt_bbox]}" if gt_bbox else "None"
            rl_str = f"{[round(v,3) for v in rl_bbox]}" if rl_bbox else "None"
            sft_str = f"{[round(v,3) for v in sft_bbox]}" if sft_bbox else "None"
            print(f"    GT:  {gt_str}")
            print(f"    RL:  {rl_str}  | IOU vs GT: {iou_rl_gt:.2f}")
            print(f"    SFT: {sft_str}  | IOU vs GT: {iou_sft_gt:.2f}")
            print(f"    RL vs SFT IOU: {iou_rl_sft:.2f}")

            result = {
                'video_id': video_id,
                'object': obj_name,
                'timestamp': timestamp,
                'context': context[:200],
                'frame_path': frame_path,
                'rollout_bbox': rollout_bbox,
                'gt_bbox': gt_bbox,
                'sft_bbox': sft_bbox,
                'rl_bbox': rl_bbox,
                'gt_response': gt_resp[:200],
                'sft_response': sft_resp[:200],
                'rl_response': rl_resp[:200],
                'iou_rl_gt': iou_rl_gt,
                'iou_sft_gt': iou_sft_gt,
                'iou_rl_sft': iou_rl_sft,
            }

            if save_vis:
                safe_obj = re.sub(r'[^\w\-]', '_', obj_name)[:30]
                vis_fn = f"{video_id}_{safe_obj}_{int(timestamp)}s.jpg"
                vis_path = os.path.join(vis_dir, vis_fn)
                save_bbox_visualization(
                    frame_path, obj_name,
                    gt_bbox, rl_bbox, sft_bbox,
                    iou_rl_gt, iou_sft_gt, iou_rl_sft,
                    vis_path,
                )
                result['vis_path'] = vis_path

            results.append(result)

    # ============== 统计汇总 ==============
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'sample_file': sample_file,
            'num_samples': num_samples,
            'gt_endpoint': gt_endpoint, 'gt_model_name': gt_model_name,
            'sft_endpoint': sft_endpoint, 'sft_model_name': sft_model_name,
            'rl_endpoint': rl_endpoint, 'rl_model_name': rl_model_name,
        },
        'summary': {},
        'results': results,
    }

    if results:
        gt_det = sum(1 for r in results if r['gt_bbox'] is not None)
        sft_det = sum(1 for r in results if r['sft_bbox'] is not None)
        rl_det = sum(1 for r in results if r['rl_bbox'] is not None)

        def stats(scores):
            if not scores:
                return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'n': 0}
            mean = sum(scores) / len(scores)
            std = (sum((s - mean)**2 for s in scores) / len(scores)) ** 0.5
            return {'mean': round(mean, 4), 'std': round(std, 4),
                    'min': round(min(scores), 4), 'max': round(max(scores), 4),
                    'n': len(scores)}

        # 以 GT 存在为前提的子集
        gt_exists = [r for r in results if r['gt_bbox'] is not None]
        # GT 存在 且 模型也检测到
        rl_and_gt = [r for r in gt_exists if r['rl_bbox'] is not None]
        sft_and_gt = [r for r in gt_exists if r['sft_bbox'] is not None]
        all_three = [r for r in gt_exists if r['rl_bbox'] is not None and r['sft_bbox'] is not None]

        # RL 赢 / SFT 赢 / 打平 (在 all_three 子集中)
        rl_wins = sum(1 for r in all_three if r['iou_rl_gt'] > r['iou_sft_gt'] + 0.05)
        sft_wins = sum(1 for r in all_three if r['iou_sft_gt'] > r['iou_rl_gt'] + 0.05)
        ties = len(all_three) - rl_wins - sft_wins

        output_data['summary'] = {
            'total_bboxes': len(results),
            'gt_detected': gt_det,
            'sft_detected': sft_det,
            'rl_detected': rl_det,
            'all_three_detected': len(all_three),
            'iou_rl_vs_gt_all': stats([r['iou_rl_gt'] for r in results]),
            'iou_sft_vs_gt_all': stats([r['iou_sft_gt'] for r in results]),
            'iou_rl_vs_gt_when_gt_exists': stats([r['iou_rl_gt'] for r in gt_exists]),
            'iou_sft_vs_gt_when_gt_exists': stats([r['iou_sft_gt'] for r in gt_exists]),
            'iou_rl_vs_gt_both_detect': stats([r['iou_rl_gt'] for r in rl_and_gt]),
            'iou_sft_vs_gt_both_detect': stats([r['iou_sft_gt'] for r in sft_and_gt]),
            'iou_rl_vs_gt_all_three': stats([r['iou_rl_gt'] for r in all_three]),
            'iou_sft_vs_gt_all_three': stats([r['iou_sft_gt'] for r in all_three]),
            'iou_rl_vs_sft': stats([r['iou_rl_sft'] for r in results]),
            'head_to_head': {
                'rl_wins': rl_wins,
                'sft_wins': sft_wins,
                'ties': ties,
                'total': len(all_three),
            },
        }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    # 打印汇总
    print(f"\n{'='*70}")
    print("SFT vs RL BBOX COMPARISON (GT = Qwen3-VL-30B)")
    print(f"{'='*70}")
    print(f"Total bboxes evaluated: {len(results)}")
    if results:
        s = output_data['summary']
        print(f"\n--- Detection Rate ---")
        print(f"  GT (30B) detected: {s['gt_detected']}/{s['total_bboxes']} ({s['gt_detected']/s['total_bboxes']*100:.1f}%)")
        print(f"  SFT detected:      {s['sft_detected']}/{s['total_bboxes']} ({s['sft_detected']/s['total_bboxes']*100:.1f}%)")
        print(f"  RL  detected:      {s['rl_detected']}/{s['total_bboxes']} ({s['rl_detected']/s['total_bboxes']*100:.1f}%)")
        print(f"  All three:         {s['all_three_detected']}/{s['total_bboxes']}")

        print(f"\n--- IOU vs GT (all samples) ---")
        print(f"  RL  vs GT: Mean={s['iou_rl_vs_gt_all']['mean']:.3f}  Std={s['iou_rl_vs_gt_all']['std']:.3f}  (n={s['iou_rl_vs_gt_all']['n']})")
        print(f"  SFT vs GT: Mean={s['iou_sft_vs_gt_all']['mean']:.3f}  Std={s['iou_sft_vs_gt_all']['std']:.3f}  (n={s['iou_sft_vs_gt_all']['n']})")

        print(f"\n--- IOU vs GT (GT exists) ---")
        print(f"  RL  vs GT: Mean={s['iou_rl_vs_gt_when_gt_exists']['mean']:.3f}  Std={s['iou_rl_vs_gt_when_gt_exists']['std']:.3f}  (n={s['iou_rl_vs_gt_when_gt_exists']['n']})")
        print(f"  SFT vs GT: Mean={s['iou_sft_vs_gt_when_gt_exists']['mean']:.3f}  Std={s['iou_sft_vs_gt_when_gt_exists']['std']:.3f}  (n={s['iou_sft_vs_gt_when_gt_exists']['n']})")

        print(f"\n--- IOU vs GT (all three detected, fair comparison) ---")
        print(f"  RL  vs GT: Mean={s['iou_rl_vs_gt_all_three']['mean']:.3f}  Std={s['iou_rl_vs_gt_all_three']['std']:.3f}  (n={s['iou_rl_vs_gt_all_three']['n']})")
        print(f"  SFT vs GT: Mean={s['iou_sft_vs_gt_all_three']['mean']:.3f}  Std={s['iou_sft_vs_gt_all_three']['std']:.3f}  (n={s['iou_sft_vs_gt_all_three']['n']})")

        h2h = s['head_to_head']
        print(f"\n--- Head-to-Head (IOU vs GT, margin > 0.05) ---")
        print(f"  RL wins:  {h2h['rl_wins']}/{h2h['total']} ({h2h['rl_wins']/max(h2h['total'],1)*100:.1f}%)")
        print(f"  SFT wins: {h2h['sft_wins']}/{h2h['total']} ({h2h['sft_wins']/max(h2h['total'],1)*100:.1f}%)")
        print(f"  Ties:     {h2h['ties']}/{h2h['total']} ({h2h['ties']/max(h2h['total'],1)*100:.1f}%)")

        # RL 比 SFT 好最多的 case
        print(f"\n{'='*70}")
        print("TOP 5: RL BETTER THAN SFT (highest RL-GT IOU - SFT-GT IOU)")
        print(f"{'='*70}")
        sorted_rl_better = sorted(all_three, key=lambda x: x['iou_rl_gt'] - x['iou_sft_gt'], reverse=True)[:5]
        for i, r in enumerate(sorted_rl_better, 1):
            delta = r['iou_rl_gt'] - r['iou_sft_gt']
            print(f"  {i}. {r['video_id']} | '{r['object'][:30]}' @ {r['timestamp']}s")
            print(f"     RL-GT={r['iou_rl_gt']:.2f}  SFT-GT={r['iou_sft_gt']:.2f}  delta=+{delta:.2f}")

        # SFT 比 RL 好最多的 case
        print(f"\n{'='*70}")
        print("TOP 5: SFT BETTER THAN RL (highest SFT-GT IOU - RL-GT IOU)")
        print(f"{'='*70}")
        sorted_sft_better = sorted(all_three, key=lambda x: x['iou_sft_gt'] - x['iou_rl_gt'], reverse=True)[:5]
        for i, r in enumerate(sorted_sft_better, 1):
            delta = r['iou_sft_gt'] - r['iou_rl_gt']
            print(f"  {i}. {r['video_id']} | '{r['object'][:30]}' @ {r['timestamp']}s")
            print(f"     SFT-GT={r['iou_sft_gt']:.2f}  RL-GT={r['iou_rl_gt']:.2f}  delta=+{delta:.2f}")

    print(f"\nResults saved to: {output_file}")
    if save_vis:
        print(f"Visualizations saved to: {vis_dir}/")
    return output_data


def main():
    parser = argparse.ArgumentParser(description="SFT vs RL BBox Comparison with GT")
    parser.add_argument("--sample_file", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--max_bboxes_per_sample", type=int, default=3)
    parser.add_argument("--gt_endpoint", type=str, default="10.96.11.3:8081")
    parser.add_argument("--gt_model_name", type=str, default="Qwen3-VL-30B-A3B-Instruct")
    parser.add_argument("--sft_endpoint", type=str, default="10.96.11.4:8081")
    parser.add_argument("--sft_model_name", type=str, default="Qwen2.5-VL-7B-sft")
    parser.add_argument("--rl_endpoint", type=str, default="10.96.11.4:8082")
    parser.add_argument("--rl_model_name", type=str, default="Qwen2.5-VL-7B-RL")
    parser.add_argument("--api_key", type=str, default="123456")
    parser.add_argument("--cache_dir", type=str, default=".cache")
    parser.add_argument("--output_file", type=str, default="bbox_gt_sft_rl_results.json")
    parser.add_argument("--vis_dir", type=str, default="./bbox_vis_gt_sft_rl")
    parser.add_argument("--save_vis", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    random.seed(args.seed)

    asyncio.run(run_comparison(
        sample_file=args.sample_file,
        num_samples=args.num_samples,
        gt_endpoint=args.gt_endpoint,
        gt_model_name=args.gt_model_name,
        sft_endpoint=args.sft_endpoint,
        sft_model_name=args.sft_model_name,
        rl_endpoint=args.rl_endpoint,
        rl_model_name=args.rl_model_name,
        api_key=args.api_key,
        cache_dir=args.cache_dir,
        output_file=args.output_file,
        vis_dir=args.vis_dir,
        save_vis=args.save_vis,
        max_bboxes_per_sample=args.max_bboxes_per_sample,
        debug=args.debug,
    ))


if __name__ == "__main__":
    main()
