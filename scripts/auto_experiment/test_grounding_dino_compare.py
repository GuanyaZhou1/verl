#!/usr/bin/env python3
"""
Grounding DINO vs Qwen3-VL 对比测试

对比三种 grounding 方案：
1. Qwen3-VL 235B (当前方案，新 prompt)
2. Grounding DINO (专用 grounding 模型)
3. 原始标注 (ground truth)

Usage:
    python test_grounding_dino_compare.py --image /path/to/image.jpg --object "man in dark suit"
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import torch
from PIL import Image, ImageDraw, ImageFont

# ============================================================
# Grounding DINO (HuggingFace Transformers)
# ============================================================
GROUNDING_DINO_AVAILABLE = False
try:
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    GROUNDING_DINO_AVAILABLE = True
except ImportError:
    print("[WARN] transformers not available or missing GroundingDino support")


class GroundingDINODetector:
    """Grounding DINO 检测器"""

    def __init__(self, model_id: str = "IDEA-Research/grounding-dino-base", device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"[GroundingDINO] Loading model {model_id} on {self.device}...")

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
        self.model.eval()
        print(f"[GroundingDINO] Model loaded successfully")

    def detect(self, image_path: str, text_query: str, threshold: float = 0.25) -> Optional[List[float]]:
        """
        检测物体并返回 bbox

        Args:
            image_path: 图片路径
            text_query: 要检测的物体描述
            threshold: 置信度阈值

        Returns:
            [x1, y1, x2, y2] 归一化到 0-1，或 None 如果未检测到
        """
        try:
            image = Image.open(image_path).convert("RGB")
            w, h = image.size

            # Grounding DINO 需要在 query 后加句号
            if not text_query.endswith('.'):
                text_query = text_query + '.'

            inputs = self.processor(images=image, text=text_query, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=threshold,
                text_threshold=threshold,
                target_sizes=[(h, w)]
            )[0]

            if len(results["boxes"]) == 0:
                return None

            # 取置信度最高的 bbox
            best_idx = results["scores"].argmax().item()
            box = results["boxes"][best_idx].tolist()
            score = results["scores"][best_idx].item()

            # 归一化到 0-1
            bbox = [
                box[0] / w,  # x1
                box[1] / h,  # y1
                box[2] / w,  # x2
                box[3] / h,  # y2
            ]

            print(f"[GroundingDINO] Found '{text_query}' with score {score:.3f}, bbox={bbox}")
            return bbox

        except Exception as e:
            print(f"[GroundingDINO] Error: {e}")
            return None


# ============================================================
# Qwen3-VL Detector (使用新 prompt)
# ============================================================
class Qwen3VLDetector:
    """Qwen3-VL 检测器"""

    def __init__(self, endpoint: str = "10.0.1.35:8081",
                 model: str = "Qwen3-VL-235B-A22B-Instruct",
                 api_key: str = "123456"):
        self.endpoint = endpoint
        self.model = model
        self.api_key = api_key

    async def detect(self, image_path: str, object_name: str) -> Optional[List[float]]:
        """检测物体"""
        # 复用 video_reasoning_async 的函数
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from verl.utils.reward_score.video_reasoning_async import get_gt_bbox_from_vlm

        bbox, response = await get_gt_bbox_from_vlm(
            frame_path=image_path,
            object_name=object_name,
            context='',
            vlm_endpoint=self.endpoint,
            vlm_model_name=self.model,
            vlm_api_key=self.api_key,
        )

        print(f"[Qwen3-VL] Response: {response[:200]}...")
        return bbox


# ============================================================
# Visualization
# ============================================================
def draw_comparison(
    image_path: str,
    object_name: str,
    bbox_qwen: Optional[List[float]],
    bbox_dino: Optional[List[float]],
    bbox_gt: Optional[List[float]] = None,
    output_path: str = None,
) -> str:
    """绘制对比图"""

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        font = ImageFont.load_default()

    def draw_bbox(bbox, label, color, offset_y=0):
        if bbox is None:
            return
        x1, y1 = int(bbox[0] * w), int(bbox[1] * h)
        x2, y2 = int(bbox[2] * w), int(bbox[3] * h)
        for i in range(3):
            draw.rectangle([x1-i, y1-i, x2+i, y2+i], outline=color)
        # Label
        tb = draw.textbbox((0, 0), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        ly = max(0, y1 - th - 8 + offset_y)
        draw.rectangle([x1, ly, x1 + tw + 8, ly + th + 4], fill=color)
        draw.text((x1 + 4, ly + 2), label, fill='white', font=font)

    # Draw bboxes
    if bbox_gt:
        draw_bbox(bbox_gt, "GT", "red", offset_y=0)
    if bbox_qwen:
        draw_bbox(bbox_qwen, "Qwen3-VL", "limegreen", offset_y=-25)
    if bbox_dino:
        draw_bbox(bbox_dino, "GroundingDINO", "dodgerblue", offset_y=-50)

    # Legend
    legend_items = [
        ("GT (Ground Truth)", "red", bbox_gt is not None),
        ("Qwen3-VL 235B", "limegreen", bbox_qwen is not None),
        ("Grounding DINO", "dodgerblue", bbox_dino is not None),
    ]

    legend_y = h - 100
    for i, (label, color, found) in enumerate(legend_items):
        y = legend_y + i * 25
        status = "✓" if found else "✗"
        draw.rectangle([10, y, 25, y + 15], fill=color if found else "gray")
        draw.text((30, y), f"{status} {label}", fill='white', font=font)

    # Title
    title = f"Object: {object_name}"
    draw.rectangle([0, 0, w, 30], fill=(0, 0, 0, 180))
    draw.text((10, 5), title, fill='yellow', font=font)

    # Save
    if output_path is None:
        output_path = image_path.replace('.jpg', '_compare.jpg').replace('.png', '_compare.png')
    img.save(output_path, quality=95)
    print(f"[Saved] {output_path}")
    return output_path


# ============================================================
# Main
# ============================================================
async def main():
    parser = argparse.ArgumentParser(description="Compare Grounding DINO vs Qwen3-VL")
    parser.add_argument("--image", type=str, required=True, help="Image path")
    parser.add_argument("--object", type=str, required=True, help="Object to detect")
    parser.add_argument("--output", type=str, default=None, help="Output image path")
    parser.add_argument("--vlm-endpoint", type=str, default="10.0.1.35:8081")
    parser.add_argument("--vlm-model", type=str, default="Qwen3-VL-235B-A22B-Instruct")
    parser.add_argument("--dino-model", type=str, default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--dino-threshold", type=float, default=0.25)
    parser.add_argument("--skip-dino", action="store_true", help="Skip Grounding DINO")
    parser.add_argument("--skip-qwen", action="store_true", help="Skip Qwen3-VL")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Grounding Comparison Test")
    print(f"Image: {args.image}")
    print(f"Object: {args.object}")
    print(f"{'='*60}\n")

    bbox_qwen = None
    bbox_dino = None

    # Test Qwen3-VL
    if not args.skip_qwen:
        print("\n[1] Testing Qwen3-VL...")
        qwen = Qwen3VLDetector(endpoint=args.vlm_endpoint, model=args.vlm_model)
        bbox_qwen = await qwen.detect(args.image, args.object)
        print(f"    Result: {bbox_qwen}")

    # Test Grounding DINO
    if not args.skip_dino and GROUNDING_DINO_AVAILABLE:
        print("\n[2] Testing Grounding DINO...")
        dino = GroundingDINODetector(model_id=args.dino_model)
        bbox_dino = dino.detect(args.image, args.object, threshold=args.dino_threshold)
        print(f"    Result: {bbox_dino}")
    elif not args.skip_dino:
        print("\n[2] Grounding DINO not available (install transformers with grounding-dino support)")

    # Generate comparison image
    print("\n[3] Generating comparison image...")
    output = draw_comparison(
        image_path=args.image,
        object_name=args.object,
        bbox_qwen=bbox_qwen,
        bbox_dino=bbox_dino,
        bbox_gt=None,
        output_path=args.output,
    )

    # Summary
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"  Qwen3-VL:      {'Found' if bbox_qwen else 'Not Found'}")
    print(f"  Grounding DINO: {'Found' if bbox_dino else 'Not Found' if not args.skip_dino else 'Skipped'}")
    print(f"  Output: {output}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(main())
