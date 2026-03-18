#!/usr/bin/env python3
"""
筛选中等难度样本并转换为训练数据格式

功能：
1. 从 realtime_results.jsonl 中筛选难度为 medium 的样本
2. 与原始训练数据匹配，保留原始格式
3. 保存为新的 parquet 文件

使用方式：
    python scripts/filter_medium_difficulty.py \
        --results_path ./difficulty_analysis/realtime_results.jsonl \
        --original_data_path ./long_video_data/longvt_selfqa/train.parquet \
        --output_dir ./long_video_data_filter/longvt_selfqa \
        --difficulty medium
"""

import os
import json
import hashlib
import argparse
import logging
from pathlib import Path
from typing import Set

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_unique_sample_id(video_id: str, question: str) -> str:
    """
    生成唯一的样本 ID，使用确定性的 hash 方法

    注意：Python 内置的 hash() 在不同进程中会产生不同结果，
    所以使用 hashlib.md5 来保证确定性
    """
    video_id = str(video_id) if video_id else ''
    question = str(question)[:100] if question else ''

    if video_id and question:
        # 使用 md5 生成确定性 hash
        hash_input = f"{video_id}_{question}".encode('utf-8')
        hash_value = int(hashlib.md5(hash_input).hexdigest(), 16) % 10000000
        return f"{video_id}_{hash_value}"
    return ""


def load_filtered_samples(results_path: str, difficulty: str) -> dict:
    """从 realtime_results.jsonl 加载指定难度的样本信息"""
    samples = {}  # video_id -> set of questions

    with open(results_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if data.get('difficulty_level') == difficulty:
                    video_id = data.get('video_id', '')
                    question = data.get('question', '')[:100]  # 截取前100字符
                    if video_id:
                        if video_id not in samples:
                            samples[video_id] = set()
                        samples[video_id].add(question)
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to parse line: {e}")
                continue

    total = sum(len(qs) for qs in samples.values())
    logger.info(f"Loaded {total} {difficulty} difficulty samples from {len(samples)} videos")
    return samples


def filter_and_save(
    original_data_path: str,
    results_path: str,
    output_dir: str,
    difficulty: str = "medium",
    val_ratio: float = 0.05,
):
    """筛选指定难度的样本并保存"""

    # 加载筛选后的样本信息 (video_id -> set of questions)
    filtered_samples = load_filtered_samples(results_path, difficulty)

    if not filtered_samples:
        logger.error(f"No {difficulty} difficulty samples found!")
        return

    # 加载原始数据
    logger.info(f"Loading original data from {original_data_path}")
    df = pd.read_parquet(original_data_path)
    logger.info(f"Original data has {len(df)} samples")

    # 通过 video_id + question 前缀匹配
    filtered_indices = []
    for idx, row in df.iterrows():
        video_id = str(row.get('video_id', ''))
        question = str(row.get('question', ''))[:100]

        if video_id in filtered_samples:
            if question in filtered_samples[video_id]:
                filtered_indices.append(idx)

    logger.info(f"Matched {len(filtered_indices)} samples from original data")

    if not filtered_indices:
        logger.error("No matching samples found! Check if question_id format matches.")
        return

    # 筛选数据
    filtered_df = df.loc[filtered_indices].reset_index(drop=True)

    # 划分训练集和验证集
    n_val = max(1, int(len(filtered_df) * val_ratio))
    n_train = len(filtered_df) - n_val

    train_df = filtered_df.iloc[:n_train]
    val_df = filtered_df.iloc[n_train:]

    logger.info(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 保存
    train_path = os.path.join(output_dir, "train.parquet")
    val_path = os.path.join(output_dir, "val.parquet")

    train_df.to_parquet(train_path)
    val_df.to_parquet(val_path)

    logger.info(f"Saved train data to {train_path}")
    logger.info(f"Saved val data to {val_path}")

    # 打印统计信息
    print("\n" + "=" * 60)
    print(f"Filter Results ({difficulty} difficulty)")
    print("=" * 60)
    print(f"Original samples: {len(df)}")
    print(f"Filtered samples: {len(filtered_df)}")
    print(f"  - Train: {len(train_df)}")
    print(f"  - Val: {len(val_df)}")
    print(f"\nOutput directory: {output_dir}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="筛选指定难度的样本并转换为训练数据格式")

    parser.add_argument(
        "--results_path",
        type=str,
        default="./difficulty_analysis/realtime_results.jsonl",
        help="难度分析结果文件路径"
    )
    parser.add_argument(
        "--original_data_path",
        type=str,
        default="./long_video_data/longvt_selfqa/train.parquet",
        help="原始训练数据路径"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./long_video_data_filter/longvt_selfqa",
        help="输出目录"
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        default="medium",
        choices=["easy", "medium", "hard"],
        help="要筛选的难度级别"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.05,
        help="验证集比例"
    )

    args = parser.parse_args()

    filter_and_save(
        original_data_path=args.original_data_path,
        results_path=args.results_path,
        output_dir=args.output_dir,
        difficulty=args.difficulty,
        val_ratio=args.val_ratio,
    )


if __name__ == "__main__":
    main()
