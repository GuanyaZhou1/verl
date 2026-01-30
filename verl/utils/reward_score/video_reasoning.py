#!/usr/bin/env python3
"""
Reward function for video reasoning task.

This module provides a reward function that evaluates the correctness of
the model's answer to video reasoning questions.
"""

import re


def extract_answer(text: str) -> str:
    """
    Extract answer from text in format <answer>...</answer>

    Args:
        text: Model output text

    Returns:
        Extracted answer content (stripped) or empty string
    """
    # Match <answer>...</answer> pattern and extract everything inside
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


def extract_option_letter(answer: str) -> str:
    """
    Extract the option letter from an answer.
    Handles formats like:
    - "B"
    - "B."
    - "B. Some explanation text"
    - "Option B"

    Args:
        answer: Answer text

    Returns:
        Single letter option (A/B/C/D/etc.) or original answer
    """
    answer = answer.strip()

    # Try to find a single letter option (A, B, C, D, etc.)
    # Match pattern: optional "Option" + letter + optional period/text
    match = re.match(r'^(?:Option\s+)?([A-Z])(?:\.|:|$|\s)', answer, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # If no pattern matched, check if the answer is just a single letter
    if len(answer) == 1 and answer.isalpha():
        return answer.upper()

    # Return the original answer if no option letter found
    return answer.strip().upper()


def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    **kwargs,
):
    """
    Compute reward for video reasoning task based on answer correctness.

    This is the main entry point called by veRL's NaiveRewardManager.

    Args:
        data_source: Dataset source identifier (should be 'video_reasoning')
        solution_str: Model's response string
        ground_truth: Correct answer (from reward_model.ground_truth)
        extra_info: Additional information (unused)
        **kwargs: Additional arguments (unused)

    Returns:
        float: Reward score (1.0 for correct, 0.0 for incorrect)
    """
    if not ground_truth:
        return 0.0

    # Extract answer from model response
    predicted_answer = extract_answer(solution_str)

    if not predicted_answer:
        # No answer found, give 0 reward
        return 0.0

    # Normalize both answers
    predicted = extract_option_letter(predicted_answer)
    correct = extract_option_letter(ground_truth)

    # Compare answers
    return 1.0 if predicted == correct else 0.0
