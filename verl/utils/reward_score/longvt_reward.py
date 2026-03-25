# Copyright 2025 Individual Contributor: Sudong Wang, Zuhao Yang, Kaichen Zhang
# Copyright 2024 Bytedance Ltd. and/or its affiliates (verl integration)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
LongVT Reward Function for verl integration.

This module ports the LongVT reward function to verl, supporting:
- LLM-as-judge for answer accuracy scoring
- Format validation for <think>/<tool_call>/<answer> structure
- IoU-based time reward for video segment localization

Expected output format:
    <think>分析...</think>
    <tool_call>{"name":"crop_video","arguments":{"start_time":10,"end_time":20}}</tool_call>
    <think>继续分析...</think>
    <answer>答案</answer>

Environment variables:
    LLM_AS_A_JUDGE_BASE: API endpoint for LLM judge (e.g., http://localhost:8081/v1)
    LLM_AS_A_JUDGE_KEY: API key for the judge model
"""

import ast
import logging
import os
import random
import re
from typing import Optional

import requests
from openai import OpenAI

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Initialize OpenAI clients from environment variables
_client_list = []
_model_name_list = []
_initialized = False


def _init_clients():
    """Lazy initialization of OpenAI clients."""
    global _client_list, _model_name_list, _initialized
    if _initialized:
        return

    openai_api_key = os.environ.get("LLM_AS_A_JUDGE_KEY", "123456")
    openai_api_base = os.environ.get("LLM_AS_A_JUDGE_BASE", None)

    if openai_api_base:
        try:
            client = OpenAI(
                api_key=openai_api_key,
                base_url=openai_api_base,
            )
            _client_list.append(client)

            # Get model name from API
            response = requests.get(
                f"{openai_api_base}/models",
                headers={"Authorization": f"Bearer {openai_api_key}"},
                timeout=10,
            )
            models = response.json()
            _model_name_list.append(models["data"][0]["id"])
            logger.info(f"Initialized LLM-as-judge with model: {_model_name_list[0]}")
        except Exception as e:
            logger.warning(f"Failed to initialize LLM-as-judge client: {e}")

    _initialized = True


def _get_chat_template():
    """Get the LLM-as-judge prompt template."""
    return """
Below are two answers to a question. Question is [Question], [Standard Answer] is the standard answer to the question,
and [Model_answer] is the answer extracted from a model's output to this question.

Judge how consistent the two answers are.

Scoring rules
• 1    — Fully consistent: they convey the same meaning (e.g., "pink" vs. "it is pink").
• 0.5 — Partially consistent: they overlap on some key points but not all.
• 0    — Inconsistent: they conflict or share no essential overlap.

Output **only** one of the following numbers: 1, 0.5, or 0.
"""


def _get_gpt4_score_ice():
    """Get in-context examples for LLM-as-judge."""
    example_1 = """
[Question]: Is the countertop tan or blue?
[Standard Answer]: The countertop is tan.
[Model_answer] : tan
Judgement: 1
"""

    example_2 = """
[Question]: On which side of the picture is the barrier?
[Standard Answer]: The barrier is on the left side of the picture.
[Model_answer] : left
Judgement: 1
"""

    example_3 = """
[Question]: What happens immediately after the fireworks illuminate the sky?
[Standard Answer]: The crowd cheers loudly and waves flags.
[Model_answer] : The crowd cheers.
Judgement: 0.5
"""

    example_4 = """
[Question]: What items does the waitress hand to the customer?
[Standard Answer]: She hands over a sandwich and a cup of coffee.
[Model_answer] : She hands over a sandwich and a cup of tea.
Judgement: 0.5
"""

    example_5 = """
[Question]: Where is the cat sitting when the dog first walks into the kitchen?
[Standard Answer]: On top of the kitchen counter.
[Model_answer] : In the kitchen, sitting on the floor near the counter.
Judgement: 0.5
"""

    example_6 = """
[Question]: Is the man phone both blue and closed?
[Standard Answer]: Yes, the man phone is both blue and closed.
[Model_answer] : No.
Judgement: 0
"""

    example_7 = """
[Question]: What color is the towel in the center of the picture?
[Standard Answer]: The towel in the center of the picture is blue.
[Model_answer] : The towel in the center of the picture is pink.
Judgement: 0
"""

    return [example_1, example_2, example_3, example_4, example_5, example_6, example_7]


def _get_prompt(predict_str: str, ground_truth: str, question: str) -> str:
    """Build the full LLM-as-judge prompt."""
    examples = _get_gpt4_score_ice()
    chat_template = _get_chat_template()
    demo_prompt = chat_template
    for example in examples:
        demo_prompt += example + "\n\n"
    test_prompt = f"""
[Question]: {question}
[Standard Answer]: {ground_truth}
[Model_answer] : {predict_str}
Judgement:"""
    return f"{demo_prompt}{test_prompt}"


def extract_answer(text: str) -> Optional[str]:
    """Extract answer from <answer>...</answer> tags."""
    pattern = r"<answer>(.*?)</answer>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def _check_format(predict_str: str) -> bool:
    """
    Check if the prediction follows the expected format.

    Expected formats:
    1. Without tool: <think>...</think><answer>...</answer>
    2. With tool: <think>...</think><tool_call>...</tool_call><think>...</think>...<answer>...</answer>
    """
    # Basic tag matching check
    count_think_1 = predict_str.count("<think>")
    count_think_2 = predict_str.count("</think>")
    count_tool_call_1 = predict_str.count("<tool_call>")
    count_tool_call_2 = predict_str.count("</tool_call>")
    count_answer_1 = predict_str.count("<answer>")
    count_answer_2 = predict_str.count("</answer>")

    # Check tag matching
    if count_think_1 != count_think_2 or count_think_1 == 0:
        return False
    if count_tool_call_1 != count_tool_call_2:
        return False
    if count_answer_1 != count_answer_2 or count_answer_1 != 1:
        return False

    # Strict format check
    if count_tool_call_1 == 0:
        # No tool case: <think>...</think><answer>...</answer>
        pattern = r"^\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$"
        if not re.match(pattern, predict_str, re.DOTALL):
            return False
    else:
        # With tool case: must follow alternating pattern
        stripped_str = predict_str.strip()

        # Check starts with <think> and ends with </answer>
        if not (stripped_str.startswith("<think>") and stripped_str.endswith("</answer>")):
            return False

        # Analyze tag sequence
        tag_pattern = r"<(think|tool_call|answer)>"
        tags = re.findall(tag_pattern, stripped_str)

        # Expected pattern: think, (tool_call, think)*, answer
        expected_pattern = ["think"]
        for _ in range(count_tool_call_1):
            expected_pattern.extend(["tool_call", "think"])
        expected_pattern.append("answer")

        if tags != expected_pattern:
            return False

    return True


def _extract_last_crop_video(predict_str: str) -> Optional[tuple[float, float]]:
    """Extract the last crop_video tool call's start_time and end_time."""
    tool_call_pattern = r"<tool_call>(.*?)</tool_call>"
    tool_calls = re.findall(tool_call_pattern, predict_str, re.DOTALL)

    for tool_call in reversed(tool_calls):
        try:
            tool_data = ast.literal_eval(tool_call.strip())
            if isinstance(tool_data, dict) and tool_data.get("name") == "crop_video":
                arguments = tool_data.get("arguments", {})
                if "start_time" in arguments and "end_time" in arguments:
                    return (float(arguments["start_time"]), float(arguments["end_time"]))
        except (ValueError, SyntaxError, KeyError):
            continue

    return None


def _compute_iou(pred_start: float, pred_end: float, gt_start: float, gt_end: float) -> float:
    """Compute IoU between predicted and ground truth time intervals."""
    intersection_start = max(pred_start, gt_start)
    intersection_end = min(pred_end, gt_end)
    intersection = max(0, intersection_end - intersection_start)

    union_start = min(pred_start, gt_start)
    union_end = max(pred_end, gt_end)
    union = union_end - union_start

    if union > 0:
        return intersection / union
    return 1.0 if intersection > 0 else 0.0


def _compute_recall(pred_start: float, pred_end: float, gt_start: float, gt_end: float) -> float:
    """Compute recall (intersection / ground_truth_length)."""
    intersection_start = max(pred_start, gt_start)
    intersection_end = min(pred_end, gt_end)
    intersection = max(0, intersection_end - intersection_start)

    gt_length = gt_end - gt_start
    if gt_length > 0:
        return intersection / gt_length
    return 1.0 if intersection > 0 else 0.0


def _get_acc_reward(answer_text: str, ground_truth: str, question_text: str) -> float:
    """Get accuracy reward using LLM-as-judge or fallback."""
    _init_clients()

    if not _client_list:
        # Fallback: simple string matching
        answer_lower = answer_text.lower().strip()
        gt_lower = ground_truth.lower().strip()
        if answer_lower == gt_lower:
            return 1.0
        if answer_lower in gt_lower or gt_lower in answer_lower:
            return 0.5
        return 0.0

    # Use LLM-as-judge
    full_prompt = _get_prompt(answer_text, ground_truth, question_text)

    client_idx = random.randint(0, len(_client_list) - 1)
    client = _client_list[client_idx]
    model_name = _model_name_list[client_idx]

    try:
        chat_response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": full_prompt},
            ],
            seed=random.randint(0, 1000000),
            temperature=0.3,
        )
        response = chat_response.choices[0].message.content.strip()

        if "Judgement:" in response:
            response = response.split("Judgement:")[-1].strip()

        if "1" in response and "0" not in response:
            return 1.0
        elif "0.5" in response:
            return 0.5
        elif "0" in response:
            return 0.0
        else:
            logger.warning(f"LLM-as-judge format error: {response=}")
            return 0.0
    except Exception as e:
        logger.warning(f"LLM-as-judge API error: {e}")
        return 0.0


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict = None,
    **kwargs,
) -> dict:
    """
    Compute the reward score for LongVT-style video reasoning output.

    This function validates the format, computes answer accuracy using LLM-as-judge,
    and optionally computes time-based rewards (IoU or recall).

    Args:
        data_source: Dataset identifier (e.g., "longvt", "vstar", "vl_agent")
        solution_str: Model's complete output string
        ground_truth: Correct answer string
        extra_info: Dictionary containing:
            - question: The question text (required for LLM-as-judge)
            - video_segment: [start_time, end_time] for time reward (optional)
        **kwargs:
            - use_iou_reward: If True, compute IoU-based time reward
            - use_time_reward: If True, compute recall-based time reward
            - tool_use_reward: If True, give reward for using tool correctly
            - use_new_reward: If True, use 1:1 acc:format ratio

    Returns:
        dict with keys:
            - score: Total score (weighted sum of components)
            - acc_score: Answer accuracy score (0, 0.5, or 1.0)
            - format_reward_score: Format validation score (0.0 or 1.0)
            - time_reward_score: Time IoU/recall score (0.0 ~ 1.0), if applicable
    """
    extra_info = extra_info or {}

    # Check format
    is_format_error = not _check_format(solution_str)

    # Extract answer
    answer_text = extract_answer(solution_str) or ""

    # Compute accuracy reward
    if not answer_text:
        acc_reward = 0.0
    elif len(answer_text) >= 1000:
        # Penalize overly long answers (possible reward hacking)
        acc_reward = 0.0
        is_format_error = True
    else:
        question_text = extra_info.get("question", "")
        acc_reward = _get_acc_reward(answer_text, ground_truth, question_text)

    format_reward = 0.0 if is_format_error else 1.0

    # Build result dict
    result = {
        "acc_score": acc_reward,
        "format_reward_score": format_reward,
    }

    # Handle different reward modes
    tool_use_reward = kwargs.get("tool_use_reward", False)
    use_time_reward = kwargs.get("use_time_reward", False)
    use_iou_reward = kwargs.get("use_iou_reward", False)
    use_new_reward = kwargs.get("use_new_reward", False)

    if tool_use_reward:
        count_tool_response = solution_str.count("<tool_response>")
        tool_reward = 1.0 if count_tool_response > 0 and acc_reward >= 0.5 else 0.0
        result["tool_reward_score"] = tool_reward
        result["score"] = 1.0 * acc_reward + 1.0 * format_reward + 1.0 * tool_reward

    elif use_time_reward or use_iou_reward:
        time_reward = 0.0
        count_tool_response = solution_str.count("<tool_response>")

        if count_tool_response > 0:
            ground_truth_time = extra_info.get("video_segment")
            last_crop_video = _extract_last_crop_video(solution_str)

            if last_crop_video and isinstance(ground_truth_time, list) and len(ground_truth_time) == 2:
                pred_start, pred_end = last_crop_video
                gt_start, gt_end = float(ground_truth_time[0]), float(ground_truth_time[1])

                if use_iou_reward:
                    time_reward = _compute_iou(pred_start, pred_end, gt_start, gt_end)
                else:  # use_time_reward (recall)
                    time_reward = _compute_recall(pred_start, pred_end, gt_start, gt_end)

        result["time_reward_score"] = time_reward
        result["score"] = 1.0 * acc_reward + 1.0 * format_reward + 1.0 * time_reward

    elif use_new_reward:
        result["score"] = 1.0 * acc_reward + 1.0 * format_reward
        # Add dummy time_reward_score for tensor compatibility
        result["time_reward_score"] = -1.0

    else:
        result["score"] = 0.8 * acc_reward + 0.2 * format_reward
        # Add dummy time_reward_score for tensor compatibility
        result["time_reward_score"] = -1.0

    # Add debug info
    result["predict_str"] = solution_str
    result["ground_truth"] = ground_truth

    return result
