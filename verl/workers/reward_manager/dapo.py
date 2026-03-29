# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from collections import defaultdict
from typing import Any

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


def _decode_token_pieces(tokenizer, token_ids: torch.Tensor) -> list[str]:
    """Decode tokens one by one so debug markup can stay aligned with token spans."""
    cache: dict[int, str] = {}
    pieces: list[str] = []
    for token_id in token_ids.tolist():
        token_id = int(token_id)
        if token_id not in cache:
            cache[token_id] = tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        pieces.append(cache[token_id])
    return pieces


def _compress_special_runs(pieces: list[str]) -> list[str]:
    """Compress long runs of special vision pads to keep debug text readable."""
    compressible = {"<|video_pad|>", "<|image_pad|>"}
    compressed: list[str] = []
    idx = 0
    while idx < len(pieces):
        piece = pieces[idx]
        if piece in compressible:
            end = idx + 1
            while end < len(pieces) and pieces[end] == piece:
                end += 1
            count = end - idx
            compressed.append(f"{piece}x{count}" if count > 1 else piece)
            idx = end
            continue
        compressed.append(piece)
        idx += 1
    return compressed


def _format_observation_meta(observation_debug: list[dict[str, Any]], obs_idx: int) -> str:
    """Format per-observation segment/frame debug info inline with the masked span."""
    if obs_idx >= len(observation_debug):
        return ""

    obs_info = observation_debug[obs_idx] or {}
    lines = [
        (
            f"<obs_meta index={obs_idx} turn={obs_info.get('turn_index', obs_idx)} "
            f"token_len={obs_info.get('obs_token_len', -1)} segments={obs_info.get('segments', [])}>"
        )
    ]
    frame_groups = obs_info.get("frame_groups", []) or []
    for seg_idx, group in enumerate(frame_groups):
        frame_paths = group.get("frame_paths", []) or []
        timestamps = group.get("timestamps", []) or []
        lines.append(
            f"[segment_{seg_idx}] range={group.get('segment', [])} num_frames={len(frame_paths)} timestamps={timestamps}"
        )
        lines.extend(frame_paths)
    lines.append("</obs_meta>")
    return "\n".join(lines) + "\n"


def _build_masked_response_debug(tokenizer, response_ids: torch.Tensor, response_mask: torch.Tensor, observation_debug) -> str:
    """Render the valid response into alternating <mask1>/<mask0> spans."""
    if response_ids.numel() == 0:
        return ""

    pieces = _decode_token_pieces(tokenizer, response_ids)
    mask_values = [int(v) for v in response_mask.tolist()]

    spans: list[tuple[int, list[str]]] = []
    current_mask = mask_values[0]
    current_pieces = [pieces[0]]
    for piece, mask_value in zip(pieces[1:], mask_values[1:], strict=True):
        if mask_value == current_mask:
            current_pieces.append(piece)
            continue
        spans.append((current_mask, _compress_special_runs(current_pieces)))
        current_mask = mask_value
        current_pieces = [piece]
    spans.append((current_mask, _compress_special_runs(current_pieces)))

    observation_debug = observation_debug or []
    obs_idx = 0
    rendered: list[str] = []
    for mask_value, span_pieces in spans:
        if mask_value == 0:
            rendered.append(_format_observation_meta(observation_debug, obs_idx))
            obs_idx += 1
        rendered.append(f"<mask{mask_value}>")
        rendered.append("".join(span_pieces))
        rendered.append(f"</mask{mask_value}>")
    return "".join(rendered)


def _build_debug_full_text(
    tokenizer,
    prompt_ids: torch.Tensor,
    response_ids: torch.Tensor,
    response_mask: torch.Tensor,
    observation_debug,
) -> str:
    """Build a prompt+response debug string with explicit mask boundaries."""
    prompt_pieces = _compress_special_runs(_decode_token_pieces(tokenizer, prompt_ids))
    prompt_text = "".join(prompt_pieces)
    response_text = _build_masked_response_debug(tokenizer, response_ids, response_mask, observation_debug)
    return (
        f"<prompt token_count={prompt_ids.numel()}>{prompt_text}</prompt>"
        f"<response token_count={response_ids.numel()}>{response_text}</response>"
    )


@register("dapo")
class DAPORewardManager(AbstractRewardManager):
    """The reward manager."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        max_resp_len=None,
        overlong_buffer_cfg=None,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, (
                f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"
            )
            assert self.max_resp_len >= self.overlong_buffer_cfg.len, (
                "max_resp_len must be larger than overlong_buffer.len"
            )

    def __call__(self, data: DataProto, return_dict: bool = False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        reward_from_rm_scores = self._extract_reward_from_rm_scores(data, return_dict)
        if reward_from_rm_scores is not None:
            return reward_from_rm_scores

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = int(data_item.batch["attention_mask"][:prompt_length].sum().item())
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = int(data_item.batch["attention_mask"][prompt_length:].sum().item())
            valid_response_ids = response_ids[:valid_response_length]
            valid_response_mask = data_item.batch["response_mask"][:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            # 保留 special tokens 的版本（包含视频 token），用于 Corrected Rollout SFT
            response_str_with_tokens = self.tokenizer.decode(valid_response_ids, skip_special_tokens=False)
            eos_token = self.tokenizer.eos_token
            if response_str.endswith(eos_token):
                response_str = response_str[: -len(eos_token)]
            if eos_token and response_str_with_tokens.endswith(eos_token):
                response_str_with_tokens = response_str_with_tokens[: -len(eos_token)]

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]

            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            extra_info = dict(data_item.non_tensor_batch.get("extra_info", {}))
            observation_debug = data_item.non_tensor_batch.get("observation_debug", None)
            if observation_debug is None:
                observation_debug = extra_info.get("observation_debug", []) or []
            extra_info["observation_debug"] = observation_debug

            # 保存原始输入文本信息到 extra_info，用于 reward_logs 保存
            extra_info["prompt_str"] = prompt_str
            # 保存带 special tokens 的版本（包含 <|video_pad|> 等），用于验证视频 token 注入
            prompt_str_with_tokens = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=False)
            extra_info["prompt_str_with_tokens"] = prompt_str_with_tokens
            # 保存带 special tokens 的 response（用于 Corrected Rollout SFT，保留视频 token）
            extra_info["response_str_with_tokens"] = response_str_with_tokens

            # 保存完整的 prompt+response 带 tokens，以及基于真实 token/mask 的调试串
            full_ids = torch.cat([valid_prompt_ids, valid_response_ids], dim=0)
            full_str_with_tokens = self.tokenizer.decode(full_ids, skip_special_tokens=False)
            extra_info["full_text_with_tokens"] = full_str_with_tokens
            extra_info["response_mask"] = valid_response_mask.cpu().tolist()
            extra_info["debug_full_text"] = _build_debug_full_text(
                self.tokenizer,
                valid_prompt_ids,
                valid_response_ids,
                valid_response_mask,
                observation_debug,
            )

            rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})

            extra_info["rollout_reward_scores"] = rollout_reward_scores

            result = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            score: float
            if isinstance(result, dict):
                score = result["score"]
                # Store the information including original reward
                for key, value in result.items():
                    reward_extra_info[key].append(value)
            else:
                score = result
                reward_extra_info["acc"].append(score)

            reward = score

            if self.overlong_buffer_cfg is not None and self.overlong_buffer_cfg.enable:
                overlong_buffer_len = self.overlong_buffer_cfg.len
                expected_len = self.max_resp_len - overlong_buffer_len
                exceed_len = valid_response_length - expected_len
                overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
                overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
                reward += overlong_reward
                if self.overlong_buffer_cfg.log:
                    reward_extra_info["overlong_reward"].append(overlong_reward)
                    reward_extra_info["overlong"].append(overlong_reward < 0)

            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(result, dict):
                    for key, value in result.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
