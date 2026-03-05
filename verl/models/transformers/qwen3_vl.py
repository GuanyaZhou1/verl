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

import functools
import logging
import os
from dataclasses import dataclass
from typing import Optional

import torch
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLCausalLMOutputWithPast,
    Qwen3VLForConditionalGeneration,
    apply_rotary_pos_emb,
    repeat_kv,
)

from verl.models.transformers.qwen2_vl import _custom_flash_attention_forward, _flash_use_top_left_mask

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def qwen3_vl_vision_forward_profiled(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs):
    """Profiled version of Qwen3VLVisionModel.forward to diagnose per-component timing."""
    import torch.nn.functional as F
    import time

    if not hasattr(qwen3_vl_vision_forward_profiled, "_call_count"):
        qwen3_vl_vision_forward_profiled._call_count = 0
    qwen3_vl_vision_forward_profiled._call_count += 1
    _do_profile = qwen3_vl_vision_forward_profiled._call_count <= 3

    if _do_profile:
        torch.cuda.synchronize()
        t0 = time.perf_counter()

    # Replace Conv3d with equivalent F.linear (bypass PyTorch 2.9 cuDNN disable for Conv3d)
    # When stride == kernel_size, Conv3d is mathematically identical to F.linear
    pe = self.patch_embed
    weight_2d = pe.proj.weight.view(pe.embed_dim, -1)  # (1152, 1536) same memory
    bias = pe.proj.bias

    hidden_states = hidden_states.view(-1, pe.in_channels * pe.temporal_patch_size * pe.patch_size * pe.patch_size)
    hidden_states = F.linear(hidden_states, weight_2d, bias)

    if _do_profile:
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        print(
            f"[PatchEmbed F.linear] call={qwen3_vl_vision_forward_profiled._call_count} | "
            f"time={t1-t0:.3f}s | patches={hidden_states.shape[0]} | "
            f"weight_dtype={weight_2d.dtype} | input_dtype={hidden_states.dtype}",
            flush=True,
        )

    pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
    hidden_states = hidden_states + pos_embeds
    rotary_pos_emb = self.rot_pos_emb(grid_thw)

    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(seq_len, -1)
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())

    cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
        dim=0,
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    if _do_profile:
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        n_windows = len(cu_seqlens) - 1
        window_sizes = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        max_win = max(window_sizes)
        min_win = min(window_sizes)
        print(
            f"[VisionEncoder] call={qwen3_vl_vision_forward_profiled._call_count} | "
            f"patch_embed={t1-t0:.3f}s | pos_embed={t2-t1:.3f}s | "
            f"patches={seq_len} | n_windows={n_windows} | "
            f"window_size: min={min_win} max={max_win} | "
            f"hidden_dtype={hidden_states.dtype} | "
            f"attn_impl={getattr(self.blocks[0].attn.config, '_attn_implementation', 'UNKNOWN')}",
            flush=True,
        )

    deepstack_feature_lists = []
    block_times = []
    for layer_num, blk in enumerate(self.blocks):
        if _do_profile:
            torch.cuda.synchronize()
            tb0 = time.perf_counter()

        hidden_states = blk(
            hidden_states,
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        if layer_num in self.deepstack_visual_indexes:
            deepstack_feature = self.deepstack_merger_list[self.deepstack_visual_indexes.index(layer_num)](
                hidden_states
            )
            deepstack_feature_lists.append(deepstack_feature)

        if _do_profile:
            torch.cuda.synchronize()
            block_times.append(time.perf_counter() - tb0)

    if _do_profile:
        torch.cuda.synchronize()
        t3 = time.perf_counter()

    hidden_states = self.merger(hidden_states)

    if _do_profile:
        torch.cuda.synchronize()
        t4 = time.perf_counter()
        total_blocks = sum(block_times)
        print(
            f"[VisionEncoder] call={qwen3_vl_vision_forward_profiled._call_count} | "
            f"blocks_total={total_blocks:.3f}s | merger={t4-t3:.3f}s | "
            f"grand_total={t4-t0:.3f}s | "
            f"block_times: first={block_times[0]:.3f}s mid={block_times[13]:.3f}s last={block_times[-1]:.3f}s | "
            f"slowest_block={max(block_times):.3f}s (idx={block_times.index(max(block_times))})",
            flush=True,
        )

    return hidden_states, deepstack_feature_lists


def qwen3_vl_attn_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """
    Patched attention forward for Qwen3VL that uses flash_attn_varlen_func + Ulysses SP
    via _custom_flash_attention_forward, avoiding the O(total_nnz^2) 4D attention mask
    created by transformers' create_causal_mask when use_remove_padding=True.
    """
    if not hasattr(qwen3_vl_attn_forward, "_logged"):
        qwen3_vl_attn_forward._logged = True
        pid = position_ids
        if pid is not None and pid.ndim == 3:
            pid = pid[0]
        if pid is not None and pid.ndim == 2:
            flat = pid.view(-1)
            resets = (flat == 0).nonzero().view(-1)
            boundaries = torch.cat([resets, torch.tensor([flat.numel()], device=flat.device)])
            sample_lens = torch.diff(boundaries).tolist()
            print(
                f"[qwen3_vl_attn_forward] ACTIVE | layer_idx={self.layer_idx} | "
                f"total_tokens={flat.numel()} | num_samples={len(sample_lens)} | "
                f"per_sample_lens={sample_lens} | "
                f"attention_mask={'None' if attention_mask is None else tuple(attention_mask.shape)} | "
                f"hidden_dtype={hidden_states.dtype} | "
                f"using varlen flash_attn (per-sample causal attention)",
                flush=True,
            )
        else:
            print(
                f"[qwen3_vl_attn_forward] ACTIVE | layer_idx={self.layer_idx} | "
                f"hidden_states={tuple(hidden_states.shape)} | "
                f"position_ids={'None' if position_ids is None else tuple(position_ids.shape)} | "
                f"attention_mask={'None' if attention_mask is None else tuple(attention_mask.shape)} | "
                f"hidden_dtype={hidden_states.dtype}",
                flush=True,
            )

    bsz, q_len, _ = hidden_states.size()

    # QKV projection with QK-Norm (Qwen3VL specific)
    query_states = self.q_norm(self.q_proj(hidden_states).view(bsz, q_len, -1, self.head_dim)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(bsz, q_len, -1, self.head_dim)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

    # Apply rotary position embeddings
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # GQA expansion
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    dropout_rate = 0.0 if not self.training else self.attention_dropout

    # Record q_len before transpose
    q_len = query_states.shape[2]

    # FA2 uses non-transposed inputs: (bsz, q_len, num_heads, head_dim)
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    # position_ids is already 2D (text_position_ids from model forward)
    if position_ids is not None and position_ids.ndim == 3:
        position_ids = position_ids[0]

    attn_output = _custom_flash_attention_forward(
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length=q_len,
        is_causal=getattr(self, "is_causal", True),
        dropout=dropout_rate,
        sliding_window=None,
        use_top_left_mask=_flash_use_top_left_mask,
        position_ids=position_ids,
    )

    attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, None


def get_rope_index(
    processor,
    input_ids: torch.Tensor,
    image_grid_thw: Optional[torch.Tensor] = None,
    video_grid_thw: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    """
    Gets the position ids for Qwen3-VL, it should be generated before sharding the sequence.
    The batch dim has been removed and the input_ids should be a 1D tensor representing a single example.
    https://github.com/huggingface/transformers/blob/v4.57.0/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py#L916
    """
    spatial_merge_size = processor.image_processor.merge_size
    image_token_id = processor.image_token_id
    video_token_id = processor.video_token_id
    vision_start_token_id = processor.vision_start_token_id

    # Since we use timestamps to separate videos,
    # like <t1> <vision_start> <frame1> <vision_end> <t2> <vision_start> <frame2> <vision_end>,
    # the video_grid_thw should also be split
    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        video_grid_thw[:, 0] = 1

    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        position_ids = torch.ones(3, input_ids.shape[0], dtype=input_ids.dtype, device=input_ids.device)
        image_index, video_index = 0, 0
        attention_mask = attention_mask.to(input_ids.device)
        input_ids = input_ids[attention_mask == 1]
        image_nums, video_nums = 0, 0
        vision_start_indices = torch.argwhere(input_ids == vision_start_token_id)
        vision_tokens = input_ids[vision_start_indices + 1]
        image_nums = (vision_tokens == image_token_id).sum()
        video_nums = (vision_tokens == video_token_id).sum()
        input_tokens = input_ids.tolist()
        llm_pos_ids_list: list = []
        st = 0
        remain_images, remain_videos = image_nums, video_nums
        for _ in range(image_nums + video_nums):
            if image_token_id in input_tokens and remain_images > 0:
                ed_image = input_tokens.index(image_token_id, st)
            else:
                ed_image = len(input_tokens) + 1
            if video_token_id in input_tokens and remain_videos > 0:
                ed_video = input_tokens.index(video_token_id, st)
            else:
                ed_video = len(input_tokens) + 1
            if ed_image < ed_video:
                t, h, w = (
                    image_grid_thw[image_index][0],
                    image_grid_thw[image_index][1],
                    image_grid_thw[image_index][2],
                )
                image_index += 1
                remain_images -= 1
                ed = ed_image
            else:
                t, h, w = (
                    video_grid_thw[video_index][0],
                    video_grid_thw[video_index][1],
                    video_grid_thw[video_index][2],
                )
                video_index += 1
                remain_videos -= 1
                ed = ed_video

            llm_grid_t, llm_grid_h, llm_grid_w = (
                t.item(),
                h.item() // spatial_merge_size,
                w.item() // spatial_merge_size,
            )
            text_len = ed - st

            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            # t_index is always 0 because llm_grid_t is always 1
            # (we use timestamps to encode the temporal information for videos)
            t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
            h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
            w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
            llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
            st = ed + llm_grid_t * llm_grid_h * llm_grid_w

        if st < len(input_tokens):
            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            text_len = len(input_tokens) - st
            llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        position_ids[..., attention_mask == 1] = llm_positions.to(position_ids.device)
    else:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1).to(attention_mask.device)
        else:
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).view(1, -1).expand(3, -1)

    return position_ids


def _get_input_embeds(
    model: "Qwen3VLForConditionalGeneration",
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.Tensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
):
    if not hasattr(_get_input_embeds, "_call_count"):
        _get_input_embeds._call_count = 0
    _get_input_embeds._call_count += 1
    _do_profile = _get_input_embeds._call_count <= 5

    if _do_profile:
        torch.cuda.synchronize()
        import time
        _t_start = time.perf_counter()

    inputs_embeds = model.get_input_embeddings()(input_ids)
    image_mask, video_mask = None, None
    if pixel_values is not None:
        if _do_profile:
            torch.cuda.synchronize()
            _t_vis_start = time.perf_counter()
        # Use inputs_embeds.dtype (bf16 under FSDP MixedPrecision) instead of model.visual.dtype
        # (which returns fp32 because FSDP stores params in fp32 before unsharding)
        pixel_values = pixel_values.to(dtype=inputs_embeds.dtype)
        image_embeds, deepstack_image_embeds = model.visual(pixel_values, grid_thw=image_grid_thw)
        if _do_profile:
            torch.cuda.synchronize()
            print(
                f"[_get_input_embeds] call={_get_input_embeds._call_count} | "
                f"image_visual_tower={time.perf_counter()-_t_vis_start:.3f}s | "
                f"pixel_values={tuple(pixel_values.shape)} | dtype={pixel_values.dtype}",
                flush=True,
            )
        n_image_tokens = (input_ids == model.config.image_token_id).sum().item()
        n_image_features = image_embeds.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )

        mask = input_ids == model.config.image_token_id
        mask_unsqueezed = mask.unsqueeze(-1)
        mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
        image_mask = mask_expanded.to(inputs_embeds.device)

        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    if pixel_values_videos is not None:
        if _do_profile:
            torch.cuda.synchronize()
            import time
            _t_vid_start = time.perf_counter()
        # Use inputs_embeds.dtype (bf16 under FSDP MixedPrecision) instead of model.visual.dtype
        pixel_values_videos = pixel_values_videos.to(dtype=inputs_embeds.dtype)
        video_embeds, deepstack_video_embeds = model.visual(pixel_values_videos, grid_thw=video_grid_thw)
        if _do_profile:
            torch.cuda.synchronize()
            print(
                f"[_get_input_embeds] call={_get_input_embeds._call_count} | "
                f"video_visual_tower={time.perf_counter()-_t_vid_start:.3f}s | "
                f"pixel_values_videos={tuple(pixel_values_videos.shape)} | dtype={pixel_values_videos.dtype}",
                flush=True,
            )
        n_video_tokens = (input_ids == model.config.video_token_id).sum().item()
        n_video_features = video_embeds.shape[0]
        if n_video_tokens != n_video_features:
            if n_video_tokens > n_video_features:
                # More tokens than features: replace excess video tokens with pad token to match
                diff = n_video_tokens - n_video_features
                logger.warning(
                    f"Video tokens ({n_video_tokens}) > features ({n_video_features}), "
                    f"replacing {diff} excess video token(s) with pad token."
                )
                pad_token_id = getattr(model.config, "pad_token_id", 0) or 0
                input_ids = input_ids.clone()  # avoid in-place modification breaking autograd
                video_token_positions = (input_ids == model.config.video_token_id).nonzero(as_tuple=True)
                # Replace the last `diff` video tokens with pad token
                for i in range(1, diff + 1):
                    input_ids[video_token_positions[0][-i], video_token_positions[1][-i]] = pad_token_id
            else:
                # More features than tokens: truncate video_embeds to match token count
                diff = n_video_features - n_video_tokens
                logger.warning(
                    f"Video features ({n_video_features}) > tokens ({n_video_tokens}), "
                    f"truncating {diff} excess video feature(s)."
                )
                video_embeds = video_embeds[:n_video_tokens]

        mask = input_ids == model.config.video_token_id
        mask_unsqueezed = mask.unsqueeze(-1)
        mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
        video_mask = mask_expanded.to(inputs_embeds.device)

        video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

    visual_pos_masks = None
    deepstack_visual_embeds = None
    if image_mask is not None and video_mask is not None:
        # aggregate visual_pos_masks and deepstack_visual_embeds
        image_mask = image_mask[..., 0]
        video_mask = video_mask[..., 0]
        visual_pos_masks = image_mask | video_mask
        deepstack_visual_embeds = []
        image_mask_joint = image_mask[visual_pos_masks]
        video_mask_joint = video_mask[visual_pos_masks]
        for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds, strict=False):
            embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
            embed_joint[image_mask_joint, :] = img_embed
            embed_joint[video_mask_joint, :] = vid_embed
            deepstack_visual_embeds.append(embed_joint)
    elif image_mask is not None:
        image_mask = image_mask[..., 0]
        visual_pos_masks = image_mask
        deepstack_visual_embeds = deepstack_image_embeds
    elif video_mask is not None:
        video_mask = video_mask[..., 0]
        visual_pos_masks = video_mask
        deepstack_visual_embeds = deepstack_video_embeds

    if pixel_values is None and pixel_values_videos is None:
        if _do_profile:
            torch.cuda.synchronize()
            import time
            _t_dummy_start = time.perf_counter()
        config = model.config.vision_config
        patch_dim = config.in_channels * config.temporal_patch_size * config.patch_size**2
        pixel_values = torch.zeros((16, patch_dim), dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        image_grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.long, device=inputs_embeds.device)
        image_embeds, dummy_deepstack_image_embeds = model.visual(pixel_values, grid_thw=image_grid_thw)
        inputs_embeds += 0.0 * image_embeds.mean()
        for emb in dummy_deepstack_image_embeds or []:
            inputs_embeds += 0.0 * emb.mean()
        if _do_profile:
            torch.cuda.synchronize()
            print(
                f"[_get_input_embeds] call={_get_input_embeds._call_count} | "
                f"dummy_visual_tower={time.perf_counter()-_t_dummy_start:.3f}s",
                flush=True,
            )

    if attention_mask is not None:
        attention_mask = attention_mask.to(inputs_embeds.device)

    if _do_profile:
        torch.cuda.synchronize()
        import time
        _t_end = time.perf_counter()
        print(
            f"[_get_input_embeds] call={_get_input_embeds._call_count} | "
            f"total={_t_end-_t_start:.3f}s | "
            f"input_ids={tuple(input_ids.shape)} | embeds_dtype={inputs_embeds.dtype}",
            flush=True,
        )

    return {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "visual_pos_masks": visual_pos_masks,
        "deepstack_visual_embeds": deepstack_visual_embeds,
    }


@dataclass
class Qwen3VLCausalLMOutputForPPO(Qwen3VLCausalLMOutputWithPast):
    log_probs: Optional[torch.FloatTensor] = None
    entropy: Optional[torch.FloatTensor] = None


def qwen3_vl_base_forward(
    self: "Qwen3VLForConditionalGeneration",
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.Tensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    **kwargs,
):
    if not hasattr(qwen3_vl_base_forward, "_call_count"):
        qwen3_vl_base_forward._call_count = 0
    qwen3_vl_base_forward._call_count += 1
    _do_profile = qwen3_vl_base_forward._call_count <= 5

    if _do_profile:
        torch.cuda.synchronize()
        import time
        _t0 = time.perf_counter()

    input_kwargs = _get_input_embeds(
        self, input_ids, attention_mask, pixel_values, pixel_values_videos, image_grid_thw, video_grid_thw
    )  # avoid lora module having multiple keyword arguments

    if _do_profile:
        torch.cuda.synchronize()
        _t1 = time.perf_counter()

    kwargs.update(input_kwargs)
    result = self.language_model(
        input_ids=None,
        **kwargs,
    )

    if _do_profile:
        torch.cuda.synchronize()
        _t2 = time.perf_counter()
        _n_vis = 0
        if pixel_values is not None:
            _n_vis += pixel_values.shape[0]
        if pixel_values_videos is not None:
            _n_vis += pixel_values_videos.shape[0]
        _seq_len = input_kwargs["inputs_embeds"].shape[1] if input_kwargs.get("inputs_embeds") is not None else 0
        _has_ds = input_kwargs.get("deepstack_visual_embeds") is not None
        _ds_layers = len(input_kwargs["deepstack_visual_embeds"]) if _has_ds else 0
        print(
            f"[qwen3_vl_base_forward] call={qwen3_vl_base_forward._call_count} | "
            f"vision_embed={_t1-_t0:.3f}s | text_model={_t2-_t1:.3f}s | total={_t2-_t0:.3f}s | "
            f"seq_len={_seq_len} | vis_patches={_n_vis} | deepstack_layers={_ds_layers} | "
            f"input_ids={'None' if input_ids is None else tuple(input_ids.shape)}",
            flush=True,
        )

    return result


def forward_with_normal_backend(
    self: "Qwen3VLForConditionalGeneration",
    input_ids: torch.LongTensor = None,
    labels: Optional[torch.LongTensor] = None,
    temperature: float = 1.0,
    **kwargs,
) -> "Qwen3VLCausalLMOutputForPPO":
    outputs = self.model(input_ids, **kwargs)
    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states)

    return Qwen3VLCausalLMOutputForPPO(
        logits=logits,
        hidden_states=outputs.hidden_states,
    )


def forward_with_torch_backend(
    self: "Qwen3VLForConditionalGeneration",
    input_ids: torch.LongTensor = None,
    labels: Optional[torch.LongTensor] = None,
    temperature: float = 1.0,
    **kwargs,
) -> "Qwen3VLCausalLMOutputForPPO":
    from verl.utils.experimental.torch_functional import FusedLinearForPPO

    if not hasattr(forward_with_torch_backend, "_call_count"):
        forward_with_torch_backend._call_count = 0
    forward_with_torch_backend._call_count += 1
    _do_profile = forward_with_torch_backend._call_count <= 5

    if _do_profile:
        torch.cuda.synchronize()
        import time
        _t0 = time.perf_counter()
        _mem0 = torch.cuda.memory_allocated() / 1e9

    outputs = self.model(input_ids, **kwargs)
    hidden_states = outputs[0]

    if _do_profile:
        torch.cuda.synchronize()
        _t1 = time.perf_counter()
        _mem1 = torch.cuda.memory_allocated() / 1e9

    # Loss calculations
    if labels is not None:
        rolled_labels = torch.roll(labels, shifts=-1, dims=-1)
    elif input_ids is not None:
        rolled_labels = torch.roll(input_ids, shifts=-1, dims=-1)
    else:
        raise RuntimeError("To use forward_with_torch_backend, either labels or input_ids must be provided.")

    fused_linear_for_ppo = FusedLinearForPPO()
    log_probs, entropy = fused_linear_for_ppo.forward(
        hidden_states=hidden_states,
        vocab_weights=self.lm_head.weight,
        input_ids=rolled_labels,
        temperature=temperature,
    )

    if _do_profile:
        torch.cuda.synchronize()
        _t2 = time.perf_counter()
        _mem2 = torch.cuda.memory_allocated() / 1e9
        print(
            f"[forward_with_torch_backend] call={forward_with_torch_backend._call_count} | "
            f"model_fwd={_t1-_t0:.3f}s | fused_linear={_t2-_t1:.3f}s | total={_t2-_t0:.3f}s | "
            f"hidden={tuple(hidden_states.shape)} | dtype={hidden_states.dtype} | "
            f"mem_gb: before={_mem0:.2f} after_model={_mem1:.2f} after_fused={_mem2:.2f}",
            flush=True,
        )

    return Qwen3VLCausalLMOutputForPPO(
        log_probs=log_probs,
        entropy=entropy,
        hidden_states=outputs.hidden_states,
    )


def forward_with_triton_backend(
    self: "Qwen3VLForConditionalGeneration",
    input_ids: torch.LongTensor = None,
    labels: Optional[torch.LongTensor] = None,
    temperature: float = 1.0,
    **kwargs,
) -> "Qwen3VLCausalLMOutputForPPO":
    from verl.utils.kernel.linear_cross_entropy import linear_cross_entropy

    outputs = self.model(input_ids, **kwargs)
    hidden_states = outputs[0]

    # Loss calculations
    if labels is not None:
        rolled_labels = torch.roll(labels, shifts=-1, dims=-1)
    elif input_ids is not None:
        rolled_labels = torch.roll(input_ids, shifts=-1, dims=-1)
    else:
        raise RuntimeError("To use forward_with_triton_backend, either labels or input_ids must be provided.")

    log_probs, entropy = linear_cross_entropy(
        hidden_states,
        self.lm_head.weight,
        rolled_labels,
        temperature,
        "none",
    )
    return Qwen3VLCausalLMOutputForPPO(
        log_probs=log_probs,
        entropy=entropy,
        hidden_states=outputs.hidden_states,
    )


def patch_qwen3_vl_moe_sparse_moe_block_forward():
    """
    Monkey patch to fix a bug in transformers 4.57.3 where Qwen3VLMoeTextSparseMoeBlock.forward
    incorrectly uses torch.zeros_like(hidden_states) instead of torch.zeros_like(router_logits)
    when creating router_weights (line 148 in modeling_qwen3_vl_moe.py).

    This is a minimal fix that only changes the problematic line while keeping the rest of the
    original implementation intact.
    """
    try:
        from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeTextSparseMoeBlock
    except ImportError:
        # Model not available, skip patching
        return

    # Store the original forward method for reference
    original_forward = Qwen3VLMoeTextSparseMoeBlock.forward

    @functools.wraps(original_forward)
    def patched_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        router_logits = self.gate(hidden_states)
        routing_weights = torch.nn.functional.softmax(router_logits, dim=-1, dtype=torch.float)
        routing_weights, router_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        # BUG FIX: Original code incorrectly uses hidden_states here, should use router_logits
        routing_weights = routing_weights.to(router_logits.dtype)
        router_weights = torch.zeros_like(router_logits).scatter_(1, router_indices, routing_weights)
        hidden_states = hidden_states.reshape(batch_size, -1, self.hidden_size)
        routed_out = self.experts(hidden_states, router_weights, router_indices)
        return routed_out

    # Apply the patch
    Qwen3VLMoeTextSparseMoeBlock.forward = patched_forward
    logger.info("Monkey patched Qwen3VLMoeTextSparseMoeBlock.forward to fix router_weights bug")
