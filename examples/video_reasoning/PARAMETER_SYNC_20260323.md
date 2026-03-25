# Parameter Sync Summary

**Date**: 2026-03-23

**Source**: `/mnt/data/home/zhengshurong/project_ori/verl/examples/video_train/run_qwen3_vl-8b.sh`

**Target**: `/mnt/data/home/zhengshurong/project/verl/examples/video_reasoning/run_video_reasoning_grpo_h200_zsr_singleturn.sh`

## Modified Parameters

| 类别 | 参数 | 旧值 | 新值 |
|------|------|------|------|
| **基础训练** | TRAIN_BATCH_SIZE | 32 | 16 |
| | GEN_BATCH_SIZE | 32 | 16 |
| | MAX_PROMPT_LENGTH | 36000 | 16384 |
| | MAX_RESPONSE_LENGTH | 16384 | 2048 |
| | TOTAL_EPOCHS | 3 | 15 |
| **奖励权重** | FORMAT_WEIGHT | 0.0 | 1.0 |
| **Checkpoint** | SAVE_FREQ | 30 | 20 |
| | TEST_FREQ | 20 | 5 |
| **Data** | +data.image_key | 无 | images |
| | +data.video_fps | 无 | 1 |
| | +data.video_max_frames | 无 | 32 |
| | +data.video_min_frames | 无 | 4 |
| | +data.max_pixels | 无 | 50176 |
| | +data.min_pixels | 无 | 3136 |
| **Actor** | ppo_mini_batch_size | 16 | $TRAIN_BATCH_SIZE |
| **Rollout** | tensor_model_parallel_size | 1 | 2 |
| | gpu_memory_utilization | 0.7 | 0.35 |
| | log_prob_micro_batch_size_per_gpu | 2 | 1 |
| | enable_chunked_prefill | 无 | False |
| | enforce_eager | 无 | False |
| | free_cache_engine | 无 | True |
| | +engine_kwargs.vllm.disable_mm_preprocessor_cache | 无 | True |
| | update_weights_bucket_megabytes | 512 | 4096 |
| **Ref** | log_prob_micro_batch_size_per_gpu | 2 | 1 |
| | fsdp_config.param_offload | False | True |
| **Trainer** | log_val_generations | 无 | 10 |
| **Ray Env** | +VLLM_USE_V1 | 无 | "1" |

## Notes

- `max_pixels=50176` corresponds to ~224x224 resolution per video frame
- `min_pixels=3136` corresponds to ~56x56 resolution
- `tensor_model_parallel_size=2` requires even number of GPUs
- `ref.fsdp_config.param_offload=True` enables CPU offloading for reference model to save GPU memory

## Hydra Config Notes

以下参数需要使用 `+` 前缀（因为不在原始 schema 中）：
- `+data.video_fps`
- `+data.video_max_frames`
- `+data.video_min_frames`
- `+data.max_pixels`
- `+data.min_pixels`
- `+actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache`
- `+ray_kwargs.ray_init.runtime_env.env_vars.VLLM_USE_V1`

## Code Version Differences

`project/verl` 和 `project_ori/verl` 存在代码版本差异：

| 功能 | project/verl | project_ori/verl |
|------|--------------|------------------|
| weights bucket 参数路径 | `rollout.update_weights_bucket_megabytes` | `rollout.checkpoint_engine.update_weights_bucket_megabytes` |

因此实际使用时需根据目标仓库调整参数路径。
