export CUDA_VISIBLE_DEVICES=4,5,6,7


vllm serve  /data_gpu/gyzhou/prj/verl/checkpoints/video-reasoning-dapo/video_reasoning_dapo_longvt_selfqa_watermark_genbs32_ep1_lr1e-6_20260210-083828/merged_model --host 0.0.0.0  --port 8082            \
                --dtype auto  --api-key 123456   \
                --tensor-parallel-size 4     \
                --async-scheduling \
                --served-model-name Qwen2.5-VL-7B-RL --gpu-memory-utilization 0.6 \
                --enable-chunked-prefill        --enable-prefix-caching      \
                --limit-mm-per-prompt '{"image":16,"video":0}' \
                --mm-processor-kwargs '{"max_pixels":12960000,"min_pixels":4096}' \
                --disable-log-requests --trust-remote-code