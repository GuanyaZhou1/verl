export CUDA_VISIBLE_DEVICES=0,1,2,3

vllm serve  /data_gpu/zhengshurong/data/project/Qwen2.5-VL/qwen-vl-finetune/checkpoints/video/Qwen2.5-VL-7B-Instruct-self_holmes_caption_233-self_longvideoreason_caption_930-openo3video_stgr_singleturn_7k-self_holmes_multiturn_1k5-self_longvideoreason_multiturn_5k3-sft-lr5e-5-b24 --host 0.0.0.0  --port 8081            \
                --dtype auto  --api-key 123456   \
                --tensor-parallel-size 4     \
                --async-scheduling \
                --served-model-name Qwen2.5-VL-7B-sft --gpu-memory-utilization 0.6 \
                --enable-chunked-prefill        --enable-prefix-caching      \
                --limit-mm-per-prompt '{"image":16,"video":0}' \
                --mm-processor-kwargs '{"max_pixels":12960000,"min_pixels":4096}' \
                --disable-log-requests --trust-remote-code