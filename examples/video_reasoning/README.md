# Video Reasoning GRPO Training

## 项目概述
基于 veRL 框架实现视频推理的强化学习训练，使用 GRPO 算法。

## 核心机制
模型输出视频帧请求，框架获取帧后继续推理，直到给出答案。

---

## 两种实现方案

### 方案一：自定义 AgentLoop（直接解析 `<segment>` 标签）
- **AgentLoop**: `video_reasoning_agent_loop.py`
- **脚本**: `run_video_reasoning_grpo.sh`
- **特点**:
  - 直接解析模型输出中的 `<segment>[(start, end)]</segment>` 标签
  - 不使用 OpenAI function calling 格式
  - 从帧缓存加载帧，作为图片添加到 observation
  - 格式与 SFT eval 脚本对齐

### 方案二：复用 Tool Call 机制（使用 `fetch_frames` 工具）
- **工具**: `verl/tools/fetch_frames_tool.py`
- **解析器**: `SegmentToolParser`（解析 `<segment>` 标签）
- **配置**: `config/tool_config.yaml`
- **脚本**: `run_video_reasoning_toolcall.sh`
- **特点**:
  - 复用 veRL 内置的 ToolAgentLoop
  - 使用自定义 `SegmentToolParser` 解析 `<segment>` 标签（与 SFT 格式一致）
  - 将 segment 标签转换为 `fetch_frames` 工具调用

---

## 方案对比

| 特性 | 方案一 | 方案二 |
|------|--------|--------|
| AgentLoop | 自定义 `video_reasoning` | 内置 `tool_agent` |
| 模型输出格式 | `<segment>[(10,20)]</segment>` | `<segment>[(10,20)]</segment>` |
| 解析器 | 内置在 AgentLoop | `SegmentToolParser` |
| 与 SFT 格式一致性 | 高 | 高（使用 segment parser） |
| 框架复用程度 | 低（需要维护自定义代码） | 高（复用内置工具系统） |
| 灵活性 | 高（可自定义解析逻辑） | 中（受限于工具系统） |

---

## 执行流程

### 方案一流程
```
Round 1: [视频 + prompt] → 模型 → <think>...</think><segment>[(10,20)]</segment>
                                      ↓
                            解析 segment，从缓存加载帧
                                      ↓
Round 2: [视频 + prompt + assistant_response + observation] → 模型 → <answer>C</answer>
```

### 方案二流程
```
Round 1: [视频 + prompt] → 模型 → <think>...</think><segment>[(10,20)]</segment>
                                      ↓
                            SegmentToolParser 解析 → FetchFramesTool 加载帧
                                      ↓
Round 2: [视频 + prompt + tool_response] → 模型 → <answer>C</answer>
```

---

## 快速开始

### 1. 数据预处理（两种方案共用）
```bash
cd /data_gpu/songlin/rl/verl

python examples/data_preprocess/video_reasoning_multiturn.py \
    --input_json ./long_video_data/results.json \
    --video_base_path /data_gpu/gyzhou/prj/data_gene/long_video_reason/data/videos \
    --output_dir ./long_video_data \
    --split train
```

### 2a. 运行方案一（自定义 AgentLoop）
```bash
export MODEL_PATH=/path/to/your/sft/model
export DATA_DIR=./long_video_data
export VIDEO_BASE_PATH=/data_gpu/gyzhou/prj/data_gene/long_video_reason/data/videos
export CACHE_DIR=./.cache
export N_GPUS=8

bash examples/video_reasoning/run_video_reasoning_grpo.sh
```

### 2b. 运行方案二（Tool Call 机制）
```bash
export MODEL_PATH=/path/to/your/sft/model
export DATA_DIR=./long_video_data
export VIDEO_BASE_PATH=/data_gpu/gyzhou/prj/data_gene/long_video_reason/data/videos
export CACHE_DIR=./.cache
export N_GPUS=8

bash examples/video_reasoning/run_video_reasoning_toolcall.sh
```

---

## 文件结构
```
examples/video_reasoning/
├── README.md                         # 本文件
├── run_video_reasoning_grpo.sh       # 方案一脚本
├── run_video_reasoning_toolcall.sh   # 方案二脚本
├── cache_video_frames.py             # 帧缓存脚本
└── config/
    ├── video_reasoning_grpo.yaml     # 训练配置
    └── tool_config.yaml              # 方案二工具配置

verl/
├── experimental/agent_loop/
│   └── video_reasoning_agent_loop.py # 方案一 AgentLoop
├── tools/
│   └── fetch_frames_tool.py          # 方案二工具
├── utils/
│   ├── video_frame_cache.py          # 帧缓存类
│   └── reward_score/video_reasoning.py # 奖励函数
```

---

## 可配置参数（环境变量）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_PATH` | - | SFT 模型路径 |
| `DATA_DIR` | `./long_video_data` | 训练数据目录 |
| `VIDEO_BASE_PATH` | - | 视频文件目录 |
| `CACHE_DIR` | `.cache` | 帧缓存目录 |
| `CACHE_FPS` | `1` | 帧采样率 |
| `CACHE_MAX_FRAMES` | `512` | 最大缓存帧数 |
| `CACHE_MAX_FRAMES_PER_SEGMENT` | `16` | 每 segment 最大帧数 |
| `TRAIN_BATCH_SIZE` | `32/64` | 训练批大小 |
| `N_GPUS` | `8` | GPU 数量 |
| `MAX_ASSISTANT_TURNS` | `5` | 最大 assistant 轮数 |
| `MAX_USER_TURNS` | `5` | 最大 user 轮数 |

---

## 进度记录

### 2026-01-28
- [x] 方案一：自定义 AgentLoop 实现（格式与 eval 对齐）
- [x] 方案二：FetchFramesTool + ToolAgentLoop
- [x] 视频帧缓存类
- [x] 奖励函数
- [x] 数据预处理（支持两种方案）
- [ ] **待测试**：完整训练流程

### 环境要求
```bash
# CUDA Driver: 12.8 (nvidia-smi)
# PyTorch: 2.9.0+cu128
pip install torch==2.9.0+cu128 --index-url https://download.pytorch.org/whl/cu128

# vLLM: 从源码编译匹配 cu128
pip install vllm==0.12.0 --no-binary vllm --no-build-isolation
```
