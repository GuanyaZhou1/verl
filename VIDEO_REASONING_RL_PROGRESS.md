# Video Reasoning RL Training - 任务进度文档

**最后更新**：2026-01-29
**当前状态**：✅ 数据格式已验证，待运行训练测试

---

## 项目目标

使用 veRL 框架对视频推理模型进行 GRPO (Group Relative Policy Optimization) 强化学习训练。

---

## 最新更新 (2026-01-29)

### 1. Parquet 格式修正（重要！）

**问题**：之前使用 `json.dumps()` 将 `prompt` 和 `videos` 存为 JSON 字符串，导致训练时报错：
```
TypeError: string indices must be integers, not 'str'
```

**解决**：参考 veRL 官方示例 `geo3k.py`，直接存储 Python 列表（pandas 会自动转为 numpy array）：

```python
# 正确格式（参考 geo3k.py）
processed = {
    "prompt": prompt_messages,  # 直接存列表，不是 json.dumps()
    "videos": videos,           # 直接存列表
    ...
}
```

**验证**：
- `prompt` 类型：`numpy.ndarray`，内部元素为 `dict`，可通过 `prompt[0]['content']` 访问
- `videos` 类型：`numpy.ndarray`，内部元素为 `dict`，可通过 `videos[0]['video']` 访问
- 这种格式与 veRL 的 `_build_messages()` 兼容

### 2. 视频路径

| 项目 | 路径 |
|------|------|
| 视频目录 | `/data_gpu/zhengshurong/data/dataset/LongVideo-Reason_videos/longvila_videos` |

### 3. 数据统计

| 数据集 | 样本数 |
|--------|--------|
| 训练集 | 1215 |
| 验证集 | 63 |
| 唯一视频数 | 430 |

---

## Token 对齐问题（已解决）

### 问题描述

SFT 训练和 RL 训练使用了不同的视觉 token：

| 阶段 | 格式 | 生成的 Token |
|------|------|-------------|
| SFT 训练 | `{"type": "video", ...}` | `<|video_pad|>` |
| RL 训练（旧） | `{"type": "image", ...}` | `<|image_pad|>` |

### 解决方案

**第一轮**：传入原始视频路径 `.mp4`
```python
{"type": "video", "video": "/path/to/video.mp4", "fps": 1, "max_frames": 512}
```

**后续轮**：从 jpg 缓存加载帧路径列表
```python
{"type": "video", "video": ["frame1.jpg", "frame2.jpg", ...]}
```

两种格式都生成 `<|video_pad|>` token，与 SFT 对齐。

---

## 核心技术要点

### 1. 数据格式（重要！）

**SFT 训练使用的是文本标签格式，不是 OpenAI function calling 格式：**

```
模型生成: <think>分析问题...</think>
         <segment>[(10.0, 20.0), (30.0, 40.0)]</segment>

         (系统返回帧)

         <think>继续推理...</think>
         <answer>C</answer>
```

**关键标签**：
- `<think>...</think>` - 推理过程
- `<segment>[(start, end), ...]</segment>` - 请求视频片段
- `<answer>...</answer>` - 最终答案

### 2. 帧缓存机制

训练时动态缓存视频帧（根据训练参数）：
- **格式**：jpg 文件（每帧一个文件）
- **采样**：fps=1, max_frames=512（可配置）
- **位置**：`.cache/{video_name}_fps{fps}_max{max}/` 目录
- **文件命名**：`frame_{idx:04d}_{timestamp:.2f}s.jpg`

---

## 两种训练方案

### 方案一：VideoReasoningAgentLoop（推荐）

- 自定义 AgentLoop，直接解析 `<segment>` 标签
- 与 eval 脚本格式完全一致
- 使用 `role: "user"` 返回 observation

**配置**：`default_agent_loop: video_reasoning`

### 方案二：ToolAgentLoop + FetchFramesTool

- 复用 veRL 的 tool 机制
- 使用 OpenAI function calling 格式
- 使用 `role: "tool"` 返回 observation

**配置**：`default_agent_loop: tool_agent`

**注意**：方案二使用 `role: "tool"`，而 eval 脚本使用 `role: "user"`，可能有细微差异。推荐优先使用方案一。

---

## 运行命令

### Step 1: 数据预处理

```bash
cd /data_gpu/songlin/rl/verl
bash examples/video_reasoning/preprocess_video_reasoning_data.sh
```

**输出**：
- `long_video_data/train.parquet` (95%)
- `long_video_data/val.parquet` (5%)

### Step 2: 运行训练

**方案一（推荐）**：
```bash
bash examples/video_reasoning/run_video_reasoning_grpo.sh
```

**方案二**：
```bash
bash examples/video_reasoning/run_video_reasoning_toolcall.sh
```

---

## 关键文件路径

### 数据文件

| 文件 | 路径 |
|------|------|
| 原始数据 | `./long_video_data/results.json` |
| 训练数据 | `./long_video_data/train.parquet` |
| 验证数据 | `./long_video_data/val.parquet` |
| 视频文件 | `/data_gpu/zhengshurong/data/dataset/LongVideo-Reason_videos/longvila_videos/{video_id}.mp4` |
| 帧缓存 | `./.cache/` |

### 代码文件

| 文件 | 路径 |
|------|------|
| 数据预处理 | `examples/data_preprocess/video_reasoning_multiturn.py` |
| 预处理脚本 | `examples/video_reasoning/preprocess_video_reasoning_data.sh` |
| 帧缓存脚本 | `examples/video_reasoning/cache_video_frames.py` |
| 帧缓存类 | `verl/utils/video_frame_cache.py` |
| AgentLoop (方案一) | `verl/experimental/agent_loop/video_reasoning_agent_loop.py` |
| ToolAgentLoop (方案二) | `verl/experimental/agent_loop/tool_agent_loop.py` |
| FetchFramesTool | `verl/tools/fetch_frames_tool.py` |
| Reward | `verl/utils/reward_score/video_reasoning.py` |
| 配置文件 | `examples/video_reasoning/config/video_reasoning_grpo.yaml` |
| 训练脚本 (方案一) | `examples/video_reasoning/run_video_reasoning_grpo.sh` |
| 训练脚本 (方案二) | `examples/video_reasoning/run_video_reasoning_toolcall.sh` |
| Eval 脚本 | `eval_holmes_qwen_multiturn_spatial.py` |

---

## Parquet 完整字段

```python
{
    # 核心字段（直接存列表，不是 JSON 字符串！）
    "prompt": list,             # [{"role": "user", "content": "<video>\n..."}]
    "videos": list,             # [{"video": path, "fps": 1, "max_frames": 512, "max_pixels": 12544}]

    # 视频信息
    "video_path": str,          # 视频文件路径
    "video_id": str,            # 视频ID

    # 问答信息
    "question_id": int,         # 问题ID
    "question": str,            # 原始问题文本
    "options": str,             # JSON: 选项字典
    "correct_answer": str,      # 正确答案 (A/B/C/D)
    "question_type": str,       # 问题类型
    "is_openended": bool,       # 是否为开放式问题
    "source": str,              # 数据来源

    # 参考信息
    "reference_reasoning": str, # 参考推理过程
    "reference_segments": str,  # 参考时间段 (JSON)

    # 训练标识
    "data_source": str,         # "video_reasoning"

    # 扩展字段（JSON 字符串）
    "extra_info": str,          # JSON: {video_path, video_duration, video_id}
    "tools_kwargs": str,        # JSON: 方案二工具参数
}
```

**注意**：`prompt` 和 `videos` 必须是 Python 列表，不能是 JSON 字符串！pandas 保存时会转为 numpy.ndarray，但内部元素仍是 dict，veRL 可以正常处理。

---

## 下一步

1. 运行预处理脚本生成 parquet
2. 运行训练脚本
3. 监控训练日志，验证多轮推理正常工作
4. 评估训练效果

---

## 模型路径

```
/data_gpu/zhengshurong/data/project/Qwen2.5-VL/qwen-vl-finetune/checkpoints/video/Qwen2.5-VL-7B-Instruct-self_holmes_multiturn_1k5-self_longvideoreason_multiturn_2k5-sft-lr5e-5-bs32
```
