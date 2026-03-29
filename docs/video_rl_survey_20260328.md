# Video RL 全面调研报告 (2025.03 - 2026.03)

> 调研日期: 2026-03-28
> 调研目的: 系统梳理 Video RL 领域所有工作，诊断当前系统问题，指导改进方向

---

## 一、当前系统配置概览 (dapo_0328_wg.sh)

| 维度 | 当前配置 | 备注 |
|------|----------|------|
| Base Model | Qwen3-VL-8B-Instruct (SFT后) | longvt_tvg + openo3video_stgr + selfconstructdata |
| RL 算法 | **GDPO** (DAPO 框架, GDPO advantage estimator) | clip_low=0.2, clip_high=0.28 |
| KL 控制 | KL loss (非 KL reward) | **coef=0.01**, type=low_var_kl |
| Advantage | **GDPO 分组件归一化** + norm_adv_by_std=**True** | 每个 reward 组件独立归一化 |
| Token Placement | **per_turn** | 轮级 reward 分配, bbox/segment 按 turn 分配 |
| 多轮机制 | Agent loop, max 5 turns | `<think>→<segment>→(新帧)→<think>→<answer>` |
| 视频表征 | 粗粒度(1fps/512帧/224px) + 细粒度(1fps/32帧/224px) | 多分辨率, max_pixels=50176 |
| 时间戳水印 | **关闭** | USE_TIMESTAMP_WATERMARK=False |
| Reward 组件 | answer(**VLM打分**) + format(规则) + bbox(规则IoU) + segment(IoU) | VLM答案打分 + 规则时空 |
| Reward 权重 | **answer=0.8, format=0.2, bbox=0.4, segment=0.5** | GDPO 独立归一化后加权 |
| BBox 验证 | **关闭** (USE_BBOX_VERIFICATION=false) | 不再调用 VLM 验证 bbox |
| 训练数据 | long_video_data/ (合并数据集) | 待确认具体大小 |
| Batch | **train=32, gen=64, rollouts=16** | 8 GPUs |
| 学习率 | **2e-6** | |
| Max Response | **8192** | 多轮空间充足 |
| Filter groups | **开启** (metric=acc) | enable_filter_groups=True |
| Overlong Buffer | enable=True, **len=2048, penalty=2.0** | |

---

## 二、核心问题诊断 (基于 dapo_0328_wg.sh)

> 相比旧脚本，新脚本已修复: filter_groups开启, KL降到0.01, format/segment/bbox权重启用, GDPO分组件归一化, per_turn token placement, rollouts增到16, response增到8192, bbox VLM验证关闭

### 仍存在的问题

### 问题 1: VLM 答案打分仍启用 — 最大的噪声源

**当前配置:**
```bash
USE_VLM_SCORING=true                              # 仍用 VLM 打分答案
VLM_MODEL_NAME="Qwen3-VL-235B-A22B-Instruct"     # 调用外部 235B 模型
```

**所有成功的系统都用规则化 reward，不用 VLM 打分:**

| 系统 | 答案 Reward | 时空 Reward | 格式 Reward |
|------|------------|------------|------------|
| Open-o3-Video | exact match / ROUGE | **规则IoU** (时间高斯+空间IoU) | 规则正则 |
| Video-Thinker | exact match (选择题) | 无 | 正则检查 `<time><caption><think>` |
| FrameThinker | exact match + **CCV验证** | 无 | 正则检查 |
| Video-R1 | exact match / ROUGE / WER | **T-GRPO对比**(规则) | 无 |
| VideoAuto-R1 | exact match / tIoU | tIoU(规则) | 正则检查 |
| LongVT | exact match | tIoU(规则) | 正则检查 |
| Video-TwG | exact match | **IoU(规则) + pseudo reward** | 正则检查 |
| **你们 (新脚本)** | **VLM 打分 (0~1连续)** | **规则 IoU (bbox关闭VLM)** | 规则正则 |

**问题:**
1. VLM 打分不稳定: 同一个答案多次调用分数不同，引入高方差噪声
2. GDPO 的核心假设是每个 reward 组件信号清晰可靠，VLM 噪声直接破坏 answer 组件的归一化
3. VLM 调用延迟大: 每个样本需要异步调用 235B 模型
4. answer 权重 0.8 是最大组件，噪声被放大

**Open-o3-Video 的做法 (对比):**
```python
# 直接规则计算，无 VLM 调用
def ans_acc_reward(completions, **kwargs):
    # exact match for MC, ROUGE for open-ended
    return 1.0 if output_ans.strip() == gt_ans.strip() else 0.0
```

### 问题 2: 缺少 Accuracy Gate — bbox/segment reward 可被 hack

**Video-TwG 的关键发现: 如果不做 accuracy gate，grounding reward 单独给会导致模型"乱定位但答错"。**

```python
# Video-TwG: accuracy-gated grounding reward
R = R_acc + R_format + I(R_acc > 0) × R_grounding
#                      ↑ 只有答对才给 grounding 奖励
```

**当前系统 (GDPO):**
GDPO 独立归一化 answer/format/bbox/segment 后加权求和，但 **答案错误时仍然会给 bbox=0.4 和 segment=0.5 的 reward**。模型可以通过输出看起来合理的 bbox/segment 来获得分数，即使答案完全错误。

### 问题 3: 没有自适应 σ 退火 (Temporal Reward)

**Open-o3-Video 的核心创新之一是 σ 退火:**
```python
# step_percent: 训练进度 [0, 1]
if step_percent < 3/4:
    sigma = 4*(1-step_percent)  # 从 4 退到 1
else:
    sigma = 1  # 最后 25% 保持严格
proximity_score = np.exp(-(time_diff ** 2) / (2 * sigma ** 2))
```

当前系统没有时间退火机制，temporal/segment reward 从一开始就要求精确对齐，模型很难在早期获得梯度。

### 问题 4: overlong_buffer.len=2048 对 8192 max_response 仍偏短

```bash
MAX_RESPONSE_LENGTH=8192
reward_model.overlong_buffer.len=2048      # 超过 8192-2048=6144 token 才惩罚
reward_model.overlong_buffer.penalty_factor=2.0
```

多轮 5 轮 × ~1200 token/轮 = 6000 token 接近阈值。建议 buffer_len 增到 4096，penalty 降到 1.0。

### 问题 5: LEARNING_RATE=2e-6 偏高

| 系统 | lr | 备注 |
|------|-----|------|
| Open-o3-Video | **1e-6** | |
| FrameThinker | **5e-7** | 最保守 |
| VideoAuto-R1 | **1e-6** | |
| Video-TwG | **1e-6** | |
| **你们** | **2e-6** | 偏高, 且 GDPO 比 GRPO 更激进 |

### 问题 6: NORM_ADV_BY_STD=true 与 GDPO 可能冲突

GDPO 已经对每个 reward 组件做了独立 batch norm 归一化，再做全局 std 归一化可能导致信号过度压缩。FrameThinker (veRL fork, DAPO) 默认不除以 std。

### 问题 7: 缺少 weight_decay 和 max_grad_norm

所有参考系统都用了这两个:
- Open-o3-Video: weight_decay=0.01, max_grad_norm=5
- Video-Thinker: weight_decay=0.01, max_grad_norm=5
- VideoAuto-R1: weight_decay=0.01, max_grad_norm=1.0

---

## 三、所有 Video RL 工作完整对比

### 3.1 训练框架对比

| 工作 | 训练框架 | RL 算法 | Base Model | LoRA? | Hardware |
|------|----------|---------|------------|-------|----------|
| Video-R1 | open-r1-video (TRL) | GRPO | Qwen2.5-VL-7B | No (full) | 8×A100-80G |
| Open-o3-Video | open-r1-video (TRL) | **GSPO** | Qwen2.5-VL-7B | No (full) | 8×H100 |
| Video-Thinker | TRL GRPOTrainer | GRPO | Qwen2.5-VL-7B | **LoRA r=64** | 4×A100 |
| FrameThinker | **veRL (fork)** | DAPO | Qwen2.5-VL-7B | No (full) | 16×H100 |
| VideoAuto-R1 | veRL | GRPO | Qwen2.5-VL-7B | No (full) | 32×H100 |
| Video-TwG | EasyR1 | GRPO | Qwen2.5-VL-7B | **LoRA r=128** | 2×H800 |
| LongVT | veRL | GRPO | Qwen2.5-VL-7B | No (full) | 未知 |
| EVA | 未公开 | SFT→KTO→GRPO | Qwen2.5-VL-7B | 未知 | 32×H100 |
| SAGE | 自建 | GRPO | Qwen2.5-VL-72B→7B蒸馏 | LoRA | 16×H100 |
| **你们** | **veRL** | **GDPO (DAPO框架)** | **Qwen3-VL-8B** | No (full) | 8×GPU |

### 3.2 Reward 设计完整对比

| 工作 | 答案Reward | 时间Reward | 空间Reward | 格式Reward | 特殊Reward |
|------|-----------|-----------|-----------|-----------|-----------|
| Open-o3-Video | EM/ROUGE/tIoU/vIoU | **高斯退火σ**(point) + IoU(segment) | **时间门控IoU**(τ=1s) | 三级(1/0.5/0) | — |
| Video-Thinker | EM(选择题) | — | — | 正则1/0 | — |
| FrameThinker | EM | — | — | 1/0 | **CCV认知一致性** + action bonus |
| Video-R1 | EM/ROUGE/WER | **T-GRPO对比** | — | — | 长度奖励ω=0.2 |
| VideoAuto-R1 | EM/tIoU | tIoU | — | 双boxed检查 | **双答案非对称**(w₁=0.9,w₂=1.1) + fallback |
| Video-TwG | EM | soft IoU + **hard bonus** | — | 正则 | **accuracy gate** + **pseudo reward** |
| LongVT | EM | IoU | — | 正则 | — |
| EVA | ROUGE/CSV | — | — | 工具补偿0.05 | 自验证(CSV) |
| SAGE | ±1.0/1.25 | — | — | ±0.05/0.10 | **GPT-4o tool judge** + args penalty |
| **你们** | **VLM打分**(0~1) | segment=0.5(规则IoU) | bbox=0.4(规则IoU,VLM关闭) | format=0.2(正则) | **GDPO分组件归一化** + per_turn |

### 3.3 数据集完整对比

| 工作 | SFT数据量 | RL数据量 | 多轮? | 时间标注 | 空间标注 | 来源 |
|------|-----------|---------|-------|---------|---------|------|
| Video-R1 | 165k | 260k | ❌ | ❌ | ❌ | Video-R1-CoT + 混合 |
| Open-o3-Video | 30k | 36k | ❌ | ✅ segment+point | ✅ bbox | STGR 自建 |
| Video-Thinker | — | 10k | ❌ | ✅ `<time>` | ❌ | ActivityNet/YouCook2/STAR |
| FrameThinker | 2.4k | 28k | ✅(多轮帧选) | ✅ frame range | ❌ | Gemini合成 |
| VideoAuto-R1 | — | 83k | ❌ | ✅ tIoU | ❌ | Video-R1/TVBench/STI |
| Video-TwG | — | 51k | ✅(2-3轮) | ✅ `<ground>` | ❌ | NExT-GQA + CG + LLaVA-Video |
| LongVT | 248k | 1.6k+15.4k | ✅(工具调用) | ✅ crop(s,e) | ❌ | VideoSIAH |
| EVA | 10k | 10.7k | ✅(4步循环) | ✅ (隐式) | ❌ | HD-VILA |
| SAGE | 418k | 7.7k | ✅(6-11步) | ❌ | ❌ | YouTube 合成 |
| **你们** | 165k+SFT | **5.8k** | ✅(5轮) | ✅ segment | ✅ bbox | Holmes/LongVR |

### 3.4 Benchmark 效果对比

| 工作 | VideoMME(w/o) | VideoMME(Long) | Video-Holmes | LongVideoBench | MLVU |
|------|-------------|----------------|--------------|----------------|------|
| Qwen2.5-VL-7B | 57.2(HR) | 46.7 | ~30% | 52.4(HR) | 55.3(HR) |
| Video-R1-7B | 57.3 | 49.1 | — | — | — |
| Open-o3-Video | — | — | — | — | — |
| FrameThinker | — | 47.6 | **56.1%** | 52.9 | 59.1 |
| VideoAuto-R1 | — | — | — | — | — |
| Video-TwG(LR) | 53.6 | 47.7 | — | 49.7 | 54.6 |
| Video-TwG(HR) | **59.7** | **50.0** | — | **56.3** | **60.3** |
| LongVILA-R1 | 65.1 | — | — | — | — |
| LongVT | 67.0 | — | — | — | — |
| **你们** | **未提升** | **未提升** | **未提升** | — | — |

---

## 四、改进建议 (基于 dapo_0328_wg.sh, 按优先级)

> 已修复: filter_groups ✅, KL=0.01 ✅, format/segment/bbox权重 ✅, GDPO ✅, per_turn ✅, rollouts=16 ✅, bbox VLM关闭 ✅

### P0 (立即修改): Reward 函数

1. **去掉 VLM 答案打分** (`USE_VLM_SCORING=false`)
   - 选择题: exact match (binary)
   - 开放题: avg(ROUGE-1/2/L)
   - 这是最大噪声源，且 answer 权重 0.8 放大了噪声

2. **加入 accuracy gate** (reward 代码层)
   ```python
   if answer_score < 0.5:
       bbox_score = 0.0
       segment_score = 0.0
   ```

3. **加入自适应 σ 退火** (参考 Open-o3-Video, reward 代码层)
   ```python
   sigma = 4 * (1 - step_percent) if step_percent < 0.75 else 1.0
   temporal_score = exp(-delta_t^2 / (2*sigma^2))
   ```

### P1 (尽快修改): 训练参数

4. **降低学习率**: `LEARNING_RATE=1e-6` (当前 2e-6 偏高, GDPO 比 GRPO 更激进)
5. **NORM_ADV_BY_STD=false**: GDPO 已做归一化, 不需要再除 std
6. **放宽 overlong buffer**: `len=4096, penalty_factor=1.0` (当前 len=2048, penalty=2.0)
7. **加 weight_decay=0.01 和 max_grad_norm=5.0**: 所有参考系统都用

### P2 (数据扩充): 增加训练数据

8. **STGR-RL-36k 完整版** (你们只用了 7k)
9. **NExT-GQA 7.1k** → 转成 `<segment>` 格式做多轮
10. **FrameThinker RL-28k** → 多轮帧选择数据

### P3 (架构改进): 参考最佳实践

11. **CCV 认知一致性验证** (FrameThinker): 防止无效多轮动作
12. **Pseudo reward** (Video-TwG): 对无标注数据也能训 grounding
13. **EMA-GRPO** (OneThinker): 解决多任务 reward 异质性

---

## 五、开源可下载数据集完整列表

| 数据集 | 大小 | 下载地址 | 标注类型 | 多轮 |
|--------|------|----------|----------|------|
| STGR-CoT-30k / STGR-RL-36k | 30k/36k | [HF](https://huggingface.co/datasets/marinero4972/Open-o3-Video) | 时间+空间+CoT | ❌ |
| VideoChat-R1 | 18k | [GitHub](https://github.com/OpenGVLab/VideoChat-R1) | 时间+空间+QA | ❌ |
| NExT-GQA | 10.5k | [HF](https://huggingface.co/datasets/jinyoungkim/NExT-GQA) | 时间+QA | ❌ |
| Temporal-R1 | ~32k | [HF](https://huggingface.co/datasets/appletea2333/temporal_r1) | 时间+CoT | ❌ |
| TimeLens-100K | 100k | [GitHub](https://github.com/TencentARC/TimeLens) | 时间 | ❌ |
| Video-R1-260k | 260k | [HF](https://huggingface.co/datasets/Video-R1/Video-R1-data) | 多种QA | ❌ |
| LLaVA-Video-178K | 178k | [HF](https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K) | 字幕+QA | ❌ |
| CG-Bench | 12.1k | [HF](https://huggingface.co/datasets/CG-Bench/CG-Bench) | 长视频QA | ❌ |
| FrameThinker RL-28k | 28k | [GitHub](https://github.com/lcqysl/FrameThinker-RL) | 多轮帧选 | ✅ |
| LongVT VideoSIAH | 264k | [GitHub](https://github.com/EvolvingLMMs-Lab/LongVT) | 工具调用 | ✅ |
| Video-Thinker-10K | 10k | [GitHub](https://github.com/shijian2001/Video-Thinker) | `<time><caption>` | ❌ |
| LoomData-8.7k | 8.7k | [GitHub](https://github.com/JPShi12/VideoLoom) | 时间+空间caption | ❌ |
| Charades-STA | 12.4k | [GitHub](https://github.com/jiyanggao/TALL) | 时间segment | ❌ |

---

## 六、核心代码对比 (atcode 分析)

### 6.1 Open-o3-Video Reward (参考标杆)

**文件:** `src/r1-v/src/open_r1/reward_func.py`

**时间 Reward (高斯退火):**
```python
def thk_temporal_point_reward(completions, **kwargs):
    step_percent = kwargs['step_percent'][0]  # 训练进度
    sigma = 4*(1-step_percent) if step_percent < 3/4 else 1  # 退火
    for time in pred_times:
        time_diff = min([abs(time - gt_time) for gt_time in gt_times])
        proximity_score = np.exp(-(time_diff ** 2) / (2 * sigma ** 2))
    return total_proximity_score / len(pred_times)
```

**空间 Reward (时间门控 IoU):**
```python
def thk_spatial_reward(completions, **kwargs):
    threshold = 1.0  # 时间门控: 1秒内才计算空间IoU
    for claim in parsed_claims:
        # 1. 时间门控
        for gt_time in gt_times:
            if gt_time - pred_time < threshold:
                ...
        # 2. 直接计算 bbox IoU (无 VLM)
        iou = calculate_iou(gt_box, pred_box)
    return total_iou_score / len(parsed_claims)
```

**格式 Reward (三级):**
```python
def format_reward(completions, **kwargs):
    # 完整格式(think+answer+obj+box+t) → 1.0
    # 只有 think+answer → 0.5
    # 其他 → 0.0
```

### 6.2 FrameThinker Reward (多轮Agent)

**文件:** `verl/workers/reward_manager/dapo.py` + agent loop

**核心特点:**
- 基于 veRL 的 DAPORewardManager
- ParallelEnv 实现多轮工具调用 (gym-like interface)
- CCV (认知一致性验证): 检查 action 是否合理
- Reward = (R_acc + R_action) × CCV
- 工具: `choose_frames(start, end)` + `get_frame_number(timestamp)`

### 6.3 Video-Thinker Reward (最简单有效)

**文件:** `train/grpo.py`

```python
def accuracy_reward(completions, solution, **kwargs):
    # 纯 exact match，只支持选择题
    return 1.0 if output_ans.strip() == gt_ans.strip() else 0.0

def format_reward(completions, **kwargs):
    # <time>...<caption>...<think>... 配对检查 + 唯一 <answer>
    return 1.0 if valid else 0.0
```

**极简但有效**: 只有 accuracy + format，10k 数据就能 Video-Holmes 43.22%。

### 6.4 你们的 Reward (过于复杂)

**文件:** `verl/utils/reward_score/video_reasoning_async.py` (2100+ 行!)

```python
async def compute_score(...):
    # 1. VLM 打分答案 (异步HTTP调用)
    answer_score = await score_answer_with_vlm(...)
    # 2. VLM 验证 bbox (异步HTTP调用 × N个bbox)
    bbox_score = await verify_bboxes_with_vlm(...)
    # 3. 格式检查
    format_score = format_reward(solution_str)
    # 4. Segment IoU (默认 weight=0, 未启用!)
    segment_score = compute_segment_score(all_segments, gt_segments)
    # 5. 加权求和 (无 accuracy gate)
    final_score = (answer_weight * answer_score + bbox_weight * bbox_score + ...)
```

---

## 七、训练参数逐项对比 (代码实证, atcode 分析)

> 以下参数全部来自各仓库的训练脚本和配置文件，通过 atcode 直接提取

### 7.1 GRPO/DAPO 核心超参数

| 参数 | Open-o3-Video | Video-Thinker | FrameThinker | VideoAuto-R1 | Video-TwG(论文) | **你们** | 建议 |
|------|:---:|:---:|:---:|:---:|:---:|:---:|------|
| **框架** | TRL (open-r1) | TRL | veRL (fork) | TRL (自改) | EasyR1 | **veRL** | — |
| **算法** | GSPO | GRPO | GRPO/DAPO | GRPO | GRPO | **GDPO** | GDPO 分组件归一化 |
| **learning_rate** | **1e-6** | **5e-6** | **5e-7** | **1e-6** | **1e-6** | **2e-6** | **偏高**, 建议降到 1e-6 |
| **lr_scheduler** | cosine | cosine | — | constant_warmup | cosine | — | 建议加 cosine |
| **beta (KL coef)** | **0.04** | **0.04** | **0** (关闭KL) | **0.01** | **0.005** | **0.01** (KL loss) | ✅ 合理 |
| **KL 方式** | KL in reward | KL in reward | 无KL | KL in reward | KL in reward | **KL loss** | 两种方式都可 |
| **num_generations (G)** | **4** | **8** | **8** | **16** | **8** | **16** | ✅ 充足 |
| **max_completion_len** | **768** | **1024** | **8192** | **2048** | — | **8192** | ✅ 多轮空间充足 |
| **max_prompt_len** | **16384** | **16384** | **8192** | **8192** | — | **36000** | 视频长 prompt 可理解 |
| **per_device_batch** | 1 | 1 | — | 8 | — | **train=32** | — |
| **num_train_epochs** | **1** | **2** | **10** | **1** | — | **3** | 合理 |
| **weight_decay** | 0.01 | 0.01 | — | 0.01 | — | **无** | 建议加 0.01 |
| **max_grad_norm** | **5** | **5** | — | **1.0** | — | **无** | 建议加 5.0 |
| **save_steps** | 500 | 500 | 50 | 200 | — | **30** | 偏频繁 |

### 7.2 DAPO 特有参数 (Clip-Higher / Filter Groups / Dr.GRPO)

| 参数 | FrameThinker (DAPO) | DAPO论文推荐 | **你们** | 建议 |
|------|:---:|:---:|:---:|------|
| **clip_ratio_low** | veRL默认 | 0.2 | **0.2** | ✅ 合理 |
| **clip_ratio_high** | veRL默认 | 0.28 | **0.28** | ✅ 合理 |
| **norm_adv_by_std** | — | **False** (Dr.GRPO) | **True** | **建议 False**, GDPO 已做归一化 |
| **filter_groups.enable** | **False** | **True** (推荐) | **True** | ✅ 已修复 |
| **filter_groups.metric** | null | acc | **acc** | ✅ 已修复 |
| **filter_groups.max_num_gen_batches** | 0 (无限) | 5 | **5** | ✅ 合理 |
| **overlong_buffer.enable** | **False** | — | **True** | — |
| **overlong_buffer.len** | 0 | — | **2048** | 建议增到 4096 |
| **overlong_buffer.penalty_factor** | 0 | — | **2.0** | 建议降到 1.0 |

### 7.3 多轮 Agent 参数

| 参数 | FrameThinker | **你们** | 建议 |
|------|:---:|:---:|------|
| **max_turns** | **5** | **5** | ✅ 一致 |
| **单轮 max_tokens** | **8192** | **2048** (+rollout.max_tokens_per_turn) | 合理 |
| **agent_workers** | 1 (concurrent) | **4** | 合理 |
| **max_vllm_images** | **128** | — | FrameThinker 限制视觉 token 总量 |
| **tool_name_key** | env_name | — | 你们用 video_reasoning agent loop |
| **param_offload** | **True** | **False** | FrameThinker 启用 offload 节省显存 |
| **optimizer_offload** | **True** | **False** | 同上 |
| **ulysses_sp_size** | **1** | **4** | 你们用序列并行，更适合长视频 |

### 7.4 视频处理参数

| 参数 | Open-o3-Video | Video-Thinker | VideoAuto-R1 | **你们** | 建议 |
|------|:---:|:---:|:---:|:---:|------|
| **max_pixels** | 401408 (28^2x512) | 401408 | video_max=768x28^2 | initial=112^2, seg=224^2 | — |
| **max_frames** | — | — | **256** | initial=**512**, seg=32 | 你们初始帧多 |
| **视频读取** | — | decord | — | 帧缓存 (PIL) | 帧缓存是独特设计 |
| **timestamp 水印** | 无 | 无 | 无 | **有 (rollout时)** | 独特设计 |
| **多分辨率** | 单一 | 单一 | 单一 | **有 (粗+细)** | 类似 TwG 思路 |

### 7.5 Reward 函数参数

| 参数 | Open-o3-Video | Video-Thinker | FrameThinker | VideoAuto-R1 | Video-TwG | **你们** | 建议 |
|------|:---:|:---:|:---:|:---:|:---:|:---:|------|
| **答案reward类型** | EM/ROUGE | EM | EM | EM/tIoU | EM | **VLM打分** | **改成EM!** |
| **答案reward权重** | 单独计 | 1.0 | 1.0 | 0.9+1.1 | 1.0 | **0.8** | 合理 |
| **格式reward** | 三级(1/0.5/0) | 1/0 | 1/0 | 1/0 | 0.2 | **0.2** | ✅ 已启用 |
| **时间reward** | 高斯(sigma退火) | — | — | tIoU | IoU+bonus | **0.5 (segment IoU)** | ✅ 已启用, 建议加sigma退火 |
| **空间reward** | IoU(门控) | — | — | — | — | **0.4 (规则IoU)** | ✅ VLM已关闭 |
| **accuracy gate** | 无 | 无 | **有**(CCV) | 无 | **有** | **无** | **加上!** |
| **sigma退火初值** | **4** | — | — | — | — | **无** | 4 |
| **sigma退火终值** | **1** | — | — | — | — | **无** | 1 |
| **时间门控 tau** | **1.0s** | — | — | — | — | — | 1~3s |
| **reward函数数量** | 5个独立 | 2个 | 2个 | 3个 | 3个 | **1个(合并)** | 拆分更清晰 |

### 7.6 模型微调参数

| 参数 | Open-o3-Video | Video-Thinker | VideoAuto-R1 | **你们** | 建议 |
|------|:---:|:---:|:---:|:---:|------|
| **全参/LoRA** | Full | **LoRA r=64** | Full | Full (SFT: freeze32) | — |
| **tune LLM** | 全部 | LoRA | **True** | 全部 (RL full) | — |
| **tune MLP** | 全部 | LoRA | **True** | 全部 (SFT: freeze; RL: full) | — |
| **tune Vision** | 全部 | LoRA | **False** | 全部 (RL full) | VideoAuto-R1 冻结 vision |
| **并行策略** | ZeRO-3 | ZeRO-3 | FSDP | ZeRO-3 | FSDP | — |
| **gradient_ckpt** | 有 | 有 | — | 有 | 有 | — |
| **attn_impl** | flash_attn_2 | flash_attn_2 | — | — | flash_attn_2 | 确认启用 |

### 7.7 关键参数差异总结

**你们与成功系统最大的参数差异:**

| 差异项 | 影响 | 优先级 |
|--------|------|--------|
| VLM 答案打分 (非规则) | 高噪声, 破坏 GDPO 组件归一化 | **P0** |
| 无 accuracy gate | bbox/segment reward 可被 hack | **P0** |
| 无 sigma 退火 | 早期时间 reward 太严格无梯度 | **P0** |
| LEARNING_RATE=2e-6 | 偏高, GDPO 更激进应更保守 | **P1** |
| NORM_ADV_BY_STD=true | 与 GDPO batch norm 冲突 | **P1** |
| overlong buffer len=2048 | 对 8192 response 偏短 | **P1** |
| overlong penalty=2.0 | 惩罚偏重 | **P1** |
| 无 weight_decay | 应加 0.01 | **P2** |
| 无 max_grad_norm | 应加 5.0 | **P2** |
| 无 lr_scheduler | 应加 cosine | **P2** |
| save_freq=30 | 偏频繁, 浪费 I/O | **P3** |

---

## 八、总结: 关键差异一览

| 维度 | 成功系统的共同点 | 你们的系统 (0328新脚本) | 差距 |
|------|-----------------|-----------|------|
| Reward 来源 | **规则化** (EM/IoU/ROUGE) | 答案仍用 VLM, 时空已改规则 | **答案 VLM 是最后的噪声源** |
| RL 数据量 | 10k~260k | 待确认 (long_video_data/) | 需扩充 |
| Accuracy Gate | 有 (TwG/FrameThinker) | 无 | 模型可 hack bbox/seg |
| sigma 退火 | 有 (Open-o3) | 无 | 早期梯度稀疏 |
| Filter Groups | 有 | **已开启** ✅ | 已修复 |
| Reward 权重 | 统一明确 | **ans=0.8, fmt=0.2, bbox=0.4, seg=0.5** ✅ | 已修复 |
| GDPO 分组件归一化 | Open-o3 用 GSPO | **GDPO + per_turn** ✅ | 已实现, 比多数系统先进 |
| KL coef | 0~0.04 | **0.01** ✅ | 已修复 |
| Overlong惩罚 | 无/宽松 | penalty=2.0, buffer=2048 | 建议放宽 |
| lr | 5e-7 ~ 1e-6 | **2e-6** | 偏高 |
| norm_adv_by_std | 多数 False | **True** | 与 GDPO 可能冲突 |
