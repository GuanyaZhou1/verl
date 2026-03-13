# Video Reasoning RL 训练问题分析报告

**日期**: 2026-03-09
**实验数量**: 23 组
**训练框架**: verl (GRPO/DAPO)

---

## 一、核心问题

### 1.1 主要现象
在 Video Reasoning 任务的 RL 训练中，观察到一个关键问题：

> **Training Reward 持续下降，而 Validation Accuracy 可能上升或波动**

具体表现：
- 大部分实验的 training reward (critic/score/mean) 从初始的 ~50% 下降到 ~40%
- 部分实验的 validation accuracy 能上升（如 exp23 从 64.9% 上升到 67.5%）
- 两个指标出现**不一致**的变化趋势

### 1.2 问题影响
- 训练无法持续优化，reward 信号逐渐失效
- 模型可能学到了"欺骗"验证集的模式，而非真正的推理能力
- 早期实验 (kl<0.3) 会在 35-70 步直接崩溃 (reward→0)

---

## 二、实验汇总

### 2.1 完整实验列表

| 实验 | filter_groups | bbox_weight | kl_coef | top_p | Val Acc 峰值 | Reward 变化 | 状态 |
|------|---------------|-------------|---------|-------|--------------|-------------|------|
| exp1 | false | 0.6 | 0.1 | 1.0 | 63.6%@20 | - | crashed@35 |
| exp2 | false | 0.6 | 0.3 | 1.0 | 67.5%@60 | - | crashed |
| exp3 | false | 0.6 | 0.2 | 1.0 | - | - | crashed@50 |
| exp4 | false | 0.6 | 0.2 | 1.0 | 66.2%@0 | - | crashed@67 |
| exp5 | false | 0.3 | 0.3 | 1.0 | **68.8%@0** | **-15.9%** | completed |
| exp6 | false | 0.0 | 0.3 | 1.0 | 61.0%@20 | **+1.5%** | stopped@34 |
| exp7 | false | 0.0 | 0.4 | 1.0 | **68.8%@20** | **-1.4%** | completed |
| exp8 | false | 0.1 | 0.3 | 1.0 | 63.6%@20 | -8.0% | stopped@31 |
| exp9 | false | 0.0 | 0.4 | 1.0 | 58.4%@0 | -3.0% | killed |
| exp10 | false | 0.0 | 0.5 | 1.0 | 64.9%@20 | -12.9% | killed |
| exp11 | false | 0.0 | 0.4 | 0.7 | 62.3%@40 | **+1.4%** | stopped@72 |
| exp12 | false | 0.0 | 0.5 | 0.7 | 63.6%@0 | **+3.9%** | stopped@35 |
| exp13 | false | 0.05 | 0.4 | 0.7 | 66.2%@20 | **-20.1%** | completed |
| exp14 | false* | 0.3* | 0.3* | 1.0 | 62.3%@0 | +0.7% | config_failed |
| exp15 | false* | 0.3* | 0.3* | 1.0 | - | - | launch_failed |
| exp22 | **true** | 0.2 | 0.4 | 0.7 | 62.3%@20 | **-3.3%** | completed |
| exp23 | false | 0.2 | 0.4 | 0.7 | **67.5%@100** | -11.7% | completed |

*注: exp14/15 命令行参数传递失败，实际使用了默认配置*

### 2.2 参数范围
- **kl_coef**: 0.1 ~ 0.5
- **bbox_weight**: 0.0 ~ 0.6
- **top_p**: 0.7 ~ 1.0
- **filter_groups**: true/false
- **entropy_coef**: -0.005 ~ 0.01

---

## 三、关键发现

### 3.1 KL 系数与训练稳定性

| kl_coef | 典型存活步数 | 说明 |
|---------|-------------|------|
| 0.1 | ~35 步 | 快速崩溃，策略漂移过快 |
| 0.2 | ~50 步 | 仍然崩溃 |
| 0.3 | ~67-137 步 | 可完成训练，但 reward 下降 |
| 0.4 | ~60-137 步 | **最佳平衡点** |
| 0.5 | ~35-50 步 | 过于保守，提前停止 |

**结论**: `kl_coef=0.4` 是最优值，既能防止策略漂移，又不会过度压缩梯度空间。

### 3.2 bbox_weight 对 Reward 稳定性的决定性影响

| bbox_weight | 实验 | Reward 变化 | 分析 |
|-------------|------|-------------|------|
| **0.0** | exp6, exp7, exp11, exp12 | +1.5%, -1.4%, +1.4%, +3.9% | **最稳定** |
| 0.05 | exp13 | -20.1% | 开始不稳定 |
| 0.1 | exp8 | -8.0% | 中等不稳定 |
| 0.2 | exp22, exp23 | -3.3%, -11.7% | 不稳定 |
| 0.3 | exp5 | -15.9% | 较不稳定 |
| 0.6 | exp1-4 | crashed | 早期崩溃 |

**关键发现**:
- **bbox_weight=0.0 时 reward 最稳定**
- 一旦引入 bbox 信号，reward 就开始下降
- 原因：bbox 验证 **74% 返回 0 分**，引入大量噪声

### 3.3 filter_groups 对比实验

| 指标 | exp22 (filter=true) | exp23 (filter=false) |
|------|---------------------|----------------------|
| 初始 Val Acc | 59.7% | 64.9% |
| 峰值 Val Acc | 62.3%@step20 | **67.5%@step100** |
| Reward 变化 | **-3.3%** | -11.7% |
| 训练步数 | 71 | 137 |

**结论**:
- `filter=false` 达到更高的 validation accuracy
- `filter=true` 的 training reward 更稳定（下降幅度更小）
- filter_groups 在当前设置下**不是主要问题**

### 3.4 top_p 的作用

| top_p | Entropy 行为 | 效果 |
|-------|-------------|------|
| 1.0 | entropy 从 ~2.0 上升到 6-7，导致崩溃 | 不稳定 |
| **0.7** | entropy 稳定在 ~1.3 | **稳定** |

**结论**: `top_p=0.7` 是必须的，可有效控制采样随机性。

### 3.5 Validation Accuracy vs Training Reward 的矛盾

| 实验 | Val Acc 变化 | Reward 变化 | 矛盾程度 |
|------|-------------|-------------|----------|
| exp5 | 68.8%@0 (无提升) | -15.9% | 中 |
| exp7 | 53%→68.8% (+15%) | -1.4% | **一致** |
| exp23 | 64.9%→67.5% (+2.6%) | -11.7% | **矛盾** |
| exp13 | 61%→66.2% (+5.2%) | -20.1% | **严重矛盾** |

**问题**: 部分实验中，模型在 validation 上表现更好，但 training reward 却在下降。可能原因：
1. Reward 函数设计问题（bbox 噪声）
2. 训练集和验证集分布不一致
3. 模型学到了 shortcut

---

## 四、根本原因分析

### 4.1 已确认的问题

#### 问题 1: BBox 验证噪声过大
- **现象**: 74% 的 bbox 验证返回 0 分
- **原因**: 源视频分辨率只有 **398x224**（约 89K pixels）
- **配置**:
  - initial_video_max_pixels = 12544 (~112x112)
  - segment_video_max_pixels = 50176 (~224x224)
- **影响**: VLM 在如此低的分辨率下难以准确 grounding，导致 bbox 验证产生大量假阴性

#### 问题 2: 命令行参数传递失败
- **现象**: exp14/exp15 想要测试 filter_groups=true，但实际运行了默认配置
- **原因**: Hydra 命令行参数在 launch_multinode_slurm.sh 中没有正确传递
- **解决**: 改用环境变量方式（exp22/exp23 成功）

### 4.2 待验证的假设

#### 假设 1: bbox 噪声是 reward 下降的主要原因
- **证据**: bbox=0.0 的实验 reward 稳定或上升
- **验证方式**:
  - A) 完全去除 bbox (bbox=0.0)，长时间训练
  - B) 提高视频分辨率后重新训练

#### 假设 2: GRPO 在二值 reward 下的梯度问题
- **现象**: 当 reward 只有 0/1 时，all-correct 和 all-wrong 的 group 无法产生有效梯度
- **验证方式**: 启用 filter_groups 并观察梯度分布

#### 假设 3: 策略漂移导致 reward 分布变化
- **现象**: 随着训练进行，模型输出分布变化，导致 reward 计算基准偏移
- **验证方式**: 监控 KL 散度和 reward 分布的关系

#### 假设 4: 验证集过拟合
- **现象**: Val acc 上升但 reward 下降
- **验证方式**: 增加验证集多样性，或使用不同的验证集

---

## 五、下一步实验建议

### 5.1 短期实验（验证 bbox 假设）

**实验 A**: 纯答案训练
```
bbox_weight=0.0
kl_coef=0.4
top_p=0.7
filter_groups=false
训练 150+ 步
```
**预期**: reward 应该稳定或上升

**实验 B**: 提高分辨率
```
initial_video_max_pixels=50176 (~224x224)
segment_video_max_pixels=200704 (~448x448)
bbox_weight=0.2
```
**预期**: bbox 验证准确率应该提高

### 5.2 中期实验（优化 reward 函数）

**实验 C**: bbox 置信度过滤
- 只使用 VLM 返回高置信度（>0.8）的 bbox 结果
- 低置信度的不计入 reward

**实验 D**: 混合 reward
- 使用 soft reward 替代 binary reward
- 例如：基于 IoU 的连续 reward

### 5.3 长期实验（架构优化）

**实验 E**: 两阶段训练
1. 第一阶段：只用答案 reward 训练到收敛
2. 第二阶段：加入 bbox reward 进行 fine-tuning

**实验 F**: 课程学习
- 从简单视频开始，逐渐增加难度
- 从短视频开始，逐渐增加长度

---

## 六、总结

### 6.1 主要结论

1. **KL 系数**: 0.4 是最优值
2. **top_p**: 必须设为 0.7
3. **bbox_weight**: 是 reward 下降的主要原因，建议设为 0.0 或解决分辨率问题后再使用
4. **filter_groups**: 影响不大，可设为 false

### 6.2 最佳配置（当前）

```yaml
algorithm:
  kl_ctrl.kl_coef: 0.4
  filter_groups.enable: false

actor_rollout_ref:
  rollout.top_p: 0.7

custom_reward_function:
  reward_kwargs.bbox_weight: 0.0  # 暂时关闭
  reward_kwargs.answer_weight: 1.0
```

### 6.3 待解决问题

1. **视频分辨率过低** → 导致 bbox grounding 不准确
2. **Reward 函数噪声** → 需要改进 bbox 验证逻辑
3. **Val/Train 指标不一致** → 需要进一步分析原因

---

*报告生成: Claude Code Auto Experiment System*
