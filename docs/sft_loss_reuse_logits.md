# Corrected Rollout SFT Loss: 复用 RL Forward Logits 的原理与实现

## 1. 问题背景

在 Video Reasoning RL 训练中，我们有一个 "Corrected Rollout SFT" 机制：

- 模型生成 rollout response（包含 bbox 坐标等）
- VLM 验证后生成修正版 response（替换不准确的 bbox）
- 对修正版 response 计算 SFT loss，混合进 RL loss

**当前实现**在 `dp_actor.py` 的 `update_policy` 中对 SFT 样本做了**第二次完整 forward pass**（包括 Vision Encoder），导致：

- 训练时间翻倍（每次 forward ~8s）
- 显存翻倍（需保存两份激活值用于 backward）→ OOM crash
- Ray worker 被 OOM killer 杀掉 → NCCL 通信断开 → 训练崩溃

**优化方案**：复用 RL forward pass 的 logits，用一次 gather 操作代替第二次 forward。

---

## 2. 两种 Loss 的定义

### 符号约定

| 符号 | 含义 |
|------|------|
| $x$ | prompt（包含视频 token） |
| $y = (y_1, y_2, \dots, y_T)$ | 模型原始 rollout response |
| $y^* = (y^*_1, y^*_2, \dots, y^*_{T'})$ | 修正后的 SFT response |
| $\pi_\theta(a \mid s)$ | 模型在状态 $s$ 下输出 token $a$ 的概率 |

### 2.1 标准 Teacher Forcing SFT Loss（当前实现，二次 forward）

$$\mathcal{L}_{\text{teacher}} = -\sum_{t=1}^{T'} \log \pi_\theta\!\left(y^*_t \;\middle|\; x,\; y^*_1, \dots, y^*_{t-1}\right)$$

**过程**：

1. 构造新的 `input_ids = [prompt, y*_1, y*_2, ..., y*_T']`
2. 对这个新序列做**完整 forward pass**，得到 $\text{logits}_{\text{sft}}$
3. $\text{logits}_{\text{sft}}[t]$ 代表分布 $P(\;\cdot\; \mid x, y^*_1, \dots, y^*_t)$
4. 在 $\text{logits}_{\text{sft}}[t-1]$ 处 gather $y^*_t$ 的 log probability

**特点**：上下文是修正后的 token。语义是"假设前面都是正确的，下一个 token 应该是什么"。

### 2.2 On-Policy SFT Loss（优化方案，复用 logits）

$$\mathcal{L}_{\text{on\text{-}policy}} = -\sum_{t=1}^{T} \log \pi_\theta\!\left(y^*_t \;\middle|\; x,\; y_1, \dots, y_{t-1}\right)$$

**过程**：

1. RL forward pass 已经用 `input_ids = [prompt, y_1, y_2, ..., y_T]` 做过 forward
2. 得到 $\text{logits}_{\text{rl}}$，其中 $\text{logits}_{\text{rl}}[t]$ 代表分布 $P(\;\cdot\; \mid x, y_1, \dots, y_t)$
3. **复用同一份** $\text{logits}_{\text{rl}}$，但在 $\text{logits}_{\text{rl}}[t-1]$ 处 gather $y^*_t$ 的 log probability

**特点**：上下文是模型自己生成的原始 token。语义是"在你实际生成到这里时，你应该输出什么"。

---

## 3. 具体计算过程

### 3.1 RL Forward Pass（已有，不额外开销）

```python
input_ids = [prompt_tokens, y_1, y_2, ..., y_T]
#                     ↓ model forward ↓
logits = model(input_ids)   # shape: (1, prompt_len + T, vocab_size)

# 取 response 部分的 logits
response_logits = logits[:, prompt_len-1 : prompt_len+T-1, :]  # (1, T, vocab_size)
```

$\text{response\_logits}[t]$ 的含义：给定 $(x, y_1, \dots, y_t)$ 作为上下文，模型对下一个 token 的概率分布。

### 3.2 RL Log Probs（标准 RL loss 需要的）

对原始 response token 做 gather：

$$\text{rl\_log\_probs}[t] = \log\text{softmax}\!\big(\text{response\_logits}[t]\big)\big[y_{t+1}\big] = \log \pi_\theta(y_{t+1} \mid x, y_1, \dots, y_t)$$

### 3.3 SFT Log Probs（复用同一份 logits，仅改 gather 的 label）

对修正 response token 做 gather（**同一份 logits**！）：

$$\text{sft\_log\_probs}[t] = \log\text{softmax}\!\big(\text{response\_logits}[t]\big)\big[y^*_{t+1}\big] = \log \pi_\theta(y^*_{t+1} \mid x, y_1, \dots, y_t)$$

> **关键区别**：不是拿两个 logits 做对比，而是同一份 logits 上用**不同的 label** 做 gather。`log_softmax` 只算一次。

### 3.4 SFT Loss 计算

$$\mathcal{L}_{\text{sft}} = -\frac{\sum_t \text{sft\_log\_probs}[t] \cdot \text{sft\_response\_mask}[t]}{\sum_t \text{sft\_response\_mask}[t]}$$

### 3.5 代码实现

```python
# 已有的 RL forward
outputs = self._forward_micro_batch(model_inputs, temperature=temperature)
log_prob = outputs["log_probs"]           # log π(y_{t+1} | x, y_{≤t})

# SFT log_probs: 在 _forward_micro_batch 内部复用 logits
# logprobs_from_logits 本质就是 logits.log_softmax(-1).gather(-1, labels)
sft_log_probs = outputs["sft_log_probs"]  # log π(y*_{t+1} | x, y_{≤t})

# SFT loss
sft_loss = masked_mean(-sft_log_probs, sft_response_mask)

# 混合 loss
total_loss = rl_loss + sft_weight * sft_loss
```

---

## 4. 关键问题：长度不一致和 token 不对齐怎么办？

### 4.1 首先澄清：这不是"两个 logits 做比较"

容易误解的点：以为需要把 $\text{logits}_{\text{origin}}$ 和 $\text{logits}_{\text{refine}}$ 做某种比较。

实际上：

- 我们只有**一份 logits**（来自 RL forward pass）
- 这份 logits 在每个位置 $t$ 给出了一个**完整的概率分布** $P(\;\cdot\; \mid x, y_{\le t})$
- RL loss：从这个分布中 gather $y_{t+1}$（原始 token）的概率
- SFT loss：从这个分布中 gather $y^*_{t+1}$（修正 token）的概率
- **两次 gather，同一份 logits**

所以问题不是"两个 logits 怎么对齐"，而是"修正 token 序列和 logits 的位置怎么对齐"。

### 4.2 当前实现的对齐方式

在 `_prepare_sft_labels`（`ray_trainer.py:1592-1684`）中：

```python
response_length = batch.batch['responses'].shape[1]        # 原始 response 的 padded 长度
sft_responses = torch.full_like(batch.batch['responses'], pad_id)  # 同样 shape
sft_response_mask = torch.zeros(bs, response_length, dtype=torch.bool)

# 修正 token 左对齐填入，多余部分 pad
sft_responses[idx, :corrected_len] = corrected_ids[:response_length]  # 截断到 response_length
sft_response_mask[idx, :corrected_len] = True                         # 标记有效位置
```

**两者 tensor shape 始终一致**（都是 `response_length`），但"语义位置"可能不对齐。

### 4.3 图解三种对齐情况

#### 情况 A：Token 完全对齐（仅替换数值 token）——最常见

bbox 坐标替换，结构和长度都不变。

```
位置:            0       1      2      3      4      5      6      7      8
原始 token:     <bbox>  0.12    ,     0.34    ,     0.56    ,     0.78  </bbox>
修正 token:     <bbox>  0.15    ,     0.38    ,     0.52    ,     0.81  </bbox>
是否相同:         ✓       ✗     ✓       ✗     ✓       ✗     ✓       ✗      ✓
logits 上下文:   完全     完全   差1个   差1个  差2个   差2个  差3个   差3个   差4个
               一致     一致   token  token  token  token  token  token  token
```

- 位置 0-1：logits 上下文完全一致 → loss 等价于 Teacher Forcing
- 位置 2+：上下文中有 1~4 个 token 不同，但它们是数值，对结构 token（`,`）的分布影响极小

**结论：几乎无损。**

#### 情况 B：长度不同（修正后更长或更短）

```
原始 response (len=8):   think ... bbox: 0.1  , 0.2  ... answer: yes <pad> <pad>
修正 response (len=10):  think ... bbox: 0.15 , 0.23 ... answer: yes <eos> <pad>
sft_response_mask:         1     1    1    1    1    1   1    1    1    1     0

logits 覆盖位置:          [0]  [1]  [2]  [3]  [4]  [5]  [6]  [7]  ×    ×
```

问题：

- logits 只有 8 个位置（由原始 response 长度决定）
- 修正 response 有 10 个 token
- 位置 8、9 没有对应的 logits

**当前处理方式**：`_prepare_sft_labels` 中 `corrected_ids[:response_length]` 截断到原始长度，`sft_response_mask` 只标记有效位置。超出部分被截断，不参与 loss 计算。

**结论**：有信息损失（最后几个 token 被截断），但核心的 bbox 修正信号保留。

#### 情况 C：完全不同的 response（极端情况）

```
原始: The answer is A because the cat is sitting on the mat.
修正: The answer is B because the dog is standing on the floor.
```

- 从第 4 个 token ("A" vs "B") 开始分叉
- 之后所有位置的 logits 上下文都是基于原始 response
- loss 信号有较大噪声

**结论**：不适用此优化。但在 Corrected Rollout SFT 场景中不会出现这种情况——修正仅针对 bbox 坐标。

### 4.4 理论误差界

设 $D = \{t : y_t \neq y^*_t\}$ 为差异位置集合，$|D| = k$。

对于位置 $t$，两种 loss 的差异为：

$$\delta_t = \left|\log P(y^*_t \mid y^*_{<t}) - \log P(y^*_t \mid y_{<t})\right|$$

- 当 $t < \min(D)$ 时，$y_{<t} = y^*_{<t}$，所以 $\delta_t = 0$
- 当 $t \geq \min(D)$ 时，上下文中有不超过 $k$ 个 token 不同

对于 Transformer 模型，单个 token 的变化对后续位置的影响随上下文长度分摊（attention weight 分散在所有 token 上）。当上下文长度为 $L$，$k$ 个 token 不同时：

$$\delta_t \approx O\!\left(\frac{k}{L}\right)$$

对于 bbox 修正场景的典型值：$k \approx 4\text{-}8$（几个坐标数字），$L \approx 2000\text{-}5000$。

所以 $\delta_t \approx O(0.001)$，**可忽略**。

---

## 5. 理论依据：DAgger (On-Policy Imitation Learning)

### 5.1 标准 Teacher Forcing 的问题

Teacher Forcing 训练时上下文来自"专家"（修正序列），推理时上下文来自模型自身。这种训练/推理分布不匹配叫做 **exposure bias**：

$$\text{训练}:\; P(y^*_t \mid x, y^*_1, \dots, y^*_{t-1}) \quad\leftarrow\text{上下文来自专家}$$

$$\text{推理}:\; P(y_t \mid x, y_1, \dots, y_{t-1}) \quad\leftarrow\text{上下文来自模型自己}$$

### 5.2 DAgger 的思路

DAgger (Dataset Aggregation, Ross et al. 2011) 的核心思想：

1. **Learner（模型）** 按自己的策略行动，访问状态 $s_1, s_2, \dots, s_T$
2. **Expert（VLM 验证器）** 在这些状态上给出正确动作 $a^*_t$
3. 训练目标：在**模型实际访问的状态**上模仿专家

$$\mathcal{L}_{\text{DAgger}} = -\mathbb{E}_{s_t \sim \pi_\theta}\!\left[\log \pi_\theta(a^*_t \mid s_t)\right]$$

映射到我们的场景：

| DAgger 概念 | 我们的场景 |
|-------------|-----------|
| Learner 的策略 | 模型的 rollout |
| Learner 访问的状态 $s_t$ | $(x, y_1, \dots, y_t)$ |
| Expert 的动作 $a^*_t$ | $y^*_{t+1}$（修正后的 token） |
| 训练 Loss | $-\log \pi_\theta(y^*_{t+1} \mid x, y_1, \dots, y_t)$ |

**这与复用 RL forward logits 的 loss 完全一致。**

### 5.3 为什么 DAgger 在 RL 混合训练中更合适

|  | Teacher Forcing | On-Policy (DAgger) |
|--|-----------------|-------------------|
| 训练时上下文 | 专家序列 $y^*$ | 模型自己的 rollout $y$ |
| 推理时上下文 | 模型自己生成 | 模型自己生成 |
| 训练-推理一致性 | 不一致 (exposure bias) | **一致** |
| 梯度信号含义 | "在专家轨迹上应该做什么" | **"在你自己的轨迹上应该做什么"** |
| 与 RL loss 的协同 | RL 用 on-policy, SFT 用 off-policy | **两者都 on-policy** |

---

## 6. 性能对比

| 指标 | Teacher Forcing（当前） | On-Policy（优化后） |
|------|----------------------|-------------------|
| Forward pass 次数 | 2x（RL + SFT） | 1x（仅 RL） |
| 额外计算 | 完整 forward (~8s) | 1 次 gather (~0.001s) |
| 额外显存 | 完整激活值 (~30GB) | 无 |
| OOM 风险 | 高（已 crash） | 无 |
| 理论依据 | 标准 SFT | DAgger |

---

## 7. 实现要点

### 7.1 需要修改的文件

1. **`dp_actor.py`**：`update_policy` 中删除第二次 forward，改用 `outputs["sft_log_probs"]`
2. **shell 脚本**：设置 `use_fused_kernels=False`（fused kernels 不返回 logits，无法 gather）

### 7.2 注意事项

- `use_fused_kernels=True` 时 logits 不可用（被融合到 fused linear 中）→ 必须关闭
- `use_fused_kernels=False` 的额外开销是 logits materialization（`vocab_size × seq_len × 4B`），但远小于第二次 forward 的开销
- `sft_response_mask` 确保只在修正 token 的有效位置计算 loss，自动处理长度不一致
