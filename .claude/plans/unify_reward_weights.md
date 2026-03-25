# 统一 Reward 权重配置重构计划

## 问题分析

当前有 **三套权重配置**，互相重叠且容易混淆：

| 配置位置 | 参数 | 生效条件 |
|---------|------|---------|
| `custom_reward_function.reward_kwargs` | `answer_weight`, `format_weight`, `bbox_weight`, `segment_weight` | **grpo** (reward function 内加权) |
| `algorithm.gdpo.reward_weights` | `answer_score`, `format_score`, `bbox_score`, `segment_score` | **gdpo** (过滤 component) |
| `algorithm.token_placement` | `answer_weight`, `format_weight`, `bbox_weight`, `segment_weight` | **gdpo + per_turn** (组合 advantage) |

**问题**：
1. grpo 和 gdpo 使用不同的权重来源
2. gdpo 有两套权重（GDPO_*_WEIGHT 和 TP_*_WEIGHT）
3. 用户难以理解哪个权重在什么时候生效

## 目标

1. **统一权重来源**：所有模式使用同一套权重配置
2. **简化配置**：删除冗余的权重参数
3. **保持 format_score 在 grpo 中生效**

## 设计方案

### 核心思路

让 `algorithm.reward_weights` 成为**唯一的权重配置来源**：

```yaml
algorithm:
  reward_weights:
    answer_score: 1.0
    format_score: 0.2
    bbox_score: 0.0      # 设为 0 表示不使用
    segment_score: 0.0
```

- **grpo 模式**：trainer 层面读取权重，对各 component 加权后作为 advantage
- **gdpo 模式**：trainer 层面读取权重，各 component 独立归一化后加权组合
- **per_turn 模式**：同样使用 `algorithm.reward_weights`

### 废弃的配置

- `custom_reward_function.reward_kwargs.*_weight` - reward function 不再做加权
- `algorithm.gdpo.reward_weights.*` - 合并到 `algorithm.reward_weights`
- `algorithm.token_placement.*_weight` - 合并到 `algorithm.reward_weights`

---

## 最小化修改步骤

### Step 1: 修改 reward function (不加权，只返回分数)

**文件**: `verl/utils/reward_score/video_reasoning_async.py`

修改 `compute_score` 函数：
- 删除 `final_score` 的加权计算逻辑
- 直接返回 `answer_score` 作为 `score`（向后兼容）
- 保留各 component 分数的返回

```python
# 修改前 (第 2107-2110 行)
final_score = (answer_weight * answer_score + bbox_weight * bbox_score +
               format_weight * format_score + segment_weight * segment_score)
total_weight = answer_weight + bbox_weight + format_weight + segment_weight
final_score = final_score / total_weight

# 修改后
final_score = answer_score  # 向后兼容，trainer 会重新计算
```

### Step 2: 添加统一权重配置

**文件**: `verl/trainer/config/algorithm.py`

添加 `reward_weights` 配置：

```python
@dataclass
class AlgorithmConfig:
    # ... existing fields ...

    # 统一的 reward component 权重
    reward_weights: dict = field(default_factory=lambda: {
        "answer_score": 1.0,
        "format_score": 0.0,
        "bbox_score": 0.0,
        "segment_score": 0.0,
    })
```

### Step 3: 修改 grpo 的 advantage 计算

**文件**: `verl/trainer/ppo/ray_trainer.py`

在 `compute_advantage` 函数的 grpo 分支中：

```python
elif adv_estimator == AdvantageEstimator.GRPO:
    # 新增：从 non_tensor_batch 获取各 component 分数
    reward_weights = config.get("reward_weights", {})

    # 计算加权总分
    weighted_scores = torch.zeros_like(data.batch["token_level_rewards"])
    total_weight = 0.0

    for name, weight in reward_weights.items():
        if weight > 0 and name in data.non_tensor_batch:
            component_scores = torch.tensor(
                data.non_tensor_batch[name],
                device=weighted_scores.device
            )
            # 放到 response 末尾位置
            for i, score in enumerate(component_scores):
                valid_len = data.batch["response_mask"][i].sum().int()
                weighted_scores[i, valid_len - 1] += weight * score
            total_weight += weight

    if total_weight > 0:
        weighted_scores = weighted_scores / total_weight

    # 使用加权后的 scores 计算 grpo advantage
    advantages, returns = core_algos.compute_grpo_outcome_advantage(
        token_level_rewards=weighted_scores,
        response_mask=data.batch["response_mask"],
        index=data.non_tensor_batch["uid"],
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
    )
```

### Step 4: 统一 gdpo 的权重来源

**文件**: `verl/trainer/ppo/ray_trainer.py`

修改 gdpo 分支，使用 `config.reward_weights` 替代 `gdpo_config.reward_weights` 和 `token_placement_config.*_weight`：

```python
elif adv_estimator == AdvantageEstimator.GDPO:
    # 使用统一的 reward_weights
    reward_weights = config.get("reward_weights", {})

    # ... 后续代码使用 reward_weights 替代原来的两套权重 ...
```

### Step 5: 更新 shell 脚本

**文件**: `examples/video_reasoning/run_video_reasoning_dapo_h200.sh`

删除冗余配置，使用统一的 `algorithm.reward_weights`：

```bash
# 删除这些
# GDPO_ANSWER_WEIGHT, GDPO_FORMAT_WEIGHT, GDPO_BBOX_WEIGHT, GDPO_SEGMENT_WEIGHT
# TP_ANSWER_WEIGHT, TP_FORMAT_WEIGHT, TP_BBOX_WEIGHT, TP_SEGMENT_WEIGHT
# custom_reward_function.reward_kwargs.answer_weight, etc.

# 新增统一配置
REWARD_WEIGHT_ANSWER=${REWARD_WEIGHT_ANSWER:-1.0}
REWARD_WEIGHT_FORMAT=${REWARD_WEIGHT_FORMAT:-0.2}
REWARD_WEIGHT_BBOX=${REWARD_WEIGHT_BBOX:-0.0}
REWARD_WEIGHT_SEGMENT=${REWARD_WEIGHT_SEGMENT:-0.0}

# 在 python 命令中使用
algorithm.reward_weights.answer_score=$REWARD_WEIGHT_ANSWER \
algorithm.reward_weights.format_score=$REWARD_WEIGHT_FORMAT \
algorithm.reward_weights.bbox_score=$REWARD_WEIGHT_BBOX \
algorithm.reward_weights.segment_score=$REWARD_WEIGHT_SEGMENT \
```

---

## 文件修改清单

| 文件 | 修改内容 |
|-----|---------|
| `verl/utils/reward_score/video_reasoning_async.py` | 删除加权逻辑，保留 `*_weight` 参数但标记为废弃 |
| `verl/trainer/config/algorithm.py` | 添加 `reward_weights` 配置 |
| `verl/trainer/ppo/ray_trainer.py` | grpo 使用统一权重；gdpo 使用统一权重 |
| `examples/video_reasoning/run_video_reasoning_dapo_h200.sh` | 使用统一的 `algorithm.reward_weights` |

---

## 向后兼容性

1. **reward function 的 `*_weight` 参数保留但标记废弃**，不会报错
2. **旧的 gdpo/token_placement 权重配置**：如果存在则发出警告，优先使用新的 `reward_weights`
3. **grpo 默认行为**：如果没配置 `reward_weights`，fallback 到原来的 `token_level_scores`

---

## 预期效果

修改后，用户只需要配置一处：

```bash
# 只用 acc + format
algorithm.reward_weights.answer_score=1.0
algorithm.reward_weights.format_score=0.2
algorithm.reward_weights.bbox_score=0
algorithm.reward_weights.segment_score=0

# grpo 或 gdpo 都使用这套权重
ADV_ESTIMATOR=grpo  # 或 gdpo
```
