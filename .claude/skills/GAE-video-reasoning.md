# Video Reasoning 实验管理 Skill

## 目标
验证 GAE 的有效性，找到一套能让 reward 持续增加的训练配置。

---

## 1. 核心路径

```
项目根目录: /mnt/data/home/zhengshurong/project/verl
训练脚本:   examples/video_reasoning/run_video_reasoning_dapo_h200.sh
启动脚本:   examples/video_reasoning/launch_multinode_slurm.sh
Checkpoint: ./checkpoints/video-reasoning-dapo/
```

**注意**: 不要使用 `_zsr` 后缀的脚本！

---

## 2. 算法选择 (三选一)

| ADV_ESTIMATOR | 说明 | Critic | 适用场景 |
|---------------|------|--------|----------|
| `grpo` | Group Relative Policy Optimization | 禁用 | 默认，简单 |
| `gdpo` | GDPO with Token Placement | 禁用 | 多轮对话，per-turn 信号 |
| `gae` | Generalized Advantage Estimation | **启用** | 需要验证有效性 |

---

## 3. 启动命令

```bash
cd /mnt/data/home/zhengshurong/project/verl

# 基础启动 (可直接指定节点，无需修改 nodes.txt)
ADV_ESTIMATOR=grpo KL_LOSS_COEF=0.1 \
nohup bash examples/video_reasoning/launch_multinode_slurm.sh \
    --jobid <JOB_ID> --nodes "node15,node19" \
    > /tmp/exp_grpo_kl0.1.log 2>&1 &

# GAE 实验 (启用 critic)
ADV_ESTIMATOR=gae KL_LOSS_COEF=0.3 \
nohup bash examples/video_reasoning/launch_multinode_slurm.sh \
    --jobid <JOB_ID> --nodes "node22,node23" \
    > /tmp/exp_gae_kl0.3.log 2>&1 &
```

---

## 4. 可调配置参数

```bash
# === 算法 ===
ADV_ESTIMATOR=grpo     # grpo / gdpo / gae

# === KL 系数 (稳定性关键) ===
KL_LOSS_COEF=0.001     # 低约束，可能不稳定
KL_LOSS_COEF=0.1       # 中等约束
KL_LOSS_COEF=0.3       # 高约束，已知稳定

# === Reward 权重 (调整信号重要性) ===
ANSWER_WEIGHT=1.0      # 答案正确性 (最重要，建议 >= 1.0)
BBOX_WEIGHT=0.6        # Bounding box
FORMAT_WEIGHT=0.1      # 格式 (通常全对，可降低)
SEGMENT_WEIGHT=0.3     # Segment 分数

# === 采样 ===
TOP_P=0.7
TEMPERATURE=0.7

# === 学习率 ===
LEARNING_RATE=1e-6
```

---

## 5. 监控命令 (每20分钟)

```bash
# 检查训练指标
grep "step:" /tmp/exp_<NAME>.log | tail -5 | while read line; do
  step=$(echo "$line" | grep -oP "step:\d+" | cut -d: -f2)
  acc=$(echo "$line" | grep -oP "reward_components/acc/mean:[0-9.]+" | cut -d: -f2)
  reward=$(echo "$line" | grep -oP "critic/score/mean:[0-9.]+" | cut -d: -f2)
  entropy=$(echo "$line" | grep -oP "actor/entropy:[0-9.]+" | cut -d: -f2)
  echo "Step $step: acc=$acc, reward=$reward, entropy=$entropy"
done

# 检查是否崩溃
tail -5 /tmp/exp_<NAME>.log | grep -E "error|Exited|stopped"
```

---

## 6. 健康指标

| 指标 | 健康 | 警告 | 崩溃 |
|------|------|------|------|
| acc | 40-70% | < 30% | < 15% 连续3步 |
| reward | 0.35-0.55 | < 0.2 | < 0.1 连续3步 |
| entropy | 0.8-1.5 | < 0.5 | 持续下降趋近 0 |

**崩溃判定** (任一满足):
1. acc < 0.15 连续 3 步
2. reward < 0.1 连续 3 步
3. acc 单调递减 5 步，且末步 < 0.3

---

## 7. 崩溃恢复策略

| 崩溃次数 | 调整 |
|---------|------|
| 第1次 | KL_LOSS_COEF → 0.01 |
| 第2次 | KL_LOSS_COEF → 0.1 |
| 第3次 | KL_LOSS_COEF → 0.3 |

```bash
# 恢复命令
kill <PID>
KL_LOSS_COEF=0.1 nohup bash examples/video_reasoning/launch_multinode_slurm.sh \
    --jobid <JOB_ID> --nodes "nodeX,nodeY" > /tmp/exp_restart.log 2>&1 &
```

---

## 8. 实验对比策略

**最多同时运行 2 组实验** (各 2 机):
- 实验 A: node15, node19
- 实验 B: node22, node23

**对比维度**:
- acc 平均值和波动
- reward 趋势 (是否持续上升)
- entropy 稳定性

**成功标准**: reward 持续上升，acc 稳定在 50%+

---

## 9. 调优思路

1. **先稳定，后优化**
   - 从高 KL (0.3) 开始确保稳定
   - 逐步降低 KL 寻找最优点

2. **GAE vs GRPO 对比**
   - 同等条件下对比 reward 曲线
   - GAE 应该更平滑，利用 critic 估计

3. **Reward 权重调整**
   - 如果 acc 波动大: 增大 ANSWER_WEIGHT
   - 如果 format 总是 1.0: 降低 FORMAT_WEIGHT

---

## 10. 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| Entropy 过低 | KL 过大或 temperature 太低 | 降低 KL 或增大 temp |
| Acc 持续下降 | 策略偏移过大 | 增大 KL |
| 系统崩溃 | 节点问题 | 直接重启，无需调参 |
| vLLM 缓存错误 | 多模态处理 bug | 已修复，会自动跳过 |
