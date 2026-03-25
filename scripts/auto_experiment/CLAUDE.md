# Auto Experiment System

这是一个自动化 RL 训练实验管理系统，支持并行实验和自动监控。

## 目录结构

```
/mnt/data/home/zhengshurong/project/verl/scripts/auto_experiment/
├── CLAUDE.md                 # 本说明文件
├── experiment_state.json     # 实验状态记录（自动更新）
├── monitor.py                # 自动监控脚本（定期检查日志、检测崩溃）
├── launch_parallel.sh        # 并行实验启动脚本
├── orchestrator.py           # 主控制器（Python函数库）
├── run_experiment.sh         # 参数化训练脚本
├── logs/                     # 监控日志目录
└── configs/                  # 实验配置目录
```

## 快速开始

```bash
# 方法1: 启动并行实验并自动监控
bash scripts/auto_experiment/launch_parallel.sh

# 方法2: 仅启动监控（实验已在运行）
bash scripts/auto_experiment/launch_parallel.sh --monitor-only

# 方法3: 后台运行监控
nohup python3 scripts/auto_experiment/monitor.py --interval 60 > logs/monitor.log 2>&1 &
```

## 实验状态文件

`experiment_state.json` 记录所有实验状态，Claude 应该读取和更新这个文件：

```json
{
  "experiments": [],           // 所有实验记录
  "next_experiment_id": 1,     // 下一个实验ID
  "active_jobs": {},           // 当前运行中的实验
  "completed_experiments": [], // 已完成的实验
  "best_config": null,         // 目前最佳配置
  "insights": []               // 分析洞察
}
```

## 启动实验

使用环境变量传递参数：

```bash
# 方法1：通过环境变量（推荐）
export EXPERIMENT_NAME="exp6_kl0.4_bbox0.2"
export KL_LOSS_COEF=0.4
export BBOX_WEIGHT=0.2
export TOP_P=0.7
export ENTROPY_COEFF=0.0

bash examples/video_reasoning/launch_multinode_slurm.sh \
    --jobid <JOB_ID> --nodes "node1,node2"

# 方法2：单行命令
KL_LOSS_COEF=0.4 BBOX_WEIGHT=0.2 TOP_P=0.7 EXPERIMENT_NAME="my_exp" \
    bash examples/video_reasoning/launch_multinode_slurm.sh \
    --jobid <JOB_ID> --nodes "node1,node2"
```

## 可调参数（环境变量）

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `EXPERIMENT_NAME` | 自动生成 | 实验名称，用于区分不同实验 |
| `KL_LOSS_COEF` | 0.3 | KL 散度损失系数 |
| `ENTROPY_COEFF` | 0.0 | Entropy 系数（负值为惩罚） |
| `BBOX_WEIGHT` | 0.3 | BBox 验证权重 |
| `ANSWER_WEIGHT` | 1.0 | 答案权重 |
| `TOP_P` | 1.0 | Top-p 采样（DAPO推荐0.7） |
| `LEARNING_RATE` | 1e-6 | 学习率 |
| `CLIP_RATIO_LOW` | 0.2 | PPO clip ratio 下界 |
| `CLIP_RATIO_HIGH` | 0.25 | PPO clip ratio 上界 |
| `TRAIN_BATCH_SIZE` | 32 | 训练批次大小 |
| `TOTAL_EPOCHS` | 3 | 训练轮数 |
| `N_ROLLOUTS` | 8 | 每个prompt的rollout数 |

## 监控实验

从日志提取指标：

```bash
# 最近10步的 reward 和 entropy
tail -50000 <LOG_FILE> | grep "step:" | tail -10 | \
    grep -oP 'step:\K[0-9]+|critic/score/mean:\K[0-9.]+|actor/entropy:\K[0-9.]+'
```

## 崩溃检测

- **阈值**: reward < 0.05
- **规则**: 连续3步或5步中有3步低于阈值则认为崩溃
- **动作**: 停止训练，记录状态，分析原因

## 停止实验

```bash
# 停止 Ray 集群
srun --jobid=<JOB_ID> --overlap -w <NODE> -n1 ray stop --force
```

## 查看资源

```bash
# 查看空闲节点
sinfo -N -h -o "%N %T" | grep idle

# 查看 GPU 使用（如果有 web 服务）
curl http://localhost:8080/

# 查看当前作业
squeue -u $USER
```

## 实验历史分析

Claude 应该分析 `completed_experiments`，找出规律：

1. KL 系数与崩溃步数的关系
2. bbox_weight 对稳定性的影响
3. top_p 对 entropy 的控制效果
4. 最佳参数组合

## 下一步实验建议

基于当前实验结果，推荐尝试：

1. 如果 entropy 持续上升导致崩溃 → 增加 KL 系数或降低 top_p
2. 如果 reward 波动大 → 降低 bbox_weight（因为 bbox 验证不稳定）
3. 如果训练稳定但 reward 低 → 尝试负的 entropy 系数（惩罚高 entropy）

## Claude 操作权限

Claude Code 可以：
- [x] 读取 experiment_state.json
- [x] 写入 experiment_state.json
- [x] 启动实验（通过 bash 命令）
- [x] 监控实验（读取日志）
- [x] 停止实验（通过 ray stop）
- [x] 分析结果并决定下一步
