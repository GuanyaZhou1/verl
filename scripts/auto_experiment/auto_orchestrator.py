#!/usr/bin/env python3
"""
全自动实验编排器 - 基于 Claude Agent SDK

功能：
1. 定期监控实验日志，提取 reward/entropy 指标
2. 检测 reward 持续下降或崩溃
3. 调用 Claude 分析实验历史，智能建议新配置
4. 自动停止失败实验，启动新配置

使用方式：
    # 前台运行
    python auto_orchestrator.py

    # 后台运行
    nohup python auto_orchestrator.py > logs/orchestrator.log 2>&1 &

依赖：
    pip install claude-agent-sdk
"""

import asyncio
import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# Claude Agent SDK
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    query,
)

# =============================================================================
# 配置
# =============================================================================
PROJECT_DIR = Path("/mnt/data/home/zhengshurong/project/verl")
AUTO_EXP_DIR = PROJECT_DIR / "scripts" / "auto_experiment"
STATE_FILE = AUTO_EXP_DIR / "experiment_state.json"
LOGS_DIR = AUTO_EXP_DIR / "logs"

# 监控配置
CHECK_INTERVAL = 600  # 检查间隔（秒）- 10分钟，每步训练约4分钟
MIN_STEPS_FOR_TREND = 5  # 最少需要多少步才能判断趋势

# 崩溃检测
CRASH_THRESHOLD = 0.05
CRASH_COUNT = 3

# 下降检测 - 放宽阈值，允许小幅波动
DECLINE_THRESHOLD = 0.25  # reward 下降超过 25% 才认为是持续下降（之前是15%）
DECLINE_STEPS = 30  # 观察更多步再判断（之前是20步）

# Claude API 配置
CLAUDE_MODEL = "claude-sonnet-4-6-cc"  # 用 Sonnet 分析，降低成本
MAX_ANALYSIS_COST = 0.50  # 单次分析最大花费


# =============================================================================
# 状态管理
# =============================================================================
def load_state() -> dict:
    """加载实验状态"""
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {
        "experiments": [],
        "next_experiment_id": 1,
        "active_jobs": {},
        "completed_experiments": [],
        "best_config": None,
        "insights": [],
        "total_ai_cost": 0.0,
    }


def save_state(state: dict):
    """保存实验状态"""
    STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False))


# =============================================================================
# 日志解析
# =============================================================================
def extract_metrics(log_file: str) -> list[dict]:
    """从日志提取训练指标"""
    try:
        result = subprocess.run(
            ["tail", "-100000", log_file],
            capture_output=True, text=True, timeout=30
        )

        metrics = []
        for line in result.stdout.split("\n"):
            # 匹配 step:X
            step_match = re.search(r'step:(\d+)', line)
            if not step_match:
                continue

            step = int(step_match.group(1))
            reward = None
            entropy = None

            # 格式1: 验证 - val-aux/video_reasoning/reward/mean@1:np.float64(Y)
            reward_match = re.search(r'video_reasoning/reward/mean@\d+:np\.float64\(([0-9.]+)\)', line)
            if reward_match:
                reward = float(reward_match.group(1))

            # 格式2: 训练 - critic/score/mean:Y
            if reward is None:
                reward_match = re.search(r'critic/score/mean:([0-9.]+)', line)
                if reward_match:
                    reward = float(reward_match.group(1))

            # 提取 entropy
            entropy_match = re.search(r'actor/entropy:([0-9.]+)', line)
            if entropy_match:
                entropy = float(entropy_match.group(1))

            if reward is not None:
                metrics.append({
                    "step": step,
                    "reward": reward,
                    "entropy": entropy,
                })
        return metrics
    except Exception as e:
        print(f"  Error reading {log_file}: {e}")
        return []


def analyze_trend(metrics: list[dict]) -> dict:
    """分析 reward 趋势"""
    if len(metrics) < MIN_STEPS_FOR_TREND:
        return {"status": "insufficient_data", "steps": len(metrics)}

    # 检测崩溃
    recent = metrics[-5:]
    crash_count = sum(1 for m in recent if m["reward"] < CRASH_THRESHOLD)
    if crash_count >= CRASH_COUNT:
        return {
            "status": "crashed",
            "current_step": metrics[-1]["step"],
            "current_reward": metrics[-1]["reward"],
        }

    # 检测下降趋势
    if len(metrics) >= DECLINE_STEPS:
        initial_reward = metrics[0]["reward"]
        recent_reward = sum(m["reward"] for m in metrics[-5:]) / 5
        decline_pct = (initial_reward - recent_reward) / initial_reward

        if decline_pct > DECLINE_THRESHOLD:
            return {
                "status": "declining",
                "initial_reward": initial_reward,
                "current_reward": recent_reward,
                "decline_pct": decline_pct,
                "current_step": metrics[-1]["step"],
            }

    # 检测完成
    if metrics[-1]["step"] >= 135:
        return {
            "status": "completed",
            "initial_reward": metrics[0]["reward"],
            "final_reward": metrics[-1]["reward"],
            "max_step": metrics[-1]["step"],
        }

    # 正常运行中
    return {
        "status": "running",
        "current_step": metrics[-1]["step"],
        "current_reward": metrics[-1]["reward"],
        "initial_reward": metrics[0]["reward"] if metrics else None,
    }


# =============================================================================
# 实验控制
# =============================================================================
def stop_experiment(job_id: int, nodes: list[str]):
    """停止实验"""
    print(f"  Stopping experiment on {nodes}...")
    for node in nodes:
        try:
            subprocess.run(
                ["srun", f"--jobid={job_id}", "--overlap", "-w", node, "-n1",
                 "ray", "stop", "--force"],
                capture_output=True, timeout=30
            )
        except Exception as e:
            print(f"  Warning: Failed to stop on {node}: {e}")


def start_experiment(job_id: int, nodes: list[str], exp_id: int, params: dict) -> dict:
    """启动新实验"""
    exp_name = f"exp{exp_id}_kl{params['kl_coef']}_bbox{params['bbox_weight']}_topp{params['top_p']}"
    log_file = f"/tmp/training_{job_id}_{exp_name}.log"
    nodes_str = ",".join(nodes)

    # 设置环境变量并启动
    env_vars = {
        "EXPERIMENT_NAME": exp_name,
        "KL_LOSS_COEF": str(params["kl_coef"]),
        "ENTROPY_COEFF": str(params.get("entropy_coef", 0.0)),
        "BBOX_WEIGHT": str(params["bbox_weight"]),
        "TOP_P": str(params["top_p"]),
    }

    cmd = f"""cd {PROJECT_DIR} && \
        export EXPERIMENT_NAME="{exp_name}" && \
        export KL_LOSS_COEF={params['kl_coef']} && \
        export ENTROPY_COEFF={params.get('entropy_coef', 0.0)} && \
        export BBOX_WEIGHT={params['bbox_weight']} && \
        export TOP_P={params['top_p']} && \
        nohup bash examples/video_reasoning/launch_multinode_slurm.sh \
            --jobid {job_id} --nodes "{nodes_str}" \
            > {log_file} 2>&1 &"""

    subprocess.Popen(cmd, shell=True)

    return {
        "exp_id": exp_id,
        "exp_name": exp_name,
        "job_id": job_id,
        "nodes": nodes,
        "params": params,
        "log_file": log_file,
        "start_time": datetime.now().isoformat(),
        "status": "starting",
    }


# =============================================================================
# Claude 智能分析
# =============================================================================
async def analyze_and_suggest_config(state: dict, failed_exp: dict, trend: dict) -> Optional[dict]:
    """
    调用 Claude 分析实验历史，建议新配置

    返回格式：{"kl_coef": x, "bbox_weight": y, "top_p": z, "reason": "..."}
    """
    # 构建实验历史摘要
    completed = state.get("completed_experiments", [])
    history_lines = []
    for exp in completed[-10:]:  # 最近10个实验
        params = exp.get("params", {})
        status = exp.get("status", "unknown")
        init_r = exp.get("initial_reward", "?")
        final_r = exp.get("final_reward", "?")
        steps = exp.get("max_step", exp.get("crash_step", "?"))
        history_lines.append(
            f"- {exp.get('exp_name')}: status={status}, steps={steps}, "
            f"reward={init_r}→{final_r}, params={params}"
        )

    history_text = "\n".join(history_lines) if history_lines else "无历史实验"

    # 当前失败实验信息
    failed_info = f"""
当前失败实验: {failed_exp.get('exp_name')}
- 参数: {failed_exp.get('params')}
- 状态: {trend.get('status')}
- 初始reward: {trend.get('initial_reward', '?')}
- 当前reward: {trend.get('current_reward', '?')}
- 下降幅度: {trend.get('decline_pct', 0) * 100:.1f}%
- 当前步数: {trend.get('current_step', '?')}
"""

    # 已知洞察
    insights = state.get("insights", [])
    insights_text = "\n".join(f"- {i}" for i in insights) if insights else "无"

    prompt = f"""你是一个强化学习实验调参专家。分析以下实验历史，为下一个实验建议最优配置。

## 实验历史
{history_text}

## 当前失败的实验
{failed_info}

## 已知洞察
{insights_text}

## 可调参数范围
- kl_coef: 0.1 ~ 0.5 (KL散度约束，越大越保守)
- bbox_weight: 0.0 ~ 0.3 (bbox验证权重，已知74%返回0分导致噪声)
- top_p: 0.7 ~ 1.0 (采样随机性，0.7更稳定)
- entropy_coef: -0.01 ~ 0.0 (entropy惩罚，负值惩罚高entropy)

## 目标
找到一个配置使得 reward 能够**上升**而不是下降。

## 输出要求
只输出一个 JSON 对象，格式如下：
```json
{{
  "kl_coef": <float>,
  "bbox_weight": <float>,
  "top_p": <float>,
  "entropy_coef": <float>,
  "reason": "<简短解释为什么选择这个配置>"
}}
```
"""

    print("\n  [AI] Analyzing experiment history...")

    suggested_config = None
    cost = 0.0

    try:
        async for message in query(
            prompt=prompt,
            options=ClaudeAgentOptions(
                model=CLAUDE_MODEL,
                allowed_tools=[],  # 纯分析，不需要工具
                permission_mode="bypassPermissions",
                max_turns=1,
                max_budget_usd=MAX_ANALYSIS_COST,
                cwd=str(PROJECT_DIR),
            ),
        ):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        text = block.text
                        # 提取 JSON
                        json_match = re.search(r'\{[^{}]*"kl_coef"[^{}]*\}', text, re.DOTALL)
                        if json_match:
                            try:
                                suggested_config = json.loads(json_match.group())
                                print(f"  [AI] Suggested: {suggested_config}")
                            except json.JSONDecodeError:
                                print(f"  [AI] Failed to parse JSON from response")

            elif isinstance(message, ResultMessage):
                cost = message.total_cost_usd or 0
                print(f"  [AI] Analysis cost: ${cost:.4f}")

    except Exception as e:
        print(f"  [AI] Error: {e}")

    # 更新总花费
    state["total_ai_cost"] = state.get("total_ai_cost", 0) + cost
    save_state(state)

    return suggested_config


# =============================================================================
# 主循环
# =============================================================================
async def monitor_and_adjust():
    """主监控循环"""
    print("=" * 60)
    print("Auto Orchestrator Started")
    print(f"Check interval: {CHECK_INTERVAL}s")
    print(f"Claude model: {CLAUDE_MODEL}")
    print("=" * 60)

    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    while True:
        try:
            state = load_state()
            active_jobs = state.get("active_jobs", {})

            if not active_jobs:
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] No active experiments. Waiting...")
                await asyncio.sleep(CHECK_INTERVAL)
                continue

            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Checking {len(active_jobs)} experiments...")

            experiments_to_stop = []
            experiments_to_complete = []

            for exp_id, exp_info in active_jobs.items():
                exp_name = exp_info.get("exp_name", f"exp{exp_id}")
                log_file = exp_info.get("log_file", "")

                if not Path(log_file).exists():
                    print(f"  [{exp_name}] Log not found, skipping")
                    continue

                metrics = extract_metrics(log_file)
                if not metrics:
                    print(f"  [{exp_name}] No metrics yet (initializing)")
                    continue

                trend = analyze_trend(metrics)
                exp_info["metrics"] = metrics[-20:]  # 保留最近20个
                exp_info["last_check"] = datetime.now().isoformat()

                current_reward = trend.get('current_reward')
                reward_str = f"{current_reward:.4f}" if isinstance(current_reward, (int, float)) else "?"
                print(f"  [{exp_name}] step={trend.get('current_step', '?')}, "
                      f"reward={reward_str}, status={trend['status']}")

                if trend["status"] == "crashed":
                    print(f"    ⚠️ CRASHED! Scheduling stop and reconfiguration...")
                    experiments_to_stop.append((exp_id, exp_info, trend))

                elif trend["status"] == "declining":
                    print(f"    📉 DECLINING ({trend['decline_pct']*100:.1f}% drop)! "
                          f"Scheduling stop and reconfiguration...")
                    experiments_to_stop.append((exp_id, exp_info, trend))

                elif trend["status"] == "completed":
                    print(f"    ✅ COMPLETED! reward: {trend['initial_reward']:.4f} → {trend['final_reward']:.4f}")
                    # 判断是否上升
                    if trend["final_reward"] > trend["initial_reward"]:
                        print(f"    🎉 REWARD INCREASED!")
                        exp_info["reward_increased"] = True
                    else:
                        exp_info["reward_increased"] = False
                    experiments_to_complete.append((exp_id, exp_info, trend))

            # 处理完成的实验
            for exp_id, exp_info, trend in experiments_to_complete:
                exp_info["status"] = "completed"
                exp_info["end_time"] = datetime.now().isoformat()
                exp_info["initial_reward"] = trend["initial_reward"]
                exp_info["final_reward"] = trend["final_reward"]
                exp_info["max_step"] = trend["max_step"]

                state["completed_experiments"].append(exp_info)
                del state["active_jobs"][exp_id]
                save_state(state)

            # 处理需要停止并重新配置的实验
            for exp_id, exp_info, trend in experiments_to_stop:
                job_id = exp_info["job_id"]
                nodes = exp_info["nodes"]

                # 1. 停止当前实验
                stop_experiment(job_id, nodes)

                # 2. 记录失败
                exp_info["status"] = trend["status"]
                exp_info["end_time"] = datetime.now().isoformat()
                exp_info["crash_step"] = trend.get("current_step")
                exp_info["initial_reward"] = trend.get("initial_reward")
                exp_info["final_reward"] = trend.get("current_reward")
                state["completed_experiments"].append(exp_info)
                del state["active_jobs"][exp_id]
                save_state(state)

                # 3. 调用 Claude 分析并建议新配置
                new_config = await analyze_and_suggest_config(state, exp_info, trend)

                if new_config:
                    # 4. 启动新实验
                    new_exp_id = state.get("next_experiment_id", 1)
                    state["next_experiment_id"] = new_exp_id + 1

                    # 添加洞察
                    reason = new_config.pop("reason", "")
                    if reason:
                        state["insights"].append(f"Exp{exp_id}失败后: {reason}")
                        # 保持最多20条洞察
                        state["insights"] = state["insights"][-20:]

                    print(f"\n  [AUTO] Starting new experiment {new_exp_id} with config: {new_config}")

                    # 等待节点清理
                    await asyncio.sleep(10)

                    new_exp_info = start_experiment(job_id, nodes, new_exp_id, new_config)
                    state["active_jobs"][str(new_exp_id)] = new_exp_info
                    save_state(state)

                    print(f"  [AUTO] Experiment {new_exp_id} started on {nodes}")
                else:
                    print(f"  [AUTO] No new config suggested, nodes {nodes} now idle")

            # 打印状态摘要
            state = load_state()
            active = len(state.get("active_jobs", {}))
            completed = len(state.get("completed_experiments", []))
            ai_cost = state.get("total_ai_cost", 0)
            print(f"\n  Summary: {active} active, {completed} completed, AI cost: ${ai_cost:.2f}")

            if active == 0:
                print("\n  No active experiments remaining. Orchestrator exiting.")
                break

            print(f"\n  Next check in {CHECK_INTERVAL}s...")
            await asyncio.sleep(CHECK_INTERVAL)

        except KeyboardInterrupt:
            print("\n\nOrchestrator stopped by user")
            break
        except Exception as e:
            print(f"\n  ERROR: {e}")
            import traceback
            traceback.print_exc()
            await asyncio.sleep(CHECK_INTERVAL)


# =============================================================================
# 入口
# =============================================================================
if __name__ == "__main__":
    asyncio.run(monitor_and_adjust())
