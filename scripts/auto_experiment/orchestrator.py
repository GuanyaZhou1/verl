#!/usr/bin/env python3
"""
Auto Experiment Orchestrator using Claude Agent SDK

这个脚本是主控制器，负责：
1. 查看空闲GPU资源
2. 申请节点
3. 启动实验（可并行多个）
4. 监控实验进度
5. 检测崩溃并记录
6. 分析结果，决定下一步实验

使用方式：
    # 使用 Claude Agent SDK 运行
    python orchestrator.py

    # 或直接用 claude 命令
    claude --prompt "请运行自动化实验系统，查看空闲GPU并启动实验"
"""

import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# =============================================================================
# 配置
# =============================================================================
PROJECT_DIR = Path("/mnt/data/home/zhengshurong/project/verl")
AUTO_EXP_DIR = PROJECT_DIR / "scripts" / "auto_experiment"
STATE_FILE = AUTO_EXP_DIR / "experiment_state.json"
LOGS_DIR = AUTO_EXP_DIR / "logs"

# 崩溃检测阈值
CRASH_THRESHOLD = 0.05
CRASH_COUNT_THRESHOLD = 3

# 实验参数搜索空间
PARAM_SEARCH_SPACE = {
    "kl_coef": [0.1, 0.2, 0.3, 0.4, 0.5],
    "entropy_coef": [0.0, -0.005, -0.01],
    "bbox_weight": [0.6, 0.3, 0.1],
    "top_p": [1.0, 0.9, 0.7],
    "learning_rate": ["1e-6", "5e-7", "2e-6"],
}

# =============================================================================
# 工具函数
# =============================================================================

def load_state() -> dict:
    """加载实验状态"""
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        "experiments": [],
        "next_experiment_id": 1,
        "active_jobs": {},
        "completed_experiments": [],
        "best_config": None,
        "insights": []
    }

def save_state(state: dict):
    """保存实验状态"""
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)

def get_idle_nodes() -> list:
    """获取空闲节点列表"""
    try:
        result = subprocess.run(
            ["sinfo", "-N", "-h", "-o", "%N %T"],
            capture_output=True, text=True
        )
        idle_nodes = []
        for line in result.stdout.strip().split("\n"):
            if line:
                parts = line.split()
                if len(parts) >= 2 and "idle" in parts[1]:
                    idle_nodes.append(parts[0])
        return idle_nodes
    except Exception as e:
        print(f"Error getting idle nodes: {e}")
        return []

def get_gpu_usage() -> dict:
    """获取GPU使用情况（通过 http://localhost:8080/）"""
    try:
        import requests
        response = requests.get("http://localhost:8080/", timeout=5)
        # 解析响应，返回每个节点的GPU使用情况
        return response.json() if response.ok else {}
    except Exception:
        return {}

def allocate_nodes(nodes: list, num_nodes: int = 2, job_name: str = "auto_exp") -> Optional[int]:
    """申请节点，返回 job_id"""
    if len(nodes) < num_nodes:
        print(f"Not enough idle nodes. Need {num_nodes}, got {len(nodes)}")
        return None

    selected_nodes = ",".join(nodes[:num_nodes])
    cmd = f"""salloc -p debug -N {num_nodes} --nodelist={selected_nodes} \
        --gres=gpu:8 --exclusive -t 128-00:00:00 \
        --job-name={job_name} srun sleep 11059200 &"""

    try:
        # 启动后台作业
        subprocess.Popen(cmd, shell=True)
        time.sleep(5)

        # 获取 job_id
        result = subprocess.run(
            ["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%i %j"],
            capture_output=True, text=True
        )
        for line in result.stdout.strip().split("\n"):
            if job_name in line:
                return int(line.split()[0])
    except Exception as e:
        print(f"Error allocating nodes: {e}")
    return None

def start_experiment(
    job_id: int,
    nodes: list,
    exp_id: int,
    params: dict
) -> dict:
    """启动一个实验"""
    exp_name = f"exp{exp_id}_kl{params.get('kl_coef', 0.3)}_bbox{params.get('bbox_weight', 0.6)}_topp{params.get('top_p', 1.0)}"
    log_file = LOGS_DIR / f"{exp_name}.log"

    # 构建启动命令
    nodes_str = ",".join(nodes)
    cmd_args = [
        f"--kl-coef {params.get('kl_coef', 0.3)}",
        f"--entropy-coef {params.get('entropy_coef', 0.0)}",
        f"--bbox-weight {params.get('bbox_weight', 0.6)}",
        f"--top-p {params.get('top_p', 1.0)}",
        f"--exp-name {exp_name}",
        f"--exp-id {exp_id}",
    ]

    launch_cmd = f"""cd {PROJECT_DIR} && \
        nohup bash examples/video_reasoning/launch_multinode_slurm.sh \
        --jobid {job_id} --nodes "{nodes_str}" \
        -- {' '.join(cmd_args)} > {log_file} 2>&1 &"""

    subprocess.Popen(launch_cmd, shell=True)

    return {
        "exp_id": exp_id,
        "exp_name": exp_name,
        "job_id": job_id,
        "nodes": nodes,
        "params": params,
        "log_file": str(log_file),
        "start_time": datetime.now().isoformat(),
        "status": "running",
        "metrics": []
    }

def monitor_experiment(log_file: str) -> dict:
    """监控实验，返回最新指标"""
    try:
        result = subprocess.run(
            ["tail", "-50000", log_file],
            capture_output=True, text=True
        )

        # 提取最新的 step 信息
        import re
        steps = []
        for line in result.stdout.split("\n"):
            if "step:" in line and "critic/score/mean" in line:
                step_match = re.search(r'step:(\d+)', line)
                reward_match = re.search(r'critic/score/mean:([0-9.]+)', line)
                entropy_match = re.search(r'actor/entropy:([0-9.]+)', line)

                if step_match and reward_match:
                    steps.append({
                        "step": int(step_match.group(1)),
                        "reward": float(reward_match.group(1)),
                        "entropy": float(entropy_match.group(1)) if entropy_match else None
                    })

        if not steps:
            return {"status": "initializing", "steps": []}

        recent_steps = steps[-5:]
        crash_count = sum(1 for s in recent_steps if s["reward"] < CRASH_THRESHOLD)

        return {
            "status": "crashed" if crash_count >= CRASH_COUNT_THRESHOLD else "running",
            "current_step": steps[-1]["step"],
            "current_reward": steps[-1]["reward"],
            "current_entropy": steps[-1].get("entropy"),
            "recent_rewards": [s["reward"] for s in recent_steps],
            "crash_count": crash_count
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

def stop_experiment(job_id: int, nodes: list):
    """停止实验"""
    for node in nodes:
        subprocess.run(
            ["srun", f"--jobid={job_id}", "--overlap", "-w", node, "-n1",
             "ray", "stop", "--force"],
            capture_output=True
        )

def record_experiment_result(state: dict, exp_info: dict, final_status: str, metrics: dict):
    """记录实验结果"""
    exp_info["status"] = final_status
    exp_info["end_time"] = datetime.now().isoformat()
    exp_info["final_metrics"] = metrics

    # 移动到已完成列表
    state["completed_experiments"].append(exp_info)

    # 更新最佳配置
    if final_status == "completed" or (metrics.get("max_step", 0) > 100):
        if state["best_config"] is None or \
           metrics.get("avg_reward", 0) > state["best_config"].get("avg_reward", 0):
            state["best_config"] = {
                "params": exp_info["params"],
                "avg_reward": metrics.get("avg_reward", 0),
                "max_step": metrics.get("max_step", 0)
            }

    save_state(state)

# =============================================================================
# Claude Agent SDK 集成说明
# =============================================================================
"""
使用 Claude Agent SDK 时，主 Agent 可以：

1. 调用 get_idle_nodes() 查看空闲节点
2. 调用 allocate_nodes() 申请资源
3. 调用 start_experiment() 启动实验
4. 调用 monitor_experiment() 监控进度
5. 调用 stop_experiment() 停止实验
6. 调用 record_experiment_result() 记录结果

子 Agent (Sonnet) 负责：
1. 持续监控单个实验的 log_file
2. 检测崩溃（reward < 0.05 连续3次）
3. 崩溃后通知主 Agent

示例 Claude 调用：

```python
from claude_agent_sdk import Agent

# 主 Agent
main_agent = Agent(model="claude-opus-4-5-20251001")

# 子 Agent（监控用）
monitor_agent = Agent(model="claude-sonnet-4-5-20251001")
```
"""

# =============================================================================
# 主函数（供 Claude 调用）
# =============================================================================

def main():
    """主入口，打印当前状态供 Claude 分析"""
    state = load_state()

    print("=" * 60)
    print("Auto Experiment System Status")
    print("=" * 60)

    # 空闲节点
    idle_nodes = get_idle_nodes()
    print(f"\nIdle Nodes: {idle_nodes}")

    # 活跃实验
    print(f"\nActive Experiments: {len(state.get('active_jobs', {}))}")
    for exp_id, exp_info in state.get('active_jobs', {}).items():
        print(f"  - {exp_info['exp_name']}: {exp_info['status']}")

    # 已完成实验
    print(f"\nCompleted Experiments: {len(state.get('completed_experiments', []))}")
    for exp in state.get('completed_experiments', [])[-5:]:
        print(f"  - {exp['exp_name']}: {exp['status']} (max_step={exp.get('final_metrics', {}).get('max_step', 'N/A')})")

    # 最佳配置
    if state.get("best_config"):
        print(f"\nBest Config So Far:")
        print(f"  Params: {state['best_config']['params']}")
        print(f"  Avg Reward: {state['best_config'].get('avg_reward', 'N/A')}")

    print("\n" + "=" * 60)

    return state

if __name__ == "__main__":
    main()
