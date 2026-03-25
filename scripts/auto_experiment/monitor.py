#!/usr/bin/env python3
"""
自动化实验监控脚本

功能：
1. 定期读取所有活跃实验的日志
2. 提取训练指标 (reward, entropy, step)
3. 检测崩溃 (reward < 0.05 连续3次)
4. 更新 experiment_state.json
5. 完成/崩溃时自动记录结果

使用方式：
    # 前台运行（每60秒检查一次）
    python monitor.py

    # 后台运行
    nohup python monitor.py > monitor.log 2>&1 &

    # 指定检查间隔（秒）
    python monitor.py --interval 120
"""

import json
import re
import subprocess
import time
import argparse
from datetime import datetime
from pathlib import Path

# 配置
PROJECT_DIR = Path("/mnt/data/home/zhengshurong/project/verl")
AUTO_EXP_DIR = PROJECT_DIR / "scripts" / "auto_experiment"
STATE_FILE = AUTO_EXP_DIR / "experiment_state.json"

# 崩溃检测阈值
CRASH_THRESHOLD = 0.05
CRASH_COUNT_THRESHOLD = 3


def load_state():
    """加载实验状态"""
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"active_jobs": {}, "completed_experiments": [], "insights": []}


def save_state(state):
    """保存实验状态"""
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def extract_metrics(log_file):
    """从日志提取训练指标"""
    try:
        result = subprocess.run(
            ["tail", "-100000", log_file],
            capture_output=True, text=True, timeout=30
        )

        metrics = []
        for line in result.stdout.split("\n"):
            if "critic/score/mean" in line:
                step_match = re.search(r'step:(\d+)', line)
                reward_match = re.search(r'critic/score/mean:([0-9.]+)', line)
                entropy_match = re.search(r'actor/entropy:([0-9.]+)', line)

                if step_match and reward_match:
                    metrics.append({
                        "step": int(step_match.group(1)),
                        "reward": float(reward_match.group(1)),
                        "entropy": float(entropy_match.group(1)) if entropy_match else None,
                        "time": datetime.now().isoformat()
                    })

        return metrics
    except Exception as e:
        print(f"  Error reading {log_file}: {e}")
        return []


def check_crash(metrics, threshold=CRASH_THRESHOLD, count=CRASH_COUNT_THRESHOLD):
    """检测是否崩溃"""
    if len(metrics) < count:
        return False, 0

    recent = metrics[-5:]
    crash_count = sum(1 for m in recent if m["reward"] < threshold)
    return crash_count >= count, crash_count


def check_completed(metrics, total_steps=138):
    """检测是否完成训练"""
    if not metrics:
        return False
    return metrics[-1]["step"] >= total_steps - 1


def check_process_alive(log_file):
    """检查训练进程是否还在运行"""
    try:
        result = subprocess.run(
            ["tail", "-5", log_file],
            capture_output=True, text=True, timeout=10
        )
        # 检查是否有错误退出信息
        if "Exited with exit code" in result.stdout:
            return False
        if "error" in result.stdout.lower() and "cuda" in result.stdout.lower():
            return False
        return True
    except:
        return True  # 读取失败时假设还在运行


def monitor_experiments(state):
    """监控所有活跃实验"""
    active_jobs = state.get("active_jobs", {})

    if not active_jobs:
        print("No active experiments to monitor")
        return state

    print(f"\n{'='*60}")
    print(f"Monitoring {len(active_jobs)} experiments at {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*60}")

    completed_ids = []

    for exp_id, exp_info in active_jobs.items():
        exp_name = exp_info.get("exp_name", f"exp{exp_id}")
        log_file = exp_info.get("log_file", "")

        print(f"\n[{exp_name}]")

        if not Path(log_file).exists():
            print(f"  Log file not found: {log_file}")
            continue

        # 提取指标
        metrics = extract_metrics(log_file)

        if not metrics:
            print(f"  No training metrics yet (initializing...)")
            continue

        # 更新实验指标
        exp_info["metrics"] = metrics[-10:]  # 保留最近10个

        latest = metrics[-1]
        initial = metrics[0] if metrics else latest

        print(f"  Step: {latest['step']}")
        print(f"  Reward: {latest['reward']:.4f} (initial: {initial['reward']:.4f})")
        if latest.get('entropy'):
            print(f"  Entropy: {latest['entropy']:.4f}")

        # 计算reward变化
        if len(metrics) > 1:
            change = (latest['reward'] - initial['reward']) / initial['reward'] * 100
            trend = "↑" if change > 0 else "↓"
            print(f"  Trend: {trend} {abs(change):.1f}%")

        # 检测崩溃
        crashed, crash_count = check_crash(metrics)
        if crashed:
            print(f"  ⚠️  CRASHED! (reward < {CRASH_THRESHOLD} for {crash_count}/5 recent steps)")
            exp_info["status"] = "crashed"
            exp_info["crash_step"] = latest["step"]
            exp_info["final_reward"] = latest["reward"]
            exp_info["initial_reward"] = initial["reward"]
            exp_info["end_time"] = datetime.now().isoformat()
            completed_ids.append(exp_id)
            continue

        # 检测完成
        if check_completed(metrics):
            print(f"  ✓ COMPLETED!")
            exp_info["status"] = "completed"
            exp_info["max_step"] = latest["step"]
            exp_info["final_reward"] = latest["reward"]
            exp_info["initial_reward"] = initial["reward"]
            exp_info["end_time"] = datetime.now().isoformat()

            # 判断reward是否上升
            if latest["reward"] > initial["reward"]:
                exp_info["reward_increased"] = True
                print(f"  🎉 REWARD INCREASED! {initial['reward']:.4f} → {latest['reward']:.4f}")
            else:
                exp_info["reward_increased"] = False

            completed_ids.append(exp_id)
            continue

        # 检查进程状态
        if not check_process_alive(log_file):
            print(f"  ⚠️  Process may have died")
            exp_info["status"] = "error"
            exp_info["end_time"] = datetime.now().isoformat()
            completed_ids.append(exp_id)
            continue

        exp_info["status"] = "running"

    # 移动完成的实验
    for exp_id in completed_ids:
        exp_info = active_jobs.pop(exp_id)
        state["completed_experiments"].append(exp_info)
        print(f"\nMoved {exp_info['exp_name']} to completed_experiments")

    return state


def print_summary(state):
    """打印实验摘要"""
    completed = state.get("completed_experiments", [])
    active = state.get("active_jobs", {})

    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"Active: {len(active)}, Completed: {len(completed)}")

    if completed:
        print("\nCompleted experiments:")
        print(f"{'Name':<35} {'Status':<10} {'Steps':<8} {'Init→Final':<20} {'Change':<10}")
        print("-" * 85)

        for exp in completed[-10:]:  # 最近10个
            name = exp.get("exp_name", "unknown")[:34]
            status = exp.get("status", "?")
            steps = exp.get("max_step", exp.get("crash_step", "?"))
            init_r = exp.get("initial_reward", 0)
            final_r = exp.get("final_reward", 0)

            if init_r and final_r:
                change = (final_r - init_r) / init_r * 100
                change_str = f"{'+' if change > 0 else ''}{change:.1f}%"
            else:
                change_str = "N/A"

            reward_str = f"{init_r:.3f} → {final_r:.3f}"
            print(f"{name:<35} {status:<10} {steps:<8} {reward_str:<20} {change_str:<10}")

    # 找到reward上升的实验
    increasing = [e for e in completed if e.get("reward_increased")]
    if increasing:
        print(f"\n🎉 Found {len(increasing)} experiments with INCREASING reward:")
        for exp in increasing:
            print(f"  - {exp['exp_name']}: {exp['params']}")


def main():
    parser = argparse.ArgumentParser(description="Auto Experiment Monitor")
    parser.add_argument("--interval", type=int, default=60, help="Check interval in seconds")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    args = parser.parse_args()

    print(f"Auto Experiment Monitor started")
    print(f"Check interval: {args.interval}s")
    print(f"State file: {STATE_FILE}")

    while True:
        try:
            state = load_state()
            state = monitor_experiments(state)
            save_state(state)
            print_summary(state)

            if args.once:
                break

            active = state.get("active_jobs", {})
            if not active:
                print("\nNo active experiments. Exiting.")
                break

            print(f"\nNext check in {args.interval}s... (Ctrl+C to stop)")
            time.sleep(args.interval)

        except KeyboardInterrupt:
            print("\nMonitor stopped by user")
            break
        except Exception as e:
            print(f"\nError: {e}")
            if args.once:
                break
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
