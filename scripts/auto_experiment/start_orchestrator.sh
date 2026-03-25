#!/bin/bash
# 启动自动编排器的脚本
# 解决 Claude Code 嵌套会话问题

# 取消 CLAUDECODE 环境变量以避免嵌套会话检测
unset CLAUDECODE

cd /mnt/data/home/zhengshurong/project/verl

LOG_FILE="scripts/auto_experiment/logs/orchestrator_$(date +%Y%m%d_%H%M%S).log"
echo "Starting Auto Orchestrator..."
echo "Log file: $LOG_FILE"

python3 -u scripts/auto_experiment/auto_orchestrator.py > "$LOG_FILE" 2>&1 &
echo "Started with PID: $!"
echo ""
echo "To monitor:"
echo "  tail -f $LOG_FILE"
