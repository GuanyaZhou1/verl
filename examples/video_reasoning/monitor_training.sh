#!/bin/bash
# 训练监控脚本 - 每小时检查一次训练状态

LOG_FILE="./logs/monitor_$(date +%Y%m%d_%H%M%S).log"
CHECK_INTERVAL=3600  # 1小时 = 3600秒

echo "[$(date)] Training monitor started" | tee -a "$LOG_FILE"
echo "Check interval: ${CHECK_INTERVAL}s (1 hour)" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

while true; do
    echo "========================================" | tee -a "$LOG_FILE"
    echo "[$(date)] Checking training status..." | tee -a "$LOG_FILE"

    # 检查 Ray 集群状态
    if ray status 2>/dev/null | grep -q "Active"; then
        echo "✓ Ray cluster is active" | tee -a "$LOG_FILE"
        ray status 2>&1 | tee -a "$LOG_FILE"
    else
        echo "✗ Ray cluster not responding!" | tee -a "$LOG_FILE"
    fi

    echo "" | tee -a "$LOG_FILE"

    # 检查训练进程
    TRAIN_PROCS=$(pgrep -f "verl.trainer.main_ppo" | wc -l)
    if [ "$TRAIN_PROCS" -gt 0 ]; then
        echo "✓ Training process running (${TRAIN_PROCS} processes)" | tee -a "$LOG_FILE"
    else
        echo "✗ No training process found!" | tee -a "$LOG_FILE"
    fi

    echo "" | tee -a "$LOG_FILE"

    # 检查最新日志
    LATEST_LOG=$(ls -t ./logs/*.log 2>/dev/null | head -1)
    if [ -n "$LATEST_LOG" ]; then
        echo "Latest log: $LATEST_LOG" | tee -a "$LOG_FILE"
        echo "Last 5 lines:" | tee -a "$LOG_FILE"
        tail -5 "$LATEST_LOG" | tee -a "$LOG_FILE"
    fi

    echo "" | tee -a "$LOG_FILE"

    # 检查 GPU 使用情况
    echo "GPU status:" | tee -a "$LOG_FILE"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>&1 | tee -a "$LOG_FILE"

    echo "" | tee -a "$LOG_FILE"
    echo "Next check in ${CHECK_INTERVAL}s (1 hour)..." | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"

    sleep "$CHECK_INTERVAL"
done
