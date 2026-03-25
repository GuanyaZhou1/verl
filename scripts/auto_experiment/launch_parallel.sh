#!/bin/bash
# =============================================================================
# 并行实验启动脚本
# =============================================================================
# 一次性启动多个实验，然后自动监控
#
# 使用方式：
#   # 启动实验并监控
#   bash launch_parallel.sh
#
#   # 仅启动监控（实验已在运行）
#   bash launch_parallel.sh --monitor-only
# =============================================================================

set -e
cd "$(dirname "$0")/../.."

PROJECT_DIR="$(pwd)"
AUTO_EXP_DIR="$PROJECT_DIR/scripts/auto_experiment"
STATE_FILE="$AUTO_EXP_DIR/experiment_state.json"

# 检查参数
MONITOR_ONLY=false
if [[ "$1" == "--monitor-only" ]]; then
    MONITOR_ONLY=true
fi

# =============================================================================
# 实验配置 - 修改这里定义要运行的实验
# =============================================================================
# 格式: "JOB_ID:NODES:EXP_NAME:KL_COEF:BBOX_WEIGHT:TOP_P"
EXPERIMENTS=(
    "21719:node33,node34:exp6_kl0.3_bbox0.0_topp0.7:0.3:0.0:0.7"
    "21758:node26,node27:exp7_kl0.4_bbox0.0_topp0.7:0.4:0.0:0.7"
    "21758:node31,node32:exp8_kl0.3_bbox0.1_topp0.7:0.3:0.1:0.7"
)

# =============================================================================
# 启动实验
# =============================================================================
start_experiments() {
    echo "=========================================="
    echo "Starting ${#EXPERIMENTS[@]} parallel experiments"
    echo "=========================================="

    for exp_config in "${EXPERIMENTS[@]}"; do
        IFS=':' read -r JOB_ID NODES EXP_NAME KL_COEF BBOX_WEIGHT TOP_P <<< "$exp_config"

        echo ""
        echo "Starting: $EXP_NAME"
        echo "  Job ID: $JOB_ID"
        echo "  Nodes: $NODES"
        echo "  KL: $KL_COEF, BBox: $BBOX_WEIGHT, TopP: $TOP_P"

        LOG_FILE="/tmp/training_${JOB_ID}_${EXP_NAME}.log"

        # 先停止可能存在的 Ray 进程
        for node in $(echo "$NODES" | tr ',' ' '); do
            srun --jobid="$JOB_ID" --overlap -w "$node" -n1 ray stop --force 2>/dev/null &
        done
        wait 2>/dev/null

        # 启动实验
        export EXPERIMENT_NAME="$EXP_NAME"
        export KL_LOSS_COEF="$KL_COEF"
        export ENTROPY_COEFF=0.0
        export BBOX_WEIGHT="$BBOX_WEIGHT"
        export TOP_P="$TOP_P"

        nohup bash examples/video_reasoning/launch_multinode_slurm.sh \
            --jobid "$JOB_ID" --nodes "$NODES" \
            > "$LOG_FILE" 2>&1 &

        echo "  Log: $LOG_FILE"
        echo "  Started with PID: $!"

        # 更新状态文件
        python3 -c "
import json
from datetime import datetime
from pathlib import Path

state_file = Path('$STATE_FILE')
state = json.loads(state_file.read_text()) if state_file.exists() else {}

exp_id = state.get('next_experiment_id', 1)
state.setdefault('active_jobs', {})

state['active_jobs'][str(exp_id)] = {
    'exp_id': exp_id,
    'exp_name': '$EXP_NAME',
    'job_id': $JOB_ID,
    'nodes': '$NODES'.split(','),
    'params': {
        'kl_coef': $KL_COEF,
        'entropy_coef': 0.0,
        'bbox_weight': $BBOX_WEIGHT,
        'top_p': $TOP_P
    },
    'log_file': '$LOG_FILE',
    'start_time': datetime.now().isoformat(),
    'status': 'starting'
}

state['next_experiment_id'] = exp_id + 1
state_file.write_text(json.dumps(state, indent=2, ensure_ascii=False))
print(f'  Registered as exp{exp_id}')
"
    done

    echo ""
    echo "All experiments started!"
    echo "Waiting 30s for initialization..."
    sleep 30
}

# =============================================================================
# 主逻辑
# =============================================================================
if [[ "$MONITOR_ONLY" == "false" ]]; then
    start_experiments
fi

echo ""
echo "=========================================="
echo "Starting monitor..."
echo "=========================================="
python3 "$AUTO_EXP_DIR/monitor.py" --interval 60
