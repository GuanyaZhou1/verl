#!/bin/bash
# =============================================================================
# Filter Groups 实验启动脚本 (直接命令行传参版本)
# =============================================================================

set -e
cd "$(dirname "$0")/../.."

PROJECT_DIR="$(pwd)"
AUTO_EXP_DIR="$PROJECT_DIR/scripts/auto_experiment"
STATE_FILE="$AUTO_EXP_DIR/experiment_state.json"

JOB_ID=21758

echo "=========================================="
echo "Filter Groups 实验 (命令行传参)"
echo "=========================================="

# =============================================================================
# 实验 14: Filter Groups + 标准参数 (nodes 26,27)
# =============================================================================
start_exp14() {
    echo ""
    echo "=========================================="
    echo "Exp14: filter_groups=true, gen_bs=64, kl=0.4"
    echo "Nodes: node26,node27"
    echo "=========================================="

    local NODES="node26,node27"
    local EXP_NAME="exp14_filter_kl0.4_genbs64"
    local LOG_FILE="/tmp/training_${JOB_ID}_${EXP_NAME}.log"

    # 停止 Ray
    for node in node26 node27; do
        srun --jobid="$JOB_ID" --overlap -w "$node" -n1 ray stop --force 2>/dev/null || true
    done
    sleep 3

    # 直接通过命令行传递 Hydra 参数
    nohup bash examples/video_reasoning/launch_multinode_slurm.sh \
        --jobid "$JOB_ID" --nodes "$NODES" \
        -- \
        algorithm.filter_groups.enable=true \
        algorithm.filter_groups.metric=score \
        algorithm.filter_groups.max_num_gen_batches=5 \
        data.gen_batch_size=64 \
        algorithm.kl_ctrl.kl_coef=0.4 \
        custom_reward_function.reward_kwargs.bbox_weight=0.0 \
        actor_rollout_ref.rollout.top_p=0.7 \
        > "$LOG_FILE" 2>&1 &

    echo "Log: $LOG_FILE"
    echo "PID: $!"

    # 更新状态
    python3 << EOF
import json
from datetime import datetime
from pathlib import Path
state_file = Path('$STATE_FILE')
state = json.loads(state_file.read_text()) if state_file.exists() else {}
exp_id = state.get('next_experiment_id', 14)
state.setdefault('active_jobs', {})
state['active_jobs'][str(exp_id)] = {
    'exp_id': exp_id, 'exp_name': '$EXP_NAME', 'job_id': $JOB_ID,
    'nodes': '$NODES'.split(','),
    'params': {'filter_groups': True, 'gen_batch_size': 64, 'kl_coef': 0.4, 'bbox_weight': 0.0},
    'log_file': '$LOG_FILE', 'start_time': datetime.now().isoformat(), 'status': 'starting'
}
state['next_experiment_id'] = exp_id + 1
state_file.write_text(json.dumps(state, indent=2, ensure_ascii=False))
print(f'Registered as exp{exp_id}')
EOF
}

# =============================================================================
# 实验 15: Filter Groups + Higher Clip (nodes 31,32)
# =============================================================================
start_exp15() {
    echo ""
    echo "=========================================="
    echo "Exp15: filter_groups=true, clip_high=0.35, kl=0.35"
    echo "Nodes: node31,node32"
    echo "=========================================="

    local NODES="node31,node32"
    local EXP_NAME="exp15_filter_kl0.35_clip0.35"
    local LOG_FILE="/tmp/training_${JOB_ID}_${EXP_NAME}.log"

    # 停止 Ray
    for node in node31 node32; do
        srun --jobid="$JOB_ID" --overlap -w "$node" -n1 ray stop --force 2>/dev/null || true
    done
    sleep 3

    # 直接通过命令行传递 Hydra 参数
    nohup bash examples/video_reasoning/launch_multinode_slurm.sh \
        --jobid "$JOB_ID" --nodes "$NODES" \
        -- \
        algorithm.filter_groups.enable=true \
        algorithm.filter_groups.metric=score \
        algorithm.filter_groups.max_num_gen_batches=5 \
        data.gen_batch_size=48 \
        algorithm.kl_ctrl.kl_coef=0.35 \
        algorithm.clip_ratio_high=0.35 \
        custom_reward_function.reward_kwargs.bbox_weight=0.0 \
        actor_rollout_ref.rollout.top_p=0.7 \
        > "$LOG_FILE" 2>&1 &

    echo "Log: $LOG_FILE"
    echo "PID: $!"

    # 更新状态
    python3 << EOF
import json
from datetime import datetime
from pathlib import Path
state_file = Path('$STATE_FILE')
state = json.loads(state_file.read_text()) if state_file.exists() else {}
exp_id = state.get('next_experiment_id', 15)
state.setdefault('active_jobs', {})
state['active_jobs'][str(exp_id)] = {
    'exp_id': exp_id, 'exp_name': '$EXP_NAME', 'job_id': $JOB_ID,
    'nodes': '$NODES'.split(','),
    'params': {'filter_groups': True, 'gen_batch_size': 48, 'kl_coef': 0.35, 'clip_ratio_high': 0.35, 'bbox_weight': 0.0},
    'log_file': '$LOG_FILE', 'start_time': datetime.now().isoformat(), 'status': 'starting'
}
state['next_experiment_id'] = exp_id + 1
state_file.write_text(json.dumps(state, indent=2, ensure_ascii=False))
print(f'Registered as exp{exp_id}')
EOF
}

# =============================================================================
# 主逻辑
# =============================================================================
start_exp14
sleep 5
start_exp15

echo ""
echo "=========================================="
echo "实验已启动！"
echo "=========================================="
echo "监控: tail -f /tmp/training_21758_exp1*.log"
