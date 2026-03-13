#!/bin/bash
# =============================================================================
# 启动两个对比实验
# 实验 A (node26,27): 新 prompt + bbox=0.2
# 实验 B (node33,34): bbox=0.0 纯答案 (baseline)
# =============================================================================

set -e
cd /mnt/data/home/zhengshurong/project/verl

echo "=========================================="
echo "启动两个对比实验"
echo "=========================================="

# =============================================================================
# 实验 A: 新 prompt + bbox=0.2 (nodes 26,27)
# =============================================================================
start_exp_a() {
    echo ""
    echo "=========================================="
    echo "Exp A: 新 prompt + bbox=0.2, kl=0.4"
    echo "Nodes: node26,node27 (job 21758)"
    echo "=========================================="

    local JOB_ID=21758
    local NODES="node26,node27"
    local EXP_NAME="exp24_newprompt_bbox0.2"
    local LOG_FILE="/tmp/training_${JOB_ID}_${EXP_NAME}.log"

    # 停止 Ray
    for node in node26 node27; do
        srun --jobid="$JOB_ID" --overlap -w "$node" -n1 ray stop --force 2>/dev/null || true
    done
    sleep 3

    # 启动训练
    ENABLE_FILTER_GROUPS=false \
    BBOX_WEIGHT=0.2 \
    KL_LOSS_COEF=0.4 \
    TOP_P=0.7 \
    EXPERIMENT_NAME="$EXP_NAME" \
    nohup bash examples/video_reasoning/launch_multinode_slurm.sh \
        --jobid "$JOB_ID" --nodes "$NODES" \
        > "$LOG_FILE" 2>&1 &

    echo "Log: $LOG_FILE"
    echo "PID: $!"
}

# =============================================================================
# 实验 B: bbox=0.0 纯答案 (nodes 33,34)
# =============================================================================
start_exp_b() {
    echo ""
    echo "=========================================="
    echo "Exp B: bbox=0.0 纯答案 baseline, kl=0.4"
    echo "Nodes: node33,node34 (job 21719)"
    echo "=========================================="

    local JOB_ID=21719
    local NODES="node33,node34"
    local EXP_NAME="exp25_bbox0.0_baseline"
    local LOG_FILE="/tmp/training_${JOB_ID}_${EXP_NAME}.log"

    # 停止 Ray
    for node in node33 node34; do
        srun --jobid="$JOB_ID" --overlap -w "$node" -n1 ray stop --force 2>/dev/null || true
    done
    sleep 3

    # 启动训练
    ENABLE_FILTER_GROUPS=false \
    BBOX_WEIGHT=0.0 \
    KL_LOSS_COEF=0.4 \
    TOP_P=0.7 \
    EXPERIMENT_NAME="$EXP_NAME" \
    nohup bash examples/video_reasoning/launch_multinode_slurm.sh \
        --jobid "$JOB_ID" --nodes "$NODES" \
        > "$LOG_FILE" 2>&1 &

    echo "Log: $LOG_FILE"
    echo "PID: $!"
}

# =============================================================================
# 主逻辑
# =============================================================================
case "${1:-all}" in
    a|A)
        start_exp_a
        ;;
    b|B)
        start_exp_b
        ;;
    all|*)
        start_exp_a
        sleep 5
        start_exp_b
        ;;
esac

echo ""
echo "=========================================="
echo "实验已启动！"
echo "=========================================="
echo "监控:"
echo "  Exp A: tail -f /tmp/training_21758_exp24_newprompt_bbox0.2.log"
echo "  Exp B: tail -f /tmp/training_21719_exp25_bbox0.0_baseline.log"
