#!/bin/bash
#SBATCH --job-name=video_dapo
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm_%j.log
#SBATCH --error=logs/slurm_%j.err

set -eo pipefail

# 打印作业信息
echo "===== Slurm Job Info ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "Num Nodes: $SLURM_NNODES"
echo "=========================="

cd /mnt/data/home/zhengshurong/project/verl

# 设置环境变量
export NNODES=$SLURM_NNODES
export N_GPUS=8
export SKIP_VIDEO_CACHE=true

# 获取节点列表
NODELIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))
HEAD_NODE=${NODELIST[0]}
HEAD_IP=$(getent hosts $HEAD_NODE | awk '{print $1}')

echo "Head node: $HEAD_NODE ($HEAD_IP)"

# 激活 conda
source /mnt/data/home/zhengshurong/miniconda3/etc/profile.d/conda.sh
conda activate verl

# 在所有节点上启动 Ray
RAY_PORT=6380

# 停止旧的 Ray 进程
srun --nodes=$SLURM_NNODES --ntasks-per-node=1 bash -c "ray stop 2>/dev/null || true"
sleep 2

# 在 head 节点启动 Ray head
echo "Starting Ray head on $HEAD_NODE..."
srun --nodes=1 --ntasks=1 --nodelist=$HEAD_NODE bash -c "
    source /mnt/data/home/zhengshurong/miniconda3/etc/profile.d/conda.sh
    conda activate verl
    ray start --head --port=$RAY_PORT --num-gpus=8 --disable-usage-stats
" &
sleep 10

# 在 worker 节点启动 Ray worker
if [ $SLURM_NNODES -gt 1 ]; then
    for i in $(seq 1 $((SLURM_NNODES - 1))); do
        WORKER_NODE=${NODELIST[$i]}
        echo "Starting Ray worker on $WORKER_NODE..."
        srun --nodes=1 --ntasks=1 --nodelist=$WORKER_NODE bash -c "
            source /mnt/data/home/zhengshurong/miniconda3/etc/profile.d/conda.sh
            conda activate verl
            ray start --address=$HEAD_IP:$RAY_PORT --num-gpus=8 --disable-usage-stats
        " &
    done
fi
sleep 15

# 等待 Ray 集群就绪
echo "Waiting for Ray cluster..."
python3 -c "
import ray
import time
ray.init(address='$HEAD_IP:$RAY_PORT')
timeout = 300
start = time.time()
while True:
    resources = ray.cluster_resources()
    num_gpus = int(resources.get('GPU', 0))
    expected = $SLURM_NNODES * 8
    print(f'  GPUs: {num_gpus}/{expected}')
    if num_gpus >= expected:
        print('Cluster ready!')
        break
    if time.time() - start > timeout:
        raise RuntimeError('Timeout waiting for cluster')
    time.sleep(5)
ray.shutdown()
"

# 运行训练
echo "Starting training..."
export RAY_ADDRESS=$HEAD_IP:$RAY_PORT
bash examples/video_reasoning/run_video_reasoning_dapo_h200.sh

echo "Training completed!"
