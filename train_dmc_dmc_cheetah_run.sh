#!/bin/bash
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --job-name=dreamerv3_dmc_cheetah_run
#SBATCH --output=logs/dmc_cheetah_run_%j.out
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --exclude=node3207

# Don't load system CUDA - JAX 0.5.0 bundles its own CUDA libraries
# module load cuda/12.4.0

# Set MuJoCo rendering backend
export MUJOCO_GL=egl

TASK="dmc_cheetah_run"

echo "=== Job Information ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Task: $TASK"
echo "Start time: $(date)"
echo ""

echo "=== Verifying GPU ==="
python3 -c 'import jax; print("Devices:", jax.devices())'
echo ""

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR=./logdir/dreamer/${TASK}_${TIMESTAMP}

echo "=== Starting DMC Training: $TASK ==="
echo "Steps: 1.1M (paper setting)"
echo "Train ratio: 256 (paper setting)"
echo "Logdir: $LOGDIR"
echo ""

python3 dreamerv3/main.py \
  --logdir $LOGDIR \
  --configs dmc_vision \
  --task $TASK

echo ""
echo "=== Training Complete ==="
echo "End time: $(date)"
