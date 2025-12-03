#!/bin/bash
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --job-name=dreamerv3_hopper_resume
#SBATCH --output=logs/dmc_hopper_resume_%j.out
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --exclude=node3207

# Don't load system CUDA - JAX 0.5.0 bundles its own CUDA libraries
# module load cuda/12.4.0

# Set MuJoCo rendering backend
export MUJOCO_GL=egl

TASK="dmc_hopper_hop"

# UPDATE THIS PATH after first training run creates a logdir
# Example: LOGDIR=./logdir/dreamer/dmc_hopper_hop_20251201_143000
LOGDIR=./logdir/dreamer/dmc_hopper_hop_20251201_201231

echo "=== Job Information ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Task: $TASK"
echo "Start time: $(date)"
echo ""

echo "=== Resuming from checkpoint ==="
echo "Logdir: $LOGDIR"

if [ ! -d "$LOGDIR" ]; then
    echo "ERROR: Logdir does not exist: $LOGDIR"
    echo "Please update LOGDIR in this script to point to your existing training directory"
    exit 1
fi

echo "=== Verifying GPU ==="
python3 -c 'import jax; print("Devices:", jax.devices())'
echo ""

echo "=== Resuming DMC Training: $TASK ==="
echo ""

python3 dreamerv3/main.py \
  --logdir $LOGDIR \
  --configs dmc_vision \
  --task $TASK

echo ""
echo "=== Training Complete ==="
echo "End time: $(date)"
