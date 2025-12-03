#!/bin/bash
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --job-name=dreamerv3_humanoid_resume
#SBATCH --output=logs/dmc_humanoid_resume_%j.out
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --exclude=node3207

export MUJOCO_GL=egl

LOGDIR=./logdir/dreamer/dmc_humanoid_walk_20251130_032240

echo "=== Resuming DMC Humanoid Walk Training ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Logdir: $LOGDIR"
echo "Start time: $(date)"
echo ""

echo "=== Verifying GPU ==="
python3 -c 'import jax; print("Devices:", jax.devices())'
echo ""

python3 dreamerv3/main.py \
  --logdir $LOGDIR \
  --configs dmc_vision \
  --task dmc_humanoid_walk

echo ""
echo "=== Training Complete ==="
echo "End time: $(date)"
