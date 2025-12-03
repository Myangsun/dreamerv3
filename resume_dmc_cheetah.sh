#!/bin/bash
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --job-name=dreamerv3_cheetah_resume
#SBATCH --output=logs/dmc_cheetah_resume_%j.out
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --exclude=node3207

# Resume training from existing checkpoint
# Simply point --logdir to the SAME directory as before!

export MUJOCO_GL=egl

LOGDIR=./logdir/dreamer/dmc_cheetah_run_20251129_211920

echo "=== Resuming DMC Cheetah Run Training ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Logdir: $LOGDIR"
echo "Start time: $(date)"
echo ""

echo "=== Verifying GPU ==="
python3 -c 'import jax; print("Devices:", jax.devices())'
echo ""

# DreamerV3 will automatically:
# 1. Detect existing checkpoint in logdir
# 2. Load model weights and optimizer state
# 3. Continue training from last step

python3 dreamerv3/main.py \
  --logdir $LOGDIR \
  --configs dmc_vision \
  --task dmc_cheetah_run

echo ""
echo "=== Training Complete ==="
echo "End time: $(date)"
