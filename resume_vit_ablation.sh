#!/bin/bash
#SBATCH --job-name=vit-ablate-resume
#SBATCH --partition=mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/home/msun14/dreamerv3/logs/dmc_ablation/resume_slurm_%A_%a.out
#SBATCH --exclude=node3207

set -euo pipefail

# Setup
WORKDIR="/home/msun14/dreamerv3"
cd "${WORKDIR}"

# Set MuJoCo rendering backend
export MUJOCO_GL=egl

# =============================================================================
# Resume from existing logdirs
# =============================================================================
LOGDIRS=(
  "/home/msun14/dreamerv3/logdir/dmc_walker_walk_vit_ae_20251205_135115"
  "/home/msun14/dreamerv3/logdir/dmc_cheetah_run_vit_ae_20251205_135115"
  "/home/msun14/dreamerv3/logdir/dmc_hopper_hop_vit_ae_20251205_195206"
)

TASKS=(dmc_walker_walk dmc_cheetah_run dmc_hopper_hop)

LOGDIR="${LOGDIRS[$SLURM_ARRAY_TASK_ID]}"
TASK_NAME="${TASKS[$SLURM_ARRAY_TASK_ID]}"

# =============================================================================
# Job Info
# =============================================================================
echo "=== Resuming ViT Ablation Training ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Task: $TASK_NAME"
echo "Encoder: vit_ae"
echo "Logdir: $LOGDIR"
echo "Start time: $(date)"
echo ""

echo "=== Verifying GPU ==="
python3 -c 'import jax; print("Devices:", jax.devices())'
echo ""

echo "=== Checking existing checkpoint ==="
ls -la "${LOGDIR}/ckpt/" 2>/dev/null || echo "No checkpoint dir found"
echo ""

# =============================================================================
# Resume Training
# DreamerV3 will automatically:
# 1. Detect existing checkpoint in logdir
# 2. Load model weights and optimizer state
# 3. Continue training from last step
# =============================================================================
python3 dreamerv3/main.py \
  --logdir "$LOGDIR" \
  --configs dmc_vision \
  --task "${TASK_NAME}" \
  --encoder_type=vit_ae

echo ""
echo "=== Training Complete ==="
echo "End time: $(date)"
