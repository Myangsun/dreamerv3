#!/bin/bash
#SBATCH --job-name=dreamer-ablate
#SBATCH --partition=mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --array=0-11%4
#SBATCH --output=/home/msun14/dreamerv3/logs/dmc_ablation/slurm_%A_%a.out
#SBATCH --exclude=node3207

set -euo pipefail

# Setup
WORKDIR="/home/msun14/dreamerv3"
LOG_ROOT="${WORKDIR}/logs/dmc_ablation"
cd "${WORKDIR}"

# Set MuJoCo rendering backend
export MUJOCO_GL=egl

# =============================================================================
# Task Configuration - 2 encoders × 3 tasks = 6 combinations
# =============================================================================
ENCODERS=(cnn_mae)
TASKS=(dmc_hopper_hop)

enc_index=$((SLURM_ARRAY_TASK_ID % ${#ENCODERS[@]}))
task_index=$((SLURM_ARRAY_TASK_ID / ${#ENCODERS[@]}))

ENCODER_TYPE="${ENCODERS[$enc_index]}"
TASK_NAME="${TASKS[$task_index]}"

MAE_FLAGS=""
if [[ "${ENCODER_TYPE}" == cnn_mae || "${ENCODER_TYPE}" == vit_mae ]]; then
  MAE_FLAGS="--mae_loss_scale=1.0 --mae_mask_ratio=0.5"
fi

# =============================================================================
# Logging
# =============================================================================
mkdir -p "${LOG_ROOT}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR="${WORKDIR}/logdir/${TASK_NAME}_${ENCODER_TYPE}_${TIMESTAMP}"

# =============================================================================
# Job Info
# =============================================================================
echo "=== Job Information ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Task: $TASK_NAME"
echo "Encoder: $ENCODER_TYPE"
echo "Logdir: $LOGDIR"
echo "Start time: $(date)"
echo ""

echo "=== Verifying GPU ==="
python3 -c 'import jax; print("Devices:", jax.devices())'
echo ""

echo "=== Starting DMC Training: $TASK_NAME ==="
echo "Steps: 0.2M (temporary ablation setting)"
echo "Train ratio: 256 (paper setting)"
echo ""

# =============================================================================
# Training
# =============================================================================
python3 dreamerv3/main.py \
  --logdir "$LOGDIR" \
  --configs dmc_vision \
  --task "${TASK_NAME}" \
  --encoder_type="${ENCODER_TYPE}" \
  --run.steps=500000 \
  ${MAE_FLAGS}

echo ""
echo "=== Training Complete ==="
echo "End time: $(date)"
