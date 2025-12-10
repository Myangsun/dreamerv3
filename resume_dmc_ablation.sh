#!/bin/bash
#SBATCH --job-name=dreamer-ablate-resume
#SBATCH --partition=mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --array=0-11%4
#SBATCH --output=/home/msun14/dreamerv3/logs/dmc_ablation/resume_slurm_%A_%a.out
#SBATCH --exclude=node3207

set -euo pipefail

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
WORKDIR="/home/msun14/dreamerv3"
LOG_ROOT="${WORKDIR}/logs/dmc_ablation"
TARGET_STEPS=1100000
cd "${WORKDIR}"
mkdir -p "${LOG_ROOT}"

# Set MuJoCo rendering backend
export MUJOCO_GL=egl

# -----------------------------------------------------------------------------
# Task Configuration - mirrors run_dmc_ablation.sh
# -----------------------------------------------------------------------------
ENCODERS=(cnn_mae vit_mae vit_ae)
TASKS=(dmc_walker_walk dmc_cheetah_run dmc_hopper_hop)

total_encoders=${#ENCODERS[@]}
total_tasks=${#TASKS[@]}
total_combos=$((total_encoders * total_tasks))

if (( SLURM_ARRAY_TASK_ID >= total_combos )); then
  echo "SLURM task id ${SLURM_ARRAY_TASK_ID} is outside ${total_combos} combos. Skipping."
  exit 0
fi

enc_index=$((SLURM_ARRAY_TASK_ID % total_encoders))
task_index=$((SLURM_ARRAY_TASK_ID / total_encoders))

ENCODER_TYPE="${ENCODERS[$enc_index]}"
TASK_NAME="${TASKS[$task_index]}"

MAE_FLAGS=""
if [[ "${ENCODER_TYPE}" == cnn_mae || "${ENCODER_TYPE}" == vit_mae ]]; then
  MAE_FLAGS="--mae_loss_scale=1.0 --mae_mask_ratio=0.5"
fi

# -----------------------------------------------------------------------------
# Locate Logdir
# -----------------------------------------------------------------------------
LOGDIR=$(ls -td "${WORKDIR}/logdir/${TASK_NAME}_${ENCODER_TYPE}"_* 2>/dev/null | head -n 1 || true)

if [[ -z "${LOGDIR}" || ! -d "${LOGDIR}" ]]; then
  echo "ERROR: Could not find existing logdir for ${TASK_NAME} (${ENCODER_TYPE})."
  echo "Expected pattern: ${WORKDIR}/logdir/${TASK_NAME}_${ENCODER_TYPE}_TIMESTAMP"
  exit 1
fi

# -----------------------------------------------------------------------------
# Job Info
# -----------------------------------------------------------------------------
echo "=== Resume Job Information ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Task: ${TASK_NAME}"
echo "Encoder: ${ENCODER_TYPE}"
echo "Logdir: ${LOGDIR}"
echo "Target steps: ${TARGET_STEPS}"
echo "Start time: $(date)"
echo ""

echo "=== Verifying GPU ==="
python3 -c 'import jax; print("Devices:", jax.devices())'
echo ""

echo "=== Resuming DMC Training: ${TASK_NAME} (${ENCODER_TYPE}) ==="
echo ""

# -----------------------------------------------------------------------------
# Resume Training
# -----------------------------------------------------------------------------
python3 dreamerv3/main.py \
  --logdir "${LOGDIR}" \
  --configs dmc_vision \
  --task "${TASK_NAME}" \
  --encoder_type="${ENCODER_TYPE}" \
  --run.steps="${TARGET_STEPS}" \
  ${MAE_FLAGS}

echo ""
echo "=== Resume Complete ==="
echo "End time: $(date)"
