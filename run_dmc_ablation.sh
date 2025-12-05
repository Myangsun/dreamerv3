#!/bin/bash
#SBATCH --job-name=dreamer-ablate
#SBATCH --partition=YOUR_PARTITION_HERE    # 修改为你的分区
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --array=0-11%4                     # 12个任务 (4 encoders × 3 tasks)

set -euo pipefail

# =============================================================================
# 配置说明 (请根据你的环境修改)
# =============================================================================
# 
# 1. --partition: 修改为你集群的 GPU 分区名
# 
# 2. 安装依赖:
#    python3 -m venv .venv
#    source .venv/bin/activate
#    pip install -U pip wheel
#    pip install jax[cuda12]==0.4.33
#    pip install nvidia-cudnn-cu12==9.16.0.29
#    pip install -r requirements.txt
#    pip install dm_control mujoco opencv-python portal einops
#
# =============================================================================

# Setup
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${WORKDIR}/.venv"
LOG_ROOT="${WORKDIR}/logs/dmc_ablation"

source "${VENV_PATH}/bin/activate"
cd "${WORKDIR}"

# Set MuJoCo rendering backend
export MUJOCO_GL=egl

# =============================================================================
# Task Configuration - 4 encoders × 3 tasks = 12 combinations
# =============================================================================
ENCODERS=(cnn_ae cnn_mae vit_ae vit_mae)
TASKS=(dmc_walker_walk dmc_cheetah_run dmc_hopper_hop)

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
LOG_FILE="${LOG_ROOT}/${SLURM_ARRAY_TASK_ID}_${TASK_NAME}_${ENCODER_TYPE}.log"
: > "${LOG_FILE}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR="${WORKDIR}/logdir/${TASK_NAME}_${ENCODER_TYPE}_${TIMESTAMP}"

# =============================================================================
# Job Info
# =============================================================================
{
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
echo "Steps: 1.1M (paper setting)"
echo "Train ratio: 256 (paper setting)"
echo ""
} 2>&1 | tee -a "${LOG_FILE}"

# =============================================================================
# Training (使用默认的 batch_size=16, batch_length=64)
# =============================================================================
python3 dreamerv3/main.py \
  --logdir "$LOGDIR" \
  --configs dmc_vision \
  --task "${TASK_NAME}" \
  --encoder_type="${ENCODER_TYPE}" \
  ${MAE_FLAGS} \
  2>&1 | tee -a "${LOG_FILE}"

echo "" | tee -a "${LOG_FILE}"
echo "=== Training Complete ===" | tee -a "${LOG_FILE}"
echo "End time: $(date)" | tee -a "${LOG_FILE}"
