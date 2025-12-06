#!/bin/bash
#SBATCH --job-name=dreamer-continue
#SBATCH --partition=cpu-gpu-rtx8000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --array=0-11                       # 12个任务

set -euo pipefail

# =============================================================================
# 继续训练脚本 - 从检查点恢复并训练相同步数
# =============================================================================

# Setup - 使用绝对路径
WORKDIR="/mnt/home/tianyuez/DL-PJ/dreamerv3"
VENV_PATH="${WORKDIR}/.venv"
LOG_ROOT="${WORKDIR}/logs/dmc_ablation_continue"

source "${VENV_PATH}/bin/activate"
cd "${WORKDIR}"

# Set MuJoCo rendering backend
export MUJOCO_GL=egl

# GPU Memory settings - avoid resource conflicts
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.80

# XLA settings for stable cuDNN
export XLA_FLAGS="--xla_gpu_deterministic_ops=true --xla_gpu_strict_conv_algorithm_picker=false"

# =============================================================================
# Task Configuration - 4 encoders × 3 tasks = 12 combinations
# =============================================================================
ENCODERS=(cnn_ae cnn_mae vit_ae vit_mae)
TASKS=(dmc_walker_walk dmc_cheetah_run dmc_hopper_hop)

# 原始运行的时间戳 (用于找到checkpoint)
declare -A ORIGINAL_TIMESTAMPS
ORIGINAL_TIMESTAMPS["dmc_walker_walk_cnn_ae"]="20251206_024217"
ORIGINAL_TIMESTAMPS["dmc_walker_walk_cnn_mae"]="20251206_024217"
ORIGINAL_TIMESTAMPS["dmc_walker_walk_vit_ae"]="20251206_024217"
ORIGINAL_TIMESTAMPS["dmc_walker_walk_vit_mae"]="20251206_024217"
ORIGINAL_TIMESTAMPS["dmc_cheetah_run_cnn_ae"]="20251206_024217"
ORIGINAL_TIMESTAMPS["dmc_cheetah_run_cnn_mae"]="20251206_024217"
ORIGINAL_TIMESTAMPS["dmc_cheetah_run_vit_ae"]="20251206_084243"
ORIGINAL_TIMESTAMPS["dmc_cheetah_run_vit_mae"]="20251206_084243"
ORIGINAL_TIMESTAMPS["dmc_hopper_hop_cnn_ae"]="20251206_084243"
ORIGINAL_TIMESTAMPS["dmc_hopper_hop_cnn_mae"]="20251206_084243"
ORIGINAL_TIMESTAMPS["dmc_hopper_hop_vit_ae"]="20251206_084243"
ORIGINAL_TIMESTAMPS["dmc_hopper_hop_vit_mae"]="20251206_084243"

enc_index=$((SLURM_ARRAY_TASK_ID % ${#ENCODERS[@]}))
task_index=$((SLURM_ARRAY_TASK_ID / ${#ENCODERS[@]}))

ENCODER_TYPE="${ENCODERS[$enc_index]}"
TASK_NAME="${TASKS[$task_index]}"

# 查找原始检查点目录
RUN_KEY="${TASK_NAME}_${ENCODER_TYPE}"
ORIG_TIMESTAMP="${ORIGINAL_TIMESTAMPS[$RUN_KEY]}"
ORIG_LOGDIR="${WORKDIR}/logdir/${RUN_KEY}_${ORIG_TIMESTAMP}"
CHECKPOINT_DIR="${ORIG_LOGDIR}/ckpt"

# 获取最新的checkpoint路径
LATEST_CKPT=$(cat "${CHECKPOINT_DIR}/latest")
FROM_CHECKPOINT="${CHECKPOINT_DIR}/${LATEST_CKPT}"

MAE_FLAGS=""
if [[ "${ENCODER_TYPE}" == cnn_mae || "${ENCODER_TYPE}" == vit_mae ]]; then
  MAE_FLAGS="--mae_loss_scale=1.0 --mae_mask_ratio=0.5"
fi

# =============================================================================
# Logging - 新的日志目录，不覆盖原有数据
# =============================================================================
mkdir -p "${LOG_ROOT}"
LOG_FILE="${LOG_ROOT}/${SLURM_ARRAY_TASK_ID}_${TASK_NAME}_${ENCODER_TYPE}.log"
: > "${LOG_FILE}"

# 新的logdir，使用 _cont 后缀表示继续训练
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR="${WORKDIR}/logdir/${TASK_NAME}_${ENCODER_TYPE}_cont_${TIMESTAMP}"

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
echo ""
echo "=== Checkpoint Info ==="
echo "Original logdir: $ORIG_LOGDIR"
echo "Checkpoint: $FROM_CHECKPOINT"
echo "New logdir: $LOGDIR"
echo ""
echo "Start time: $(date)"
echo ""

echo "=== Verifying GPU ==="
python3 -c 'import jax; print("Devices:", jax.devices())'
echo ""

echo "=== Continuing Training: $TASK_NAME ==="
echo "Loading from checkpoint and training 550K more steps"
echo ""
} 2>&1 | tee -a "${LOG_FILE}"

# =============================================================================
# Training - 从检查点恢复，训练相同步数 (550K)
# 注意: run.steps 是总步数，从checkpoint恢复后会继续到这个总数
# 所以设置为 550K * 2 = 1.1M 步
# =============================================================================
python3 dreamerv3/main.py \
  --logdir "$LOGDIR" \
  --configs dmc_vision size50m \
  --task "${TASK_NAME}" \
  --encoder_type="${ENCODER_TYPE}" \
  --run.envs=8 \
  --run.train_ratio=64 \
  --run.steps=1100000 \
  --run.from_checkpoint="${FROM_CHECKPOINT}" \
  ${MAE_FLAGS} \
  2>&1 | tee -a "${LOG_FILE}"

echo "" | tee -a "${LOG_FILE}"
echo "=== Training Complete ===" | tee -a "${LOG_FILE}"
echo "End time: $(date)" | tee -a "${LOG_FILE}"

