#!/bin/bash
#SBATCH --job-name=dreamer-ablate
#SBATCH --partition=YOUR_PARTITION_HERE    # 修改为你的分区，如 gpu, normal_gpu 等
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32                 # 与 run.envs 保持一致
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --array=0-11%4                     # 12个任务 (4 encoders × 3 tasks)，并发4个

set -euo pipefail

# =============================================================================
# 配置说明 (请根据你的环境修改)
# =============================================================================
# 
# 1. --partition: 修改为你集群的 GPU 分区名
# 
# 2. GPU 显存建议:
#    - 48GB GPU (如 RTX 8000, A6000): batch_size=64, batch_length=64
#    - 32GB GPU (如 V100-32G):        batch_size=32, batch_length=64
#    - 24GB GPU (如 RTX 3090, A5000): batch_size=16, batch_length=64
#    - 16GB GPU (如 V100-16G):        batch_size=8,  batch_length=32
#
# 3. JAX/cuDNN 版本兼容性:
#    - RTX 系列 / Ampere+: JAX 0.4.33 + cuDNN 9.x (推荐)
#    - V100 (Volta):       需要 JAX <= 0.4.20 + cuDNN 8.x (CUDA 版本敏感)
#
# 4. 安装依赖:
#    python3 -m venv .venv
#    source .venv/bin/activate
#    pip install -U pip wheel
#    pip install jax[cuda12]==0.4.33
#    pip install nvidia-cudnn-cu12==9.16.0.29
#    pip install -r requirements.txt
#    pip install dm_control mujoco opencv-python portal
#
# =============================================================================

# =============================================================================
# Setup
# =============================================================================
module purge 2>/dev/null || true

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${WORKDIR}/.venv"
LOG_ROOT="${WORKDIR}/logs/dmc_ablation"

source "${VENV_PATH}/bin/activate"
cd "${WORKDIR}"

# Environment
export MUJOCO_GL=egl
export SDL_VIDEODRIVER=dummy

# =============================================================================
# XLA FLAGS - 根据 GPU 调整
# =============================================================================
# 通用设置，适用于大多数现代 GPU
export XLA_FLAGS="--xla_gpu_deterministic_ops=true --xla_gpu_strict_conv_algorithm_picker=false"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

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
# Batch Size - 根据 GPU 显存调整
# =============================================================================
# 默认配置 (48GB GPU)
BATCH_SIZE=64
BATCH_LENGTH=64
NUM_ENVS=32

# 如果显存不足，取消下面的注释并调整:
# BATCH_SIZE=16
# BATCH_LENGTH=64
# NUM_ENVS=8

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

echo "=== Training Config ==="
echo "Batch size: $BATCH_SIZE"
echo "Batch length: $BATCH_LENGTH"
echo "Num envs: $NUM_ENVS"
echo ""

echo "=== GPU Info ==="
nvidia-smi --query-gpu=driver_version,name,memory.total --format=csv,noheader
echo ""

echo "=== JAX Version ==="
python3 -c '
import jax
import jaxlib
print("JAX version:", jax.__version__)
print("jaxlib version:", jaxlib.__version__)
print("Devices:", jax.devices())
'
echo ""

echo "=== Starting Training ==="
echo ""
} 2>&1 | tee -a "${LOG_FILE}"

# =============================================================================
# Training
# =============================================================================
python3 dreamerv3/main.py \
  --logdir "$LOGDIR" \
  --configs dmc_vision \
  --task "${TASK_NAME}" \
  --encoder_type="${ENCODER_TYPE}" \
  --batch_size=${BATCH_SIZE} \
  --batch_length=${BATCH_LENGTH} \
  --run.envs=${NUM_ENVS} \
  --run.steps=1.1e6 \
  ${MAE_FLAGS} \
  2>&1 | tee -a "${LOG_FILE}"

echo "" | tee -a "${LOG_FILE}"
echo "=== Training Complete ===" | tee -a "${LOG_FILE}"
echo "End time: $(date)" | tee -a "${LOG_FILE}"
