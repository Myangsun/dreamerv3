#!/bin/bash
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --job-name=dreamerv3_full
#SBATCH --output=logs/full_%j.out
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# Configuration - Edit these variables as needed
TASK_CONFIG="crafter"           # Options: crafter, atari, dmc_vision, etc.
TASK_NAME=""                     # For Atari: atari_pong, atari_breakout, etc.
TRAIN_STEPS=1000000             # Total training steps (1M is common)
TRAIN_RATIO=32                  # Ratio of training to environment steps
LOG_EVERY=1000                  # Log metrics every N steps
MODEL_SIZE=""                   # Options: "", "size50m", "size200m" (empty = default)

# Load CUDA module
module load cuda/12.4.0

# Verify GPU access
echo "=== Job Information ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

echo "=== Verifying GPU access ==="
python3 -c 'import jax; print("JAX version:", jax.__version__); print("JAX devices:", jax.devices()); print("GPU count:", jax.device_count())'
echo ""

# Build config string
CONFIGS="$TASK_CONFIG"
if [ -n "$MODEL_SIZE" ]; then
    CONFIGS="$CONFIGS $MODEL_SIZE"
fi

# Build task string
TASK_ARG=""
if [ -n "$TASK_NAME" ]; then
    TASK_ARG="--task $TASK_NAME"
fi

# Create logdir with timestamp and config info
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
if [ -n "$TASK_NAME" ]; then
    LOGDIR=./logdir/dreamer/${TASK_NAME}_${TIMESTAMP}
else
    LOGDIR=./logdir/dreamer/${TASK_CONFIG}_${TIMESTAMP}
fi

echo "=== Training Configuration ==="
echo "Config: $CONFIGS"
echo "Task: ${TASK_NAME:-default}"
echo "Training steps: $TRAIN_STEPS"
echo "Train ratio: $TRAIN_RATIO"
echo "Log directory: $LOGDIR"
echo ""

echo "=== Starting DreamerV3 Full Training ==="
python3 dreamerv3/main.py \
  --logdir $LOGDIR \
  --configs $CONFIGS \
  $TASK_ARG \
  --run.train_ratio $TRAIN_RATIO \
  --run.steps $TRAIN_STEPS \
  --run.log_every $LOG_EVERY

EXIT_CODE=$?
echo ""
echo "=== Training Complete ==="
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "Log directory: $LOGDIR"

exit $EXIT_CODE
