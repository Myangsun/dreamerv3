#!/bin/bash
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --job-name=dreamerv3_atari100k
#SBATCH --output=logs/atari100k_%j.out
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# Atari 100k game to train on
# Examples: atari100k_pong, atari100k_breakout, atari100k_qbert
GAME="atari100k_pong"

module load cuda/12.4.0

# Prevent JAX from preallocating all GPU memory (helps avoid CUDA conflicts with ALE)
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75

# Force ALE to use dummy video driver (no GPU rendering)
export SDL_VIDEODRIVER=dummy

# Set Atari ROM path for ale-py 0.8.1
export ALE_ROM_PATH=~/.local/lib/python3.11/site-packages/AutoROM/roms

echo "=== Job Information ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Game: $GAME (100k benchmark)"
echo "Start time: $(date)"
echo ""

echo "=== Verifying GPU ==="
python3 -c 'import jax; print("Devices:", jax.devices())'
echo ""

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR=./logdir/dreamer/${GAME}_${TIMESTAMP}

echo "=== Starting Atari 100k Training: $GAME ==="
echo "This is the DATA-EFFICIENT benchmark (110k steps)"
echo "Expected duration: 2-4 hours"
echo "Logdir: $LOGDIR"
echo ""

python3 dreamerv3/main.py \
  --logdir $LOGDIR \
  --configs atari100k \
  --task $GAME \
  --jax.platform cpu

echo ""
echo "=== Training Complete ==="
echo "End time: $(date)"
echo "This was Atari 100k (sample-efficient benchmark)"
