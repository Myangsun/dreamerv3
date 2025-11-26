#!/bin/bash
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --job-name=dreamerv3_atari
#SBATCH --output=logs/atari_%j.out
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# Atari game to train on - CHANGE THIS to your desired game
# Examples: atari_pong, atari_breakout, atari_seaquest, atari_space_invaders
GAME="atari_pong"

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
echo "Game: $GAME"
echo "Start time: $(date)"
echo ""

echo "=== Verifying GPU ==="
python3 -c 'import jax; print("Devices:", jax.devices())'
echo ""

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR=./logdir/dreamer/${GAME}_${TIMESTAMP}

echo "=== Starting Atari Training: $GAME ==="
echo "Logdir: $LOGDIR"
echo ""

python3 dreamerv3/main.py \
  --logdir $LOGDIR \
  --configs atari \
  --task $GAME \
  --run.train_ratio 32 \
  --jax.platform cpu

echo ""
echo "=== Training Complete ==="
echo "End time: $(date)"
