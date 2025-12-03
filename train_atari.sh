#!/bin/bash
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --job-name=dreamerv3_atari
#SBATCH --output=logs/atari_%j.out
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --exclude=node3207

# Atari game to train on - CHANGE THIS to your desired game
# Examples: atari_pong, atari_breakout, atari_seaquest, atari_space_invaders
GAME="atari_pong"

# Don't load system CUDA - JAX 0.5.0 bundles its own CUDA libraries
# module load cuda/12.4.0

# Force ALE to use dummy video driver (no GPU rendering)
export SDL_VIDEODRIVER=dummy

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
  --script parallel

echo ""
echo "=== Training Complete ==="
echo "End time: $(date)"
