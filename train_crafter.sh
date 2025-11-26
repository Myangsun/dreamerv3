#!/bin/bash
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --job-name=dreamerv3_crafter
#SBATCH --output=logs/crafter_%j.out
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

module load cuda/12.4.0

echo "=== Job Information ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

echo "=== Verifying GPU ==="
python3 -c 'import jax; print("Devices:", jax.devices())'
echo ""

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR=./logdir/dreamer/crafter_${TIMESTAMP}

echo "=== Starting Crafter Training ==="
echo "Logdir: $LOGDIR"
echo ""

python3 dreamerv3/main.py \
  --logdir $LOGDIR \
  --configs crafter \
  --run.train_ratio 32

echo ""
echo "=== Training Complete ==="
echo "End time: $(date)"
