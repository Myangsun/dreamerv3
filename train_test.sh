#!/bin/bash
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --job-name=dreamerv3_test
#SBATCH --output=logs/test_%j.out
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Load CUDA module
module load cuda/12.4.0

# Verify GPU access
echo "=== Verifying GPU access ==="
python3 -c 'import jax; print("JAX devices:", jax.devices()); print("GPU count:", jax.device_count())'
echo ""

# Run DreamerV3 test
echo "=== Starting DreamerV3 test run ==="
python3 dreamerv3/main.py \
  --logdir ~/logdir/dreamer/test_$(date +%Y%m%d_%H%M%S) \
  --configs debug crafter \
  --run.train_ratio 32 \
  --run.steps 10000 \
  --run.log_every 500

echo "=== Training complete ==="
