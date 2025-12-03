# DreamerV3 Training Guide

## Quick Start

### 1. Test Run (Verify Setup)
```bash
sbatch train_test.sh
```
- Uses debug config (small network, JAX on CPU)
- Runs for 10,000 steps (~5-10 minutes)
- Verifies installation and GPU access

### 2. Full Training

**Atari 100k (Data-Efficient, 2-4 hours):**
```bash
# Edit train_atari100k.sh to change game if needed
sbatch train_atari100k.sh
```

**Full Atari (51M steps, ~24 hours):**
```bash
# Edit train_atari.sh to set GAME variable
sbatch train_atari.sh
```

**Crafter:**
```bash
sbatch train_crafter.sh
```

## Environment Setup

### Initial Setup (Already Completed)
```bash
# Python version
python3 --version  # 3.11.3

# Load CUDA
module load cuda/12.4.0

# Install dependencies
pip install -r requirements.txt

# Install Atari ROMs
AutoROM --accept-license  # Installs 108 ROMs
```

## Training Scripts

### train_atari100k.sh
- **Task**: Atari 100k benchmark (data-efficient)
- **Steps**: 110,000 steps
- **Duration**: 2-4 hours
- **Config**: `atari100k` with parallel mode
- **Edit**: Change `GAME="atari100k_pong"` to other games

### train_atari.sh
- **Task**: Standard Atari training
- **Steps**: 51M steps (configurable)
- **Duration**: ~24 hours
- **Config**: `atari` with parallel mode
- **Edit**: Change `GAME="atari_pong"` to other games

### train_crafter.sh
- **Task**: Crafter survival game
- **Steps**: 1.1M steps
- **Duration**: 12-24 hours
- **Config**: `crafter`

### train_test.sh
- **Task**: Quick verification test
- **Steps**: 10,000 steps
- **Duration**: 5-10 minutes
- **Config**: `debug` (CPU-only JAX, small model)

## Available Games

### Atari 100k Tasks
- `atari100k_pong`
- `atari100k_breakout`
- `atari100k_qbert`
- See `dreamerv3/configs.yaml` for full list

### Standard Atari Tasks
- `atari_pong`
- `atari_breakout`
- `atari_seaquest`
- `atari_space_invaders`
- `atari_mspacman`
- See `dreamerv3/configs.yaml` for full list

## Monitoring Training

### Check Job Status
```bash
squeue -u $USER
```

### Check Job History
```bash
sacct -u $USER --format=JobID,JobName,State,Elapsed -S today
```

### View Log File
```bash
# While job is running or after completion
tail -f logs/atari100k_<JOBID>.out

# View last 50 lines
tail -50 logs/atari100k_<JOBID>.out
```

### Check Training Metrics
```bash
# View JSONL metrics
cat ./logdir/dreamer/<run_name>/metrics.jsonl | tail -20

# Count steps completed
grep -c "step" ./logdir/dreamer/<run_name>/metrics.jsonl
```

### Cancel a Job
```bash
scancel <JOBID>
```

## SLURM Job Management

### Job States
- **PD** (Pending): Waiting for GPU resources
- **R** (Running): Currently executing
- **CG** (Completing): Finishing up
- **CD** (Completed): Finished successfully
- **F** (Failed): Job failed
- **CA** (Cancelled): User cancelled

### Check Specific Job
```bash
squeue -j <JOBID>
sacct -j <JOBID> --format=JobID,State,ExitCode,Elapsed
```

## Training Configuration

### Key Parameters

**From command line:**
```bash
--logdir <path>           # Where to save checkpoints and logs
--configs <config_name>   # Config preset (atari, atari100k, crafter, etc.)
--task <task_name>        # Specific task/game
--script parallel         # Run envs in separate processes (REQUIRED for Atari!)
--run.steps <number>      # Total training steps
--run.train_ratio <num>   # Gradient updates per env step
--run.log_every <num>     # Log frequency
```

**Example:**
```bash
python3 dreamerv3/main.py \
  --logdir ./logdir/my_experiment \
  --configs atari100k \
  --task atari100k_breakout \
  --script parallel \
  --run.steps 150000 \
  --run.log_every 1000
```

### Config Files
- **Main config**: `dreamerv3/configs.yaml`
- **Defaults**: See `defaults:` section in configs.yaml
- **Task-specific**: See individual config sections (atari, crafter, etc.)

## Resource Requirements

| Task | GPU | Memory | CPUs | Time |
|------|-----|--------|------|------|
| Test run | Any | 16GB | 4 | 30min |
| Atari 100k | 1x GPU | 32GB | 8 | 2-4h |
| Full Atari | 1x GPU | 32GB | 8 | 24h |
| Crafter | 1x GPU | 32GB | 8 | 12-24h |

**Available GPUs on cluster:**
- H200 (141GB VRAM)
- H100 (80GB VRAM)
- L40S (48GB VRAM)

### Request Specific GPU Type
Edit SBATCH header in training script:
```bash
#SBATCH --gres=gpu:h200:1  # Request H200
#SBATCH --gres=gpu:h100:1  # Request H100
#SBATCH --gres=gpu:1       # Any available GPU
```

## Resuming Training

To continue a stopped run, use the same `--logdir`:

```bash
python3 dreamerv3/main.py \
  --logdir ./logdir/dreamer/atari100k_pong_20251125_194315 \
  --configs atari100k \
  --task atari100k_pong \
  --script parallel
```

The agent will:
- Load the latest checkpoint
- Resume from where it left off
- Continue logging to the same directory

## File Locations

```
/home/msun14/dreamerv3/
├── train_atari100k.sh      # Atari 100k training script
├── train_atari.sh          # Full Atari training script
├── train_crafter.sh        # Crafter training script
├── train_test.sh           # Quick test script
├── logs/                   # SLURM output logs
│   └── atari100k_<jobid>.out
├── logdir/                 # Training outputs (gitignored)
│   └── dreamer/
│       └── <run_name>/
│           ├── checkpoint/
│           ├── metrics.jsonl
│           └── scores.jsonl
└── dreamerv3/
    ├── main.py            # Main entry point
    ├── configs.yaml       # All configurations
    └── embodied/
        └── envs/
            └── atari.py   # Atari env (with CUDA fix)
```

## Troubleshooting

### Atari Segmentation Fault (FIXED)

**Problem**: Atari training crashed with `Segmentation fault (core dumped)` when using JAX on GPU.

**Root Cause**: CUDA/ALE incompatibility
- ALE (Atari Learning Environment version 0.8.1) has fundamental incompatibility with JAX using CUDA 12.4 on this cluster
- Both JAX and ALE try to access CUDA, causing memory conflicts
- Results in immediate segmentation fault during environment initialization

**Solution**: Use JAX on CPU for Atari training
- Add `--jax.platform cpu` flag to training command
- JAX runs on CPU (no CUDA), ALE can initialize without conflicts
- Training works correctly but slower than GPU

**Performance Impact**:
- CPU training is significantly slower than GPU
- Atari 100k: ~4-8 hours (vs 2-4 hours on GPU if it worked)
- Still functional for benchmarking and research

**Files Modified**:
- [train_atari100k.sh:50](train_atari100k.sh#L50) - Added `--jax.platform cpu`
- [train_atari.sh:49](train_atari.sh#L49) - Added `--jax.platform cpu`
- [embodied/envs/atari.py:51-78](embodied/envs/atari.py#L51-L78) - CUDA isolation during ALE init (defense in depth)

## Tips

1. **Start small**: Always run `train_test.sh` first to verify setup
2. **Monitor early**: Check logs after 5 minutes to catch errors quickly
3. **Use parallel mode for Atari**: Required to avoid CUDA crashes
4. **Save logdir path**: You'll need it to resume or analyze results
5. **Check queue times**: Run `squeue -p mit_normal_gpu` to see cluster load

## Getting Help

- **DreamerV3 Paper**: [arxiv.org/abs/2301.04104](https://arxiv.org/abs/2301.04104)
- **Original Repo**: [github.com/danijar/dreamerv3](https://github.com/danijar/dreamerv3)
- **Your Fork**: [github.com/Myangsun/dreamerv3](https://github.com/Myangsun/dreamerv3)
- **Configs**: `dreamerv3/configs.yaml`
- **SLURM Docs**: Your cluster's documentation
