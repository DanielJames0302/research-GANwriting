#!/bin/bash

#SBATCH --output=logs/slurm-training-%j.out
#SBATCH --error=logs/slurm-training-%j.err
#SBATCH --job-name=cs4243-ganwriting
#SBATCH --mail-user=e1129786@comp.nus.edu.sg
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:a100-40:1
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

# ---------- user-tunable knobs via --export or defaults ----------
TRAIN_MODE=${TRAIN_MODE:-scratch}           # sbatch --export=TRAIN_MODE=pretrain,START_EPOCH=1000 job_train.sbatch
START_EPOCH=${START_EPOCH:-}                # Auto-detected if not provided in pretrain mode
CAPTCHA_DIR=${CAPTCHA_DIR:-test/correct_test}  # Directory containing captcha images
LOG_NAME=${LOG_NAME:-exp1}                  # Experiment name for logging

# ---------- environment ----------
set -euo pipefail
export PYTHONUNBUFFERED=1                   # flush prints immediately
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# If using conda:
# shellcheck disable=SC1090
# source /path/to/your/conda/bin/activate
# conda activate your_env_name

# (Alternative: venv)
# source /path/to/your/venv/bin/activate

# Optional NCCL tweaks (uncomment if needed)
# export NCCL_DEBUG=WARN
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=0

# ---------- go to code ----------
# Change to the workspace root directory
WORKSPACE_DIR="/home/m/man0302/CS4243/project/research-GANwriting"
cd "$WORKSPACE_DIR" || { echo "Error: Cannot cd to $WORKSPACE_DIR"; exit 1; }

echo "Node: $(hostname)"
echo "=== Slurm GANwriting Training Script ==="
echo "Starting training at $(date)"
echo "Running on host: $(hostname)"
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits || echo "No GPU detected"

# Make sure logs dir exists (for SBATCH -o/-e above)
mkdir -p logs

echo "== SLURM JOB INFO =="
echo "Job ID     : $SLURM_JOB_ID"
echo "Node list  : $SLURM_NODELIST"
echo "GPUs       : $SLURM_GPUS_ON_NODE"
echo "CPUs/Task  : $SLURM_CPUS_PER_TASK"
echo "Train Mode : $TRAIN_MODE"
echo "Log name   : $LOG_NAME"
echo "Captcha Dir: $CAPTCHA_DIR"
echo "===================="
pwd

# --- Count visible CUDA devices ---
NUM_GPUS=$(python3 - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)

echo "Detected $NUM_GPUS visible CUDA device(s)"

# ---------- determine training mode and epoch ----------
EPOCH_ARG=""

if [ "$TRAIN_MODE" = "scratch" ]; then
    echo "=== Training from scratch ==="
    echo "Cleaning old model files and images..."
    python3 clean.py
    EPOCH_ARG="0"
elif [ "$TRAIN_MODE" = "pretrain" ]; then
    echo "=== Resuming training from checkpoint ==="
    if [ -n "$START_EPOCH" ]; then
        EPOCH_ARG="$START_EPOCH"
        echo "Using specified epoch: $START_EPOCH"
    else
        # Auto-detect latest epoch from save_weights directory
        if [ -d "save_weights" ]; then
            LAST_EPOCH=$(find save_weights/ -name "contran-*.model" 2>/dev/null | \
                sed "s/save_weights\/contran-\([0-9][0-9]*\)\.model/\1/" | \
                sort -n | \
                tail -n 1)
            if [ -n "$LAST_EPOCH" ]; then
                EPOCH_ARG="$LAST_EPOCH"
                echo "Auto-detected latest epoch: $LAST_EPOCH"
            else
                echo "Error: No model files found in save_weights/ directory"
                exit 1
            fi
        else
            echo "Error: save_weights/ directory does not exist"
            exit 1
        fi
    fi
else
    echo "Error: Invalid TRAIN_MODE='$TRAIN_MODE'. Must be 'scratch' or 'pretrain'"
    exit 1
fi

# ---------- run training ----------
echo "=== Starting GANwriting training ==="
echo "Training mode: $TRAIN_MODE"
echo "Starting epoch: $EPOCH_ARG"
echo "Dataset: captcha"
echo "Captcha directory: $CAPTCHA_DIR"
date

if srun python3 -u main_run.py "$EPOCH_ARG" --dataset captcha --captcha_dir "$CAPTCHA_DIR"; then
    TRAIN_EXIT=0
    echo "=== Training succeeded ==="
else
    TRAIN_EXIT=$?
    echo "=== Training failed with code $TRAIN_EXIT ==="
    exit $TRAIN_EXIT
fi

date
echo "=== Training completed at $(date) ==="

