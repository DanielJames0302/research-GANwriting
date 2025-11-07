#!/bin/bash

#SBATCH --output=logs/slurm-visualize-%j.out
#SBATCH --error=logs/slurm-visualize-%j.err
#SBATCH --job-name=cs4243-visualize
#SBATCH --mail-user=e1129786@comp.nus.edu.sg
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:a100-40:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# ---------- user-tunable knobs via --export or defaults ----------
CHECKPOINT=${CHECKPOINT:-}                    # Required: path to checkpoint file
TEXT=${TEXT:-hello}                           # Text to generate (default: hello)
DATASET=${DATASET:-captcha}                   # Dataset type: iam or captcha
CAPTCHA_DIR=${CAPTCHA_DIR:-test/correct_test} # Directory containing captcha images
OUTPUT=${OUTPUT:-}                            # Output PNG file path (optional)
USE_DUMMY_REF=${USE_DUMMY_REF:-false}         # Use dummy reference images

# ---------- environment ----------
set -euo pipefail
export PYTHONUNBUFFERED=1                     # flush prints immediately
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# If using conda:
# shellcheck disable=SC1090
# source /path/to/your/conda/bin/activate
# conda activate your_env_name

# (Alternative: venv)
# source /path/to/your/venv/bin/activate

# ---------- go to code ----------
# Change to the workspace root directory
WORKSPACE_DIR="/home/m/man0302/CS4243/project/research-GANwriting"
cd "$WORKSPACE_DIR" || { echo "Error: Cannot cd to $WORKSPACE_DIR"; exit 1; }

echo "Node: $(hostname)"
echo "=== Slurm GANwriting Visualization Script ==="
echo "Starting visualization at $(date)"
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
echo "Checkpoint : $CHECKPOINT"
echo "Text       : $TEXT"
echo "Dataset    : $DATASET"
echo "Output     : $OUTPUT"
echo "===================="
pwd

# --- Validate checkpoint path ---
if [ -z "$CHECKPOINT" ]; then
    echo "Error: CHECKPOINT environment variable is required"
    echo "Usage: sbatch --export=CHECKPOINT=path/to/checkpoint.model,TEXT='your text' job_visualize.sbatch"
    exit 1
fi

if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint file not found: $CHECKPOINT"
    exit 1
fi

# --- Count visible CUDA devices ---
NUM_GPUS=$(python3 - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)

echo "Detected $NUM_GPUS visible CUDA device(s)"

# ---------- build command arguments ----------
VISUALIZE_ARGS=(
    "$CHECKPOINT"
    --text "$TEXT"
    --dataset "$DATASET"
)

if [ "$DATASET" = "captcha" ]; then
    VISUALIZE_ARGS+=(--captcha_dir "$CAPTCHA_DIR")
fi

if [ -n "$OUTPUT" ]; then
    VISUALIZE_ARGS+=(--output "$OUTPUT")
fi

if [ "$USE_DUMMY_REF" = "true" ]; then
    VISUALIZE_ARGS+=(--use_dummy_ref)
fi

# ---------- run visualization ----------
echo "=== Starting image generation ==="
echo "Checkpoint: $CHECKPOINT"
echo "Text: $TEXT"
echo "Dataset: $DATASET"
date

if srun python3 -u visualize_checkpoint.py "${VISUALIZE_ARGS[@]}"; then
    VISUALIZE_EXIT=0
    echo "=== Visualization succeeded ==="
else
    VISUALIZE_EXIT=$?
    echo "=== Visualization failed with code $VISUALIZE_EXIT ==="
    exit $VISUALIZE_EXIT
fi

date
echo "=== Visualization completed at $(date) ==="

