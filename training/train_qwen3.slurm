#!/bin/bash
#SBATCH --job-name=qwen3-finetune
#SBATCH --output=logs/qwen3-%j.out
#SBATCH --error=logs/qwen3-%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G

# ============================================================================
# Dyslexic Writer - Qwen3 Model Fine-tuning on Nibi Cluster
#
# Trains 4 Qwen3 models for local inference:
#   - Qwen3-0.6B  (~400MB quantized) - low-end devices
#   - Qwen3-1.7B  (~1GB quantized)   - default for most
#   - Qwen3-4B    (~2.5GB quantized) - good balance
#   - Qwen3-8B    (~5GB quantized)   - premium quality
#
# Uses Alliance Canada wheels and PyTorch's native SDPA for Flash Attention
# Expected runtime: ~8-12 hours total on H100
# ============================================================================

echo "=========================================="
echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="

# Create logs directory
mkdir -p logs

# Load modules (DRAC best practice)
module load python/3.11 cuda/12.6 arrow/21.0.0

echo ""
echo "Loaded modules:"
module list

# Check GPU
echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv

# Create virtual environment in fast local storage
VENV_DIR=$SLURM_TMPDIR/venv
echo ""
echo "Creating virtual environment in $VENV_DIR..."
virtualenv --no-download $VENV_DIR
source $VENV_DIR/bin/activate

# Upgrade pip using Alliance wheel
pip install --no-index --upgrade pip

# Install dependencies from Alliance wheels
echo ""
echo "Installing dependencies from Alliance wheels..."
pip install --no-index torch torchvision
pip install --no-index transformers datasets trl accelerate peft
pip install --no-index sentencepiece

# Install any packages not available as wheels from PyPI
echo ""
echo "Installing additional packages from PyPI if needed..."
pip install tokenizers --upgrade 2>/dev/null || true

# Show installed versions
echo ""
echo "Installed packages:"
pip list | grep -E "torch|transformers|datasets|trl|accelerate"

# Change to training directory
cd $SLURM_SUBMIT_DIR

# Train all Qwen3 models
echo ""
echo "=========================================="
echo "Starting Qwen3 training..."
echo "=========================================="
python finetune_qwen3.py --model all --output-dir ./outputs_qwen3 --epochs 3

echo ""
echo "=========================================="
echo "Job finished: $(date)"
echo "=========================================="

# List output models
echo ""
echo "Trained models:"
ls -la outputs_qwen3/ 2>/dev/null || echo "No outputs directory found"
