#!/bin/bash
#SBATCH --job-name=spelling-finetune
#SBATCH --output=logs/spelling-%j.out
#SBATCH --error=logs/spelling-%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# ============================================================================
# Dyslexic Writer - Spelling Model Fine-tuning on Nibi Cluster
#
# Trains 4 small models for local inference:
#   - SmolLM2-360M-Instruct
#   - Qwen2.5-0.5B-Instruct
#   - Qwen2.5-1.5B-Instruct
#   - SmolLM2-1.7B-Instruct
#
# Uses Alliance Canada wheels and PyTorch's native SDPA for Flash Attention
# Expected runtime: ~4-6 hours total on H100
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
# This is the DRAC-recommended approach for job-specific environments
VENV_DIR=$SLURM_TMPDIR/venv
echo ""
echo "Creating virtual environment in $VENV_DIR..."
virtualenv --no-download $VENV_DIR
source $VENV_DIR/bin/activate

# Upgrade pip using Alliance wheel
pip install --no-index --upgrade pip

# Install dependencies from Alliance wheels (--no-index = use local wheels only)
echo ""
echo "Installing dependencies from Alliance wheels..."
pip install --no-index torch torchvision
pip install --no-index transformers datasets trl accelerate peft
pip install --no-index sentencepiece

# Install any packages not available as wheels from PyPI
# (tokenizers may need newer version for transformers compatibility)
echo ""
echo "Installing additional packages from PyPI if needed..."
pip install tokenizers --upgrade 2>/dev/null || true

# Show installed versions
echo ""
echo "Installed packages:"
pip list | grep -E "torch|transformers|datasets|trl|accelerate"

# Change to training directory
cd $SLURM_SUBMIT_DIR

# Train all models
echo ""
echo "=========================================="
echo "Starting training..."
echo "=========================================="
python finetune.py --model all --output-dir ./outputs --epochs 3

echo ""
echo "=========================================="
echo "Job finished: $(date)"
echo "=========================================="

# List output models
echo ""
echo "Trained models:"
ls -la outputs/ 2>/dev/null || echo "No outputs directory found"
