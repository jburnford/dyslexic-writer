#!/bin/bash
#SBATCH --job-name=qwen3-lora-v2
#SBATCH --output=logs/qwen3-lora-v2-%j.out
#SBATCH --error=logs/qwen3-lora-v2-%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G

# ============================================================================
# Dyslexic Writer v2 - Qwen3 LoRA/QLoRA Training on Nibi (H100)
#
# Trains Qwen3 models with LoRA for spelling correction:
#   - Qwen3-0.6B  (LoRA, r=64)   - low-end devices
#   - Qwen3-1.7B  (LoRA, r=64)   - default for most
#   - Qwen3-4B    (LoRA, r=64)   - good balance
#   - Qwen3-8B    (LoRA, r=64)   - premium quality
#   - Qwen3-14B   (LoRA, r=32)   - high quality
#   - Qwen3-32B   (QLoRA 4-bit, r=32) - maximum quality
#
# Anti-overfitting measures:
#   - 2 epochs (reduced from 3)
#   - weight_decay=0.05 (increased from 0.01)
#   - Early stopping with patience=3
#   - More frequent eval (every 250 steps)
#
# Expected runtime: ~18-24 hours total on H100
# ============================================================================

set -e  # Exit on first error

echo "=========================================="
echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="

# Create logs directory
mkdir -p logs

# Load modules (arrow MUST be loaded before venv for pyarrow)
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

# Upgrade pip
pip install --no-index --upgrade pip

# Install dependencies from Alliance wheels
echo ""
echo "Installing dependencies..."
pip install --no-index torch torchvision
pip install --no-index transformers accelerate peft datasets sentencepiece
# trl and bitsandbytes may not be in Alliance wheels — fall back to PyPI
pip install --no-index trl 2>/dev/null || pip install trl
pip install --no-index bitsandbytes 2>/dev/null || pip install bitsandbytes

# Verify critical packages
echo ""
echo "Installed packages:"
pip list | grep -E "torch|transformers|datasets|trl|accelerate|peft|bitsandbytes"
python -c "
from datasets import Dataset
from trl import SFTTrainer
from peft import LoraConfig
print('All imports OK')
try:
    import bitsandbytes
    print('bitsandbytes available (QLoRA supported)')
except ImportError:
    print('WARNING: bitsandbytes not available (QLoRA not supported)')
"

# Change to training directory
cd $SLURM_SUBMIT_DIR

# ============================================================================
# Configure which models to train
# ============================================================================
# Edit this list to select which models to train
# For a quick test: use just one small model
# For full training: include all models

# Default: train 0.6B through 8B with LoRA
MODELS_SMALL="Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B Qwen/Qwen3-4B"
MODELS_LARGE="Qwen/Qwen3-8B"
MODELS_XLARGE="Qwen/Qwen3-14B"
MODELS_QLORA="Qwen/Qwen3-32B"

# ============================================================================
# Phase 1: Small models with LoRA (0.6B, 1.7B, 4B)
# ============================================================================
echo ""
echo "=========================================="
echo "Phase 1: Small models with LoRA"
echo "=========================================="

for MODEL in $MODELS_SMALL; do
    echo ""
    echo "=========================================="
    echo "Training: $MODEL (LoRA)"
    echo "Start: $(date)"
    echo "=========================================="
    python finetune_qwen3.py \
        --model "$MODEL" \
        --output-dir ./outputs_qwen3_v2 \
        --epochs 2 \
        --lora \
        --train-file v2_data/train.jsonl \
        --eval-file v2_data/eval.jsonl \
        --early-stopping 3
    echo "Finished $MODEL at $(date)"
done

# ============================================================================
# Phase 2: Large model with LoRA (8B)
# ============================================================================
echo ""
echo "=========================================="
echo "Phase 2: 8B model with LoRA"
echo "=========================================="

for MODEL in $MODELS_LARGE; do
    echo ""
    echo "=========================================="
    echo "Training: $MODEL (LoRA)"
    echo "Start: $(date)"
    echo "=========================================="
    python finetune_qwen3.py \
        --model "$MODEL" \
        --output-dir ./outputs_qwen3_v2 \
        --epochs 2 \
        --lora \
        --train-file v2_data/train.jsonl \
        --eval-file v2_data/eval.jsonl \
        --early-stopping 3
    echo "Finished $MODEL at $(date)"
done

# ============================================================================
# Phase 3: XLarge model with LoRA (14B)
# ============================================================================
echo ""
echo "=========================================="
echo "Phase 3: 14B model with LoRA"
echo "=========================================="

for MODEL in $MODELS_XLARGE; do
    echo ""
    echo "=========================================="
    echo "Training: $MODEL (LoRA)"
    echo "Start: $(date)"
    echo "=========================================="
    python finetune_qwen3.py \
        --model "$MODEL" \
        --output-dir ./outputs_qwen3_v2 \
        --epochs 2 \
        --lora \
        --train-file v2_data/train.jsonl \
        --eval-file v2_data/eval.jsonl \
        --early-stopping 3
    echo "Finished $MODEL at $(date)"
done

# ============================================================================
# Phase 4: 32B model with QLoRA (4-bit quantized)
# ============================================================================
echo ""
echo "=========================================="
echo "Phase 4: 32B model with QLoRA"
echo "=========================================="

for MODEL in $MODELS_QLORA; do
    echo ""
    echo "=========================================="
    echo "Training: $MODEL (QLoRA 4-bit)"
    echo "Start: $(date)"
    echo "=========================================="
    python finetune_qwen3.py \
        --model "$MODEL" \
        --output-dir ./outputs_qwen3_v2 \
        --epochs 2 \
        --qlora \
        --train-file v2_data/train.jsonl \
        --eval-file v2_data/eval.jsonl \
        --early-stopping 3
    echo "Finished $MODEL at $(date)"
done

echo ""
echo "=========================================="
echo "All training complete: $(date)"
echo "=========================================="

# List output models
echo ""
echo "Trained models:"
ls -la outputs_qwen3_v2/ 2>/dev/null || echo "No outputs directory found"

echo ""
echo "Disk usage:"
du -sh outputs_qwen3_v2/*/ 2>/dev/null || echo "No model directories found"
