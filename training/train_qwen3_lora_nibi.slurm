#!/bin/bash
#SBATCH --job-name=qwen3-lora-v2
#SBATCH --output=logs/qwen3-lora-v2-%j.out
#SBATCH --error=logs/qwen3-lora-v2-%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G

# ============================================================================
# Dyslexic Writer v2 - Qwen3 LoRA Training on Nibi (H100)
#
# Trains 4 models sequentially:
#   Phase 1: Qwen3-0.6B, 1.7B, 4B  (LoRA, r=64)
#   Phase 2: Qwen3-14B             (LoRA, r=32)
#
# Data: v2_data/train.jsonl (5638 examples), v2_data/eval.jsonl (627 examples)
# Anti-overfitting: 2 epochs, weight_decay=0.05, early stopping patience=3
# Expected runtime: ~12-18 hours on H100
# ============================================================================

set -e

echo "=========================================="
echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="

mkdir -p logs

# Load modules (arrow MUST be loaded before venv for pyarrow)
module load python/3.11 cuda/12.6 arrow/21.0.0

echo ""
echo "Loaded modules:"
module list

echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv

# Create virtual environment in fast local storage
VENV_DIR=$SLURM_TMPDIR/venv
echo ""
echo "Creating virtual environment in $VENV_DIR..."
virtualenv --no-download $VENV_DIR
source $VENV_DIR/bin/activate

pip install --no-index --upgrade pip

echo ""
echo "Installing dependencies..."
pip install --no-index torch torchvision
pip install --no-index transformers accelerate peft datasets sentencepiece
pip install --no-index trl 2>/dev/null || pip install trl
pip install --no-index bitsandbytes 2>/dev/null || pip install bitsandbytes

echo ""
echo "Installed packages:"
pip list | grep -E "torch|transformers|datasets|trl|accelerate|peft|bitsandbytes"
python -c "
from datasets import Dataset
from trl import SFTTrainer
from peft import LoraConfig
print('All imports OK')
"

cd $SLURM_SUBMIT_DIR

# ============================================================================
# Phase 1: Small models with LoRA (0.6B, 1.7B, 4B)
# ============================================================================
echo ""
echo "=========================================="
echo "Phase 1: Small models with LoRA"
echo "=========================================="

for MODEL in Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B Qwen/Qwen3-4B; do
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
# Phase 2: 14B model with LoRA
# ============================================================================
echo ""
echo "=========================================="
echo "Phase 2: 14B model with LoRA"
echo "=========================================="

echo ""
echo "=========================================="
echo "Training: Qwen/Qwen3-14B (LoRA)"
echo "Start: $(date)"
echo "=========================================="
python finetune_qwen3.py \
    --model "Qwen/Qwen3-14B" \
    --output-dir ./outputs_qwen3_v2 \
    --epochs 2 \
    --lora \
    --train-file v2_data/train.jsonl \
    --eval-file v2_data/eval.jsonl \
    --early-stopping 3
echo "Finished Qwen/Qwen3-14B at $(date)"

echo ""
echo "=========================================="
echo "All training complete: $(date)"
echo "=========================================="

echo ""
echo "Trained models:"
ls -la outputs_qwen3_v2/ 2>/dev/null || echo "No outputs directory found"

echo ""
echo "Disk usage:"
du -sh outputs_qwen3_v2/*/ 2>/dev/null || echo "No model directories found"
