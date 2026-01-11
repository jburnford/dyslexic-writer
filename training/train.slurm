#!/bin/bash
#SBATCH --job-name=spelling-finetune
#SBATCH --output=logs/spelling-%j.out
#SBATCH --error=logs/spelling-%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# ============================================================================
# Dyslexic Writer - Spelling Model Fine-tuning
#
# Trains 4 small models for local inference:
#   - SmolLM2-360M-Instruct
#   - Qwen2.5-0.5B-Instruct
#   - Qwen2.5-1.5B-Instruct
#   - SmolLM2-1.7B-Instruct
#
# Expected runtime: ~4-6 hours total on H100
# ============================================================================

echo "=========================================="
echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "=========================================="

# Load modules (adjust for your cluster)
# module load cuda/12.1
# module load python/3.11

# Create logs directory
mkdir -p logs

# Activate virtual environment (create if needed)
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets trl accelerate
pip install flash-attn --no-build-isolation

# Prepare training data
echo ""
echo "Preparing training data..."
python prepare_finetune_data.py

# Train all models
echo ""
echo "Starting training..."
python finetune.py --model all --output-dir ./outputs --epochs 3

echo ""
echo "=========================================="
echo "Job finished: $(date)"
echo "=========================================="

# List output models
echo ""
echo "Trained models:"
ls -la outputs/
