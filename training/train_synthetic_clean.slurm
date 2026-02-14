#!/bin/bash
#SBATCH --job-name=synth-spelling
#SBATCH --output=logs/synth-spelling-%j.out
#SBATCH --error=logs/synth-spelling-%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# ============================================================================
# Dyslexic Writer - Stage 2 Fine-tuning on Synthetic Bob-Story-Calibrated Data
#
# Trains on 50k+ synthetic sentence pairs with:
#   - 40% identity (pass-through) pairs
#   - Bob-story-calibrated error injection
#   - Phonological-dominant error distribution
#
# Models: SmolLM2-1.7B, Qwen2.5-0.5B, Qwen2.5-1.5B, SmolLM2-360M
# Expected runtime: ~4-6 hours on H100
# ============================================================================

echo "=========================================="
echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="

mkdir -p logs

# Load modules
module load python/3.11 cuda/12.6 arrow/21.0.0

echo ""
echo "Loaded modules:"
module list

echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv

# Create virtual environment
VENV_DIR=$SLURM_TMPDIR/venv
echo ""
echo "Creating virtual environment in $VENV_DIR..."
virtualenv --no-download $VENV_DIR
source $VENV_DIR/bin/activate

pip install --no-index --upgrade pip
echo ""
echo "Installing dependencies from Alliance wheels..."
pip install --no-index torch torchvision
pip install --no-index transformers datasets trl accelerate peft
pip install --no-index sentencepiece
pip install tokenizers --upgrade 2>/dev/null || true

echo ""
echo "Installed packages:"
pip list | grep -E "torch|transformers|datasets|trl|accelerate|peft"

cd $SLURM_SUBMIT_DIR

echo ""
echo "=========================================="
echo "Training data stats:"
echo "  Train: $(wc -l < synthetic_train_clean_instruction.jsonl) pairs"
echo "  Eval:  $(wc -l < synthetic_eval_clean_instruction.jsonl) pairs"
echo "=========================================="

# Train all 4 models on the new synthetic data
echo ""
echo "=========================================="
echo "Starting training on synthetic data..."
echo "=========================================="
python finetune.py \
    --model all \
    --train-file synthetic_train_clean_instruction.jsonl \
    --eval-file synthetic_eval_clean_instruction.jsonl \
    --output-dir ./outputs_synthetic \
    --epochs 3 \
    --batch-size 8 \
    --lr 5e-5

echo ""
echo "=========================================="
echo "Job finished: $(date)"
echo "=========================================="

echo ""
echo "Trained models:"
ls -la outputs_synthetic/ 2>/dev/null || echo "No outputs directory found"
for d in outputs_synthetic/*/; do
    echo "  $d: $(du -sh $d 2>/dev/null | cut -f1)"
done
