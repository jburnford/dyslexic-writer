#!/bin/bash
#SBATCH --job-name=spell-proper-nouns
#SBATCH --output=logs/proper_nouns_%j.out
#SBATCH --error=logs/proper_nouns_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# ============================================================================
# Train SmolLM2-1.7B and Qwen3-4B on proper nouns data
# Two approaches per model:
#   Option A: From scratch with combined data
#   Option B: Continue training existing model
# ============================================================================

echo "=========================================="
echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="

mkdir -p logs outputs

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

# Install dependencies
pip install --no-index --upgrade pip
pip install --no-index torch torchvision
pip install --no-index transformers datasets trl accelerate peft
pip install --no-index sentencepiece
pip install tokenizers --upgrade 2>/dev/null || true

echo ""
echo "Installed packages:"
pip list | grep -E "torch|transformers|datasets|trl|accelerate"

cd $SLURM_SUBMIT_DIR

# ==========================================
# SmolLM2-1.7B
# ==========================================
echo ""
echo "=========================================="
echo "SmolLM2-1.7B: OPTION A (from scratch)"
echo "=========================================="
python finetune.py \
    --model HuggingFaceTB/SmolLM2-1.7B-Instruct \
    --output-dir ./outputs/smol_1.7b_combined \
    --train-file combined_train.jsonl \
    --eval-file combined_eval.jsonl \
    --epochs 3 \
    --batch-size 8 \
    --lr 2e-5

echo ""
echo "=========================================="
echo "SmolLM2-1.7B: OPTION B (continue training)"
echo "=========================================="
# Check if we have a previously trained model to continue from
if [ -d "./outputs/SmolLM2-1.7B-Instruct" ]; then
    python finetune_continue.py \
        --base-model ./outputs/SmolLM2-1.7B-Instruct \
        --output-dir ./outputs/smol_1.7b_continued \
        --train-file proper_nouns_train.jsonl \
        --eval-file proper_nouns_eval.jsonl \
        --epochs 3 \
        --batch-size 8 \
        --lr 5e-6
else
    echo "No existing SmolLM2-1.7B model found, skipping continue training"
    echo "Using the freshly trained model instead..."
    python finetune_continue.py \
        --base-model ./outputs/smol_1.7b_combined/SmolLM2-1.7B-Instruct \
        --output-dir ./outputs/smol_1.7b_continued \
        --train-file proper_nouns_train.jsonl \
        --eval-file proper_nouns_eval.jsonl \
        --epochs 3 \
        --batch-size 8 \
        --lr 5e-6
fi

# ==========================================
# Qwen3-4B
# ==========================================
echo ""
echo "=========================================="
echo "Qwen3-4B: OPTION A (from scratch)"
echo "=========================================="
python finetune_qwen3.py \
    --model Qwen/Qwen3-4B \
    --output-dir ./outputs/qwen3_4b_combined \
    --train-file combined_train.jsonl \
    --eval-file combined_eval.jsonl \
    --epochs 3 \
    --batch-size 4 \
    --lr 2e-5

echo ""
echo "=========================================="
echo "Qwen3-4B: OPTION B (continue training)"
echo "=========================================="
if [ -d "./outputs/Qwen3-4B" ]; then
    python finetune_continue.py \
        --base-model ./outputs/Qwen3-4B \
        --output-dir ./outputs/qwen3_4b_continued \
        --train-file proper_nouns_train.jsonl \
        --eval-file proper_nouns_eval.jsonl \
        --epochs 3 \
        --batch-size 4 \
        --lr 5e-6
else
    echo "No existing Qwen3-4B model found, using freshly trained..."
    python finetune_continue.py \
        --base-model ./outputs/qwen3_4b_combined/Qwen3-4B \
        --output-dir ./outputs/qwen3_4b_continued \
        --train-file proper_nouns_train.jsonl \
        --eval-file proper_nouns_eval.jsonl \
        --epochs 3 \
        --batch-size 4 \
        --lr 5e-6
fi

# ==========================================
# Run tests on all models
# ==========================================
echo ""
echo "=========================================="
echo "Running tests on all models..."
echo "=========================================="
python run_tests.py --model-dir ./outputs/smol_1.7b_combined --output results_smol_combined.json
python run_tests.py --model-dir ./outputs/smol_1.7b_continued --output results_smol_continued.json
python run_tests.py --model-dir ./outputs/qwen3_4b_combined --output results_qwen3_combined.json
python run_tests.py --model-dir ./outputs/qwen3_4b_continued --output results_qwen3_continued.json

echo ""
echo "=========================================="
echo "DONE! Job finished: $(date)"
echo "=========================================="
echo "Results:"
ls -la results_*.json 2>/dev/null || echo "No results files found"
echo ""
echo "Models:"
ls -la outputs/ 2>/dev/null || echo "No outputs found"
