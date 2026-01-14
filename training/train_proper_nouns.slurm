#!/bin/bash
#SBATCH --job-name=spell-proper-nouns
#SBATCH --output=logs/proper_nouns_%j.out
#SBATCH --error=logs/proper_nouns_%j.err
#SBATCH --time=8:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8

# Train SmolLM2-1.7B and Qwen3-4B on proper nouns data
# Two approaches per model:
#   Option A: From scratch with combined data
#   Option B: Continue training existing model

module load cuda/12.1
source ~/venvs/spelling/bin/activate

cd $SLURM_SUBMIT_DIR
mkdir -p logs outputs

# Prepare the datasets
echo "Preparing datasets..."
python prepare_proper_nouns.py

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
python finetune_continue.py \
    --base-model ./outputs/SmolLM2-1.7B-Instruct \
    --output-dir ./outputs/smol_1.7b_continued \
    --train-file proper_nouns_train.jsonl \
    --eval-file proper_nouns_eval.jsonl \
    --epochs 3 \
    --batch-size 8 \
    --lr 5e-6

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
python finetune_continue.py \
    --base-model ./outputs/Qwen3-4B \
    --output-dir ./outputs/qwen3_4b_continued \
    --train-file proper_nouns_train.jsonl \
    --eval-file proper_nouns_eval.jsonl \
    --epochs 3 \
    --batch-size 4 \
    --lr 5e-6

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
echo "DONE! Results:"
echo "=========================================="
echo "  SmolLM2-1.7B combined:  results_smol_combined.json"
echo "  SmolLM2-1.7B continued: results_smol_continued.json"
echo "  Qwen3-4B combined:      results_qwen3_combined.json"
echo "  Qwen3-4B continued:     results_qwen3_continued.json"
