#!/bin/bash
#SBATCH --job-name=test-spelling
#SBATCH --output=logs/test-%j.out
#SBATCH --error=logs/test-%j.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

echo "=========================================="
echo "Testing ALL trained spelling models"
echo "Job started: $(date)"
echo "=========================================="

mkdir -p logs

# Load modules
module load python/3.11 cuda/12.6

# Create venv
VENV_DIR=$SLURM_TMPDIR/venv
virtualenv --no-download $VENV_DIR
source $VENV_DIR/bin/activate

pip install --no-index --upgrade pip
pip install --no-index torch transformers accelerate

cd $SLURM_SUBMIT_DIR

echo ""
echo "============================================"
echo "Testing Run 1 models (SmolLM2/Qwen2.5)..."
echo "============================================"
python run_tests.py --model-dir ./outputs --output test_results_run1.json

echo ""
echo "============================================"
echo "Testing Run 2 models (Qwen3)..."
echo "============================================"
python run_tests.py --model-dir ./outputs_qwen3 --output test_results_qwen3.json

echo ""
echo "=========================================="
echo "Job finished: $(date)"
echo "=========================================="

echo ""
echo "Results saved to:"
echo "  - test_results_run1.json"
echo "  - test_results_qwen3.json"
