#!/bin/bash
#SBATCH --job-name=test-qwen3
#SBATCH --output=logs/test-qwen3-%j.out
#SBATCH --error=logs/test-qwen3-%j.err
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

echo "=========================================="
echo "Testing Qwen3 models (with thinking mode disabled)"
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
echo "Testing Qwen3 models..."
python run_tests.py --model-dir ./outputs_qwen3 --output test_results_qwen3_fixed.json

echo ""
echo "=========================================="
echo "Job finished: $(date)"
echo "=========================================="
