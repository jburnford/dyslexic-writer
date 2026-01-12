#!/bin/bash
#SBATCH --job-name=test-spelling
#SBATCH --output=logs/test-%j.out
#SBATCH --error=logs/test-%j.err
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

echo "=========================================="
echo "Testing trained spelling models"
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
echo "Running hallucination tests..."
python run_tests.py --model-dir ./outputs --output test_results.json

echo ""
echo "=========================================="
echo "Job finished: $(date)"
echo "=========================================="
