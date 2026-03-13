#!/bin/bash
#SBATCH --job-name=hockey-v2e-14b
#SBATCH --output=logs/hockey-v2e-14b-%j.out
#SBATCH --error=logs/hockey-v2e-14b-%j.err
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G

module load python/3.11 cuda/12.6 arrow/21.0.0
VENV_DIR=$SLURM_TMPDIR/venv
virtualenv --no-download $VENV_DIR
source $VENV_DIR/bin/activate
pip install --no-index --upgrade pip
pip install --no-index torch torchvision
pip install --no-index transformers accelerate sentencepiece

cd $SLURM_SUBMIT_DIR

python3 test_hockey_story_v2e_14b.py
