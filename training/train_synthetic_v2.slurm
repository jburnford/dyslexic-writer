#!/bin/bash
#SBATCH --job-name=synth-spell-v2
#SBATCH --output=/scratch/jic823/dyslexic-writer/logs/synth-spell-%j.out
#SBATCH --error=/scratch/jic823/dyslexic-writer/logs/synth-spell-%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# ============================================================================
# Dyslexic Writer - Stage 2 Fine-tuning (v2 - using scratch for output)
# All model outputs go to /scratch to avoid home quota issues
# ============================================================================

echo "=========================================="
echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="

# Setup directories on scratch
WORK_DIR=/scratch/jic823/dyslexic-writer
OUTPUT_DIR=$WORK_DIR/outputs_v2
mkdir -p $WORK_DIR/logs
mkdir -p $OUTPUT_DIR

# Set HF cache to scratch too
export HF_HOME=$WORK_DIR/hf_cache
export TRANSFORMERS_CACHE=$WORK_DIR/hf_cache
mkdir -p $HF_HOME

# Load modules
module load python/3.11 cuda/12.6 arrow/21.0.0

echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv

# Create virtual environment in fast local storage
VENV_DIR=$SLURM_TMPDIR/venv
echo ""
echo "Creating virtual environment..."
virtualenv --no-download $VENV_DIR
source $VENV_DIR/bin/activate

pip install --no-index --upgrade pip
pip install --no-index torch torchvision
pip install --no-index transformers datasets trl accelerate peft
pip install --no-index sentencepiece
pip install tokenizers --upgrade 2>/dev/null || true

echo ""
pip list | grep -E "torch|transformers|datasets|trl|accelerate|peft"

# Copy training script to scratch (in case home is slow)
cp ~/dyslexic-writer/training/finetune.py $WORK_DIR/

cd $WORK_DIR

echo ""
echo "=========================================="
echo "Training data:"
echo "  Train: $(wc -l < synthetic_train_clean_instruction.jsonl) pairs"
echo "  Eval:  $(wc -l < synthetic_eval_clean_instruction.jsonl) pairs"
echo "  Output: $OUTPUT_DIR"
echo "=========================================="

# Train all 4 models
python finetune.py \
    --model all \
    --train-file synthetic_train_clean_instruction.jsonl \
    --eval-file synthetic_eval_clean_instruction.jsonl \
    --output-dir $OUTPUT_DIR \
    --epochs 3 \
    --batch-size 8 \
    --lr 5e-5

echo ""
echo "=========================================="
echo "Job finished: $(date)"
echo "=========================================="

echo ""
echo "Trained models:"
for d in $OUTPUT_DIR/*/; do
    echo "  $(basename $d): $(du -sh $d 2>/dev/null | cut -f1)"
done

echo ""
echo "Disk usage:"
du -sh $OUTPUT_DIR
