#!/bin/bash
#SBATCH --job-name=spell-large-v3
#SBATCH --output=/scratch/jic823/dyslexic-writer/logs/spell-large-%j.out
#SBATCH --error=/scratch/jic823/dyslexic-writer/logs/spell-large-%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# ============================================================================
# Dyslexic Writer - v3 Large Model Training (round 3)
# Train Llama-3.2-3B (gate access now approved)
# Qwen2.5-7B already trained, Phi-3.5 incompatible with TRL
# ============================================================================

echo "=========================================="
echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="

# Setup directories on scratch
WORK_DIR=/scratch/jic823/dyslexic-writer
OUTPUT_DIR=$WORK_DIR/outputs_v3
mkdir -p $WORK_DIR/logs
mkdir -p $OUTPUT_DIR

# Set HF cache to scratch + auth token for gated models
export HF_HOME=$WORK_DIR/hf_cache
export TRANSFORMERS_CACHE=$WORK_DIR/hf_cache
mkdir -p $HF_HOME
export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null)

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
pip install --no-index sentencepiece protobuf
pip install tokenizers --upgrade 2>/dev/null || true

echo ""
pip list | grep -E "torch|transformers|datasets|trl|accelerate|peft|sentencepiece"

# Copy training script to scratch
cp ~/dyslexic-writer/training/finetune_v2.py $WORK_DIR/

cd $WORK_DIR

echo ""
echo "=========================================="
echo "Training data:"
echo "  Train: $(wc -l < synthetic_train_clean_instruction.jsonl) pairs"
echo "  Eval:  $(wc -l < synthetic_eval_clean_instruction.jsonl) pairs"
echo "  Output: $OUTPUT_DIR"
echo "=========================================="

COMMON_ARGS="--train-file synthetic_train_clean_instruction.jsonl \
    --eval-file synthetic_eval_clean_instruction.jsonl \
    --output-dir $OUTPUT_DIR \
    --epochs 3 \
    --lr 5e-5 \
    --gradient-checkpointing"

# ---- Llama-3.2-3B-Instruct ----
echo ""
echo "############################################################"
echo "# meta-llama/Llama-3.2-3B-Instruct (3B)"
echo "############################################################"
python finetune_v2.py \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --batch-size 8 \
    --grad-accum 4 \
    $COMMON_ARGS

echo ""
echo "=========================================="
echo "Job finished: $(date)"
echo "=========================================="

echo ""
echo "Trained models:"
for d in $OUTPUT_DIR/*/; do
    if [ -d "$d" ]; then
        echo "  $(basename $d): $(du -sh $d 2>/dev/null | cut -f1)"
    fi
done

echo ""
echo "Total disk usage:"
du -sh $OUTPUT_DIR
