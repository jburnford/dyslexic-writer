#!/bin/bash
#SBATCH --job-name=gen-texts-v2
#SBATCH --output=logs/gen-texts-v2-%j.out
#SBATCH --error=logs/gen-texts-v2-%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G

# ============================================================================
# Generate diverse source texts for Dyslexic Writer v2 using GPT-OSS-120B
#
# Serves GPT-OSS-120B with vLLM on H100, then runs text generation.
# Generates 2,000 texts + 700 identity per age band (3 bands = 8,100 total)
#
# Expected runtime: ~6-10 hours
# ============================================================================

set -e

echo "=========================================="
echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="

mkdir -p logs

# Load modules
module load python/3.11 cuda/12.6 arrow/21.0.0

echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# Create virtual environment
VENV_DIR=$SLURM_TMPDIR/venv
echo ""
echo "Creating virtual environment in $VENV_DIR..."
virtualenv --no-download $VENV_DIR
source $VENV_DIR/bin/activate

pip install --no-index --upgrade pip
pip install --no-index torch torchvision
pip install --no-index transformers accelerate

# Install vLLM (may need PyPI fallback)
pip install --no-index vllm 2>/dev/null || pip install vllm

# Install requests for the generation script
pip install --no-index requests 2>/dev/null || pip install requests

echo ""
echo "Installed packages:"
pip list | grep -E "vllm|torch|transformers|requests"

cd $SLURM_SUBMIT_DIR

# ============================================================================
# Configuration
# ============================================================================
MODEL_PATH="$HOME/projects/def-jic823/models/gpt-oss-120b"
VLLM_PORT=8000
TEXTS_PER_BAND=2000
IDENTITY_PER_BAND=700

echo ""
echo "Model: $MODEL_PATH"
echo "Texts per band: $TEXTS_PER_BAND"
echo "Identity per band: $IDENTITY_PER_BAND"

# ============================================================================
# Start vLLM server in background
# ============================================================================
echo ""
echo "=========================================="
echo "Starting vLLM server..."
echo "=========================================="

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --port $VLLM_PORT \
    --tensor-parallel-size 1 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.95 \
    --max-num-batched-tokens 1024 \
    --dtype auto \
    --trust-remote-code \
    &

VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

# Wait for server to be ready
echo "Waiting for vLLM server to start..."
MAX_WAIT=1200  # 20 minutes max (183GB model takes time to load)
WAITED=0
while ! curl -s http://localhost:$VLLM_PORT/v1/models > /dev/null 2>&1; do
    sleep 10
    WAITED=$((WAITED + 10))
    if [ $WAITED -ge $MAX_WAIT ]; then
        echo "ERROR: vLLM server failed to start after ${MAX_WAIT}s"
        kill $VLLM_PID 2>/dev/null
        exit 1
    fi
    echo "  Waiting... (${WAITED}s)"
done

echo "vLLM server ready! (took ${WAITED}s)"

# Get model name from vLLM
MODEL_NAME=$(curl -s http://localhost:$VLLM_PORT/v1/models | python -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])")
echo "Model name: $MODEL_NAME"

# ============================================================================
# Run text generation
# ============================================================================
echo ""
echo "=========================================="
echo "Generating diverse texts..."
echo "=========================================="

export LLM_API_URL="http://localhost:$VLLM_PORT"
export LLM_MODEL="$MODEL_NAME"

for BAND in young middle teen; do
    echo ""
    echo "--- Generating $BAND band ---"
    python training/generate_diverse_texts.py \
        --age-band "$BAND" \
        --count $TEXTS_PER_BAND \
        --identity $IDENTITY_PER_BAND \
        --batch-size 50 \
        --seed 42
done

# ============================================================================
# Build training data from generated texts
# ============================================================================
echo ""
echo "=========================================="
echo "Building training data..."
echo "=========================================="

python training/build_training_data.py \
    --source-dir training/generated_diverse \
    --output-dir training/v2_data \
    --target-count 300000 \
    --seed 42

# ============================================================================
# Cleanup
# ============================================================================
echo ""
echo "Shutting down vLLM server..."
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null

echo ""
echo "=========================================="
echo "Job finished: $(date)"
echo "=========================================="

# Show results
echo ""
echo "Generated texts:"
find training/generated_diverse -name "*.jsonl" | wc -l
echo "files"
wc -l training/generated_diverse/*/*.jsonl 2>/dev/null | tail -1

echo ""
echo "Training data:"
wc -l training/v2_data/train.jsonl training/v2_data/eval.jsonl 2>/dev/null

echo ""
echo "Dataset stats:"
cat training/v2_data/dataset_stats.json 2>/dev/null
