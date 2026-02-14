#!/bin/bash
#SBATCH --job-name=convert-gguf
#SBATCH --output=/scratch/jic823/dyslexic-writer/logs/convert-gguf-%j.out
#SBATCH --error=/scratch/jic823/dyslexic-writer/logs/convert-gguf-%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# No GPU needed for conversion - CPU only

echo "=========================================="
echo "Converting models to GGUF format"
echo "Job started: $(date)"
echo "=========================================="

mkdir -p /scratch/jic823/dyslexic-writer/logs

cd ~/projects/def-jic823

# Clone llama.cpp if not present
if [ ! -d "llama.cpp" ]; then
    echo "Cloning llama.cpp..."
    git clone https://github.com/ggerganov/llama.cpp.git
fi

cd llama.cpp

# Load modules
module load python/3.11 cmake gcc/12.3

# Create venv for conversion
VENV_DIR=$SLURM_TMPDIR/venv
virtualenv --no-download $VENV_DIR
source $VENV_DIR/bin/activate

# Install requirements for conversion
pip install --no-index torch numpy sentencepiece transformers
pip install gguf  # May need internet for this one

# If gguf fails, try from PyPI
pip install gguf 2>/dev/null || pip install --user gguf

# Output directory for GGUF files
GGUF_DIR=/scratch/jic823/dyslexic-writer/gguf_models
mkdir -p $GGUF_DIR

# Models to convert - v2 best + v3 7B
MODELS=(
    "v2-SmolLM2-1.7B|/scratch/jic823/dyslexic-writer/outputs_v2/SmolLM2-1.7B-Instruct"
    "v3-Qwen2.5-7B|/scratch/jic823/dyslexic-writer/outputs_v3/Qwen2.5-7B-Instruct"
)

for entry in "${MODELS[@]}"; do
    model_name="${entry%%|*}"
    full_path="${entry##*|}"

    echo ""
    echo "============================================"
    echo "Converting: $model_name"
    echo "Source: $full_path"
    echo "============================================"

    if [ ! -d "$full_path" ]; then
        echo "ERROR: Model not found at $full_path"
        continue
    fi

    # Step 1: Convert to GGUF F16
    echo "Step 1: Converting to GGUF F16..."
    python convert_hf_to_gguf.py "$full_path" \
        --outfile "$GGUF_DIR/${model_name}-f16.gguf" \
        --outtype f16

    if [ $? -ne 0 ]; then
        echo "ERROR: F16 conversion failed for $model_name"
        continue
    fi

    # Step 2: Quantize to Q4_K_M (good balance of size and quality)
    echo "Step 2: Quantizing to Q4_K_M..."

    # Build quantize tool if needed
    if [ ! -f "build/bin/llama-quantize" ]; then
        echo "Building llama.cpp quantize tool..."
        mkdir -p build && cd build
        cmake .. -DGGML_NATIVE=OFF
        cmake --build . --target llama-quantize -j 8
        cd ..
    fi

    ./build/bin/llama-quantize \
        "$GGUF_DIR/${model_name}-f16.gguf" \
        "$GGUF_DIR/${model_name}-q4_k_m.gguf" \
        Q4_K_M

    if [ $? -eq 0 ]; then
        echo "SUCCESS: Created $GGUF_DIR/${model_name}-q4_k_m.gguf"
        # Keep F16 for now, can delete later if space needed
        # rm "$GGUF_DIR/${model_name}-f16.gguf"
    else
        echo "ERROR: Quantization failed for $model_name"
    fi

    echo ""
done

echo ""
echo "=========================================="
echo "Conversion complete!"
echo "=========================================="

echo ""
echo "GGUF models created:"
ls -lh $GGUF_DIR/*.gguf 2>/dev/null || echo "No GGUF files found"

echo ""
echo "Job finished: $(date)"
