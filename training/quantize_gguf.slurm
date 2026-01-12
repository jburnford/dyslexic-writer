#!/bin/bash
#SBATCH --job-name=quantize-gguf
#SBATCH --output=logs/quantize-gguf-%j.out
#SBATCH --error=logs/quantize-gguf-%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

echo "=========================================="
echo "Quantizing GGUF models to Q4_K_M"
echo "Job started: $(date)"
echo "=========================================="

# Load modules
module load python/3.11 cmake gcc/12.3

# Build llama-quantize in scratch (faster, no quota issues)
echo "Building llama-quantize in SLURM_TMPDIR..."
cd $SLURM_TMPDIR
git clone --depth 1 https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
mkdir build && cd build
cmake .. -DGGML_NATIVE=OFF -DCMAKE_BUILD_TYPE=Release -DGGML_CCACHE=OFF
make -j8 llama-quantize

QUANTIZE_BIN="$SLURM_TMPDIR/llama.cpp/build/bin/llama-quantize"

if [ ! -f "$QUANTIZE_BIN" ]; then
    echo "ERROR: llama-quantize build failed"
    exit 1
fi

echo "llama-quantize built successfully at $QUANTIZE_BIN"

# Quantize each F16 model
GGUF_DIR=~/projects/def-jic823/dyslexic-writer/training/gguf_models

for f16_file in $GGUF_DIR/*-f16.gguf; do
    if [ ! -f "$f16_file" ]; then
        continue
    fi

    model_name=$(basename "$f16_file" -f16.gguf)
    q4_file="$GGUF_DIR/${model_name}-q4_k_m.gguf"

    echo ""
    echo "============================================"
    echo "Quantizing: $model_name"
    echo "  Input:  $f16_file"
    echo "  Output: $q4_file"
    echo "============================================"

    $QUANTIZE_BIN "$f16_file" "$q4_file" Q4_K_M

    if [ $? -eq 0 ]; then
        echo "SUCCESS: Created $q4_file"
        ls -lh "$q4_file"
    else
        echo "ERROR: Quantization failed for $model_name"
    fi
done

echo ""
echo "=========================================="
echo "Quantization complete!"
echo "=========================================="

echo ""
echo "All GGUF models:"
ls -lh $GGUF_DIR/*.gguf

echo ""
echo "Job finished: $(date)"
