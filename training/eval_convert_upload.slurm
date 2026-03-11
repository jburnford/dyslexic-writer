#!/bin/bash
#SBATCH --job-name=eval-gguf-hf
#SBATCH --output=logs/eval-gguf-hf-%j.out
#SBATCH --error=logs/eval-gguf-hf-%j.err
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G

# ============================================================================
# Dyslexic Writer v2 - Evaluate, Convert to GGUF, Upload to HuggingFace
#
# Phase 1: Evaluate merged models on v2 eval set
# Phase 2: Convert to GGUF + quantize (Q4_K_M, Q8_0)
# Phase 3: Upload to HuggingFace
# ============================================================================

set -e

echo "=========================================="
echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="

mkdir -p logs

module load python/3.11 cuda/12.6 arrow/21.0.0 cmake gcc/12.3

echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv

VENV_DIR=$SLURM_TMPDIR/venv
echo "Creating virtual environment in $VENV_DIR..."
virtualenv --no-download $VENV_DIR
source $VENV_DIR/bin/activate

pip install --no-index --upgrade pip
pip install --no-index torch torchvision
pip install --no-index transformers accelerate datasets sentencepiece numpy
pip install gguf huggingface_hub 2>/dev/null || pip install --user gguf huggingface_hub

echo "Installed packages:"
pip list | grep -E "torch|transformers|gguf|hugging"

cd $SLURM_SUBMIT_DIR

MODELS_DIR=outputs_qwen3_v2
GGUF_DIR=gguf_models_v2
HF_USER=jburnford

MODEL_LIST="Qwen3-0.6B-merged Qwen3-1.7B-merged Qwen3-4B-merged Qwen3-14B-merged"

mkdir -p $GGUF_DIR

# ============================================================================
# PHASE 1: Evaluate models
# ============================================================================
echo ""
echo "=========================================="
echo "PHASE 1: Evaluating models"
echo "=========================================="

python evaluate_models.py \
    --models-dir $MODELS_DIR \
    --eval-file v2_data/eval.jsonl \
    --max-examples 627 \
    --models Qwen3-0.6B-merged Qwen3-1.7B-merged Qwen3-4B-merged Qwen3-14B-merged

echo ""
echo "PHASE 1 COMPLETE"

# ============================================================================
# PHASE 2: Convert to GGUF
# ============================================================================
echo ""
echo "=========================================="
echo "PHASE 2: Converting models to GGUF"
echo "=========================================="

LLAMA_DIR=$SLURM_TMPDIR/llama.cpp
if [ ! -d "$LLAMA_DIR" ]; then
    echo "Cloning llama.cpp..."
    git clone --depth 1 https://github.com/ggerganov/llama.cpp.git $LLAMA_DIR
fi

echo "Building llama-quantize..."
cd $LLAMA_DIR
mkdir -p build && cd build
CCACHE_DISABLE=1 cmake .. -DGGML_NATIVE=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build . --target llama-quantize -j 8
QUANTIZE=$LLAMA_DIR/build/bin/llama-quantize
cd $SLURM_SUBMIT_DIR

for MODEL_MERGED in $MODEL_LIST; do
    # e.g. Qwen3-4B-merged -> Qwen3-4B
    MODEL_NAME=${MODEL_MERGED%-merged}
    MODEL_PATH=$MODELS_DIR/$MODEL_MERGED
    F16_GGUF=$GGUF_DIR/${MODEL_NAME}-f16.gguf
    Q4_GGUF=$GGUF_DIR/${MODEL_NAME}-q4_k_m.gguf
    Q8_GGUF=$GGUF_DIR/${MODEL_NAME}-q8_0.gguf

    echo ""
    echo "--- Converting $MODEL_NAME ---"

    if [ ! -d "$MODEL_PATH" ]; then
        echo "ERROR: Model not found at $MODEL_PATH, skipping"
        continue
    fi

    echo "  Converting to F16 GGUF..."
    python $LLAMA_DIR/convert_hf_to_gguf.py "$MODEL_PATH" \
        --outfile "$F16_GGUF" \
        --outtype f16

    if [ $? -ne 0 ]; then
        echo "ERROR: F16 conversion failed for $MODEL_NAME"
        continue
    fi
    echo "  F16: $(du -h "$F16_GGUF" | cut -f1)"

    echo "  Quantizing to Q4_K_M..."
    $QUANTIZE "$F16_GGUF" "$Q4_GGUF" Q4_K_M
    echo "  Q4_K_M: $(du -h "$Q4_GGUF" | cut -f1)"

    echo "  Quantizing to Q8_0..."
    $QUANTIZE "$F16_GGUF" "$Q8_GGUF" Q8_0
    echo "  Q8_0: $(du -h "$Q8_GGUF" | cut -f1)"

    # For 14B also do Q5_K_M (good balance for larger model)
    if [[ "$MODEL_NAME" == *"14B"* ]]; then
        Q5_GGUF=$GGUF_DIR/${MODEL_NAME}-q5_k_m.gguf
        echo "  Quantizing to Q5_K_M..."
        $QUANTIZE "$F16_GGUF" "$Q5_GGUF" Q5_K_M
        echo "  Q5_K_M: $(du -h "$Q5_GGUF" | cut -f1)"
    fi

    echo "  Done: $MODEL_NAME"
done

echo ""
echo "PHASE 2 COMPLETE"
echo ""
echo "GGUF models:"
ls -lh $GGUF_DIR/*.gguf 2>/dev/null

# ============================================================================
# PHASE 3: Upload to HuggingFace
# ============================================================================
echo ""
echo "=========================================="
echo "PHASE 3: Uploading to HuggingFace"
echo "=========================================="

HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null)
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: No HuggingFace token found. Skipping upload."
    echo "To fix: huggingface-cli login"
    echo "GGUF files are ready for manual upload in: $(pwd)/$GGUF_DIR/"
else
    python3 << 'UPLOAD_SCRIPT'
import os
import json
from pathlib import Path
from huggingface_hub import HfApi, create_repo

api = HfApi()
hf_user = "jburnford"
gguf_dir = Path("gguf_models_v2")
models_dir = Path("outputs_qwen3_v2")

# Load eval results if available
eval_results = []
eval_file = models_dir / "eval_results.json"
if eval_file.exists():
    with open(eval_file) as f:
        eval_results = json.load(f)

models = [
    ("Qwen3-0.6B", "qwen3-0.6b"),
    ("Qwen3-1.7B", "qwen3-1.7b"),
    ("Qwen3-4B", "qwen3-4b"),
    ("Qwen3-14B", "qwen3-14b"),
]

for model_name, hf_suffix in models:
    repo_id = f"{hf_user}/dyslexic-writer-{hf_suffix}"
    merged_dir = models_dir / f"{model_name}-merged"
    print(f"\n--- Uploading {model_name} to {repo_id} ---")

    # Find eval metrics
    metrics = {}
    for r in eval_results:
        if model_name in r.get("model", ""):
            metrics = r
            break

    accuracy = metrics.get("exact_match_accuracy", 0) * 100
    error_fix = metrics.get("has_error_accuracy", 0) * 100
    no_err = metrics.get("no_error_accuracy", 0) * 100

    readme = f"""---
language: en
license: apache-2.0
tags:
  - spelling-correction
  - dyslexia
  - text-correction
  - education
base_model: Qwen/{model_name}
pipeline_tag: text-generation
---

# Dyslexic Writer v2 - {model_name}

Fine-tuned **Qwen/{model_name}** for spelling and grammar correction, optimized for dyslexic writers.

## Performance (v2)

| Metric | Score |
|--------|-------|
| Exact Match Accuracy | {accuracy:.1f}% |
| Error Fix Rate | {error_fix:.1f}% |
| No-Error Preservation | {no_err:.1f}% |

## Usage

### With Ollama (GGUF)

```bash
ollama create dyslexic-writer -f Modelfile
```

### With Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")

messages = [
    {{"role": "system", "content": "You are a spelling correction assistant."}},
    {{"role": "user", "content": "Fix any spelling mistakes in this text. If there are no mistakes, output the text unchanged.\\n\\nI went to teh store."}},
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Training Data (v2)

Trained on ~6,265 curated examples with 3-step pipeline:
1. Clean text generation (Gemini API, age-appropriate stories)
2. Python error injection (42 dyslexic error rules + homophones)
3. Gemini augmentation (review and improve error quality)

## Model Variants

| Model | GGUF Q4_K_M | Best For |
|-------|-------------|----------|
| [0.6B](https://huggingface.co/{hf_user}/dyslexic-writer-qwen3-0.6b) | ~380 MB | Mobile/embedded |
| [1.7B](https://huggingface.co/{hf_user}/dyslexic-writer-qwen3-1.7b) | ~1 GB | Default |
| [4B](https://huggingface.co/{hf_user}/dyslexic-writer-qwen3-4b) | ~2.5 GB | Good balance |
| [14B](https://huggingface.co/{hf_user}/dyslexic-writer-qwen3-14b) | ~9 GB | Best quality |
"""

    try:
        create_repo(repo_id, exist_ok=True, repo_type="model")

        # Upload merged model weights
        if merged_dir.exists():
            print(f"  Uploading HF model weights from {merged_dir}...")
            for f in sorted(merged_dir.iterdir()):
                if f.name.startswith("checkpoint-") or f.name.startswith("."):
                    continue
                if f.is_file():
                    print(f"    Uploading {f.name}...")
                    api.upload_file(
                        path_or_fileobj=str(f),
                        path_in_repo=f.name,
                        repo_id=repo_id,
                    )

        # Upload GGUF files
        for suffix in ["f16", "q4_k_m", "q5_k_m", "q8_0"]:
            gguf_file = gguf_dir / f"{model_name}-{suffix}.gguf"
            if gguf_file.exists():
                print(f"  Uploading {gguf_file.name} ({gguf_file.stat().st_size / 1e9:.1f} GB)...")
                api.upload_file(
                    path_or_fileobj=str(gguf_file),
                    path_in_repo=gguf_file.name,
                    repo_id=repo_id,
                )

        # Upload README
        readme_path = Path(f"/tmp/README-{model_name}.md")
        readme_path.write_text(readme)
        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=repo_id,
        )

        print(f"  SUCCESS: https://huggingface.co/{repo_id}")

    except Exception as e:
        print(f"  ERROR uploading {model_name}: {e}")
        continue

print("\nAll uploads complete!")
UPLOAD_SCRIPT
fi

echo ""
echo "=========================================="
echo "ALL PHASES COMPLETE"
echo "Job finished: $(date)"
echo "=========================================="
