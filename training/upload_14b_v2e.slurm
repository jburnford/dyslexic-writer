#!/bin/bash
#SBATCH --job-name=upload-14b
#SBATCH --output=logs/upload-14b-%j.out
#SBATCH --error=logs/upload-14b-%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Upload v2e 14B GGUF models to HuggingFace
# No GPU needed - just network access

set -e

echo "Job started: $(date)"
echo "Node: $(hostname)"

mkdir -p logs
module load python/3.11

VENV_DIR=$SLURM_TMPDIR/venv
virtualenv --no-download $VENV_DIR
source $VENV_DIR/bin/activate

pip install --no-index --upgrade pip
pip install gguf huggingface_hub 2>/dev/null || pip install --user gguf huggingface_hub

cd $SLURM_SUBMIT_DIR

HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null)
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: No HuggingFace token found."
    exit 1
fi

python3 << 'UPLOAD_SCRIPT'
import os
import json
from pathlib import Path
from huggingface_hub import HfApi, create_repo

api = HfApi()
repo_id = "jburnford/dyslexic-writer-qwen3-14b"
gguf_dir = Path("gguf_models_v2e")

# Load v2e eval results
eval_file = Path("outputs_qwen3_v2e/eval_results.json")
metrics = {}
if eval_file.exists():
    with open(eval_file) as f:
        data = json.load(f)
    for r in data.get("synthetic", data if isinstance(data, list) else []):
        if "14B" in r.get("model", ""):
            metrics = r
            break

accuracy = metrics.get("exact_match_accuracy", 0) * 100
error_fix = metrics.get("has_error_accuracy", 0) * 100
no_err = metrics.get("no_error_accuracy", 0) * 100

readme = f"""---
license: apache-2.0
base_model: Qwen/Qwen3-14B
tags:
  - dyslexia
  - spelling-correction
  - assistive-technology
  - gguf
  - qwen3
language:
  - en
pipeline_tag: text-generation
---

# Dyslexic Writer - Qwen3-14B (v2e)

A fine-tuned Qwen3-14B model that corrects dyslexic spelling errors while preserving the writer's unique voice, style, and word choices.

## Model Details

- **Base Model**: Qwen/Qwen3-14B
- **Training**: LoRA (r=32) with embed_tokens and lm_head on v2e dataset
- **Training Data**: 5,638 examples of dyslexic text → corrected text pairs
- **Task**: Fix spelling/grammar errors WITHOUT changing voice, style, or word choice

## Eval Results (v2e)

| Metric | Score |
|--------|-------|
| Exact Match | {accuracy:.1f}% |
| Error Fix Rate | {error_fix:.1f}% |
| No-Error Preservation | {no_err:.1f}% |

## GGUF Quantizations

| File | Size | Description |
|------|------|-------------|
| Qwen3-14B-f16.gguf | 28 GB | Full precision |
| Qwen3-14B-q8_0.gguf | 15 GB | 8-bit quantized |
| Qwen3-14B-q5_k_m.gguf | 9.8 GB | 5-bit quantized (recommended) |
| Qwen3-14B-q4_k_m.gguf | 8.4 GB | 4-bit quantized |

## Usage

Use with Ollama or llama.cpp. The prompt format is:

```
Fix only the spelling and grammar errors in the following text. Keep the author's voice, word choices, and style exactly as they are. Do not rewrite or rephrase anything. Only correct clear mistakes:

[text here]
```

**Important**: Disable Qwen3 thinking mode (`enable_thinking=False` / `/no_think`).

## Training Details

- **Dataset**: Synthetic dyslexic text pairs generated via phonetic misspeller + Gemini augmentation
- **Error Types**: Phonetic misspellings, letter transpositions, missing letters, wrong homophones, missing apostrophes
- **LoRA Targets**: All attention + MLP layers + embed_tokens + lm_head
"""

print(f"Creating/updating repo: {repo_id}")
create_repo(repo_id, exist_ok=True, repo_type="model")

# Upload README
api.upload_file(
    path_or_fileobj=readme.encode(),
    path_in_repo="README.md",
    repo_id=repo_id,
    commit_message="Update README for v2e model",
)
print("README uploaded")

# Upload GGUF files
gguf_files = sorted(gguf_dir.glob("Qwen3-14B*.gguf"))
for f in gguf_files:
    size_gb = f.stat().st_size / (1024**3)
    print(f"Uploading {f.name} ({size_gb:.1f} GB)...")
    api.upload_file(
        path_or_fileobj=str(f),
        path_in_repo=f.name,
        repo_id=repo_id,
        commit_message=f"Upload {f.name} (v2e)",
    )
    print(f"  Done: {f.name}")

print("\nAll uploads complete!")
print(f"https://huggingface.co/{repo_id}")
UPLOAD_SCRIPT

echo ""
echo "Job finished: $(date)"
