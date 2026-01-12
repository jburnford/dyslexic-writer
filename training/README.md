# Spelling Correction Model Training

Fine-tune small LLMs for spelling correction, optimized for local inference on consumer hardware.

## Models

We train 4 small models that can run on MacBook Air / Windows laptops:

| Model | Size (Q4) | RAM | Good For |
|-------|-----------|-----|----------|
| SmolLM2-360M-Instruct | ~200MB | ~500MB | Ultra-low resource devices |
| Qwen2.5-0.5B-Instruct | ~350MB | ~1GB | Best quality at tiny size |
| Qwen2.5-1.5B-Instruct | ~1GB | ~2GB | Sweet spot for quality/size |
| SmolLM2-1.7B-Instruct | ~1GB | ~2GB | Edge-optimized |

## Training Data

~99K unique examples (~192K with sentence context variants) from:
- GitHub Typo Corpus (51K real typos)
- Birkbeck corpus (36K from poor spellers)
- Holbrook corpus (1.8K from schoolchildren)
- Extra misspellings (4.2K various)
- Synthetic data (6.1K Gemini 3 Pro generated):
  - Kids/elementary writing patterns (3,968 sentences)
  - Word pairs for K-3 vocabulary (2,149 pairs)
  - Covers: homophones, contractions, dyslexia reversals, compound words, silent letters

## Quick Start on DARC Cluster

```bash
# Clone repo (if not already)
git clone git@github.com:jburnford/dyslexic-writer.git
cd dyslexic-writer/training

# Submit job
sbatch train.slurm
```

## Manual Steps

### 1. Prepare Data

```bash
python prepare_finetune_data.py
```

Creates:
- `train.jsonl` - 90% of data (~173K examples)
- `eval.jsonl` - 10% of data (~19K examples)

### 2. Train Models

```bash
# Train all 4 models
python finetune.py --model all --output-dir ./outputs

# Or train a specific model
python finetune.py --model Qwen/Qwen2.5-0.5B-Instruct
```

### 3. Evaluate

```bash
# Evaluate all trained models
python evaluate.py --all --output-dir ./outputs

# Evaluate specific model
python evaluate.py --model-dir ./outputs/Qwen2.5-0.5B-Instruct
```

## Test Results (January 2026)

| Model | Accuracy | Notes |
|-------|----------|-------|
| **SmolLM2-1.7B** | **97.0%** | Best performer, production ready |
| Qwen2.5-1.5B | 89.9% | Good but some hallucinations |
| SmolLM2-360M | 89.9% | Too conservative (misses contractions) |
| Qwen2.5-0.5B | 82.8% | Most hallucinations |

Training time on H100 80GB (~4-6 hours total for all 4 models):
- SmolLM2-360M: ~30 min
- Qwen2.5-0.5B: ~45 min
- Qwen2.5-1.5B: ~1.5 hours
- SmolLM2-1.7B: ~1.5 hours

See [../TRAINING_PROGRESS.md](../TRAINING_PROGRESS.md) for detailed analysis.

## Output Files

After training:
```
outputs/
├── SmolLM2-360M-Instruct/
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── training_metrics.json
├── Qwen2.5-0.5B-Instruct/
│   └── ...
└── evaluation_results.json
```

## Converting for Local Use

After training, convert to GGUF for llama.cpp/Ollama:

```bash
# Install llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && make

# Convert to GGUF
python convert_hf_to_gguf.py ../outputs/Qwen2.5-0.5B-Instruct --outfile spelling-qwen-0.5b.gguf

# Quantize to Q4
./llama-quantize spelling-qwen-0.5b.gguf spelling-qwen-0.5b-q4.gguf Q4_K_M
```

Then use with Ollama:
```bash
ollama create spelling-model -f Modelfile
```

## Adjusting for Your Cluster

Edit `train.slurm`:

```bash
# Uncomment and adjust module loads
module load cuda/12.1
module load python/3.11

# Adjust partition name
#SBATCH --partition=your-gpu-partition

# Adjust GPU type if needed
#SBATCH --gres=gpu:h100:1
```
