# Model Training Progress

**Last Updated**: January 2026

## Overview

Fine-tuning small LLMs for spelling correction on the Nibi cluster (DRAC/Alliance Canada). The goal is to create models that run locally on consumer hardware while catching spelling errors that dyslexic writers commonly make.

**Key Design Principle**: This is a **human-in-the-loop** system. Users review all suggestions before accepting them. Therefore:
- **False negatives (missed errors) are worse** than false positives
- We want aggressive models that catch errors, even at the cost of occasional bad suggestions
- Users can always reject incorrect suggestions with a single click

## Training Data

**Total**: ~192,000 examples

| Source | Examples | Description |
|--------|----------|-------------|
| GitHub Typo Corpus | 51,140 | Real typos from commit messages |
| Birkbeck corpus | 35,995 | Poor spellers, native English |
| Extra misspellings | 4,200 | Various sources |
| Holbrook corpus | 1,780 | Schoolchildren's writing |
| Synthetic (Gemini 3 Pro) | 6,117 | Generated for weak areas |

### Synthetic Data Breakdown

Generated with Gemini 3 Pro to cover patterns underrepresented in organic data:

| Category | Count | Examples |
|----------|-------|----------|
| Kids sentences | 3,968 | Full sentences with contextual errors |
| Word pairs | 2,149 | Targeted misspelling→correct pairs |
| **Contractions** | 186 | "dont" → "don't", "Im" → "I'm" |
| **Homophones** | 270 | "your/you're", "their/there/they're" |
| **Dyslexia patterns** | 187 | b/d confusion, letter reversals |

### Data Imbalance Issue

Contractions and homophones together represent only **0.1%** of training data. This caused smaller models to underperform on these critical categories for dyslexic writers.

## Training Runs

### Run 1: Qwen2.5 + SmolLM2 (Completed)

**Job**: 12-hour run on Nibi H100
**Status**: ✅ Completed successfully

Models trained:
- SmolLM2-360M-Instruct
- SmolLM2-1.7B-Instruct
- Qwen2.5-0.5B-Instruct
- Qwen2.5-1.5B-Instruct

**Issues encountered**:
1. First attempt got MIG slice (10GB) instead of full H100 → OOM on larger models
   - Fix: Changed `--gres=gpu:1` to `--gres=gpu:h100:1`
2. trl 0.26+ API change broke training
   - Fix: Changed `max_seq_length` to `max_length` in SFTConfig
3. flash-attn not available as Alliance wheel
   - Fix: Used PyTorch native SDPA (`attn_implementation="sdpa"`)

### Run 2: Qwen3 (Completed)

**Job ID**: 6680232
**Status**: ✅ Completed successfully (4h 13m)

Models trained:
- Qwen3-0.6B
- Qwen3-1.7B
- Qwen3-4B
- Qwen3-8B

**Issue encountered**:
- Qwen3 models default to "thinking mode" which outputs `<think>` reasoning tags
- Fix: Added `enable_thinking=False` to `apply_chat_template()` in test script
- Also added post-processing to strip any remaining `<think>...</think>` tags

## Test Results

### Test Suite

99 tests across 5 categories:

| Category | Tests | Purpose |
|----------|-------|---------|
| Pass-through | 10 | Correct text should stay unchanged |
| Known corrections | 60 | Basic typos, contractions, homophones, compound words |
| Edge cases | 8 | Proper nouns, technical terms, abbreviations |
| Dyslexia reversals | 11 | b/d, p/q, m/w confusion, letter order |
| Word preservation | 10 | No words added or deleted |

### Complete Results (All 8 Models)

| Model | Score | Disk Size | ~GGUF Q4 | Pass | Corrections | Edge | Dyslexia | Words |
|-------|-------|-----------|----------|------|-------------|------|----------|-------|
| **Qwen3-4B** | **98.0%** | 53 GB | ~2.5 GB | 10/10 | 58/60 | 8/8 | 11/11 | 10/10 |
| **Qwen3-8B** | **98.0%** | 107 GB | ~5 GB | 10/10 | 58/60 | 8/8 | 11/11 | 10/10 |
| SmolLM2-1.7B | 97.0% | 23 GB | ~1 GB | 10/10 | 57/60 | 8/8 | 11/11 | 10/10 |
| Qwen3-1.7B | 96.0% | 23 GB | ~1 GB | 10/10 | 56/60 | 8/8 | 11/11 | 10/10 |
| Qwen3-0.6B | 90.9% | 7.9 GB | ~400 MB | 10/10 | 51/60 | 8/8 | 11/11 | 10/10 |
| Qwen2.5-1.5B | 89.9% | 21 GB | ~1 GB | 10/10 | 50/60 | 8/8 | 10/11 | 10/10 |
| SmolLM2-360M | 89.9% | 4.8 GB | ~200 MB | 10/10 | 50/60 | 8/8 | 10/11 | 10/10 |
| Qwen2.5-0.5B | 82.8% | 6.5 GB | ~350 MB | 10/10 | 43/60 | 8/8 | 10/11 | 10/10 |

### Key Findings

#### Qwen3-4B/8B (Best Performers - 98%)
- **Near-perfect accuracy** at 98%
- Only 2 failures each (minor issues):
  - "Shouldnt go" → "You shouldn't go" (added subject)
  - "Nevermind that" → "Mind never that" (Qwen3-4B) or "What's up?" punctuation (Qwen3-8B)
- Excellent on all categories including dyslexia patterns
- 4B is the sweet spot: same accuracy as 8B at half the size

#### SmolLM2-1.7B (97%)
- Excellent for its size
- Only 3 failures, strong on contractions and homophones
- Best choice for ~1GB VRAM budget

#### Qwen3-0.6B (91%)
- Best small model (beats SmolLM2-360M)
- Good for low-resource devices
- Handles contractions better than older Qwen2.5

#### Qwen2.5 Models
- More aggressive (higher recall)
- Some semantic hallucinations (e.g., "their" → "to bed")
- Superseded by Qwen3 in all cases

### Failure Analysis

**Common failures across models**:

1. **Subject addition**: "Shouldnt go" → "You shouldn't go" (adds "You")
2. **Punctuation changes**: "Whats up" → "What's up?" (adds question mark)
3. **Rare compound words**: "Nevermind" handling varies

**Qwen2.5-specific issues** (fixed in Qwen3):
- Semantic drift: changing meaning while "correcting"
- Example: "I went their yesterday" → "I went to bed yesterday" (should be "there")

## Recommendations

### For Production Use

| Use Case | Model | Accuracy | VRAM |
|----------|-------|----------|------|
| **Best quality** | Qwen3-4B | 98% | ~2.5 GB |
| **Good quality, smaller** | SmolLM2-1.7B | 97% | ~1 GB |
| **Low-resource devices** | Qwen3-0.6B | 91% | ~400 MB |

**Key insight**: Qwen3-4B achieves 98% accuracy - same as 8B but half the size. This is the sweet spot for quality vs. resource tradeoff.

### Model Selection Guide

| Device Type | Recommended Model | VRAM | Accuracy |
|-------------|-------------------|------|----------|
| Gaming desktop (Intel Arc 12GB) | **Qwen3-4B** | ~2.5 GB | 98% |
| Laptop with dedicated GPU | SmolLM2-1.7B | ~1 GB | 97% |
| MacBook Air M1/M2 (8GB) | Qwen3-1.7B | ~1 GB | 96% |
| Integrated graphics / low RAM | Qwen3-0.6B | ~400 MB | 91% |
| Ultra-low-resource | SmolLM2-360M | ~200 MB | 90% |

### For Improving Smaller Models

If sub-1GB model performance matters, generate more targeted training data:

| Category | Current | Target | Gap |
|----------|---------|--------|-----|
| Contractions | 186 | 1,000+ | +814 |
| Homophones | 270 | 1,000+ | +730 |
| Dyslexia patterns | 187 | 500+ | +313 |

This would bring these critical categories from 0.1% to ~1.5% of training data.

## GGUF Conversion

### Status: ✅ Complete

Converted top models to GGUF Q4_K_M format for Ollama/llama.cpp:

| Model | Accuracy | F16 Size | Q4_K_M Size |
|-------|----------|----------|-------------|
| Qwen3-0.6B | 91% | 1.2 GB | **379 MB** |
| SmolLM2-1.7B | 97% | 3.2 GB | **1007 MB** |
| Qwen3-4B | 98% | 7.5 GB | **2.4 GB** |
| Qwen3-8B | 98% | 16 GB | **4.7 GB** |

**Output location**: `~/projects/def-jic823/dyslexic-writer/training/gguf_models/`

### Conversion Process

1. Clone llama.cpp to scratch (`$SLURM_TMPDIR`)
2. Build `llama-quantize` with ccache disabled (avoid home quota issues)
3. Convert HF model to GGUF F16 using `convert_hf_to_gguf.py`
4. Quantize to Q4_K_M using `llama-quantize`

## Next Steps

1. ✅ ~~Wait for Qwen3 training to complete~~
2. ✅ ~~Run test suite on Qwen3 models~~
3. ✅ ~~Compare all 8 models~~
4. ✅ ~~Convert best models to GGUF~~
5. 🧪 Test GGUF models locally with Ollama or llama.cpp
6. 📝 Create Ollama Modelfiles for each model
7. 🧪 Test on Intel Arc GPU with IPEX-LLM
8. 📤 (Optional) Upload to Hugging Face for sharing
9. 🔧 (Optional) Generate more Gemini data if sub-1GB performance needs improvement

## Files

| File | Purpose |
|------|---------|
| `training/finetune.py` | Main training script (Qwen2.5/SmolLM2) |
| `training/finetune_qwen3.py` | Qwen3 training script |
| `training/run_tests.py` | Comprehensive test suite (99 tests) |
| `training/train_nibi.slurm` | SLURM job for Qwen2.5/SmolLM2 |
| `training/train_qwen3.slurm` | SLURM job for Qwen3 |
| `training/test_models.slurm` | SLURM job for running tests |
| `training/convert_to_gguf.slurm` | SLURM job for GGUF conversion |
| `training/prepare_finetune_data.py` | Data preparation |
| `training/convert_synthetic_data.py` | Gemini data conversion |

### Model Directories (on Nibi)

**Base path**: `~/projects/def-jic823/dyslexic-writer/training/`

| Directory | Contents |
|-----------|----------|
| `outputs/` | Run 1 HF models (Qwen2.5, SmolLM2) |
| `outputs_qwen3/` | Run 2 HF models (Qwen3) |
| `gguf_models/` | Converted GGUF models |

**Full paths to GGUF models (ready for Ollama/llama.cpp):**
```
~/projects/def-jic823/dyslexic-writer/training/gguf_models/
├── Qwen3-0.6B-q4_k_m.gguf        # 379 MB, 91% accuracy
├── SmolLM2-1.7B-Instruct-q4_k_m.gguf  # 1007 MB, 97% accuracy
├── Qwen3-4B-q4_k_m.gguf          # 2.4 GB, 98% accuracy
├── Qwen3-8B-q4_k_m.gguf          # 4.7 GB, 98% accuracy
├── Qwen3-0.6B-f16.gguf           # 1.2 GB (backup)
├── SmolLM2-1.7B-Instruct-f16.gguf    # 3.2 GB (backup)
├── Qwen3-4B-f16.gguf             # 7.5 GB (backup)
└── Qwen3-8B-f16.gguf             # 16 GB (backup)
```

**Full paths to HuggingFace models (for further fine-tuning):**
```
~/projects/def-jic823/dyslexic-writer/training/outputs/
├── Qwen2.5-0.5B-Instruct/
├── Qwen2.5-1.5B-Instruct/
├── SmolLM2-360M-Instruct/
└── SmolLM2-1.7B-Instruct/

~/projects/def-jic823/dyslexic-writer/training/outputs_qwen3/
├── Qwen3-0.6B/
├── Qwen3-1.7B/
├── Qwen3-4B/
└── Qwen3-8B/
```

## Cluster Notes (Nibi/DRAC)

### Working Configuration

```bash
#SBATCH --gres=gpu:h100:1  # Must specify h100 to avoid MIG slice
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

module load python/3.11 cuda/12.6 arrow/21.0.0

# Use Alliance wheels
pip install --no-index torch transformers datasets trl accelerate peft

# Use PyTorch SDPA instead of flash-attn (not available as wheel)
# Set in code: attn_implementation="sdpa"
```

### Known Issues

1. **MIG slices**: Default `gpu:1` may give 10GB slice, not full 80GB H100
2. **flash-attn**: Not available as Alliance wheel, would require 2+ hour build
3. **trl 0.26+**: API changed from `max_seq_length` to `max_length`
4. **Qwen3 thinking mode**: Qwen3 models default to outputting `<think>` reasoning tags
   - Fix: Use `enable_thinking=False` in `apply_chat_template()`
   - Also strip any remaining `<think>...</think>` tags from output
