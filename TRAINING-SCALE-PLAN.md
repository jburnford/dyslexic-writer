# Plan: Scale Training Data Generation Using DGX Spark + GPT-OSS-120B

## Context

We have a working dyslexic spelling correction product deployed on GitHub Pages with a fine-tuned Qwen3-4B model (98% accuracy) served via Modal. The goal is to train **smaller models (0.5B-1.5B)** that match 4B quality so the tool can run on any laptop (8GB RAM, no GPU). To do this, we need significantly more training data — especially sentence-level examples across the full age range (7-17).

**Current data**: ~93K word-level pairs + ~6K synthetic sentences. The synthetic pipeline currently uses the paid Gemini API (~$15/3500 stories). We want to replace this with the **DGX Spark** (GB10 Blackwell, 128GB unified memory) running **GPT-OSS-120B** locally for free, unlimited generation.

**Key research insight**: Dyslexic spelling errors are a developmental delay — same error types as typical kids but persisting longer. Error profiles shift with age: phonological-dominant (7-9) → orthographic-dominant (10-12) → morphological-dominant (13-17). The existing error injection rule engine (`inject_errors.py`) is well-calibrated but only covers the 8-10 age band.

---

## Step 1: Set Up GPT-OSS-120B on DGX Spark — DONE

**Goal**: Get the 120B model running with maximum throughput.

**What we found**:
- Model already running via **Ollama** (not llama.cpp) — 100% GPU, 70GB VRAM
- Measured throughput: **~42 tok/s** decode
- Effective story generation rate: **~160 stories/hr** (~22s per story including reasoning overhead)
- Full 21K generation (15K stories + 6K negatives across 3 age bands) = **~130 hours (~5 days)**

**Critical discovery**: GPT-OSS-120B is a **reasoning model** — it uses internal thinking tokens before producing content. This means:
- Ollama native `/api/generate` works best (thinking goes in `thinking` field, content in `response`)
- OpenAI-compatible `/v1/chat/completions` puts reasoning in `reasoning` field, often consuming all `max_tokens` before producing content
- Need **1536-2048 num_predict** to budget for both reasoning (~500-700 tok) and content (~200-400 tok)
- Script auto-detects Ollama vs vLLM and uses the appropriate API

**Also planned**: vLLM on USask Plato cluster as alternative/parallel generation endpoint. Script supports both via `--url` flag.

---

## Step 2: Expand Vocabulary Targets for Ages 7-17 — DONE

**File**: `training/vocab_targets.json`

Added 8 new age-band categories:
- **Young**: `young_cvc_short_vowel` (40 words), `young_consonant_clusters` (42 words), `young_sight_words` (44 words)
- **Middle**: `middle_double_consonants` (49 words), `middle_silent_letters` (45 words), `middle_academic_words` (36 words), `middle_homophones` (46 words)
- **Teen**: `teen_latin_greek_roots` (32 words), `teen_complex_morphology` (32 words), `teen_subject_vocabulary` (36 words), `teen_proper_nouns` (20 words), `teen_word_boundary` (10 phrases)

---

## Step 3: Create `generate_stories_local.py` — Local LLM Story Generator — DONE (script ready, generation pending)

**File**: `training/generate_stories_local.py`

**Implemented**:
- Auto-detects Ollama (native API) vs vLLM/OpenAI-compatible backend
- `--age-band young/middle/teen/all` with age-specific system prompts and genre weights
- Age-specific negative example prompts (4 variants per age band)
- Reasoning model handling: extracts stories from `thinking` field when `response` is empty
- Quality filter: rejects meta-commentary and responses under 40 words
- `--resume-from` for interrupted runs, `--pilot` for 10-story test batches
- `--url` and `--model` flags for flexible deployment (DGX Spark, Plato cluster, etc.)

**Pilot results**: 10/10 stories generated successfully across all 3 age bands. Quality is age-appropriate.

**Generation targets** (unchanged):
| Age Band | Clean Stories | Negative Examples | Total |
|----------|-------------|-------------------|-------|
| Young (7-9) | 5,000 | 2,000 | 7,000 |
| Middle (10-12) | 5,000 | 2,000 | 7,000 |
| Teen (13-17) | 5,000 | 2,000 | 7,000 |
| **Total** | **15,000** | **6,000** | **21,000** |

**Revised time estimate**: ~130 hours (~5 days) on DGX Spark at 160 stories/hr.

**To run**: `python3 -u training/generate_stories_local.py --url http://localhost:11434 --model gpt-oss:120b --age-band all --seed 42`

---

## Step 4: Expand Error Injection Engine for Age Bands — DONE

**File**: `training/inject_errors.py`

**Implemented**:
- `--age-band young/middle/teen` flag with distinct error profiles
- Age-band category weights: young=phonological-dominant, middle=orthographic-dominant, teen=morphological-dominant
- Error density scaling: young ~15%, middle ~10%, teen ~7%
- `--inconsistency` flag for within-text variation

**8 new rules added** (all tested and firing at correct rates):

| Rule | Band | Rate | Example |
|------|------|------|---------|
| `double_consonant_omission` | middle | 40% | stopped→stoped |
| `double_consonant_insertion` | middle | 20% | dining→dinning |
| `silent_letter_drop` | middle | 50% | knife→nife |
| `vowel_digraph_swap` | middle | 30% | heat→heet |
| `latin_prefix_error` | teen | 35% | unnecessary→unecessary |
| `suffix_confusion` | teen | 40% | definitely→definitly |
| `unstressed_vowel_reduction` | teen | 45% | government→govrnment |
| `real_word_substitution` | teen | 15% | quiet→quite |
| `word_boundary` (text-level) | teen | 50% | a lot→alot |

---

## Step 5: Build Confusion Matrix from Existing Corpora — DONE

**Files**: `training/build_confusion_matrix.py` → `training/confusion_matrix.json`

**Results**: Processed 131,768 pairs → 285,919 error alignments:
- Substitutions: 151,754
- Deletions: 88,388
- Insertions: 45,777

**Top confusions**: z→s (.645), i→e (.304), a→e (.287), c→s (.324), d→t (.221)

---

## Step 6: Incorporate Lancaster LCCPW Corpus

**Goal**: Extract real children's writing with authentic spelling errors.

**Source**: https://www.lancaster.ac.uk/fass/projects/lever/
- 37 UK school children, ages 8-11, written over 3 years
- SGML transcriptions with original spelling
- Core sample: 12 children with complete data across 3 project series

**Tasks**:
1. Download available SGML transcriptions from the LEVER website
2. Parse SGML to extract raw text with original (uncorrected) spelling
3. Create correction pairs by aligning with any available corrected versions, or manually reviewing
4. Add to training data with `source: "lancaster_lccpw"` tag

**Note**: This is a small but high-value dataset — real errors from real kids at exactly the right age range. Quality > quantity here.

**Estimated dev time**: 2-3 hours

---

## Step 7: Assemble Training Dataset and Prepare for Fine-tuning — DONE (pipeline ready)

**Files**: `training-data/combine_all.py` and `training/prepare_finetune_data.py` (both updated)

**Pipeline tested** with existing data (before new story generation):
- `combine_all.py` now loads age-band story batches + negatives from `generated_stories/<age_band>/`
- `prepare_finetune_data.py` now applies `inject_errors.py` age-band error injection to clean stories automatically
- Existing 3,498 stories + 750 negatives expanded to **50,955 training examples** via error injection
- Current total: **237,185 examples** (93K word pairs + 93K sentence context + 51K synthetic)
- Train/eval split: 213K / 24K

**After 15K new stories are generated**, re-running the pipeline will produce the final dataset.

**Output formats** (same as existing):
- `train.jsonl` / `eval.jsonl` (90/10 split)
- Chat template format for Qwen3 and SmolLM2

---

## Step 8: Train Smaller Models on Nibi H100

**Goal**: Fine-tune 0.5B and 1.5B models, compare to 4B baseline.

**Reuse existing**: `training/finetune_qwen3.py`, SLURM scripts

**Candidates**:
| Model | GGUF Size | Target |
|-------|----------|--------|
| Qwen3-0.6B | 379MB | Runs on anything |
| SmolLM2-1.7B | 1.0GB | Sweet spot for 8GB laptops |
| Qwen3-1.7B | ~1.0GB | Alternative 1.5B candidate |

**Evaluation**: Run existing 99-test suite + Bob Story eval. Target: 1.5B model reaching **95%+ accuracy** (vs current 4B at 98%).

**Estimated cluster time**: 4-6 hours on H100

---

## Verification

1. **Story generation quality**: Generate 10 pilot stories per age band, manually review for age-appropriateness and vocabulary
2. **Error injection quality**: Run inject_errors.py on pilot stories, check error rates match targets per age band
3. **Training data format**: Verify JSONL output loads correctly in SFTTrainer
4. **Model evaluation**: Run `training/run_tests.py` (99 tests) + `training/evaluate_bob_story.py` on trained models
5. **End-to-end**: Load trained GGUF into Ollama, test via Flask backend (`app/server.py`), verify web UI works

---

## File Summary

| Status | File | Description |
|--------|------|-------------|
| DONE | `training/vocab_targets.json` | Added 8 age-band categories (young/middle/teen) |
| DONE | `training/inject_errors.py` | Added `--age-band`, 8 new rules, inconsistency injection |
| DONE | `training/generate_stories_local.py` | Local LLM story gen (Ollama + vLLM auto-detect) |
| DONE | `training/build_confusion_matrix.py` | Character confusion matrix (131K pairs → 286K alignments) |
| DONE | `training/confusion_matrix.json` | Output: substitution/deletion/insertion probabilities |
| DONE | `training-data/combine_all.py` | Updated for age-band story batches |
| DONE | `training/prepare_finetune_data.py` | Auto-applies error injection to clean stories |
| TODO | `training/scrape_lancaster.py` | LCCPW corpus extraction (Step 6) |
| PENDING | Story generation | `python3 -u training/generate_stories_local.py --url http://localhost:11434 --model gpt-oss:120b --age-band all --seed 42` (~5 days) |
