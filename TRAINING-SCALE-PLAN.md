# Plan: Scale Training Data Generation Using DGX Spark + GPT-OSS-120B

## Context

We have a working dyslexic spelling correction product deployed on GitHub Pages with a fine-tuned Qwen3-4B model (98% accuracy) served via Modal. The goal is to train **smaller models (0.5B-1.5B)** that match 4B quality so the tool can run on any laptop (8GB RAM, no GPU). To do this, we need significantly more training data — especially sentence-level examples across the full age range (7-17).

**Current data**: ~93K word-level pairs + ~6K synthetic sentences. The synthetic pipeline currently uses the paid Gemini API (~$15/3500 stories). We want to replace this with the **DGX Spark** (GB10 Blackwell, 128GB unified memory) running **GPT-OSS-120B** locally for free, unlimited generation.

**Key research insight**: Dyslexic spelling errors are a developmental delay — same error types as typical kids but persisting longer. Error profiles shift with age: phonological-dominant (7-9) → orthographic-dominant (10-12) → morphological-dominant (13-17). The existing error injection rule engine (`inject_errors.py`) is well-calibrated but only covers the 8-10 age band.

---

## Step 1: Set Up GPT-OSS-120B on DGX Spark

**Goal**: Get the 120B model running with maximum throughput.

**Critical optimization**: Use **MXFP4 format** (not Q4_K_M GGUF) to exploit Blackwell's FP4 tensor cores. Benchmarks show:
- MXFP4: **~40-60 tok/s** decode → 10K stories in ~7 hours
- Q4_K_M GGUF: **~4-10 tok/s** decode → 10K stories in 2-4 days

**Tasks**:
1. Check if GPT-OSS-120B is available in MXFP4 format on Hugging Face (NVIDIA's official quantization)
2. If not, use Q4_K_M GGUF — slower but still workable over a weekend
3. Build llama.cpp from source targeting Blackwell (`-DCMAKE_CUDA_ARCHITECTURES=121`)
4. Run as a server (`llama-server`) with flash attention enabled (`-fa 1`, `--no-mmap`)
5. Verify throughput with a test prompt

**Estimated time**: 2-3 hours setup

---

## Step 2: Expand Vocabulary Targets for Ages 7-17

**Goal**: The current `vocab_targets.json` targets grade 2-3 vocabulary. Expand to cover the full age range.

**File**: `training/vocab_targets.json`

**New age-band categories to add**:

**Young (7-9)** — Already covered. Add:
- CVC words with short vowel confusion targets (bed/bid, pen/pin, set/sit)
- Simple consonant clusters (bl, cr, st, sp, fl)
- High-frequency sight words that get misspelled (because, friend, people, said)

**Middle (10-12)** — New:
- Double consonant words (running, stopped, beginning, swimming, different)
- Silent letter words (knight, island, whistle, castle, science)
- Multi-syllable academic words (experiment, temperature, important, interesting)
- Homophone pairs in context (their/there/they're, to/too/two, your/you're)

**Teen (13-17)** — New:
- Latin/Greek root words (government, parliament, psychology, environment)
- Complex morphology (unnecessary, disappear, definitely, accommodation)
- Subject-specific vocabulary (photosynthesis, democracy, hypothesis, analysis)
- Teen-relevant proper nouns (Instagram, PlayStation, TikTok)

**Estimated time**: 1-2 hours to curate word lists

---

## Step 3: Create `generate_stories_local.py` — Local LLM Story Generator

**Goal**: Replace Gemini API with local GPT-OSS-120B inference. Generate 15,000+ clean stories.

**Base on**: `training/generate_clean_stories.py` (existing Gemini script)

**Key changes**:
- Replace `google.genai` client with HTTP calls to local llama-server (`http://localhost:8080/completion`)
- Add `--age-band` parameter: `young` (7-9), `middle` (10-12), `teen` (13-17)
- Age-specific prompts:
  - **Young**: "Write as an 8-year-old. Simple plot, short sentences, 80-120 words."
  - **Middle**: "Write as an 11-year-old doing a school assignment. 100-180 words."
  - **Teen**: "Write as a 14-year-old. Essay-style or journal entry. 150-250 words."
- Expand genre weights per age band (teens write more essays, young kids more adventure)
- Keep negative example generation (correct text that should pass through unchanged)
- Add resume capability (already exists in base script)
- No rate limiting needed (local server)

**Generation targets**:
| Age Band | Clean Stories | Negative Examples | Total |
|----------|-------------|-------------------|-------|
| Young (7-9) | 5,000 | 2,000 | 7,000 |
| Middle (10-12) | 5,000 | 2,000 | 7,000 |
| Teen (13-17) | 5,000 | 2,000 | 7,000 |
| **Total** | **15,000** | **6,000** | **21,000** |

At ~40 tok/s (MXFP4): ~8-10 hours. At ~5 tok/s (Q4): ~3-4 days.

**Estimated dev time**: 2-3 hours

---

## Step 4: Expand Error Injection Engine for Age Bands

**Goal**: Add age-weighted error profiles to `inject_errors.py`.

**File**: `training/inject_errors.py`

**New parameter**: `age_band` controls which rules fire and at what rates.

**Age-band error distributions** (from research):

| | Phonological | Orthographic | Morphological |
|-|-------------|--------------|---------------|
| Young (7-9) | **65%** | 22% | 13% |
| Middle (10-12) | 30% | **43%** | 27% |
| Teen (13-17) | 18% | **37%** | **33%** + 12% real-word |

**New rules to add**:

For **middle** band:
- Double consonant omission: running→runing, stopped→stoped (rate: 40%)
- Double consonant insertion: dining→dinning, coming→comming (rate: 20%)
- Silent letter drop: knife→nife, wrong→rong, island→iland (rate: 50%)
- Vowel digraph swap: heat→heet, fear→feer (rate: 30%)

For **teen** band:
- Latin prefix errors: disappear→disapear, unnecessary→unnessary (rate: 35%)
- Suffix confusion: definitely→definitly, separately→seperately (rate: 40%)
- Unstressed vowel reduction: government→goverment, different→diffrent (rate: 45%)
- Real-word substitution: form→from, tried→tired, quiet→quite (rate: 15%)
- Word boundary errors: a lot→alot, all right→alright (rate: 50%)

**Also add**:
- **Inconsistency injection**: Same word misspelled differently within one text (~10% chance). This is a hallmark of dyslexic writing that distinguishes it from typical errors.
- **Error density scaling by age**: Young ~15%, Middle ~10%, Teen ~7% (error rates decrease with age but persist on harder words)

**Estimated dev time**: 3-4 hours

---

## Step 5: Build Confusion Matrix from Existing Corpora

**Goal**: Create a character-level confusion matrix for data augmentation (NeuSpell-style noising).

**New script**: `training/build_confusion_matrix.py`

**Input data**:
- `training-data/all_pairs.csv` (93K pairs with edit distance)
- `training-data/birkbeck.dat` (36K misspellings)
- `training-data/holbrook.dat` (1.8K misspellings)

**Output**: `training/confusion_matrix.json` — character substitution probabilities weighted by real dyslexic error frequency.

**Use cases**:
- Additional data augmentation: apply confusion matrix noising to clean stories as an alternative to rule-based injection
- Can be combined with rule engine for richer error diversity
- Validates that rule engine probabilities align with corpus statistics

**Estimated dev time**: 2 hours

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

## Step 7: Assemble Training Dataset and Prepare for Fine-tuning

**Goal**: Combine all data sources into a unified training set.

**Reuse existing**: `training-data/combine_all.py` and `training/prepare_finetune_data.py`

**Final dataset composition target**:

| Source | Type | Approx Count |
|--------|------|-------------|
| Birkbeck + Holbrook (existing) | Word pairs | 38K |
| GitHub Typos (existing) | Word pairs | 51K |
| Extra misspellings (existing) | Word pairs | 4K |
| Old synthetic (existing) | Sentences | 6K |
| **New: Rule-injected stories** | **Sentences** | **~15K** |
| **New: Negative examples** | **Identity pairs** | **~6K** |
| **New: Confusion matrix augmented** | **Sentences** | **~10K** |
| **New: Lancaster LCCPW** | **Real children's writing** | **~500-1K** |
| **Total** | | **~130-140K** |

With sentence-level data going from ~6K to ~30K+ (5x increase), and the full 7-17 age range covered.

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

| Action | File |
|--------|------|
| Modify | `training/vocab_targets.json` — add middle/teen word lists |
| Modify | `training/inject_errors.py` — add age bands, new rules |
| Create | `training/generate_stories_local.py` — local LLM story generation |
| Create | `training/build_confusion_matrix.py` — character confusion matrix |
| Create | `training/scrape_lancaster.py` — LCCPW corpus extraction |
| Modify | `training-data/combine_all.py` — incorporate new data sources |
| Modify | `training/prepare_finetune_data.py` — handle new data volume |
