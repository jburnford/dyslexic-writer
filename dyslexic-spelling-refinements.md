# Refined Synthetic Training Data Pipeline: Dyslexic Spelling Patterns

## Addendum to Plan v2.0 — Critical Refinements for Quality Control at Scale

*February 2026*

---

## 1. The Central Methodological Problem

We have a rich but tiny sample (one child, one story), solid research literature, and a powerful generation API (Gemini). The risk is that at scale we drift from authentic patterns into plausible-but-ungrounded synthetic noise.

We are essentially trying to generalize from n=1 while staying empirically honest about doing so. The Bob story gives us a **specific error profile** that may or may not be representative of "dyslexic spelling" broadly. The research literature gives us the **distributional envelope** (what kinds of errors are attested, their relative frequencies in population studies), but our sample gives us a **particular instantiation** within that envelope.

### Key Decision: Scope of the Model

We need to decide whether we're training a model to correct **this child's** spelling or **dyslexic spelling generally** — because those are different targets with different data requirements.

**Recommendation**: Design the pipeline to do both. Build a **core profile** tightly calibrated to the Bob story, then build **variant profiles** that explore the research-attested parameter space more broadly. Label them distinctly so we can train and evaluate separately.

---

## 2. New Stage 0: Empirical Calibration from the Source Text

Before generating anything, conduct a systematic error audit of the Bob story.

### 2.1 What to Count

- Total word count
- Total content words (excluding top-50 Dolch sight words, articles, prepositions)
- Total errors
- Error rate (errors / content words)
- Classification of every error into the three-category taxonomy (phonological, morphological, orthographic)
- Distribution across categories (actual percentages, not assumed)
- Identification of which specific transformation rules are **directly attested** vs. **inferred**

### 2.2 Why This Matters

The probability values in the original plan (e.g., 0.7 for ea→e, 0.6 for sc→sk) are currently intuitive estimates. They should derive from the sample. If "screamed→skremed" appears twice and there are three total instances of 'ea' digraphs in the text, that's a 67% rate for that specific rule — but with enormous uncertainty given the sample size. Be explicit about confidence intervals, even informally.

### 2.3 Output

A reference document: **Bob Story Error Profile** — the ground truth against which all synthetic data is validated.

---

## 3. Architectural Principle: Separate Generation from Error Injection

### 3.1 Use Gemini for Clean Story Generation, NOT Error Injection

This is a critical architectural decision.

**Use the Gemini API to generate** clean, age-appropriate stories with vocabulary profiles that match the Bob story's complexity level.

**Do error injection locally** with your own rule engine, calibrated to the Bob story and the literature.

### 3.2 Rationale

If you ask Gemini to "write a story with dyslexic spelling errors," you'll get something that reflects the model's training data *about* dyslexia, which likely includes stereotypes, Hollywood representations, and generic "bad spelling" that isn't grounded in the research. Your rule engine is your quality control mechanism. Outsourcing error generation to the API is where drift will happen fastest.

### 3.3 Three Approved Uses of the Gemini API

1. **Clean story generation** with controlled vocabulary targeting (words containing consonant clusters, vowel digraphs, irregular past tenses, silent letters, multi-syllable words with unstressed vowels)
2. **Variant generation** — producing multiple clean versions of the same narrative with different phrasing, giving natural augmentation
3. **Batch formatting and validation** of your rule engine's output

### 3.4 Three Things to NOT Use the Gemini API For

1. Deciding what errors to introduce
2. Estimating error probabilities
3. Generating "dyslexic-style" text directly

Those decisions should stay in the calibrated rule engine.

---

## 4. Transformation Rule Provenance System

### 4.1 Three-Tier Provenance Classification

For each transformation rule, tag it with one of:

| Provenance | Definition | Example | Weight in Training |
|---|---|---|---|
| **Attested** | Directly observed in the Bob story | ea→e in "skremed" | High confidence; use freely |
| **Inferred** | Consistent with the Bob story's patterns but not directly observed | ea→e in other words like "please→plese" | Medium confidence; use with caution |
| **Literature-based** | Attested in dyslexia research but not observed in this sample | b/d reversals, word boundary errors | Lower confidence; label distinctly |

### 4.2 Why This Matters for Evaluation

This lets you later analyze whether the literature-based extrapolations actually improved model performance or whether you'd have been better off staying close to the attested patterns. You can train on attested-only, attested+inferred, and all three tiers separately and compare results.

---

## 5. Corrections to Original Plan

### 5.1 R-Intrusion Rule (§3.3.3: "back"→"bark")

The original plan includes an R-Intrusion/Metathesis category based on "back→bark" in the sample. This may be over-fitted to a single instance. R-intrusion is real in some dialects but isn't well-attested as a *spelling* error pattern in the dyslexia literature the way consonant cluster reduction is. 

**Recommendation**: Either drop it entirely or assign it a very low probability (~0.1) and tag it as provenance: "attested-uncertain."

### 5.2 Over-Regularization of Irregular Past Tenses (§3.2.1: "fell"→"falled")

The "falled" for "fell" example is worth being precise about. Over-regularization of irregular past tenses is more characteristic of younger typically-developing children and children with Specific Language Impairment (SLI) than it is specifically of dyslexia. The citation to Egan & Tainturier (2011) is appropriate, but their finding is more about **under-use of -ed on regular verbs** than over-regularization of irregulars.

**Recommendation**: Keep the rule but classify its provenance as "literature-based" rather than "attested," and be aware that including it may blur the boundary between dyslexic and SLI error profiles. Note this as a limitation.

### 5.3 Missing Rule Category: Word Boundary Errors

The original plan lacks a category for word boundary errors — running words together or splitting them incorrectly (e.g., "a lot"→"alot," "into"→"in to," "because"→"be cause"). These are common in dyslexic handwriting and would add ecological validity.

**Caveat**: This matters for ecological validity but may be out of scope depending on the input modality. If processing handwriting through OCR, word boundaries are partly an OCR problem. If working with typed input, word boundary errors are less common. Worth noting as a known limitation either way.

---

## 6. Adversarial Validation Loop (New Stage)

This is the key missing element from the original plan for quality control at scale.

### 6.1 Automated Adversarial Check

After generating a batch, use Gemini (or Claude) as a **critic** rather than a generator:

1. Present it with pairs of error-injected sentences — one from your rule engine, one with random plausible-looking errors
2. Ask it to identify which looks more like authentic dyslexic writing and why
3. Flag sentences where the critic can't distinguish or finds the synthetic version implausible

This isn't definitive, but it catches obvious implausibilities at scale.

### 6.2 Expert Review (Critical)

If you can get even a small amount of feedback from a specialist — an educational psychologist, a literacy specialist, or a speech-language pathologist who works with dyslexic children — a few hours of their review on a pilot batch would be worth more than thousands of additional synthetic pairs.

**Suggested protocol**: Present 50 synthetic error-injected sentences alongside 10 real sentences from the Bob story (unlabelled). Ask the specialist to rate each on a 1–5 scale for "authenticity as dyslexic writing" and flag any that feel wrong. Use their feedback to recalibrate rules before scaling up.

---

## 7. Batch-Based Quality Control Protocol

For a 2,000-pair corpus, generate in batches of 200 pairs.

### After Each Batch:

1. **Automated checks**:
   - Edit distance within expected range
   - Preservation list respected (top-50 sight words, proper nouns)
   - Error density within ±5% of target rate derived from Stage 0 audit
   - No impossible letter combinations introduced
   - Error type distribution within expected range (phonological ~45%, orthographic ~30%, morphological ~25%)

2. **Manual review** of 20 pairs (~10%):
   - Ask: "Could this sentence plausibly appear in the same notebook as the Bob story?"
   - Flag any transformation rule that produces implausible results
   - Adjust probability or remove rule before next batch

3. **Changelog**:
   - Track all rule adjustments with rationale
   - Record which rules were added, modified, or removed at each batch
   - This makes the process reproducible and auditable

---

## 8. Revised Annotation Schema

The original schema (§7.3) should be expanded with provenance and confidence fields:

```json
{
  "id": "story_001_sent_01",
  "input": "Bob was a Bowring guy",
  "target": "Bob was a boring guy",
  "errors": [
    {
      "target_word": "boring",
      "written_as": "Bowring",
      "rule": "vowel_digraph_phonetic",
      "category": "phonological",
      "provenance": "attested",
      "confidence": "high"
    }
  ],
  "error_count": 1,
  "word_error_rate": 0.17,
  "source_story": "story_001",
  "batch": "batch_003",
  "reviewed": false
}
```

### New Fields Explained

| Field | Purpose |
|---|---|
| `provenance` | "attested", "inferred", or "literature-based" — ties each error to its evidential basis |
| `confidence` | "high", "medium", "low" — reflects certainty that this error type belongs in the profile |
| `batch` | Tracks which generation batch produced this pair, for diagnosing systematic issues |
| `reviewed` | Whether a human has validated this pair |

---

## 9. Gemini Prompt Design for Clean Story Generation

### 9.1 Principles

- Specify reading level, word count, genre, and structural requirements
- Include a **vocabulary target list** of words that will trigger known error rules (words with consonant clusters, vowel digraphs, irregular past tenses, etc.)
- Request varied sentence complexity (simple, compound, complex)
- Request inclusion of dialogue
- Do NOT mention dyslexia, spelling errors, or the downstream purpose

### 9.2 Example Prompt Template

```
Write a short adventure story for a child aged 8-10. The story should:
- Be 150-300 words long
- Have a clear beginning, middle, and end
- Include at least 3 lines of dialogue
- Use a mix of simple and compound sentences
- Be written at approximately a Grade 3 reading level

The story must naturally include the following words (or close variants): 
[INSERT VOCABULARY TARGETS FROM RULE ENGINE — e.g., screamed, building, 
dropped, apartment, silence, balcony, grabbed, walked, morning, different, 
frightened, beautiful, climbing, breakfast, knocked]

Do not force these words unnaturally. If some don't fit, skip them.
Write only the story, no title or commentary.
```

### 9.3 Vocabulary Target Generation

Maintain a master list of English words organized by which transformation rules they trigger. Rotate through this list across stories to ensure broad coverage. Examples:

| Rule Triggered | Target Words |
|---|---|
| ea digraph | screamed, dreaming, pleased, reason, reached, treated, feared |
| Consonant cluster | grabbed, dropped, trust, climbing, struggled, sprint, crashed |
| Irregular past tense | fell, flew, ran, caught, thought, brought, knew, threw |
| Silent letters | knight, wrong, climb, honest, island, knife, listen |
| Unstressed syllables | apartment, different, beautiful, elephant, hospital, important |

---

## 10. Fine-Tuning Considerations

### 10.1 For Parameter-Efficient Fine-Tuning (LoRA or Similar)

A 2,000–3,000 pair corpus is reasonable if fine-tuning an existing model with strong GEC (Grammatical Error Correction) capabilities.

### 10.2 Key Caveat on Existing GEC Corpora

The BEA 2019 and C4_200M corpora cited in the original plan are dominated by **L2 English errors**, which have a substantially different distribution from dyslexic L1 errors. L2 errors often involve article usage, preposition choice, and verb agreement — categories largely absent from dyslexic writing. Transfer from L2 GEC baselines may be limited, and this should be evaluated explicitly rather than assumed.

### 10.3 Evaluation Strategy

- Hold out 10% of generated pairs as a test set
- Additionally, hold out a small set of **real** Bob story sentences (if available beyond the two-page sample) as an ecological validity check
- Compare model performance on attested-only vs. attested+inferred vs. all-provenance training sets
- Report performance broken down by error category (phonological, morphological, orthographic)

---

## 11. Known Limitations to Document

1. **n=1 source sample**: All "attested" rules derive from a single child's single story. This child's error profile may not generalize.
2. **Age and severity unknown**: Without knowing the child's exact age and diagnostic profile, we can't calibrate error density or type distribution precisely.
3. **Handwriting vs. typed modality**: The Bob story is handwritten; some errors may reflect motor/graphomotor difficulties rather than spelling knowledge. Our synthetic data assumes typed input.
4. **Word boundary errors excluded**: A known feature of dyslexic writing not modelled in this pipeline.
5. **L2 vs. L1 baseline mismatch**: Available GEC benchmarks are predominantly L2, limiting direct comparability.
6. **Over-regularization ambiguity**: Some morphological rules (irregular past tense over-regularization) overlap with SLI profiles and younger typically-developing writers' patterns.

---

## 12. Revised Implementation Roadmap

### Phase 0: Empirical Calibration (Week 1)
- [ ] Conduct systematic error audit of Bob story
- [ ] Compute actual error rates and category distributions
- [ ] Classify each observed error by provenance
- [ ] Produce **Bob Story Error Profile** reference document
- [ ] Establish confidence intervals / uncertainty notes for low-frequency rules

### Phase 1: Rule Engine Development (Week 2-3)
- [ ] Finalize transformation rule library with provenance tags
- [ ] Calibrate probabilities to Stage 0 audit
- [ ] Build error injection module (local, not API-dependent)
- [ ] Create preservation word lists (Dolch top 50, proper nouns)
- [ ] Build vocabulary target lists organized by triggerable rules

### Phase 2: Clean Story Generation (Week 3-4)
- [ ] Design and test Gemini prompts for clean story generation
- [ ] Generate 50 pilot stories
- [ ] Review vocabulary coverage against rule targets
- [ ] Iterate on prompt design

### Phase 3: Pilot Error Injection + Validation (Week 4-6)
- [ ] Apply error injection to pilot stories
- [ ] Run adversarial validation loop (LLM-as-critic)
- [ ] Seek expert review on pilot batch (if possible)
- [ ] Refine rules based on feedback
- [ ] Establish batch QA protocol and changelog

### Phase 4: Scale Production (Week 6-9)
- [ ] Generate full corpus (2,000+ pairs) in batches of 200
- [ ] Run automated validation after each batch
- [ ] Manual review of 10% sample per batch
- [ ] Track and document all rule adjustments
- [ ] Stratify final corpus by provenance tier

### Phase 5: Training + Evaluation (Week 9-12)
- [ ] Split corpus: 90% train, 10% test (stratified by provenance)
- [ ] Fine-tune correction model (LoRA on existing GEC-capable model)
- [ ] Evaluate on held-out test set, broken down by error category
- [ ] Compare attested-only vs. attested+inferred vs. full training sets
- [ ] Compare to baseline GEC model performance
- [ ] If available, evaluate on real Bob story sentences
- [ ] Document results and iterate

---

*This document is an addendum to the Synthetic Training Data Plan v2.0. It should be read alongside the original plan, which contains the full error taxonomy, research framework, and references.*
