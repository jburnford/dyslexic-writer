#!/bin/bash
#SBATCH --job-name=bob-eval
#SBATCH --output=/scratch/jic823/dyslexic-writer/logs/bob-eval-%j.out
#SBATCH --error=/scratch/jic823/dyslexic-writer/logs/bob-eval-%j.err
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

echo "=========================================="
echo "Bob Story Evaluation - $(date)"
echo "=========================================="

module load python/3.11 cuda/12.6 arrow/21.0.0

VENV_DIR=$SLURM_TMPDIR/venv
virtualenv --no-download $VENV_DIR
source $VENV_DIR/bin/activate

pip install --no-index --upgrade pip
pip install --no-index torch torchvision
pip install --no-index transformers accelerate sentencepiece
pip install tokenizers --upgrade 2>/dev/null || true

export HF_HOME=/scratch/jic823/dyslexic-writer/hf_cache

nvidia-smi --query-gpu=name,memory.total --format=csv

WORK_DIR=/scratch/jic823/dyslexic-writer
OUTPUT_DIR=$WORK_DIR/outputs_v2
EVAL_DIR=$WORK_DIR/eval_results
mkdir -p $EVAL_DIR

python3 << 'PYEOF'
import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# Bob story evaluation pairs
BOB_STORY_EVAL = [
    {"input": "Bob was a Bowring guy, he never went to a party he didn't need to for work.",
     "target": "Bob was a boring guy, he never went to a party he didn't need to for work.",
     "errors": [{"written": "Bowring", "target": "boring", "category": "phonological"}]},
    {"input": "he walked out on to his belkany to get some fresh air",
     "target": "he walked out on to his balcony to get some fresh air",
     "errors": [{"written": "belkany", "target": "balcony", "category": "phonological"}]},
    {"input": "he hate going outside",
     "target": "he hated going outside",
     "errors": [{"written": "hate", "target": "hated", "category": "morphological"}]},
    {"input": "he skremed at the hights",
     "target": "he screamed at the heights",
     "errors": [{"written": "skremed", "target": "screamed", "category": "phonological"},
                {"written": "hights", "target": "heights", "category": "orthographic"}]},
    {"input": "the ailens falled from the selen",
     "target": "the aliens fell from the ceiling",
     "errors": [{"written": "ailens", "target": "aliens", "category": "orthographic"},
                {"written": "falled", "target": "fell", "category": "morphological"},
                {"written": "selen", "target": "ceiling", "category": "phonological"}]},
    {"input": "he fented and the he got up agan",
     "target": "he fainted and then he got up again",
     "errors": [{"written": "fented", "target": "fainted", "category": "phonological"},
                {"written": "the", "target": "then", "category": "orthographic"},
                {"written": "agan", "target": "again", "category": "phonological"}]},
    {"input": "he hered skreming from the hights",
     "target": "he heard screaming from the heights",
     "errors": [{"written": "hered", "target": "heard", "category": "phonological"},
                {"written": "skreming", "target": "screaming", "category": "phonological"},
                {"written": "hights", "target": "heights", "category": "orthographic"}]},
    {"input": "he see a man fall from a bailden",
     "target": "he saw a man fell from a building",
     "errors": [{"written": "see", "target": "saw", "category": "morphological"},
                {"written": "fall", "target": "fell", "category": "morphological"},
                {"written": "bailden", "target": "building", "category": "phonological"}]},
    {"input": "the bird flaw fast and gab him",
     "target": "the bird flew fast and grabbed him",
     "errors": [{"written": "flaw", "target": "flew", "category": "phonological"},
                {"written": "gab", "target": "grabbed", "category": "phonological"}]},
    {"input": "he drope him on the roof and he was safe",
     "target": "he dropped him on the roof and he was safe",
     "errors": [{"written": "drope", "target": "dropped", "category": "orthographic"}]},
    {"input": "thats how Bob fall from the hights",
     "target": "that's how Bob fell from the heights",
     "errors": [{"written": "thats", "target": "that's", "category": "orthographic"},
                {"written": "fall", "target": "fell", "category": "morphological"},
                {"written": "hights", "target": "heights", "category": "orthographic"}]},
    {"input": "he tried bugy-juming from the belkany",
     "target": "he tried bungee jumping from the balcony",
     "errors": [{"written": "bugy-juming", "target": "bungee jumping", "category": "phonological"},
                {"written": "belkany", "target": "balcony", "category": "phonological"}]},
    {"input": "Mow he is broken but safe",
     "target": "Now he is broken but safe",
     "errors": [{"written": "Mow", "target": "Now", "category": "orthographic"}]},
]

SYSTEM_PROMPT = "You are a spelling correction assistant. Fix only spelling and grammar errors. Do not change meaning, names, or correct text. If the text is already correct, return it unchanged."

OUTPUT_DIR = Path("/scratch/jic823/dyslexic-writer/outputs_v2")
EVAL_DIR = Path("/scratch/jic823/dyslexic-writer/eval_results")

MODELS = [
    ("SmolLM2-360M-Instruct", OUTPUT_DIR / "SmolLM2-360M-Instruct"),
    ("Qwen2.5-0.5B-Instruct", OUTPUT_DIR / "Qwen2.5-0.5B-Instruct"),
    ("Qwen2.5-1.5B-Instruct", OUTPUT_DIR / "Qwen2.5-1.5B-Instruct"),
    ("SmolLM2-1.7B-Instruct", OUTPUT_DIR / "SmolLM2-1.7B-Instruct"),
]


def run_inference(model, tokenizer, input_text, device):
    """Run spelling correction inference."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Fix any spelling or grammar errors in this text. If there are no errors, return the text unchanged.\n\n{input_text}"},
    ]

    if hasattr(tokenizer, 'apply_chat_template'):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = f"### System:\n{SYSTEM_PROMPT}\n\n### User:\n{input_text}\n\n### Assistant:\n"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the generated tokens
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    result = tokenizer.decode(generated, skip_special_tokens=True).strip()

    # Clean up: take first line/sentence if model generates extra
    for stop in ["\n\n", "\n###", "\n\nUser", "\n\nSystem"]:
        if stop in result:
            result = result[:result.index(stop)].strip()

    return result


def evaluate_model(model_name, model_path, device):
    """Evaluate a single model on Bob story."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    results = []
    exact_matches = 0
    total_errors = 0
    errors_fixed = 0
    false_positives = 0
    hallucinations = 0
    category_stats = {}  # category -> {total, fixed}

    for i, pair in enumerate(BOB_STORY_EVAL):
        predicted = run_inference(model, tokenizer, pair["input"], device)
        exact = predicted.strip() == pair["target"].strip()
        if exact:
            exact_matches += 1

        # Word-level analysis
        pred_words = predicted.lower().split()
        target_words = pair["target"].lower().split()
        input_words = pair["input"].lower().split()

        # Check each expected error
        pair_fixed = 0
        pair_missed = 0
        for err in pair["errors"]:
            cat = err["category"]
            if cat not in category_stats:
                category_stats[cat] = {"total": 0, "fixed": 0}
            category_stats[cat]["total"] += 1
            total_errors += 1

            target_w = err["target"].lower()
            if target_w in " ".join(pred_words):
                errors_fixed += 1
                pair_fixed += 1
                category_stats[cat]["fixed"] += 1
            else:
                pair_missed += 1

        # Check for false positives (correct words changed)
        for w in input_words:
            is_error_word = any(e["written"].lower() == w for e in pair["errors"])
            if not is_error_word and w not in " ".join(pred_words) and w in " ".join(target_words):
                false_positives += 1

        status = "EXACT" if exact else ("PARTIAL" if pair_fixed > 0 else "MISS")
        print(f"\n[{i+1:2d}] {status} (fixed {pair_fixed}/{len(pair['errors'])})")
        print(f"  Input:     {pair['input']}")
        print(f"  Target:    {pair['target']}")
        print(f"  Predicted: {predicted}")

        results.append({
            "input": pair["input"],
            "target": pair["target"],
            "predicted": predicted,
            "exact_match": exact,
            "errors_fixed": pair_fixed,
            "errors_total": len(pair["errors"]),
        })

    # Summary
    n = len(BOB_STORY_EVAL)
    print(f"\n{'='*60}")
    print(f"SUMMARY: {model_name}")
    print(f"{'='*60}")
    print(f"Exact match rate:    {exact_matches}/{n} ({exact_matches/n*100:.1f}%)")
    print(f"Errors fixed:        {errors_fixed}/{total_errors} ({errors_fixed/total_errors*100:.1f}%)")
    print(f"False positives:     {false_positives}")
    print()
    print("Per-category:")
    for cat, stats in sorted(category_stats.items()):
        t, f = stats["total"], stats["fixed"]
        print(f"  {cat:20s}: {f}/{t} ({f/t*100:.1f}%)")

    # Save results
    report = {
        "model": model_name,
        "exact_match_rate": exact_matches / n,
        "errors_fixed": errors_fixed,
        "total_errors": total_errors,
        "error_fix_rate": errors_fixed / total_errors,
        "false_positives": false_positives,
        "category_accuracy": {k: v["fixed"]/v["total"] for k, v in category_stats.items()},
        "sentences": results,
    }
    report_path = EVAL_DIR / f"bob_eval_{model_name}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved to {report_path}")

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    return report


# Main
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

all_reports = []
for model_name, model_path in MODELS:
    if model_path.exists():
        report = evaluate_model(model_name, model_path, device)
        all_reports.append(report)
    else:
        print(f"Skipping {model_name}: {model_path} not found")

# Final comparison
print(f"\n{'='*60}")
print("MODEL COMPARISON")
print(f"{'='*60}")
print(f"{'Model':<25s} {'Exact Match':>12s} {'Errors Fixed':>13s} {'FP':>4s}")
print("-" * 60)
for r in all_reports:
    print(f"{r['model']:<25s} {r['exact_match_rate']*100:>11.1f}% {r['errors_fixed']:>5d}/{r['total_errors']:<5d}   {r['false_positives']:>3d}")

# Save comparison
with open(EVAL_DIR / "bob_eval_comparison.json", "w") as f:
    json.dump(all_reports, f, indent=2)
print(f"\nComparison saved to {EVAL_DIR}/bob_eval_comparison.json")
PYEOF

echo ""
echo "=========================================="
echo "Evaluation complete: $(date)"
echo "=========================================="
