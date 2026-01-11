#!/usr/bin/env python3
"""
Evaluate fine-tuned spelling correction models.
Tests accuracy on held-out eval set and real dyslexic misspellings.

Usage:
    python evaluate.py --model-dir ./outputs/SmolLM2-360M-Instruct
    python evaluate.py --all --output-dir ./outputs
"""

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Real dyslexic misspellings for testing
# These are actual common errors, not in training data
TEST_CASES = [
    # Phonetic misspellings
    ("enuff", "enough"),
    ("becuase", "because"),
    ("wich", "which"),
    ("thay", "they"),
    ("sed", "said"),
    ("woz", "was"),
    ("cud", "could"),
    ("wud", "would"),
    ("shud", "should"),
    ("frend", "friend"),
    ("peple", "people"),
    ("diffrent", "different"),
    ("definatly", "definitely"),
    ("probly", "probably"),
    ("tomarrow", "tomorrow"),
    ("seperate", "separate"),
    ("occured", "occurred"),
    ("recieve", "receive"),
    ("wierd", "weird"),
    ("untill", "until"),
    # Letter reversals (common in dyslexia)
    ("teh", "the"),
    ("adn", "and"),
    ("waht", "what"),
    ("form", "from"),  # Also valid word, context needed
    ("hte", "the"),
    # Missing letters
    ("whn", "when"),
    ("ther", "there"),
    ("hav", "have"),
    # Extra letters
    ("whenn", "when"),
    ("therre", "there"),
]


def load_model(model_path: Path):
    """Load a fine-tuned model."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def get_correction(model, tokenizer, misspelling: str) -> tuple[str, float]:
    """Get spelling correction from model. Returns (correction, time_seconds)."""
    prompt = f"Fix the spelling of this word.\n\n{misspelling}"

    if hasattr(tokenizer, 'apply_chat_template'):
        messages = [
            {"role": "system", "content": "You are a spelling correction assistant."},
            {"role": "user", "content": prompt}
        ]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        input_text = f"### Instruction:\nFix the spelling of this word.\n\n### Input:\n{misspelling}\n\n### Response:\n"

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    elapsed = time.time() - start

    # Decode only the generated part
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    # Clean up response (take first word/line)
    correction = response.strip().split('\n')[0].split()[0] if response.strip() else ""

    return correction, elapsed


def evaluate_model(model_path: Path) -> dict:
    """Evaluate a single model."""
    print(f"\nEvaluating: {model_path.name}")
    print("-" * 50)

    model, tokenizer = load_model(model_path)

    results = {
        "model": model_path.name,
        "correct": 0,
        "total": len(TEST_CASES),
        "times": [],
        "details": []
    }

    for misspelling, expected in TEST_CASES:
        correction, elapsed = get_correction(model, tokenizer, misspelling)
        results["times"].append(elapsed)

        is_correct = correction.lower() == expected.lower()
        if is_correct:
            results["correct"] += 1

        results["details"].append({
            "input": misspelling,
            "expected": expected,
            "output": correction,
            "correct": is_correct,
            "time": elapsed
        })

        status = "PASS" if is_correct else "FAIL"
        print(f"  [{status}] {misspelling} -> {correction} (expected: {expected}) [{elapsed:.2f}s]")

    # Calculate metrics
    accuracy = results["correct"] / results["total"] * 100
    avg_time = sum(results["times"]) / len(results["times"])

    results["accuracy"] = accuracy
    results["avg_time"] = avg_time

    print(f"\nAccuracy: {accuracy:.1f}% ({results['correct']}/{results['total']})")
    print(f"Avg time: {avg_time:.3f}s")

    # Clean up
    del model
    torch.cuda.empty_cache()

    return results


def evaluate_on_eval_set(model_path: Path, eval_path: Path, max_samples: int = 500) -> dict:
    """Evaluate on held-out eval set."""
    print(f"\nEvaluating on eval set: {eval_path}")

    with open(eval_path) as f:
        eval_data = [json.loads(line) for line in f][:max_samples]

    model, tokenizer = load_model(model_path)

    correct = 0
    for example in eval_data:
        # Only evaluate simple word corrections
        if example["instruction"] == "Fix the spelling of this word.":
            correction, _ = get_correction(model, tokenizer, example["input"])
            if correction.lower() == example["output"].lower():
                correct += 1

    accuracy = correct / len(eval_data) * 100 if eval_data else 0
    print(f"Eval set accuracy: {accuracy:.1f}% ({correct}/{len(eval_data)})")

    del model
    torch.cuda.empty_cache()

    return {"eval_accuracy": accuracy, "eval_samples": len(eval_data)}


def main():
    parser = argparse.ArgumentParser(description="Evaluate spelling correction models")
    parser.add_argument("--model-dir", type=str, help="Path to specific model")
    parser.add_argument("--all", action="store_true", help="Evaluate all models in output dir")
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Output directory with models")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    script_dir = Path(__file__).parent
    eval_path = script_dir / "eval.jsonl"

    # Find models to evaluate
    if args.model_dir:
        model_dirs = [Path(args.model_dir)]
    elif args.all:
        model_dirs = [d for d in output_dir.iterdir() if d.is_dir() and (d / "config.json").exists()]
    else:
        print("Specify --model-dir or --all")
        return

    if not model_dirs:
        print(f"No models found in {output_dir}")
        return

    # Evaluate each model
    all_results = []
    for model_dir in sorted(model_dirs):
        results = evaluate_model(model_dir)

        # Also evaluate on eval set if available
        if eval_path.exists():
            eval_results = evaluate_on_eval_set(model_dir, eval_path)
            results.update(eval_results)

        all_results.append(results)

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Model':<30} {'Accuracy':<12} {'Eval Acc':<12} {'Avg Time':<10}")
    print("-" * 70)

    for r in sorted(all_results, key=lambda x: -x["accuracy"]):
        eval_acc = f"{r.get('eval_accuracy', 0):.1f}%" if 'eval_accuracy' in r else "N/A"
        print(f"{r['model']:<30} {r['accuracy']:>5.1f}%      {eval_acc:<12} {r['avg_time']:.3f}s")

    # Save results
    results_path = output_dir / "evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Recommendation
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    best = max(all_results, key=lambda x: x["accuracy"])
    print(f"Best model: {best['model']}")
    print(f"  Test accuracy: {best['accuracy']:.1f}%")
    print(f"  Avg inference time: {best['avg_time']:.3f}s")


if __name__ == "__main__":
    main()
