#!/usr/bin/env python3
"""
Run hallucination tests on trained models using HuggingFace transformers.
Designed to run on Nibi cluster with GPU.
"""

import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from difflib import SequenceMatcher

# Test cases
PASS_THROUGH_TESTS = [
    "The quick brown fox jumps over the lazy dog.",
    "I live in Saskatoon, Saskatchewan.",
    "My email is john.smith@example.com.",
    "The meeting is at 3:30 PM.",
    "She has a PhD in Computer Science.",
    "The iPhone 15 Pro Max costs $1,199.",
    "My favorite color is blue.",
    "They went to the store.",
    "I like pizza and pasta.",
    "The weather is nice today.",
]

KNOWN_CORRECTIONS = [
    ("I went to teh store.", "I went to the store."),
    ("She is my freind.", "She is my friend."),
    ("The wheather is nice.", "The weather is nice."),
    ("I recieved your message.", "I received your message."),
    ("This is definately correct.", "This is definitely correct."),
    ("I dont know.", "I don't know."),
    ("Its a nice day.", "It's a nice day."),
    ("Your the best.", "You're the best."),
    ("I went their yesterday.", "I went there yesterday."),
    ("alot of people came.", "a lot of people came."),
]

EDGE_CASES = [
    ("I met Elon Musk yesterday.", "I met Elon Musk yesterday."),
    ("She works at Google.", "She works at Google."),
    ("The API endpoint returns JSON.", "The API endpoint returns JSON."),
    ("Run pip install numpy.", "Run pip install numpy."),
    ("The CEO and CFO met today.", "The CEO and CFO met today."),
]


def load_model(model_path: str):
    """Load model and tokenizer."""
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def generate_correction(model, tokenizer, text: str, max_new_tokens: int = 128) -> str:
    """Generate spelling correction for input text."""
    # Format as instruction
    if hasattr(tokenizer, 'apply_chat_template'):
        messages = [
            {"role": "system", "content": "You are a spelling correction assistant."},
            {"role": "user", "content": f"Fix the spelling mistakes in this sentence. Only output the corrected sentence.\n\n{text}"}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = f"### Instruction:\nFix the spelling mistakes in this sentence. Only output the corrected sentence.\n\n### Input:\n{text}\n\n### Response:\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the new tokens
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    # Clean up response - take first line/sentence
    response = response.strip()
    if '\n' in response:
        response = response.split('\n')[0]

    return response.strip()


def similarity(a: str, b: str) -> float:
    """Calculate string similarity ratio."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def run_tests(model, tokenizer, model_name: str) -> dict:
    """Run all tests on a model."""
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"{'='*60}")

    results = {
        "model": model_name,
        "pass_through": {"passed": 0, "failed": 0, "details": []},
        "corrections": {"passed": 0, "failed": 0, "details": []},
        "edge_cases": {"passed": 0, "failed": 0, "details": []},
        "word_preservation": {"passed": 0, "failed": 0, "details": []},
    }

    # 1. Pass-through tests
    print("\n1. Pass-through tests (correct text should stay unchanged):")
    for text in PASS_THROUGH_TESTS:
        output = generate_correction(model, tokenizer, text)
        sim = similarity(output, text)
        passed = sim > 0.95

        if passed:
            results["pass_through"]["passed"] += 1
            status = "✓"
        else:
            results["pass_through"]["failed"] += 1
            status = "✗"

        results["pass_through"]["details"].append({
            "input": text,
            "output": output,
            "similarity": sim,
            "passed": passed
        })
        print(f"  {status} [{sim:.2f}] {text[:40]}...")
        if not passed:
            print(f"       Got: {output[:40]}...")

    # 2. Known corrections
    print("\n2. Known corrections (should fix misspellings):")
    for input_text, expected in KNOWN_CORRECTIONS:
        output = generate_correction(model, tokenizer, input_text)
        sim = similarity(output, expected)
        passed = sim > 0.90

        if passed:
            results["corrections"]["passed"] += 1
            status = "✓"
        else:
            results["corrections"]["failed"] += 1
            status = "✗"

        results["corrections"]["details"].append({
            "input": input_text,
            "expected": expected,
            "output": output,
            "similarity": sim,
            "passed": passed
        })
        print(f"  {status} [{sim:.2f}] {input_text[:30]}... → {output[:30]}...")

    # 3. Edge cases
    print("\n3. Edge cases (proper nouns, tech terms):")
    for input_text, expected in EDGE_CASES:
        output = generate_correction(model, tokenizer, input_text)
        sim = similarity(output, expected)
        passed = sim > 0.95

        if passed:
            results["edge_cases"]["passed"] += 1
            status = "✓"
        else:
            results["edge_cases"]["failed"] += 1
            status = "✗"

        results["edge_cases"]["details"].append({
            "input": input_text,
            "expected": expected,
            "output": output,
            "similarity": sim,
            "passed": passed
        })
        print(f"  {status} [{sim:.2f}] {input_text[:40]}...")
        if not passed:
            print(f"       Got: {output[:40]}...")

    # 4. Word preservation
    print("\n4. Word preservation (no words added/deleted):")
    test_texts = PASS_THROUGH_TESTS[:5] + [t[0] for t in KNOWN_CORRECTIONS[:5]]
    for text in test_texts:
        output = generate_correction(model, tokenizer, text)
        input_words = count_words(text)
        output_words = count_words(output)
        diff = abs(input_words - output_words)
        passed = diff <= 1  # Allow 1 word difference for contractions

        if passed:
            results["word_preservation"]["passed"] += 1
            status = "✓"
        else:
            results["word_preservation"]["failed"] += 1
            status = "✗"

        results["word_preservation"]["details"].append({
            "input": text,
            "output": output,
            "input_words": input_words,
            "output_words": output_words,
            "passed": passed
        })
        if not passed:
            print(f"  {status} Words: {input_words} → {output_words} | {text[:30]}...")

    word_passed = results["word_preservation"]["passed"]
    word_total = word_passed + results["word_preservation"]["failed"]
    print(f"  Passed: {word_passed}/{word_total}")

    # Summary
    total_passed = sum(r["passed"] for r in results.values() if isinstance(r, dict) and "passed" in r)
    total_failed = sum(r["failed"] for r in results.values() if isinstance(r, dict) and "failed" in r)
    total = total_passed + total_failed

    print(f"\n{'='*60}")
    print(f"SUMMARY: {total_passed}/{total} tests passed ({100*total_passed/total:.1f}%)")
    print(f"  Pass-through: {results['pass_through']['passed']}/{results['pass_through']['passed']+results['pass_through']['failed']}")
    print(f"  Corrections:  {results['corrections']['passed']}/{results['corrections']['passed']+results['corrections']['failed']}")
    print(f"  Edge cases:   {results['edge_cases']['passed']}/{results['edge_cases']['passed']+results['edge_cases']['failed']}")
    print(f"  Word preservation: {results['word_preservation']['passed']}/{results['word_preservation']['passed']+results['word_preservation']['failed']}")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default="./outputs", help="Directory containing trained models")
    parser.add_argument("--model", type=str, help="Specific model to test (e.g., SmolLM2-360M-Instruct)")
    parser.add_argument("--output", type=str, help="Save results to JSON")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    all_results = {}

    if args.model:
        models = [args.model]
    else:
        # Find all model directories
        models = [d.name for d in model_dir.iterdir() if d.is_dir() and (d / "config.json").exists()]

    print(f"Found models: {models}")

    for model_name in models:
        model_path = model_dir / model_name
        if not (model_path / "config.json").exists():
            print(f"Skipping {model_name} - no config.json found")
            continue

        try:
            model, tokenizer = load_model(str(model_path))
            results = run_tests(model, tokenizer, model_name)
            all_results[model_name] = results

            # Free memory
            del model
            del tokenizer
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error testing {model_name}: {e}")
            continue

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
