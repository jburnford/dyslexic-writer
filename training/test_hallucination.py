#!/usr/bin/env python3
"""
Test fine-tuned spelling models for hallucination and accuracy.

Tests:
1. Pass-through: Correct text should return unchanged
2. Word preservation: No words added or deleted
3. Character preservation: Only spelling changes, not meaning
4. Known corrections: Verify expected fixes work
5. Edge cases: Proper nouns, technical terms, etc.
"""

import json
import argparse
from pathlib import Path
from difflib import SequenceMatcher

# Test cases organized by category
PASS_THROUGH_TESTS = [
    # These are correct - should return unchanged
    "The quick brown fox jumps over the lazy dog.",
    "I live in Saskatoon, Saskatchewan.",
    "My email is john.smith@example.com.",
    "The meeting is at 3:30 PM.",
    "She has a PhD in Computer Science.",
    "The iPhone 15 Pro Max costs $1,199.",
    "We drove 500 km yesterday.",
    "The API returns JSON data.",
    "My favorite color is blue.",
    "They went to the store.",
]

KNOWN_CORRECTIONS = [
    # (input, expected_output)
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
    # Proper nouns - should NOT be changed
    ("I met Elon Musk yesterday.", "I met Elon Musk yesterday."),
    ("She works at Google.", "She works at Google."),
    ("We visited the Louvre in Paris.", "We visited the Louvre in Paris."),
    # Technical terms
    ("The API endpoint returns JSON.", "The API endpoint returns JSON."),
    ("Run pip install numpy.", "Run pip install numpy."),
    # Abbreviations
    ("The CEO and CFO met today.", "The CEO and CFO met today."),
    ("I have a PhD and an MBA.", "I have a PhD and an MBA."),
    # Numbers and mixed
    ("I bought 3 apples for $2.50.", "I bought 3 apples for $2.50."),
]

WORD_PRESERVATION_TESTS = [
    # Model should NOT add or remove words
    ("I like cats.", "I like cats."),  # Don't add "really" or similar
    ("The book is on the table.", "The book is on the table."),
    ("She went home.", "She went home."),  # Don't add "back"
]


def count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def similarity(a: str, b: str) -> float:
    """Calculate string similarity ratio."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def run_model(model_path: str, text: str) -> str:
    """Run the model on input text and return output."""
    # This would be replaced with actual model inference
    # For now, return placeholder
    raise NotImplementedError("Implement model inference")


def run_ollama(model_name: str, text: str) -> str:
    """Run model via Ollama."""
    import subprocess
    prompt = f"Fix the spelling mistakes in this sentence. Only output the corrected sentence.\n\n{text}"
    result = subprocess.run(
        ["ollama", "run", model_name, prompt],
        capture_output=True,
        text=True,
        timeout=30
    )
    return result.stdout.strip()


def test_pass_through(model_func, model_name: str) -> dict:
    """Test that correct text passes through unchanged."""
    results = {"passed": 0, "failed": 0, "failures": []}

    for text in PASS_THROUGH_TESTS:
        output = model_func(model_name, text)
        if output.strip() == text.strip():
            results["passed"] += 1
        else:
            results["failed"] += 1
            results["failures"].append({
                "input": text,
                "expected": text,
                "got": output,
                "issue": "Changed correct text"
            })

    return results


def test_known_corrections(model_func, model_name: str) -> dict:
    """Test that known misspellings are corrected properly."""
    results = {"passed": 0, "failed": 0, "failures": []}

    for input_text, expected in KNOWN_CORRECTIONS:
        output = model_func(model_name, input_text)
        if similarity(output.strip(), expected.strip()) > 0.95:
            results["passed"] += 1
        else:
            results["failed"] += 1
            results["failures"].append({
                "input": input_text,
                "expected": expected,
                "got": output,
                "similarity": similarity(output, expected)
            })

    return results


def test_word_preservation(model_func, model_name: str) -> dict:
    """Test that word count is preserved (no additions/deletions)."""
    results = {"passed": 0, "failed": 0, "failures": []}

    all_tests = PASS_THROUGH_TESTS + [t[0] for t in KNOWN_CORRECTIONS]

    for text in all_tests:
        output = model_func(model_name, text)
        input_words = count_words(text)
        output_words = count_words(output)

        # Allow small variance for contractions (don't -> do not)
        if abs(input_words - output_words) <= 1:
            results["passed"] += 1
        else:
            results["failed"] += 1
            results["failures"].append({
                "input": text,
                "output": output,
                "input_words": input_words,
                "output_words": output_words,
                "issue": f"Word count changed by {abs(input_words - output_words)}"
            })

    return results


def test_edge_cases(model_func, model_name: str) -> dict:
    """Test proper nouns, technical terms, abbreviations."""
    results = {"passed": 0, "failed": 0, "failures": []}

    for input_text, expected in EDGE_CASES:
        output = model_func(model_name, input_text)
        if similarity(output.strip(), expected.strip()) > 0.95:
            results["passed"] += 1
        else:
            results["failed"] += 1
            results["failures"].append({
                "input": input_text,
                "expected": expected,
                "got": output,
                "issue": "Modified edge case incorrectly"
            })

    return results


def run_all_tests(model_func, model_name: str) -> dict:
    """Run all hallucination tests."""
    print(f"\nTesting model: {model_name}")
    print("=" * 60)

    results = {}

    print("\n1. Pass-through test (correct text unchanged)...")
    results["pass_through"] = test_pass_through(model_func, model_name)
    print(f"   Passed: {results['pass_through']['passed']}/{results['pass_through']['passed'] + results['pass_through']['failed']}")

    print("\n2. Known corrections test...")
    results["known_corrections"] = test_known_corrections(model_func, model_name)
    print(f"   Passed: {results['known_corrections']['passed']}/{results['known_corrections']['passed'] + results['known_corrections']['failed']}")

    print("\n3. Word preservation test...")
    results["word_preservation"] = test_word_preservation(model_func, model_name)
    print(f"   Passed: {results['word_preservation']['passed']}/{results['word_preservation']['passed'] + results['word_preservation']['failed']}")

    print("\n4. Edge cases test (proper nouns, technical terms)...")
    results["edge_cases"] = test_edge_cases(model_func, model_name)
    print(f"   Passed: {results['edge_cases']['passed']}/{results['edge_cases']['passed'] + results['edge_cases']['failed']}")

    # Summary
    total_passed = sum(r["passed"] for r in results.values())
    total_failed = sum(r["failed"] for r in results.values())
    total = total_passed + total_failed

    print("\n" + "=" * 60)
    print(f"TOTAL: {total_passed}/{total} passed ({100*total_passed/total:.1f}%)")

    # Show failures
    all_failures = []
    for category, r in results.items():
        for f in r["failures"]:
            f["category"] = category
            all_failures.append(f)

    if all_failures:
        print(f"\nFailures ({len(all_failures)}):")
        for f in all_failures[:10]:  # Show first 10
            print(f"  [{f['category']}] {f.get('issue', 'Unknown')}")
            print(f"    Input: {f['input'][:50]}...")
            print(f"    Got:   {f['got'][:50]}...")

    return results


def main():
    parser = argparse.ArgumentParser(description="Test spelling model for hallucinations")
    parser.add_argument("--model", type=str, required=True, help="Model name (for Ollama) or path")
    parser.add_argument("--backend", type=str, default="ollama", choices=["ollama", "local"])
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    args = parser.parse_args()

    if args.backend == "ollama":
        model_func = run_ollama
    else:
        model_func = run_model

    results = run_all_tests(model_func, args.model)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
