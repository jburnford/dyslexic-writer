#!/usr/bin/env python3
"""
Benchmark different Ollama models for homophone detection.
Tests speed and accuracy on our use case.
"""

import requests
import time
import json

MODELS = [
    "gemma3n:e2b",
    "phi4-mini",
    "ministral-3",
]

SYSTEM_PROMPT = """You are a reading tutor for a dyslexic child. Your job is to HINT, never correct.

Rules:
1. If the word is wrong for the context, respond: HINT: "[correct_word]" means [5-word definition]
2. If the word is correct, respond: OK
3. Keep response under 15 words. No explanations."""

TEST_CASES = [
    # (sentence, word_to_check, expected_contains)
    ("Their going to the store.", "Their", "They're"),
    ("I want to go to.", "to", "too"),  # Second 'to' should be 'too'
    ("I saw there dog.", "there", "their"),
    ("Your the best!", "Your", "You're"),
    ("I don't no the answer.", "no", "know"),
    ("I want to by a toy.", "by", "buy"),
    ("I can here the music.", "here", "hear"),
    ("The dog wagged it's tail.", "it's", "its"),
    # These should be OK (correct usage)
    ("I saw a bird in the tree.", "bird", "OK"),
    ("They went to the park.", "to", "OK"),
]


def test_model(model: str, sentence: str, word: str) -> tuple[str, float]:
    """Test a model on a single case. Returns (response, time_seconds)."""
    prompt = f'Sentence: "{sentence}"\nWord to check: "{word}"\n\nIs this word correct for the context?'

    start = time.time()
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "system": SYSTEM_PROMPT,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 50,
                }
            },
            timeout=60
        )
        response.raise_for_status()
        elapsed = time.time() - start
        return response.json().get("response", "").strip(), elapsed
    except Exception as e:
        return f"ERROR: {e}", time.time() - start


def run_benchmark():
    print("=" * 70)
    print("HOMOPHONE DETECTION BENCHMARK")
    print("=" * 70)

    results = {model: {"correct": 0, "total": 0, "times": []} for model in MODELS}

    for model in MODELS:
        print(f"\n{'='*70}")
        print(f"MODEL: {model}")
        print("=" * 70)

        # Warm up (first call loads model)
        print("Warming up (loading model)...")
        test_model(model, "Test sentence.", "Test")

        for sentence, word, expected in TEST_CASES:
            response, elapsed = test_model(model, sentence, word)
            results[model]["times"].append(elapsed)
            results[model]["total"] += 1

            # Check if response contains expected
            is_correct = expected.lower() in response.lower()
            if is_correct:
                results[model]["correct"] += 1
                status = "PASS"
            else:
                status = "FAIL"

            print(f"\n[{status}] {elapsed:.2f}s")
            print(f"  Sentence: {sentence}")
            print(f"  Word: '{word}' | Expected: '{expected}'")
            print(f"  Response: {response[:80]}{'...' if len(response) > 80 else ''}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Model':<20} {'Accuracy':<12} {'Avg Time':<12} {'Min':<10} {'Max':<10}")
    print("-" * 70)

    for model in MODELS:
        r = results[model]
        accuracy = r["correct"] / r["total"] * 100 if r["total"] > 0 else 0
        avg_time = sum(r["times"]) / len(r["times"]) if r["times"] else 0
        min_time = min(r["times"]) if r["times"] else 0
        max_time = max(r["times"]) if r["times"] else 0

        print(f"{model:<20} {accuracy:>5.1f}%      {avg_time:>6.2f}s      {min_time:>5.2f}s    {max_time:>5.2f}s")

    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)

    # Find best balance of speed and accuracy
    best = None
    best_score = 0
    for model in MODELS:
        r = results[model]
        accuracy = r["correct"] / r["total"] if r["total"] > 0 else 0
        avg_time = sum(r["times"]) / len(r["times"]) if r["times"] else 999
        # Score: accuracy matters most, but penalize slow models
        score = accuracy * 100 - avg_time * 5  # 1 second = 5 points penalty
        if score > best_score:
            best_score = score
            best = model

    if best:
        r = results[best]
        accuracy = r["correct"] / r["total"] * 100
        avg_time = sum(r["times"]) / len(r["times"])
        print(f"Best model for this task: {best}")
        print(f"  Accuracy: {accuracy:.1f}%")
        print(f"  Avg response time: {avg_time:.2f}s")


if __name__ == "__main__":
    run_benchmark()
