#!/usr/bin/env python3
"""
Benchmark phonetic/dyslexic spelling correction.
These are the REAL challenges - not homophones, but creative phonetic spellings.
"""

import requests
import time

MODEL = "phi4-mini"

SYSTEM_PROMPT = """You are a spelling helper for a child who spells words the way they sound.

When given a misspelled word in a sentence:
1. Figure out what word they meant based on how it sounds
2. Respond ONLY with: WORD: [correct_spelling]
3. If the word is already correct, respond: OK

Examples:
- "enuff" sounds like "enough" → WORD: enough
- "sed" sounds like "said" → WORD: said
- "fone" sounds like "phone" → WORD: phone

Keep response under 10 words. Just give the correct spelling."""

# Common dyslexic phonetic spellings
# Format: (sentence_with_error, misspelled_word, correct_word)
TEST_CASES = [
    # Phonetic vowel substitutions
    ("I have enuff food.", "enuff", "enough"),
    ("She sed hello.", "sed", "said"),
    ("I herd a noise.", "herd", "heard"),
    ("The skool is big.", "skool", "school"),
    ("I wuz happy.", "wuz", "was"),

    # Silent letters ignored
    ("I can rite my name.", "rite", "write"),
    ("I no the answer.", "no", "know"),
    ("Use the nife carefully.", "nife", "knife"),
    ("I need to cof.", "cof", "cough"),
    ("My nee hurts.", "nee", "knee"),

    # Common phonetic patterns
    ("Call me on the fone.", "fone", "phone"),
    ("I want sum water.", "sum", "some"),
    ("That is reely cool.", "reely", "really"),
    ("I luv my dog.", "luv", "love"),
    ("Becuz I want to.", "becuz", "because"),

    # Double letter confusion
    ("I am hapy today.", "hapy", "happy"),
    ("The bal is red.", "bal", "ball"),
    ("I can spel words.", "spel", "spell"),
    ("She is prety.", "prety", "pretty"),
    ("I am runing fast.", "runing", "running"),

    # Ending sounds
    ("I wil go home.", "wil", "will"),
    ("The burd flew away.", "burd", "bird"),
    ("I hav a cat.", "hav", "have"),
    ("She walkd home.", "walkd", "walked"),
    ("I playd games.", "playd", "played"),

    # Complex phonetic attempts
    ("That is difikult.", "difikult", "difficult"),
    ("I am exited!", "exited", "excited"),
    ("The peple are nice.", "peple", "people"),
    ("I saw a bewtiful sunset.", "bewtiful", "beautiful"),
    ("That is intresting.", "intresting", "interesting"),
]


def test_word(sentence: str, word: str) -> tuple[str, float]:
    """Test the model on a phonetic spelling."""
    prompt = f'Sentence: "{sentence}"\nMisspelled word: "{word}"\n\nWhat word did they mean?'

    start = time.time()
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": MODEL,
                "prompt": prompt,
                "system": SYSTEM_PROMPT,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 30,
                }
            },
            timeout=30
        )
        response.raise_for_status()
        elapsed = time.time() - start
        return response.json().get("response", "").strip(), elapsed
    except Exception as e:
        return f"ERROR: {e}", time.time() - start


def run_benchmark():
    print("=" * 70)
    print("PHONETIC SPELLING BENCHMARK - phi4-mini")
    print("Testing dyslexic phonetic spelling patterns")
    print("=" * 70)

    # Warm up
    print("\nWarming up model...")
    test_word("Test sentence.", "test")

    correct = 0
    total = 0
    times = []
    failures = []

    for sentence, misspelled, expected in TEST_CASES:
        response, elapsed = test_word(sentence, misspelled)
        times.append(elapsed)
        total += 1

        # Check if response contains the correct word
        is_correct = expected.lower() in response.lower()
        if is_correct:
            correct += 1
            status = "✓"
        else:
            status = "✗"
            failures.append((misspelled, expected, response))

        print(f"{status} {elapsed:.2f}s | {misspelled:12} → {expected:12} | Got: {response[:40]}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Accuracy: {correct}/{total} ({correct/total*100:.1f}%)")
    print(f"Avg time: {sum(times)/len(times):.2f}s")
    print(f"Min time: {min(times):.2f}s")
    print(f"Max time: {max(times):.2f}s")

    if failures:
        print(f"\n{len(failures)} FAILURES:")
        for misspelled, expected, got in failures[:10]:
            print(f"  {misspelled} → expected '{expected}', got: {got[:50]}")

    print("\n" + "=" * 70)
    if correct/total >= 0.8:
        print("VERDICT: Good enough for phonetic spelling! ✓")
    elif correct/total >= 0.6:
        print("VERDICT: Decent, but may need prompt tuning")
    else:
        print("VERDICT: Needs work - consider different model or approach")


if __name__ == "__main__":
    run_benchmark()
