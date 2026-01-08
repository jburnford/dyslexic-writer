#!/usr/bin/env python3
"""
Homophone Hint Prototype
========================
Test the LLM-based homophone detection logic.

Usage:
  1. Install Ollama: https://ollama.com
  2. Pull a small model: ollama pull llama3.2:1b
  3. Run: python homophone_hint.py

Or use llama-cpp-python:
  pip install llama-cpp-python
  python homophone_hint.py --backend llama-cpp --model path/to/model.gguf
"""

import argparse
import json
import re
from typing import Optional

# Common homophones - static list for fast lookup
HOMOPHONES = {
    "there": ["there", "their", "they're"],
    "their": ["there", "their", "they're"],
    "they're": ["there", "their", "they're"],
    "to": ["to", "too", "two"],
    "too": ["to", "too", "two"],
    "two": ["to", "too", "two"],
    "your": ["your", "you're"],
    "you're": ["your", "you're"],
    "its": ["its", "it's"],
    "it's": ["its", "it's"],
    "here": ["here", "hear"],
    "hear": ["here", "hear"],
    "where": ["where", "wear", "were"],
    "wear": ["where", "wear", "were"],
    "were": ["where", "wear", "were"],
    "no": ["no", "know"],
    "know": ["no", "know"],
    "by": ["by", "buy", "bye"],
    "buy": ["by", "buy", "bye"],
    "bye": ["by", "buy", "bye"],
    "right": ["right", "write"],
    "write": ["right", "write"],
    "which": ["which", "witch"],
    "witch": ["which", "witch"],
}

DEFINITIONS = {
    "there": "a place (over there)",
    "their": "belonging to them",
    "they're": "they are (contraction)",
    "to": "direction (go to school)",
    "too": "also, or very much",
    "two": "the number 2",
    "your": "belonging to you",
    "you're": "you are (contraction)",
    "its": "belonging to it",
    "it's": "it is (contraction)",
    "here": "this place",
    "hear": "to listen with ears",
    "where": "asking about place",
    "wear": "to put on clothes",
    "were": "past tense of are",
    "no": "opposite of yes",
    "know": "to understand something",
    "by": "near, or done by someone",
    "buy": "to purchase something",
    "bye": "short for goodbye",
    "right": "correct, or direction",
    "write": "to put words on paper",
    "which": "asking about choice",
    "witch": "magical person in stories",
}

SYSTEM_PROMPT = """You are a reading tutor for a dyslexic child. Your job is to HINT, never correct.

Rules:
1. Identify if the word is a homophone error (there/their/they're, to/too/two, etc.)
2. If it IS an error, respond ONLY with: HINT: "[correct_word]" means [5-word definition]
3. If the word is correct for the context, respond ONLY with: OK
4. DO NOT output the corrected sentence
5. DO NOT explain your reasoning
6. Keep responses under 15 words"""


def is_homophone(word: str) -> bool:
    """Check if a word has homophones we track."""
    return word.lower() in HOMOPHONES


def get_homophone_group(word: str) -> list[str]:
    """Get all homophones for a word."""
    return HOMOPHONES.get(word.lower(), [])


def check_with_ollama(sentence: str, word: str) -> str:
    """Use Ollama API for context checking."""
    try:
        import requests
    except ImportError:
        return "ERROR: requests library not installed. Run: pip install requests"

    prompt = f"""Sentence: "{sentence}"
Focused word: "{word}"
Homophone options: {get_homophone_group(word)}

Is "{word}" correct in this context? If not, which homophone should be used?"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2:1b",  # Small, fast model
                "prompt": prompt,
                "system": SYSTEM_PROMPT,
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Low temp for consistency
                    "num_predict": 50,   # Short response
                }
            },
            timeout=30
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        return "ERROR: Cannot connect to Ollama. Is it running? (ollama serve)"
    except Exception as e:
        return f"ERROR: {e}"


def check_with_llama_cpp(sentence: str, word: str, model_path: str) -> str:
    """Use llama-cpp-python for context checking."""
    try:
        from llama_cpp import Llama
    except ImportError:
        return "ERROR: llama-cpp-python not installed. Run: pip install llama-cpp-python"

    prompt = f"""<|system|>
{SYSTEM_PROMPT}
<|user|>
Sentence: "{sentence}"
Focused word: "{word}"
Homophone options: {get_homophone_group(word)}

Is "{word}" correct in this context?
<|assistant|>"""

    try:
        llm = Llama(model_path=model_path, n_ctx=512, verbose=False)
        output = llm(prompt, max_tokens=50, temperature=0.1, stop=["<|", "\n\n"])
        return output["choices"][0]["text"].strip()
    except Exception as e:
        return f"ERROR: {e}"


def generate_tts_script(sentence: str, error_word: str, correct_word: str) -> str:
    """Generate a TTS script with emphasis for the sound-out feature."""
    # Find position of error word
    words = sentence.split()
    script_parts = []

    for i, w in enumerate(words):
        clean_w = re.sub(r'[^\w]', '', w.lower())
        if clean_w == error_word.lower():
            script_parts.append(f"[pause] {w.upper()}.")
            script_parts.append(f"{error_word.upper()} means {DEFINITIONS.get(error_word.lower(), 'unknown')}.")
            script_parts.append(f"Did you mean {correct_word.upper()}, as in {DEFINITIONS.get(correct_word.lower(), 'unknown')}?")
        else:
            script_parts.append(w)

    return " ".join(script_parts)


def interactive_mode(backend: str = "ollama", model_path: Optional[str] = None):
    """Run interactive testing."""
    print("\n" + "="*60)
    print("  HOMOPHONE HINT PROTOTYPE")
    print("  Type a sentence with a homophone error to test")
    print("  Example: 'Their going to the store'")
    print("  Type 'quit' to exit")
    print("="*60 + "\n")

    while True:
        sentence = input("\nSentence: ").strip()
        if sentence.lower() in ('quit', 'exit', 'q'):
            break

        if not sentence:
            continue

        # Find homophones in sentence
        words = sentence.split()
        homophones_found = []

        for word in words:
            clean = re.sub(r'[^\w]', '', word.lower())
            if is_homophone(clean):
                homophones_found.append((word, clean))

        if not homophones_found:
            print("  No homophones detected in this sentence.")
            continue

        print(f"\n  Homophones found: {[w[0] for w in homophones_found]}")

        # Check each homophone
        for original, clean in homophones_found:
            print(f"\n  Checking '{original}'...")
            print(f"  Options: {get_homophone_group(clean)}")

            # Get LLM verdict
            if backend == "ollama":
                result = check_with_ollama(sentence, clean)
            else:
                result = check_with_llama_cpp(sentence, clean, model_path)

            print(f"  LLM says: {result}")

            # If error detected, generate TTS script
            if result.startswith("HINT:"):
                # Extract correct word from hint
                match = re.search(r'"(\w+)"', result)
                if match:
                    correct = match.group(1)
                    print(f"\n  TTS Script:")
                    print(f"  {generate_tts_script(sentence, clean, correct)}")


def test_examples():
    """Test with known examples."""
    examples = [
        ("Their going to the store.", "their"),
        ("I want to go to.", "to"),  # Second 'to' should be 'too'
        ("I saw there dog.", "there"),
        ("Your the best!", "your"),
        ("I don't no the answer.", "no"),
        ("I want to by a toy.", "by"),
        ("Which witch is which?", "witch"),  # This one is actually correct!
    ]

    print("\n" + "="*60)
    print("  RUNNING TEST EXAMPLES")
    print("="*60)

    for sentence, word in examples:
        print(f"\n  Sentence: {sentence}")
        print(f"  Checking: '{word}'")
        result = check_with_ollama(sentence, word)
        print(f"  Result: {result}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test homophone detection with LLM")
    parser.add_argument("--backend", choices=["ollama", "llama-cpp"], default="ollama",
                        help="LLM backend to use")
    parser.add_argument("--model", type=str, help="Path to GGUF model (for llama-cpp)")
    parser.add_argument("--test", action="store_true", help="Run test examples")

    args = parser.parse_args()

    if args.test:
        test_examples()
    else:
        interactive_mode(args.backend, args.model)
