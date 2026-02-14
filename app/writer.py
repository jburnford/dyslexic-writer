#!/usr/bin/env python3
"""
Dyslexic Writer - Interactive CLI tool.
Type text, get spelling corrections, hear them read aloud.

Usage:
    python writer.py                         # TTS only (no correction model)
    python writer.py --backend ollama        # with Ollama model
    python writer.py --backend transformers --model /path/to/model
"""

import argparse
import sys

from corrector import create_corrector
from tts import speak, VOICE


def highlight_changes(original: str, corrected: str) -> str:
    """Show what changed between original and corrected text."""
    orig_words = original.split()
    corr_words = corrected.split()

    # Simple word-level diff with color
    output = []
    max_i = min(len(orig_words), len(corr_words))
    for i in range(max_i):
        if orig_words[i] != corr_words[i]:
            # strikethrough original, bold corrected
            output.append(f"\033[9;31m{orig_words[i]}\033[0m -> \033[1;32m{corr_words[i]}\033[0m")
        else:
            output.append(corr_words[i])

    # Handle length differences
    for i in range(max_i, len(corr_words)):
        output.append(f"\033[1;32m{corr_words[i]}\033[0m")

    return " ".join(output)


def main():
    parser = argparse.ArgumentParser(description="Dyslexic Writer - Interactive spelling tool")
    parser.add_argument("--backend", default="passthrough",
                        choices=["ollama", "transformers", "passthrough"])
    parser.add_argument("--model", default="smollm2-1.7b-spell")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--voice", default=VOICE)
    parser.add_argument("--no-speak", action="store_true", help="Disable TTS")
    args = parser.parse_args()

    # Create corrector
    if args.backend == "ollama":
        corrector = create_corrector("ollama", model=args.model, base_url=args.ollama_url)
    elif args.backend == "transformers":
        corrector = create_corrector("transformers", model_path=args.model)
    else:
        corrector = create_corrector("passthrough")

    print("=== Dyslexic Writer ===")
    print(f"Backend: {args.backend}")
    print(f"Voice: {args.voice}")
    print("Type your text and press Enter. Type 'quit' to exit.\n")

    while True:
        try:
            text = input("\033[1mYou:\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not text:
            continue
        if text.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        # Correct
        result = corrector.correct(text)

        if result.changed:
            print(f"\033[1mFixed:\033[0m {highlight_changes(result.original, result.corrected)}")
            print(f"\033[1mResult:\033[0m {result.corrected}")
        else:
            print(f"\033[1mResult:\033[0m {result.corrected} \033[2m(no changes)\033[0m")

        # Speak
        if not args.no_speak:
            try:
                speak(result.corrected, args.voice)
            except Exception as e:
                print(f"\033[33mTTS error: {e}\033[0m")

        print()


if __name__ == "__main__":
    main()
