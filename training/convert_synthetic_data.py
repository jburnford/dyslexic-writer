#!/usr/bin/env python3
"""
Convert synthetic Gemini-generated data to training format.

Converts:
- Sentence pairs (original/corrected) -> sentence instruction format
- Word pairs (misspelled/correct) -> word instruction format

Output files go to training-data/ directory.
"""

import json
from pathlib import Path

SENTENCE_INSTRUCTION = "Fix the spelling mistakes in this sentence. Only output the corrected sentence."
WORD_INSTRUCTION = "Fix the spelling of this word."

def load_json(path: Path) -> list[dict]:
    """Load JSON file (array of objects)."""
    with open(path) as f:
        return json.load(f)

def save_jsonl(data: list[dict], path: Path):
    """Save to JSONL file."""
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    print(f"  Saved {len(data)} examples to {path}")

def convert_sentences(data: list[dict]) -> list[dict]:
    """Convert sentence pairs to instruction format."""
    converted = []
    for item in data:
        converted.append({
            "instruction": SENTENCE_INSTRUCTION,
            "input": item["original"],
            "output": item["corrected"]
        })
    return converted

def convert_word_pairs(data: list[dict]) -> list[dict]:
    """Convert word pairs to instruction format."""
    converted = []
    for item in data:
        converted.append({
            "instruction": WORD_INSTRUCTION,
            "input": item["misspelled"],
            "output": item["correct"]
        })
    return converted

def main():
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "training-data"

    # Sentence files (original/corrected format)
    sentence_files = [
        "synthetic_academic2_gemini3.json",
        "synthetic_academic_writing_gemini3.json",
        "synthetic_casual_kids_gemini3.json",
        "synthetic_compound_words_gemini3.json",
        "synthetic_contractions_gemini3.json",
        "synthetic_daily_life_gemini3.json",
        "synthetic_dyslexia_reversals_gemini3.json",
        "synthetic_elementary_k3_gemini3.json",
        "synthetic_everyday_gemini3.json",
        "synthetic_hard_words_gemini3.json",
        "synthetic_homophones_gemini3.json",
        "synthetic_kids_writing_gemini3.json",
        "synthetic_mixed2_gemini3.json",
        "synthetic_mixed3_gemini3.json",
        "synthetic_mixed_writing_gemini3.json",
        "synthetic_silent_letters_gemini3.json",
    ]

    # Word pair files (misspelled/correct format)
    word_pair_files = [
        "synthetic_word_pairs_k3_gemini3.json",
        "synthetic_word_pairs_vocab_gemini3.json",
    ]

    print("Converting synthetic sentence data...")
    all_sentences = []
    for filename in sentence_files:
        filepath = script_dir / filename
        if filepath.exists():
            data = load_json(filepath)
            converted = convert_sentences(data)
            all_sentences.extend(converted)
            print(f"  {filename}: {len(data)} examples")
        else:
            print(f"  WARNING: {filename} not found")

    print(f"\nTotal sentences: {len(all_sentences)}")

    print("\nConverting synthetic word pair data...")
    all_word_pairs = []
    for filename in word_pair_files:
        filepath = script_dir / filename
        if filepath.exists():
            data = load_json(filepath)
            converted = convert_word_pairs(data)
            all_word_pairs.extend(converted)
            print(f"  {filename}: {len(data)} examples")
        else:
            print(f"  WARNING: {filename} not found")

    print(f"\nTotal word pairs: {len(all_word_pairs)}")

    # Save to training-data directory
    print("\nSaving converted data...")
    save_jsonl(all_sentences, data_dir / "synthetic_sentences.jsonl")
    save_jsonl(all_word_pairs, data_dir / "synthetic_word_pairs.jsonl")

    # Also save combined
    combined = all_sentences + all_word_pairs
    save_jsonl(combined, data_dir / "synthetic_all.jsonl")

    print(f"\nDone! Created in {data_dir}:")
    print(f"  - synthetic_sentences.jsonl ({len(all_sentences)} sentence examples)")
    print(f"  - synthetic_word_pairs.jsonl ({len(all_word_pairs)} word pair examples)")
    print(f"  - synthetic_all.jsonl ({len(combined)} total examples)")

if __name__ == "__main__":
    main()
