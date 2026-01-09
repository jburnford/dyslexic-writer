#!/usr/bin/env python3
"""
Convert Birkbeck/Holbrook spelling corpora to training data format.

Input format (corpus):
$correct_word
misspelling1
misspelling2
...

Output formats:
1. JSONL for instruction fine-tuning (Llama, Phi, etc.)
2. CSV pairs for analysis
3. Stats on the data
"""

import json
import csv
import re
from pathlib import Path
from collections import defaultdict

def parse_corpus(filepath: Path) -> list[tuple[str, str]]:
    """Parse corpus file into (misspelling, correct) pairs."""
    pairs = []
    current_correct = None

    with open(filepath, 'r', encoding='latin-1') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith('$'):
                current_correct = line[1:].replace('_', ' ')
            elif current_correct:
                misspelling = line.replace('_', ' ')
                # Skip if they're the same
                if misspelling.lower() != current_correct.lower():
                    pairs.append((misspelling, current_correct))

    return pairs


def create_instruction_dataset(pairs: list[tuple[str, str]], output_path: Path):
    """Create JSONL for instruction fine-tuning."""
    with open(output_path, 'w') as f:
        for misspelling, correct in pairs:
            entry = {
                "instruction": "Fix the spelling of this word.",
                "input": misspelling,
                "output": correct
            }
            f.write(json.dumps(entry) + '\n')

    print(f"Created {output_path} with {len(pairs)} examples")


def create_sentence_dataset(pairs: list[tuple[str, str]], output_path: Path):
    """Create JSONL with sentence context for better training."""
    templates = [
        "I have {} food.",
        "The {} is big.",
        "We went to the {}.",
        "She said {} to me.",
        "It was very {}.",
        "I want to {} today.",
        "The {} was great.",
        "He is my {}.",
    ]

    with open(output_path, 'w') as f:
        for i, (misspelling, correct) in enumerate(pairs):
            template = templates[i % len(templates)]

            entry = {
                "instruction": "Fix the spelling mistakes in this sentence. Only output the corrected sentence.",
                "input": template.format(misspelling),
                "output": template.format(correct)
            }
            f.write(json.dumps(entry) + '\n')

    print(f"Created {output_path} with {len(pairs)} examples")


def create_change_format_dataset(pairs: list[tuple[str, str]], output_path: Path):
    """Create JSONL in the CHANGES format we use."""
    with open(output_path, 'w') as f:
        for misspelling, correct in pairs:
            entry = {
                "instruction": "Fix spelling mistakes. Output format: CHANGES: misspelled->correct",
                "input": f'Fix spelling: "I have {misspelling} here."',
                "output": f"CHANGES: {misspelling}->{correct}"
            }
            f.write(json.dumps(entry) + '\n')

    print(f"Created {output_path} with {len(pairs)} examples")


def create_csv(pairs: list[tuple[str, str]], output_path: Path):
    """Create simple CSV for analysis."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['misspelling', 'correct', 'edit_distance', 'len_diff'])

        for misspelling, correct in pairs:
            edit_dist = levenshtein(misspelling.lower(), correct.lower())
            len_diff = len(misspelling) - len(correct)
            writer.writerow([misspelling, correct, edit_dist, len_diff])

    print(f"Created {output_path} with {len(pairs)} pairs")


def levenshtein(s1: str, s2: str) -> int:
    """Calculate edit distance."""
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    if len(s2) == 0:
        return len(s1)

    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row

    return prev_row[-1]


def print_stats(pairs: list[tuple[str, str]], name: str):
    """Print statistics about the corpus."""
    print(f"\n{'='*50}")
    print(f"Stats for {name}")
    print(f"{'='*50}")

    print(f"Total pairs: {len(pairs)}")

    # Unique correct words
    correct_words = set(c.lower() for _, c in pairs)
    print(f"Unique correct words: {len(correct_words)}")

    # Edit distance distribution
    distances = defaultdict(int)
    for m, c in pairs:
        d = levenshtein(m.lower(), c.lower())
        distances[d] += 1

    print("\nEdit distance distribution:")
    for d in sorted(distances.keys())[:10]:
        pct = distances[d] / len(pairs) * 100
        print(f"  {d}: {distances[d]} ({pct:.1f}%)")

    # Common patterns
    print("\nSample misspellings:")
    for m, c in pairs[:10]:
        print(f"  {m} -> {c}")


def main():
    script_dir = Path(__file__).parent

    # Process Birkbeck corpus
    birkbeck_path = script_dir / "birkbeck.dat"
    if birkbeck_path.exists():
        birkbeck_pairs = parse_corpus(birkbeck_path)
        print_stats(birkbeck_pairs, "Birkbeck")

        create_instruction_dataset(birkbeck_pairs, script_dir / "birkbeck_instruction.jsonl")
        create_change_format_dataset(birkbeck_pairs, script_dir / "birkbeck_changes.jsonl")
        create_csv(birkbeck_pairs, script_dir / "birkbeck_pairs.csv")

    # Process Holbrook corpus
    holbrook_path = script_dir / "holbrook.dat"
    if holbrook_path.exists():
        holbrook_pairs = parse_corpus(holbrook_path)
        print_stats(holbrook_pairs, "Holbrook")

        create_instruction_dataset(holbrook_pairs, script_dir / "holbrook_instruction.jsonl")
        create_change_format_dataset(holbrook_pairs, script_dir / "holbrook_changes.jsonl")
        create_csv(holbrook_pairs, script_dir / "holbrook_pairs.csv")

    # Combined dataset
    if birkbeck_path.exists() and holbrook_path.exists():
        combined = birkbeck_pairs + holbrook_pairs
        print_stats(combined, "Combined")

        create_instruction_dataset(combined, script_dir / "combined_instruction.jsonl")
        create_change_format_dataset(combined, script_dir / "combined_changes.jsonl")
        create_sentence_dataset(combined, script_dir / "combined_sentences.jsonl")


if __name__ == "__main__":
    main()
