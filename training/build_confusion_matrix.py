#!/usr/bin/env python3
"""
Build a character-level confusion matrix from existing spelling error corpora.

Aligns misspelling→correct pairs at the character level and counts
substitution, insertion, and deletion frequencies. Output is a JSON file
usable for NeuSpell-style data augmentation.

Input data:
  - training-data/all_pairs.csv  (93K word-level pairs)
  - training-data/birkbeck.dat   (36K misspellings)
  - training-data/holbrook.dat   (1.8K misspellings)

Output:
  - training/confusion_matrix.json
"""

import csv
import json
import re
from collections import defaultdict
from pathlib import Path


_SCRIPT_DIR = Path(__file__).parent
_DATA_DIR = _SCRIPT_DIR.parent / "training-data"


# ---------------------------------------------------------------------------
# Edit distance alignment (Needleman-Wunsch style)
# ---------------------------------------------------------------------------

def align_chars(source: str, target: str) -> list[tuple[str, str]]:
    """
    Align two strings character-by-character using edit distance.
    Returns list of (source_char, target_char) pairs.
    '-' represents a gap (insertion/deletion).
    """
    m, n = len(source), len(target)

    # DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if source[i - 1] == target[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # deletion
                    dp[i][j - 1],      # insertion
                    dp[i - 1][j - 1],  # substitution
                )

    # Traceback
    alignment = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and (
            source[i - 1] == target[j - 1] or
            dp[i][j] == dp[i - 1][j - 1] + 1
        ):
            alignment.append((source[i - 1], target[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            alignment.append((source[i - 1], "-"))  # deletion
            i -= 1
        else:
            alignment.append(("-", target[j - 1]))  # insertion
            j -= 1

    alignment.reverse()
    return alignment


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_all_pairs_csv(path: Path) -> list[tuple[str, str]]:
    """Load misspelling→correct pairs from all_pairs.csv."""
    pairs = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            m = row.get("misspelling", "").strip().lower()
            c = row.get("correct", "").strip().lower()
            if m and c and m != c:
                pairs.append((m, c))
    return pairs


def load_birkbeck(path: Path) -> list[tuple[str, str]]:
    """Load pairs from birkbeck.dat ($correct then variants)."""
    pairs = []
    correct = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("$"):
                correct = line[1:].lower().replace("_", " ")
            elif correct:
                misspelling = line.lower().replace("_", " ")
                if misspelling != correct:
                    pairs.append((misspelling, correct))
    return pairs


def load_holbrook(path: Path) -> list[tuple[str, str]]:
    """Load pairs from holbrook.dat ($correct then misspelling+count lines)."""
    pairs = []
    correct = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("$"):
                correct = line[1:].lower()
            elif correct:
                # Format: "misspelling count"
                parts = line.rsplit(None, 1)
                if len(parts) == 2:
                    misspelling = parts[0].lower()
                    try:
                        count = int(parts[1])
                    except ValueError:
                        misspelling = line.lower()
                        count = 1
                else:
                    misspelling = line.lower()
                    count = 1
                if misspelling != correct:
                    # Add multiple times based on frequency
                    for _ in range(count):
                        pairs.append((misspelling, correct))
    return pairs


# ---------------------------------------------------------------------------
# Build confusion matrix
# ---------------------------------------------------------------------------

def build_matrix(pairs: list[tuple[str, str]]) -> dict:
    """
    Build confusion matrix from misspelling→correct pairs.

    Returns dict with:
      - substitutions: {correct_char: {wrong_char: count}}
      - deletions: {correct_char: count}  (char in correct, missing in misspelling)
      - insertions: {wrong_char: count}   (char in misspelling, not in correct)
      - position_weights: {position: weight}  (where errors tend to occur)
    """
    substitutions = defaultdict(lambda: defaultdict(int))
    deletions = defaultdict(int)
    insertions = defaultdict(int)
    position_counts = defaultdict(int)
    total_aligned = 0

    for misspelling, correct in pairs:
        # Only process single words, skip multi-word
        if " " in misspelling or " " in correct:
            continue
        # Skip very long or very short
        if len(correct) < 2 or len(correct) > 30:
            continue

        alignment = align_chars(misspelling, correct)

        for idx, (src, tgt) in enumerate(alignment):
            # Normalize position to [0, 1] range
            pos_bucket = min(9, int(idx / max(len(alignment), 1) * 10))

            if src == tgt:
                continue  # match, no error

            if src == "-":
                # Deletion: char exists in correct but missing in misspelling
                deletions[tgt] += 1
            elif tgt == "-":
                # Insertion: extra char in misspelling
                insertions[src] += 1
            else:
                # Substitution: wrong char for correct char
                substitutions[tgt][src] += 1

            position_counts[pos_bucket] += 1
            total_aligned += 1

    # Normalize position weights
    position_weights = {}
    if total_aligned > 0:
        for pos, count in position_counts.items():
            position_weights[str(pos)] = round(count / total_aligned, 4)

    # Convert to regular dicts and compute probabilities
    sub_probs = {}
    for correct_char, wrong_chars in substitutions.items():
        total = sum(wrong_chars.values())
        sub_probs[correct_char] = {
            wc: round(count / total, 4)
            for wc, count in sorted(wrong_chars.items(), key=lambda x: -x[1])
        }

    del_total = sum(deletions.values()) or 1
    del_probs = {
        c: round(count / del_total, 4)
        for c, count in sorted(deletions.items(), key=lambda x: -x[1])
    }

    ins_total = sum(insertions.values()) or 1
    ins_probs = {
        c: round(count / ins_total, 4)
        for c, count in sorted(insertions.items(), key=lambda x: -x[1])
    }

    return {
        "substitutions": sub_probs,
        "deletions": del_probs,
        "insertions": ins_probs,
        "position_weights": position_weights,
        "stats": {
            "total_pairs_processed": len(pairs),
            "total_errors_aligned": total_aligned,
            "substitution_count": sum(sum(v.values()) for v in substitutions.values()),
            "deletion_count": sum(deletions.values()),
            "insertion_count": sum(insertions.values()),
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Building character-level confusion matrix...\n")

    all_pairs = []

    # Load all_pairs.csv
    csv_path = _DATA_DIR / "all_pairs.csv"
    if csv_path.exists():
        pairs = load_all_pairs_csv(csv_path)
        print(f"  all_pairs.csv: {len(pairs):,} pairs")
        all_pairs.extend(pairs)
    else:
        print(f"  all_pairs.csv: NOT FOUND")

    # Load birkbeck.dat
    birk_path = _DATA_DIR / "birkbeck.dat"
    if birk_path.exists():
        pairs = load_birkbeck(birk_path)
        print(f"  birkbeck.dat:  {len(pairs):,} pairs")
        all_pairs.extend(pairs)
    else:
        print(f"  birkbeck.dat: NOT FOUND")

    # Load holbrook.dat
    holb_path = _DATA_DIR / "holbrook.dat"
    if holb_path.exists():
        pairs = load_holbrook(holb_path)
        print(f"  holbrook.dat:  {len(pairs):,} pairs")
        all_pairs.extend(pairs)
    else:
        print(f"  holbrook.dat: NOT FOUND")

    print(f"\n  Total pairs: {len(all_pairs):,}")

    # Build matrix
    matrix = build_matrix(all_pairs)

    # Print summary
    stats = matrix["stats"]
    print(f"\n  Errors aligned: {stats['total_errors_aligned']:,}")
    print(f"    Substitutions: {stats['substitution_count']:,}")
    print(f"    Deletions:     {stats['deletion_count']:,}")
    print(f"    Insertions:    {stats['insertion_count']:,}")

    # Top substitutions
    print("\n  Top 15 substitutions (correct → wrong):")
    all_subs = []
    for correct_char, wrong_chars in matrix["substitutions"].items():
        for wrong_char, prob in wrong_chars.items():
            all_subs.append((correct_char, wrong_char, prob))
    all_subs.sort(key=lambda x: -x[2])
    for c, w, p in all_subs[:15]:
        print(f"    '{c}' → '{w}': {p:.3f}")

    # Save
    output_path = _SCRIPT_DIR / "confusion_matrix.json"
    with open(output_path, "w") as f:
        json.dump(matrix, f, indent=2)
    print(f"\n  Saved to {output_path}")


if __name__ == "__main__":
    main()
