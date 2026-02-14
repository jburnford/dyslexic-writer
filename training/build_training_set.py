#!/usr/bin/env python3
"""
Assemble the final training dataset from generated stories + error injection.

Pipeline:
1. Load clean stories from Gemini generation
2. Split into sentences
3. Apply error injection with target distribution
4. Add identity pairs from negative examples
5. Record full annotation metadata
6. Shuffle and split into train/eval
7. Hold out Bob story sentences for ecological validation

Target composition:
- ~60% error-injected sentences (positive examples)
- ~40% identity/pass-through sentences (negative examples)
- Error sentences: ~35% with 1 error, ~20% with 2 errors, ~5% with 3+
"""

import json
import os
import random
import re
import argparse
from pathlib import Path
from typing import Optional

from inject_errors import inject_errors_sentence, SentenceResult

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).parent
_STORIES_DIR = _DATA_DIR / "generated_stories"
_OUTPUT_DIR = _DATA_DIR

# Target distribution for the final dataset
TARGET_IDENTITY_RATIO = 0.40     # 40% identity pairs
TARGET_1_ERROR_RATIO = 0.35      # 35% with 1 error
TARGET_2_ERROR_RATIO = 0.20      # 20% with 2 errors
TARGET_3PLUS_ERROR_RATIO = 0.05  # 5% with 3+ errors

# Bob story sentences for held-out evaluation (NEVER in training)
BOB_STORY_EVAL_PAIRS = [
    # (corrupted, correct) pairs from error profile
    {"input": "Bob was a Bowring guy, he never went to a party he didn't need to for work.",
     "target": "Bob was a boring guy, he never went to a party he didn't need to for work.",
     "errors": [{"written": "Bowring", "target": "boring", "category": "phonological"}]},

    {"input": "he walked out on to his belkany to get some fresh air",
     "target": "he walked out on to his balcony to get some fresh air",
     "errors": [{"written": "belkany", "target": "balcony", "category": "phonological"}]},

    {"input": "he hate going outside",
     "target": "he hated going outside",
     "errors": [{"written": "hate", "target": "hated", "category": "morphological"}]},

    {"input": "he skremed at the hights",
     "target": "he screamed at the heights",
     "errors": [{"written": "skremed", "target": "screamed", "category": "phonological"},
                {"written": "hights", "target": "heights", "category": "orthographic"}]},

    {"input": "the ailens falled from the selen",
     "target": "the aliens fell from the ceiling",
     "errors": [{"written": "ailens", "target": "aliens", "category": "orthographic"},
                {"written": "falled", "target": "fell", "category": "morphological"},
                {"written": "selen", "target": "ceiling", "category": "phonological"}]},

    {"input": "he fented and the he got up agan",
     "target": "he fainted and then he got up again",
     "errors": [{"written": "fented", "target": "fainted", "category": "phonological"},
                {"written": "the", "target": "then", "category": "orthographic"},
                {"written": "agan", "target": "again", "category": "phonological"}]},

    {"input": "he hered skreming from the hights",
     "target": "he heard screaming from the heights",
     "errors": [{"written": "hered", "target": "heard", "category": "phonological"},
                {"written": "skreming", "target": "screaming", "category": "phonological"},
                {"written": "hights", "target": "heights", "category": "orthographic"}]},

    {"input": "he see a man fall from a bailden",
     "target": "he saw a man fell from a building",
     "errors": [{"written": "see", "target": "saw", "category": "morphological"},
                {"written": "fall", "target": "fell", "category": "morphological"},
                {"written": "bailden", "target": "building", "category": "phonological"}]},

    {"input": "the bird flaw fast and gab him",
     "target": "the bird flew fast and grabbed him",
     "errors": [{"written": "flaw", "target": "flew", "category": "phonological"},
                {"written": "gab", "target": "grabbed", "category": "phonological"}]},

    {"input": "he drope him on the roof and he was safe",
     "target": "he dropped him on the roof and he was safe",
     "errors": [{"written": "drope", "target": "dropped", "category": "orthographic"}]},

    {"input": "thats how Bob fall from the hights",
     "target": "that's how Bob fell from the heights",
     "errors": [{"written": "thats", "target": "that's", "category": "orthographic"},
                {"written": "fall", "target": "fell", "category": "morphological"},
                {"written": "hights", "target": "heights", "category": "orthographic"}]},

    {"input": "he tried bugy-juming from the belkany",
     "target": "he tried bungee jumping from the balcony",
     "errors": [{"written": "bugy-juming", "target": "bungee jumping", "category": "phonological"},
                {"written": "belkany", "target": "balcony", "category": "phonological"}]},

    {"input": "Mow he is broken but safe",
     "target": "Now he is broken but safe",
     "errors": [{"written": "Mow", "target": "Now", "category": "orthographic"}]},
]


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences. Handle dialogue and run-ons."""
    # Split on sentence-ending punctuation followed by space or end
    parts = re.split(r'(?<=[.!?])\s+', text.strip())

    sentences = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Skip very short fragments
        if len(part.split()) < 3:
            continue
        # Skip very long sentences (probably run-ons, split them)
        words = part.split()
        if len(words) > 25:
            # Split at commas or conjunctions
            sub_parts = re.split(r',\s+(?:and|but|so|then)\s+', part)
            for sp in sub_parts:
                sp = sp.strip()
                if len(sp.split()) >= 3:
                    sentences.append(sp)
        else:
            sentences.append(part)

    return sentences


# ---------------------------------------------------------------------------
# Load generated data
# ---------------------------------------------------------------------------

def load_stories(stories_dir: Path) -> list[dict]:
    """Load all story batch files."""
    stories = []
    for f in sorted(stories_dir.glob("stories_batch_*.json")):
        with open(f) as fh:
            batch = json.load(fh)
            stories.extend(batch)
    print(f"Loaded {len(stories)} clean stories")
    return stories


def load_negatives(stories_dir: Path) -> list[dict]:
    """Load all negative example batch files."""
    negatives = []
    for f in sorted(stories_dir.glob("negatives_batch_*.json")):
        with open(f) as fh:
            batch = json.load(fh)
            negatives.extend(batch)
    print(f"Loaded {len(negatives)} negative examples")
    return negatives


# ---------------------------------------------------------------------------
# Build training pairs
# ---------------------------------------------------------------------------

def build_error_pairs(stories: list[dict]) -> list[dict]:
    """Apply error injection to clean story sentences."""
    pairs = []
    stats = {"0_errors": 0, "1_error": 0, "2_errors": 0, "3plus_errors": 0}

    for story in stories:
        sentences = split_into_sentences(story["text"])
        for sent_idx, sentence in enumerate(sentences):
            result = inject_errors_sentence(sentence)

            pair = {
                "id": f"{story['id']}_sent_{sent_idx:02d}",
                "input": result.corrupted,
                "target": result.original,
                "errors": [
                    {
                        "target_word": e.target_word,
                        "written_as": e.written_as,
                        "rule": e.rule,
                        "category": e.category,
                        "provenance": e.provenance,
                        "confidence": e.confidence,
                    }
                    for e in result.errors
                ],
                "error_count": result.error_count,
                "word_error_rate": round(result.word_error_rate, 3),
                "source_story": story["id"],
                "genre": story.get("genre", "unknown"),
                "type": "error_injected" if result.error_count > 0 else "identity",
                "batch": "synthetic_v1",
                "reviewed": False,
            }
            pairs.append(pair)

            # Track stats
            if result.error_count == 0:
                stats["0_errors"] += 1
            elif result.error_count == 1:
                stats["1_error"] += 1
            elif result.error_count == 2:
                stats["2_errors"] += 1
            else:
                stats["3plus_errors"] += 1

    print(f"\nError injection stats:")
    total = sum(stats.values())
    for k, v in stats.items():
        pct = v / total * 100 if total > 0 else 0
        print(f"  {k}: {v} ({pct:.1f}%)")

    return pairs


def build_identity_pairs(negatives: list[dict]) -> list[dict]:
    """Convert negative examples to identity pairs (input == target)."""
    pairs = []
    for neg in negatives:
        sentences = split_into_sentences(neg["text"])
        for sent_idx, sentence in enumerate(sentences):
            pair = {
                "id": f"{neg['id']}_sent_{sent_idx:02d}",
                "input": sentence,
                "target": sentence,
                "errors": [],
                "error_count": 0,
                "word_error_rate": 0.0,
                "source_story": neg["id"],
                "genre": neg.get("variant", "negative"),
                "type": "identity",
                "batch": "synthetic_v1",
                "reviewed": False,
            }
            pairs.append(pair)
    return pairs


# ---------------------------------------------------------------------------
# Distribution balancing
# ---------------------------------------------------------------------------

def balance_dataset(
    error_pairs: list[dict],
    identity_pairs: list[dict],
    target_size: Optional[int] = None,
) -> list[dict]:
    """
    Balance the dataset to match target distribution.

    Target:
    - ~40% identity pairs
    - ~35% with 1 error
    - ~20% with 2 errors
    - ~5% with 3+ errors
    """
    # Separate error pairs by count
    by_count = {"0": [], "1": [], "2": [], "3+": []}
    for p in error_pairs:
        ec = p["error_count"]
        if ec == 0:
            by_count["0"].append(p)
        elif ec == 1:
            by_count["1"].append(p)
        elif ec == 2:
            by_count["2"].append(p)
        else:
            by_count["3+"].append(p)

    # Combine identity sources
    all_identity = by_count["0"] + identity_pairs

    if target_size is None:
        # Use all available data, adjust ratios
        target_size = len(error_pairs) + len(identity_pairs)

    # Calculate target counts
    n_identity = int(target_size * TARGET_IDENTITY_RATIO)
    n_1_error = int(target_size * TARGET_1_ERROR_RATIO)
    n_2_error = int(target_size * TARGET_2_ERROR_RATIO)
    n_3plus = int(target_size * TARGET_3PLUS_ERROR_RATIO)

    # Sample (with replacement if needed)
    def sample_or_all(source, n):
        if len(source) == 0:
            return []
        if len(source) >= n:
            return random.sample(source, n)
        # Oversample with replacement
        return random.choices(source, k=n)

    balanced = []
    balanced.extend(sample_or_all(all_identity, n_identity))
    balanced.extend(sample_or_all(by_count["1"], n_1_error))
    balanced.extend(sample_or_all(by_count["2"], n_2_error))
    balanced.extend(sample_or_all(by_count["3+"], n_3plus))

    random.shuffle(balanced)

    print(f"\nBalanced dataset composition ({len(balanced)} total):")
    counts = {"identity": 0, "1_error": 0, "2_errors": 0, "3+_errors": 0}
    for p in balanced:
        ec = p["error_count"]
        if ec == 0:
            counts["identity"] += 1
        elif ec == 1:
            counts["1_error"] += 1
        elif ec == 2:
            counts["2_errors"] += 1
        else:
            counts["3+_errors"] += 1

    for k, v in counts.items():
        pct = v / len(balanced) * 100 if balanced else 0
        print(f"  {k}: {v} ({pct:.1f}%)")

    return balanced


# ---------------------------------------------------------------------------
# Split and save
# ---------------------------------------------------------------------------

def split_train_eval(
    pairs: list[dict],
    eval_ratio: float = 0.10,
) -> tuple[list[dict], list[dict]]:
    """Split into train and eval sets (stratified by error count)."""
    random.shuffle(pairs)

    # Stratified split
    by_type = {}
    for p in pairs:
        t = p["type"]
        by_type.setdefault(t, []).append(p)

    train = []
    eval_set = []
    for t, items in by_type.items():
        split_idx = max(1, int(len(items) * eval_ratio))
        eval_set.extend(items[:split_idx])
        train.extend(items[split_idx:])

    random.shuffle(train)
    random.shuffle(eval_set)

    return train, eval_set


def save_dataset(
    pairs: list[dict],
    filename: str,
    output_dir: Path,
) -> None:
    """Save dataset as JSONL."""
    filepath = output_dir / filename
    with open(filepath, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")
    print(f"Saved {len(pairs)} pairs to {filepath}")


def save_bob_eval(output_dir: Path) -> None:
    """Save Bob story evaluation set."""
    filepath = output_dir / "bob_story_eval.json"
    with open(filepath, "w") as f:
        json.dump(BOB_STORY_EVAL_PAIRS, f, indent=2)
    print(f"Saved {len(BOB_STORY_EVAL_PAIRS)} Bob story eval pairs to {filepath}")


# ---------------------------------------------------------------------------
# Validation checks
# ---------------------------------------------------------------------------

def validate_dataset(pairs: list[dict]) -> dict:
    """Run automated validation checks on the dataset."""
    checks = {
        "total_pairs": len(pairs),
        "identity_pairs": 0,
        "error_pairs": 0,
        "avg_error_rate": 0.0,
        "category_distribution": {"phonological": 0, "orthographic": 0, "morphological": 0},
        "provenance_distribution": {"attested": 0, "inferred": 0, "literature-based": 0},
        "issues": [],
    }

    total_wer = 0
    for p in pairs:
        if p["error_count"] == 0:
            checks["identity_pairs"] += 1
        else:
            checks["error_pairs"] += 1
            total_wer += p["word_error_rate"]

        for e in p.get("errors", []):
            cat = e.get("category", "unknown")
            prov = e.get("provenance", "unknown")
            checks["category_distribution"][cat] = checks["category_distribution"].get(cat, 0) + 1
            checks["provenance_distribution"][prov] = checks["provenance_distribution"].get(prov, 0) + 1

    if checks["error_pairs"] > 0:
        checks["avg_error_rate"] = round(total_wer / checks["error_pairs"], 3)

    # Check identity ratio
    identity_ratio = checks["identity_pairs"] / len(pairs) if pairs else 0
    if identity_ratio < 0.30:
        checks["issues"].append(f"Identity ratio too low: {identity_ratio:.2f} (target: ~0.40)")
    elif identity_ratio > 0.55:
        checks["issues"].append(f"Identity ratio too high: {identity_ratio:.2f} (target: ~0.40)")

    # Check category distribution
    total_errors = sum(checks["category_distribution"].values())
    if total_errors > 0:
        phon_ratio = checks["category_distribution"].get("phonological", 0) / total_errors
        if phon_ratio < 0.40:
            checks["issues"].append(f"Phonological ratio low: {phon_ratio:.2f} (target: ~0.54)")

    return checks


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_dataset(
    stories_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    target_size: Optional[int] = None,
    seed: Optional[int] = None,
) -> None:
    """Run the full data assembly pipeline."""
    if stories_dir is None:
        stories_dir = _STORIES_DIR
    if output_dir is None:
        output_dir = _OUTPUT_DIR
    if seed is not None:
        random.seed(seed)

    stories_dir = Path(stories_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Building Training Dataset")
    print("=" * 60)

    # Step 1: Load data
    stories = load_stories(stories_dir)
    negatives = load_negatives(stories_dir)

    if not stories:
        print("\nNo stories found! Run generate_clean_stories.py first.")
        print(f"Expected files in: {stories_dir}")
        return

    # Step 2: Build pairs
    print("\n--- Error Injection ---")
    error_pairs = build_error_pairs(stories)

    print("\n--- Identity Pairs ---")
    identity_pairs = build_identity_pairs(negatives)
    print(f"Created {len(identity_pairs)} identity pairs from negatives")

    # Step 3: Balance
    print("\n--- Balancing Dataset ---")
    balanced = balance_dataset(error_pairs, identity_pairs, target_size)

    # Step 4: Validate
    print("\n--- Validation ---")
    checks = validate_dataset(balanced)
    print(f"Total pairs: {checks['total_pairs']}")
    print(f"Identity: {checks['identity_pairs']} | Errors: {checks['error_pairs']}")
    print(f"Avg error rate: {checks['avg_error_rate']}")
    print(f"Category dist: {checks['category_distribution']}")
    print(f"Provenance dist: {checks['provenance_distribution']}")
    if checks["issues"]:
        print("ISSUES:")
        for issue in checks["issues"]:
            print(f"  WARNING: {issue}")

    # Step 5: Split
    print("\n--- Train/Eval Split ---")
    train, eval_set = split_train_eval(balanced)

    # Step 6: Save
    print("\n--- Saving ---")
    save_dataset(train, "synthetic_train.jsonl", output_dir)
    save_dataset(eval_set, "synthetic_eval.jsonl", output_dir)
    save_bob_eval(output_dir)

    # Save validation report
    report_path = output_dir / "dataset_report.json"
    with open(report_path, "w") as f:
        json.dump(checks, f, indent=2)
    print(f"Saved validation report to {report_path}")

    print("\n" + "=" * 60)
    print("Dataset build complete!")
    print(f"  Train: {len(train)} pairs")
    print(f"  Eval:  {len(eval_set)} pairs")
    print(f"  Bob story eval: {len(BOB_STORY_EVAL_PAIRS)} pairs (held out)")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Assemble training dataset from generated stories + error injection"
    )
    parser.add_argument(
        "--stories-dir", type=str, default=None,
        help="Directory containing generated story batches"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for training files"
    )
    parser.add_argument(
        "--target-size", type=int, default=None,
        help="Target total dataset size (default: use all data)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    args = parser.parse_args()

    build_dataset(
        stories_dir=Path(args.stories_dir) if args.stories_dir else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        target_size=args.target_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
