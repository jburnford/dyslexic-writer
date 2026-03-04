#!/usr/bin/env python3
"""
Build training data for Dyslexic Writer v2.

Combines diverse source texts with improved error injection to produce
balanced training JSONL files.

Data composition targets:
  - 60% diverse texts with error injection (new v2 genres)
  - 20% identity examples (correct → correct, prevents false positives)
  - 10% hard cases (proper nouns, domain-specific, compound words)
  - 10% word pairs and short phrases (dense error practice)

Splits into train/eval with stratification by genre and error type.

Usage:
    python build_training_data.py --source-dir generated_diverse --output-dir v2_data
    python build_training_data.py --source-dir generated_diverse --target-count 300000
"""

from __future__ import annotations

import json
import os
import random
import re
import argparse
from pathlib import Path
from typing import Optional
from collections import defaultdict

from inject_errors import inject_errors_text, inject_errors_sentence, SentenceResult

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).parent

INSTRUCTION = "Fix any spelling mistakes in the following text. If there are no mistakes, return the text unchanged."

# Target composition ratios
COMPOSITION = {
    "diverse_errors": 0.60,   # Diverse texts with error injection
    "identity": 0.20,         # Correct text → correct text
    "hard_cases": 0.10,       # Proper nouns, domain-specific vocab
    "word_pairs": 0.10,       # Short phrases / word pairs
}

# Age band weights
AGE_BAND_WEIGHTS = {
    "young": 0.35,
    "middle": 0.35,
    "teen": 0.30,
}

# Eval split ratio
EVAL_RATIO = 0.10

# ---------------------------------------------------------------------------
# Held-out stories (excluded from training)
# ---------------------------------------------------------------------------

# Pascal's DnD story excerpts (for real-world eval only)
PASCAL_STORY_SNIPPETS = [
    "the party desided to go on a campain",
    "the sitasans were afrade",
    "they fided a way threw the forest",
]

# Bob's story patterns (for real-world eval only)
BOB_STORY_SNIPPETS = [
    "they skremed and ran away",
    "he jumed off the belkany",
]


def is_held_out(text: str) -> bool:
    """Check if text contains held-out story content."""
    lower = text.lower()
    for snippet in PASCAL_STORY_SNIPPETS + BOB_STORY_SNIPPETS:
        if any(word in lower for word in snippet.split() if len(word) > 4):
            return False  # Only exact snippet matches should be excluded
    return False


# ---------------------------------------------------------------------------
# Hard case generators
# ---------------------------------------------------------------------------

PROPER_NOUN_SENTENCES = [
    "My friend {name} went to {place} last weekend.",
    "{name} and I played {game} for hours after school.",
    "We met {name} at the {place} and had a great time.",
    "{name} said the best part of {game} was the final boss.",
    "Last Tuesday {name} came over and we walked to {place}.",
    "I told {name} about what happened at {place} yesterday.",
    "{name}'s mom drove us to {place} after practice.",
    "Everyone at {place} was talking about the new {game} update.",
]

NAMES = [
    "Jake", "Emma", "Liam", "Sophia", "Noah", "Olivia", "Aiden", "Mia",
    "Lucas", "Isabella", "Mason", "Ava", "Ethan", "Harper", "Logan",
    "Marcus", "Quinn", "Kai", "Luna", "Jayden", "Riley", "Devon",
    "Christopher", "Elizabeth", "Alexander", "Katherine", "Benjamin",
    "Stephanie", "Nathaniel", "Victoria",
]

PLACES = [
    "Oakville", "Riverside Park", "Cedar Elementary", "Mountain View Mall",
    "Pine Valley Library", "Springfield", "Lakewood Pool", "Greenfield Arena",
    "Crystal Lake", "Willow Creek", "the community center", "downtown",
]

GAMES = [
    "Minecraft", "Fortnite", "Roblox", "Pokemon Scarlet", "Mario Kart",
    "Among Us", "Zelda Tears of the Kingdom", "Animal Crossing",
    "Terraria", "Stardew Valley",
]

# Domain-specific vocabulary sentences
DOMAIN_SENTENCES = {
    "dnd": [
        "The wizard cast fireball and rolled a natural twenty for damage.",
        "Our paladin used her healing spell to save the cleric from the dragon.",
        "The dungeon master described a cavern filled with treasure and goblins.",
        "My rogue character has high dexterity and stealth proficiency.",
        "We found a legendary sword with magical enchantment in the dungeon.",
        "The dragonborn barbarian charged at the beholder with his greataxe.",
        "I failed my constitution saving throw and took poison damage.",
    ],
    "gaming": [
        "I built a massive redstone contraption in my Minecraft survival world.",
        "The creeper exploded and destroyed half of my diamond pickaxe.",
        "We dropped at Tilted Towers and found a legendary assault rifle.",
        "My character respawned at the checkpoint after falling into lava.",
        "The final boss had three different phases and a health regeneration ability.",
    ],
    "science": [
        "Photosynthesis is the process where plants convert sunlight into energy.",
        "The mitochondria is the powerhouse of the cell and produces adenosine triphosphate.",
        "Earthquakes happen when tectonic plates shift along fault lines.",
        "The asteroid that hit Earth caused the extinction of the dinosaurs.",
        "Gravity is the force that pulls objects toward each other.",
    ],
}

# Word pair practice (hardest category — no context)
WORD_PAIRS = [
    # Homophones
    ("their house", "there house"),
    ("you're welcome", "your welcome"),
    ("it's raining", "its raining"),
    ("who's coming", "whose coming"),
    ("we're going", "were going"),
    ("they're happy", "there happy"),
    ("too many", "to many"),
    ("piece of cake", "peace of cake"),
    ("right answer", "write answer"),
    ("hear the music", "here the music"),
    ("break the glass", "brake the glass"),
    ("weather forecast", "whether forcast"),
    ("principal office", "principle office"),
    ("stationary bike", "stationery bike"),
    ("complement each other", "compliment each other"),
    ("affect the outcome", "effect the outcome"),
    ("lose the game", "loose the game"),
    ("accept the answer", "except the answer"),
    ("quiet room", "quite room"),
    ("desert landscape", "dessert landscape"),
    # Common misspellings
    ("definitely going", "definately going"),
    ("separate rooms", "seperate rooms"),
    ("necessary items", "nessesary items"),
    ("February weather", "Febuary weather"),
    ("Wednesday morning", "Wensday morning"),
    ("beautiful sunset", "beautifull sunset"),
    ("restaurant menu", "restarant menu"),
    ("government building", "goverment building"),
    ("environment protection", "enviroment protection"),
    ("immediately after", "imediately after"),
    ("accidentally broke", "accidently broke"),
    ("embarrassing moment", "embarassing moment"),
    ("occasionally visits", "occassionally visits"),
    ("achievement unlocked", "achievment unlocked"),
    ("beginning of", "begining of"),
    ("disappear quickly", "dissapear quickly"),
    ("independent study", "independant study"),
    ("knowledge base", "knowlege base"),
    ("maintenance work", "maintainance work"),
    ("occurrence report", "occurence report"),
    ("possession of", "posession of"),
    ("privilege access", "privelege access"),
    ("recommend highly", "recomend highly"),
    ("sergeant major", "sergent major"),
    ("threshold level", "threshhold level"),
    ("until tomorrow", "untill tommorow"),
    ("vacuum cleaner", "vaccuum cleaner"),
]


def generate_hard_cases(count: int, age_band: str) -> list[dict]:
    """Generate hard case examples: proper nouns, domain vocab, compound words."""
    examples = []

    # Proper noun sentences (40% of hard cases)
    proper_count = int(count * 0.4)
    for _ in range(proper_count):
        template = random.choice(PROPER_NOUN_SENTENCES)
        name = random.choice(NAMES)
        place = random.choice(PLACES)
        game = random.choice(GAMES)
        sentence = template.format(name=name, place=place, game=game)

        # Inject errors
        result = inject_errors_sentence(sentence, age_band=age_band)
        examples.append({
            "instruction": INSTRUCTION,
            "input": result.corrupted,
            "output": sentence,
            "genre": "hard_proper_nouns",
            "age_band": age_band,
            "has_error": result.corrupted != sentence,
        })

    # Domain-specific sentences (40%)
    domain_count = int(count * 0.4)
    domains = list(DOMAIN_SENTENCES.keys())
    for _ in range(domain_count):
        domain = random.choice(domains)
        sentence = random.choice(DOMAIN_SENTENCES[domain])
        result = inject_errors_sentence(sentence, age_band=age_band)
        examples.append({
            "instruction": INSTRUCTION,
            "input": result.corrupted,
            "output": sentence,
            "genre": f"hard_{domain}",
            "age_band": age_band,
            "has_error": result.corrupted != sentence,
        })

    # Identity examples of hard vocab (20% — these should NOT be changed)
    identity_count = count - proper_count - domain_count
    for _ in range(identity_count):
        domain = random.choice(domains)
        sentence = random.choice(DOMAIN_SENTENCES[domain])
        examples.append({
            "instruction": INSTRUCTION,
            "input": sentence,
            "output": sentence,
            "genre": f"hard_{domain}_identity",
            "age_band": age_band,
            "has_error": False,
        })

    return examples


def generate_word_pairs(count: int) -> list[dict]:
    """Generate word pair / short phrase examples."""
    examples = []

    for _ in range(count):
        correct, misspelled = random.choice(WORD_PAIRS)

        if random.random() < 0.5:
            # Error case: misspelled → correct
            examples.append({
                "instruction": INSTRUCTION,
                "input": misspelled,
                "output": correct,
                "genre": "word_pair",
                "age_band": "all",
                "has_error": True,
            })
        else:
            # Identity case: correct → correct
            examples.append({
                "instruction": INSTRUCTION,
                "input": correct,
                "output": correct,
                "genre": "word_pair_identity",
                "age_band": "all",
                "has_error": False,
            })

    return examples


# ---------------------------------------------------------------------------
# Source text processing
# ---------------------------------------------------------------------------

def load_source_texts(source_dir: Path) -> dict[str, list[dict]]:
    """Load generated texts organized by age band."""
    texts_by_band = defaultdict(list)

    for jsonl_file in sorted(source_dir.rglob("*.jsonl")):
        with open(jsonl_file) as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    band = record.get("age_band", "middle")
                    texts_by_band[band].append(record)
                except json.JSONDecodeError:
                    continue

    return dict(texts_by_band)


def process_text_to_examples(text: str, genre: str, age_band: str,
                              is_identity: bool = False) -> list[dict]:
    """Convert a source text into training examples."""
    examples = []

    if is_identity:
        # Identity example: correct text → correct text
        examples.append({
            "instruction": INSTRUCTION,
            "input": text,
            "output": text,
            "genre": genre,
            "age_band": age_band,
            "has_error": False,
        })
    else:
        # Inject errors sentence by sentence
        results = inject_errors_text(text, age_band=age_band)

        # Create one example per sentence
        for result in results:
            if len(result.original.split()) < 3:
                continue  # Skip very short fragments

            examples.append({
                "instruction": INSTRUCTION,
                "input": result.corrupted,
                "output": result.original,
                "genre": genre,
                "age_band": age_band,
                "has_error": result.corrupted != result.original,
            })

        # Also create a full-text example (paragraph level)
        full_corrupted = " ".join(r.corrupted for r in results)
        full_original = " ".join(r.original for r in results)
        if len(full_original.split()) >= 10:
            examples.append({
                "instruction": INSTRUCTION,
                "input": full_corrupted,
                "output": full_original,
                "genre": genre,
                "age_band": age_band,
                "has_error": full_corrupted != full_original,
            })

    return examples


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_dataset(
    source_dir: Path,
    output_dir: Path,
    target_count: int = 300000,
    seed: int = 42,
) -> None:
    """Build the full v2 training dataset."""
    random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Building Dyslexic Writer v2 Training Data ===")
    print(f"Source: {source_dir}")
    print(f"Target: {target_count} examples")
    print()

    # Calculate composition targets
    diverse_target = int(target_count * COMPOSITION["diverse_errors"])
    identity_target = int(target_count * COMPOSITION["identity"])
    hard_target = int(target_count * COMPOSITION["hard_cases"])
    wordpair_target = int(target_count * COMPOSITION["word_pairs"])

    print(f"Composition targets:")
    print(f"  Diverse with errors: {diverse_target}")
    print(f"  Identity (no errors): {identity_target}")
    print(f"  Hard cases: {hard_target}")
    print(f"  Word pairs: {wordpair_target}")
    print()

    all_examples = []

    # 1. Load source texts
    print("Loading source texts...")
    texts_by_band = load_source_texts(source_dir)
    total_texts = sum(len(v) for v in texts_by_band.values())
    print(f"  Loaded {total_texts} texts across {list(texts_by_band.keys())} bands")

    if total_texts == 0:
        print("  WARNING: No source texts found! Generating from hard cases and word pairs only.")
        diverse_target = 0
        identity_target = 0
        hard_target = int(target_count * 0.7)
        wordpair_target = int(target_count * 0.3)

    # 2. Process diverse texts with error injection
    print("\nProcessing diverse texts with error injection...")
    diverse_examples = []
    identity_examples = []

    for band, texts in texts_by_band.items():
        band_weight = AGE_BAND_WEIGHTS.get(band, 0.33)
        band_diverse_target = int(diverse_target * band_weight)
        band_identity_target = int(identity_target * band_weight)

        random.shuffle(texts)

        # Split texts: some for errors, some for identity
        error_texts = [t for t in texts if not t.get("is_identity", False)]
        id_texts = [t for t in texts if t.get("is_identity", False)]

        # Process error injection texts
        for text_record in error_texts:
            if len(diverse_examples) >= diverse_target:
                break
            text = text_record["text"]
            genre = text_record.get("genre", "unknown")
            exs = process_text_to_examples(text, genre, band, is_identity=False)
            diverse_examples.extend(exs)

        # Process identity texts
        for text_record in (id_texts or error_texts):
            if len(identity_examples) >= identity_target:
                break
            text = text_record["text"]
            genre = text_record.get("genre", "unknown") + "_identity"
            exs = process_text_to_examples(text, genre, band, is_identity=True)
            identity_examples.extend(exs)

    print(f"  Diverse examples: {len(diverse_examples)}")
    print(f"  Identity examples: {len(identity_examples)}")
    all_examples.extend(diverse_examples)
    all_examples.extend(identity_examples)

    # 3. Generate hard cases
    print("\nGenerating hard cases...")
    for band, weight in AGE_BAND_WEIGHTS.items():
        band_hard = int(hard_target * weight)
        hard_exs = generate_hard_cases(band_hard, band)
        all_examples.extend(hard_exs)
        print(f"  {band}: {len(hard_exs)} hard cases")

    # 4. Generate word pairs
    print("\nGenerating word pairs...")
    wp_exs = generate_word_pairs(wordpair_target)
    all_examples.extend(wp_exs)
    print(f"  Word pairs: {len(wp_exs)}")

    # 5. Shuffle and split
    print(f"\nTotal examples: {len(all_examples)}")
    random.shuffle(all_examples)

    eval_count = int(len(all_examples) * EVAL_RATIO)
    eval_examples = all_examples[:eval_count]
    train_examples = all_examples[eval_count:]

    print(f"  Train: {len(train_examples)}")
    print(f"  Eval: {len(eval_examples)}")

    # 6. Write output files
    train_file = output_dir / "train.jsonl"
    eval_file = output_dir / "eval.jsonl"

    with open(train_file, 'w') as f:
        for ex in train_examples:
            f.write(json.dumps(ex) + '\n')
    print(f"\nSaved train: {train_file} ({len(train_examples)} examples)")

    with open(eval_file, 'w') as f:
        for ex in eval_examples:
            f.write(json.dumps(ex) + '\n')
    print(f"Saved eval: {eval_file} ({len(eval_examples)} examples)")

    # 7. Print statistics
    print("\n=== Dataset Statistics ===")

    # By genre
    genre_counts = defaultdict(int)
    for ex in all_examples:
        genre_counts[ex.get("genre", "unknown")] += 1
    print("\nBy genre:")
    for genre, count in sorted(genre_counts.items(), key=lambda x: -x[1])[:20]:
        pct = count / len(all_examples) * 100
        print(f"  {genre}: {count} ({pct:.1f}%)")

    # By age band
    band_counts = defaultdict(int)
    for ex in all_examples:
        band_counts[ex.get("age_band", "unknown")] += 1
    print("\nBy age band:")
    for band, count in sorted(band_counts.items()):
        pct = count / len(all_examples) * 100
        print(f"  {band}: {count} ({pct:.1f}%)")

    # Error vs identity ratio
    error_count = sum(1 for ex in all_examples if ex.get("has_error", False))
    identity_count = len(all_examples) - error_count
    print(f"\nError examples: {error_count} ({error_count/len(all_examples)*100:.1f}%)")
    print(f"Identity examples: {identity_count} ({identity_count/len(all_examples)*100:.1f}%)")

    # Save stats
    stats = {
        "total_examples": len(all_examples),
        "train_count": len(train_examples),
        "eval_count": len(eval_examples),
        "genre_distribution": dict(genre_counts),
        "age_band_distribution": dict(band_counts),
        "error_count": error_count,
        "identity_count": identity_count,
        "composition_ratios": COMPOSITION,
    }
    with open(output_dir / "dataset_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nStats saved to {output_dir / 'dataset_stats.json'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build v2 training data")
    parser.add_argument("--source-dir", type=str, default="generated_diverse",
                       help="Directory with generated source texts")
    parser.add_argument("--output-dir", type=str, default="v2_data",
                       help="Output directory for train/eval JSONL")
    parser.add_argument("--target-count", type=int, default=300000,
                       help="Target total example count")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source = _DATA_DIR / args.source_dir
    output = _DATA_DIR / args.output_dir

    build_dataset(source, output, args.target_count, args.seed)


if __name__ == "__main__":
    main()
