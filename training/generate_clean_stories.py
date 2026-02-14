#!/usr/bin/env python3
"""
Generate clean, age-appropriate stories via Gemini API for error injection.

Uses vocabulary targeting to ensure generated stories contain words that trigger
specific error rules from the Bob story error profile. Stories are generated
across 6 genre categories matching what children actually write about.

Budget: ~$15 for ~3,500 stories at ~$0.004/story.
"""

import json
import os
import random
import time
import argparse
from pathlib import Path
from typing import Optional

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("ERROR: google-genai package required. Install with:")
    print("  pip install google-genai")
    raise SystemExit(1)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).parent
_ENV_PATH = _DATA_DIR / ".env"
_VOCAB_PATH = _DATA_DIR / "vocab_targets.json"
_OUTPUT_DIR = _DATA_DIR / "generated_stories"

# Genre categories and their weights
GENRE_WEIGHTS = {
    "adventure_action": 0.25,
    "school_daily_life": 0.20,
    "animals_nature": 0.15,
    "sports_games": 0.15,
    "family_friends": 0.15,
    "fantasy_scifi": 0.10,
}

GENRE_DESCRIPTIONS = {
    "adventure_action": "an adventure story where characters explore, face danger, or go on a quest",
    "school_daily_life": "a story about school, homework, recess, lunchtime, or a regular day",
    "animals_nature": "a story about animals, pets, nature, or exploring outside",
    "sports_games": "a story about playing sports, video games, or competing in something",
    "family_friends": "a story about family, friends, a birthday, holiday, or sleepover",
    "fantasy_scifi": "a fantasy or science fiction story with magic, aliens, robots, or superpowers",
}

SYSTEM_PROMPT = """You are a creative writing assistant that writes short stories
at a child's level. Write naturally and simply. Do not use sophisticated vocabulary
or complex sentence structures. Write as a real 8-10 year old child would write —
simple plots, direct action, lots of dialogue, and not too polished."""

# ---------------------------------------------------------------------------
# Load environment and vocab
# ---------------------------------------------------------------------------

def load_api_key() -> str:
    """Load Gemini API key from .env file or environment."""
    # Check environment first
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key

    # Try .env file
    if _ENV_PATH.exists():
        with open(_ENV_PATH) as f:
            for line in f:
                line = line.strip()
                if line.startswith("GEMINI_API_KEY="):
                    return line.split("=", 1)[1].strip()

    raise ValueError(
        "GEMINI_API_KEY not found. Set it in environment or in training/.env"
    )


def load_vocab_targets() -> dict:
    """Load vocabulary target lists."""
    with open(_VOCAB_PATH) as f:
        data = json.load(f)
    # Remove comment keys
    return {k: v for k, v in data.items() if not k.startswith("_")}


def select_vocab_targets(vocab: dict, count: int = 12) -> list[str]:
    """Select a diverse set of vocabulary targets across rule categories."""
    # Pick words from different categories to maximize rule coverage
    categories = list(vocab.keys())
    random.shuffle(categories)

    selected = []
    for cat in categories:
        if len(selected) >= count:
            break
        words = vocab[cat]
        # Pick 1-2 words from each category
        n = min(2, count - len(selected))
        selected.extend(random.sample(words, min(n, len(words))))

    random.shuffle(selected)
    return selected


def choose_genre() -> str:
    """Weighted random genre selection."""
    genres = list(GENRE_WEIGHTS.keys())
    weights = list(GENRE_WEIGHTS.values())
    return random.choices(genres, weights=weights, k=1)[0]


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_story_prompt(genre: str, vocab_words: list[str]) -> str:
    """Build a prompt for Gemini to generate one clean story."""
    genre_desc = GENRE_DESCRIPTIONS[genre]
    vocab_str = ", ".join(vocab_words)

    return f"""Write a short story a child aged 8-10 might write. It should be {genre_desc}.

The story should:
- Be 80-150 words long
- Use simple and compound sentences (no complex subordinate clauses)
- Average 8-12 words per sentence
- Include at least 2 lines of dialogue
- Be written at Grade 2-3 reading level
- Include some of these words naturally: {vocab_str}

Write ONLY the story. No title. Write it as a child would structure it (simple plot, not too polished, some run-on sentences are fine)."""


def build_negative_prompt(variant: str) -> str:
    """Build a prompt for negative examples (identity pairs)."""
    prompts = {
        "tricky_words": """Write a short paragraph (60-100 words) a child aged 8-10 might write
that includes words that look unusual but are spelled correctly: knight, island, caught,
through, enough, thought, daughter, straight. Use at least 4 of these words naturally.
Write at Grade 2-3 level. Write ONLY the paragraph.""",

        "informal_correct": """Write a short paragraph (60-100 words) of informal but
grammatically acceptable children's writing. Include contractions (don't, can't, it's,
we're), exclamations (Wow!, No way!, Oh man!), and sentence fragments that sound natural
in kids' speech. Everything should be spelled correctly. Write ONLY the paragraph.""",

        "proper_nouns": """Write a short paragraph (60-100 words) a child might write about
playing video games or watching shows. Include proper nouns like Minecraft, Pokemon, Roblox,
Fortnite, Spider-Man, or similar. All spelling should be correct. Write at Grade 2-3 level.
Write ONLY the paragraph.""",

        "simple_correct": """Write a very simple paragraph (60-100 words) using only common,
easy-to-spell words. A child aged 8 writing about their day. Short sentences. Simple words.
Everything spelled correctly. Write ONLY the paragraph.""",
    }
    return prompts.get(variant, prompts["simple_correct"])


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_stories(
    num_stories: int = 3500,
    num_negative: int = 1500,
    batch_size: int = 50,
    output_dir: Optional[Path] = None,
    delay: float = 0.5,
    resume_from: int = 0,
) -> None:
    """
    Generate clean stories and negative examples via Gemini API.

    Args:
        num_stories: Number of clean stories for error injection.
        num_negative: Number of negative/identity examples.
        batch_size: Stories per output file.
        output_dir: Where to save outputs.
        delay: Seconds between API calls to avoid rate limiting.
        resume_from: Story index to resume from (for interrupted runs).
    """
    if output_dir is None:
        output_dir = _OUTPUT_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    api_key = load_api_key()
    vocab = load_vocab_targets()

    # Initialize Gemini client
    client = genai.Client(api_key=api_key)
    model_id = "gemini-3-pro-preview"

    print(f"Generating {num_stories} stories + {num_negative} negative examples")
    print(f"Using model: {model_id}")
    print(f"Output directory: {output_dir}")
    print()

    # --- Generate clean stories for error injection ---
    stories = []
    batch_num = resume_from // batch_size
    failed = 0
    max_failures = 50

    for i in range(resume_from, num_stories):
        genre = choose_genre()
        vocab_words = select_vocab_targets(vocab, count=random.randint(8, 15))
        prompt = build_story_prompt(genre, vocab_words)

        try:
            response = client.models.generate_content(
                model=model_id,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.9,
                    max_output_tokens=4096,
                    thinking_config=types.ThinkingConfig(thinking_budget=128),
                ),
            )

            story_text = response.text.strip() if response.text else ""
            if not story_text:
                failed += 1
                print(f"  [{i}] Empty response, skipping")
                continue

            stories.append({
                "id": f"story_{i:05d}",
                "text": story_text,
                "genre": genre,
                "vocab_targets": vocab_words,
                "type": "clean_for_injection",
            })

            if (i + 1) % 10 == 0:
                print(f"  [{i + 1}/{num_stories}] Generated ({genre}): {story_text[:60]}...")

        except Exception as e:
            failed += 1
            print(f"  [{i}] Error: {e}")
            if failed > max_failures:
                print(f"Too many failures ({failed}). Stopping.")
                break
            time.sleep(2)  # back off on errors
            continue

        # Save batch
        if len(stories) >= batch_size:
            batch_file = output_dir / f"stories_batch_{batch_num:04d}.json"
            with open(batch_file, "w") as f:
                json.dump(stories, f, indent=2)
            print(f"  Saved batch {batch_num} ({len(stories)} stories) -> {batch_file}")
            stories = []
            batch_num += 1

        time.sleep(delay)

    # Save remaining stories
    if stories:
        batch_file = output_dir / f"stories_batch_{batch_num:04d}.json"
        with open(batch_file, "w") as f:
            json.dump(stories, f, indent=2)
        print(f"  Saved final batch {batch_num} ({len(stories)} stories)")

    # --- Generate negative examples ---
    print(f"\nGenerating {num_negative} negative examples...")
    negatives = []
    neg_batch = 0
    neg_variants = ["tricky_words", "informal_correct", "proper_nouns", "simple_correct"]

    for i in range(num_negative):
        variant = random.choice(neg_variants)
        prompt = build_negative_prompt(variant)

        try:
            response = client.models.generate_content(
                model=model_id,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.8,
                    max_output_tokens=4096,
                    thinking_config=types.ThinkingConfig(thinking_budget=128),
                ),
            )

            text = response.text.strip() if response.text else ""
            if not text:
                continue

            negatives.append({
                "id": f"negative_{i:05d}",
                "text": text,
                "variant": variant,
                "type": "negative_identity",
            })

            if (i + 1) % 10 == 0:
                print(f"  [{i + 1}/{num_negative}] Negative ({variant}): {text[:60]}...")

        except Exception as e:
            print(f"  [{i}] Error: {e}")
            time.sleep(2)
            continue

        # Save batch
        if len(negatives) >= batch_size:
            batch_file = output_dir / f"negatives_batch_{neg_batch:04d}.json"
            with open(batch_file, "w") as f:
                json.dump(negatives, f, indent=2)
            print(f"  Saved negative batch {neg_batch} ({len(negatives)} examples)")
            negatives = []
            neg_batch += 1

        time.sleep(delay)

    # Save remaining negatives
    if negatives:
        batch_file = output_dir / f"negatives_batch_{neg_batch:04d}.json"
        with open(batch_file, "w") as f:
            json.dump(negatives, f, indent=2)
        print(f"  Saved final negative batch ({len(negatives)} examples)")

    print(f"\nDone! Failed calls: {failed}")
    print(f"Total stories: {num_stories - failed}")
    print(f"Output directory: {output_dir}")


# ---------------------------------------------------------------------------
# Pilot mode (small test run)
# ---------------------------------------------------------------------------

def generate_pilot(count: int = 10) -> None:
    """Generate a small pilot batch for manual review."""
    print(f"=== PILOT MODE: Generating {count} stories ===\n")
    generate_stories(
        num_stories=count,
        num_negative=max(3, count // 3),
        batch_size=count,
        output_dir=_OUTPUT_DIR / "pilot",
        delay=1.0,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate clean stories via Gemini API for error injection"
    )
    parser.add_argument(
        "--pilot", action="store_true",
        help="Generate a small pilot batch (10 stories) for review"
    )
    parser.add_argument(
        "--stories", type=int, default=3500,
        help="Number of clean stories to generate (default: 3500)"
    )
    parser.add_argument(
        "--negative", type=int, default=1500,
        help="Number of negative examples to generate (default: 1500)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=50,
        help="Stories per output file (default: 50)"
    )
    parser.add_argument(
        "--delay", type=float, default=0.5,
        help="Seconds between API calls (default: 0.5)"
    )
    parser.add_argument(
        "--resume-from", type=int, default=0,
        help="Story index to resume from (for interrupted runs)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: training/generated_stories)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    if args.pilot:
        generate_pilot()
    else:
        output_dir = Path(args.output_dir) if args.output_dir else None
        generate_stories(
            num_stories=args.stories,
            num_negative=args.negative,
            batch_size=args.batch_size,
            output_dir=output_dir,
            delay=args.delay,
            resume_from=args.resume_from,
        )


if __name__ == "__main__":
    main()
