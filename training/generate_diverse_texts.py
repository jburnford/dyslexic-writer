#!/usr/bin/env python3
"""
Generate diverse source texts for spelling correction training data v2.

Expands beyond the 6-genre story generator to 20+ formats/genres that
better represent how kids actually write: gaming recaps, social media,
school assignments, diary entries, how-to guides, letters, fan fiction,
sports recaps, personal narratives with proper nouns, etc.

Also generates texts with imperfect grammar (run-on sentences, fragments,
informal style) since real kids don't write perfectly.

Usage:
    python generate_diverse_texts.py --age-band young --count 2000
    python generate_diverse_texts.py --age-band teen --count 3000
    python generate_diverse_texts.py --age-band all --count 3000
"""

from __future__ import annotations

import json
import os
import random
import re
import time
import argparse
import requests
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).parent
_OUTPUT_DIR = _DATA_DIR / "generated_diverse"

API_URL = os.environ.get("LLM_API_URL", "http://platogpu002:8000")
API_MODEL = os.environ.get("LLM_MODEL", "Valdemardi/DeepSeek-R1-Distill-Llama-70B-AWQ")

# ---------------------------------------------------------------------------
# Genre/format definitions (20+ categories)
# ---------------------------------------------------------------------------

GENRES = {
    # --- Narrative ---
    "adventure_action": {
        "desc": "an adventure story with action, exploring, danger, or a quest",
        "ages": ["young", "middle", "teen"],
    },
    "fantasy_scifi": {
        "desc": "a fantasy or sci-fi story with magic, aliens, robots, or superpowers",
        "ages": ["young", "middle", "teen"],
    },
    "fan_fiction": {
        "desc": "fan fiction based on a popular book, movie, or game (Minecraft, Harry Potter, Pokemon, etc.)",
        "ages": ["middle", "teen"],
    },
    "horror_mystery": {
        "desc": "a spooky story or mystery — something strange happens",
        "ages": ["middle", "teen"],
    },

    # --- Gaming ---
    "dnd_recap": {
        "desc": "a Dungeons & Dragons campaign recap — describing characters, battles, quests, and loot",
        "ages": ["middle", "teen"],
    },
    "minecraft_story": {
        "desc": "a Minecraft adventure — building, mining, fighting mobs, exploring biomes",
        "ages": ["young", "middle"],
    },
    "game_review": {
        "desc": "a review of a video game — what's fun, what's hard, would you recommend it",
        "ages": ["middle", "teen"],
    },
    "game_guide": {
        "desc": "a how-to guide for a game — tips, tricks, strategies, walkthroughs",
        "ages": ["middle", "teen"],
    },
    "roblox_fortnite": {
        "desc": "a story about playing Roblox or Fortnite with friends",
        "ages": ["young", "middle"],
    },

    # --- School ---
    "book_report": {
        "desc": "a book report — summarize a book you read and give your opinion",
        "ages": ["young", "middle", "teen"],
    },
    "science_report": {
        "desc": "a science report — explain something you learned about in science class",
        "ages": ["middle", "teen"],
    },
    "history_essay": {
        "desc": "a history essay about an event, person, or time period you studied",
        "ages": ["middle", "teen"],
    },
    "school_daily_life": {
        "desc": "a story about a day at school — classes, lunch, recess, friends, teachers",
        "ages": ["young", "middle"],
    },

    # --- Personal ---
    "diary_journal": {
        "desc": "a diary or journal entry about your day, feelings, or something that happened",
        "ages": ["young", "middle", "teen"],
    },
    "personal_narrative": {
        "desc": "a personal story about something real that happened — use specific names and places",
        "ages": ["young", "middle", "teen"],
    },
    "letter_email": {
        "desc": "a letter or email to a friend or family member about something exciting",
        "ages": ["young", "middle", "teen"],
    },
    "texting_social": {
        "desc": "a social media post or text message conversation — very informal, casual",
        "ages": ["middle", "teen"],
    },

    # --- Non-fiction ---
    "how_to_guide": {
        "desc": "instructions or a how-to guide — how to make something, do something, build something",
        "ages": ["young", "middle", "teen"],
    },
    "animals_nature": {
        "desc": "a report or story about animals, pets, nature, or exploring outside",
        "ages": ["young", "middle"],
    },
    "sports_recap": {
        "desc": "a sports recap — describe a game you played or watched, the score, big plays",
        "ages": ["young", "middle", "teen"],
    },
    "nonfiction_report": {
        "desc": "a non-fiction report about something interesting — dinosaurs, space, volcanoes, etc.",
        "ages": ["young", "middle", "teen"],
    },

    # --- Opinion / Argument ---
    "opinion_essay": {
        "desc": "a persuasive essay or opinion piece about something you feel strongly about",
        "ages": ["middle", "teen"],
    },
    "debate_argument": {
        "desc": "an argument for one side of a debate — school uniforms, homework, screen time, etc.",
        "ages": ["teen"],
    },

    # --- Creative ---
    "poetry": {
        "desc": "a poem — any style (rhyming, free verse, haiku collection, etc.)",
        "ages": ["young", "middle", "teen"],
    },
    "family_friends": {
        "desc": "a story about family or friends — a birthday, holiday, sleepover, trip",
        "ages": ["young", "middle"],
    },
    "youtube_desc": {
        "desc": "a YouTube video description or script — describing what happens in a video",
        "ages": ["middle", "teen"],
    },
}

# Genre weights per age band
GENRE_WEIGHTS = {
    "young": {
        "adventure_action": 15, "fantasy_scifi": 8, "minecraft_story": 12,
        "roblox_fortnite": 8, "school_daily_life": 12, "book_report": 5,
        "diary_journal": 8, "personal_narrative": 8, "letter_email": 5,
        "how_to_guide": 3, "animals_nature": 8, "sports_recap": 5,
        "nonfiction_report": 3, "poetry": 2, "family_friends": 8,
    },
    "middle": {
        "adventure_action": 8, "fantasy_scifi": 8, "fan_fiction": 8,
        "horror_mystery": 5, "dnd_recap": 10, "minecraft_story": 5,
        "game_review": 5, "game_guide": 3, "roblox_fortnite": 3,
        "book_report": 5, "science_report": 3, "school_daily_life": 5,
        "diary_journal": 5, "personal_narrative": 5, "letter_email": 3,
        "texting_social": 3, "how_to_guide": 3, "animals_nature": 3,
        "sports_recap": 5, "nonfiction_report": 3, "opinion_essay": 2,
        "family_friends": 3, "youtube_desc": 2,
    },
    "teen": {
        "adventure_action": 5, "fantasy_scifi": 5, "fan_fiction": 8,
        "horror_mystery": 5, "dnd_recap": 10, "game_review": 5,
        "game_guide": 3, "book_report": 5, "science_report": 5,
        "history_essay": 5, "diary_journal": 8, "personal_narrative": 5,
        "letter_email": 3, "texting_social": 5, "how_to_guide": 3,
        "sports_recap": 3, "nonfiction_report": 3, "opinion_essay": 5,
        "debate_argument": 3, "poetry": 2, "youtube_desc": 3,
    },
}

# Word count ranges per age band
WORD_RANGES = {
    "young": (60, 140),
    "middle": (90, 200),
    "teen": (140, 280),
}

# ---------------------------------------------------------------------------
# Style variation prompts
# ---------------------------------------------------------------------------

# These inject realistic imperfections into the generated text
STYLE_VARIANTS = {
    "young": [
        "Write with some run-on sentences connected with 'and' and 'then'.",
        "Use short, choppy sentences. A young child's writing style.",
        "Include lots of dialogue with simple he said/she said tags.",
        "Write with excitement — use exclamation marks! A lot of them!",
        "Write simply and directly. Don't use any fancy vocabulary.",
    ],
    "middle": [
        "Write naturally — mix short and longer sentences.",
        "Include some dialogue and description.",
        "Write as a real student would for a school assignment — not too polished.",
        "Use some slang or casual language mixed with more formal writing.",
        "Start some sentences with 'And' or 'But' like kids actually do.",
    ],
    "teen": [
        "Write with personality and voice — this is a teenager's writing.",
        "Use some informal language, contractions, and casual phrasing.",
        "Include personal opinions and reflective moments.",
        "Mix formal and informal style like a real teen assignment.",
        "Write with some sarcasm or humor where appropriate.",
    ],
}

# Proper noun banks for more realistic writing
PROPER_NOUNS = {
    "character_names": [
        "Jake", "Emma", "Liam", "Sophia", "Noah", "Olivia", "Aiden", "Mia",
        "Lucas", "Isabella", "Mason", "Ava", "Ethan", "Harper", "Logan",
        "Ella", "James", "Aria", "Alex", "Lily", "Ryan", "Zoe", "Tyler",
        "Chloe", "Dylan", "Nora", "Marcus", "Quinn", "Kai", "Luna",
        "Jayden", "Riley", "Devon", "Skyler", "Jordan", "Taylor", "Sam",
        "Max", "Ben", "Charlie", "Leo", "Theo", "Finn", "Oscar",
    ],
    "place_names": [
        "Oakville", "Riverside", "Cedar Park", "Mountain View", "Pine Valley",
        "Springfield", "Lakewood", "Greenfield", "Sunset Hills", "Crystal Lake",
        "Willow Creek", "Eagle Point", "Silver Springs", "Maple Grove",
        "the park", "the mall", "the library", "the gym", "the pool",
    ],
    "game_names": [
        "Minecraft", "Fortnite", "Roblox", "Pokemon", "Mario Kart",
        "Among Us", "Zelda", "Animal Crossing", "Super Smash Bros",
        "Terraria", "Stardew Valley", "Splatoon", "Kirby", "Sonic",
    ],
    "dnd_terms": [
        "dungeon master", "hit points", "armor class", "saving throw",
        "critical hit", "natural twenty", "initiative", "spell slot",
        "perception check", "charisma", "dexterity", "constitution",
        "wizard", "paladin", "rogue", "bard", "cleric", "ranger",
        "dragonborn", "tiefling", "halfling", "dwarf", "goblin",
        "dragon", "ogre", "lich", "beholder", "owlbear",
    ],
}

# ---------------------------------------------------------------------------
# System prompts per age band
# ---------------------------------------------------------------------------

SYSTEM_PROMPTS = {
    "young": """You are a creative writing simulator that produces text exactly as a real 7-9 year old child would write it. Write naturally and simply — short sentences, basic vocabulary, lots of action words. Use 'and then' connectors. Include dialogue with simple tags. Don't be too polished or literary. Write like a real kid, not a professional author writing for kids.

CRITICAL: Output ONLY the story/text. No title, no meta-commentary, no "Here's a story about..." — just the actual writing.""",

    "middle": """You are a creative writing simulator that produces text exactly as a real 10-12 year old would write it. More variety in sentences but still straightforward. Can describe things with some detail but isn't overly literary. Occasionally starts sentences with 'And' or 'But'. Has a personal voice. Writes like a real 11-year-old doing a school assignment.

CRITICAL: Output ONLY the text. No title, no meta-commentary — just the actual writing.""",

    "teen": """You are a creative writing simulator that produces text exactly as a real 13-17 year old teenager would write it. More complex vocabulary and sentence structure. Has strong opinions and personal voice. Uses some informal language, contractions, and casual phrasing. Can be reflective or analytical. Writes like a real teenager.

CRITICAL: Output ONLY the text. No title, no meta-commentary — just the actual writing.""",
}

# ---------------------------------------------------------------------------
# LLM API interface (reused from generate_stories_local.py)
# ---------------------------------------------------------------------------

_BACKEND = None


def _detect_backend():
    """Detect whether server is Ollama or OpenAI-compatible."""
    global _BACKEND
    if _BACKEND:
        return _BACKEND
    try:
        resp = requests.get(f"{API_URL}/api/tags", timeout=5)
        if resp.status_code == 200 and "models" in resp.json():
            _BACKEND = "ollama"
            print(f"  Backend: Ollama (native API)")
            return _BACKEND
    except Exception:
        pass
    _BACKEND = "openai"
    print(f"  Backend: OpenAI-compatible")
    return _BACKEND


def llm_generate(prompt: str, system: str, temperature: float = 0.9,
                 max_tokens: int = 4096) -> Optional[str]:
    """Generate text via LLM API."""
    backend = _detect_backend()

    try:
        if backend == "ollama":
            full_prompt = f"{system}\n\n{prompt}"
            resp = requests.post(
                f"{API_URL}/api/generate",
                json={
                    "model": API_MODEL,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    },
                },
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            text = data.get("response", "").strip()
            if not text and data.get("thinking"):
                text = _extract_from_thinking(data["thinking"])
        else:
            resp = requests.post(
                f"{API_URL}/v1/chat/completions",
                json={
                    "model": API_MODEL,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                timeout=300,
            )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"].get("content", "").strip()
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

        # Quality filter
        if text and _is_meta(text):
            return None

        return text if text else None
    except Exception as e:
        print(f"  LLM API error: {e}")
        return None


def _extract_from_thinking(thinking: str) -> str:
    """Extract actual content from reasoning model's thinking output."""
    lines = thinking.strip().split("\n")
    meta_prefixes = (
        "we need", "use ", "include", "must ", "will ", "word count",
        "let's ", "let me", "i need", "i'll ", "now ", "count:",
        "draft", "plan", "outline", "target", "check", "ok ",
        "total", "final", "revision", "edit", "hmm", "wait",
        "---", "===", "***",
    )
    blocks = []
    current = []
    for line in lines:
        s = line.strip()
        if not s:
            if current:
                blocks.append(current)
                current = []
            continue
        if any(s.lower().startswith(p) for p in meta_prefixes):
            if current:
                blocks.append(current)
                current = []
        else:
            if s.startswith('"'): s = s[1:]
            if s.endswith('"'): s = s[:-1]
            current.append(s)
    if current:
        blocks.append(current)
    if not blocks:
        return ""
    best = max(blocks, key=lambda b: sum(len(l) for l in b))
    result = "\n".join(best).strip()
    return result if len(result.split()) >= 30 else ""


def _is_meta(text: str) -> bool:
    """Detect model meta-commentary."""
    lower = text.strip().lower()
    meta_starts = [
        "we'll write", "we will write", "let's write", "let me write",
        "i'll write", "i will write", "here's a", "here is a",
        "this story", "the story", "i need to", "i should",
        "word count", "draft:", "outline:", "structure:",
        "write as", "write a ", "sure!", "of course",
    ]
    return any(lower.startswith(p) for p in meta_starts) or len(text.split()) < 30


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_prompt(genre: str, age_band: str) -> str:
    """Build a generation prompt for the given genre and age band."""
    genre_info = GENRES[genre]
    word_min, word_max = WORD_RANGES[age_band]
    style = random.choice(STYLE_VARIANTS[age_band])

    # Pick some proper nouns to include
    names = random.sample(PROPER_NOUNS["character_names"], random.randint(1, 3))
    place = random.choice(PROPER_NOUNS["place_names"])

    # Genre-specific additions
    extras = ""
    if genre in ("dnd_recap",):
        dnd_terms = random.sample(PROPER_NOUNS["dnd_terms"], random.randint(3, 6))
        extras = f"\nInclude these D&D terms naturally: {', '.join(dnd_terms)}"
    elif genre in ("minecraft_story", "roblox_fortnite"):
        game = random.choice(PROPER_NOUNS["game_names"][:5])
        extras = f"\nThis is about playing {game}."
    elif genre == "game_review":
        game = random.choice(PROPER_NOUNS["game_names"])
        extras = f"\nReview the game: {game}"

    prompt = f"""Write {genre_info['desc']}.

Requirements:
- Write {word_min}-{word_max} words
- {style}
- Use these character names: {', '.join(names)}
- Set it in or around: {place}
{extras}

Write ONLY the text — no title, no commentary. Start directly with the writing."""

    return prompt


def build_identity_prompt(age_band: str) -> str:
    """Build a prompt for generating clean text (for identity/negative examples)."""
    word_min, word_max = WORD_RANGES[age_band]

    topics = [
        "a normal day", "your favorite hobby", "something you learned recently",
        "a trip you took", "your pet or a pet you wish you had",
        "what you did last weekend", "your favorite food and how to make it",
        "a time you helped someone", "your favorite season and why",
        "something funny that happened", "a sport or game you enjoy",
        "a place you want to visit", "your best friend",
    ]

    prompt = f"""Write a short paragraph ({word_min}-{word_max} words) about {random.choice(topics)}.

Write it as a {{'young': '8-year-old', 'middle': '11-year-old', 'teen': '15-year-old'}}['{age_band}'] would write it. Natural, not too polished.

Write ONLY the text — no title, no commentary."""

    return prompt


# ---------------------------------------------------------------------------
# Generation loop
# ---------------------------------------------------------------------------

def select_genre(age_band: str) -> str:
    """Randomly select a genre weighted by age band preferences."""
    weights = GENRE_WEIGHTS[age_band]
    genres = list(weights.keys())
    probs = [weights[g] for g in genres]
    return random.choices(genres, weights=probs, k=1)[0]


def generate_texts(
    age_band: str,
    num_texts: int = 3000,
    num_identity: int = 1000,
    batch_size: int = 50,
    output_dir: Optional[Path] = None,
    resume_from: int = 0,
) -> None:
    """Generate diverse texts for training data."""
    if output_dir is None:
        output_dir = _OUTPUT_DIR / age_band
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    system_prompt = SYSTEM_PROMPTS[age_band]

    print(f"=== Generating diverse {age_band} band texts ===")
    print(f"Model: {API_MODEL} @ {API_URL}")
    print(f"Texts: {num_texts}, Identity: {num_identity}")
    print(f"Output: {output_dir}")
    print()

    # Track genre distribution
    genre_counts = {}
    texts_generated = 0
    identity_generated = 0
    failed = 0
    batch = []
    batch_num = resume_from // batch_size

    total_needed = num_texts + num_identity
    i = resume_from

    while i < total_needed:
        # Decide whether to generate a genre text or identity text
        if texts_generated < num_texts:
            genre = select_genre(age_band)
            prompt = build_prompt(genre, age_band)
            is_identity = False
        elif identity_generated < num_identity:
            prompt = build_identity_prompt(age_band)
            genre = "identity"
            is_identity = True
        else:
            break

        text = llm_generate(prompt, system_prompt)

        if text is None:
            failed += 1
            if failed % 10 == 0:
                print(f"  {failed} failed generations so far")
            if failed > total_needed * 0.3:
                print("  Too many failures, stopping")
                break
            continue

        # Track
        genre_counts[genre] = genre_counts.get(genre, 0) + 1
        if is_identity:
            identity_generated += 1
        else:
            texts_generated += 1

        record = {
            "text": text,
            "genre": genre,
            "age_band": age_band,
            "is_identity": is_identity,
            "word_count": len(text.split()),
        }
        batch.append(record)
        i += 1

        # Progress
        if i % 10 == 0:
            pct = i / total_needed * 100
            print(f"  [{i}/{total_needed}] ({pct:.1f}%) - "
                  f"texts: {texts_generated}, identity: {identity_generated}, "
                  f"failed: {failed}")

        # Save batch
        if len(batch) >= batch_size:
            batch_file = output_dir / f"batch_{batch_num:04d}.jsonl"
            with open(batch_file, 'w') as f:
                for rec in batch:
                    f.write(json.dumps(rec) + '\n')
            print(f"  Saved {batch_file.name} ({len(batch)} texts)")
            batch = []
            batch_num += 1

    # Save remaining
    if batch:
        batch_file = output_dir / f"batch_{batch_num:04d}.jsonl"
        with open(batch_file, 'w') as f:
            for rec in batch:
                f.write(json.dumps(rec) + '\n')
        print(f"  Saved {batch_file.name} ({len(batch)} texts)")

    # Summary
    print(f"\n=== Generation complete ===")
    print(f"  Texts: {texts_generated}, Identity: {identity_generated}")
    print(f"  Failed: {failed}")
    print(f"  Genre distribution:")
    for genre, count in sorted(genre_counts.items(), key=lambda x: -x[1]):
        print(f"    {genre}: {count}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate diverse source texts for v2 training")
    parser.add_argument("--age-band", "-a", choices=["young", "middle", "teen", "all"],
                       default="all", help="Age band to generate for")
    parser.add_argument("--count", "-n", type=int, default=3000,
                       help="Number of texts per age band")
    parser.add_argument("--identity", type=int, default=1000,
                       help="Number of identity/negative examples per age band")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--resume-from", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    if args.age_band == "all":
        bands = ["young", "middle", "teen"]
    else:
        bands = [args.age_band]

    for band in bands:
        out = Path(args.output_dir) / band if args.output_dir else None
        generate_texts(
            age_band=band,
            num_texts=args.count,
            num_identity=args.identity,
            batch_size=args.batch_size,
            output_dir=out,
            resume_from=args.resume_from,
        )


if __name__ == "__main__":
    main()
