#!/usr/bin/env python3
"""
Generate clean, age-appropriate stories via local LLM for error injection.

Replaces generate_clean_stories.py (Gemini API) with local inference via any
OpenAI-compatible API (vLLM, Ollama, llama.cpp server).
Supports three age bands: young (7-9), middle (10-12), teen (13-17).

No rate limiting needed — local server, unlimited generation.
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
_VOCAB_PATH = _DATA_DIR / "vocab_targets.json"
_OUTPUT_DIR = _DATA_DIR / "generated_stories"

API_URL = os.environ.get("LLM_API_URL", "http://platogpu002:8000")
API_MODEL = os.environ.get("LLM_MODEL", "Valdemardi/DeepSeek-R1-Distill-Llama-70B-AWQ")

# Age-band specific genre weights
GENRE_WEIGHTS = {
    "young": {
        "adventure_action": 0.30,
        "school_daily_life": 0.20,
        "animals_nature": 0.20,
        "sports_games": 0.15,
        "family_friends": 0.10,
        "fantasy_scifi": 0.05,
    },
    "middle": {
        "adventure_action": 0.20,
        "school_daily_life": 0.25,
        "animals_nature": 0.10,
        "sports_games": 0.15,
        "family_friends": 0.15,
        "fantasy_scifi": 0.15,
    },
    "teen": {
        "adventure_action": 0.10,
        "school_daily_life": 0.15,
        "essay_journal": 0.25,
        "sports_games": 0.10,
        "family_friends": 0.15,
        "fantasy_scifi": 0.10,
        "opinion_argument": 0.15,
    },
}

GENRE_DESCRIPTIONS = {
    "adventure_action": "an adventure story where characters explore, face danger, or go on a quest",
    "school_daily_life": "a story about school, homework, recess, lunchtime, or a regular day",
    "animals_nature": "a story about animals, pets, nature, or exploring outside",
    "sports_games": "a story about playing sports, video games, or competing in something",
    "family_friends": "a story about family, friends, a birthday, holiday, or sleepover",
    "fantasy_scifi": "a fantasy or science fiction story with magic, aliens, robots, or superpowers",
    "essay_journal": "a journal entry, personal essay, or reflection on a topic that matters to them",
    "opinion_argument": "a persuasive piece or opinion essay about something they feel strongly about",
}

SYSTEM_PROMPTS = {
    "young": """You are a creative writing assistant that writes short stories
at a young child's level. Write naturally and simply. Do not use sophisticated
vocabulary or complex sentence structures. Write as a real 8-year-old child
would write — simple plots, direct action, lots of dialogue, and not too polished.""",

    "middle": """You are a creative writing assistant that writes at an 11-year-old's
level. Write naturally with some variety in sentence structure. Include some
descriptive language but keep it age-appropriate. Write as a real 11-year-old
doing a school assignment would — organized paragraphs, some detail, but still
straightforward.""",

    "teen": """You are a creative writing assistant that writes at a 14-year-old's
level. Write with more complex sentence structures and vocabulary. Include
personal reflection, opinions, or analysis. Write as a real teenager would —
more sophisticated than a child but not adult-level. Some informal language
and personal voice is fine.""",
}

# Age-band vocab category preferences
VOCAB_CATEGORIES_BY_AGE = {
    "young": [
        "young_cvc_short_vowel", "young_consonant_clusters", "young_sight_words",
        "vowel_digraph_ea", "vowel_digraph_ee", "vowel_digraph_ai",
        "consonant_cluster_ng", "consonant_cluster_mp", "consonant_cluster_initial",
        "irregular_past_tense", "regular_past_tense_ed", "silent_letters",
    ],
    "middle": [
        "middle_double_consonants", "middle_silent_letters", "middle_academic_words",
        "middle_homophones", "vowel_digraph_ea", "vowel_digraph_ei",
        "vowel_digraph_ou", "multi_syllable_complex", "consonant_cluster_sc",
        "irregular_past_tense", "regular_past_tense_ed", "silent_letters",
    ],
    "teen": [
        "teen_latin_greek_roots", "teen_complex_morphology", "teen_subject_vocabulary",
        "teen_proper_nouns", "teen_word_boundary", "middle_academic_words",
        "middle_homophones", "multi_syllable_complex", "vowel_digraph_ei",
        "silent_letters", "tricky_correct_words",
    ],
}

# ---------------------------------------------------------------------------
# Load vocab
# ---------------------------------------------------------------------------

def load_vocab_targets() -> dict:
    """Load vocabulary target lists."""
    with open(_VOCAB_PATH) as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if not k.startswith("_")}


def select_vocab_targets(vocab: dict, age_band: str, count: int = 12) -> list[str]:
    """Select a diverse set of vocabulary targets for the given age band."""
    preferred = VOCAB_CATEGORIES_BY_AGE.get(age_band, list(vocab.keys()))
    # Filter to categories that exist in vocab
    categories = [c for c in preferred if c in vocab]
    if not categories:
        categories = [k for k in vocab.keys() if not k.startswith("_")]
    random.shuffle(categories)

    selected = []
    for cat in categories:
        if len(selected) >= count:
            break
        words = vocab[cat]
        n = min(2, count - len(selected))
        selected.extend(random.sample(words, min(n, len(words))))

    random.shuffle(selected)
    return selected


def choose_genre(age_band: str) -> str:
    """Weighted random genre selection for the given age band."""
    weights = GENRE_WEIGHTS.get(age_band, GENRE_WEIGHTS["young"])
    genres = list(weights.keys())
    probs = list(weights.values())
    return random.choices(genres, weights=probs, k=1)[0]


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

AGE_SPECS = {
    "young": {"word_range": "80-120", "grade": "Grade 2-3", "age_desc": "8-year-old", "dialogue": 2},
    "middle": {"word_range": "100-180", "grade": "Grade 4-6", "age_desc": "11-year-old", "dialogue": 1},
    "teen": {"word_range": "150-250", "grade": "Grade 7-9", "age_desc": "14-year-old", "dialogue": 0},
}


def build_story_prompt(genre: str, vocab_words: list[str], age_band: str) -> str:
    """Build a prompt for the LLM to generate one clean story."""
    genre_desc = GENRE_DESCRIPTIONS[genre]
    vocab_str = ", ".join(vocab_words)
    spec = AGE_SPECS[age_band]

    dialogue_line = ""
    if spec["dialogue"] > 0:
        dialogue_line = f"\n- Include at least {spec['dialogue']} lines of dialogue"

    return f"""Write a short story a child aged {spec['age_desc']} might write. It should be {genre_desc}.

The story should:
- Be {spec['word_range']} words long
- Be written at {spec['grade']} reading level
- Use natural, age-appropriate language{dialogue_line}
- Include some of these words naturally: {vocab_str}

Write ONLY the story. No title. Write it as a {spec['age_desc']} would structure it."""


NEGATIVE_PROMPTS = {
    "young": {
        "tricky_words": """Write a short paragraph (60-100 words) an 8-year-old might write
that includes words that look unusual but are spelled correctly: knight, island, caught,
through, enough, thought, daughter, straight. Use at least 4 of these words naturally.
Write at Grade 2-3 level. Write ONLY the paragraph.""",

        "informal_correct": """Write a short paragraph (60-100 words) of informal but correct
children's writing. Include contractions (don't, can't, it's, we're), exclamations
(Wow!, No way!, Oh man!), and sentence fragments that sound natural. Everything should
be spelled correctly. Write ONLY the paragraph.""",

        "proper_nouns": """Write a short paragraph (60-100 words) a child might write about
playing video games or watching shows. Include proper nouns like Minecraft, Pokemon,
Roblox, Fortnite, Spider-Man. All spelling should be correct. Write at Grade 2-3 level.
Write ONLY the paragraph.""",

        "simple_correct": """Write a very simple paragraph (60-100 words) using only common,
easy-to-spell words. A child aged 8 writing about their day. Short sentences. Simple words.
Everything spelled correctly. Write ONLY the paragraph.""",
    },
    "middle": {
        "academic_correct": """Write a short paragraph (80-150 words) an 11-year-old might write
for a school report. Include academic words like experiment, temperature, important,
information, explanation, measurement. All spelling should be correct.
Write ONLY the paragraph.""",

        "homophone_correct": """Write a short paragraph (80-150 words) that correctly uses
homophones: their/there/they're, to/too/two, your/you're, its/it's, hear/here.
Write as an 11-year-old would. All spelling and usage should be correct.
Write ONLY the paragraph.""",

        "silent_letter_correct": """Write a short paragraph (80-150 words) that uses words with
silent letters: knight, island, knowledge, answer, sword, autumn, whistle, listen, honest.
All spelling should be correct. Write at Grade 4-6 level. Write ONLY the paragraph.""",

        "informal_correct": """Write a short paragraph (80-150 words) of natural 11-year-old
writing. Include contractions, some longer sentences, and everyday vocabulary.
Everything should be spelled correctly. Write ONLY the paragraph.""",
    },
    "teen": {
        "complex_correct": """Write a short paragraph (120-200 words) a 14-year-old might write
using complex vocabulary: government, environment, psychology, unnecessary, definitely,
accommodation, embarrassment, occasionally. All spelling should be correct.
Write ONLY the paragraph.""",

        "essay_correct": """Write a short persuasive paragraph (120-200 words) a 14-year-old
might write for English class. Use formal language with transition words and some
sophisticated vocabulary. Everything should be spelled correctly. Write ONLY the paragraph.""",

        "science_correct": """Write a short paragraph (120-200 words) from a 14-year-old's
science report. Include terms like photosynthesis, hypothesis, experiment, analysis,
observation, conclusion. All spelling should be correct. Write ONLY the paragraph.""",

        "journal_correct": """Write a short journal entry (120-200 words) a teenager might write.
Include reflective language, some informal expressions, and personal opinions.
Everything should be spelled correctly. Write ONLY the paragraph.""",
    },
}


def build_negative_prompt(age_band: str) -> tuple[str, str]:
    """Build a prompt for negative examples. Returns (variant, prompt)."""
    variants = NEGATIVE_PROMPTS.get(age_band, NEGATIVE_PROMPTS["young"])
    variant = random.choice(list(variants.keys()))
    return variant, variants[variant]


# ---------------------------------------------------------------------------
# LLM API — auto-detects Ollama (native) vs OpenAI-compatible (vLLM etc.)
# ---------------------------------------------------------------------------

# Budget for reasoning models: <think> block (~1000-2000 tok) + content (~100-400 tok)
MAX_TOKENS_BY_AGE = {"young": 4096, "middle": 4096, "teen": 4096}
_BACKEND = None  # "ollama" or "openai", auto-detected on first call


def _detect_backend():
    """Detect whether server is Ollama (use native API) or vLLM/other (use OpenAI API)."""
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
                 age_band: str = "young") -> Optional[str]:
    """Generate text via LLM API. Uses Ollama native API or OpenAI-compatible."""
    backend = _detect_backend()
    max_tokens = MAX_TOKENS_BY_AGE.get(age_band, 512)

    try:
        if backend == "ollama":
            # Ollama native /api/generate — avoids reasoning token overhead
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
            # Reasoning models may put content in 'thinking' field
            if not text and data.get("thinking"):
                text = _extract_story_from_thinking(data["thinking"])
        else:
            # OpenAI-compatible (vLLM, llama.cpp server, etc.)
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
            # DeepSeek-R1 models emit <think>...</think> reasoning blocks — strip them
            text = _strip_think_tags(text)

        # Quality filter: reject meta-commentary or very short responses
        if text and _is_meta_commentary(text):
            return None

        return text if text else None
    except Exception as e:
        print(f"  LLM API error: {e}")
        return None


def _strip_think_tags(text: str) -> str:
    """Strip <think>...</think> reasoning blocks from DeepSeek-R1 style output."""
    stripped = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Also handle unclosed <think> (model hit max_tokens mid-reasoning)
    if "<think>" in stripped:
        stripped = stripped.split("</think>")[-1].strip()
        if stripped.startswith("<think>"):
            stripped = ""
    return stripped


def _is_meta_commentary(text: str) -> bool:
    """Detect model meta-commentary that isn't an actual story."""
    lower = text.strip().lower()
    meta_starts = [
        "we'll write", "we will write", "let's write", "let me write",
        "i'll write", "i will write", "here's a", "here is a",
        "this story", "the story", "i need to", "i should",
        "word count", "draft:", "outline:", "structure:",
        "also \"", "also '", "must use", "must include",
        "write as", "write a ",
    ]
    if any(lower.startswith(p) for p in meta_starts):
        return True
    # Reject very short responses (likely truncated or meta)
    if len(text.split()) < 40:
        return True
    return False


def _extract_story_from_thinking(thinking: str) -> str:
    """Extract the actual story from a reasoning model's thinking output.

    The model deliberates first (word count, plan, etc.) then drafts the story.
    We want the last substantial narrative block.
    """
    lines = thinking.strip().split("\n")

    # Strategy: find the last block of 3+ consecutive narrative lines
    # (not starting with meta-commentary patterns)
    meta_prefixes = (
        "we need", "use ", "include", "must ", "will ", "word count",
        "let's ", "let me", "i need", "i'll ", "now ", "count:",
        "draft", "plan", "outline", "target", "check", "ok ",
        "total", "final", "revision", "edit", "hmm", "wait",
        "---", "===", "***",
    )

    blocks = []
    current_block = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current_block:
                blocks.append(current_block)
                current_block = []
            continue
        lower = stripped.lower()
        is_meta = any(lower.startswith(p) for p in meta_prefixes)
        if is_meta:
            if current_block:
                blocks.append(current_block)
                current_block = []
        else:
            # Clean up quote wrapping
            if stripped.startswith('"'):
                stripped = stripped[1:]
            if stripped.endswith('"'):
                stripped = stripped[:-1]
            current_block.append(stripped)
    if current_block:
        blocks.append(current_block)

    # Find the longest block (most likely the actual story)
    if not blocks:
        return ""
    best = max(blocks, key=lambda b: sum(len(l) for l in b))
    result = "\n".join(best).strip()
    return result if len(result.split()) >= 30 else ""


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_stories(
    age_band: str,
    num_stories: int = 5000,
    num_negative: int = 2000,
    batch_size: int = 50,
    output_dir: Optional[Path] = None,
    resume_from: int = 0,
) -> None:
    """
    Generate clean stories and negative examples via local Ollama.

    Args:
        age_band: One of "young", "middle", "teen".
        num_stories: Number of clean stories for error injection.
        num_negative: Number of negative/identity examples.
        batch_size: Stories per output file.
        output_dir: Where to save outputs.
        resume_from: Story index to resume from (for interrupted runs).
    """
    if output_dir is None:
        output_dir = _OUTPUT_DIR / age_band
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vocab = load_vocab_targets()
    system_prompt = SYSTEM_PROMPTS[age_band]

    age_ranges = {"young": "7-9", "middle": "10-12", "teen": "13-17"}
    print(f"=== Generating {age_band} band (ages {age_ranges[age_band]}) ===")
    print(f"Model: {API_MODEL} @ {API_URL}")
    print(f"Stories: {num_stories}, Negatives: {num_negative}")
    print(f"Output: {output_dir}")
    print()

    # Verify server is reachable
    try:
        resp = requests.get(f"{API_URL}/v1/models", timeout=10)
        models = [m["id"] for m in resp.json().get("data", [])]
        print(f"Available models: {models}")
        if API_MODEL not in models and models:
            print(f"WARNING: {API_MODEL} not in model list. Using first available: {models[0]}")
    except Exception as e:
        print(f"WARNING: Could not connect to {API_URL}: {e}")
        print("Continuing anyway (server may come up)...")

    # --- Generate clean stories ---
    stories = []
    batch_num = resume_from // batch_size
    failed = 0
    max_failures = 100
    start_time = time.time()

    for i in range(resume_from, num_stories):
        genre = choose_genre(age_band)
        vocab_words = select_vocab_targets(vocab, age_band, count=random.randint(8, 15))
        prompt = build_story_prompt(genre, vocab_words, age_band)

        text = llm_generate(prompt, system_prompt, temperature=0.9, age_band=age_band)

        if not text:
            failed += 1
            print(f"  [{i}] Empty response, skipping (failures: {failed})")
            if failed > max_failures:
                print(f"Too many failures ({failed}). Stopping.")
                break
            continue

        stories.append({
            "id": f"{age_band}_story_{i:05d}",
            "text": text,
            "genre": genre,
            "age_band": age_band,
            "vocab_targets": vocab_words,
            "type": "clean_for_injection",
        })

        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1 - resume_from) / elapsed * 3600
            print(f"  [{i + 1}/{num_stories}] {genre} | {rate:.0f} stories/hr | {text[:60]}...")

        # Save batch
        if len(stories) >= batch_size:
            batch_file = output_dir / f"stories_batch_{batch_num:04d}.json"
            with open(batch_file, "w") as f:
                json.dump(stories, f, indent=2)
            print(f"  Saved batch {batch_num} ({len(stories)} stories) -> {batch_file.name}")
            stories = []
            batch_num += 1

    # Save remaining
    if stories:
        batch_file = output_dir / f"stories_batch_{batch_num:04d}.json"
        with open(batch_file, "w") as f:
            json.dump(stories, f, indent=2)
        print(f"  Saved final batch ({len(stories)} stories)")

    # --- Generate negative examples ---
    print(f"\nGenerating {num_negative} negative examples for {age_band}...")
    negatives = []
    neg_batch = 0

    for i in range(num_negative):
        variant, prompt = build_negative_prompt(age_band)

        text = llm_generate(prompt, system_prompt, temperature=0.8, age_band=age_band)

        if not text:
            continue

        negatives.append({
            "id": f"{age_band}_negative_{i:05d}",
            "text": text,
            "variant": variant,
            "age_band": age_band,
            "type": "negative_identity",
        })

        if (i + 1) % 10 == 0:
            print(f"  [{i + 1}/{num_negative}] Negative ({variant}): {text[:60]}...")

        if len(negatives) >= batch_size:
            batch_file = output_dir / f"negatives_batch_{neg_batch:04d}.json"
            with open(batch_file, "w") as f:
                json.dump(negatives, f, indent=2)
            print(f"  Saved negative batch {neg_batch} ({len(negatives)} examples)")
            negatives = []
            neg_batch += 1

    if negatives:
        batch_file = output_dir / f"negatives_batch_{neg_batch:04d}.json"
        with open(batch_file, "w") as f:
            json.dump(negatives, f, indent=2)
        print(f"  Saved final negative batch ({len(negatives)} examples)")

    elapsed = time.time() - start_time
    print(f"\nDone {age_band}! Elapsed: {elapsed/3600:.1f}h | Failed: {failed}")


def generate_pilot(age_band: str, count: int = 10) -> None:
    """Generate a small pilot batch for manual review."""
    print(f"=== PILOT MODE: {count} stories, age band: {age_band} ===\n")
    generate_stories(
        age_band=age_band,
        num_stories=count,
        num_negative=max(3, count // 3),
        batch_size=count,
        output_dir=_OUTPUT_DIR / "pilot" / age_band,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate clean stories via local LLM (vLLM/Ollama) for error injection"
    )
    parser.add_argument(
        "--age-band", choices=["young", "middle", "teen", "all"], default="all",
        help="Age band to generate for (default: all)"
    )
    parser.add_argument(
        "--pilot", action="store_true",
        help="Generate a small pilot batch (10 stories per age band)"
    )
    parser.add_argument(
        "--stories", type=int, default=5000,
        help="Clean stories per age band (default: 5000)"
    )
    parser.add_argument(
        "--negative", type=int, default=2000,
        help="Negative examples per age band (default: 2000)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=50,
        help="Stories per output file (default: 50)"
    )
    parser.add_argument(
        "--resume-from", type=int, default=0,
        help="Story index to resume from (for interrupted runs)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: training/generated_stories/<age_band>)"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Model name for API (default: from LLM_MODEL env or gpt-oss-120b)"
    )
    parser.add_argument(
        "--url", type=str, default=None,
        help="OpenAI-compatible API URL (default: from LLM_API_URL env or http://localhost:8000)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
    _apply_overrides(args.model, args.url)

    bands = ["young", "middle", "teen"] if args.age_band == "all" else [args.age_band]

    for band in bands:
        if args.pilot:
            generate_pilot(band)
        else:
            out = Path(args.output_dir) / band if args.output_dir else None
            generate_stories(
                age_band=band,
                num_stories=args.stories,
                num_negative=args.negative,
                batch_size=args.batch_size,
                output_dir=out,
                resume_from=args.resume_from,
            )


def _apply_overrides(model: Optional[str], url: Optional[str]):
    global API_MODEL, API_URL
    if model:
        API_MODEL = model
    if url:
        API_URL = url


if __name__ == "__main__":
    main()
