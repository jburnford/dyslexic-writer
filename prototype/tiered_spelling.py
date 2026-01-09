#!/usr/bin/env python3
"""
LLM-First Spelling Correction for Dyslexic Writers
===================================================

For severe dyslexic spelling, traditional spell checkers fail because:
- "enuff" is closer to "snuff" than "enough" by edit distance
- "fone" is closer to "one" than "phone"
- "platem" won't match "platinum" at all

Solution: Use LLM with full sentence context. Cache results for speed.

Usage:
    python tiered_spelling.py
"""

import re
import time
import json
import hashlib
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

try:
    import requests
except ImportError:
    requests = None

# Configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "phi4-mini"
CACHE_FILE = Path(__file__).parent / ".spelling_cache.json"


@dataclass
class Correction:
    """A spelling correction with metadata."""
    original: str
    corrected: str
    source: str  # "llm" or "cache"


class SpellingCache:
    """Cache for LLM corrections - makes repeated misspellings instant."""

    def __init__(self, cache_file: Path = CACHE_FILE):
        self.cache_file = cache_file
        self.cache: dict[str, str] = {}
        self._load()

    def _load(self):
        if self.cache_file.exists():
            try:
                self.cache = json.loads(self.cache_file.read_text())
            except:
                self.cache = {}

    def _save(self):
        self.cache_file.write_text(json.dumps(self.cache, indent=2))

    def get(self, word: str) -> Optional[str]:
        """Get cached correction for a word."""
        return self.cache.get(word.lower())

    def set(self, word: str, correction: str):
        """Cache a correction."""
        if word.lower() != correction.lower():
            self.cache[word.lower()] = correction
            self._save()

    def clear(self):
        """Clear the cache."""
        self.cache = {}
        if self.cache_file.exists():
            self.cache_file.unlink()


# Common English words that should NEVER be "corrected"
VALID_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need", "its", "it's",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
    "us", "them", "my", "your", "his", "her", "its", "our", "their",
    "this", "that", "these", "those", "what", "which", "who", "whom",
    "and", "or", "but", "if", "because", "when", "where", "how", "why",
    "for", "to", "of", "in", "on", "at", "by", "with", "from", "as",
    "go", "going", "went", "gone", "come", "coming", "came", "see", "saw",
    "think", "know", "want", "get", "make", "take", "say", "said",
    "cool", "win", "won't", "will", "kids", "never", "always", "reading",
    "writing", "hard", "easy", "words", "spelling", "dyslexic", "platinum",
}


class LLMSpellChecker:
    """LLM-based spell checker for dyslexic phonetic spelling."""

    SYSTEM_PROMPT = """You fix spelling for a dyslexic child who spells phonetically.

RULES:
- ONLY fix spelling mistakes
- NEVER change correctly spelled words
- NEVER change word choice or meaning
- NEVER add or remove words

Output format:
CHANGES: misspelled1->correct1, misspelled2->correct2

If no errors:
CHANGES: none

Examples:
"I have enuff fud" -> CHANGES: enuff->enough, fud->food
"i will win platem" -> CHANGES: platem->platinum
"I am happy" -> CHANGES: none"""

    def __init__(self, model: str = MODEL):
        self.model = model
        self.cache = SpellingCache()

    def check_sentence(self, sentence: str) -> tuple[str, list[Correction], float]:
        """
        Check and correct a sentence.
        Returns: (corrected_sentence, list of corrections, time_taken)
        """
        start = time.time()
        corrections = []

        # Step 1: Apply cached corrections first
        words = sentence.split()
        uncached_sentence = sentence
        cached_corrections = []

        for i, word in enumerate(words):
            clean = re.sub(r'[^\w]', '', word.lower())
            cached = self.cache.get(clean)
            if cached:
                # Apply cached correction
                punct = re.sub(r'[\w]', '', word)  # preserve punctuation
                words[i] = cached + punct
                cached_corrections.append(Correction(clean, cached, "cache"))

        if cached_corrections:
            uncached_sentence = " ".join(words)
            corrections.extend(cached_corrections)
            print(f"  [CACHE] Applied {len(cached_corrections)} cached corrections")

        # Step 2: Send to LLM for remaining errors
        if not requests:
            return uncached_sentence, corrections, time.time() - start

        # Skip LLM if we already have corrections and the sentence looks clean
        # This avoids the LLM "over-correcting" already-fixed text
        if cached_corrections and len(cached_corrections) >= 2:
            print("  [SKIP] Cache handled main corrections, skipping LLM")
            return uncached_sentence, corrections, time.time() - start

        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": self.model,
                    "prompt": f'Fix spelling: "{uncached_sentence}"',
                    "system": self.SYSTEM_PROMPT,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 150,
                        "num_gpu": 20,
                    }
                },
                timeout=60
            )
            response.raise_for_status()
            result = response.json().get("response", "").strip()
            print(f"  [LLM] Response: {result}")

            # Parse LLM response
            llm_changes = self._parse_response(result)

            # Apply LLM corrections
            corrected = uncached_sentence
            for original, fixed in llm_changes:
                pattern = re.compile(re.escape(original), re.IGNORECASE)
                corrected = pattern.sub(fixed, corrected, count=1)
                corrections.append(Correction(original, fixed, "llm"))
                # Cache for next time
                self.cache.set(original, fixed)

            return corrected, corrections, time.time() - start

        except Exception as e:
            print(f"  [LLM] Error: {e}")
            return uncached_sentence, corrections, time.time() - start

    def _parse_response(self, response: str) -> list[tuple[str, str]]:
        """Parse LLM response into corrections."""
        changes = []

        # Only look at first line containing CHANGES
        for line in response.split('\n'):
            if 'CHANGES:' in line.upper():
                response = line
                break

        match = re.search(r'CHANGES:\s*(.+?)$', response, re.IGNORECASE)
        if not match:
            return changes

        changes_str = match.group(1).strip()
        if changes_str.lower() == "none":
            return changes

        for change in changes_str.split(","):
            if "->" in change:
                parts = change.strip().split("->")
                if len(parts) == 2:
                    original = parts[0].strip()
                    corrected = parts[1].strip()
                    # Strip punctuation from both sides
                    original = re.sub(r'^[^\w]+|[^\w]+$', '', original)
                    corrected = re.sub(r'^[^\w]+|[^\w]+$', '', corrected)
                    # Skip if they're the same or either is empty
                    if not original or not corrected:
                        continue
                    if original.lower() == corrected.lower():
                        continue
                    # IMPORTANT: Never "correct" valid English words
                    if original.lower() in VALID_WORDS:
                        print(f"  [SKIP] '{original}' is a valid word, ignoring correction")
                        continue
                    changes.append((original, corrected))

        return changes


def demo():
    """Demo the spell checker."""
    print("=" * 70)
    print("LLM SPELLING CORRECTION FOR DYSLEXIC WRITERS")
    print("=" * 70)

    checker = LLMSpellChecker()
    checker.cache.clear()  # Start fresh for demo

    test_sentences = [
        # Easy phonetic
        "I have enuff food.",
        "Call me on the fone.",

        # Harder cases
        "i think its cool and i think i will win platem.",
        "for dysxeic kids reading and writing is hard because never rememer the spelling of words.",

        # Full example
        "i think its cool and i think i will win platem. for dysxeic kids reading and writing is hard because never rememer the spelling of words.",

        # Test caching - run same sentence again
        "I have enuff food.",
    ]

    for sentence in test_sentences:
        print(f"\n{'─' * 70}")
        print(f"INPUT:  {sentence}")

        corrected, corrections, elapsed = checker.check_sentence(sentence)

        print(f"OUTPUT: {corrected}")
        print(f"TIME:   {elapsed:.2f}s")

        if corrections:
            print("FIXES:")
            for c in corrections:
                print(f"  • {c.original} → {c.corrected} ({c.source})")

    print(f"\n{'=' * 70}")
    print("CACHE CONTENTS:")
    print("=" * 70)
    for word, correction in checker.cache.cache.items():
        print(f"  {word} → {correction}")


if __name__ == "__main__":
    demo()
