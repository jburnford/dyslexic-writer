#!/usr/bin/env python3
"""
Error injection rule engine calibrated to the Bob Story error profile.

Takes clean text and injects dyslexic spelling errors based on empirically-derived
transformation rules from bob-story-error-profile.md.

Supports three age bands with distinct error profiles:
  - young  (7-9):  phonological-dominant (65%), higher error density (~15%)
  - middle (10-12): orthographic-dominant (43%), medium density (~10%)
  - teen   (13-17): morphological-dominant (33%), lower density (~7%)

Each rule is tagged with provenance (attested, inferred, literature-based) and
confidence level. Error density follows realistic distributions observed in the
Bob story (~12% word-level error rate for young band).
"""

import json
import random
import re
import os
from dataclasses import dataclass, field
from typing import Optional

from phonetic_misspeller import (
    generate_misspelling,
    keyboard_typo,
    visual_similarity,
    creative_phonetic,
    phoneme_misspell,
    EXPANDED_REAL_WORD_SUBS,
    EXPANDED_WORD_BOUNDARY_ERRORS,
)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ErrorAnnotation:
    """Tracks a single error injected into a word."""
    target_word: str
    written_as: str
    rule: str
    category: str        # phonological, orthographic, morphological
    provenance: str      # attested, inferred, literature-based
    confidence: str      # high, medium, low
    position: int = 0    # word index in sentence


@dataclass
class SentenceResult:
    """Result of error injection on a single sentence."""
    original: str
    corrupted: str
    errors: list = field(default_factory=list)
    error_count: int = 0
    word_error_rate: float = 0.0


# ---------------------------------------------------------------------------
# Load data files
# ---------------------------------------------------------------------------

_DATA_DIR = os.path.dirname(os.path.abspath(__file__))

def _load_json(filename: str) -> dict:
    path = os.path.join(_DATA_DIR, filename)
    with open(path) as f:
        return json.load(f)


def _build_preservation_set() -> set:
    """Build a flat set of all words that should never be corrupted."""
    data = _load_json("preservation_list.json")
    words = set()
    for key, values in data.items():
        if key.startswith("_"):
            continue
        for w in values:
            words.add(w.lower())
    return words


PRESERVATION_SET: set = set()  # initialized lazily


def _get_preservation_set() -> set:
    global PRESERVATION_SET
    if not PRESERVATION_SET:
        PRESERVATION_SET = _build_preservation_set()
    return PRESERVATION_SET


# ---------------------------------------------------------------------------
# Age-band configuration
# ---------------------------------------------------------------------------

AGE_BAND_CONFIG = {
    "young": {
        "category_weights": {"phonological": 0.65, "orthographic": 0.22, "morphological": 0.13},
        "error_density": 0.15,  # ~15% of words get errors
        "error_distribution": [(0, 0.25), (1, 0.35), (2, 0.25), (3, 0.15)],
    },
    "middle": {
        "category_weights": {"phonological": 0.30, "orthographic": 0.43, "morphological": 0.27},
        "error_density": 0.10,
        "error_distribution": [(0, 0.30), (1, 0.35), (2, 0.25), (3, 0.10)],
    },
    "teen": {
        "category_weights": {"phonological": 0.18, "orthographic": 0.37, "morphological": 0.33, "real_word": 0.12},
        "error_density": 0.07,
        "error_distribution": [(0, 0.35), (1, 0.35), (2, 0.20), (3, 0.10)],
    },
}

# Default (original) config matches young band
DEFAULT_AGE_BAND = "young"


# ---------------------------------------------------------------------------
# Transformation rules
# ---------------------------------------------------------------------------
# Each rule is a callable: (word: str) -> Optional[tuple[str, ErrorAnnotation]]
# Returns None if the rule doesn't apply or the random check fails.
# ---------------------------------------------------------------------------

def _has_pattern(word: str, pattern: str) -> bool:
    """Check if word contains a pattern (case-insensitive)."""
    return pattern in word.lower()


# --- Tier 1: Vowel digraph reduction (highest frequency, attested) ---

def rule_ea_to_e(word: str) -> Optional[tuple[str, ErrorAnnotation]]:
    """ea -> e: screamed -> skremed, dreaming -> dreming"""
    lower = word.lower()
    if "ea" not in lower:
        return None
    if random.random() > 0.70:  # 70% application rate (attested: 2/2)
        return None
    # Don't apply to 'ea' at word boundaries where it changes pronunciation
    idx = lower.index("ea")
    corrupted = word[:idx] + word[idx + 1:]  # remove the 'a' after 'e'
    if corrupted.lower() == word.lower():
        return None
    return (corrupted, ErrorAnnotation(
        target_word=word, written_as=corrupted,
        rule="vowel_digraph_ea_to_e", category="phonological",
        provenance="attested", confidence="high"
    ))


def rule_ei_to_i(word: str) -> Optional[tuple[str, ErrorAnnotation]]:
    """ei -> i: heights -> hights"""
    lower = word.lower()
    if "ei" not in lower:
        return None
    if random.random() > 0.70:  # attested: 2/2
        return None
    idx = lower.index("ei")
    corrupted = word[:idx] + word[idx + 1:]  # remove 'e', keep 'i'
    return (corrupted, ErrorAnnotation(
        target_word=word, written_as=corrupted,
        rule="vowel_digraph_ei_to_i", category="phonological",
        provenance="attested", confidence="high"
    ))


def rule_ai_to_e(word: str) -> Optional[tuple[str, ErrorAnnotation]]:
    """ai -> e: fainted -> fented"""
    lower = word.lower()
    if "ai" not in lower:
        return None
    if random.random() > 0.45:  # attested: 1/3 for 'e' variant
        return None
    idx = lower.index("ai")
    corrupted = word[:idx] + "e" + word[idx + 2:]
    return (corrupted, ErrorAnnotation(
        target_word=word, written_as=corrupted,
        rule="vowel_digraph_ai_to_e", category="phonological",
        provenance="attested", confidence="medium"
    ))


def rule_ai_to_a(word: str) -> Optional[tuple[str, ErrorAnnotation]]:
    """ai -> a: again -> agan"""
    lower = word.lower()
    if "ai" not in lower:
        return None
    if random.random() > 0.45:  # attested: 1/3 for 'a' variant
        return None
    idx = lower.index("ai")
    corrupted = word[:idx] + "a" + word[idx + 2:]
    return (corrupted, ErrorAnnotation(
        target_word=word, written_as=corrupted,
        rule="vowel_digraph_ai_to_a", category="phonological",
        provenance="attested", confidence="medium"
    ))


def rule_ew_to_aw(word: str) -> Optional[tuple[str, ErrorAnnotation]]:
    """ew -> aw: flew -> flaw"""
    lower = word.lower()
    if "ew" not in lower:
        return None
    if random.random() > 0.50:  # attested: 1/1
        return None
    idx = lower.index("ew")
    corrupted = word[:idx] + "aw" + word[idx + 2:]
    return (corrupted, ErrorAnnotation(
        target_word=word, written_as=corrupted,
        rule="vowel_digraph_ew_to_aw", category="phonological",
        provenance="attested", confidence="medium"
    ))


def rule_ear_to_ere(word: str) -> Optional[tuple[str, ErrorAnnotation]]:
    """ear -> ere: heard -> hered"""
    lower = word.lower()
    if "ear" not in lower:
        return None
    if random.random() > 0.50:  # attested: 1/1
        return None
    idx = lower.index("ear")
    corrupted = word[:idx] + "ere" + word[idx + 3:]
    return (corrupted, ErrorAnnotation(
        target_word=word, written_as=corrupted,
        rule="vowel_digraph_ear_to_ere", category="phonological",
        provenance="attested", confidence="medium"
    ))


def rule_ee_to_y(word: str) -> Optional[tuple[str, ErrorAnnotation]]:
    """ee -> y: bungee -> bungy (word-final ee)"""
    lower = word.lower()
    if not lower.endswith("ee"):
        return None
    if random.random() > 0.40:  # attested: 1/1, lower rate
        return None
    corrupted = word[:-2] + "y"
    return (corrupted, ErrorAnnotation(
        target_word=word, written_as=corrupted,
        rule="vowel_digraph_ee_to_y", category="phonological",
        provenance="attested", confidence="medium"
    ))


def rule_oa_to_o(word: str) -> Optional[tuple[str, ErrorAnnotation]]:
    """oa -> o: floating -> floting (inferred from digraph reduction pattern)"""
    lower = word.lower()
    if "oa" not in lower:
        return None
    if random.random() > 0.40:  # inferred
        return None
    idx = lower.index("oa")
    corrupted = word[:idx] + "o" + word[idx + 2:]
    return (corrupted, ErrorAnnotation(
        target_word=word, written_as=corrupted,
        rule="vowel_digraph_oa_to_o", category="phonological",
        provenance="inferred", confidence="medium"
    ))


def rule_ou_to_ow(word: str) -> Optional[tuple[str, ErrorAnnotation]]:
    """ou -> ow: shouted -> showted (inferred)"""
    lower = word.lower()
    if "ou" not in lower:
        return None
    # Skip words where 'ou' has different pronunciation (though, through, etc.)
    skip_patterns = ["ough", "ould", "ous"]
    if any(p in lower for p in skip_patterns):
        return None
    if random.random() > 0.30:  # inferred, lower confidence
        return None
    idx = lower.index("ou")
    corrupted = word[:idx] + "ow" + word[idx + 2:]
    return (corrupted, ErrorAnnotation(
        target_word=word, written_as=corrupted,
        rule="vowel_digraph_ou_to_ow", category="phonological",
        provenance="inferred", confidence="low"
    ))


# --- Tier 1: Consonant cluster/digraph substitution ---

def rule_sc_to_sk(word: str) -> Optional[tuple[str, ErrorAnnotation]]:
    """sc -> sk: screamed -> skremed"""
    lower = word.lower()
    if not lower.startswith("sc"):
        return None
    if random.random() > 0.70:  # attested: 2/2
        return None
    corrupted = "sk" + word[2:]
    # Preserve original capitalization
    if word[0].isupper():
        corrupted = "Sk" + word[2:]
    return (corrupted, ErrorAnnotation(
        target_word=word, written_as=corrupted,
        rule="consonant_sc_to_sk", category="phonological",
        provenance="attested", confidence="high"
    ))


def rule_gr_to_g(word: str) -> Optional[tuple[str, ErrorAnnotation]]:
    """gr -> g: grabbed -> gabed"""
    lower = word.lower()
    if not lower.startswith("gr"):
        return None
    if random.random() > 0.35:  # attested: 1/~4 gr- words
        return None
    corrupted = word[0] + word[2:]  # drop 'r'
    return (corrupted, ErrorAnnotation(
        target_word=word, written_as=corrupted,
        rule="consonant_cluster_gr_to_g", category="phonological",
        provenance="attested", confidence="medium"
    ))


def rule_ng_to_g(word: str) -> Optional[tuple[str, ErrorAnnotation]]:
    """ng -> g: jumping -> juming, bungee -> bugee"""
    lower = word.lower()
    # Find 'ng' not at position 0
    idx = lower.find("ng", 1)
    if idx == -1:
        return None
    if random.random() > 0.60:  # attested: 2/2
        return None
    corrupted = word[:idx] + word[idx + 1:]  # remove 'n', keep 'g'
    return (corrupted, ErrorAnnotation(
        target_word=word, written_as=corrupted,
        rule="consonant_cluster_ng_to_g", category="phonological",
        provenance="attested", confidence="high"
    ))


def rule_mp_to_m(word: str) -> Optional[tuple[str, ErrorAnnotation]]:
    """mp -> m: jumping -> juming"""
    lower = word.lower()
    if "mp" not in lower:
        return None
    if random.random() > 0.40:  # attested: 1/1
        return None
    idx = lower.index("mp")
    corrupted = word[:idx + 1] + word[idx + 2:]  # keep 'm', drop 'p'
    return (corrupted, ErrorAnnotation(
        target_word=word, written_as=corrupted,
        rule="consonant_cluster_mp_to_m", category="phonological",
        provenance="attested", confidence="medium"
    ))


def rule_initial_cluster_reduction(word: str) -> Optional[tuple[str, ErrorAnnotation]]:
    """Reduce initial consonant clusters: cr->c, tr->t, sp->s, etc."""
    lower = word.lower()
    clusters = [
        ("cr", "c"), ("tr", "t"), ("sp", "s"), ("st", "s"),
        ("fl", "f"), ("bl", "b"), ("cl", "c"), ("pl", "p"),
        ("sl", "s"), ("dr", "d"), ("br", "b"), ("fr", "f"),
        ("pr", "p"), ("spr", "sp"), ("str", "st"),
    ]
    for cluster, replacement in clusters:
        if lower.startswith(cluster):
            if random.random() > 0.15:  # low rate: attested 1/~4
                continue
            if word[0].isupper():
                corrupted = replacement.capitalize() + word[len(cluster):]
            else:
                corrupted = replacement + word[len(cluster):]
            return (corrupted, ErrorAnnotation(
                target_word=word, written_as=corrupted,
                rule="initial_cluster_reduction", category="phonological",
                provenance="inferred", confidence="low"
            ))
    return None


# --- Tier 1: Multi-syllable phonetic restructuring ---

def _count_vowel_groups(word: str) -> int:
    """Rough syllable count by counting vowel groups."""
    return len(re.findall(r'[aeiouy]+', word.lower()))


def rule_multisyllable_vowel_shift(word: str) -> Optional[tuple[str, ErrorAnnotation]]:
    """Shift vowels in multi-syllable words: balcony -> belkany, building -> bailden"""
    if _count_vowel_groups(word) < 3:
        return None
    if len(word) < 6:
        return None
    if random.random() > 0.35:  # attested: 5/~8 complex words
        return None

    lower = word.lower()
    vowels = "aeiou"
    # Find vowel positions
    vowel_positions = [i for i, c in enumerate(lower) if c in vowels]
    if len(vowel_positions) < 2:
        return None

    # Swap or shift a random vowel
    shift_map = {"a": "e", "e": "i", "i": "a", "o": "u", "u": "o"}
    pos = random.choice(vowel_positions)
    old_vowel = lower[pos]
    new_vowel = shift_map.get(old_vowel, old_vowel)

    chars = list(word)
    chars[pos] = new_vowel if word[pos].islower() else new_vowel.upper()
    corrupted = "".join(chars)

    if corrupted.lower() == word.lower():
        return None

    return (corrupted, ErrorAnnotation(
        target_word=word, written_as=corrupted,
        rule="multisyllable_vowel_shift", category="phonological",
        provenance="attested", confidence="medium"
    ))


# --- Tier 2: Morphological errors ---

# Irregular past tense lookup: present -> (past, error_form, error_type)
IRREGULAR_VERBS = {
    "fall": ("fell", "falled", "over-regularization"),
    "see": ("saw", "see", "base-form"),
    "fly": ("flew", "flyed", "over-regularization"),
    "run": ("ran", "runned", "over-regularization"),
    "catch": ("caught", "catched", "over-regularization"),
    "think": ("thought", "thinked", "over-regularization"),
    "bring": ("brought", "bringed", "over-regularization"),
    "know": ("knew", "knowed", "over-regularization"),
    "throw": ("threw", "throwed", "over-regularization"),
    "break": ("broke", "breaked", "over-regularization"),
    "choose": ("chose", "choosed", "over-regularization"),
    "drive": ("drove", "drived", "over-regularization"),
    "freeze": ("froze", "freezed", "over-regularization"),
    "grow": ("grew", "growed", "over-regularization"),
    "hold": ("held", "holded", "over-regularization"),
    "keep": ("kept", "keeped", "over-regularization"),
    "leave": ("left", "leaved", "over-regularization"),
    "lose": ("lost", "losed", "over-regularization"),
    "send": ("sent", "sended", "over-regularization"),
    "shake": ("shook", "shaked", "over-regularization"),
    "speak": ("spoke", "speaked", "over-regularization"),
    "stand": ("stood", "standed", "over-regularization"),
    "swim": ("swam", "swimmed", "over-regularization"),
    "teach": ("taught", "teached", "over-regularization"),
    "tell": ("told", "telled", "over-regularization"),
    "wake": ("woke", "waked", "over-regularization"),
    "win": ("won", "winned", "over-regularization"),
    "write": ("wrote", "writed", "over-regularization"),
    "buy": ("bought", "buyed", "over-regularization"),
    "fight": ("fought", "fighted", "over-regularization"),
    "find": ("found", "finded", "over-regularization"),
    "hear": ("heard", "heared", "over-regularization"),
    "build": ("built", "builded", "over-regularization"),
    "sing": ("sang", "singed", "over-regularization"),
    "sit": ("sat", "sitted", "over-regularization"),
    "sleep": ("slept", "sleeped", "over-regularization"),
    "ride": ("rode", "rided", "over-regularization"),
}

# Build reverse lookup: past_form -> (base, error_form, error_type)
IRREGULAR_PAST_LOOKUP = {}
for base, (past, error, etype) in IRREGULAR_VERBS.items():
    IRREGULAR_PAST_LOOKUP[past] = (base, error, etype)


def rule_irregular_past_tense(word: str) -> Optional[tuple[str, ErrorAnnotation]]:
    """Replace irregular past tense with over-regularized or base form."""
    lower = word.lower()
    if lower not in IRREGULAR_PAST_LOOKUP:
        return None
    if random.random() > 0.40:  # attested: 3/~5 irregular verbs
        return None

    base, error_form, error_type = IRREGULAR_PAST_LOOKUP[lower]

    # 60% over-regularization, 40% base form substitution
    if error_type == "over-regularization" and random.random() < 0.6:
        corrupted = error_form
    else:
        corrupted = base

    # Preserve capitalization
    if word[0].isupper():
        corrupted = corrupted.capitalize()

    return (corrupted, ErrorAnnotation(
        target_word=word, written_as=corrupted,
        rule="irregular_past_tense", category="morphological",
        provenance="attested", confidence="medium"
    ))


def rule_missing_ed_suffix(word: str) -> Optional[tuple[str, ErrorAnnotation]]:
    """Drop -ed/-d suffix: walked -> walk, hated -> hate, grabbed -> grab"""
    lower = word.lower()
    if not lower.endswith("ed"):
        return None
    if len(lower) < 4:
        return None
    if random.random() > 0.20:  # attested: 3/~15 past tense verbs
        return None

    # Handle doubled consonants: grabbed -> grab, dropped -> drop
    if len(lower) > 4 and lower[-3] == lower[-4] and lower[-3] not in "aeiouy":
        corrupted = word[:-3]
    # Handle consonant+ed: walked -> walk, jumped -> jump
    elif lower[-3] not in "aeiouy":
        corrupted = word[:-2]
    # Handle vowel+d: hated -> hate (where base ends in 'e')
    elif lower.endswith("ted") or lower.endswith("ded") or lower.endswith("ned"):
        corrupted = word[:-1]  # just drop the 'd'
    else:
        corrupted = word[:-2]

    if not corrupted or corrupted.lower() == word.lower():
        return None

    return (corrupted, ErrorAnnotation(
        target_word=word, written_as=corrupted,
        rule="missing_ed_suffix", category="morphological",
        provenance="attested", confidence="medium"
    ))


def rule_missing_ing_suffix(word: str) -> Optional[tuple[str, ErrorAnnotation]]:
    """Truncate -ing: running -> run, climbing -> climb"""
    lower = word.lower()
    if not lower.endswith("ing"):
        return None
    if len(lower) < 5:
        return None
    if random.random() > 0.10:  # literature-based, lower rate
        return None

    # Handle doubled consonant before -ing: running -> run
    stem = word[:-3]
    if len(stem) >= 2 and stem[-1] == stem[-2] and stem[-1] not in "aeiouy":
        corrupted = stem[:-1]
    # Handle consonant+ing: jumping -> jump
    else:
        corrupted = stem

    # Some verbs lost an 'e': making -> mak (should be make)
    # Add 'e' back if stem ends in consonant and original likely had silent-e
    if corrupted and corrupted[-1] not in "aeiouy" and len(corrupted) > 2:
        # Check if adding 'e' makes a real-ish word
        pass  # keep as-is for realism

    if not corrupted or corrupted.lower() == word.lower():
        return None

    return (corrupted, ErrorAnnotation(
        target_word=word, written_as=corrupted,
        rule="missing_ing_suffix", category="morphological",
        provenance="literature-based", confidence="low"
    ))


# --- Tier 2: Orthographic errors ---

def rule_letter_transposition(word: str) -> Optional[tuple[str, ErrorAnnotation]]:
    """Swap two adjacent letters: aliens -> ailens"""
    if len(word) < 4:
        return None
    if random.random() > 0.05:  # ~5% rate on content words
        return None

    # Pick a random position (not first or last)
    positions = list(range(1, len(word) - 1))
    random.shuffle(positions)
    pos = positions[0]

    chars = list(word)
    chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
    corrupted = "".join(chars)

    if corrupted.lower() == word.lower():
        return None

    return (corrupted, ErrorAnnotation(
        target_word=word, written_as=corrupted,
        rule="letter_transposition", category="orthographic",
        provenance="attested", confidence="medium"
    ))


def rule_final_consonant_drop(word: str) -> Optional[tuple[str, ErrorAnnotation]]:
    """Drop final consonant: then -> the, just -> jus"""
    if len(word) < 3:
        return None
    lower = word.lower()
    if lower[-1] in "aeiouy":
        return None
    if random.random() > 0.03:  # ~3% rate
        return None

    corrupted = word[:-1]
    return (corrupted, ErrorAnnotation(
        target_word=word, written_as=corrupted,
        rule="final_consonant_drop", category="orthographic",
        provenance="attested", confidence="medium"
    ))


def rule_missing_apostrophe(word: str) -> Optional[tuple[str, ErrorAnnotation]]:
    """Remove apostrophe: that's -> thats, don't -> dont"""
    if "'" not in word:
        return None
    if random.random() > 0.60:  # common in children's writing
        return None

    corrupted = word.replace("'", "")
    return (corrupted, ErrorAnnotation(
        target_word=word, written_as=corrupted,
        rule="missing_apostrophe", category="orthographic",
        provenance="attested", confidence="high"
    ))


def rule_silent_e_addition(word: str) -> Optional[tuple[str, ErrorAnnotation]]:
    """Add unnecessary silent-e: dropped -> droped/drope"""
    lower = word.lower()
    if not lower.endswith("ed") or len(lower) < 5:
        return None
    # Only on doubled consonant + ed: dropped, grabbed, etc.
    if lower[-3] != lower[-4]:
        return None
    if random.random() > 0.15:
        return None

    # Drop one consonant and possibly the 'ed': dropped -> drope
    corrupted = word[:-3] + "e"
    return (corrupted, ErrorAnnotation(
        target_word=word, written_as=corrupted,
        rule="silent_e_addition", category="orthographic",
        provenance="attested", confidence="medium"
    ))


# ---------------------------------------------------------------------------
# NEW: Middle band rules (ages 10-12)
# ---------------------------------------------------------------------------

def rule_double_consonant_omission(word: str) -> Optional[tuple[str, ErrorAnnotation]]:
    """Drop one of a double consonant: running -> runing, stopped -> stoped"""
    lower = word.lower()
    # Find doubled consonants
    for i in range(len(lower) - 1):
        if lower[i] == lower[i + 1] and lower[i] not in "aeiouy":
            if random.random() > 0.40:  # 40% rate
                continue
            corrupted = word[:i] + word[i + 1:]
            return (corrupted, ErrorAnnotation(
                target_word=word, written_as=corrupted,
                rule="double_consonant_omission", category="orthographic",
                provenance="literature-based", confidence="high"
            ))
    return None


def rule_double_consonant_insertion(word: str) -> Optional[tuple[str, ErrorAnnotation]]:
    """Add extra consonant: dining -> dinning, coming -> comming"""
    lower = word.lower()
    if len(lower) < 4:
        return None
    if random.random() > 0.20:  # 20% rate
        return None
    # Find single consonants between vowels (likely doubling targets)
    vowels = "aeiouy"
    for i in range(1, len(lower) - 1):
        if (lower[i] not in vowels and
            lower[i - 1] in vowels and
            i + 1 < len(lower) and lower[i + 1] in vowels and
            # Don't double if already doubled
            (i == 0 or lower[i] != lower[i - 1])):
            corrupted = word[:i + 1] + word[i] + word[i + 1:]
            return (corrupted, ErrorAnnotation(
                target_word=word, written_as=corrupted,
                rule="double_consonant_insertion", category="orthographic",
                provenance="literature-based", confidence="medium"
            ))
    return None


def rule_silent_letter_drop(word: str) -> Optional[tuple[str, ErrorAnnotation]]:
    """Drop silent letters: knife -> nife, wrong -> rong, island -> iland"""
    lower = word.lower()
    silent_patterns = [
        ("kn", "n", 0),     # knife -> nife, knock -> nock
        ("wr", "r", 0),     # wrong -> rong, write -> rite
        ("gn", "n", 0),     # gnaw -> naw
        ("mb", "m", None),  # climb -> clim, lamb -> lam (word-final)
        ("bt", "t", None),  # doubt -> dout, debt -> det (word-final-ish)
    ]
    for pattern, replacement, pos in silent_patterns:
        if pos == 0 and lower.startswith(pattern):
            if random.random() > 0.50:  # 50% rate
                continue
            if word[0].isupper():
                corrupted = replacement.capitalize() + word[len(pattern):]
            else:
                corrupted = replacement + word[len(pattern):]
            return (corrupted, ErrorAnnotation(
                target_word=word, written_as=corrupted,
                rule="silent_letter_drop", category="orthographic",
                provenance="literature-based", confidence="high"
            ))
        elif pos is None and lower.endswith(pattern):
            if random.random() > 0.50:
                continue
            corrupted = word[:-len(pattern)] + replacement
            return (corrupted, ErrorAnnotation(
                target_word=word, written_as=corrupted,
                rule="silent_letter_drop", category="orthographic",
                provenance="literature-based", confidence="high"
            ))
    return None


def rule_vowel_digraph_swap(word: str) -> Optional[tuple[str, ErrorAnnotation]]:
    """Swap vowel digraphs: heat -> heet, fear -> feer"""
    lower = word.lower()
    swaps = [
        ("ea", "ee"),  # heat -> heet
        ("ee", "ea"),  # feet -> feat
        ("ie", "ei"),  # field -> feild
        ("ei", "ie"),  # receive -> recieve
        ("ai", "ay"),  # rain -> rayn
        ("ay", "ai"),  # play -> plai
    ]
    for old, new in swaps:
        if old in lower:
            if random.random() > 0.30:  # 30% rate
                continue
            idx = lower.index(old)
            corrupted = word[:idx] + new + word[idx + len(old):]
            if corrupted.lower() != word.lower():
                return (corrupted, ErrorAnnotation(
                    target_word=word, written_as=corrupted,
                    rule="vowel_digraph_swap", category="orthographic",
                    provenance="literature-based", confidence="medium"
                ))
    return None


# ---------------------------------------------------------------------------
# NEW: Teen band rules (ages 13-17)
# ---------------------------------------------------------------------------

def rule_latin_prefix_error(word: str) -> Optional[tuple[str, ErrorAnnotation]]:
    """Errors in Latin prefixes: disappear -> disapear, unnecessary -> unnessary"""
    lower = word.lower()
    prefix_errors = [
        ("dis", "dis"),     # disappear -> disapear (double->single after prefix)
        ("un", "un"),       # unnecessary -> unnessary
        ("mis", "mis"),     # misspell -> mispell
        ("re", "re"),       # recommend -> recomend
        ("over", "over"),   # overrate -> overate
        ("pre", "pre"),     # prerequisite -> prerequiste
    ]
    for prefix, _ in prefix_errors:
        if lower.startswith(prefix) and len(lower) > len(prefix) + 2:
            rest = lower[len(prefix):]
            # If char after prefix is same as last char of prefix, drop it
            if rest and prefix[-1] == rest[0]:
                if random.random() > 0.35:
                    continue
                corrupted = word[:len(prefix)] + word[len(prefix) + 1:]
                return (corrupted, ErrorAnnotation(
                    target_word=word, written_as=corrupted,
                    rule="latin_prefix_error", category="morphological",
                    provenance="literature-based", confidence="medium"
                ))
            # Or if double consonant follows prefix, reduce it
            if len(rest) > 1 and rest[0] == rest[1] and rest[0] not in "aeiouy":
                if random.random() > 0.35:
                    continue
                corrupted = word[:len(prefix)] + word[len(prefix) + 1:]
                return (corrupted, ErrorAnnotation(
                    target_word=word, written_as=corrupted,
                    rule="latin_prefix_error", category="morphological",
                    provenance="literature-based", confidence="medium"
                ))
    return None


def rule_suffix_confusion(word: str) -> Optional[tuple[str, ErrorAnnotation]]:
    """Common suffix errors: definitely -> definitly, separately -> seperately"""
    lower = word.lower()
    suffix_swaps = [
        ("ately", "atly"),     # definitely -> definitly, separately -> sepratly
        ("itely", "itly"),     # definitely -> definitly
        ("ately", "etely"),    # separately -> seperately
        ("ance", "ence"),      # performance -> performence
        ("ence", "ance"),      # difference -> differance
        ("able", "ible"),      # comfortable -> comfortible
        ("ible", "able"),      # possible -> possable
        ("tion", "shun"),      # education -> educashun
        ("sion", "tion"),      # decision -> decition
        ("ous", "us"),         # dangerous -> dangerus
        ("ious", "ius"),       # serious -> serius
        ("ally", "ly"),        # occasionally -> occasionly
    ]
    for old_suffix, new_suffix in suffix_swaps:
        if lower.endswith(old_suffix):
            if random.random() > 0.40:  # 40% rate
                continue
            corrupted = word[:-len(old_suffix)] + new_suffix
            if corrupted.lower() != word.lower():
                return (corrupted, ErrorAnnotation(
                    target_word=word, written_as=corrupted,
                    rule="suffix_confusion", category="morphological",
                    provenance="literature-based", confidence="medium"
                ))
    return None


def rule_unstressed_vowel_reduction(word: str) -> Optional[tuple[str, ErrorAnnotation]]:
    """Drop unstressed vowels: government -> goverment, different -> diffrent"""
    lower = word.lower()
    if _count_vowel_groups(lower) < 3 or len(lower) < 7:
        return None
    if random.random() > 0.45:  # 45% rate
        return None

    # Find vowels in middle positions (likely unstressed)
    vowels = "aeiou"
    candidates = []
    for i in range(2, len(lower) - 2):
        if lower[i] in vowels and lower[i - 1] not in vowels and lower[i + 1] not in vowels:
            candidates.append(i)

    if not candidates:
        return None

    pos = random.choice(candidates)
    corrupted = word[:pos] + word[pos + 1:]

    if corrupted.lower() == word.lower() or len(corrupted) < 3:
        return None

    return (corrupted, ErrorAnnotation(
        target_word=word, written_as=corrupted,
        rule="unstressed_vowel_reduction", category="orthographic",
        provenance="literature-based", confidence="medium"
    ))


# ---------------------------------------------------------------------------
# NEW: Gap-filling rules based on Pascal story analysis (March 2026)
# These address error types our model couldn't handle.
# ---------------------------------------------------------------------------

# --- Phonological gaps ---

def rule_ght_simplification(word: str) -> Optional[tuple[str, ErrorAnnotation]]:
    """ght -> t: fought -> fot, night -> nit, light -> lit, right -> rit"""
    lower = word.lower()
    if "ght" not in lower:
        return None
    if random.random() > 0.55:
        return None
    idx = lower.index("ght")
    # Remove 'gh', keep 't': fought -> fot
    corrupted = word[:idx] + word[idx + 2:]
    if corrupted.lower() == word.lower():
        return None
    return (corrupted, ErrorAnnotation(
        target_word=word, written_as=corrupted,
        rule="ght_simplification", category="phonological",
        provenance="attested", confidence="high"
    ))


def rule_wh_to_w(word: str) -> Optional[tuple[str, ErrorAnnotation]]:
    """wh -> w: whale -> wale, whistle -> wistle, wheel -> weel"""
    lower = word.lower()
    if not lower.startswith("wh"):
        return None
    # Skip very common words handled elsewhere or where wh->w creates a real word confusingly
    skip = {"what", "who", "which", "while", "why", "white", "whole"}
    if lower in skip:
        return None
    if random.random() > 0.45:
        return None
    if word[0].isupper():
        corrupted = "W" + word[2:]
    else:
        corrupted = "w" + word[2:]
    return (corrupted, ErrorAnnotation(
        target_word=word, written_as=corrupted,
        rule="wh_to_w", category="phonological",
        provenance="attested", confidence="medium"
    ))


def rule_ph_to_f(word: str) -> Optional[tuple[str, ErrorAnnotation]]:
    """ph -> f: phone -> fone, elephant -> elefant, graph -> graf"""
    lower = word.lower()
    if "ph" not in lower:
        return None
    if random.random() > 0.50:
        return None
    idx = lower.index("ph")
    if word[idx].isupper():
        corrupted = word[:idx] + "F" + word[idx + 2:]
    else:
        corrupted = word[:idx] + "f" + word[idx + 2:]
    return (corrupted, ErrorAnnotation(
        target_word=word, written_as=corrupted,
        rule="ph_to_f", category="phonological",
        provenance="literature-based", confidence="high"
    ))


def rule_wrong_silent_letter(word: str) -> Optional[tuple[str, ErrorAnnotation]]:
    """Use wrong silent letter: gnome -> knome, gnaw -> knaw (knows silent letter exists, picks wrong one)"""
    lower = word.lower()
    # gn- at start -> kn- (most common wrong choice)
    if lower.startswith("gn"):
        if random.random() > 0.55:
            return None
        if word[0].isupper():
            corrupted = "Kn" + word[2:]
        else:
            corrupted = "kn" + word[2:]
        return (corrupted, ErrorAnnotation(
            target_word=word, written_as=corrupted,
            rule="wrong_silent_letter", category="phonological",
            provenance="attested", confidence="medium"
        ))
    # ps- at start -> s- (psychology -> sychology — less common, already covered by silent_letter_drop)
    return None


def rule_ough_simplification(word: str) -> Optional[tuple[str, ErrorAnnotation]]:
    """Simplify 'ough' patterns: enough -> enuf, through -> thru, though -> tho"""
    lower = word.lower()
    if "ough" not in lower:
        return None
    if random.random() > 0.50:
        return None

    # Handle by pronunciation pattern
    if lower == "through":
        corrupted = "thru"
    elif lower == "though":
        corrupted = "tho"
    elif lower == "thought":
        corrupted = "thot"
    elif lower == "thorough":
        corrupted = "thuro"
    elif lower.endswith("ought"):
        # bought -> bot, fought -> fot, brought -> brot
        idx = lower.index("ought")
        corrupted = word[:idx] + "ot"
    elif lower.endswith("ough"):
        # enough -> enuf, rough -> ruf, tough -> tuf
        idx = lower.index("ough")
        corrupted = word[:idx] + "uf"
    else:
        return None

    if word[0].isupper():
        corrupted = corrupted[0].upper() + corrupted[1:]
    if corrupted.lower() == word.lower():
        return None

    return (corrupted, ErrorAnnotation(
        target_word=word, written_as=corrupted,
        rule="ough_simplification", category="phonological",
        provenance="literature-based", confidence="medium"
    ))


# --- Orthographic gaps ---

def rule_word_split(word: str) -> Optional[tuple[str, ErrorAnnotation]]:
    """Split a word with accidental space: then -> the n, and -> an d, around -> a round"""
    if len(word) < 4:
        return None
    if random.random() > 0.06:  # ~6% rate on eligible words
        return None
    # Prefer splitting 1-2 chars from end (most realistic: "the n", "an d")
    positions = list(range(2, len(word) - 1))
    if not positions:
        return None
    # Weight toward end of word
    weights = list(range(1, len(positions) + 1))
    pos = random.choices(positions, weights=weights, k=1)[0]
    corrupted = word[:pos] + " " + word[pos:]
    return (corrupted, ErrorAnnotation(
        target_word=word, written_as=corrupted,
        rule="word_split", category="orthographic",
        provenance="attested", confidence="medium"
    ))


# --- Sentence-level rules (operate on full text, not individual words) ---

def rule_word_concatenation(text: str) -> Optional[tuple[str, str, ErrorAnnotation]]:
    """Join two adjacent words: locked door -> lockeddoor, rest of -> restof"""
    words = text.split()
    if len(words) < 4:
        return None
    if random.random() > 0.12:  # ~12% chance per sentence
        return None

    # Find candidate pairs (skip punctuation-heavy boundaries)
    candidates = []
    for i in range(len(words) - 1):
        w1 = re.sub(r'[^a-zA-Z]', '', words[i])
        w2 = re.sub(r'[^a-zA-Z]', '', words[i + 1])
        if len(w1) >= 2 and len(w2) >= 2:
            candidates.append(i)

    if not candidates:
        return None

    pos = random.choice(candidates)
    joined = words[pos] + words[pos + 1]
    new_words = words[:pos] + [joined] + words[pos + 2:]
    corrupted = " ".join(new_words)
    original_phrase = words[pos] + " " + words[pos + 1]

    return (text, corrupted, ErrorAnnotation(
        target_word=original_phrase, written_as=joined,
        rule="word_concatenation", category="orthographic",
        provenance="attested", confidence="medium"
    ))


def rule_duplicate_word(text: str) -> Optional[tuple[str, str, ErrorAnnotation]]:
    """Accidentally double a small function word: a door -> a a door"""
    words = text.split()
    if len(words) < 4:
        return None
    if random.random() > 0.05:  # ~5% chance per sentence
        return None

    function_words = {"a", "an", "the", "to", "in", "on", "at", "is", "it", "of", "or", "no", "we", "he"}
    candidates = [i for i, w in enumerate(words) if w.lower().rstrip('.,!?') in function_words]

    if not candidates:
        return None

    pos = random.choice(candidates)
    new_words = words[:pos] + [words[pos], words[pos]] + words[pos + 1:]
    corrupted = " ".join(new_words)

    return (text, corrupted, ErrorAnnotation(
        target_word=words[pos], written_as=words[pos] + " " + words[pos],
        rule="duplicate_word", category="orthographic",
        provenance="attested", confidence="medium"
    ))


def rule_lowercase_i_sentence(text: str) -> Optional[tuple[str, str, ErrorAnnotation]]:
    """Lowercase standalone 'I': I went -> i went"""
    # Find standalone I (word boundary check)
    if not re.search(r'\bI\b', text):
        return None
    if random.random() > 0.60:  # Very common in kids' writing
        return None

    corrupted = re.sub(r'\bI\b', 'i', text)
    if corrupted == text:
        return None

    return (text, corrupted, ErrorAnnotation(
        target_word="I", written_as="i",
        rule="lowercase_i", category="orthographic",
        provenance="attested", confidence="high"
    ))


def rule_skip_sentence_caps(text: str) -> Optional[tuple[str, str, ErrorAnnotation]]:
    """Skip capitalizing the first word: Then we -> then we"""
    if not text or not text[0].isupper():
        return None
    if random.random() > 0.25:  # 25% chance
        return None

    corrupted = text[0].lower() + text[1:]
    return (text, corrupted, ErrorAnnotation(
        target_word=text.split()[0], written_as=text.split()[0][0].lower() + text.split()[0][1:],
        rule="skip_sentence_caps", category="orthographic",
        provenance="attested", confidence="high"
    ))


# Real-word substitution lookup (expanded from phonetic_misspeller module)
REAL_WORD_SUBS = EXPANDED_REAL_WORD_SUBS


def rule_real_word_substitution(word: str) -> Optional[tuple[str, ErrorAnnotation]]:
    """Substitute with a real but wrong word: form -> from, quiet -> quite"""
    lower = word.lower()
    if lower not in REAL_WORD_SUBS:
        return None
    if random.random() > 0.15:  # 15% rate — subtle error
        return None

    corrupted = REAL_WORD_SUBS[lower]
    if word[0].isupper():
        corrupted = corrupted.capitalize()

    return (corrupted, ErrorAnnotation(
        target_word=word, written_as=corrupted,
        rule="real_word_substitution", category="real_word",
        provenance="literature-based", confidence="high"
    ))


# Word boundary lookup (expanded from phonetic_misspeller module)
WORD_BOUNDARY_ERRORS = EXPANDED_WORD_BOUNDARY_ERRORS


def rule_word_boundary(text: str) -> Optional[tuple[str, str, ErrorAnnotation]]:
    """Join words that should be separate: a lot -> alot (operates on text, not word)"""
    lower = text.lower()
    for phrase, joined in WORD_BOUNDARY_ERRORS.items():
        if phrase in lower:
            if random.random() > 0.50:  # 50% rate
                continue
            idx = lower.index(phrase)
            corrupted = text[:idx] + joined + text[idx + len(phrase):]
            return (text, corrupted, ErrorAnnotation(
                target_word=phrase, written_as=joined,
                rule="word_boundary_error", category="orthographic",
                provenance="literature-based", confidence="high"
            ))
    return None


# ---------------------------------------------------------------------------
# Phonetic misspeller rules (from phonetic_misspeller module)
# ---------------------------------------------------------------------------

def rule_phoneme_misspell(word: str) -> Optional[tuple[str, ErrorAnnotation]]:
    """CMU-dict-based phoneme perturbation: campaign -> campain"""
    if len(word) < 4:
        return None
    if random.random() > 0.25:  # 25% rate
        return None
    result = phoneme_misspell(word)
    if result is None:
        return None
    if word[0].isupper():
        result = result[0].upper() + result[1:]
    return (result, ErrorAnnotation(
        target_word=word, written_as=result,
        rule="phoneme_misspell", category="phonological",
        provenance="data-driven", confidence="medium"
    ))


def rule_keyboard_typo(word: str) -> Optional[tuple[str, ErrorAnnotation]]:
    """Keyboard proximity error: building -> biilding"""
    if len(word) < 4:
        return None
    if random.random() > 0.12:  # 12% rate — subtle
        return None
    result = keyboard_typo(word)
    if result is None:
        return None
    return (result, ErrorAnnotation(
        target_word=word, written_as=result,
        rule="keyboard_typo", category="orthographic",
        provenance="data-driven", confidence="medium"
    ))


def rule_visual_similarity(word: str) -> Optional[tuple[str, ErrorAnnotation]]:
    """Visual similarity error: morning -> rnoining"""
    if len(word) < 4:
        return None
    if random.random() > 0.15:  # 15% rate
        return None
    result = visual_similarity(word)
    if result is None:
        return None
    if word[0].isupper():
        result = result[0].upper() + result[1:]
    return (result, ErrorAnnotation(
        target_word=word, written_as=result,
        rule="visual_similarity", category="orthographic",
        provenance="data-driven", confidence="low"
    ))


def rule_creative_phonetic(word: str) -> Optional[tuple[str, ErrorAnnotation]]:
    """Kid-style phonetic spelling: beautiful -> butiful, nation -> nashun"""
    if len(word) < 5:
        return None
    if random.random() > 0.20:  # 20% rate
        return None
    result = creative_phonetic(word)
    if result is None:
        return None
    if word[0].isupper():
        result = result[0].upper() + result[1:]
    return (result, ErrorAnnotation(
        target_word=word, written_as=result,
        rule="creative_phonetic", category="phonological",
        provenance="data-driven", confidence="medium"
    ))


# ---------------------------------------------------------------------------
# Rule registry — organized by age band
# ---------------------------------------------------------------------------

# Original rules (all ages, weighted differently)
TIER_1_RULES = [
    rule_ea_to_e,
    rule_ei_to_i,
    rule_sc_to_sk,
    rule_ng_to_g,
    rule_multisyllable_vowel_shift,
    rule_ght_simplification,       # NEW: fought -> fot
    rule_wh_to_w,                  # NEW: whale -> wale
    rule_ph_to_f,                  # NEW: phone -> fone
    rule_ough_simplification,      # NEW: enough -> enuf
]

TIER_2_RULES = [
    rule_ai_to_e,
    rule_ai_to_a,
    rule_ew_to_aw,
    rule_ear_to_ere,
    rule_ee_to_y,
    rule_oa_to_o,
    rule_ou_to_ow,
    rule_gr_to_g,
    rule_mp_to_m,
    rule_initial_cluster_reduction,
    rule_irregular_past_tense,
    rule_missing_ed_suffix,
    rule_missing_ing_suffix,
    rule_word_split,               # NEW: then -> the n
]

TIER_3_RULES = [
    rule_letter_transposition,
    rule_final_consonant_drop,
    rule_missing_apostrophe,
    rule_silent_e_addition,
]

# Phonetic misspeller rules (data-driven, all ages)
PHONETIC_RULES = [
    rule_phoneme_misspell,             # CMU dict phoneme perturbation
    rule_creative_phonetic,            # Kid-style phonetic spelling
    rule_keyboard_typo,                # Keyboard proximity errors
    rule_visual_similarity,            # Visual similarity (rn↔m, etc.)
]

# Middle band additions
MIDDLE_RULES = [
    rule_double_consonant_omission,
    rule_double_consonant_insertion,
    rule_silent_letter_drop,
    rule_vowel_digraph_swap,
    rule_wrong_silent_letter,      # NEW: gnome -> knome
]

# Teen band additions
TEEN_RULES = [
    rule_latin_prefix_error,
    rule_suffix_confusion,
    rule_unstressed_vowel_reduction,
    rule_real_word_substitution,
]

ALL_RULES = TIER_1_RULES + TIER_2_RULES + TIER_3_RULES + PHONETIC_RULES

# Sentence-level rules (applied after word-level corruption)
SENTENCE_RULES = [
    rule_word_concatenation,       # NEW: locked door -> lockeddoor
    rule_duplicate_word,           # NEW: a door -> a a door
    rule_lowercase_i_sentence,     # NEW: I went -> i went
    rule_skip_sentence_caps,       # NEW: Then -> then
]

# Age-band rule sets
RULES_BY_AGE = {
    "young": ALL_RULES,
    "middle": ALL_RULES + MIDDLE_RULES,
    "teen": ALL_RULES + MIDDLE_RULES + TEEN_RULES,
}


# ---------------------------------------------------------------------------
# Error density controller
# ---------------------------------------------------------------------------

def choose_error_count(age_band: str = "young") -> int:
    """
    Choose number of errors for a sentence based on age-band distribution.
    """
    config = AGE_BAND_CONFIG.get(age_band, AGE_BAND_CONFIG["young"])
    dist = config["error_distribution"]

    r = random.random()
    cumulative = 0.0
    for count, prob in dist:
        cumulative += prob
        if r < cumulative:
            if count == 3:
                return random.choice([3, 3, 4])
            return count
    return 1


# ---------------------------------------------------------------------------
# Core injection logic
# ---------------------------------------------------------------------------

def _is_content_word(word: str) -> bool:
    """Check if a word is a content word (not a function/sight word)."""
    clean = re.sub(r'[^a-zA-Z\']', '', word).lower()
    if len(clean) < 3:
        return False
    preservation = _get_preservation_set()
    return clean not in preservation


def _strip_punctuation(word: str) -> tuple[str, str, str]:
    """Split word into (leading_punct, core, trailing_punct)."""
    match = re.match(r'^([^a-zA-Z\']*)([a-zA-Z\']+)([^a-zA-Z\']*)$', word)
    if not match:
        return ("", word, "")
    return match.group(1), match.group(2), match.group(3)


def _try_rules_on_word(word: str, rules: list) -> Optional[tuple[str, ErrorAnnotation]]:
    """Try each rule on a word in random order. Return first match or None."""
    shuffled = list(rules)
    random.shuffle(shuffled)
    for rule in shuffled:
        result = rule(word)
        if result is not None:
            return result
    return None


def _select_rules_by_category(age_band: str, rules: list) -> list:
    """Reorder rules based on age-band category weights."""
    config = AGE_BAND_CONFIG.get(age_band, AGE_BAND_CONFIG["young"])
    weights = config["category_weights"]

    # Categorize rules
    categorized = {"phonological": [], "orthographic": [], "morphological": [], "real_word": []}
    for rule in rules:
        # Peek at a test call to determine category, or use heuristics from name
        name = rule.__name__
        if any(x in name for x in ["vowel", "ea_to", "ei_to", "ai_to", "ew_to", "ear_to",
                                     "ee_to", "oa_to", "ou_to", "sc_to", "gr_to", "ng_to",
                                     "mp_to", "cluster", "multisyllable",
                                     "ght_", "wh_to", "ph_to", "wrong_silent", "ough_",
                                     "phoneme_misspell", "creative_phonetic"]):
            categorized["phonological"].append(rule)
        elif any(x in name for x in ["transposition", "consonant_drop", "apostrophe",
                                       "silent_e", "double_consonant", "silent_letter",
                                       "digraph_swap", "unstressed_vowel", "word_boundary",
                                       "word_split", "keyboard_typo", "visual_similarity"]):
            categorized["orthographic"].append(rule)
        elif any(x in name for x in ["past_tense", "ed_suffix", "ing_suffix",
                                       "prefix", "suffix"]):
            categorized["morphological"].append(rule)
        elif "real_word" in name:
            categorized["real_word"].append(rule)
        else:
            categorized["phonological"].append(rule)

    # Build weighted rule order
    r = random.random()
    cumulative = 0.0
    primary = "phonological"
    for cat, weight in weights.items():
        cumulative += weight
        if r < cumulative:
            primary = cat
            break

    # Put primary category first, then others
    ordered = list(categorized.get(primary, []))
    for cat in weights:
        if cat != primary:
            ordered.extend(categorized.get(cat, []))

    return ordered


def inject_errors_sentence(
    sentence: str,
    target_errors: Optional[int] = None,
    category_weights: Optional[dict] = None,
    age_band: str = "young",
) -> SentenceResult:
    """
    Inject errors into a single sentence.

    Args:
        sentence: Clean input sentence.
        target_errors: Number of errors to inject. If None, uses density controller.
        category_weights: Optional dict to bias category distribution.
        age_band: One of "young", "middle", "teen".

    Returns:
        SentenceResult with corrupted text and annotations.
    """
    if target_errors is None:
        target_errors = choose_error_count(age_band)

    if target_errors == 0:
        return SentenceResult(
            original=sentence, corrupted=sentence,
            errors=[], error_count=0, word_error_rate=0.0
        )

    words = sentence.split()
    if not words:
        return SentenceResult(
            original=sentence, corrupted=sentence,
            errors=[], error_count=0, word_error_rate=0.0
        )

    # Find content word indices eligible for corruption
    eligible = []
    for i, w in enumerate(words):
        lead, core, trail = _strip_punctuation(w)
        if core and _is_content_word(core):
            eligible.append(i)

    if not eligible:
        return SentenceResult(
            original=sentence, corrupted=sentence,
            errors=[], error_count=0, word_error_rate=0.0
        )

    # Don't try to inject more errors than eligible words
    target_errors = min(target_errors, len(eligible))

    # Get age-appropriate rules
    available_rules = RULES_BY_AGE.get(age_band, ALL_RULES)

    # Shuffle eligible positions and try to inject
    random.shuffle(eligible)
    errors_injected = []
    result_words = list(words)

    for pos in eligible:
        if len(errors_injected) >= target_errors:
            break

        lead, core, trail = _strip_punctuation(words[pos])

        # Select rules weighted by age-band category distribution
        rules_to_try = _select_rules_by_category(age_band, available_rules)

        result = _try_rules_on_word(core, rules_to_try)
        if result:
            corrupted_core, annotation = result
            annotation.position = pos
            result_words[pos] = lead + corrupted_core + trail
            errors_injected.append(annotation)

    corrupted_sentence = " ".join(result_words)

    # Apply word boundary errors for teen band
    if age_band == "teen":
        wb_result = rule_word_boundary(corrupted_sentence)
        if wb_result:
            _, corrupted_sentence, wb_annotation = wb_result
            errors_injected.append(wb_annotation)

    # Apply sentence-level rules (all age bands, max 1 per sentence to avoid piling up)
    shuffled_sent_rules = list(SENTENCE_RULES)
    random.shuffle(shuffled_sent_rules)
    sent_level_applied = 0
    for sent_rule in shuffled_sent_rules:
        if sent_level_applied >= 1:
            break
        result = sent_rule(corrupted_sentence)
        if result:
            _, corrupted_sentence, annotation = result
            errors_injected.append(annotation)
            sent_level_applied += 1

    word_count = len(words)

    return SentenceResult(
        original=sentence,
        corrupted=corrupted_sentence,
        errors=errors_injected,
        error_count=len(errors_injected),
        word_error_rate=len(errors_injected) / word_count if word_count > 0 else 0.0
    )


def inject_errors_text(
    text: str,
    sentence_split: bool = True,
    age_band: str = "young",
    inconsistency: bool = False,
) -> list[SentenceResult]:
    """
    Inject errors into a full text, sentence by sentence.

    Args:
        text: Clean input text (one or more sentences).
        sentence_split: If True, split on sentence boundaries.
        age_band: One of "young", "middle", "teen".
        inconsistency: If True, same word may be misspelled differently (~10% chance).

    Returns:
        List of SentenceResult objects, one per sentence.
    """
    if sentence_split:
        # Simple sentence splitter (handles ., !, ?)
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    else:
        sentences = [text]

    results = []
    # Track word->corruption mapping for inconsistency injection
    word_corruptions = {}

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        result = inject_errors_sentence(sent, age_band=age_band)

        # Inconsistency injection: sometimes use a different corruption for the same word
        if inconsistency and result.errors:
            for err in result.errors:
                word = err.target_word.lower()
                if word in word_corruptions and random.random() < 0.10:
                    # Re-corrupt this word differently (leave as-is, already different)
                    pass
                else:
                    word_corruptions[word] = err.written_as

        results.append(result)

    return results


# ---------------------------------------------------------------------------
# CLI interface
# ---------------------------------------------------------------------------

def main():
    """Command-line interface for testing error injection."""
    import argparse

    parser = argparse.ArgumentParser(description="Inject dyslexic spelling errors into clean text")
    parser.add_argument("text", nargs="?", help="Text to corrupt (or use --file)")
    parser.add_argument("--file", "-f", help="Path to clean text file")
    parser.add_argument("--errors", "-e", type=int, default=None, help="Target error count per sentence")
    parser.add_argument("--count", "-n", type=int, default=1, help="Number of variants to generate")
    parser.add_argument("--json", "-j", action="store_true", help="Output as JSON")
    parser.add_argument("--seed", "-s", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--age-band", "-a", choices=["young", "middle", "teen"], default="young",
                       help="Age band for error profile (default: young)")
    parser.add_argument("--inconsistency", action="store_true",
                       help="Enable inconsistency injection (same word misspelled differently)")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    if args.file:
        with open(args.file) as f:
            text = f.read()
    elif args.text:
        text = args.text
    else:
        import sys
        text = sys.stdin.read()

    for i in range(args.count):
        results = inject_errors_text(text, age_band=args.age_band, inconsistency=args.inconsistency)
        for r in results:
            if args.json:
                print(json.dumps({
                    "original": r.original,
                    "corrupted": r.corrupted,
                    "error_count": r.error_count,
                    "word_error_rate": round(r.word_error_rate, 3),
                    "age_band": args.age_band,
                    "errors": [
                        {
                            "target_word": e.target_word,
                            "written_as": e.written_as,
                            "rule": e.rule,
                            "category": e.category,
                            "provenance": e.provenance,
                            "confidence": e.confidence,
                        }
                        for e in r.errors
                    ]
                }))
            else:
                print(f"Original:  {r.original}")
                print(f"Corrupted: {r.corrupted}")
                if r.errors:
                    for e in r.errors:
                        print(f"  [{e.category}/{e.provenance}] {e.target_word} -> {e.written_as} ({e.rule})")
                print()


if __name__ == "__main__":
    main()
