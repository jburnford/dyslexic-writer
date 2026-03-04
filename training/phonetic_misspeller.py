#!/usr/bin/env python3
"""
Phonetic misspelling generator for diverse, realistic spelling errors.

Three complementary strategies:
  1. CMU Pronouncing Dictionary phoneme perturbation
  2. Keyboard-proximity and visual-similarity character perturbations
  3. Creative phonetic spelling (kid-style "write what you hear")

Also provides an expanded real-word substitution list (200+ pairs).
"""

import json
import os
import random
import re
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# CMU Pronouncing Dictionary
# ---------------------------------------------------------------------------

_CMU_DICT: dict[str, list[list[str]]] = {}
_CMU_LOADED = False


def _load_cmu_dict():
    """Load CMU Pronouncing Dictionary. Tries nltk first, then direct download."""
    global _CMU_DICT, _CMU_LOADED
    if _CMU_LOADED:
        return

    try:
        import nltk
        try:
            from nltk.corpus import cmudict
            entries = cmudict.entries()
        except LookupError:
            nltk.download('cmudict', quiet=True)
            from nltk.corpus import cmudict
            entries = cmudict.entries()

        for word, phones in entries:
            word_lower = word.lower()
            if word_lower not in _CMU_DICT:
                _CMU_DICT[word_lower] = []
            _CMU_DICT[word_lower].append(phones)
    except ImportError:
        # Fallback: try to load from a cached file
        cache_path = Path(__file__).parent / "cmu_dict_cache.json"
        if cache_path.exists():
            with open(cache_path) as f:
                _CMU_DICT.update(json.load(f))

    _CMU_LOADED = True


def get_pronunciations(word: str) -> list[list[str]]:
    """Get CMU pronunciations for a word."""
    _load_cmu_dict()
    return _CMU_DICT.get(word.lower(), [])


# ---------------------------------------------------------------------------
# Phoneme-to-grapheme mappings (how sounds map to written English)
# ---------------------------------------------------------------------------

# For each phoneme, possible ways it can be written
PHONEME_TO_GRAPHEME = {
    # Vowels
    "AA": ["a", "o", "ah", "au"],         # father, hot
    "AE": ["a", "e", "ai"],               # cat, bad
    "AH": ["u", "a", "o", "e"],           # but, about
    "AO": ["aw", "au", "o", "al"],        # caught, all
    "AW": ["ow", "ou", "aw"],             # cow, out
    "AY": ["i", "igh", "y", "ie", "ai"],  # my, high
    "EH": ["e", "ea", "a", "ai"],         # bed, said
    "ER": ["er", "ur", "ir", "or", "ar"], # her, fur
    "EY": ["a", "ay", "ai", "ei", "ey"],  # say, rain
    "IH": ["i", "e", "y"],                # sit, pretty
    "IY": ["ee", "ea", "y", "ie", "i"],   # see, easy
    "OW": ["o", "ow", "oe", "oa"],        # go, show
    "OY": ["oy", "oi"],                   # boy, oil
    "UH": ["oo", "u", "ou"],              # book, put
    "UW": ["oo", "ue", "ew", "u", "ou"],  # food, blue
    # Consonants
    "B": ["b", "bb"],
    "CH": ["ch", "tch"],
    "D": ["d", "dd", "ed"],
    "DH": ["th"],                          # the, this
    "F": ["f", "ff", "ph", "gh"],
    "G": ["g", "gg", "gh"],
    "HH": ["h"],
    "JH": ["j", "g", "dg"],               # judge, edge
    "K": ["k", "c", "ck", "ch", "q"],
    "L": ["l", "ll"],
    "M": ["m", "mm", "mb"],
    "N": ["n", "nn", "kn", "gn"],
    "NG": ["ng", "n"],
    "P": ["p", "pp"],
    "R": ["r", "rr", "wr"],
    "S": ["s", "ss", "c", "sc"],
    "SH": ["sh", "ti", "ci", "si"],       # ship, nation
    "T": ["t", "tt", "ed"],
    "TH": ["th"],                          # think
    "V": ["v", "ve"],
    "W": ["w", "wh"],
    "Y": ["y"],
    "Z": ["z", "zz", "s", "se"],
    "ZH": ["si", "ge"],                   # vision, garage
}

# Phoneme substitutions (perceptually similar sounds)
PHONEME_SUBS = {
    "AA": ["AH", "AO"],
    "AE": ["EH", "AH"],
    "AH": ["AA", "AE", "IH"],
    "AO": ["AA", "OW"],
    "AW": ["OW", "AO"],
    "AY": ["EY", "IY"],
    "EH": ["AE", "IH", "AH"],
    "ER": ["AH"],
    "EY": ["IY", "AY", "EH"],
    "IH": ["EH", "IY", "AH"],
    "IY": ["IH", "EY"],
    "OW": ["AO", "UW"],
    "OY": ["AW"],
    "UH": ["UW", "AH"],
    "UW": ["UH", "OW"],
    "B": ["P"],
    "CH": ["SH", "JH"],
    "D": ["T", "DH"],
    "DH": ["TH", "D"],
    "F": ["V", "TH"],
    "G": ["K"],
    "JH": ["CH", "SH"],
    "K": ["G"],
    "L": ["R"],
    "M": ["N"],
    "N": ["M"],
    "P": ["B"],
    "R": ["L"],
    "S": ["Z", "SH"],
    "SH": ["S", "CH"],
    "T": ["D"],
    "TH": ["DH", "F"],
    "V": ["F", "B"],
    "Z": ["S"],
    "ZH": ["SH", "JH"],
}


def phoneme_misspell(word: str) -> Optional[str]:
    """
    Generate a phonetically plausible misspelling by perturbing the
    phoneme-to-grapheme mapping.

    Returns None if word not in CMU dict or no perturbation possible.
    """
    pronuns = get_pronunciations(word.lower())
    if not pronuns:
        return None

    phones = random.choice(pronuns)
    # Strip stress markers
    phones_clean = [re.sub(r'\d', '', p) for p in phones]

    # Choose 1-2 phonemes to perturb
    n_perturb = random.choice([1, 1, 1, 2])
    indices = list(range(len(phones_clean)))
    if not indices:
        return None

    random.shuffle(indices)
    perturbed_graphemes = []

    for i, phone in enumerate(phones_clean):
        if i in indices[:n_perturb] and random.random() < 0.7:
            # Either substitute the phoneme or use a different grapheme
            if random.random() < 0.4 and phone in PHONEME_SUBS:
                # Substitute with similar phoneme
                new_phone = random.choice(PHONEME_SUBS[phone])
                graphemes = PHONEME_TO_GRAPHEME.get(new_phone, [new_phone.lower()])
            else:
                # Use different grapheme for same phoneme
                graphemes = PHONEME_TO_GRAPHEME.get(phone, [phone.lower()])

            # Pick a grapheme that's different from what the word actually uses
            chosen = random.choice(graphemes)
            perturbed_graphemes.append(chosen)
        else:
            # Use any valid grapheme for this phoneme
            graphemes = PHONEME_TO_GRAPHEME.get(phone, [phone.lower()])
            perturbed_graphemes.append(random.choice(graphemes))

    result = "".join(perturbed_graphemes)

    # Don't return if identical to original
    if result.lower() == word.lower():
        return None

    # Don't return very short results for longer words (sanity check)
    if len(result) < len(word) * 0.4:
        return None

    return result


# ---------------------------------------------------------------------------
# Keyboard proximity perturbations
# ---------------------------------------------------------------------------

KEYBOARD_NEIGHBORS = {
    'q': 'wa', 'w': 'qase', 'e': 'wsdr', 'r': 'edft', 't': 'rfgy',
    'y': 'tghu', 'u': 'yhji', 'i': 'ujko', 'o': 'iklp', 'p': 'ol',
    'a': 'qwsz', 's': 'awedxz', 'd': 'serfcx', 'f': 'drtgvc',
    'g': 'ftyhbv', 'h': 'gyujnb', 'j': 'huikmn', 'k': 'jiolm',
    'l': 'kop', 'z': 'asx', 'x': 'zsdc', 'c': 'xdfv',
    'v': 'cfgb', 'b': 'vghn', 'n': 'bhjm', 'm': 'njk',
}


def keyboard_typo(word: str) -> Optional[str]:
    """
    Introduce a keyboard-proximity typo: swap one character for a neighbor.
    """
    if len(word) < 3:
        return None

    # Pick a random position (not first or last for more realism)
    positions = list(range(1, len(word) - 1)) if len(word) > 3 else list(range(len(word)))
    if not positions:
        return None

    pos = random.choice(positions)
    char = word[pos].lower()

    if char not in KEYBOARD_NEIGHBORS:
        return None

    neighbors = KEYBOARD_NEIGHBORS[char]
    replacement = random.choice(list(neighbors))

    # Preserve case
    if word[pos].isupper():
        replacement = replacement.upper()

    result = word[:pos] + replacement + word[pos + 1:]
    return result if result.lower() != word.lower() else None


# ---------------------------------------------------------------------------
# Visual similarity perturbations
# ---------------------------------------------------------------------------

VISUAL_SUBS = {
    "rn": "m",
    "m": "rn",
    "cl": "d",
    "d": "cl",
    "vv": "w",
    "w": "vv",
    "li": "h",
    "nn": "m",
    "ii": "u",
    "u": "ii",
}


def visual_similarity(word: str) -> Optional[str]:
    """
    Replace character sequences that look similar (rn↔m, cl↔d, etc.).
    """
    if len(word) < 3:
        return None

    lower = word.lower()
    candidates = []

    for pattern, replacement in VISUAL_SUBS.items():
        if pattern in lower:
            idx = lower.index(pattern)
            new = lower[:idx] + replacement + lower[idx + len(pattern):]
            if new != lower:
                candidates.append(new)

    if not candidates:
        return None

    result = random.choice(candidates)

    # Restore original casing (best effort)
    if word[0].isupper() and result:
        result = result[0].upper() + result[1:]

    return result


# ---------------------------------------------------------------------------
# Character-level random perturbations
# ---------------------------------------------------------------------------

def char_perturbation(word: str) -> Optional[str]:
    """
    Apply a random character-level perturbation:
    - insertion, deletion, substitution, or transposition
    """
    if len(word) < 3:
        return None

    op = random.choice(["insert", "delete", "substitute", "transpose"])
    lower = word.lower()

    if op == "insert":
        # Insert a random letter
        pos = random.randint(1, len(lower) - 1)
        char = random.choice("aeioubcdfghjklmnpqrstvwxyz")
        result = lower[:pos] + char + lower[pos:]

    elif op == "delete":
        # Delete a random letter (not first)
        pos = random.randint(1, len(lower) - 1)
        result = lower[:pos] + lower[pos + 1:]
        if len(result) < 2:
            return None

    elif op == "substitute":
        # Substitute with a random vowel/consonant
        pos = random.randint(1, len(lower) - 1)
        if lower[pos] in "aeiou":
            char = random.choice("aeiou")
        else:
            char = random.choice("bcdfghjklmnpqrstvwxyz")
        result = lower[:pos] + char + lower[pos + 1:]

    elif op == "transpose":
        # Swap two adjacent letters
        pos = random.randint(0, len(lower) - 2)
        chars = list(lower)
        chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
        result = "".join(chars)

    else:
        return None

    if result == lower:
        return None

    # Restore casing
    if word[0].isupper():
        result = result[0].upper() + result[1:]

    return result


# ---------------------------------------------------------------------------
# Creative phonetic spelling (kid-style "write what you hear")
# ---------------------------------------------------------------------------

# Common phonetic simplification patterns children use
PHONETIC_PATTERNS = [
    # (regex_pattern, replacement, description)
    (r'tion$', 'shun', 'tion → shun'),
    (r'sion$', 'shun', 'sion → shun'),
    (r'ious$', 'eus', 'ious → eus'),
    (r'ous$', 'us', 'ous → us'),
    (r'ious$', 'yus', 'ious → yus'),
    (r'ture$', 'cher', 'ture → cher'),
    (r'sure$', 'sher', 'sure → sher'),
    (r'cial$', 'shul', 'cial → shul'),
    (r'tial$', 'shul', 'tial → shul'),
    (r'cious$', 'shus', 'cious → shus'),
    (r'tious$', 'shus', 'tious → shus'),
    (r'ough', 'uf', 'ough → uf'),
    (r'ight', 'ite', 'ight → ite'),
    (r'ould', 'ood', 'ould → ood'),
    (r'ould', 'ud', 'ould → ud'),
    (r'ough$', 'o', 'ough → o'),
    (r'ough', 'off', 'ough → off'),
    (r'ough', 'ow', 'ough → ow'),
    (r'augh', 'aff', 'augh → aff'),
    (r'eigh', 'ay', 'eigh → ay'),
    (r'(?<=[aeiou])ght', 't', 'ght → t after vowel'),
    (r'wh', 'w', 'wh → w'),
    (r'wr', 'r', 'wr → r'),
    (r'kn', 'n', 'kn → n'),
    (r'gn', 'n', 'gn → n'),
    (r'mb$', 'm', 'mb → m'),
    (r'mn$', 'm', 'mn → m'),
    (r'(?<=[aeiou])ble$', 'bul', 'ble → bul'),
    (r'(?<=[aeiou])ple$', 'pul', 'ple → pul'),
    (r'ence$', 'ens', 'ence → ens'),
    (r'ance$', 'ans', 'ance → ans'),
    (r'ment$', 'mint', 'ment → mint'),
    (r'ness$', 'nis', 'ness → nis'),
    (r'able$', 'ubul', 'able → ubul'),
    (r'ible$', 'ubul', 'ible → ubul'),
    (r'ally$', 'aly', 'ally → aly'),
    (r'ful$', 'full', 'ful → full'),
    (r'ful$', 'ful', 'ful → ful'),
    (r'ology$', 'olojy', 'ology → olojy'),
    (r'ph', 'f', 'ph → f'),
    (r'qu', 'kw', 'qu → kw'),
    (r'x', 'ks', 'x → ks'),
    (r'(?<=[aeiou])ce$', 'se', 'ce → se'),
    (r'(?<=[aeiou])ge$', 'je', 'ge → je'),
    (r'(?<=[aeiou])se$', 'ze', 'se → ze'),
    (r'ey$', 'ee', 'ey → ee'),
    (r'ey$', 'y', 'ey → y'),
    (r'ie$', 'y', 'ie → y'),
    (r'ck', 'k', 'ck → k'),
    (r'dge', 'j', 'dge → j'),
    (r'(?<=[bcdfghjklmnpqrstvwxyz])le$', 'ul', 'consonant+le → ul'),
    (r'(?<=[bcdfghjklmnpqrstvwxyz])er$', 'r', 'consonant+er → r'),
    (r'(?<=[bcdfghjklmnpqrstvwxyz])or$', 'r', 'consonant+or → r'),
    (r'(?<=[aeiou])r(?=[aeiou])', 'rr', 'intervocalic r → rr'),
    (r'(?<=[aeiou])([bcdfghjklmnpqrstvwxyz])\1', lambda m: m.group(1), 'double consonant → single'),
    (r'(?<=[aeiou])([bcdfghjklmnpqrstvwxyz])(?=[aeiou])', lambda m: m.group(1) * 2, 'single → double between vowels'),
    # Vowel simplifications
    (r'ai', 'a', 'ai → a'),
    (r'ea', 'e', 'ea → e'),
    (r'ou', 'ow', 'ou → ow'),
    (r'ei', 'ee', 'ei → ee'),
    (r'oa', 'o', 'oa → o'),
    (r'ui', 'oo', 'ui → oo'),
    (r'ie', 'ee', 'ie → ee'),
]


def creative_phonetic(word: str) -> Optional[str]:
    """
    Apply kid-style phonetic spelling. Applies 1-2 phonetic patterns.
    """
    if len(word) < 4:
        return None

    lower = word.lower()
    applicable = []

    for pattern, replacement, desc in PHONETIC_PATTERNS:
        try:
            if re.search(pattern, lower):
                applicable.append((pattern, replacement, desc))
        except Exception:
            continue

    if not applicable:
        return None

    # Apply 1-2 patterns
    n_apply = min(random.choice([1, 1, 2]), len(applicable))
    random.shuffle(applicable)

    result = lower
    for pattern, replacement, desc in applicable[:n_apply]:
        try:
            if callable(replacement):
                result = re.sub(pattern, replacement, result, count=1)
            else:
                result = re.sub(pattern, replacement, result, count=1)
        except Exception:
            continue

    if result == lower:
        return None

    # Restore casing
    if word[0].isupper():
        result = result[0].upper() + result[1:]

    return result


# ---------------------------------------------------------------------------
# Expanded real-word substitution list (200+ pairs)
# ---------------------------------------------------------------------------

EXPANDED_REAL_WORD_SUBS = {
    # Original 14 pairs
    "form": "from", "from": "form",
    "tried": "tired", "tired": "tried",
    "quiet": "quite", "quite": "quiet",
    "angel": "angle", "angle": "angel",
    "diary": "dairy", "dairy": "diary",
    "desert": "dessert", "dessert": "desert",
    "loose": "lose", "lose": "loose",
    "accept": "except", "except": "accept",
    "affect": "effect", "effect": "affect",
    "advice": "advise", "advise": "advice",
    "than": "then", "then": "than",
    "were": "where", "where": "were",
    "through": "thorough", "thorough": "through",
    "thought": "though", "though": "thought",

    # Homophones and near-homophones
    "their": "there", "there": "their",
    "they're": "their",
    "your": "you're", "you're": "your",
    "its": "it's", "it's": "its",
    "to": "too", "too": "to",
    "two": "too",
    "know": "no", "no": "know",
    "knew": "new", "new": "knew",
    "right": "write", "write": "right",
    "night": "knight", "knight": "night",
    "sea": "see", "see": "sea",
    "be": "bee", "bee": "be",
    "hear": "here", "here": "hear",
    "for": "four", "four": "for",
    "ate": "eight", "eight": "ate",
    "wear": "where",
    "one": "won", "won": "one",
    "son": "sun", "sun": "son",
    "way": "weigh", "weigh": "way",
    "wait": "weight", "weight": "wait",
    "peace": "piece", "piece": "peace",
    "break": "brake", "brake": "break",
    "sale": "sail", "sail": "sale",
    "tail": "tale", "tale": "tail",
    "mail": "male", "male": "mail",
    "main": "mane", "mane": "main",
    "pain": "pane", "pane": "pain",
    "rain": "reign", "reign": "rain",
    "plain": "plane", "plane": "plain",
    "stair": "stare", "stare": "stair",
    "pair": "pare", "pare": "pair",
    "fair": "fare", "fare": "fair",
    "hair": "hare", "hare": "hair",
    "bear": "bare", "bare": "bear",
    "flower": "flour", "flour": "flower",
    "hour": "our",
    "whole": "hole", "hole": "whole",
    "role": "roll", "roll": "role",
    "sole": "soul", "soul": "sole",
    "pole": "poll", "poll": "pole",
    "steal": "steel", "steel": "steal",
    "heal": "heel", "heel": "heal",
    "peel": "peal", "peal": "peel",
    "real": "reel", "reel": "real",
    "week": "weak", "weak": "week",
    "peek": "peak", "peak": "peek",
    "meet": "meat", "meat": "meet",
    "beat": "beet", "beet": "beat",
    "feet": "feat", "feat": "feet",
    "sweet": "suite",
    "board": "bored", "bored": "board",
    "ward": "word",
    "morning": "mourning", "mourning": "morning",
    "warn": "worn", "worn": "warn",
    "war": "wore",
    "shore": "sure",
    "poor": "pour", "pour": "poor",
    "ore": "or",
    "road": "rode", "rode": "road",
    "toad": "towed", "towed": "toad",
    "load": "lode",
    "groan": "grown", "grown": "groan",
    "thrown": "throne", "throne": "thrown",
    "shown": "shone", "shone": "shown",
    "loan": "lone", "lone": "loan",
    "moan": "mown",
    "sew": "so",
    "sow": "sew",
    "toe": "tow", "tow": "toe",
    "doe": "dough", "dough": "doe",
    "foe": "faux",
    "die": "dye", "dye": "die",
    "buy": "by", "by": "buy",
    "eye": "I",
    "tied": "tide", "tide": "tied",
    "side": "sighed",
    "find": "fined",
    "mind": "mined",
    "sign": "sine",
    "wine": "whine", "whine": "wine",
    "dine": "dyne",
    "might": "mite", "mite": "might",
    "site": "sight", "sight": "site",
    "cite": "sight",
    "bite": "bight",
    "lite": "light",
    "guise": "guys",
    "prize": "pries",
    "wise": "whys",
    "days": "daze", "daze": "days",
    "gaze": "gays",
    "raise": "rays",
    "praise": "prays",
    "made": "maid", "maid": "made",
    "paid": "played",
    "aid": "aide",
    "led": "lead",
    "red": "read", "read": "red",
    "bred": "bread", "bread": "bred",
    "said": "sed",
    "head": "hed",
    "dead": "ded",
    "led": "lead",
    "guessed": "guest",
    "past": "passed", "passed": "past",
    "last": "lassed",
    "cast": "caste",
    "waste": "waist", "waist": "waste",
    "paste": "paced",
    "base": "bass",
    "race": "raise",
    "lace": "lase",
    "which": "witch", "witch": "which",
    "rich": "riche",
    "beach": "beech", "beech": "beach",
    "leach": "leech", "leech": "leach",
    "birth": "berth", "berth": "birth",
    "fir": "fur", "fur": "fir",
    "herd": "heard", "heard": "herd",
    "curb": "kerb",
    "born": "borne",
    "cord": "chord", "chord": "cord",
    "coarse": "course", "course": "coarse",
    "horse": "hoarse",
    "source": "sauce",
    "would": "wood", "wood": "would",
    "could": "cud",
    "should": "shud",
    "root": "route",
    "boot": "bout",
    "through": "threw",
    "blue": "blew", "blew": "blue",
    "clue": "clew",
    "due": "dew", "dew": "due",
    "flew": "flu", "flu": "flew",
    "stew": "stue",
    "crew": "crue",
    "who's": "whose", "whose": "who's",
    "we're": "were",
    "we'll": "weal",
    "been": "bin",
    "scene": "seen", "seen": "scene",
    "bean": "been",
    "ceil": "seal",
    "seize": "sees",
    "allowed": "aloud", "aloud": "allowed",
    "alley": "ally",
    "altar": "alter", "alter": "altar",
    "assent": "ascent", "ascent": "assent",
    "bridal": "bridle", "bridle": "bridal",
    "canvas": "canvass",
    "capital": "capitol",
    "cereal": "serial", "serial": "cereal",
    "complement": "compliment", "compliment": "complement",
    "council": "counsel", "counsel": "council",
    "descent": "dissent",
    "device": "devise", "devise": "device",
    "discreet": "discrete", "discrete": "discreet",
    "elicit": "illicit", "illicit": "elicit",
    "eminent": "imminent", "imminent": "eminent",
    "gorilla": "guerrilla",
    "hangar": "hanger", "hanger": "hangar",
    "licence": "license", "license": "licence",
    "lightening": "lightning", "lightning": "lightening",
    "marshal": "martial",
    "medal": "meddle", "meddle": "medal",
    "metal": "mettle", "mettle": "metal",
    "miner": "minor", "minor": "miner",
    "moral": "morale", "morale": "moral",
    "muscle": "mussel", "mussel": "muscle",
    "naval": "navel", "navel": "naval",
    "pedal": "peddle", "peddle": "pedal",
    "personal": "personnel",
    "principal": "principle", "principle": "principal",
    "profit": "prophet", "prophet": "profit",
    "stationary": "stationery", "stationery": "stationary",
    "storey": "story",
    "weather": "whether", "whether": "weather",
    "wander": "wonder", "wonder": "wander",
    "choose": "chose",
    "dose": "doze",
    "breath": "breathe",
    "bath": "bathe",
    "cloth": "clothe",
}


# ---------------------------------------------------------------------------
# Expanded word boundary errors
# ---------------------------------------------------------------------------

EXPANDED_WORD_BOUNDARY_ERRORS = {
    "a lot": "alot",
    "all right": "alright",
    "each other": "eachother",
    "no one": "noone",
    "in fact": "infact",
    "as well": "aswell",
    "at least": "atleast",
    "in front": "infront",
    "any way": "anyway",
    "some times": "sometimes",
    "every one": "everyone",
    "every thing": "everything",
    "some thing": "something",
    "mean while": "meanwhile",
    "never the less": "nevertheless",
    "can not": "cannot",
    "may be": "maybe",
}


# ---------------------------------------------------------------------------
# Unified misspelling interface
# ---------------------------------------------------------------------------

def generate_misspelling(word: str, strategy: Optional[str] = None) -> Optional[str]:
    """
    Generate a plausible misspelling of a word using one of several strategies.

    Args:
        word: The correctly-spelled word.
        strategy: One of 'phonetic', 'phoneme', 'keyboard', 'visual', 'char',
                  'creative', or None (random selection).

    Returns:
        A misspelled version, or None if no perturbation was possible.
    """
    if len(word) < 3:
        return None

    strategies = ['phoneme', 'keyboard', 'visual', 'char', 'creative']
    if strategy is None:
        # Weight creative and phoneme higher for realism
        strategy = random.choices(
            strategies,
            weights=[30, 15, 10, 15, 30],
            k=1,
        )[0]

    if strategy == 'phoneme':
        return phoneme_misspell(word)
    elif strategy == 'keyboard':
        return keyboard_typo(word)
    elif strategy == 'visual':
        return visual_similarity(word)
    elif strategy == 'char':
        return char_perturbation(word)
    elif strategy == 'creative':
        return creative_phonetic(word)
    else:
        return None


def get_real_word_sub(word: str) -> Optional[str]:
    """Look up a real-word substitution for the given word."""
    return EXPANDED_REAL_WORD_SUBS.get(word.lower())


# ---------------------------------------------------------------------------
# CLI testing
# ---------------------------------------------------------------------------

def main():
    """Test the phonetic misspeller on sample words."""
    import argparse

    parser = argparse.ArgumentParser(description="Phonetic misspelling generator")
    parser.add_argument("words", nargs="*", default=[
        "campaign", "citizens", "beautiful", "necessary", "government",
        "definitely", "restaurant", "environment", "separate", "basically",
        "elephant", "telephone", "knight", "whistle", "enough",
        "receive", "believe", "friend", "chocolate", "February",
    ])
    parser.add_argument("--strategy", choices=['phoneme', 'keyboard', 'visual', 'char', 'creative'],
                       default=None, help="Specific strategy to use")
    parser.add_argument("--count", "-n", type=int, default=5, help="Variants per word")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    print(f"Strategy: {args.strategy or 'random'}\n")

    for word in args.words:
        variants = set()
        for _ in range(args.count * 3):  # try extra times to get enough unique
            result = generate_misspelling(word, strategy=args.strategy)
            if result and result.lower() != word.lower():
                variants.add(result)
            if len(variants) >= args.count:
                break

        print(f"  {word:20s} -> {', '.join(sorted(variants)) if variants else '(no variants)'}")


if __name__ == "__main__":
    main()
