#!/usr/bin/env python3
"""
Extract spelling corrections from GitHub Typo Corpus.

Filters for actual word typos (not code formatting/spacing issues).
"""

import json
import gzip
import re
import csv
from pathlib import Path
from collections import defaultdict

def is_word_typo(src_text: str, tgt_text: str) -> list[tuple[str, str]]:
    """
    Check if this is a word-level spelling typo (not code/formatting).
    Returns list of (misspelling, correction) pairs.
    """
    pairs = []

    # Tokenize both strings
    src_words = re.findall(r'\b[a-zA-Z]+\b', src_text.lower())
    tgt_words = re.findall(r'\b[a-zA-Z]+\b', tgt_text.lower())

    # Skip if word counts are very different (structural change, not typo)
    if abs(len(src_words) - len(tgt_words)) > 1:
        return []

    # Find differences
    if len(src_words) == len(tgt_words):
        for src_w, tgt_w in zip(src_words, tgt_words):
            if src_w != tgt_w:
                # Skip very short words
                if len(src_w) < 3 or len(tgt_w) < 3:
                    continue
                # Skip if too different (probably not a typo)
                if abs(len(src_w) - len(tgt_w)) > 3:
                    continue
                # Skip programming terms and abbreviations
                if is_code_term(src_w) or is_code_term(tgt_w):
                    continue
                # Skip semantic changes (both are valid different words)
                if is_semantic_change(src_w, tgt_w):
                    continue
                pairs.append((src_w, tgt_w))

    return pairs

def is_code_term(word: str) -> bool:
    """Check if word looks like a programming term."""
    code_patterns = [
        r'^(get|set|has|is|on|to|str|int|bool|arr|obj|func|var|val|def|cls|src|tgt|tmp|ptr|idx|len|max|min|err|msg|buf|ctx|cfg|env|api|url|uri|sql|css|html|xml|json|yaml|http|tcp|udp)$',
        r'[A-Z]{2,}',  # ALLCAPS
        r'^\d',  # starts with number
        r'_',  # underscore
    ]
    for pattern in code_patterns:
        if re.search(pattern, word):
            return True
    return False

# Common words that shouldn't be "corrected" to each other (semantic, not spelling)
VALID_WORDS = {
    'first', 'last', 'next', 'prev', 'previous', 'open', 'close', 'opens', 'closes',
    'list', 'lists', 'update', 'updates', 'create', 'creates', 'delete', 'deletes',
    'add', 'adds', 'remove', 'removes', 'start', 'stop', 'starts', 'stops',
    'read', 'write', 'reads', 'writes', 'send', 'receive', 'sends', 'receives',
    'high', 'low', 'highest', 'lowest', 'left', 'right', 'up', 'down',
    'name', 'names', 'value', 'values', 'key', 'keys', 'type', 'types',
    'file', 'files', 'path', 'paths', 'node', 'nodes', 'item', 'items',
    'child', 'parent', 'children', 'parents', 'sibling', 'siblings',
    'true', 'false', 'null', 'none', 'empty', 'full',
    'one', 'two', 'three', 'four', 'five', 'single', 'double', 'triple',
    'excerpt', 'excerpts', 'prime', 'primer', 'primes',
}

def is_semantic_change(src: str, tgt: str) -> bool:
    """Check if both words are valid and this is a semantic change, not a typo."""
    # If both are common words, it's likely a semantic fix, not a typo
    if src in VALID_WORDS and tgt in VALID_WORDS:
        return True
    # Singular/plural changes on common suffixes
    if src + 's' == tgt or tgt + 's' == src:
        if len(src) > 4 and len(tgt) > 4:  # Both reasonably long words
            return True
    return False

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

def main():
    input_path = Path("/home/burnford/Downloads/github-typo-corpus.v1.0.0.jsonl.gz")
    output_dir = Path(__file__).parent

    pairs = []
    seen = set()

    print("Processing GitHub Typo Corpus...")

    with gzip.open(input_path, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i % 50000 == 0:
                print(f"  Processed {i:,} entries, found {len(pairs):,} spelling pairs...")

            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            for edit in entry.get('edits', []):
                # Only process confirmed typos
                if not edit.get('is_typo', False):
                    continue

                # Only English
                if edit.get('src', {}).get('lang') != 'eng':
                    continue

                src_text = edit.get('src', {}).get('text', '')
                tgt_text = edit.get('tgt', {}).get('text', '')

                # Extract word-level typos
                word_pairs = is_word_typo(src_text, tgt_text)

                for misspelling, correct in word_pairs:
                    key = (misspelling, correct)
                    if key not in seen:
                        seen.add(key)
                        pairs.append((misspelling, correct))

    print(f"\nExtracted {len(pairs):,} unique spelling pairs")

    # Save as CSV
    csv_path = output_dir / "github_typos_pairs.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['misspelling', 'correct', 'edit_distance'])
        for m, c in pairs:
            writer.writerow([m, c, levenshtein(m, c)])
    print(f"Saved to {csv_path}")

    # Save as JSONL for training
    jsonl_path = output_dir / "github_typos_instruction.jsonl"
    with open(jsonl_path, 'w') as f:
        for m, c in pairs:
            entry = {
                "instruction": "Fix the spelling of this word.",
                "input": m,
                "output": c
            }
            f.write(json.dumps(entry) + '\n')
    print(f"Saved to {jsonl_path}")

    # Stats
    print("\nSample pairs:")
    for m, c in pairs[:20]:
        print(f"  {m} -> {c}")

    # Edit distance distribution
    distances = defaultdict(int)
    for m, c in pairs:
        d = levenshtein(m, c)
        distances[d] += 1

    print("\nEdit distance distribution:")
    for d in sorted(distances.keys())[:6]:
        pct = distances[d] / len(pairs) * 100
        print(f"  {d}: {distances[d]:,} ({pct:.1f}%)")

if __name__ == "__main__":
    main()
