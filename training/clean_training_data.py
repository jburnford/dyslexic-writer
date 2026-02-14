#!/usr/bin/env python3
"""
Clean training data: fix Gemini artifacts, newline mismatches,
truncated sentences, homograph corruptions, duplicates, and high-WER pairs.
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

_DATA_DIR = Path(__file__).parent

# Real English words that shouldn't appear as "misspellings"
# These are the most common homograph collisions from error injection
HOMOGRAPH_BLOCKLIST = {
    "bar", "rig", "had", "man", "her", "run", "mat", "far",
    "fun", "sin", "sit", "led", "red", "wed", "bid", "hid",
    "rid", "god", "rod", "dug", "hug", "mug", "rug", "tug",
    "ban", "can", "fan", "pan", "ran", "van", "bat", "cat",
    "fat", "hat", "pat", "rat", "sat", "bit", "fit", "hit",
    "kit", "lit", "pit", "wit", "cot", "dot", "got", "hot",
    "lot", "not", "pot", "rot", "but", "cut", "gut", "hut",
    "nut", "put", "rut", "bet", "get", "jet", "let", "met",
    "net", "pet", "set", "vet", "wet",
}


def strip_markdown(text: str) -> str:
    """Remove markdown bold/italic markers."""
    # Remove ** bold markers
    text = text.replace("**", "")
    # Remove ## headers
    text = re.sub(r'^#{1,3}\s+', '', text, flags=re.MULTILINE)
    # Remove ``` code blocks
    text = text.replace("```", "")
    # Remove bullet markers at line start
    text = re.sub(r'^\* ', '', text, flags=re.MULTILINE)
    return text


def normalize_whitespace(text: str) -> str:
    """Normalize newlines and extra whitespace to single spaces."""
    # Replace newlines with spaces
    text = text.replace("\n\n", " ").replace("\n", " ")
    # Collapse multiple spaces
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def is_truncated(text: str) -> bool:
    """Check if a sentence appears truncated."""
    text = text.strip()
    if not text:
        return True
    # Ends with a title abbreviation (Mrs., Mr., Dr., etc.)
    if re.search(r'\b(Mrs?|Dr|Ms|St)\.\s*$', text):
        return True
    # Very short and no ending punctuation
    words = text.split()
    if len(words) <= 2 and not text[-1] in '.!?"\'':
        return True
    return False


def has_homograph_corruption(pair: dict) -> bool:
    """Check if any error creates a real English word."""
    for error in pair.get("errors", []):
        written = error.get("written_as", "").lower().strip(".,!?;:'\"")
        if written in HOMOGRAPH_BLOCKLIST:
            return True
    return False


def clean_pair(pair: dict) -> Optional[dict]:
    """Clean a single training pair. Returns None if pair should be dropped."""
    # Strip markdown from both fields
    pair["input"] = strip_markdown(pair["input"])
    pair["target"] = strip_markdown(pair["target"])

    # Normalize whitespace
    pair["input"] = normalize_whitespace(pair["input"])
    pair["target"] = normalize_whitespace(pair["target"])

    # Drop truncated sentences
    if is_truncated(pair["input"]) or is_truncated(pair["target"]):
        return None

    # Drop if input or target is empty after cleaning
    if not pair["input"].strip() or not pair["target"].strip():
        return None

    # Drop pairs where error creates a common real word (homograph)
    if has_homograph_corruption(pair):
        return None

    # Drop very high WER pairs (>= 0.5)
    if pair.get("word_error_rate", 0) >= 0.5:
        return None

    # Fix identity pairs where input/target differ only by whitespace
    if pair.get("error_count", 0) == 0 or pair.get("type") == "identity":
        pair["target"] = pair["input"]
        pair["error_count"] = 0
        pair["word_error_rate"] = 0.0
        pair["type"] = "identity"

    return pair


def clean_file(input_path: Path, output_path: Path) -> dict:
    """Clean an entire JSONL file. Returns stats."""
    stats = {
        "input_count": 0,
        "output_count": 0,
        "dropped_markdown": 0,
        "dropped_truncated": 0,
        "dropped_homograph": 0,
        "dropped_high_wer": 0,
        "dropped_duplicate": 0,
        "dropped_empty": 0,
        "markdown_fixed": 0,
        "whitespace_fixed": 0,
    }

    pairs = []
    with open(input_path) as f:
        for line in f:
            if not line.strip():
                continue
            stats["input_count"] += 1
            pair = json.loads(line)

            # Track what we're fixing
            orig_input = pair["input"]
            orig_target = pair["target"]

            if "**" in orig_input or "**" in orig_target:
                stats["markdown_fixed"] += 1
            if "\n" in orig_input or "\n" in orig_target:
                stats["whitespace_fixed"] += 1

            cleaned = clean_pair(pair)

            if cleaned is None:
                # Figure out why it was dropped
                if is_truncated(normalize_whitespace(strip_markdown(orig_input))) or \
                   is_truncated(normalize_whitespace(strip_markdown(orig_target))):
                    stats["dropped_truncated"] += 1
                elif has_homograph_corruption(pair):
                    stats["dropped_homograph"] += 1
                elif pair.get("word_error_rate", 0) >= 0.5:
                    stats["dropped_high_wer"] += 1
                else:
                    stats["dropped_empty"] += 1
                continue

            pairs.append(cleaned)

    # Deduplicate by input text (keep first occurrence)
    seen_inputs = set()
    deduped = []
    for p in pairs:
        key = p["input"]
        if key in seen_inputs:
            stats["dropped_duplicate"] += 1
            continue
        seen_inputs.add(key)
        deduped.append(p)

    stats["output_count"] = len(deduped)

    with open(output_path, "w") as f:
        for p in deduped:
            f.write(json.dumps(p) + "\n")

    return stats


def main():
    print("=" * 60)
    print("Cleaning Training Data")
    print("=" * 60)

    for split in ["train", "eval"]:
        input_path = _DATA_DIR / f"synthetic_{split}.jsonl"
        output_path = _DATA_DIR / f"synthetic_{split}_clean.jsonl"

        if not input_path.exists():
            print(f"Skipping {split}: {input_path} not found")
            continue

        print(f"\n--- Cleaning {split} ---")
        stats = clean_file(input_path, output_path)

        print(f"  Input:              {stats['input_count']}")
        print(f"  Output:             {stats['output_count']}")
        print(f"  Dropped total:      {stats['input_count'] - stats['output_count']}")
        print(f"    Truncated:        {stats['dropped_truncated']}")
        print(f"    Homograph:        {stats['dropped_homograph']}")
        print(f"    High WER:         {stats['dropped_high_wer']}")
        print(f"    Duplicate:        {stats['dropped_duplicate']}")
        print(f"    Empty:            {stats['dropped_empty']}")
        print(f"  Markdown fixed:     {stats['markdown_fixed']}")
        print(f"  Whitespace fixed:   {stats['whitespace_fixed']}")
        print(f"  Saved to: {output_path}")

    # Also convert clean files to instruction format for finetune.py
    print("\n--- Converting to instruction format ---")
    for split in ["train", "eval"]:
        clean_path = _DATA_DIR / f"synthetic_{split}_clean.jsonl"
        instr_path = _DATA_DIR / f"synthetic_{split}_clean_instruction.jsonl"

        if not clean_path.exists():
            continue

        count = 0
        with open(clean_path) as fin, open(instr_path, "w") as fout:
            for line in fin:
                pair = json.loads(line)
                record = {
                    "instruction": "Fix any spelling or grammar errors in this text. If there are no errors, return the text unchanged.",
                    "input": pair["input"],
                    "output": pair["target"],
                }
                fout.write(json.dumps(record) + "\n")
                count += 1
        print(f"  {split}: {count} pairs -> {instr_path}")

    # Verify final distribution
    print("\n--- Final Distribution Check ---")
    clean_path = _DATA_DIR / "synthetic_train_clean.jsonl"
    if clean_path.exists():
        error_counts = Counter()
        cats = Counter()
        total = 0
        with open(clean_path) as f:
            for line in f:
                p = json.loads(line)
                total += 1
                error_counts[p["error_count"]] += 1
                for e in p.get("errors", []):
                    cats[e["category"]] += 1

        print(f"  Total pairs: {total}")
        for k in sorted(error_counts):
            print(f"    {k} errors: {error_counts[k]} ({error_counts[k]/total*100:.1f}%)")
        print(f"  Categories:")
        for k, v in cats.most_common():
            print(f"    {k}: {v} ({v/sum(cats.values())*100:.1f}%)")

    print("\nDone!")


if __name__ == "__main__":
    main()
