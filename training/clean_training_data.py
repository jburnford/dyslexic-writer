#!/usr/bin/env python3
"""
Clean training data: fix source-level issues, then post-process train/eval.

Addresses all issues from the quality review:
  CRITICAL: GitHub semantic substitutions, conflicting examples
  HIGH: Holbrook trailing numbers, offensive words, </think> contamination
  MEDIUM: Duplicates, train/eval overlap, markdown artifacts
"""

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "training-data"
STORY_DIR = SCRIPT_DIR / "generated_stories"

# ---------------------------------------------------------------------------
# Offensive word filter
# ---------------------------------------------------------------------------

OFFENSIVE_OUTPUTS = {
    "shit", "bitch", "bitchy", "cunt", "fag", "faggot",
    "nigger", "nigga", "retard", "retarded",
}

# Words that error injection can accidentally create
OFFENSIVE_INJECTED = {
    "cock", "fag", "fags", "cok", "dik", "dic",
    "tit", "tits", "cum", "piss",
}


def _has_offensive_injection(input_text, output_text):
    """Check if error injection accidentally created an offensive word."""
    if input_text == output_text:
        return False
    input_words = set(input_text.lower().split())
    output_words = set(output_text.lower().split())
    # Words in input but not output = injected errors
    injected = input_words - output_words
    return bool(injected & OFFENSIVE_INJECTED)


# ---------------------------------------------------------------------------
# String similarity
# ---------------------------------------------------------------------------

def similarity(a, b):
    """String similarity ratio between two strings."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


# ---------------------------------------------------------------------------
# Phase 1: Clean source CSV files
# ---------------------------------------------------------------------------

def clean_holbrook():
    """Fix Holbrook CSV: strip trailing frequency numbers, remove ? outputs."""
    path = DATA_DIR / "holbrook_pairs.csv"
    if not path.exists():
        print("  Holbrook CSV not found, skipping")
        return 0

    rows = []
    removed = 0
    fixed = 0
    with open(path) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            m = row.get("misspelling", "").strip()
            c = row.get("correct", "").strip()

            # Remove entries with ? output
            if c == "?":
                removed += 1
                continue

            # Strip trailing frequency numbers: "rigth 2" -> "rigth"
            m_clean = re.sub(r'\s+\d+$', '', m)
            if m_clean != m:
                fixed += 1
                row["misspelling"] = m_clean

            rows.append(row)

    # Write back
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Holbrook: removed {removed} '?' entries, fixed {fixed} trailing numbers")
    return removed + fixed


def clean_github_typos():
    """Filter GitHub typos: remove semantic substitutions and code identifiers."""
    path = DATA_DIR / "github_typos_pairs.csv"
    if not path.exists():
        print("  GitHub typos CSV not found, skipping")
        return 0

    rows = []
    removed_semantic = 0
    removed_code = 0
    removed_offensive = 0

    # Code-like patterns: camelCase (strict: requires 2+ lowercase then uppercase)
    # and specific tech terms
    code_pattern = re.compile(
        r'[a-z]{2}[A-Z][a-z]|'  # strict camelCase like "forEach", "fileName"
        r'[A-Z]{3,}|'           # ALL_CAPS identifiers (3+ chars)
        r'\b(npm|webpack|kubernetes|django|flask|reactjs|redux|'
        r'rackspace|github|gitlab|docker|ansible|nginx|postgres|'
        r'mongodb|elasticsearch|localhost|stderr|stdout|argv)\b'
    )

    with open(path) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            m = row.get("misspelling", "").strip()
            c = row.get("correct", "").strip()

            if not m or not c:
                continue

            # Filter: similarity too low = semantic substitution
            sim = similarity(m, c)
            if sim < 0.4:
                removed_semantic += 1
                continue

            # Filter: code identifiers
            if code_pattern.search(m) or code_pattern.search(c):
                removed_code += 1
                continue

            # Filter: contains underscore (code variable names)
            if '_' in m or '_' in c:
                removed_code += 1
                continue

            # Filter: offensive outputs
            if c.lower() in OFFENSIVE_OUTPUTS:
                removed_offensive += 1
                continue

            rows.append(row)

    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    total = removed_semantic + removed_code + removed_offensive
    print(f"  GitHub: removed {removed_semantic} semantic subs, "
          f"{removed_code} code IDs, {removed_offensive} offensive ({total} total)")
    return total


# ---------------------------------------------------------------------------
# Phase 2: Clean synthetic stories
# ---------------------------------------------------------------------------

def clean_synthetic_stories():
    """Fix </think> tag contamination and markdown in generated stories."""
    fixed_think = 0
    fixed_markdown = 0

    for age_band in ["young", "middle", "teen"]:
        band_dir = STORY_DIR / age_band
        if not band_dir.exists():
            continue

        for json_file in sorted(band_dir.glob("stories_batch_*.json")):
            with open(json_file) as f:
                stories = json.load(f)

            modified = False
            for story in stories:
                text = story.get("text", "")

                # Fix </think> contamination: keep only content after last </think>
                if "</think>" in text:
                    text = text.split("</think>")[-1].strip()
                    story["text"] = text
                    fixed_think += 1
                    modified = True

                # Also handle unclosed <think>
                if "<think>" in text:
                    parts = text.split("<think>")
                    clean_parts = [p.strip() for p in parts if len(p.strip()) > 50]
                    if clean_parts:
                        text = clean_parts[-1]
                    else:
                        text = parts[0].strip()
                    story["text"] = text
                    fixed_think += 1
                    modified = True

                # Strip markdown bold
                if "**" in text:
                    text = text.replace("**", "")
                    story["text"] = text
                    fixed_markdown += 1
                    modified = True

            if modified:
                with open(json_file, 'w') as f:
                    json.dump(stories, f, indent=2)

    print(f"  Stories: fixed {fixed_think} <think> contaminations, "
          f"{fixed_markdown} markdown artifacts")
    return fixed_think + fixed_markdown


# ---------------------------------------------------------------------------
# Phase 3: Post-process train/eval JSONL
# ---------------------------------------------------------------------------

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(data, path):
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def clean_example(ex):
    """Clean a single example. Returns None to remove it."""
    inp = ex.get("input", "")
    out = ex.get("output", "")
    instruction = ex.get("instruction", "")

    # Remove empty
    if not inp.strip() or not out.strip():
        return None

    # Remove ? outputs
    if out.strip() == "?":
        return None

    # Strip trailing frequency numbers from word-level inputs
    if instruction == "Fix the spelling of this word.":
        inp_clean = re.sub(r'\s+\d+$', '', inp)
        if inp_clean != inp:
            ex["input"] = inp_clean
            inp = inp_clean

    # Same for CHANGES format inputs
    if "CHANGES:" in instruction:
        match = re.match(r'Fix spelling: "The (.+) is here\."', inp)
        if match:
            word = match.group(1)
            word_clean = re.sub(r'\s+\d+$', '', word)
            if word_clean != word:
                ex["input"] = f'Fix spelling: "The {word_clean} is here."'
                # Also fix the CHANGES output
                out_match = re.match(r'CHANGES: (.+)->(.+)', out)
                if out_match:
                    old_word = out_match.group(1)
                    old_clean = re.sub(r'\s+\d+$', '', old_word)
                    if old_clean != old_word:
                        ex["output"] = f"CHANGES: {old_clean}->{out_match.group(2)}"
                inp = ex["input"]
                out = ex["output"]

    # Filter word-level pairs with low similarity (semantic subs)
    if instruction == "Fix the spelling of this word.":
        sim = similarity(inp, out)
        if sim < 0.4:
            return None

    # Filter CHANGES format with low similarity
    if "CHANGES:" in out:
        match = re.match(r'CHANGES: (.+)->(.+)', out)
        if match:
            old, new = match.group(1), match.group(2)
            sim = similarity(old, new)
            if sim < 0.4:
                return None

    # Filter offensive word outputs (word-level)
    if instruction == "Fix the spelling of this word.":
        if out.lower().strip() in OFFENSIVE_OUTPUTS:
            return None

    # Filter offensive error injections (sentence-level)
    if "Fix any spelling mistakes" in instruction:
        if _has_offensive_injection(inp, out):
            return None

    # Strip markdown bold from texts
    if "**" in inp:
        ex["input"] = inp.replace("**", "")
    if "**" in out:
        ex["output"] = out.replace("**", "")

    # Remove non-English content (Chinese characters etc.)
    if re.search(r'[\u4e00-\u9fff]', inp) or re.search(r'[\u4e00-\u9fff]', out):
        return None

    # Remove single-character or empty-after-strip
    if len(ex["input"].strip()) < 2 or len(ex["output"].strip()) < 2:
        return None

    return ex


def deduplicate(data, max_copies=3):
    """Remove excessive duplicates. Keep at most max_copies of each (input, output) pair."""
    seen = Counter()
    result = []
    removed = 0

    for ex in data:
        key = (ex.get("input", ""), ex.get("output", ""))
        seen[key] += 1
        if seen[key] <= max_copies:
            result.append(ex)
        else:
            removed += 1

    print(f"  Dedup: removed {removed:,} excessive duplicates (max {max_copies} copies)")
    return result


def remove_conflicting(data):
    """Remove word-level examples where a common input maps to many different outputs."""
    # Count how many different outputs each input has (word pairs only)
    input_outputs = defaultdict(set)
    for ex in data:
        if ex.get("instruction", "") != "Fix the spelling of this word.":
            continue
        inp = ex.get("input", "").lower().strip()
        out = ex.get("output", "").lower().strip()
        if inp != out:
            input_outputs[inp].add(out)

    # Flag inputs with too many different outputs
    conflicting_inputs = {inp for inp, outs in input_outputs.items() if len(outs) > 5}

    if not conflicting_inputs:
        print("  Conflicts: none found")
        return data

    before = len(data)
    data = [
        ex for ex in data
        if not (
            ex.get("instruction", "") == "Fix the spelling of this word."
            and ex.get("input", "").lower().strip() in conflicting_inputs
            and ex.get("input", "").lower().strip() != ex.get("output", "").lower().strip()
        )
    ]
    removed = before - len(data)
    print(f"  Conflicts: removed {removed:,} word pairs with >5 outputs "
          f"({len(conflicting_inputs)} conflicting inputs)")
    return data


def remove_conflicting_changes(data):
    """Also remove conflicting CHANGES-format examples."""
    input_outputs = defaultdict(set)
    for ex in data:
        if "CHANGES:" not in ex.get("output", ""):
            continue
        match = re.match(r'CHANGES: (.+)->(.+)', ex.get("output", ""))
        if match:
            old = match.group(1).lower().strip()
            new = match.group(2).lower().strip()
            if old != new:
                input_outputs[old].add(new)

    conflicting = {inp for inp, outs in input_outputs.items() if len(outs) > 5}

    if not conflicting:
        print("  CHANGES conflicts: none found")
        return data

    before = len(data)
    data_out = []
    for ex in data:
        if "CHANGES:" in ex.get("output", ""):
            match = re.match(r'CHANGES: (.+)->(.+)', ex.get("output", ""))
            if match and match.group(1).lower().strip() in conflicting:
                continue
        data_out.append(ex)

    removed = before - len(data_out)
    print(f"  CHANGES conflicts: removed {removed:,} ({len(conflicting)} conflicting inputs)")
    return data_out


def remove_overlap(train, eval_data):
    """Remove examples from eval that also appear in train."""
    train_keys = set()
    for ex in train:
        key = (ex.get("input", ""), ex.get("output", ""))
        train_keys.add(key)

    before = len(eval_data)
    clean_eval = [
        ex for ex in eval_data
        if (ex.get("input", ""), ex.get("output", "")) not in train_keys
    ]

    removed = before - len(clean_eval)
    print(f"  Overlap: removed {removed:,} eval examples also in train "
          f"({removed/before*100:.1f}%)")
    return clean_eval


def convert_to_chat_format(examples):
    """Convert instruction format to chat format."""
    chat_examples = []
    for ex in examples:
        chat_examples.append({
            "messages": [
                {"role": "system",
                 "content": "You are a spelling correction assistant. "
                            "Fix spelling mistakes accurately."},
                {"role": "user",
                 "content": f"{ex['instruction']}\n\n{ex['input']}"},
                {"role": "assistant",
                 "content": ex['output']}
            ]
        })
    return chat_examples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("TRAINING DATA CLEANUP")
    print("=" * 60)

    # --- Phase 1: Clean source CSVs ---
    print("\n--- Phase 1: Clean source data ---")
    clean_holbrook()
    clean_github_typos()

    # --- Phase 2: Clean synthetic stories ---
    print("\n--- Phase 2: Clean synthetic stories ---")
    clean_synthetic_stories()

    # --- Phase 3: Re-run combine_all.py ---
    print("\n--- Phase 3: Re-combine source data ---")
    import subprocess
    result = subprocess.run(
        [sys.executable, str(DATA_DIR / "combine_all.py")],
        capture_output=True, text=True, cwd=str(DATA_DIR)
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"ERROR: combine_all.py failed:\n{result.stderr}")
        return

    # --- Phase 4: Re-run error injection pipeline ---
    print("\n--- Phase 4: Re-run error injection pipeline ---")
    result = subprocess.run(
        [sys.executable, "-u", str(SCRIPT_DIR / "prepare_finetune_data.py")],
        capture_output=True, text=True, cwd=str(SCRIPT_DIR)
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"ERROR: prepare_finetune_data.py failed:\n{result.stderr}")
        return

    # --- Phase 5: Post-process train/eval ---
    print("\n--- Phase 5: Post-process train/eval ---")

    print("Loading train.jsonl...")
    train = load_jsonl(SCRIPT_DIR / "train.jsonl")
    print(f"  Loaded {len(train):,} examples")

    print("Loading eval.jsonl...")
    eval_data = load_jsonl(SCRIPT_DIR / "eval.jsonl")
    print(f"  Loaded {len(eval_data):,} examples")

    # Clean individual examples
    print("\nCleaning individual examples...")
    train_clean = []
    eval_clean = []
    train_removed = 0
    eval_removed = 0

    for ex in train:
        cleaned = clean_example(ex)
        if cleaned:
            train_clean.append(cleaned)
        else:
            train_removed += 1

    for ex in eval_data:
        cleaned = clean_example(ex)
        if cleaned:
            eval_clean.append(cleaned)
        else:
            eval_removed += 1

    print(f"  Train: removed {train_removed:,} bad examples")
    print(f"  Eval: removed {eval_removed:,} bad examples")

    # Remove conflicting word pairs
    print("\nRemoving conflicting examples...")
    train_clean = remove_conflicting(train_clean)
    train_clean = remove_conflicting_changes(train_clean)

    # Deduplicate
    print("\nDeduplicating...")
    train_clean = deduplicate(train_clean, max_copies=3)
    eval_clean = deduplicate(eval_clean, max_copies=1)

    # Remove train/eval overlap
    print("\nRemoving train/eval overlap...")
    eval_clean = remove_overlap(train_clean, eval_clean)

    # Save cleaned data
    print("\nSaving cleaned data...")
    save_jsonl(train_clean, SCRIPT_DIR / "train.jsonl")
    save_jsonl(eval_clean, SCRIPT_DIR / "eval.jsonl")

    # Also save chat format
    print("Generating chat format...")
    train_chat = convert_to_chat_format(train_clean)
    eval_chat = convert_to_chat_format(eval_clean)
    save_jsonl(train_chat, SCRIPT_DIR / "train_chat.jsonl")
    save_jsonl(eval_chat, SCRIPT_DIR / "eval_chat.jsonl")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("CLEANUP COMPLETE")
    print("=" * 60)
    print(f"  Train: {len(train_clean):,} examples")
    print(f"  Eval:  {len(eval_clean):,} examples")
    print(f"  Total: {len(train_clean) + len(eval_clean):,}")

    removed_train = len(train) - len(train_clean)
    removed_eval = len(eval_data) - len(eval_clean)
    print(f"\n  Removed from train: {removed_train:,} ({removed_train/len(train)*100:.1f}%)")
    print(f"  Removed from eval:  {removed_eval:,} ({removed_eval/len(eval_data)*100:.1f}%)")

    # Quick sanity check
    print("\n--- Sanity check ---")
    pos = sum(1 for ex in train_clean if ex["input"] != ex["output"])
    neg = sum(1 for ex in train_clean if ex["input"] == ex["output"])
    print(f"  Positive (has errors): {pos:,} ({pos/len(train_clean)*100:.1f}%)")
    print(f"  Negative (identity):   {neg:,} ({neg/len(train_clean)*100:.1f}%)")


if __name__ == "__main__":
    main()
