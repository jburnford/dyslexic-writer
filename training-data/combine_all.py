#!/usr/bin/env python3
"""
Combine all spelling correction datasets into unified training files.

Supports both word-level pairs (CSV) and sentence-level examples (JSON batches
from generate_stories_local.py + inject_errors.py).
"""

import json
import csv
from pathlib import Path
from collections import defaultdict

def load_csv_pairs(filepath: Path) -> list[tuple[str, str]]:
    """Load pairs from CSV file."""
    pairs = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Try different column name conventions
            m = (row.get('misspelling') or row.get('wrong') or
                 row.get('input') or '').strip()
            c = (row.get('correct') or row.get('right') or
                 row.get('label') or '').strip()
            if m and c and m.lower() != c.lower():
                pairs.append((m.lower(), c.lower()))
    return pairs

def load_jsonl_pairs(filepath: Path) -> list[tuple[str, str]]:
    """Load pairs from JSONL file."""
    pairs = []
    with open(filepath, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line)
                m = entry.get('input', '').strip()
                c = entry.get('output', '').strip()
                if m and c and m.lower() != c.lower():
                    pairs.append((m.lower(), c.lower()))
            except json.JSONDecodeError:
                continue
    return pairs

def load_story_batches(story_dir: Path) -> list[dict]:
    """Load generated story batches (JSON files from generate_stories_local.py)."""
    stories = []
    if not story_dir.exists():
        return stories
    for json_file in sorted(story_dir.glob("stories_batch_*.json")):
        with open(json_file) as f:
            batch = json.load(f)
            stories.extend(batch)
    return stories

def load_negative_batches(story_dir: Path) -> list[dict]:
    """Load negative example batches."""
    negatives = []
    if not story_dir.exists():
        return negatives
    for json_file in sorted(story_dir.glob("negatives_batch_*.json")):
        with open(json_file) as f:
            batch = json.load(f)
            negatives.extend(batch)
    return negatives

def main():
    script_dir = Path(__file__).parent
    training_dir = script_dir.parent / "training"

    all_pairs = []
    seen = set()

    # --- Word-level pairs ---
    sources = {
        'birkbeck_pairs.csv': 'Birkbeck',
        'holbrook_pairs.csv': 'Holbrook',
        'extra_misspellings.csv': 'Extra',
        'github_typos_pairs.csv': 'GitHub',
    }

    stats = defaultdict(int)

    for filename, source in sources.items():
        filepath = script_dir / filename
        if not filepath.exists():
            print(f"Skipping {filename} (not found)")
            continue

        pairs = load_csv_pairs(filepath)
        added = 0
        for m, c in pairs:
            key = (m, c)
            if key not in seen:
                seen.add(key)
                all_pairs.append((m, c, source))
                added += 1

        stats[source] = added
        print(f"{source}: {len(pairs):,} total, {added:,} unique new")

    print(f"\nTotal unique word pairs: {len(all_pairs):,}")

    # Save combined CSV
    csv_path = script_dir / "all_pairs.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['misspelling', 'correct', 'source'])
        for m, c, s in all_pairs:
            writer.writerow([m, c, s])
    print(f"Saved to {csv_path}")

    # Save combined JSONL for instruction fine-tuning
    jsonl_path = script_dir / "all_instruction.jsonl"
    with open(jsonl_path, 'w') as f:
        for m, c, _ in all_pairs:
            entry = {
                "instruction": "Fix the spelling of this word.",
                "input": m,
                "output": c
            }
            f.write(json.dumps(entry) + '\n')
    print(f"Saved to {jsonl_path}")

    # Save in CHANGES format
    changes_path = script_dir / "all_changes.jsonl"
    with open(changes_path, 'w') as f:
        for m, c, _ in all_pairs:
            entry = {
                "instruction": "Fix spelling mistakes. Output format: CHANGES: misspelled->correct",
                "input": f'Fix spelling: "The {m} is here."',
                "output": f"CHANGES: {m}->{c}"
            }
            f.write(json.dumps(entry) + '\n')
    print(f"Saved to {changes_path}")

    # --- Sentence-level data from generated stories ---
    story_dir = training_dir / "generated_stories"
    sentence_count = 0

    synthetic_path = script_dir / "synthetic_all.jsonl"
    synthetic_f = open(synthetic_path, 'w')

    for age_band in ["young", "middle", "teen"]:
        band_dir = story_dir / age_band
        stories = load_story_batches(band_dir)
        negatives = load_negative_batches(band_dir)

        if stories:
            print(f"\n{age_band} stories: {len(stories):,}")
            for story in stories:
                text = story.get("text", "")
                if not text:
                    continue
                # Stories are clean — they need error injection applied separately
                # Store as synthetic data for the injection pipeline
                entry = {
                    "text": text,
                    "age_band": age_band,
                    "genre": story.get("genre", ""),
                    "type": "clean_for_injection",
                }
                synthetic_f.write(json.dumps(entry) + '\n')
                sentence_count += 1

        if negatives:
            print(f"{age_band} negatives: {len(negatives):,}")
            for neg in negatives:
                text = neg.get("text", "")
                if not text:
                    continue
                # Negative examples are identity pairs (correct -> correct)
                entry = {
                    "instruction": "Fix any spelling mistakes in this text. If there are no mistakes, output the text unchanged.",
                    "input": text,
                    "output": text,
                }
                synthetic_f.write(json.dumps(entry) + '\n')
                sentence_count += 1

    # Also load stories from flat directory (no age band subdirs)
    flat_stories = load_story_batches(story_dir)
    flat_negatives = load_negative_batches(story_dir)
    if flat_stories:
        print(f"\nFlat stories: {len(flat_stories):,}")
        for story in flat_stories:
            text = story.get("text", "")
            if text:
                entry = {
                    "text": text,
                    "age_band": story.get("age_band", "young"),
                    "genre": story.get("genre", ""),
                    "type": "clean_for_injection",
                }
                synthetic_f.write(json.dumps(entry) + '\n')
                sentence_count += 1
    if flat_negatives:
        print(f"Flat negatives: {len(flat_negatives):,}")
        for neg in flat_negatives:
            text = neg.get("text", "")
            if text:
                entry = {
                    "instruction": "Fix any spelling mistakes in this text. If there are no mistakes, output the text unchanged.",
                    "input": text,
                    "output": text,
                }
                synthetic_f.write(json.dumps(entry) + '\n')
                sentence_count += 1

    synthetic_f.close()
    print(f"\nTotal sentence-level examples: {sentence_count:,}")
    print(f"Saved to {synthetic_path}")

    print("\nBreakdown by source:")
    for source, count in sorted(stats.items(), key=lambda x: -x[1]):
        pct = count / len(all_pairs) * 100 if all_pairs else 0
        print(f"  {source}: {count:,} ({pct:.1f}%)")

if __name__ == "__main__":
    main()
