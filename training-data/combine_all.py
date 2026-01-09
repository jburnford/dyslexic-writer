#!/usr/bin/env python3
"""
Combine all spelling correction datasets into unified training files.
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

def main():
    script_dir = Path(__file__).parent

    all_pairs = []
    seen = set()

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

    print(f"\nTotal unique pairs: {len(all_pairs):,}")

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

    print("\nBreakdown by source:")
    for source, count in sorted(stats.items(), key=lambda x: -x[1]):
        pct = count / len(all_pairs) * 100
        print(f"  {source}: {count:,} ({pct:.1f}%)")

if __name__ == "__main__":
    main()
