#!/usr/bin/env python3
"""
Convert proper nouns training data to instruction format and create training datasets.

Creates:
1. proper_nouns_train.jsonl - Proper nouns in instruction format (for continue training)
2. combined_train.jsonl - All data combined (for training from scratch)
"""

import json
from pathlib import Path
import random

def convert_to_instruction(data: dict) -> dict:
    """Convert {"input": ..., "output": ...} to instruction format."""
    return {
        "instruction": "Fix the spelling mistakes in this sentence. Only output the corrected sentence.",
        "input": data["input"],
        "output": data["output"]
    }

def main():
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "training-data"

    # Load proper nouns data
    print("Loading proper nouns data...")
    proper_nouns = []
    with open(data_dir / "proper_nouns_instruction.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                proper_nouns.append(json.loads(line))
    print(f"  Loaded {len(proper_nouns)} proper noun examples")

    # Convert to instruction format
    proper_nouns_instruction = [convert_to_instruction(d) for d in proper_nouns]

    # Split proper nouns into train/eval (90/10)
    random.seed(42)
    random.shuffle(proper_nouns_instruction)
    split_idx = int(len(proper_nouns_instruction) * 0.9)
    proper_train = proper_nouns_instruction[:split_idx]
    proper_eval = proper_nouns_instruction[split_idx:]

    # Save proper nouns only (for continue training - Option B)
    print(f"\nOption B: Continue training data")
    with open(script_dir / "proper_nouns_train.jsonl", 'w') as f:
        for item in proper_train:
            f.write(json.dumps(item) + '\n')
    print(f"  Saved {len(proper_train)} to proper_nouns_train.jsonl")

    with open(script_dir / "proper_nouns_eval.jsonl", 'w') as f:
        for item in proper_eval:
            f.write(json.dumps(item) + '\n')
    print(f"  Saved {len(proper_eval)} to proper_nouns_eval.jsonl")

    # Load existing training data
    print(f"\nLoading existing training data...")
    existing_train = []
    with open(script_dir / "train.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                existing_train.append(json.loads(line))
    print(f"  Loaded {len(existing_train)} existing train examples")

    existing_eval = []
    with open(script_dir / "eval.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                existing_eval.append(json.loads(line))
    print(f"  Loaded {len(existing_eval)} existing eval examples")

    # Combine all data (for training from scratch - Option A)
    combined_train = existing_train + proper_train
    combined_eval = existing_eval + proper_eval

    # Shuffle combined data
    random.shuffle(combined_train)
    random.shuffle(combined_eval)

    print(f"\nOption A: Combined training data")
    with open(script_dir / "combined_train.jsonl", 'w') as f:
        for item in combined_train:
            f.write(json.dumps(item) + '\n')
    print(f"  Saved {len(combined_train)} to combined_train.jsonl")

    with open(script_dir / "combined_eval.jsonl", 'w') as f:
        for item in combined_eval:
            f.write(json.dumps(item) + '\n')
    print(f"  Saved {len(combined_eval)} to combined_eval.jsonl")

    print(f"\n{'='*60}")
    print("Summary:")
    print(f"{'='*60}")
    print(f"Option A (from scratch): {len(combined_train)} train + {len(combined_eval)} eval")
    print(f"Option B (continue):     {len(proper_train)} train + {len(proper_eval)} eval")

if __name__ == "__main__":
    main()
