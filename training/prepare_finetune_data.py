#!/usr/bin/env python3
"""
Prepare training data for fine-tuning.

Combines word pairs, sentence context, and synthetic/generated examples.
Applies error injection to clean stories and creates train/eval split.
"""

import json
import random
import sys
from pathlib import Path

SEED = 42
EVAL_RATIO = 0.1

# Add parent to path for inject_errors import
sys.path.insert(0, str(Path(__file__).parent))
from inject_errors import inject_errors_text


def load_jsonl(path: Path) -> list[dict]:
    """Load JSONL file."""
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(data: list[dict], path: Path):
    """Save to JSONL file."""
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    print(f"Saved {len(data)} examples to {path}")


def convert_to_chat_format(examples: list[dict]) -> list[dict]:
    """Convert instruction format to chat format for training."""
    chat_examples = []
    for ex in examples:
        chat_examples.append({
            "messages": [
                {"role": "system", "content": "You are a spelling correction assistant. Fix spelling mistakes accurately."},
                {"role": "user", "content": f"{ex['instruction']}\n\n{ex['input']}"},
                {"role": "assistant", "content": ex['output']}
            ]
        })
    return chat_examples


def inject_errors_into_stories(synthetic_data: list[dict]) -> list[dict]:
    """
    Process synthetic data: apply error injection to clean stories,
    pass through identity pairs unchanged.

    Returns list of instruction-format dicts.
    """
    results = []
    injected = 0
    identity = 0

    for entry in synthetic_data:
        # Identity pairs (negative examples) — already in instruction format
        if "instruction" in entry and "input" in entry and "output" in entry:
            results.append(entry)
            identity += 1
            continue

        # Clean stories — apply error injection
        text = entry.get("text", "")
        age_band = entry.get("age_band", "young")
        if not text:
            continue

        sentence_results = inject_errors_text(text, age_band=age_band, inconsistency=True)

        for sr in sentence_results:
            if sr.error_count > 0:
                results.append({
                    "instruction": "Fix any spelling mistakes in this text. If there are no mistakes, output the text unchanged.",
                    "input": sr.corrupted,
                    "output": sr.original,
                })
                injected += 1
            else:
                # No errors injected — use as identity pair
                results.append({
                    "instruction": "Fix any spelling mistakes in this text. If there are no mistakes, output the text unchanged.",
                    "input": sr.original,
                    "output": sr.original,
                })
                identity += 1

    print(f"  Error-injected sentences: {injected:,}")
    print(f"  Identity pairs: {identity:,}")
    return results


def main():
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "training-data"

    # Load both formats
    print("Loading training data...")

    # Simple word pairs
    instruction_data = load_jsonl(data_dir / "all_instruction.jsonl")
    print(f"  Loaded {len(instruction_data):,} word pair examples")

    # Sentence context (CHANGES format)
    changes_data = load_jsonl(data_dir / "all_changes.jsonl")
    print(f"  Loaded {len(changes_data):,} sentence context examples")

    # Synthetic data (generated stories + negatives)
    synthetic_path = data_dir / "synthetic_all.jsonl"
    raw_synthetic = load_jsonl(synthetic_path)
    if raw_synthetic:
        print(f"  Loaded {len(raw_synthetic):,} synthetic entries")
        print("  Applying error injection to clean stories...")
        synthetic_data = inject_errors_into_stories(raw_synthetic)
        print(f"  Total synthetic training examples: {len(synthetic_data):,}")
    else:
        synthetic_data = []
        print("  No synthetic data found (run generate_stories_local.py + combine_all.py first)")

    # Combine all
    combined = instruction_data + changes_data + synthetic_data
    print(f"\nTotal combined: {len(combined):,} examples")

    # Shuffle
    random.seed(SEED)
    random.shuffle(combined)

    # Split into train/eval
    split_idx = int(len(combined) * (1 - EVAL_RATIO))
    train_data = combined[:split_idx]
    eval_data = combined[split_idx:]

    print(f"\nSplit:")
    print(f"  Train: {len(train_data):,} examples")
    print(f"  Eval:  {len(eval_data):,} examples")

    # Save instruction format (for SFTTrainer)
    save_jsonl(train_data, script_dir / "train.jsonl")
    save_jsonl(eval_data, script_dir / "eval.jsonl")

    # Also save chat format (for some models)
    train_chat = convert_to_chat_format(train_data)
    eval_chat = convert_to_chat_format(eval_data)
    save_jsonl(train_chat, script_dir / "train_chat.jsonl")
    save_jsonl(eval_chat, script_dir / "eval_chat.jsonl")

    print("\nDone! Created:")
    print("  - train.jsonl / eval.jsonl (instruction format)")
    print("  - train_chat.jsonl / eval_chat.jsonl (chat format)")


if __name__ == "__main__":
    main()
