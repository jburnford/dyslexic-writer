#!/usr/bin/env python3
"""
Prepare training data for fine-tuning.
Combines both formats and creates train/eval split.
"""

import json
import random
from pathlib import Path

SEED = 42
EVAL_RATIO = 0.1

def load_jsonl(path: Path) -> list[dict]:
    """Load JSONL file."""
    with open(path) as f:
        return [json.loads(line) for line in f]


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


def main():
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "training-data"

    # Load both formats
    print("Loading training data...")

    # Simple word pairs
    instruction_data = load_jsonl(data_dir / "all_instruction.jsonl")
    print(f"  Loaded {len(instruction_data)} word pair examples")

    # Sentence context (CHANGES format)
    changes_data = load_jsonl(data_dir / "all_changes.jsonl")
    print(f"  Loaded {len(changes_data)} sentence context examples")

    # Synthetic data (Gemini 3 Pro generated)
    synthetic_path = data_dir / "synthetic_all.jsonl"
    if synthetic_path.exists():
        synthetic_data = load_jsonl(synthetic_path)
        print(f"  Loaded {len(synthetic_data)} synthetic examples")
    else:
        synthetic_data = []
        print("  No synthetic data found (run convert_synthetic_data.py first)")

    # Combine all
    combined = instruction_data + changes_data + synthetic_data
    print(f"\nTotal combined: {len(combined)} examples")

    # Shuffle
    random.seed(SEED)
    random.shuffle(combined)

    # Split into train/eval
    split_idx = int(len(combined) * (1 - EVAL_RATIO))
    train_data = combined[:split_idx]
    eval_data = combined[split_idx:]

    print(f"\nSplit:")
    print(f"  Train: {len(train_data)} examples")
    print(f"  Eval:  {len(eval_data)} examples")

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
