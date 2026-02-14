#!/usr/bin/env python3
"""
Convert assembled training data to chat/instruction format for fine-tuning.

Outputs JSONL files with system/user/assistant message format compatible
with HuggingFace transformers and common fine-tuning frameworks.

Two output formats:
1. Chat format (messages array) - for chat-style fine-tuning
2. Instruction format (instruction/input/output) - for instruction tuning
"""

import json
import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).parent

SYSTEM_PROMPT = (
    "You are a spelling correction assistant. Fix only spelling and grammar errors. "
    "Do not change meaning, names, or correct text. If the text is already correct, "
    "return it unchanged."
)

SYSTEM_PROMPT_DETAILED = (
    "You are a spelling correction assistant for children's writing. "
    "Fix only spelling and grammar errors. Preserve the child's voice and meaning. "
    "Do not change proper nouns, names, or intentional words. "
    "If the text is already correct, return it unchanged."
)


# ---------------------------------------------------------------------------
# Format converters
# ---------------------------------------------------------------------------

def to_chat_format(pair: dict, detailed_system: bool = False) -> dict:
    """
    Convert a training pair to chat message format.

    Output:
    {
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "<corrupted text>"},
            {"role": "assistant", "content": "<correct text>"}
        ]
    }
    """
    system = SYSTEM_PROMPT_DETAILED if detailed_system else SYSTEM_PROMPT
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": pair["input"]},
            {"role": "assistant", "content": pair["target"]},
        ]
    }


def to_instruction_format(pair: dict) -> dict:
    """
    Convert a training pair to instruction format.

    Output:
    {
        "instruction": "Fix any spelling or grammar errors in this text.",
        "input": "<corrupted text>",
        "output": "<correct text>"
    }
    """
    return {
        "instruction": "Fix any spelling or grammar errors in this text. If there are no errors, return the text unchanged.",
        "input": pair["input"],
        "output": pair["target"],
    }


def to_completion_format(pair: dict) -> dict:
    """
    Convert to simple prompt-completion format.

    Output:
    {
        "prompt": "Correct: <corrupted text>\n\nCorrected:",
        "completion": " <correct text>"
    }
    """
    return {
        "prompt": f"Correct: {pair['input']}\n\nCorrected:",
        "completion": f" {pair['target']}",
    }


# ---------------------------------------------------------------------------
# Processing pipeline
# ---------------------------------------------------------------------------

def load_pairs(filepath: Path) -> list[dict]:
    """Load training pairs from JSONL."""
    pairs = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


def convert_file(
    input_path: Path,
    output_path: Path,
    fmt: str = "chat",
    detailed_system: bool = False,
) -> int:
    """Convert a JSONL file to the specified format."""
    pairs = load_pairs(input_path)

    with open(output_path, "w") as out:
        for pair in pairs:
            if fmt == "chat":
                converted = to_chat_format(pair, detailed_system)
            elif fmt == "instruction":
                converted = to_instruction_format(pair)
            elif fmt == "completion":
                converted = to_completion_format(pair)
            else:
                raise ValueError(f"Unknown format: {fmt}")

            out.write(json.dumps(converted) + "\n")

    return len(pairs)


def convert_bob_eval(
    input_path: Path,
    output_path: Path,
    fmt: str = "chat",
) -> int:
    """Convert Bob story eval set to the specified format."""
    with open(input_path) as f:
        pairs = json.load(f)

    with open(output_path, "w") as out:
        for pair in pairs:
            if fmt == "chat":
                converted = to_chat_format(pair)
            elif fmt == "instruction":
                converted = to_instruction_format(pair)
            else:
                converted = to_completion_format(pair)
            out.write(json.dumps(converted) + "\n")

    return len(pairs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert training data to fine-tuning format"
    )
    parser.add_argument(
        "--format", choices=["chat", "instruction", "completion"],
        default="chat", help="Output format (default: chat)"
    )
    parser.add_argument(
        "--input-dir", type=str, default=None,
        help="Directory with synthetic_train.jsonl and synthetic_eval.jsonl"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: same as input)"
    )
    parser.add_argument(
        "--detailed-system", action="store_true",
        help="Use detailed system prompt mentioning children's writing"
    )
    parser.add_argument(
        "--suffix", type=str, default="",
        help="Suffix for output filenames (e.g., '_v2')"
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir) if args.input_dir else _DATA_DIR
    output_dir = Path(args.output_dir) if args.output_dir else input_dir

    fmt = args.format
    suffix = args.suffix

    print(f"Converting to {fmt} format")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print()

    # Convert training set
    train_in = input_dir / "synthetic_train.jsonl"
    if train_in.exists():
        train_out = output_dir / f"synthetic_train_{fmt}{suffix}.jsonl"
        n = convert_file(train_in, train_out, fmt, args.detailed_system)
        print(f"Train: {n} pairs -> {train_out}")

    # Convert eval set
    eval_in = input_dir / "synthetic_eval.jsonl"
    if eval_in.exists():
        eval_out = output_dir / f"synthetic_eval_{fmt}{suffix}.jsonl"
        n = convert_file(eval_in, eval_out, fmt, args.detailed_system)
        print(f"Eval:  {n} pairs -> {eval_out}")

    # Convert Bob story eval
    bob_in = input_dir / "bob_story_eval.json"
    if bob_in.exists():
        bob_out = output_dir / f"bob_story_eval_{fmt}{suffix}.jsonl"
        n = convert_bob_eval(bob_in, bob_out, fmt)
        print(f"Bob:   {n} pairs -> {bob_out}")

    print("\nDone!")


if __name__ == "__main__":
    main()
