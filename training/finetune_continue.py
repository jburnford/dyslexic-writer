#!/usr/bin/env python3
"""
Continue fine-tuning an already trained model on new data.
Uses lower learning rate to avoid catastrophic forgetting.

Usage:
    python finetune_continue.py --base-model ./outputs/SmolLM2-1.7B-Instruct \
                                --train-file proper_nouns_train.jsonl
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig


def load_jsonl(path: Path) -> list[dict]:
    """Load JSONL file."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def format_prompt(example: dict, tokenizer) -> str:
    """Format example for training."""
    if hasattr(tokenizer, 'apply_chat_template'):
        messages = [
            {"role": "system", "content": "You are a spelling correction assistant."},
            {"role": "user", "content": f"{example['instruction']}\n\n{example['input']}"},
            {"role": "assistant", "content": example['output']}
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False)
    else:
        return f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"


def prepare_dataset(data: list[dict], tokenizer) -> Dataset:
    """Prepare dataset for training."""
    formatted = [{"text": format_prompt(ex, tokenizer)} for ex in data]
    return Dataset.from_list(formatted)


def main():
    parser = argparse.ArgumentParser(description="Continue fine-tuning on new data")
    parser.add_argument("--base-model", type=str, required=True, help="Path to already fine-tuned model")
    parser.add_argument("--output-dir", type=str, default="./outputs/continued", help="Output directory")
    parser.add_argument("--train-file", type=str, default="proper_nouns_train.jsonl", help="Training data")
    parser.add_argument("--eval-file", type=str, default="proper_nouns_eval.jsonl", help="Eval data")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate (lower for continue training)")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading base model from {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    # Load new training data
    print(f"Loading training data from {args.train_file}...")
    train_data = load_jsonl(script_dir / args.train_file)
    eval_data = load_jsonl(script_dir / args.eval_file)

    train_dataset = prepare_dataset(train_data, tokenizer)
    eval_dataset = prepare_dataset(eval_data, tokenizer)

    print(f"Train examples: {len(train_dataset)}")
    print(f"Eval examples: {len(eval_dataset)}")

    # Training config with lower LR
    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        learning_rate=args.lr,  # Lower LR for continue training
        weight_decay=0.01,
        warmup_ratio=0.05,  # Less warmup since model is already trained
        lr_scheduler_type="cosine",
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        dataloader_num_workers=4,
        report_to="none",
        max_length=256,
        packing=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    print(f"\nContinue training with LR={args.lr}...")
    trainer.train()

    print(f"\nSaving model to {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    # Save metrics
    metrics = trainer.state.log_history
    with open(output_dir / "training_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    print("Done!")


if __name__ == "__main__":
    main()
