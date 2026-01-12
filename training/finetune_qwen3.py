#!/usr/bin/env python3
"""
Fine-tune Qwen3 models for spelling correction.
Designed for H100 80GB with 24-hour time limit.

Usage:
    python finetune_qwen3.py --model Qwen/Qwen3-1.7B
    python finetune_qwen3.py --model all
"""

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from trl import SFTTrainer, SFTConfig

# Qwen3 models to train
MODELS = [
    "Qwen/Qwen3-0.6B",      # ~400MB quantized - low-end devices
    "Qwen/Qwen3-1.7B",      # ~1GB quantized - default for most
    "Qwen/Qwen3-4B",        # ~2.5GB quantized - good balance
    "Qwen/Qwen3-8B",        # ~5GB quantized - premium quality
]


def load_jsonl(path: Path) -> list[dict]:
    """Load JSONL file."""
    with open(path) as f:
        return [json.loads(line) for line in f]


def format_prompt(example: dict, tokenizer) -> str:
    """Format example for training."""
    # Use chat template if available
    if hasattr(tokenizer, 'apply_chat_template'):
        messages = [
            {"role": "system", "content": "You are a spelling correction assistant."},
            {"role": "user", "content": f"{example['instruction']}\n\n{example['input']}"},
            {"role": "assistant", "content": example['output']}
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False)
    else:
        # Fallback to simple format
        return f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"


def prepare_dataset(data: list[dict], tokenizer) -> Dataset:
    """Prepare dataset for training."""
    formatted = [{"text": format_prompt(ex, tokenizer)} for ex in data]
    return Dataset.from_list(formatted)


def train_model(
    model_name: str,
    train_data: list[dict],
    eval_data: list[dict],
    output_dir: Path,
    num_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    max_seq_length: int = 256,
):
    """Fine-tune a single model."""
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"{'='*60}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",  # Use PyTorch native SDPA (includes Flash Attention)
    )

    # Prepare datasets
    train_dataset = prepare_dataset(train_data, tokenizer)
    eval_dataset = prepare_dataset(eval_data, tokenizer)

    print(f"Train examples: {len(train_dataset)}")
    print(f"Eval examples: {len(eval_dataset)}")

    # Model-specific output directory
    model_short_name = model_name.split("/")[-1]
    model_output_dir = output_dir / model_short_name

    # Adjust batch size for larger models
    if "8B" in model_name:
        batch_size = 4
        gradient_accumulation_steps = 8
    elif "4B" in model_name:
        batch_size = 6
        gradient_accumulation_steps = 6
    else:
        gradient_accumulation_steps = 4

    # Training config
    training_args = SFTConfig(
        output_dir=str(model_output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        dataloader_num_workers=4,
        report_to="none",  # Disable wandb etc
        max_length=max_seq_length,  # renamed from max_seq_length in trl 0.26+
        packing=True,  # Pack short sequences for efficiency
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    # Save final model
    print(f"\nSaving model to {model_output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(model_output_dir)

    # Save training metrics
    metrics = trainer.state.log_history
    with open(model_output_dir / "training_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"Done training {model_name}")
    return model_output_dir


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3 spelling correction models")
    parser.add_argument("--model", type=str, help="Specific model to train (or 'all')")
    parser.add_argument("--output-dir", type=str, default="./outputs_qwen3", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per device")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load training data
    print("Loading training data...")
    train_data = load_jsonl(script_dir / "train.jsonl")
    eval_data = load_jsonl(script_dir / "eval.jsonl")
    print(f"Loaded {len(train_data)} train, {len(eval_data)} eval examples")

    # Determine which models to train
    if args.model and args.model != "all":
        models_to_train = [args.model]
    else:
        models_to_train = MODELS

    # Train each model
    trained_models = []
    for model_name in models_to_train:
        try:
            model_dir = train_model(
                model_name=model_name,
                train_data=train_data,
                eval_data=eval_data,
                output_dir=output_dir,
                num_epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
            )
            trained_models.append((model_name, model_dir))
        except Exception as e:
            print(f"ERROR training {model_name}: {e}")
            continue

    # Summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    for model_name, model_dir in trained_models:
        print(f"  {model_name} -> {model_dir}")


if __name__ == "__main__":
    main()
