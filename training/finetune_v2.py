#!/usr/bin/env python3
"""
Fine-tune small LLMs for spelling correction.
Designed for H100 80GB with 24-hour time limit.

v2: Adds gradient checkpointing and configurable grad accumulation
    for larger models (3B-7B).

Usage:
    python finetune_v2.py --model Qwen/Qwen2.5-7B-Instruct --batch-size 2 --grad-accum 16 --gradient-checkpointing
    python finetune_v2.py --model meta-llama/Llama-3.2-3B-Instruct --batch-size 8 --gradient-checkpointing
"""

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig


def load_jsonl(path: Path) -> list:
    """Load JSONL file."""
    with open(path) as f:
        return [json.loads(line) for line in f]


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
        return (
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Input:\n{example['input']}\n\n"
            f"### Response:\n{example['output']}"
        )


def prepare_dataset(data: list, tokenizer) -> Dataset:
    """Prepare dataset for training."""
    formatted = [{"text": format_prompt(ex, tokenizer)} for ex in data]
    return Dataset.from_list(formatted)


def train_model(
    model_name: str,
    train_data: list,
    eval_data: list,
    output_dir: Path,
    num_epochs: int = 3,
    batch_size: int = 8,
    grad_accum: int = 4,
    learning_rate: float = 2e-5,
    max_seq_length: int = 256,
    gradient_checkpointing: bool = False,
):
    """Fine-tune a single model."""
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"  batch_size={batch_size}, grad_accum={grad_accum}, effective={batch_size*grad_accum}")
    print(f"  gradient_checkpointing={gradient_checkpointing}")
    print(f"{'='*60}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Phi models don't support SDPA yet, fall back to eager
    attn_impl = "sdpa"
    if "phi" in model_name.lower():
        attn_impl = "eager"
        print(f"  Using attn_implementation='eager' for Phi model")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )

    # Prepare datasets
    train_dataset = prepare_dataset(train_data, tokenizer)
    eval_dataset = prepare_dataset(eval_data, tokenizer)

    print(f"Train examples: {len(train_dataset)}")
    print(f"Eval examples: {len(eval_dataset)}")

    # Model-specific output directory
    model_short_name = model_name.split("/")[-1]
    model_output_dir = output_dir / model_short_name

    # Training config
    sft_kwargs = dict(
        output_dir=str(model_output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=50,
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
        report_to="none",
        max_length=max_seq_length,
        packing=True,
    )

    if gradient_checkpointing:
        sft_kwargs["gradient_checkpointing"] = True
        sft_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}

    training_args = SFTConfig(**sft_kwargs)

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

    # Print final metrics
    final_train = [m for m in metrics if "train_loss" in m]
    final_eval = [m for m in metrics if "eval_loss" in m]
    if final_train:
        print(f"  Final train loss: {final_train[-1]['train_loss']:.4f}")
    if final_eval:
        print(f"  Final eval loss: {final_eval[-1]['eval_loss']:.4f}")

    print(f"Done training {model_name}")

    # Free GPU memory before next model
    del model, trainer
    torch.cuda.empty_cache()

    return model_output_dir


def main():
    parser = argparse.ArgumentParser(description="Fine-tune spelling correction models (v2)")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model ID")
    parser.add_argument("--output-dir", type=str, default="./outputs_v3", help="Output directory")
    parser.add_argument("--train-file", type=str, default="synthetic_train_clean_instruction.jsonl")
    parser.add_argument("--eval-file", type=str, default="synthetic_eval_clean_instruction.jsonl")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load training data
    script_dir = Path(__file__).parent
    train_path = script_dir / args.train_file
    eval_path = script_dir / args.eval_file

    # Also check current directory
    if not train_path.exists():
        train_path = Path(args.train_file)
        eval_path = Path(args.eval_file)

    print(f"Loading training data from {train_path}...")
    train_data = load_jsonl(train_path)
    eval_data = load_jsonl(eval_path)
    print(f"Loaded {len(train_data)} train, {len(eval_data)} eval examples")

    try:
        model_dir = train_model(
            model_name=args.model,
            train_data=train_data,
            eval_data=eval_data,
            output_dir=output_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            learning_rate=args.lr,
            gradient_checkpointing=args.gradient_checkpointing,
        )
        print(f"\nModel saved to: {model_dir}")
    except Exception as e:
        print(f"\nERROR training {args.model}: {e}")
        import traceback
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
