#!/usr/bin/env python3
"""
Fine-tune Qwen3 models for spelling correction with LoRA/QLoRA.
Designed for H100 80GB with 24-hour time limit.

Supports full fine-tuning (small models) and LoRA/QLoRA (large models).

Usage:
    python finetune_qwen3.py --model Qwen/Qwen3-1.7B
    python finetune_qwen3.py --model Qwen/Qwen3-4B --lora
    python finetune_qwen3.py --model Qwen/Qwen3-32B --qlora
    python finetune_qwen3.py --model all --lora
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
    BitsAndBytesConfig,
    EarlyStoppingCallback,
)
from trl import SFTTrainer, SFTConfig

# Qwen3 models to train
MODELS = [
    "Qwen/Qwen3-0.6B",      # ~400MB quantized - low-end devices
    "Qwen/Qwen3-1.7B",      # ~1GB quantized - default for most
    "Qwen/Qwen3-4B",        # ~2.5GB quantized - good balance
    "Qwen/Qwen3-8B",        # ~5GB quantized - premium quality
    "Qwen/Qwen3-14B",       # ~9GB quantized - high quality
    "Qwen/Qwen3-32B",       # ~20GB quantized - maximum quality (QLoRA only)
]

# LoRA configuration per model size
LORA_CONFIGS = {
    "0.6B":  {"r": 64, "alpha": 128, "dropout": 0.05},
    "1.7B":  {"r": 64, "alpha": 128, "dropout": 0.05},
    "4B":    {"r": 64, "alpha": 128, "dropout": 0.05},
    "8B":    {"r": 64, "alpha": 128, "dropout": 0.05},
    "14B":   {"r": 32, "alpha": 64,  "dropout": 0.05},
    "32B":   {"r": 32, "alpha": 64,  "dropout": 0.05},
}

# Target modules for LoRA
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def get_model_size(model_name: str) -> str:
    """Extract model size string from model name."""
    for size in ["0.6B", "1.7B", "4B", "8B", "14B", "32B"]:
        if size in model_name:
            return size
    return "4B"  # default


def load_jsonl(path: Path) -> list[dict]:
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
        return (f"### Instruction:\n{example['instruction']}\n\n"
                f"### Input:\n{example['input']}\n\n"
                f"### Response:\n{example['output']}")


def prepare_dataset(data: list[dict], tokenizer) -> Dataset:
    """Prepare dataset for training."""
    formatted = [{"text": format_prompt(ex, tokenizer)} for ex in data]
    return Dataset.from_list(formatted)


def create_lora_config(model_size: str):
    """Create PEFT LoRA configuration."""
    from peft import LoraConfig, TaskType

    config = LORA_CONFIGS.get(model_size, LORA_CONFIGS["4B"])
    return LoraConfig(
        r=config["r"],
        lora_alpha=config["alpha"],
        lora_dropout=config["dropout"],
        target_modules=LORA_TARGET_MODULES,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )


def create_qlora_config():
    """Create 4-bit quantization config for QLoRA."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def train_model(
    model_name: str,
    train_data: list[dict],
    eval_data: list[dict],
    output_dir: Path,
    num_epochs: int = 2,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    max_seq_length: int = 256,
    use_lora: bool = False,
    use_qlora: bool = False,
    early_stopping_patience: int = 3,
):
    """Fine-tune a single model with optional LoRA/QLoRA."""
    model_size = get_model_size(model_name)

    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"Mode: {'QLoRA (4-bit)' if use_qlora else 'LoRA' if use_lora else 'Full fine-tuning'}")
    print(f"{'='*60}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with appropriate quantization
    model_kwargs = {
        "trust_remote_code": True,
        "attn_implementation": "sdpa",
    }

    if use_qlora:
        model_kwargs["quantization_config"] = create_qlora_config()
        model_kwargs["torch_dtype"] = torch.bfloat16
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    # Enable gradient checkpointing for memory efficiency with LoRA
    if use_lora or use_qlora:
        model.gradient_checkpointing_enable()
        if use_qlora:
            from peft import prepare_model_for_kbit_training
            model = prepare_model_for_kbit_training(model)

    # Prepare datasets
    train_dataset = prepare_dataset(train_data, tokenizer)
    eval_dataset = prepare_dataset(eval_data, tokenizer)

    print(f"Train examples: {len(train_dataset)}")
    print(f"Eval examples: {len(eval_dataset)}")

    # Model-specific output directory
    model_short_name = model_name.split("/")[-1]
    suffix = "-lora" if (use_lora or use_qlora) else ""
    model_output_dir = output_dir / f"{model_short_name}{suffix}"

    # Adjust batch size and grad accumulation for model size
    if model_size in ("32B",):
        batch_size = 2
        gradient_accumulation_steps = 16
    elif model_size in ("14B",):
        batch_size = 2
        gradient_accumulation_steps = 16
    elif model_size in ("8B",):
        batch_size = 4
        gradient_accumulation_steps = 8
    elif model_size in ("4B",):
        batch_size = 6
        gradient_accumulation_steps = 6
    else:
        gradient_accumulation_steps = 4

    # LoRA learning rate is typically higher
    if use_lora or use_qlora:
        learning_rate = learning_rate * 5  # 1e-4 default for LoRA

    # Training config
    training_args = SFTConfig(
        output_dir=str(model_output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=0.05,  # Increased from 0.01 for regularization
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=250,  # More frequent eval
        save_strategy="steps",
        save_steps=250,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        dataloader_num_workers=4,
        report_to="none",
        max_length=max_seq_length,
        packing=True,
    )

    # Build trainer kwargs
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "processing_class": tokenizer,
        "callbacks": [EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
    }

    # Add LoRA config if using PEFT
    if use_lora or use_qlora:
        peft_config = create_lora_config(model_size)
        trainer_kwargs["peft_config"] = peft_config

    # Trainer
    trainer = SFTTrainer(**trainer_kwargs)

    # Train
    print("\nStarting training...")
    trainer.train()

    # Save final model
    print(f"\nSaving model to {model_output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(model_output_dir)

    # For LoRA, also merge and save full model
    if use_lora or use_qlora:
        print("Merging LoRA weights...")
        merged_dir = output_dir / f"{model_short_name}-merged"
        try:
            merged_model = trainer.model.merge_and_unload()
            merged_model.save_pretrained(merged_dir)
            tokenizer.save_pretrained(merged_dir)
            print(f"  Merged model saved to {merged_dir}")
        except Exception as e:
            print(f"  Warning: Could not merge LoRA weights: {e}")
            print(f"  LoRA adapter saved to {model_output_dir}")

    # Save training metrics
    metrics = trainer.state.log_history
    with open(model_output_dir / "training_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"Done training {model_name}")

    # Free GPU memory
    del model, trainer
    torch.cuda.empty_cache()

    return model_output_dir


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3 spelling correction models")
    parser.add_argument("--model", type=str, help="Specific model to train (or 'all')")
    parser.add_argument("--output-dir", type=str, default="./outputs_qwen3_v2", help="Output directory")
    parser.add_argument("--train-file", type=str, default="train.jsonl", help="Training data file")
    parser.add_argument("--eval-file", type=str, default="eval.jsonl", help="Eval data file")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs (default 2 for anti-overfitting)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per device")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--lora", action="store_true", help="Use LoRA fine-tuning")
    parser.add_argument("--qlora", action="store_true", help="Use QLoRA (4-bit) fine-tuning")
    parser.add_argument("--early-stopping", type=int, default=3,
                       help="Early stopping patience (eval steps without improvement)")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load training data
    print("Loading training data...")
    train_data = load_jsonl(script_dir / args.train_file)
    eval_data = load_jsonl(script_dir / args.eval_file)
    print(f"Loaded {len(train_data)} train, {len(eval_data)} eval examples")

    # Determine which models to train
    if args.model and args.model != "all":
        models_to_train = [args.model]
    else:
        models_to_train = MODELS

    # QLoRA implies LoRA
    use_qlora = args.qlora
    use_lora = args.lora or use_qlora

    # For 14B+ models, force LoRA if not already set
    for model_name in models_to_train:
        size = get_model_size(model_name)
        model_lora = use_lora
        model_qlora = use_qlora

        if size in ("14B", "32B") and not model_lora:
            print(f"Note: Forcing LoRA for {model_name} (too large for full fine-tuning on single GPU)")
            model_lora = True
        if size == "32B" and not model_qlora:
            print(f"Note: Forcing QLoRA for {model_name} (too large for bf16 LoRA on single GPU)")
            model_qlora = True

    # Train each model
    trained_models = []
    for model_name in models_to_train:
        size = get_model_size(model_name)
        model_lora = use_lora or size in ("14B", "32B")
        model_qlora = use_qlora or size == "32B"

        try:
            model_dir = train_model(
                model_name=model_name,
                train_data=train_data,
                eval_data=eval_data,
                output_dir=output_dir,
                num_epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                use_lora=model_lora,
                use_qlora=model_qlora,
                early_stopping_patience=args.early_stopping,
            )
            trained_models.append((model_name, model_dir))
        except Exception as e:
            print(f"ERROR training {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    for model_name, model_dir in trained_models:
        print(f"  {model_name} -> {model_dir}")


if __name__ == "__main__":
    main()
