#!/usr/bin/env python3
"""
Evaluate fine-tuned Qwen3 spelling correction models.

Runs inference on eval set and computes:
- Exact match accuracy (output == expected)
- Error detection precision/recall (did it correctly identify/fix errors?)
- Inference speed (tokens/second)

Usage:
    python evaluate_models.py --models-dir outputs_qwen3 --eval-file eval.jsonl
    python evaluate_models.py --models-dir outputs_qwen3 --eval-file eval.jsonl --max-examples 2000
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def strip_thinking(text: str) -> str:
    """Remove Qwen3 <think>...</think> blocks from generated text."""
    # Remove <think>...</think> blocks (greedy, handles multiline)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return text.strip()


def load_eval_data(path: Path, max_examples: int = 0) -> list[dict]:
    """Load eval JSONL, optionally limiting count."""
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
            if max_examples and len(data) >= max_examples:
                break
    return data


def evaluate_model(model_dir: str, eval_data: list[dict], batch_size: int = 32) -> dict:
    """Evaluate a single model on the eval set."""
    model_name = Path(model_dir).name
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*60}")

    # Load model and tokenizer
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    ).cuda().eval()

    # Metrics
    exact_matches = 0
    true_positives = 0  # correctly fixed an error
    false_positives = 0  # changed text that had no error
    false_negatives = 0  # missed an error (output == input when input != expected)
    true_negatives = 0   # correctly left unchanged
    total = 0
    total_tokens = 0
    total_time = 0.0

    # Track error types
    results_by_type = {
        "has_error": {"correct": 0, "total": 0},
        "no_error": {"correct": 0, "total": 0},
    }

    print(f"Running inference on {len(eval_data)} examples...")
    for i, example in enumerate(eval_data):
        if (i + 1) % 500 == 0:
            pct = (i + 1) / len(eval_data) * 100
            speed = total_tokens / total_time if total_time > 0 else 0
            print(f"  [{i+1}/{len(eval_data)}] ({pct:.1f}%) - "
                  f"Exact match: {exact_matches/(i+1)*100:.1f}% - "
                  f"{speed:.0f} tok/s")

        input_text = example["input"]
        expected_output = example["output"]
        has_error = (input_text != expected_output)

        # Build prompt using chat template (disable thinking mode if supported)
        messages = [
            {"role": "system", "content": "You are a spelling correction assistant."},
            {"role": "user", "content": f"{example['instruction']}\n\n{input_text}"},
        ]
        try:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            # Older models (SmolLM2, Qwen2.5) don't support enable_thinking
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to("cuda")
        prompt_len = inputs["input_ids"].shape[1]

        # Generate (extra tokens in case model still produces thinking)
        max_new_tokens = min(len(tokenizer.encode(expected_output)) + 50, 512)

        start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # greedy for eval
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.time() - start

        # Decode only the generated part, strip thinking tags
        generated_ids = outputs[0][prompt_len:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        generated_text = strip_thinking(generated_text)

        total_tokens += len(generated_ids)
        total_time += elapsed
        total += 1

        # Debug: print first 3 examples to verify output format
        if i < 3:
            print(f"  Example {i}: input='{input_text[:60]}...'")
            print(f"    expected: '{expected_output[:60]}...'")
            print(f"    got:      '{generated_text[:60]}...'")
            print(f"    match: {generated_text == expected_output}")

        # Exact match
        is_match = (generated_text == expected_output)
        if is_match:
            exact_matches += 1

        # Error detection metrics
        if has_error:
            results_by_type["has_error"]["total"] += 1
            if generated_text == expected_output:
                true_positives += 1
                results_by_type["has_error"]["correct"] += 1
            elif generated_text == input_text:
                false_negatives += 1  # missed the error entirely
            # else: attempted fix but wrong
        else:
            results_by_type["no_error"]["total"] += 1
            if generated_text == expected_output:
                true_negatives += 1
                results_by_type["no_error"]["correct"] += 1
            else:
                false_positives += 1  # changed correct text

    # Compute final metrics
    speed = total_tokens / total_time if total_time > 0 else 0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    metrics = {
        "model": model_name,
        "total_examples": total,
        "exact_match_accuracy": exact_matches / total if total > 0 else 0,
        "error_detection": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "true_negatives": true_negatives,
            "false_negatives": false_negatives,
        },
        "has_error_accuracy": (
            results_by_type["has_error"]["correct"] / results_by_type["has_error"]["total"]
            if results_by_type["has_error"]["total"] > 0 else 0
        ),
        "no_error_accuracy": (
            results_by_type["no_error"]["correct"] / results_by_type["no_error"]["total"]
            if results_by_type["no_error"]["total"] > 0 else 0
        ),
        "inference_speed_tokens_per_sec": speed,
        "total_inference_time_sec": total_time,
        "total_tokens_generated": total_tokens,
    }

    # Print summary
    print(f"\n--- {model_name} Results ---")
    print(f"  Exact match accuracy: {metrics['exact_match_accuracy']*100:.2f}%")
    print(f"  Has-error accuracy:   {metrics['has_error_accuracy']*100:.2f}%")
    print(f"  No-error accuracy:    {metrics['no_error_accuracy']*100:.2f}%")
    print(f"  Precision: {precision*100:.2f}%  Recall: {recall*100:.2f}%  F1: {f1*100:.2f}%")
    print(f"  Speed: {speed:.0f} tokens/sec")
    print(f"  Total time: {total_time:.1f}s")

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    return metrics


def evaluate_real_world(model_dir: str, stories: list[dict]) -> dict:
    """Evaluate a model on real-world writing samples."""
    model_name = Path(model_dir).name
    print(f"\n--- Real-World Evaluation: {model_name} ---")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="sdpa",
    ).cuda().eval()

    results = []
    for story in stories:
        story_results = {"name": story["name"], "examples": []}

        for example in story["examples"]:
            input_text = example["input"]
            expected = example["output"]
            instruction = "Fix any spelling mistakes in the following text. If there are no mistakes, return the text unchanged."

            messages = [
                {"role": "system", "content": "You are a spelling correction assistant."},
                {"role": "user", "content": f"{instruction}\n\n{input_text}"},
            ]
            try:
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to("cuda")
            prompt_len = inputs["input_ids"].shape[1]
            max_new_tokens = min(len(tokenizer.encode(expected)) + 50, 512)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=max_new_tokens,
                    do_sample=False, temperature=1.0,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            generated_ids = outputs[0][prompt_len:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            generated_text = strip_thinking(generated_text)

            is_match = (generated_text == expected)
            has_error = (input_text != expected)

            story_results["examples"].append({
                "input": input_text,
                "expected": expected,
                "generated": generated_text,
                "exact_match": is_match,
                "has_error": has_error,
                "fixed": is_match and has_error,
            })

        # Compute story-level metrics
        exs = story_results["examples"]
        total = len(exs)
        exact = sum(1 for e in exs if e["exact_match"])
        with_error = [e for e in exs if e["has_error"]]
        fixed = sum(1 for e in with_error if e["fixed"])
        no_error = [e for e in exs if not e["has_error"]]
        preserved = sum(1 for e in no_error if e["exact_match"])

        story_results["metrics"] = {
            "total": total,
            "exact_match": exact / total if total > 0 else 0,
            "fix_rate": fixed / len(with_error) if with_error else 0,
            "preservation_rate": preserved / len(no_error) if no_error else 0,
        }

        print(f"  {story['name']}: "
              f"exact={story_results['metrics']['exact_match']*100:.1f}% "
              f"fix={story_results['metrics']['fix_rate']*100:.1f}% "
              f"preserve={story_results['metrics']['preservation_rate']*100:.1f}%")

        results.append(story_results)

    del model
    torch.cuda.empty_cache()

    return {"model": model_name, "stories": results}


# Real-world test stories (held out from training)
REAL_WORLD_STORIES = [
    {
        "name": "Pascal DnD Story",
        "examples": [
            {"input": "the party desided to go on a campain to the northen lands",
             "output": "the party decided to go on a campaign to the northern lands"},
            {"input": "the sitasans were afrade of the dragons",
             "output": "the citizens were afraid of the dragons"},
            {"input": "they fided a way threw the dence forest",
             "output": "they found a way through the dense forest"},
            {"input": "the rouge sneked past the gards and stole the tresure",
             "output": "the rogue sneaked past the guards and stole the treasure"},
            {"input": "our wizard casted a powerfull spell and defeted the boss",
             "output": "our wizard cast a powerful spell and defeated the boss"},
            {"input": "we travled for three days befor reching the castel",
             "output": "we traveled for three days before reaching the castle"},
            {"input": "the vilagers were greatful for our help",
             "output": "the villagers were grateful for our help"},
            {"input": "my carracter has high strangth and constitusion",
             "output": "my character has high strength and constitution"},
            {"input": "the dungen master discribed a dark and scarey cave",
             "output": "the dungeon master described a dark and scary cave"},
            {"input": "we neaded to rest beacuse everyone was exausted",
             "output": "we needed to rest because everyone was exhausted"},
        ],
    },
    {
        "name": "Bob Story Excerpts",
        "examples": [
            {"input": "they skremed and ran away from the monstar",
             "output": "they screamed and ran away from the monster"},
            {"input": "he jumed off the belkany and landed in the garden",
             "output": "he jumped off the balcony and landed in the garden"},
            {"input": "the alien was floting in the sky above the bilding",
             "output": "the alien was floating in the sky above the building"},
            {"input": "she fented when she saw the hights of the clif",
             "output": "she fainted when she saw the heights of the cliff"},
            {"input": "he gabed the rope and started climing the wal",
             "output": "he grabbed the rope and started climbing the wall"},
        ],
    },
    {
        "name": "Diverse Real-World",
        "examples": [
            # Identity examples (should not be changed)
            {"input": "I went to the store and bought some bread.",
             "output": "I went to the store and bought some bread."},
            {"input": "My friend Jake plays Minecraft every day after school.",
             "output": "My friend Jake plays Minecraft every day after school."},
            # Challenging misspellings
            {"input": "I went to teh stor to by sum bred",
             "output": "I went to the store to buy some bread"},
            {"input": "my frend sed he wood come to my hous tomorow",
             "output": "my friend said he would come to my house tomorrow"},
            {"input": "the teecher gave us alot of homwork today",
             "output": "the teacher gave us a lot of homework today"},
        ],
    },
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", type=str, default="outputs_qwen3")
    parser.add_argument("--eval-file", type=str, default="eval.jsonl")
    parser.add_argument("--max-examples", type=int, default=5000,
                        help="Max eval examples (0 = all). Default 5000 for speed.")
    parser.add_argument("--models", type=str, nargs="*",
                        help="Specific model names to evaluate (e.g. Qwen3-0.6B Qwen3-4B)")
    parser.add_argument("--real-world-only", action="store_true",
                        help="Only run real-world evaluation (skip synthetic)")
    parser.add_argument("--real-world-file", type=str, default=None,
                        help="Path to custom real-world test stories (JSON)")
    args = parser.parse_args()

    models_dir = Path(args.models_dir)

    # Find models to evaluate
    if args.models:
        model_dirs = [models_dir / m for m in args.models]
    else:
        model_dirs = sorted([d for d in models_dir.iterdir() if d.is_dir() and (d / "config.json").exists()])

    print(f"Models to evaluate: {[d.name for d in model_dirs]}")

    # Synthetic evaluation
    all_metrics = []
    if not args.real_world_only:
        eval_data = load_eval_data(Path(args.eval_file), args.max_examples)
        print(f"Loaded {len(eval_data)} eval examples")

        for model_dir in model_dirs:
            if not (model_dir / "config.json").exists():
                print(f"Skipping {model_dir} - no config.json")
                continue
            metrics = evaluate_model(str(model_dir), eval_data)
            all_metrics.append(metrics)

    # Real-world evaluation
    stories = REAL_WORLD_STORIES
    if args.real_world_file:
        with open(args.real_world_file) as f:
            stories = json.load(f)

    all_real_world = []
    for model_dir in model_dirs:
        if not (model_dir / "config.json").exists():
            continue
        rw_metrics = evaluate_real_world(str(model_dir), stories)
        all_real_world.append(rw_metrics)

    # Save results
    output_file = models_dir / "eval_results.json"
    with open(output_file, "w") as f:
        json.dump({"synthetic": all_metrics, "real_world": all_real_world}, f, indent=2)
    print(f"\nResults saved to {output_file}")

    # Print synthetic comparison table
    if all_metrics:
        print(f"\n{'='*80}")
        print("SYNTHETIC EVAL COMPARISON")
        print(f"{'='*80}")
        print(f"{'Model':<15} {'Exact Match':>12} {'Error Fix':>10} {'No-Err Keep':>12} "
              f"{'Precision':>10} {'Recall':>8} {'F1':>8} {'Tok/s':>8}")
        print("-" * 80)
        for m in all_metrics:
            print(f"{m['model']:<15} "
                  f"{m['exact_match_accuracy']*100:>11.2f}% "
                  f"{m['has_error_accuracy']*100:>9.2f}% "
                  f"{m['no_error_accuracy']*100:>11.2f}% "
                  f"{m['error_detection']['precision']*100:>9.2f}% "
                  f"{m['error_detection']['recall']*100:>7.2f}% "
                  f"{m['error_detection']['f1']*100:>7.2f}% "
                  f"{m['inference_speed_tokens_per_sec']:>7.0f}")

    # Print real-world comparison table
    if all_real_world:
        print(f"\n{'='*80}")
        print("REAL-WORLD EVAL COMPARISON")
        print(f"{'='*80}")
        for rw in all_real_world:
            print(f"\n  {rw['model']}:")
            for story in rw["stories"]:
                m = story["metrics"]
                print(f"    {story['name']:<25} "
                      f"exact={m['exact_match']*100:5.1f}%  "
                      f"fix={m['fix_rate']*100:5.1f}%  "
                      f"preserve={m['preservation_rate']*100:5.1f}%")


if __name__ == "__main__":
    main()
