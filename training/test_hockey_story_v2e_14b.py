#!/usr/bin/env python3
"""Test Bob's hockey story against v2e 14B model - two prompts, greedy + 5 sampled runs each."""

import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

with open("bob_hockey_story.json") as f:
    story = json.load(f)

input_text = story["input"]
expected = story["expected"]

PROMPTS = {
    "simple": {
        "system": "You are a spelling correction assistant.",
        "user": f"Fix any spelling mistakes in this text. If there are no mistakes, output the text unchanged.\n\n{input_text}",
    },
    "dyslexia": {
        "system": "You are a spelling correction assistant helping a dyslexic student. Fix spelling and basic grammar errors only. Preserve the student's voice, sentence structure, and word choices exactly. Do not rewrite, rephrase, add punctuation beyond what's needed, or \"improve\" the writing. It should still read like a kid wrote it. Output only the corrected text with no explanation.",
        "user": f"Fix the spelling and basic grammar mistakes in this text while preserving the student's voice. The student is dyslexic and just needs help with spelling — don't make it read like an adult rewrote it. Output only the corrected text.\n\n{input_text}",
    },
}

model_path = "outputs_qwen3_v2e/Qwen3-14B-merged"
print(f"Loading {model_path}...")

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)


def word_diff(response, input_text, expected):
    """Print word-level diff."""
    out_words = response.strip().split()
    exp_words = expected.strip().split()
    inp_words = input_text.strip().split()
    fixed, missed, wrong = [], [], []
    for i, (inp, exp) in enumerate(zip(inp_words, exp_words)):
        out = out_words[i] if i < len(out_words) else ""
        if inp != exp:
            if out == exp:
                fixed.append(f"  {inp} -> {out}")
            else:
                missed.append(f"  {inp} -> {out} (expected {exp})")
        elif out != exp:
            wrong.append(f"  {exp} -> {out} (was correct)")

    print(f"\nFixed ({len(fixed)}):")
    for f in fixed:
        print(f)
    print(f"\nMissed ({len(missed)}):")
    for m in missed:
        print(m)
    if wrong:
        print(f"\nWrongly changed ({len(wrong)}):")
        for w in wrong:
            print(w)


for prompt_name, prompt_config in PROMPTS.items():
    print(f"\n\n{'#' * 70}")
    print(f"# PROMPT: {prompt_name.upper()}")
    print(f"# System: {prompt_config['system'][:80]}...")
    print(f"{'#' * 70}")

    messages = [
        {"role": "system", "content": prompt_config["system"]},
        {"role": "user", "content": prompt_config["user"]},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Greedy
    print("\n" + "=" * 60)
    print(f"GREEDY (temperature=0) — {prompt_name}")
    print("=" * 60)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False, temperature=1.0)
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    print(f"Output: {response}")
    print(f"Exact match: {response.strip() == expected.strip()}")
    word_diff(response, input_text, expected)

    # 5 sampled runs
    print("\n\n" + "=" * 60)
    print(f"5 SAMPLED RUNS (temp=0.7) — {prompt_name}")
    print("=" * 60)
    for run in range(5):
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
        exact = response.strip() == expected.strip()
        print(f"\nRun {run + 1}: exact={exact}")
        print(f"  {response}")

print(f"\n\n{'=' * 60}")
print("EXPECTED")
print("=" * 60)
print(expected)
