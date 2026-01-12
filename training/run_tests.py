#!/usr/bin/env python3
"""
Run hallucination tests on trained models using HuggingFace transformers.
Designed to run on Nibi cluster with GPU.
"""

import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from difflib import SequenceMatcher

# Test cases
PASS_THROUGH_TESTS = [
    "The quick brown fox jumps over the lazy dog.",
    "I live in Saskatoon, Saskatchewan.",
    "My email is john.smith@example.com.",
    "The meeting is at 3:30 PM.",
    "She has a PhD in Computer Science.",
    "The iPhone 15 Pro Max costs $1,199.",
    "My favorite color is blue.",
    "They went to the store.",
    "I like pizza and pasta.",
    "The weather is nice today.",
]

KNOWN_CORRECTIONS = [
    # Basic typos
    ("I went to teh store.", "I went to the store."),
    ("She is my freind.", "She is my friend."),
    ("The wheather is nice.", "The weather is nice."),
    ("I recieved your message.", "I received your message."),
    ("This is definately correct.", "This is definitely correct."),

    # Contractions - CRITICAL for dyslexic writers
    ("I dont know.", "I don't know."),
    ("Its a nice day.", "It's a nice day."),
    ("Im going home.", "I'm going home."),
    ("Shes my sister.", "She's my sister."),
    ("Hes very tall.", "He's very tall."),
    ("Were going out.", "We're going out."),
    ("Theyre coming too.", "They're coming too."),
    ("Youre welcome.", "You're welcome."),
    ("Cant do that.", "Can't do that."),
    ("Wont be long.", "Won't be long."),
    ("Didnt see it.", "Didn't see it."),
    ("Doesnt matter.", "Doesn't matter."),
    ("Wouldnt help.", "Wouldn't help."),
    ("Couldnt find it.", "Couldn't find it."),
    ("Shouldnt go.", "Shouldn't go."),
    ("Lets go now.", "Let's go now."),
    ("Thats great.", "That's great."),
    ("Whats up.", "What's up."),
    ("Whos there.", "Who's there."),
    ("Theres a cat.", "There's a cat."),
    ("Heres your book.", "Here's your book."),

    # Homophones - CRITICAL for dyslexic writers
    ("Your the best.", "You're the best."),
    ("I went their yesterday.", "I went there yesterday."),
    ("There dog is big.", "Their dog is big."),
    ("Its over they're.", "It's over there."),
    ("I no the answer.", "I know the answer."),
    ("He through the ball.", "He threw the ball."),
    ("I want to by a car.", "I want to buy a car."),
    ("The whether is cold.", "The weather is cold."),
    ("I herd a noise.", "I heard a noise."),
    ("She past the test.", "She passed the test."),
    ("The plain landed.", "The plane landed."),
    ("I need a brake.", "I need a break."),
    ("The dear ran away.", "The deer ran away."),
    ("I eight lunch.", "I ate lunch."),
    ("The son is bright.", "The sun is bright."),
    ("I red the book.", "I read the book."),
    ("The knight was dark.", "The night was dark."),
    ("I sea the ocean.", "I see the ocean."),
    ("The flour is white.", "The flour is white."),  # This one is tricky - could be flower
    ("He one the game.", "He won the game."),
    ("The hole team came.", "The whole team came."),
    ("I weight too much.", "I weigh too much."),
    ("She has blond hare.", "She has blond hair."),
    ("The pear of shoes.", "The pair of shoes."),

    # Compound words / word boundaries
    ("alot of people came.", "a lot of people came."),
    ("I need to goto the store.", "I need to go to the store."),
    ("Thankyou for helping.", "Thank you for helping."),
    ("I went infront of him.", "I went in front of him."),
    ("Its apart of life.", "It's a part of life."),
    ("Everyday I wake up.", "Every day I wake up."),  # When used as adverb
    ("Infact its true.", "In fact it's true."),
    ("Nevermind that.", "Never mind that."),
    ("Noone came.", "No one came."),
    ("Atleast try.", "At least try."),
]

DYSLEXIA_REVERSALS = [
    # b/d confusion
    ("The dag barked.", "The dog barked."),
    ("I went to bed.", "I went to bed."),  # Correct - should pass through
    ("The boor is open.", "The door is open."),
    ("He has a big doard.", "He has a big board."),

    # p/q confusion
    ("She is puite tall.", "She is quite tall."),
    ("I have a puestion.", "I have a question."),

    # m/w confusion
    ("The water is marm.", "The water is warm."),
    ("I mant to go.", "I want to go."),

    # Letter order reversals
    ("Form vs from.", "From vs from."),  # form/from
    ("I saw tow cats.", "I saw two cats."),
    ("The gril ran fast.", "The girl ran fast."),
]

EDGE_CASES = [
    ("I met Elon Musk yesterday.", "I met Elon Musk yesterday."),
    ("She works at Google.", "She works at Google."),
    ("The API endpoint returns JSON.", "The API endpoint returns JSON."),
    ("Run pip install numpy.", "Run pip install numpy."),
    ("The CEO and CFO met today.", "The CEO and CFO met today."),
    # More edge cases
    ("My iPhone is broken.", "My iPhone is broken."),
    ("I use macOS daily.", "I use macOS daily."),
    ("The URL is https://example.com.", "The URL is https://example.com."),
]


def load_model(model_path: str):
    """Load model and tokenizer."""
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def generate_correction(model, tokenizer, text: str, max_new_tokens: int = 128) -> str:
    """Generate spelling correction for input text."""
    # Format as instruction
    if hasattr(tokenizer, 'apply_chat_template'):
        messages = [
            {"role": "system", "content": "You are a spelling correction assistant."},
            {"role": "user", "content": f"Fix the spelling mistakes in this sentence. Only output the corrected sentence.\n\n{text}"}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = f"### Instruction:\nFix the spelling mistakes in this sentence. Only output the corrected sentence.\n\n### Input:\n{text}\n\n### Response:\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the new tokens
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    # Clean up response - take first line/sentence
    response = response.strip()
    if '\n' in response:
        response = response.split('\n')[0]

    return response.strip()


def similarity(a: str, b: str) -> float:
    """Calculate string similarity ratio."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def run_tests(model, tokenizer, model_name: str) -> dict:
    """Run all tests on a model."""
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"{'='*60}")

    results = {
        "model": model_name,
        "pass_through": {"passed": 0, "failed": 0, "details": []},
        "corrections": {"passed": 0, "failed": 0, "details": []},
        "edge_cases": {"passed": 0, "failed": 0, "details": []},
        "word_preservation": {"passed": 0, "failed": 0, "details": []},
    }

    # 1. Pass-through tests
    print("\n1. Pass-through tests (correct text should stay unchanged):")
    for text in PASS_THROUGH_TESTS:
        output = generate_correction(model, tokenizer, text)
        sim = similarity(output, text)
        passed = sim > 0.95

        if passed:
            results["pass_through"]["passed"] += 1
            status = "✓"
        else:
            results["pass_through"]["failed"] += 1
            status = "✗"

        results["pass_through"]["details"].append({
            "input": text,
            "output": output,
            "similarity": sim,
            "passed": passed
        })
        print(f"  {status} [{sim:.2f}] {text[:40]}...")
        if not passed:
            print(f"       Got: {output[:40]}...")

    # 2. Known corrections
    print("\n2. Known corrections (should fix misspellings):")
    for input_text, expected in KNOWN_CORRECTIONS:
        output = generate_correction(model, tokenizer, input_text)
        sim = similarity(output, expected)
        passed = sim > 0.90

        if passed:
            results["corrections"]["passed"] += 1
            status = "✓"
        else:
            results["corrections"]["failed"] += 1
            status = "✗"

        results["corrections"]["details"].append({
            "input": input_text,
            "expected": expected,
            "output": output,
            "similarity": sim,
            "passed": passed
        })
        print(f"  {status} [{sim:.2f}] {input_text[:30]}... → {output[:30]}...")

    # 3. Edge cases
    print("\n3. Edge cases (proper nouns, tech terms):")
    for input_text, expected in EDGE_CASES:
        output = generate_correction(model, tokenizer, input_text)
        sim = similarity(output, expected)
        passed = sim > 0.95

        if passed:
            results["edge_cases"]["passed"] += 1
            status = "✓"
        else:
            results["edge_cases"]["failed"] += 1
            status = "✗"

        results["edge_cases"]["details"].append({
            "input": input_text,
            "expected": expected,
            "output": output,
            "similarity": sim,
            "passed": passed
        })
        print(f"  {status} [{sim:.2f}] {input_text[:40]}...")
        if not passed:
            print(f"       Got: {output[:40]}...")

    # 4. Dyslexia reversals
    print("\n4. Dyslexia reversals (b/d, p/q, letter swaps):")
    results["dyslexia"] = {"passed": 0, "failed": 0, "details": []}
    for input_text, expected in DYSLEXIA_REVERSALS:
        output = generate_correction(model, tokenizer, input_text)
        sim = similarity(output, expected)
        passed = sim > 0.90

        if passed:
            results["dyslexia"]["passed"] += 1
            status = "✓"
        else:
            results["dyslexia"]["failed"] += 1
            status = "✗"

        results["dyslexia"]["details"].append({
            "input": input_text,
            "expected": expected,
            "output": output,
            "similarity": sim,
            "passed": passed
        })
        print(f"  {status} [{sim:.2f}] {input_text[:30]}... → {output[:30]}...")

    # 5. Word preservation
    print("\n5. Word preservation (no words added/deleted):")
    test_texts = PASS_THROUGH_TESTS[:5] + [t[0] for t in KNOWN_CORRECTIONS[:5]]
    for text in test_texts:
        output = generate_correction(model, tokenizer, text)
        input_words = count_words(text)
        output_words = count_words(output)
        diff = abs(input_words - output_words)
        passed = diff <= 1  # Allow 1 word difference for contractions

        if passed:
            results["word_preservation"]["passed"] += 1
            status = "✓"
        else:
            results["word_preservation"]["failed"] += 1
            status = "✗"

        results["word_preservation"]["details"].append({
            "input": text,
            "output": output,
            "input_words": input_words,
            "output_words": output_words,
            "passed": passed
        })
        if not passed:
            print(f"  {status} Words: {input_words} → {output_words} | {text[:30]}...")

    word_passed = results["word_preservation"]["passed"]
    word_total = word_passed + results["word_preservation"]["failed"]
    print(f"  Passed: {word_passed}/{word_total}")

    # Summary
    total_passed = sum(r["passed"] for r in results.values() if isinstance(r, dict) and "passed" in r)
    total_failed = sum(r["failed"] for r in results.values() if isinstance(r, dict) and "failed" in r)
    total = total_passed + total_failed

    print(f"\n{'='*60}")
    print(f"SUMMARY: {total_passed}/{total} tests passed ({100*total_passed/total:.1f}%)")
    print(f"  Pass-through: {results['pass_through']['passed']}/{results['pass_through']['passed']+results['pass_through']['failed']}")
    print(f"  Corrections:  {results['corrections']['passed']}/{results['corrections']['passed']+results['corrections']['failed']}")
    print(f"  Edge cases:   {results['edge_cases']['passed']}/{results['edge_cases']['passed']+results['edge_cases']['failed']}")
    print(f"  Dyslexia:     {results['dyslexia']['passed']}/{results['dyslexia']['passed']+results['dyslexia']['failed']}")
    print(f"  Word preservation: {results['word_preservation']['passed']}/{results['word_preservation']['passed']+results['word_preservation']['failed']}")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default="./outputs", help="Directory containing trained models")
    parser.add_argument("--model", type=str, help="Specific model to test (e.g., SmolLM2-360M-Instruct)")
    parser.add_argument("--output", type=str, help="Save results to JSON")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    all_results = {}

    if args.model:
        models = [args.model]
    else:
        # Find all model directories
        models = [d.name for d in model_dir.iterdir() if d.is_dir() and (d / "config.json").exists()]

    print(f"Found models: {models}")

    for model_name in models:
        model_path = model_dir / model_name
        if not (model_path / "config.json").exists():
            print(f"Skipping {model_name} - no config.json found")
            continue

        try:
            model, tokenizer = load_model(str(model_path))
            results = run_tests(model, tokenizer, model_name)
            all_results[model_name] = results

            # Free memory
            del model
            del tokenizer
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error testing {model_name}: {e}")
            continue

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
