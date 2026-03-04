#!/usr/bin/env python3
"""
Test spelling correction models on Pascal's D&D story.

Feeds each paragraph to the model and compares output to gold standard.
Uses the light-edit gold standard (spelling only, no grammar restructuring).
"""

import json
import subprocess
import sys
import re
from pathlib import Path
from difflib import SequenceMatcher

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent

# The original story, split into paragraphs (matching line structure)
TEST_STORY_PARAGRAPHS = [
    "Pascals dnd champain",
    "We came to his house at 10:00Am.",
    "To get started we traved to a littal town. we where trining to fided the colt leder. We went to the pup and our drinks got spik. atte had it the wars. then we got kidknapt down to the caves onder the town. the first cave we saw some pots and a a lockeddoor ronansmash half the pots and fond the key. the secend room had aboom i forgt how we got out of it. but in the next room we fot our first battle thare was a trole an d some skelatens  masan grabehis chain then smash the skellys to jost bone. ronan canvenst the trol to come with us then we left.",
    "then we traved to a knome town we get informashon from the sitasans.",
    "after that we traved to a forst there we fond a army of skelatens and one big skelly. we one hit the big skelly with masons chain then kill the restof them off.",
    "then we got on a boat then saled to the mittal of the sea the n we order uber eat we get it because mamoude got the 10% tip the we get eten by a wale.",
    "thats all we get to in the first seshen.",
]

# Gold standard paragraphs (light edit - spelling only)
GOLD_PARAGRAPHS = [
    "Pascal's dnd campaign",
    "We came to his house at 10:00Am.",
    "To get started we travelled to a little town. we were trying to find the cult leader. We went to the pub and our drinks got spiked. Atte had it the worst. then we got kidnapped down to the caves under the town. the first cave we saw some pots and a locked door Ronan smashed half the pots and found the key. the second room had a bomb I forgot how we got out of it. but in the next room we fought our first battle there was a troll and some skeletons Mason grabbed his chain then smashed the skeletons to just bone. Ronan convinced the troll to come with us then we left.",
    "then we travelled to a gnome town we got information from the citizens.",
    "after that we travelled to a forest there we found an army of skeletons and one big skeleton. we one hit the big skeleton with Mason's chain then killed the rest of them off.",
    "then we got on a boat then sailed to the middle of the sea then we ordered Uber Eats we got it because Mamoude got the 10% tip then we got eaten by a whale.",
    "that's all we got to in the first session.",
]


def run_ollama(model_name: str, text: str) -> str:
    """Run inference via Ollama CLI."""
    prompt = f"Fix any spelling mistakes in this text. If there are no mistakes, output the text unchanged.\n\n{text}"
    try:
        result = subprocess.run(
            ["ollama", "run", model_name, prompt],
            capture_output=True, text=True, timeout=120,
        )
        output = result.stdout.strip()
        # Strip any <think>...</think> tags from Qwen3
        output = re.sub(r"<think>.*?</think>", "", output, flags=re.DOTALL).strip()
        return output
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT on: {text[:50]}...")
        return text
    except FileNotFoundError:
        print("ERROR: Ollama not found. Is it installed and running?")
        sys.exit(1)


def word_diff(original: str, corrected: str) -> list[str]:
    """Show word-level differences."""
    orig_words = original.split()
    corr_words = corrected.split()
    diffs = []
    matcher = SequenceMatcher(None, orig_words, corr_words)
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "replace":
            for ow, cw in zip(orig_words[i1:i2], corr_words[j1:j2]):
                if ow != cw:
                    diffs.append(f"  {ow} -> {cw}")
        elif op == "delete":
            for w in orig_words[i1:i2]:
                diffs.append(f"  {w} -> [DELETED]")
        elif op == "insert":
            for w in corr_words[j1:j2]:
                diffs.append(f"  [INSERTED] -> {w}")
    return diffs


def similarity(a: str, b: str) -> float:
    """String similarity ratio."""
    return SequenceMatcher(None, a.lower().split(), b.lower().split()).ratio()


def evaluate_model(model_name: str):
    """Run full paragraph-level evaluation for one model."""
    print(f"\n{'='*70}")
    print(f"MODEL: {model_name}")
    print(f"{'='*70}\n")

    results = []
    total_errors_in_gold = 0
    total_corrections_made = 0
    total_correct_corrections = 0
    total_false_positives = 0

    for i, (original, gold) in enumerate(zip(TEST_STORY_PARAGRAPHS, GOLD_PARAGRAPHS)):
        print(f"--- Paragraph {i+1} ---")
        print(f"INPUT:     {original[:80]}{'...' if len(original)>80 else ''}")

        predicted = run_ollama(model_name, original)
        print(f"MODEL:     {predicted[:80]}{'...' if len(predicted)>80 else ''}")
        print(f"GOLD:      {gold[:80]}{'...' if len(gold)>80 else ''}")

        sim_to_gold = similarity(predicted, gold)
        exact = predicted.strip() == gold.strip()
        print(f"MATCH:     {'EXACT' if exact else f'similarity={sim_to_gold:.1%}'}")

        # Show what the model changed
        changes = word_diff(original, predicted)
        if changes:
            print(f"CHANGES ({len(changes)}):")
            for c in changes:
                print(f"    {c}")
        else:
            print("CHANGES: none")

        # Show what should have changed (gold vs original)
        gold_changes = word_diff(original, gold)

        # Count corrections
        gold_set = set(word_diff(original, gold))
        pred_set = set(word_diff(original, predicted))
        correct = gold_set & pred_set  # changes matching gold
        missed = gold_set - pred_set   # gold changes not made
        extra = pred_set - gold_set    # changes not in gold

        total_errors_in_gold += len(gold_set)
        total_correct_corrections += len(correct)
        total_false_positives += len(extra)

        if missed:
            print(f"MISSED ({len(missed)}):")
            for m in sorted(missed):
                print(f"    {m}")
        if extra:
            print(f"EXTRA ({len(extra)}):")
            for e in sorted(extra):
                print(f"    {e}")

        results.append({
            "paragraph": i + 1,
            "input": original,
            "predicted": predicted,
            "gold": gold,
            "exact_match": exact,
            "similarity": round(sim_to_gold, 4),
            "correct_fixes": len(correct),
            "missed_fixes": len(missed),
            "false_positives": len(extra),
            "total_gold_fixes": len(gold_set),
        })
        print()

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY: {model_name}")
    print(f"{'='*70}")
    exact_matches = sum(1 for r in results if r["exact_match"])
    print(f"Exact paragraph matches: {exact_matches}/{len(results)}")
    print(f"Correct corrections:     {total_correct_corrections}/{total_errors_in_gold} ({total_correct_corrections/total_errors_in_gold*100:.1f}%)" if total_errors_in_gold > 0 else "")
    print(f"False positives:         {total_false_positives}")
    avg_sim = sum(r["similarity"] for r in results) / len(results)
    print(f"Average similarity:      {avg_sim:.1%}")
    print(f"{'='*70}\n")

    return results


def main():
    models = sys.argv[1:] if len(sys.argv) > 1 else ["dyslexic-writer-4b-q5"]

    all_results = {}
    for model in models:
        all_results[model] = evaluate_model(model)

    # Save results
    output_path = SCRIPT_DIR / "pascal_story_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Print comparison table if multiple models
    if len(models) > 1:
        print(f"\n{'='*70}")
        print("COMPARISON")
        print(f"{'='*70}")
        print(f"{'Model':<30} {'Exact':>6} {'Fix%':>6} {'FP':>4} {'Sim':>6}")
        print("-" * 56)
        for model in models:
            r = all_results[model]
            exact = sum(1 for x in r if x["exact_match"])
            fixes = sum(x["correct_fixes"] for x in r)
            total = sum(x["total_gold_fixes"] for x in r)
            fp = sum(x["false_positives"] for x in r)
            sim = sum(x["similarity"] for x in r) / len(r)
            pct = fixes / total * 100 if total > 0 else 0
            print(f"{model:<30} {exact:>4}/7 {pct:>5.1f}% {fp:>4} {sim:>5.1%}")


if __name__ == "__main__":
    main()
