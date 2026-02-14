#!/usr/bin/env python3
"""
Ecological validation of spelling correction models on the Bob story.

Uses the held-out Bob story error pairs (real dyslexic writing from a child)
to evaluate model performance. This is the ultimate test — if the model can't
correct these real errors, the training data pipeline needs work.

Metrics:
- Per-sentence exact match rate
- Per-word correction accuracy
- False positive rate (words changed that shouldn't be)
- Per-category accuracy (phonological, orthographic, morphological)
- Hallucination rate (corrections that introduce new errors)
- Over-correction rate (correct words that were changed)
- Under-correction rate (errors that were not fixed)
"""

import json
import argparse
import subprocess
import sys
from pathlib import Path
from dataclasses import dataclass, field
from difflib import SequenceMatcher

# ---------------------------------------------------------------------------
# Bob story evaluation data
# ---------------------------------------------------------------------------

# These pairs are the ground truth from bob-story-error-profile.md
# They must NEVER appear in training data
BOB_STORY_EVAL = [
    {
        "input": "Bob was a Bowring guy, he never went to a party he didn't need to for work.",
        "target": "Bob was a boring guy, he never went to a party he didn't need to for work.",
        "errors": [{"written": "Bowring", "target": "boring", "category": "phonological", "rule": "vowel_digraph_insertion"}],
    },
    {
        "input": "he walked out on to his belkany to get some fresh air",
        "target": "he walked out on to his balcony to get some fresh air",
        "errors": [{"written": "belkany", "target": "balcony", "category": "phonological", "rule": "multisyllable_restructuring"}],
    },
    {
        "input": "he hate going outside",
        "target": "he hated going outside",
        "errors": [{"written": "hate", "target": "hated", "category": "morphological", "rule": "missing_ed_suffix"}],
    },
    {
        "input": "he skremed at the hights",
        "target": "he screamed at the heights",
        "errors": [
            {"written": "skremed", "target": "screamed", "category": "phonological", "rule": "consonant_sc_to_sk+vowel_ea_to_e"},
            {"written": "hights", "target": "heights", "category": "orthographic", "rule": "vowel_digraph_ei_to_i"},
        ],
    },
    {
        "input": "the ailens falled from the selen",
        "target": "the aliens fell from the ceiling",
        "errors": [
            {"written": "ailens", "target": "aliens", "category": "orthographic", "rule": "letter_transposition"},
            {"written": "falled", "target": "fell", "category": "morphological", "rule": "irregular_past_over_regularization"},
            {"written": "selen", "target": "ceiling", "category": "phonological", "rule": "multisyllable_restructuring"},
        ],
    },
    {
        "input": "he fented and the he got up agan",
        "target": "he fainted and then he got up again",
        "errors": [
            {"written": "fented", "target": "fainted", "category": "phonological", "rule": "vowel_digraph_ai_to_e"},
            {"written": "the", "target": "then", "category": "orthographic", "rule": "final_consonant_drop"},
            {"written": "agan", "target": "again", "category": "phonological", "rule": "vowel_digraph_ai_to_a"},
        ],
    },
    {
        "input": "he hered skreming from the hights",
        "target": "he heard screaming from the heights",
        "errors": [
            {"written": "hered", "target": "heard", "category": "phonological", "rule": "vowel_digraph_ear_to_ere"},
            {"written": "skreming", "target": "screaming", "category": "phonological", "rule": "consonant_sc_to_sk+vowel_ea_to_e"},
            {"written": "hights", "target": "heights", "category": "orthographic", "rule": "vowel_digraph_ei_to_i"},
        ],
    },
    {
        "input": "he see a man fall from a bailden",
        "target": "he saw a man fell from a building",
        "errors": [
            {"written": "see", "target": "saw", "category": "morphological", "rule": "irregular_past_base_form"},
            {"written": "fall", "target": "fell", "category": "morphological", "rule": "irregular_past_base_form"},
            {"written": "bailden", "target": "building", "category": "phonological", "rule": "multisyllable_restructuring"},
        ],
    },
    {
        "input": "the bird flaw fast and gab him",
        "target": "the bird flew fast and grabbed him",
        "errors": [
            {"written": "flaw", "target": "flew", "category": "phonological", "rule": "vowel_digraph_ew_to_aw"},
            {"written": "gab", "target": "grabbed", "category": "phonological", "rule": "cluster_reduction+missing_suffix"},
        ],
    },
    {
        "input": "he drope him on the roof and he was safe",
        "target": "he dropped him on the roof and he was safe",
        "errors": [{"written": "drope", "target": "dropped", "category": "orthographic", "rule": "silent_e_addition+missing_doubling"}],
    },
    {
        "input": "thats how Bob fall from the hights",
        "target": "that's how Bob fell from the heights",
        "errors": [
            {"written": "thats", "target": "that's", "category": "orthographic", "rule": "missing_apostrophe"},
            {"written": "fall", "target": "fell", "category": "morphological", "rule": "irregular_past_base_form"},
            {"written": "hights", "target": "heights", "category": "orthographic", "rule": "vowel_digraph_ei_to_i"},
        ],
    },
    {
        "input": "he tried bugy-juming from the belkany",
        "target": "he tried bungee jumping from the balcony",
        "errors": [
            {"written": "bugy-juming", "target": "bungee jumping", "category": "phonological", "rule": "nasal_cluster_ng_to_g+ee_to_y"},
            {"written": "belkany", "target": "balcony", "category": "phonological", "rule": "multisyllable_restructuring"},
        ],
    },
    {
        "input": "Mow he is broken but safe",
        "target": "Now he is broken but safe",
        "errors": [{"written": "Mow", "target": "Now", "category": "orthographic", "rule": "letter_confusion_M_N"}],
    },
]


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

@dataclass
class WordResult:
    """Result for a single word comparison."""
    input_word: str
    target_word: str
    predicted_word: str
    is_error: bool       # Was the input word an error?
    was_corrected: bool  # Did the model change it?
    correct: bool        # Did the model get the right answer?
    category: str = ""   # Error category if applicable


@dataclass
class SentenceResult:
    """Result for a single sentence."""
    input_text: str
    target_text: str
    predicted_text: str
    exact_match: bool
    word_results: list = field(default_factory=list)
    errors_expected: int = 0
    errors_fixed: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    hallucinations: int = 0


@dataclass
class EvalReport:
    """Overall evaluation report."""
    sentences: list = field(default_factory=list)
    exact_match_rate: float = 0.0
    word_accuracy: float = 0.0
    false_positive_rate: float = 0.0
    false_negative_rate: float = 0.0
    hallucination_rate: float = 0.0
    category_accuracy: dict = field(default_factory=dict)
    total_errors: int = 0
    total_corrections: int = 0
    total_words: int = 0


def tokenize_simple(text: str) -> list[str]:
    """Simple whitespace tokenizer preserving punctuation."""
    return text.split()


def align_words(source: list[str], target: list[str]) -> list[tuple]:
    """Align words between source and target using SequenceMatcher."""
    matcher = SequenceMatcher(None, source, target)
    alignments = []

    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            for i, j in zip(range(i1, i2), range(j1, j2)):
                alignments.append((source[i], target[j], "equal"))
        elif op == "replace":
            for i, j in zip(range(i1, i2), range(j1, j2)):
                alignments.append((source[i], target[j], "replace"))
            # Handle unequal lengths
            if i2 - i1 > j2 - j1:
                for i in range(j2 - j1 + i1, i2):
                    alignments.append((source[i], "", "delete"))
            elif j2 - j1 > i2 - i1:
                for j in range(i2 - i1 + j1, j2):
                    alignments.append(("", target[j], "insert"))
        elif op == "delete":
            for i in range(i1, i2):
                alignments.append((source[i], "", "delete"))
        elif op == "insert":
            for j in range(j1, j2):
                alignments.append(("", target[j], "insert"))

    return alignments


def evaluate_sentence(
    input_text: str,
    target_text: str,
    predicted_text: str,
    expected_errors: list[dict],
) -> SentenceResult:
    """Evaluate model output for a single sentence."""
    result = SentenceResult(
        input_text=input_text,
        target_text=target_text,
        predicted_text=predicted_text,
        exact_match=(predicted_text.strip() == target_text.strip()),
    )

    input_words = tokenize_simple(input_text)
    target_words = tokenize_simple(target_text)
    pred_words = tokenize_simple(predicted_text)

    # Build error lookup: input_word -> error info
    error_lookup = {}
    for err in expected_errors:
        error_lookup[err["written"].lower()] = err

    result.errors_expected = len(expected_errors)

    # Compare input -> prediction vs input -> target
    for i, (inp_w, tgt_w, pred_w) in enumerate(
        zip_longest_simple(input_words, target_words, pred_words)
    ):
        is_error = inp_w.lower() in error_lookup
        was_corrected = inp_w != pred_w
        correct_target = tgt_w

        wr = WordResult(
            input_word=inp_w,
            target_word=tgt_w,
            predicted_word=pred_w,
            is_error=is_error,
            was_corrected=was_corrected,
            correct=(pred_w == tgt_w),
            category=error_lookup.get(inp_w.lower(), {}).get("category", ""),
        )
        result.word_results.append(wr)

        if is_error and pred_w == tgt_w:
            result.errors_fixed += 1
        elif is_error and pred_w != tgt_w and was_corrected:
            result.hallucinations += 1  # Changed but to wrong thing
        elif is_error and not was_corrected:
            result.false_negatives += 1  # Error not fixed
        elif not is_error and was_corrected and pred_w != tgt_w:
            result.false_positives += 1  # Changed a correct word incorrectly

    return result


def zip_longest_simple(*iterables):
    """Simple zip longest with empty string padding."""
    max_len = max(len(it) for it in iterables)
    for i in range(max_len):
        yield tuple(it[i] if i < len(it) else "" for it in iterables)


def compile_report(sentence_results: list[SentenceResult]) -> EvalReport:
    """Compile sentence results into an overall report."""
    report = EvalReport(sentences=sentence_results)

    total_words = 0
    total_correct = 0
    total_errors = 0
    total_fixed = 0
    total_fp = 0
    total_fn = 0
    total_hallucinations = 0
    category_correct = {}
    category_total = {}

    for sr in sentence_results:
        for wr in sr.word_results:
            total_words += 1
            if wr.correct:
                total_correct += 1
            if wr.is_error:
                total_errors += 1
                cat = wr.category or "unknown"
                category_total[cat] = category_total.get(cat, 0) + 1
                if wr.correct:
                    category_correct[cat] = category_correct.get(cat, 0) + 1

        total_fixed += sr.errors_fixed
        total_fp += sr.false_positives
        total_fn += sr.false_negatives
        total_hallucinations += sr.hallucinations

    n_sent = len(sentence_results)
    exact_matches = sum(1 for sr in sentence_results if sr.exact_match)

    report.exact_match_rate = exact_matches / n_sent if n_sent > 0 else 0
    report.word_accuracy = total_correct / total_words if total_words > 0 else 0
    report.false_positive_rate = total_fp / (total_words - total_errors) if (total_words - total_errors) > 0 else 0
    report.false_negative_rate = total_fn / total_errors if total_errors > 0 else 0
    report.hallucination_rate = total_hallucinations / total_errors if total_errors > 0 else 0
    report.total_errors = total_errors
    report.total_corrections = total_fixed
    report.total_words = total_words

    for cat in category_total:
        ct = category_total[cat]
        cc = category_correct.get(cat, 0)
        report.category_accuracy[cat] = cc / ct if ct > 0 else 0

    return report


# ---------------------------------------------------------------------------
# Model inference
# ---------------------------------------------------------------------------

def run_ollama_inference(model_name: str, input_text: str) -> str:
    """Run inference via Ollama CLI."""
    prompt = (
        "You are a spelling correction assistant. Fix only spelling and grammar errors. "
        "Do not change meaning, names, or correct text. If the text is already correct, "
        "return it unchanged.\n\n"
        f"Correct this text: {input_text}"
    )

    try:
        result = subprocess.run(
            ["ollama", "run", model_name, prompt],
            capture_output=True, text=True, timeout=30,
        )
        return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"  Ollama error: {e}")
        return input_text  # Return input unchanged on error


def run_dummy_inference(input_text: str) -> str:
    """Dummy inference that returns input unchanged (baseline)."""
    return input_text


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model_name: str = None,
    inference_fn=None,
    eval_data: list[dict] = None,
    verbose: bool = True,
) -> EvalReport:
    """
    Run full evaluation on Bob story data.

    Args:
        model_name: Ollama model name (if using Ollama).
        inference_fn: Custom inference function(input_text) -> predicted_text.
        eval_data: Custom eval data (default: BOB_STORY_EVAL).
        verbose: Print detailed results.
    """
    if eval_data is None:
        eval_data = BOB_STORY_EVAL

    if inference_fn is None:
        if model_name:
            inference_fn = lambda x: run_ollama_inference(model_name, x)
        else:
            print("No model specified, using dummy baseline (no corrections)")
            inference_fn = run_dummy_inference

    print(f"Evaluating on {len(eval_data)} sentences...")
    print()

    sentence_results = []
    for i, pair in enumerate(eval_data):
        predicted = inference_fn(pair["input"])
        result = evaluate_sentence(
            pair["input"], pair["target"], predicted, pair["errors"]
        )
        sentence_results.append(result)

        if verbose:
            status = "EXACT" if result.exact_match else "MISS"
            print(f"[{i + 1:2d}] {status}")
            print(f"  Input:     {pair['input']}")
            print(f"  Target:    {pair['target']}")
            print(f"  Predicted: {predicted}")
            if result.errors_fixed > 0:
                print(f"  Fixed: {result.errors_fixed}/{result.errors_expected}")
            if result.false_positives > 0:
                print(f"  False positives: {result.false_positives}")
            if result.hallucinations > 0:
                print(f"  Hallucinations: {result.hallucinations}")
            print()

    report = compile_report(sentence_results)

    # Print summary
    print("=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Sentences:           {len(sentence_results)}")
    print(f"Exact match rate:    {report.exact_match_rate:.1%}")
    print(f"Word accuracy:       {report.word_accuracy:.1%}")
    print(f"False positive rate: {report.false_positive_rate:.1%}")
    print(f"Under-correction:    {report.false_negative_rate:.1%}")
    print(f"Hallucination rate:  {report.hallucination_rate:.1%}")
    print(f"Errors fixed:        {report.total_corrections}/{report.total_errors}")
    print()

    if report.category_accuracy:
        print("Per-category accuracy:")
        for cat, acc in sorted(report.category_accuracy.items()):
            print(f"  {cat:20s}: {acc:.1%}")

    print("=" * 60)

    return report


def save_report(report: EvalReport, filepath: Path) -> None:
    """Save evaluation report to JSON."""
    data = {
        "exact_match_rate": round(report.exact_match_rate, 4),
        "word_accuracy": round(report.word_accuracy, 4),
        "false_positive_rate": round(report.false_positive_rate, 4),
        "false_negative_rate": round(report.false_negative_rate, 4),
        "hallucination_rate": round(report.hallucination_rate, 4),
        "total_errors": report.total_errors,
        "total_corrections": report.total_corrections,
        "total_words": report.total_words,
        "category_accuracy": {k: round(v, 4) for k, v in report.category_accuracy.items()},
        "sentences": [
            {
                "input": sr.input_text,
                "target": sr.target_text,
                "predicted": sr.predicted_text,
                "exact_match": sr.exact_match,
                "errors_fixed": sr.errors_fixed,
                "errors_expected": sr.errors_expected,
                "false_positives": sr.false_positives,
                "hallucinations": sr.hallucinations,
            }
            for sr in report.sentences
        ],
    }
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nReport saved to {filepath}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate spelling correction model on Bob story data"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Ollama model name to evaluate"
    )
    parser.add_argument(
        "--eval-file", type=str, default=None,
        help="Path to custom eval JSONL file (default: built-in Bob story pairs)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save evaluation report JSON"
    )
    parser.add_argument(
        "--baseline", action="store_true",
        help="Run baseline evaluation (no corrections)"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-sentence output"
    )
    args = parser.parse_args()

    eval_data = None
    if args.eval_file:
        filepath = Path(args.eval_file)
        if filepath.suffix == ".json":
            with open(filepath) as f:
                eval_data = json.load(f)
        else:
            eval_data = []
            with open(filepath) as f:
                for line in f:
                    if line.strip():
                        eval_data.append(json.loads(line))

    inference_fn = None
    if args.baseline:
        inference_fn = run_dummy_inference
        print("Running BASELINE evaluation (no corrections)\n")

    report = evaluate(
        model_name=args.model,
        inference_fn=inference_fn,
        eval_data=eval_data,
        verbose=not args.quiet,
    )

    if args.output:
        save_report(report, Path(args.output))
    else:
        # Default output path
        output_dir = _DATA_DIR
        model_tag = args.model.replace("/", "_").replace(":", "_") if args.model else "baseline"
        save_report(report, output_dir / f"eval_report_{model_tag}.json")


_DATA_DIR = Path(__file__).parent

if __name__ == "__main__":
    main()
