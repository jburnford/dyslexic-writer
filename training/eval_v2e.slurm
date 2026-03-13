#!/bin/bash
#SBATCH --job-name=eval-v2e
#SBATCH --output=logs/eval-v2e-%j.out
#SBATCH --error=logs/eval-v2e-%j.err
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G

# ============================================================================
# Dyslexic Writer v2e - Evaluate, Hockey Story Tests, GGUF Convert
#
# Phase 1: Evaluate merged models on eval set
# Phase 2: Hockey story single-run test (greedy)
# Phase 3: Hockey story 100-run statistical test (sampling)
# Phase 4: Convert to GGUF + quantize
# ============================================================================

set -e

echo "=========================================="
echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="

mkdir -p logs

module load python/3.11 cuda/12.6 arrow/21.0.0 cmake gcc/12.3

echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv

VENV_DIR=$SLURM_TMPDIR/venv
echo "Creating virtual environment in $VENV_DIR..."
virtualenv --no-download $VENV_DIR
source $VENV_DIR/bin/activate

pip install --no-index --upgrade pip
pip install --no-index torch torchvision
pip install --no-index transformers accelerate datasets sentencepiece numpy
pip install gguf 2>/dev/null || pip install --user gguf

echo "Installed packages:"
pip list | grep -E "torch|transformers|gguf"

cd $SLURM_SUBMIT_DIR

MODELS_DIR=outputs_qwen3_v2e

# ============================================================================
# PHASE 1: Evaluate models on eval set
# ============================================================================
echo ""
echo "=========================================="
echo "PHASE 1: Evaluating v2e models"
echo "=========================================="

python evaluate_models.py \
    --models-dir $MODELS_DIR \
    --eval-file v2_data/eval.jsonl \
    --max-examples 627 \
    --models Qwen3-0.6B-merged Qwen3-1.7B-merged Qwen3-4B-merged Qwen3-14B-merged

echo ""
echo "PHASE 1 COMPLETE: $(date)"

# ============================================================================
# PHASE 2: Hockey story single test (greedy, all models)
# ============================================================================
echo ""
echo "=========================================="
echo "PHASE 2: Hockey story single test"
echo "=========================================="

python3 << 'HOCKEY_SINGLE'
import json, re, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

with open("bob_hockey_story.json") as f:
    story = json.load(f)

input_text = story["input"]
expected = story["expected"]
instruction = "Fix any spelling mistakes in this text. If there are no mistakes, output the text unchanged."

models = [
    "outputs_qwen3_v2e/Qwen3-0.6B-merged",
    "outputs_qwen3_v2e/Qwen3-1.7B-merged",
    "outputs_qwen3_v2e/Qwen3-4B-merged",
    "outputs_qwen3_v2e/Qwen3-14B-merged",
]

for model_path in models:
    name = model_path.split("/")[-1]
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"{name} (v2e)")
    print(sep)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )

    messages = [
        {"role": "system", "content": "You are a spelling correction assistant."},
        {"role": "user", "content": f"{instruction}\n\n{input_text}"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False, temperature=1.0)

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

    print(f"Output: {response}")
    print(f"Exact match: {response.strip() == expected.strip()}")

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
    for f in fixed: print(f)
    print(f"\nMissed ({len(missed)}):")
    for m in missed: print(m)
    if wrong:
        print(f"\nWrongly changed ({len(wrong)}):")
        for w in wrong: print(w)

    del model
    torch.cuda.empty_cache()

print(f"\n{'=' * 60}")
print("EXPECTED")
print("=" * 60)
print(expected)
HOCKEY_SINGLE

echo ""
echo "PHASE 2 COMPLETE: $(date)"

# ============================================================================
# PHASE 3: Hockey story 100-run statistical test
# ============================================================================
echo ""
echo "=========================================="
echo "PHASE 3: Hockey story 100-run test"
echo "=========================================="

# Use the existing test_hockey_100.py but with v2e models
python3 << 'HOCKEY_100'
import json, re, time, torch
from collections import Counter
from difflib import SequenceMatcher
from transformers import AutoModelForCausalLM, AutoTokenizer

RUNS = 100

with open("bob_hockey_story.json") as f:
    story = json.load(f)

input_text = story["input"]
expected = story["expected"]
instruction = "Fix any spelling mistakes in this text. If there are no mistakes, output the text unchanged."

KEY_ERRORS = [
    ("doing", "going", "wrong word"),
    ("playoff.", "playoffs.", "missing plural"),
    ("hole", "whole", "homophone"),
    ("theres", "there's", "missing apostrophe"),
    ("tow", "two", "phonetic"),
    ("minits", "minutes", "phonetic"),
    ("minis", "minutes", "phonetic"),
    ("3ed", "3rd", "ordinal"),
    ("Oscor", "Oscar", "name"),
    ("socord.", "scored.", "phonetic"),
    ("secend", "second", "phonetic"),
    ("agens", "against", "phonetic"),
    ("leg.", "league.", "truncation"),
    ("geting", "getting", "double consonant"),
    ("goas", "goals", "missing letter"),
    ("scord", "scored", "missing letter"),
    ("thrid.", "third.", "transposition"),
    ("fater", "after", "transposition"),
    ("tird.", "third.", "missing letter"),
]


def check_corrections(input_text, output_text, key_errors):
    inp_words = input_text.split()
    out_words = output_text.split()
    matcher = SequenceMatcher(None, inp_words, out_words)
    inp_to_out = {}
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            for k in range(i2 - i1):
                inp_to_out[i1 + k] = out_words[j1 + k]
        elif op == "replace":
            for k in range(min(i2 - i1, j2 - j1)):
                inp_to_out[i1 + k] = out_words[j1 + k]

    results = {}
    for err_word, correct_word, err_type in key_errors:
        for i, w in enumerate(inp_words):
            if w == err_word:
                out_word = inp_to_out.get(i, "")
                results[f"{err_word}->{correct_word}"] = {
                    "fixed": out_word == correct_word,
                    "output": out_word,
                }
                break

    for i, w in enumerate(inp_words):
        if w == "He" and i + 1 < len(inp_words) and inp_words[i + 1] == "lost":
            out_word = inp_to_out.get(i, "")
            results["He->We (pronoun)"] = {"fixed": out_word == "We", "output": out_word}
        if w == "he" and i + 1 < len(inp_words) and inp_words[i + 1] == "won":
            out_word = inp_to_out.get(i, "")
            results["he->we (pronoun)"] = {"fixed": out_word == "we", "output": out_word}

    return results


models = [
    "outputs_qwen3_v2e/Qwen3-0.6B-merged",
    "outputs_qwen3_v2e/Qwen3-1.7B-merged",
    "outputs_qwen3_v2e/Qwen3-4B-merged",
    "outputs_qwen3_v2e/Qwen3-14B-merged",
]

messages = [
    {"role": "system", "content": "You are a spelling correction assistant."},
    {"role": "user", "content": f"{instruction}\n\n{input_text}"},
]

all_results = {}
all_error_keys = [f"{e[0]}->{e[1]}" for e in KEY_ERRORS] + ["He->We (pronoun)", "he->we (pronoun)"]

for model_path in models:
    name = model_path.split("/")[-1]
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"{name} (v2e) - {RUNS} runs")
    print(sep)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    exact_matches = 0
    fix_counts = Counter()
    alt_outputs = {}
    unique_outputs = set()

    start = time.time()

    for run in range(RUNS):
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

        if response == expected.strip():
            exact_matches += 1

        unique_outputs.add(response)

        corrections = check_corrections(input_text, response, KEY_ERRORS)
        for key, result in corrections.items():
            if result["fixed"]:
                fix_counts[key] += 1
            if key not in alt_outputs:
                alt_outputs[key] = Counter()
            alt_outputs[key][result["output"]] += 1

        if (run + 1) % 25 == 0:
            elapsed = time.time() - start
            print(f"  [{run+1}/{RUNS}] exact={exact_matches} ({elapsed:.0f}s)", flush=True)

    elapsed = time.time() - start

    print(f"\n--- {name} Summary ({elapsed:.0f}s) ---")
    print(f"Exact match: {exact_matches}/{RUNS} ({exact_matches/RUNS*100:.1f}%)")
    print(f"Unique outputs: {len(unique_outputs)}")

    print(f"\nError fix rates:")
    model_rates = {}
    for key in all_error_keys:
        rate = fix_counts[key]
        model_rates[key] = rate / RUNS
        alts = alt_outputs.get(key, Counter())
        if rate == RUNS:
            print(f"  {key:30s} {rate:3d}/{RUNS} ({rate/RUNS*100:5.1f}%)")
        else:
            alternatives = [(w, c) for w, c in alts.most_common(3) if w != key.split("->")[1]]
            alt_str = ", ".join(f"{w}({c})" for w, c in alternatives[:3])
            print(f"  {key:30s} {rate:3d}/{RUNS} ({rate/RUNS*100:5.1f}%) alt: {alt_str}")

    all_results[name] = {
        "exact_matches": exact_matches,
        "unique_outputs": len(unique_outputs),
        "fix_rates": model_rates,
    }

    del model
    torch.cuda.empty_cache()

# Final comparison table
print(f"\n{'=' * 90}")
print("V2E COMPARISON TABLE (fix rate %)")
print(f"{'=' * 90}")
header = f"{'Error':30s} {'Type':18s}"
for name in all_results:
    header += f" {name.replace('-merged',''):>10s}"
print(header)
print("-" * 90)

error_types = {f"{e[0]}->{e[1]}": e[2] for e in KEY_ERRORS}
error_types["He->We (pronoun)"] = "pronoun"
error_types["he->we (pronoun)"] = "pronoun"

for key in all_error_keys:
    etype = error_types.get(key, "")
    row = f"{key:30s} {etype:18s}"
    for name in all_results:
        rate = all_results[name]["fix_rates"].get(key, 0) * 100
        row += f" {rate:9.0f}%"
    print(row)

print("-" * 90)
row = f"{'EXACT MATCH':30s} {'':18s}"
for name in all_results:
    em = all_results[name]["exact_matches"]
    row += f" {em:8d}/{RUNS}"
print(row)
row = f"{'UNIQUE OUTPUTS':30s} {'':18s}"
for name in all_results:
    row += f" {all_results[name]['unique_outputs']:9d}"
print(row)

with open("hockey_test_v2e_100_results.json", "w") as f:
    json.dump(all_results, f, indent=2)
print("\nResults saved to hockey_test_v2e_100_results.json")
HOCKEY_100

echo ""
echo "PHASE 3 COMPLETE: $(date)"

# ============================================================================
# PHASE 4: Convert to GGUF
# ============================================================================
echo ""
echo "=========================================="
echo "PHASE 4: Converting v2e models to GGUF"
echo "=========================================="

GGUF_DIR=gguf_models_v2e
mkdir -p $GGUF_DIR

LLAMA_DIR=$SLURM_TMPDIR/llama.cpp
if [ ! -d "$LLAMA_DIR" ]; then
    echo "Cloning llama.cpp..."
    git clone --depth 1 https://github.com/ggerganov/llama.cpp.git $LLAMA_DIR
fi

echo "Building llama-quantize..."
cd $LLAMA_DIR
mkdir -p build && cd build
CCACHE_DISABLE=1 cmake .. -DGGML_NATIVE=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build . --target llama-quantize -j 8
QUANTIZE=$LLAMA_DIR/build/bin/llama-quantize
cd $SLURM_SUBMIT_DIR

MODEL_LIST="Qwen3-0.6B-merged Qwen3-1.7B-merged Qwen3-4B-merged Qwen3-14B-merged"

for MODEL_MERGED in $MODEL_LIST; do
    MODEL_NAME=${MODEL_MERGED%-merged}
    MODEL_PATH=$MODELS_DIR/$MODEL_MERGED
    F16_GGUF=$GGUF_DIR/${MODEL_NAME}-f16.gguf
    Q4_GGUF=$GGUF_DIR/${MODEL_NAME}-q4_k_m.gguf
    Q8_GGUF=$GGUF_DIR/${MODEL_NAME}-q8_0.gguf

    echo ""
    echo "--- Converting $MODEL_NAME ---"

    if [ ! -d "$MODEL_PATH" ]; then
        echo "ERROR: Model not found at $MODEL_PATH, skipping"
        continue
    fi

    echo "  Converting to F16 GGUF..."
    python $LLAMA_DIR/convert_hf_to_gguf.py "$MODEL_PATH" \
        --outfile "$F16_GGUF" \
        --outtype f16

    if [ $? -ne 0 ]; then
        echo "ERROR: F16 conversion failed for $MODEL_NAME"
        continue
    fi
    echo "  F16: $(du -h "$F16_GGUF" | cut -f1)"

    echo "  Quantizing to Q4_K_M..."
    $QUANTIZE "$F16_GGUF" "$Q4_GGUF" Q4_K_M
    echo "  Q4_K_M: $(du -h "$Q4_GGUF" | cut -f1)"

    echo "  Quantizing to Q8_0..."
    $QUANTIZE "$F16_GGUF" "$Q8_GGUF" Q8_0
    echo "  Q8_0: $(du -h "$Q8_GGUF" | cut -f1)"

    if [[ "$MODEL_NAME" == *"14B"* ]]; then
        Q5_GGUF=$GGUF_DIR/${MODEL_NAME}-q5_k_m.gguf
        echo "  Quantizing to Q5_K_M..."
        $QUANTIZE "$F16_GGUF" "$Q5_GGUF" Q5_K_M
        echo "  Q5_K_M: $(du -h "$Q5_GGUF" | cut -f1)"
    fi

    echo "  Done: $MODEL_NAME"
done

echo ""
echo "PHASE 4 COMPLETE"
echo ""
echo "GGUF models:"
ls -lh $GGUF_DIR/*.gguf 2>/dev/null

echo ""
echo "=========================================="
echo "ALL PHASES COMPLETE"
echo "Job finished: $(date)"
echo "=========================================="
