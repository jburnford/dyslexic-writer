"""
Dyslexic Writer — Hugging Face Spaces demo.
Uses ZeroGPU to run the fine-tuned Qwen3-4B spelling correction model for free.
"""

import difflib
import re

import gradio as gr
import spaces
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

MODEL_ID = "jburnford/dyslexic-writer-qwen3-4b"

SYSTEM_PROMPT = (
    "You are a spelling correction assistant. "
    "Fix only spelling and grammar errors. "
    "Do not change meaning, names, or correct text. "
    "If the text is already correct, return it unchanged."
)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    trust_remote_code=True,
)
model.eval()
print("Model loaded.")

# ---------------------------------------------------------------------------
# Correction logic (mirrors app/corrector.py)
# ---------------------------------------------------------------------------


def compute_changes(original: str, corrected: str) -> list[tuple[str, str]]:
    """Word-level diff using difflib — same algorithm as the local app."""
    orig_words = original.split()
    corr_words = corrected.split()
    diffs = []
    matcher = difflib.SequenceMatcher(None, orig_words, corr_words)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "replace":
            orig_phrase = " ".join(orig_words[i1:i2])
            corr_phrase = " ".join(corr_words[j1:j2])
            if orig_phrase != corr_phrase:
                diffs.append((orig_phrase, corr_phrase))
    return diffs


def strip_think_tags(text: str) -> str:
    """Remove Qwen3 <think>...</think> reasoning tags if present."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


@spaces.GPU(duration=120)
def correct(text: str) -> tuple[str, str]:
    """Run spelling correction on the input text. Returns (corrected, details)."""
    if not text or not text.strip():
        return "", ""

    text = text.strip()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.1,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = outputs[0][prompt_len:]
    corrected = tokenizer.decode(generated, skip_special_tokens=True).strip()
    corrected = strip_think_tags(corrected)

    changes = compute_changes(text, corrected)

    if not changes:
        details = "No errors found!"
    else:
        lines = []
        for orig, fixed in changes:
            lines.append(f"  {orig}  →  {fixed}")
        details = "Corrections:\n" + "\n".join(lines)

    return corrected, details


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

EXAMPLES = [
    "I have enuff food becuase Im hungrey.",
    "My freind went to the libary to studdy for the mathamatics test.",
    "She was very exited to recieve the presant from her bruther.",
    "The wether was beautful so we desided to go for a wlak.",
    "I dont no ware my backpak is.",
]

CSS = """
@import url('https://fonts.cdnfonts.com/css/opendyslexic');

.gradio-container {
    max-width: 800px !important;
    margin: auto !important;
}
textarea {
    font-family: 'OpenDyslexic', sans-serif !important;
    font-size: 20px !important;
    line-height: 1.8 !important;
}
.output-text {
    font-family: 'OpenDyslexic', sans-serif !important;
    font-size: 20px !important;
    line-height: 1.8 !important;
}
"""

with gr.Blocks(theme=gr.themes.Base(primary_hue="purple"), css=CSS) as demo:
    gr.Markdown(
        """
        # Dyslexic Writer
        ### A spelling tool that helps without doing the work for you

        Type (or paste) a sentence with spelling mistakes and click **Check Spelling**.
        The model is a fine-tuned Qwen3-4B trained on 192 000 misspelling examples — 98% accuracy.
        """
    )

    with gr.Row():
        inp = gr.Textbox(
            label="Your writing",
            placeholder="Type a sentence here...",
            lines=4,
        )

    btn = gr.Button("Check Spelling", variant="primary", size="lg")

    with gr.Row():
        out_corrected = gr.Textbox(
            label="Corrected",
            lines=4,
            interactive=False,
            elem_classes=["output-text"],
        )

    out_details = gr.Textbox(label="What changed", lines=4, interactive=False)

    btn.click(fn=correct, inputs=inp, outputs=[out_corrected, out_details])

    gr.Examples(examples=EXAMPLES, inputs=inp, label="Try these examples")

    gr.Markdown(
        """
        ---
        *Built for a science fair — a free, open-source tool for dyslexic kids.*
        *[GitHub](https://github.com/jburnford/dyslexic-writer)
        · Model: [jburnford/dyslexic-writer-qwen3-4b](https://huggingface.co/jburnford/dyslexic-writer-qwen3-4b)*
        """
    )

demo.launch()
