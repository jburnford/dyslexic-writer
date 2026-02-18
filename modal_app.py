"""
Dyslexic Writer — Modal serverless GPU backend.
Runs the fine-tuned Qwen3-4B spelling correction model on a T4 GPU.
Pay-per-second (~$0.50/hr), cold starts ~30s.

Deploy:  modal deploy modal_app.py
Test:    modal serve modal_app.py
"""

import modal

app = modal.App("dyslexic-writer")

# Build the container image with all dependencies + model weights baked in
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "accelerate", "fastapi[standard]")
    .run_commands(
        # Pre-download model weights into the image so cold starts are fast
        "python3 -c \""
        "from transformers import AutoModelForCausalLM, AutoTokenizer; "
        "AutoTokenizer.from_pretrained('jburnford/dyslexic-writer-qwen3-4b', trust_remote_code=True); "
        "AutoModelForCausalLM.from_pretrained('jburnford/dyslexic-writer-qwen3-4b', trust_remote_code=True)"
        "\"",
    )
)


@app.cls(
    image=image,
    gpu="T4",
    scaledown_window=300,  # keep warm for 5 min after last request
)
@modal.concurrent(max_inputs=4)
class SpellingModel:
    @modal.enter()
    def load_model(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_id = "jburnford/dyslexic-writer-qwen3-4b"

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

        self.system_prompt = (
            "You are a spelling correction assistant. "
            "Fix only spelling and grammar errors. "
            "Do not change meaning, names, or correct text. "
            "If the text is already correct, return it unchanged."
        )

    @modal.method()
    def correct(self, text: str) -> dict:
        import difflib
        import re
        import torch

        text = text.strip()
        if not text:
            return {"original": text, "corrected": text, "changed": False, "changes": []}

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": text},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated = outputs[0][prompt_len:]
        corrected = self.tokenizer.decode(generated, skip_special_tokens=True).strip()

        # Strip Qwen3 think tags
        corrected = re.sub(r"<think>.*?</think>", "", corrected, flags=re.DOTALL).strip()

        # Word-level diff
        orig_words = text.split()
        corr_words = corrected.split()
        changes = []
        matcher = difflib.SequenceMatcher(None, orig_words, corr_words)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "replace":
                orig_phrase = " ".join(orig_words[i1:i2])
                corr_phrase = " ".join(corr_words[j1:j2])
                if orig_phrase != corr_phrase:
                    changes.append([orig_phrase, corr_phrase])

        return {
            "original": text,
            "corrected": corrected,
            "changed": text != corrected,
            "changes": changes,
        }


@app.function(image=image)
@modal.concurrent(max_inputs=10)
@modal.asgi_app()
def web():
    """ASGI web endpoint that wraps the GPU model class."""
    from fastapi import FastAPI, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse

    web_app = FastAPI()
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    model = SpellingModel()

    @web_app.get("/health")
    async def health():
        return {"status": "ok", "backend": "modal", "model": "qwen3-4b"}

    @web_app.post("/correct")
    async def correct(request: Request):
        data = await request.json()
        text = data.get("text", "").strip()
        if not text:
            return JSONResponse({"error": "Missing 'text' field"}, status_code=400)
        result = model.correct.remote(text)
        return result

    return web_app
