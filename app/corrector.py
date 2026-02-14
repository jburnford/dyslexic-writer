"""
Spelling correction module.
Supports multiple backends: Ollama (local GGUF), Transformers (HF models), or API.
"""

import json
import re
from dataclasses import dataclass
from typing import Optional

SYSTEM_PROMPT = (
    "You are a spelling correction assistant. "
    "Fix only spelling and grammar errors. "
    "Do not change meaning, names, or correct text. "
    "If the text is already correct, return it unchanged."
)


@dataclass
class Correction:
    original: str
    corrected: str
    changed: bool

    @property
    def changes(self) -> list[tuple[str, str]]:
        """Return list of (original_word, corrected_word) pairs that differ."""
        orig_words = self.original.split()
        corr_words = self.corrected.split()
        diffs = []
        # Simple word-level diff
        max_len = max(len(orig_words), len(corr_words))
        for i in range(min(len(orig_words), len(corr_words))):
            if orig_words[i] != corr_words[i]:
                diffs.append((orig_words[i], corr_words[i]))
        return diffs


class OllamaCorrector:
    """Use a local Ollama model for correction."""

    def __init__(self, model: str = "smollm2-1.7b-spell", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def correct(self, text: str) -> Correction:
        import requests

        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                "stream": False,
                "options": {"temperature": 0.1, "top_p": 0.9},
            },
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()
        corrected = result["message"]["content"].strip()

        return Correction(
            original=text,
            corrected=corrected,
            changed=text != corrected,
        )


class TransformersCorrector:
    """Use a HuggingFace Transformers model directly."""

    def __init__(self, model_path: str, device: str = "auto"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            trust_remote_code=True,
        ).to(device)
        self.model.eval()

    def correct(self, text: str) -> Correction:
        import torch

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
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

        return Correction(
            original=text,
            corrected=corrected,
            changed=text != corrected,
        )


class PassthroughCorrector:
    """Dummy corrector that returns text unchanged. For testing TTS without a model."""

    def correct(self, text: str) -> Correction:
        return Correction(original=text, corrected=text, changed=False)


def create_corrector(backend: str = "ollama", **kwargs):
    """Factory function to create the right corrector backend."""
    if backend == "ollama":
        return OllamaCorrector(**kwargs)
    elif backend == "transformers":
        return TransformersCorrector(**kwargs)
    elif backend == "passthrough":
        return PassthroughCorrector()
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'ollama', 'transformers', or 'passthrough'.")
