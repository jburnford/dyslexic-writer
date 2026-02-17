---
title: Dyslexic Writer
emoji: ✏️
colorFrom: purple
colorTo: indigo
sdk: gradio
sdk_version: "5.0"
app_file: app.py
pinned: false
license: mit
hardware: zero-a10g
---

# Dyslexic Writer

A free spelling correction tool for dyslexic kids that helps without doing the work for them.

Uses a fine-tuned **Qwen3-4B** model (98% accuracy) trained on 192,000 misspelling examples from real-world corpora including schoolchildren's writing and dyslexia-specific patterns.

## How it works

1. Type a sentence with spelling mistakes
2. Click "Check Spelling"
3. See the corrected text with changes highlighted

The model fixes spelling and grammar errors while preserving meaning, names, and writing style.

## Links

- [GitHub Repository](https://github.com/jburnford/dyslexic-writer)
- [Training Details](https://github.com/jburnford/dyslexic-writer/blob/main/TRAINING_PROGRESS.md)
- [Model Weights](https://huggingface.co/jburnford/dyslexic-writer-qwen3-4b)
