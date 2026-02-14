# Dyslexic Writer - Local Model Setup

## Requirements

- 16 GB RAM
- 12 GB VRAM (NVIDIA GPU)
- [Ollama](https://ollama.com/) installed

## Available Models

| Model | Size | Bob Story Accuracy | Best For |
|-------|------|-------------------|----------|
| **v3-Qwen2.5-7B** (Q4_K_M) | 4.4 GB | 69% exact, 85% errors fixed, 0 FP | Best quality |
| v2-SmolLM2-1.7B (Q4_K_M) | 1.0 GB | 62% exact, 81% errors fixed, 1 FP | Low resource / fast |

Models are hosted (public, no login required) at: https://huggingface.co/jburnford/dyslexic-writer-spelling

## Quick Start with Ollama

### 1. Download the GGUF file

```bash
# Best model (recommended)
huggingface-cli download jburnford/dyslexic-writer-spelling v3-Qwen2.5-7B-q4_k_m.gguf --local-dir .

# Or the smaller model
huggingface-cli download jburnford/dyslexic-writer-spelling v2-SmolLM2-1.7B-q4_k_m.gguf --local-dir .
```

If you don't have `huggingface-cli`:
```bash
pip install huggingface-hub
```

### 2. Create an Ollama Modelfile

Create a file called `Modelfile`:

```
FROM ./v3-Qwen2.5-7B-q4_k_m.gguf

SYSTEM "You are a spelling correction assistant. Fix only spelling and grammar errors. Do not change meaning, names, or correct text. If the text is already correct, return it unchanged."

PARAMETER temperature 0.1
PARAMETER num_ctx 512
```

For the smaller model, change the `FROM` line:
```
FROM ./v2-SmolLM2-1.7B-q4_k_m.gguf
```

### 3. Build and run

```bash
# Create the model
ollama create dyslexic-writer -f Modelfile

# Test it
ollama run dyslexic-writer "Fix any spelling or grammar errors in this text. If there are no errors, return the text unchanged.

he walked out on to his belkany to get some fresh air"
```

Expected output: `he walked out on to his balcony to get some fresh air`

## Using with the App

The `app/` directory contains the web server that connects to Ollama:

```bash
cd app
pip install -r requirements.txt
python server.py
```

The server expects an Ollama model named `dyslexic-writer` running locally.

## Alternative: llama.cpp directly

If you prefer llama.cpp over Ollama:

```bash
# Install llama.cpp (or use a pre-built binary)
brew install llama.cpp   # macOS
# or download from https://github.com/ggerganov/llama.cpp/releases

# Run inference
llama-cli -m v3-Qwen2.5-7B-q4_k_m.gguf \
  -p "You are a spelling correction assistant.\n\nFix any spelling or grammar errors: he skremed at the hights" \
  --temp 0.1 -n 100
```

## VRAM Usage

| Model | VRAM at Idle | VRAM During Inference |
|-------|-------------|----------------------|
| v3-Qwen2.5-7B Q4_K_M | ~4.5 GB | ~5-6 GB |
| v2-SmolLM2-1.7B Q4_K_M | ~1.1 GB | ~1.5 GB |

Both fit comfortably within 12 GB VRAM.

## Model Performance (Bob Story Eval)

13 sentences of real dyslexic writing, 26 total errors across phonological, orthographic, and morphological categories.

### v3-Qwen2.5-7B (recommended)
- Exact match: 9/13 (69.2%)
- Errors fixed: 22/26 (84.6%)
- False positives: 0
- Phonological: 92.3% | Orthographic: 87.5% | Morphological: 60.0%

### v2-SmolLM2-1.7B
- Exact match: 8/13 (61.5%)
- Errors fixed: 21/26 (80.8%)
- False positives: 1
- Phonological: 84.6% | Orthographic: 75.0% | Morphological: 80.0%

## Troubleshooting

- **"model not found"**: Make sure you ran `ollama create` from the directory containing the GGUF file
- **Out of memory**: Try the smaller v2-SmolLM2-1.7B model (1 GB)
- **Slow inference**: Ensure Ollama is using your GPU (`ollama ps` to check)
