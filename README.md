# Dyslexic Writer

A lightweight writing tool for dyslexic kids that helps with spelling without doing the work for them.

## Philosophy

Most spell checkers either auto-correct (removing the learning opportunity) or just underline errors in red (triggering anxiety). This tool takes a different approach:

- **Phonetic understanding** - Accepts "enuff", "sed", "fone" and figures out what the child meant
- **Hints with definitions** - Shows what each word means so the child learns, not just fixes
- **Two modes** - Click-to-replace for getting work done, or retype-only for practice
- **Audio feedback** - Read words aloud to distinguish similar-looking words
- **No red underlines** - Uses soft purple dots to avoid failure signals

### Correction Modes

| Mode | How it works | Best for |
|------|--------------|----------|
| **Get It Done** (default) | Click dropdown to replace, like MS Word | Homework, longer documents |
| **Learning Mode** | Must delete and retype the word | Practice sessions, building muscle memory |

Toggle in Settings: "Enable click-to-replace" (uncheck for learning mode)

## How It Works

1. Child types: "I want to go to the stor and by sum fud"
2. System highlights uncertain words with purple dots
3. Child clicks a word to see suggestions with audio
4. Child **deletes and retypes** the correct spelling (building motor memory)

## Tech Stack

| Component | Choice | Why |
|-----------|--------|-----|
| App Shell | Tauri v2 | Lightweight (~10MB vs Electron's 150MB) |
| Editor | TipTap | Headless, full control over UI |
| Spelling | SymSpell + LLM | Fast fuzzy matching + context awareness |
| LLM | Phi-4 Mini (via Ollama) | 96.7% accuracy on phonetic spelling, runs locally |
| TTS | Piper | Neural voices, works offline |

## Current Status

**Prototype phase** - Testing LLM models for phonetic spelling correction.

Benchmark results (Phi-4 Mini):
- Phonetic spelling: **96.7% accuracy** @ 0.25s response
- Homophones: 70% accuracy @ 1.2s response

## Setup

### Requirements

- Python 3.8+
- ~3GB disk space for Phi-4 Mini model
- GPU with 4GB+ VRAM recommended (runs on CPU too, just slower)

### Install Ollama & Phi-4 Mini

```bash
# 1. Install Ollama (Linux/Mac)
curl -fsSL https://ollama.com/install.sh | sh

# 2. Pull Phi-4 Mini (~2.5GB download)
ollama pull phi4-mini

# 3. Verify it's working
ollama run phi4-mini "What word did they mean: 'I have enuff food'"
# Should respond: "enough"
```

### Run the Prototype

```bash
# Test phonetic spelling correction
python3 prototype/benchmark_phonetic.py

# Interactive homophone testing
python3 prototype/homophone_hint.py
```

## Project Structure

```
dyslexic-writer/
├── PLAN.md                    # Full development plan
├── dictionaries/
│   └── homophones.json        # Homophone definitions
└── prototype/
    ├── homophone_hint.py      # Interactive homophone tester
    ├── benchmark_models.py    # Model comparison (homophones)
    └── benchmark_phonetic.py  # Phonetic spelling benchmark
```

## License

MIT
