# Dyslexic Writer

A free, open-source writing tool for dyslexic kids that helps with spelling without doing the work for them.

**Goal**: Create a free alternative to expensive commercial dyslexia tools.

## Current Status: Prototype

Working editor with:
- Hybrid phonetic + LLM spelling correction
- Dark mode default (white on black) with light mode option
- OpenDyslexic font default with standard font option
- Large font size (20px vs Word's 12pt)
- Purple dot underlines (not anxiety-inducing red)
- Text-to-speech for words and sentences
- Learning mode (must retype) vs Get It Done mode (click to replace)
- Logging for analysis and model improvement

## Architecture

```
User types sentence
        │
    Ends with period
        ▼
┌─────────────────────────┐
│  Hybrid Spell Check     │
│                         │
│  1. Cache lookup        │ → instant
│  2. Phonetic matching   │ → <1ms (Double Metaphone)
│  3. LLM fallback        │ → ~1-2s (phi4-mini via Ollama)
└─────────────────────────┘
        │
        ▼
   Purple dot highlights
        │
        ▼
   Click word → suggestion popup with audio
```

### Why Hybrid?

Traditional spell checkers use edit distance, which fails for dyslexic spelling:
- "becuase" → edit distance says "because" is 2 edits away, but so is "becalms"
- "gameing" → might match "gaming" or "gamine"
- "enuff" → nowhere near "enough" by edit distance

Our approach:
1. **Phonetic matching** catches most dyslexic spellings because kids spell how words SOUND
2. **LLM fallback** uses sentence context for ambiguous cases

## Tech Stack

| Component | Choice | Why |
|-----------|--------|-----|
| Frontend | React + Vite | Fast dev, hot reload |
| Editor | TipTap | Headless, full control |
| Phonetics | Double Metaphone | Matches words by sound |
| LLM | Ollama + phi4-mini | Local, private, no API costs |
| TTS | Web Speech API | Built-in, works offline |

## Training Data

We've collected **93,115 unique** misspelling → correct word pairs for fine-tuning:

| Source | Pairs | Description |
|--------|-------|-------------|
| GitHub Typo Corpus | 51,140 | Real typos from commit messages |
| Birkbeck corpus | 35,995 | Poor spellers, native English |
| Extra misspellings | 4,200 | Various sources |
| Holbrook corpus | 1,780 | Schoolchildren's writing |

Located in `training-data/`:
- `all_instruction.jsonl` - Combined dataset for instruction fine-tuning
- `all_changes.jsonl` - In our CHANGES format
- `all_pairs.csv` - Raw pairs with source attribution

## Development Plan

### Phase 1: Working Prototype ✅
- [x] TipTap editor with dyslexia-friendly styling
- [x] Ollama integration for LLM spelling
- [x] Phonetic matching with Double Metaphone
- [x] Suggestion popup with audio
- [x] Logging for analysis
- [x] Dark mode, font options

### Phase 2: Fine-tuned Model (In Progress)
- [x] Fine-tune small models on spelling data (H100 training run)
- [x] Evaluate accuracy - SmolLM2-1.7B achieves 97%
- [ ] Integrate fine-tuned model into app

See [TRAINING_PROGRESS.md](TRAINING_PROGRESS.md) for detailed results.

### Phase 3: Polish
- [ ] Piper TTS (better voices than Web Speech API)
- [ ] File save/load
- [ ] Progress tracking (optional)
- [ ] Tauri packaging for desktop app

### Phase 4: Ship
- [ ] Package for Windows/Mac/Linux
- [ ] Landing page at dyslexi.co
- [ ] User testing with real dyslexic kids

## Setup

### Requirements
- Node.js 18+
- Ollama with phi4-mini model

### Install

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull the model
ollama pull phi4-mini

# Install dependencies
npm install

# Run dev server
npm run dev
```

Open http://localhost:3000

### Testing the Spell Checker

Type a sentence ending with a period:
```
I have enuff food becuase Im hungrey.
```

Expected corrections:
- enuff → enough (phonetic match)
- becuase → because (phonetic match)
- Im → I'm (LLM)
- hungrey → hungry (phonetic match)

## Project Structure

```
dyslexic-writer/
├── src/
│   ├── components/
│   │   └── Editor.tsx      # Main editor component
│   ├── services/
│   │   ├── spelling.ts     # Hybrid spell checker
│   │   └── phonetic.ts     # Double Metaphone matching
│   ├── styles/
│   │   └── editor.css      # Dyslexia-friendly styles
│   ├── App.tsx
│   └── main.tsx
├── training-data/
│   ├── birkbeck.dat        # Birkbeck corpus (raw)
│   ├── holbrook.dat        # Holbrook corpus (raw)
│   ├── combined_*.jsonl    # Training data formats
│   └── prepare_training_data.py
├── prototype/
│   └── tiered_spelling.py  # Python prototype
└── dictionaries/
    └── homophones.json     # Homophone definitions
```

## Key Design Decisions

### Why not just use LLM for everything?
- Too slow for real-time typing (~1-2s per check)
- Sometimes hallucinates corrections
- Overkill for obvious phonetic matches

### Why phonetic matching?
Dyslexic kids spell phonetically. "becuase" sounds like "because". Traditional edit-distance spell checkers miss this because they only count character differences.

### Why dark mode default?
Feedback from a dyslexic kid: white on black is easier to read. Many dyslexic users prefer high contrast and find bright white backgrounds harder.

### Why purple dots instead of red underlines?
Red signals failure and triggers anxiety. Purple is neutral and less stressful.

### Why OpenDyslexic font?
Weighted letter bottoms help prevent visual letter-flipping (b/d, p/q confusion). But some kids don't like it, so we offer a toggle.

## Resources

- [Birkbeck Spelling Corpora](https://titan.dcs.bbk.ac.uk/~roger/corpora.html)
- [Double Metaphone Algorithm](https://en.wikipedia.org/wiki/Metaphone#Double_Metaphone)
- [TipTap Editor](https://tiptap.dev/)
- [Ollama](https://ollama.com/)

## License

MIT

## Contributing

This is a science fair project that aims to become a free tool for dyslexic kids everywhere. Contributions welcome!

Areas where we need help:
- More training data (especially from dyslexic writers)
- Fine-tuning experiments
- UI/UX improvements
- Testing with real users
