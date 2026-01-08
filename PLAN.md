# Dyslexic Writer - Development Plan

A lightweight writing tool for dyslexic kids that helps with spelling without doing the work for them.

## Core Philosophy

1. **Scaffold, don't replace** - Show candidates, let the child choose
2. **Phonetic awareness** - Accept "close enough" spelling, explain the correction
3. **Audio reinforcement** - Read words aloud to distinguish similar-looking words
4. **Fast feedback** - Sub-100ms spelling suggestions (no waiting for LLM)
5. **Works offline** - Everything runs locally, no internet required

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Tauri v2 Shell                           │
│  (Rust backend + OS webview, ~10MB vs Electron's 150MB+)        │
├─────────────────────────────────────────────────────────────────┤
│                         Frontend (React)                        │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    TipTap Editor                          │  │
│  │  - ProseMirror-based, headless                            │  │
│  │  - Custom decorations for spelling highlights             │  │
│  │  - Word-level selection for TTS                           │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              │                                   │
│  ┌───────────────────────────▼───────────────────────────────┐  │
│  │              Spelling Service (TypeScript)                │  │
│  │                                                           │  │
│  │  Layer 1: SymSpell (instant, edit-distance)               │  │
│  │     ↓ candidates                                          │  │
│  │  Layer 2: Double Metaphone (phonetic sound-alikes)        │  │
│  │     ↓ ranked candidates                                   │  │
│  │  Layer 3: LLM Context (only for homophones/ambiguous)     │  │
│  │                                                           │  │
│  └───────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                      Rust Sidecar Services                      │
│                                                                 │
│  ┌─────────────────────┐    ┌─────────────────────────────┐    │
│  │   llama.cpp         │    │        Piper TTS            │    │
│  │   (Llama 3.2 1B)    │    │   (local neural voices)     │    │
│  │                     │    │                             │    │
│  │  - Context resolver │    │  - Word pronunciation       │    │
│  │  - Homophone picker │    │  - Sentence reading         │    │
│  │  - Grammar hints    │    │  - Phonetic feedback        │    │
│  └─────────────────────┘    └─────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. App Shell: Tauri v2

**Why Tauri over Electron:**
- Binary size: ~10MB vs 150MB+
- RAM usage: ~50MB vs 300MB+
- Uses OS webview (no bundled Chromium)
- Rust backend = safe sidecar management
- Cross-platform (Windows, Mac, Linux)

**Tauri handles:**
- Window management
- File system access (save/load documents)
- Spawning/managing llama.cpp and Piper processes
- IPC between frontend and Rust services

### 2. Editor: TipTap (Headless)

**Why TipTap:**
- Built on ProseMirror (battle-tested)
- Headless = complete UI control
- Extension system for custom behavior
- React-friendly but framework-agnostic

**Custom extensions needed:**
- `SpellingDecoration` - Underline misspelled words
- `WordSelector` - Click word to hear it / see suggestions
- `SuggestionPopup` - Show candidates with audio buttons
- `PhoneticHint` - Display "sounds like: X"

### 3. Spelling Pipeline

```
User types: "I went to the stor to by sum fud"
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ Layer 1: SymSpell (edit distance)                       │
│                                                         │
│ "stor" → candidates: [store, story, stir, star]        │
│ "by"   → candidates: [by, be, buy, bye]  ← AMBIGUOUS   │
│ "fud"  → candidates: [fud, fed, mud]     ← NO MATCH    │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ Layer 2: Double Metaphone (phonetic)                    │
│                                                         │
│ "fud" sounds like "FT" → matches "food" (FT)           │
│ Now: [food, feud]                                       │
│                                                         │
│ "stor" sounds like "STR" → confirms "store"            │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼ (only for ambiguous cases)
┌─────────────────────────────────────────────────────────┐
│ Layer 3: LLM Context Resolution                         │
│                                                         │
│ Prompt: "Which word fits: 'went to the store to ___ '  │
│          Options: by, buy, bye"                         │
│                                                         │
│ Response: "buy" (purchasing context)                    │
└─────────────────────────────────────────────────────────┘
```

**Key insight:** LLM is only called for homophones/context-dependent cases. 90%+ of corrections happen in <10ms with SymSpell + Metaphone.

### 4. LLM Sidecar: llama.cpp + Llama 3.2 1B

**Model options (in order of preference):**

| Model | Size | RAM | Why |
|-------|------|-----|-----|
| Llama 3.2 1B Q4 | ~700MB | ~1.5GB | Best quality at tiny size |
| SmolLM2 1.7B | ~1GB | ~2GB | Good alternative |
| Qwen2.5 0.5B | ~400MB | ~1GB | Even smaller, still capable |

**Use cases:**
1. **Homophone resolution** - "their/there/they're" in context
2. **Grammar suggestions** - "He go to school" → "goes"
3. **Phonetic explanation** - "Why does 'enough' sound like 'enuff'?"

**NOT used for:**
- Raw spell checking (too slow, 200ms+ vs 10ms)
- Rewriting sentences (violates "no easy button" principle)

### 5. TTS: Piper

**Why Piper:**
- Fast: <50ms latency for single words
- Quality: Neural voices, not robotic
- Offline: No internet required
- Small: ~50MB per voice model
- Battle-tested: Used in Home Assistant, Rhasspy

**Voice selection:**
- Clear, slightly slower voices for young readers
- US/UK English options
- Possibly kid-friendly voice if available

**TTS triggers:**
- Click any word to hear it
- Click speaker icon on suggestion popup
- "Read my sentence" button
- Automatic reading of correction ("You wrote 'fone', did you mean 'phone'?")

---

## User Experience Flow

### Design Principles (Dyslexia-Specific)

1. **No red underlines** - Red signals failure/anxiety. Use soft purple under-dots instead.
2. **OpenDyslexic font default** - Weighted letter bottoms prevent visual flipping (b/d, p/q).
3. **Two correction modes** - User chooses their workflow (see below).
4. **Hints, not answers** - Show definitions/context clues, never auto-correct.
5. **2-second pause trigger** - LLM only invoked after child stops typing (not per-keystroke).
6. **Bubble menu hijack** - Use TipTap's formatting menu pattern for spelling suggestions.

### Correction Modes (Settings Toggle)

Users can choose their correction workflow in Settings:

**Get It Done Mode** (default)
- Click dropdown to select correct spelling (like MS Word)
- Practical for homework, longer documents
- Still shows hints/definitions to reinforce learning

**Learning Mode** (toggle: "Disable click-to-replace")
- Dropdown disabled - must delete and retype the word
- Builds muscle memory through motor practice
- Best for dedicated practice sessions
- Parents/teachers can enable this for practice time

Both modes:
- Show the same suggestions with audio
- Display definitions so child understands the difference
- Track which words were corrected (optional progress feature)

Settings UI:
```
☑ Enable click-to-replace (uncheck for learning mode)
☑ Play sound on correction
☑ Show word definitions
☐ Track my progress
```

### Writing Flow

```
1. Child types: "I saw a brd in the tre"
                      ↓
2. Soft purple under-dots appear under "brd" and "tre"
   (NOT red wavy lines - those signal failure)
                      ↓
3. Child clicks "brd" (or selects word)
                      ↓
4. Dropdown popup appears:
   ┌────────────────────────────────────────┐
   │  Did you mean...                       │
   │                                        │
   │  🔊 bird - an animal with feathers  ←  │
   │  🔊 bred - past tense of breed         │
   │  🔊 bard - a poet who tells stories    │
   │                                        │
   │  [Tap 🔊 to hear each word]            │
   └────────────────────────────────────────┘
                      ↓
5. Child taps 🔊 next to "bird" - hears it spoken clearly
                      ↓
6. GET IT DONE MODE: Child clicks "bird" - word is replaced
   LEARNING MODE: Child must delete "brd" and retype "bird"
                      ↓
7. Purple dot disappears. Subtle positive feedback.
```

### Homophone Flow (The "Sound-Out" Feature)

This is the killer feature. Dyslexic processing often disconnects visual form from phoneme.

```
1. Child types: "I want to go to."
   (Intended: "too" as in "also")
                      ↓
2. After 2-second pause, system detects homophone in context
   "to" gets an orange under-dot (context issue, not misspelled)
                      ↓
3. Child clicks "to" OR hits hotkey (Ctrl+Space)
                      ↓
4. TTS speaks with emphasis:
   "I want to go [pause] TO. Note: 'To' means direction.
    Did you mean 'Too', as in 'also'?"
                      ↓
5. Popup shows:
   ┌────────────────────────────────────────┐
   │  These words sound the same:           │
   │                                        │
   │  🔊 to  - direction ("go to school")   │
   │  🔊 too - also, or very ("me too")     │
   │  🔊 two - the number 2                 │
   │                                        │
   │  💡 Hint: You're saying "also",        │
   │     so "too" fits better here!         │
   │                                        │
   │  [Keep as is]                          │
   └────────────────────────────────────────┘
                      ↓
6. Child must DELETE "to" and RETYPE "too"
   (Reinforces correct spelling through motor memory)
```

### "Read My Writing" Flow

```
1. Child finishes paragraph
                      ↓
2. Clicks "Read My Writing" button (or Ctrl+Enter)
                      ↓
3. Piper reads the paragraph aloud:
   - Current word highlighted as it's read
   - Child hears what they actually wrote
   - Often catches errors they missed visually
   - Slower pace option in settings
```

---

## Project Structure

```
dyslexic-writer/
├── src-tauri/                 # Rust backend
│   ├── src/
│   │   ├── main.rs           # Tauri entry point
│   │   ├── llm.rs            # llama.cpp integration
│   │   ├── tts.rs            # Piper integration
│   │   └── spelling.rs       # SymSpell + Metaphone (Rust impl)
│   ├── Cargo.toml
│   └── tauri.conf.json
│
├── src/                       # React frontend
│   ├── components/
│   │   ├── Editor/
│   │   │   ├── Editor.tsx
│   │   │   ├── SpellingExtension.ts
│   │   │   ├── SuggestionPopup.tsx
│   │   │   └── WordHighlight.tsx
│   │   ├── Toolbar/
│   │   │   ├── Toolbar.tsx
│   │   │   └── ReadAloudButton.tsx
│   │   └── Settings/
│   │       └── Settings.tsx
│   ├── services/
│   │   ├── spelling.ts       # Frontend spelling service
│   │   ├── tts.ts            # TTS wrapper
│   │   └── llm.ts            # LLM context queries
│   ├── hooks/
│   │   ├── useSpelling.ts
│   │   └── useTTS.ts
│   ├── App.tsx
│   └── main.tsx
│
├── dictionaries/              # Spelling data
│   ├── en_US.txt             # SymSpell dictionary
│   ├── phonetic_map.json     # Common dyslexic substitutions
│   └── homophones.json       # Homophone groups
│
├── models/                    # ML models (gitignored, downloaded on first run)
│   ├── llama-3.2-1b-q4.gguf
│   └── piper-en-us-amy.onnx
│
├── package.json
├── vite.config.ts
└── PLAN.md
```

---

## Code to Fork/Borrow From

### Editor Templates
- **TipTap Starter**: Clone a clean TipTap + React template
  - Hijack the Bubble Menu (normally Bold/Italic) for spelling suggestions
  - https://tiptap.dev/docs/examples/basics/default-text-editor

- **Zettlr**: Electron-based academic Markdown editor
  - Excellent architecture for "inspecting" text
  - Good reference for file handling patterns
  - https://github.com/Zettlr/Zettlr

### Spelling/Grammar
- **LanguageTool**: Open source (Java) - study their `grammar.xml` rules
  - Too heavy to run full server, but rules are valuable reference
  - Python wrapper available for prototyping

- **typo-js / simple-spellchecker**: Simple Node.js spell checking
  - Good for Phase 1 MVP before implementing SymSpell

### LLM Integration
- **Ollama**: Easiest way to manage local models
  - Just install Ollama, point app to `localhost:11434`
  - Handles model downloads, quantization, GPU detection
  - https://ollama.com

---

## Development Phases

### Phase 1: "Dumb" Editor (Weekend Project)

**Goal:** Working editor with basic spell check + TTS

**Tasks:**
- [ ] Clone Tauri v2 + React + TipTap template
- [ ] Install `typo-js` or `simple-spellchecker` for basic underlining
- [ ] Add "Read Selection" button using browser's `window.speechSynthesis` API
- [ ] Style with OpenDyslexic font, purple under-dots (not red)
- [ ] Basic file save/load (.md format)

**Deliverable:** Editor that marks misspelled words, can read text aloud

### Phase 2: Local LLM Brain

**Goal:** Add contextual intelligence via local LLM

**Tasks:**
- [ ] Download Llama 3.2 3B Instruct (GGUF format)
- [ ] Set up Ollama OR `python -m llama_cpp.server`
- [ ] Connect editor to local LLM server
- [ ] Trigger LLM only after 2-second typing pause
- [ ] Test homophone detection prompts

**Deliverable:** LLM provides context-aware suggestions on pause

### Phase 3: Pedagogical UI

**Goal:** Implement "no easy button" philosophy

**Tasks:**
- [ ] Create static homophone list (there/their/they're, to/too/two, etc.)
- [ ] Trigger LLM context check when homophone typed
- [ ] Orange highlighting for context issues (vs purple for spelling)
- [ ] Implement "Try Again" loop - child must DELETE and RETYPE
- [ ] Add hint popup with definitions (not corrections)
- [ ] Subtle positive feedback when word corrected

**Deliverable:** Child learns through retyping, not clicking

### Phase 4: Better Spelling + TTS

**Goal:** Production-quality spelling and voice

**Tasks:**
- [ ] Replace typo-js with SymSpell (JS port or Rust)
- [ ] Add Double Metaphone for phonetic matching
- [ ] Replace browser TTS with Piper (neural voices)
- [ ] Word highlighting during read-aloud
- [ ] "Sound-out" feature with emphasis on errors

**Deliverable:** Professional-grade spelling + natural TTS

### Phase 5: Polish & Ship

**Goal:** Ready for real users

**Tasks:**
- [ ] Progress tracking (optional, opt-in)
- [ ] High-contrast and dark themes
- [ ] Font selection (OpenDyslexic, Lexie Readable, etc.)
- [ ] First-run tutorial/onboarding
- [ ] Settings persistence
- [ ] Package for Windows/Mac/Linux distribution
- [ ] Parent/teacher dashboard (future)

---

## Technical Decisions

### Why not use an LLM for all spelling?

1. **Latency**: LLMs take 200-500ms even for small models. SymSpell takes <10ms.
2. **Hallucination**: LLMs sometimes "correct" words incorrectly
3. **Overkill**: 90% of spelling errors are simple edit-distance fixes
4. **Resource usage**: Constant LLM inference drains battery

**Solution**: Tiered system where LLM is only consulted for ambiguous cases.

### Why SymSpell over other spell checkers?

- **Speed**: O(1) lookup via pre-computed delete combinations
- **Accuracy**: Handles up to 2-3 edit distance errors
- **Memory**: ~50MB for full English dictionary
- **No dependencies**: Pure algorithm, easy to port

### Why Double Metaphone specifically?

- Handles English phonetic irregularities better than Soundex
- Produces two codes (primary + alternate) for words with multiple pronunciations
- Well-documented, implementations available in many languages

### Why local LLM vs API?

- **Privacy**: Kids' writing shouldn't go to cloud servers
- **Offline**: Works without internet
- **Cost**: No API fees
- **Speed**: Local inference is faster than network round-trip for small models

---

## Open Questions

1. **Dictionary source**: Use standard dictionary or build dyslexia-specific one?
2. **Model distribution**: Bundle models or download on first run?
3. **Platform priority**: Start with Mac (best Tauri support) or Windows (most users)?
4. **Accessibility**: Screen reader compatibility requirements?
5. **Age range**: Targeting 7-10? 10-14? Different UI needs.

---

## Resources & References

### Similar Projects to Study
- [Ghotit](https://www.ghotit.com/) - Commercial dyslexia writing tool
- [Read&Write](https://www.texthelp.com/products/read-and-write-education/) - Texthelp's literacy software
- [Co:Writer](https://www.donjohnston.com/cowriter/) - Word prediction for struggling writers

### Technical Resources
- [SymSpell](https://github.com/wolfgarbe/SymSpell) - Original C# implementation
- [TipTap](https://tiptap.dev/) - Editor framework
- [Tauri](https://tauri.app/) - App framework
- [Piper](https://github.com/rhasspy/piper) - Neural TTS
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - LLM inference

### Research
- [Phonetic Spell Checking for Dyslexia](https://www.assistivetechnologycenter.org/spell-checking-tools-for-phonetic-spellers-and-individuals-with-dyslexia/)
- [Amazon RAG Spelling Correction](https://arxiv.org/html/2410.11655v1) - Hybrid approach validation

---

## LLM Prompt Strategy

### System Prompt for Homophone Detection

```
You are a reading tutor for a dyslexic child. Your job is to HINT, never correct.

The user has typed the sentence: '{sentence}'
They have focused on the word: '{word}'

Rules:
1. Identify if the word is a homophone error (there/their/they're, to/too/two, etc.)
2. If it IS an error, provide a 5-word-max definition for the CORRECT word
3. DO NOT output the corrected sentence
4. DO NOT say "you should use X"
5. Only output the hint in this format:

If error detected:
HINT: "[correct_word]" means [5-word definition]

If no error:
OK: Word fits the context.
```

### Example Interactions

**Input:**
- Sentence: "I want to go to."
- Focused word: "to" (second occurrence)

**Output:**
```
HINT: "too" means also, or very much
```

**Input:**
- Sentence: "Their going to the store."
- Focused word: "Their"

**Output:**
```
HINT: "They're" means they are (contraction)
```

**Input:**
- Sentence: "I saw a bird in the tree."
- Focused word: "bird"

**Output:**
```
OK: Word fits the context.
```

### TTS Script Generation Prompt

For the "Sound-Out" feature, generate emphasized read-aloud:

```
You are generating a read-aloud script for a dyslexic child.

Sentence: '{sentence}'
Potential error word: '{word}'
Correct word (if error): '{correction}'

Generate a script that:
1. Reads the sentence normally up to the error
2. PAUSES before the error word
3. Speaks the error word clearly and slowly
4. Explains what that word means
5. Suggests what they might have meant

Format: Use [pause] for pauses, CAPS for emphasis.

Example output:
"I want to go [pause] TO. TO means direction, like going TO a place. Did you mean TOO, as in also?"
```

---

## Common Homophones List

Start with these high-frequency confusions:

```json
{
  "homophones": [
    {
      "words": ["there", "their", "they're"],
      "definitions": {
        "there": "a place (over there)",
        "their": "belonging to them",
        "they're": "they are (contraction)"
      }
    },
    {
      "words": ["to", "too", "two"],
      "definitions": {
        "to": "direction (go to school)",
        "too": "also, or very much",
        "two": "the number 2"
      }
    },
    {
      "words": ["your", "you're"],
      "definitions": {
        "your": "belonging to you",
        "you're": "you are (contraction)"
      }
    },
    {
      "words": ["its", "it's"],
      "definitions": {
        "its": "belonging to it",
        "it's": "it is (contraction)"
      }
    },
    {
      "words": ["here", "hear"],
      "definitions": {
        "here": "this place",
        "hear": "to listen with ears"
      }
    },
    {
      "words": ["where", "wear", "were"],
      "definitions": {
        "where": "asking about place",
        "wear": "to put on clothes",
        "were": "past tense of are"
      }
    },
    {
      "words": ["no", "know"],
      "definitions": {
        "no": "opposite of yes",
        "know": "to understand something"
      }
    },
    {
      "words": ["by", "buy", "bye"],
      "definitions": {
        "by": "near, or done by someone",
        "buy": "to purchase something",
        "bye": "short for goodbye"
      }
    },
    {
      "words": ["right", "write"],
      "definitions": {
        "right": "correct, or direction",
        "write": "to put words on paper"
      }
    },
    {
      "words": ["which", "witch"],
      "definitions": {
        "which": "asking about choice",
        "witch": "magical person in stories"
      }
    }
  ]
}
```
