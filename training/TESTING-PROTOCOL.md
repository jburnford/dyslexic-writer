# Pascal's Story — Spelling Correction Testing Protocol

## Test Story
File: `test_story` — Pascal's D&D campaign writeup (real child's writing, ~170 words, 66 identified errors)

## Gold Standards
- **Light edit** (`pascal_story_gold.txt`): Spelling/word corrections only. No sentence restructuring, no added punctuation, no capitalization changes at sentence starts.
- **Full edit** (`pascal_story_gold_full.txt`): Same as light edit + sentence-start capitalization and title case. No added periods or restructuring.

---

## Testing Our Models (automated)

Run: `python training/test_pascal_story.py dyslexic-writer-4b-q5 dyslexic-writer-1.7b-q8 dyslexic-writer-0.6b-q8`

This feeds each paragraph to each model via Ollama and compares against the light-edit gold standard.

---

## Testing Microsoft Word

### Setup
1. Open **Microsoft Word** (desktop version, not online)
2. Make sure proofing language is set to **English**
3. Create a **new blank document**

### Test Procedure
1. Open `test_story` in Notepad and **copy all text**
2. **Paste into Word** (Ctrl+V)
3. Wait 5-10 seconds for all underlines to appear
4. **Screenshot** the full document showing red/blue underlines — save as `word_screenshot.png`

### Record Each Suggestion
5. Go to **Review > Spelling & Grammar** (or press F7)
6. Word will walk through each flagged item. For EACH one:
   - Note the **original word** Word flagged
   - Note Word's **top suggestion** (or "no suggestion" if none)
   - Click **Change** to accept the top suggestion, OR **Ignore** if no suggestion
7. After going through everything, **select all text** (Ctrl+A) and **copy**
8. Open Notepad, paste, save as `test_story_word_output.txt` in the project root

### Important Notes
- Accept ONLY the **top suggestion** each time — don't pick from the list
- If Word offers no suggestion for a flagged word, click **Ignore** (the word stays unchanged)
- If Word flags a proper noun (Ronan, Atte, Mamoude), click **Ignore** — note it was flagged
- Do NOT manually fix anything — we want to see what Word does on its own

---

## Testing Grammarly

### Setup
1. Go to **https://app.grammarly.com/** in your browser
2. Log in (free account is fine)
3. Click **New** to create a new document
4. Set goals: **Audience: General**, **Formality: Informal**, **Domain: General**

### Test Procedure
1. Open `test_story` in Notepad and **copy all text**
2. **Paste into Grammarly** editor
3. Wait for all suggestions to load (number appears in bottom-left)
4. **Screenshot** the editor showing the suggestion count — save as `grammarly_screenshot.png`

### Record Each Suggestion
5. Click on each underlined section in order (left to right, top to bottom)
6. For EACH suggestion Grammarly shows:
   - Note the **original text** it flagged
   - Note Grammarly's **suggested replacement**
   - Note the **category** (Correctness, Clarity, Engagement, etc.)
   - Click the **green button** to accept
7. After accepting ALL suggestions, **select all text** and **copy**
8. Open Notepad, paste, save as `test_story_grammarly_output.txt` in the project root

### Important Notes
- Accept ALL suggestions — even ones you disagree with
- If Grammarly suggests deleting or restructuring a sentence, accept it
- Note if Grammarly flags proper nouns (Ronan, Atte, Mamoude) as errors
- Free Grammarly only shows "Correctness" suggestions; Premium shows all 4 categories

---

## After Testing

Put both output files in the project root:
```
dyslexic-writer/
  test_story                      <- original
  test_story_word_output.txt      <- Word's corrections
  test_story_grammarly_output.txt <- Grammarly's corrections
```

Then run the comparison script (we'll build this together after collecting all outputs).

---

## What We're Measuring

| Metric | What it means |
|--------|---------------|
| **Errors found** | How many of the 66 errors the tool flagged/changed |
| **Correctly fixed** | Changed AND matches our gold standard |
| **Missed** | Error that the tool didn't flag at all |
| **False positives** | Tool changed something that was already correct |
| **Hallucinations** | Tool flagged an error but gave the wrong correction |
| **Fix rate** | Correctly fixed / total errors (higher = better) |
