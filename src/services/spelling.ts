/**
 * Spelling correction service
 *
 * Hybrid approach:
 * 1. Phonetic matching (fast, no LLM) - catches most dyslexic spellings
 * 2. LLM fallback (slower) - for ambiguous cases or unknown words
 */

import { findPhoneticMatches, isLikelyMisspelled } from './phonetic'

const OLLAMA_URL = 'http://localhost:11434/api/generate'
const MODEL = 'phi4-mini'

const SYSTEM_PROMPT = `Fix spelling mistakes for a dyslexic child.

FIND AND FIX:
- Phonetic spellings: enuff->enough, fone->phone, becuase->because
- Missing letters: gameing->gaming, settup->setup, helllo->hello
- Missing apostrophes: Im->I'm, dont->don't, cant->can't
- Letter swaps: teh->the, freind->friend

RULES:
- ONLY output words that need fixing
- Keep original case (lowercase stays lowercase)
- Never add "(no change needed)" or explanations
- Never fix capitalization or grammar

FORMAT (exactly like this):
CHANGES: wrong1->right1, wrong2->right2

If nothing to fix:
CHANGES: none

EXAMPLES:
"i have enuff fud" -> CHANGES: enuff->enough, fud->food
"gameing settup becuase" -> CHANGES: gameing->gaming, settup->setup, becuase->because
"hello world" -> CHANGES: none`

// Valid words that should never be "corrected"
const VALID_WORDS = new Set([
  'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
  'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
  'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'its', "it's",
  'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her',
  'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their',
  'this', 'that', 'these', 'those', 'what', 'which', 'who', 'whom',
  'and', 'or', 'but', 'if', 'because', 'when', 'where', 'how', 'why',
  'for', 'to', 'of', 'in', 'on', 'at', 'by', 'with', 'from', 'as',
  'go', 'going', 'went', 'gone', 'come', 'coming', 'came', 'see', 'saw',
  'think', 'know', 'want', 'get', 'make', 'take', 'say', 'said',
  'cool', 'win', "won't", 'will', 'kids', 'never', 'always', 'reading',
  'writing', 'hard', 'easy', 'words', 'spelling', 'dyslexic', 'platinum',
])

export interface Correction {
  original: string
  corrected: string
  position: number // character position in text
}

// Simple cache for repeated misspellings
const cache = new Map<string, string>()

// Logging for analysis
export interface LogEntry {
  timestamp: string
  input: string
  llmResponse: string
  corrections: { original: string; corrected: string; source: 'phonetic' | 'llm' | 'cache' }[]
  success: boolean
}

const LOG_KEY = 'dyslexic-writer-log'

function loadLog(): LogEntry[] {
  try {
    const stored = localStorage.getItem(LOG_KEY)
    return stored ? JSON.parse(stored) : []
  } catch {
    return []
  }
}

function saveLog(entries: LogEntry[]): void {
  localStorage.setItem(LOG_KEY, JSON.stringify(entries))
}

function addLogEntry(entry: LogEntry): void {
  const log = loadLog()
  log.push(entry)
  // Keep last 100 entries
  if (log.length > 100) log.shift()
  saveLog(log)
}

export function getLog(): LogEntry[] {
  return loadLog()
}

export function clearLog(): void {
  localStorage.removeItem(LOG_KEY)
}

export function exportLog(): string {
  const log = loadLog()
  return JSON.stringify(log, null, 2)
}

export function getLogStats(): { total: number; phonetic: number; llm: number; cache: number } {
  const log = loadLog()
  let phonetic = 0, llm = 0, cache = 0

  for (const entry of log) {
    for (const c of entry.corrections) {
      if (c.source === 'phonetic') phonetic++
      else if (c.source === 'llm') llm++
      else if (c.source === 'cache') cache++
    }
  }

  return { total: phonetic + llm + cache, phonetic, llm, cache }
}

/**
 * Check a sentence for spelling errors
 * Uses phonetic matching first, then LLM for remaining words
 */
export async function checkSpelling(sentence: string): Promise<Correction[]> {
  const corrections: Correction[] = []
  const wordsNeedingLLM: string[] = []

  // Split into words, keeping track of positions
  const words = sentence.split(/(\s+)/)
  let pos = 0

  for (const word of words) {
    const clean = word.replace(/[^\w]/g, '').toLowerCase()

    if (!clean || clean.length < 2) {
      pos += word.length
      continue
    }

    // Skip valid words
    if (VALID_WORDS.has(clean)) {
      pos += word.length
      continue
    }

    // Check cache first
    if (cache.has(clean)) {
      corrections.push({
        original: clean,
        corrected: cache.get(clean)!,
        position: pos,
      })
      pos += word.length
      continue
    }

    // Try phonetic matching
    if (isLikelyMisspelled(clean)) {
      const match = findPhoneticMatches(clean)

      if (match.bestMatch && match.confidence > 0.5) {
        // High confidence phonetic match - use it
        console.log(`[Phonetic] ${clean} -> ${match.bestMatch} (${(match.confidence * 100).toFixed(0)}%)`)
        corrections.push({
          original: clean,
          corrected: match.bestMatch,
          position: pos,
        })
        cache.set(clean, match.bestMatch)
      } else if (match.candidates.length > 0) {
        // Low confidence - need LLM to pick from candidates
        console.log(`[Phonetic] ${clean} -> ambiguous, candidates: ${match.candidates.slice(0, 3).join(', ')}`)
        wordsNeedingLLM.push(clean)
      } else {
        // No phonetic match - need LLM
        console.log(`[Phonetic] ${clean} -> no match, sending to LLM`)
        wordsNeedingLLM.push(clean)
      }
    }

    pos += word.length
  }

  // If we have words that need LLM, send the whole sentence
  let llmResponse = ''
  if (wordsNeedingLLM.length > 0) {
    console.log(`[LLM] Checking ${wordsNeedingLLM.length} words: ${wordsNeedingLLM.join(', ')}`)
    const { corrections: llmCorrections, response } = await checkWithLLM(sentence)
    llmResponse = response
    for (const c of llmCorrections) {
      // Avoid duplicates
      if (!corrections.find(existing => existing.original.toLowerCase() === c.original.toLowerCase())) {
        corrections.push(c)
      }
    }
  }

  // Log all corrections with their sources
  if (corrections.length > 0) {
    addLogEntry({
      timestamp: new Date().toISOString(),
      input: sentence,
      llmResponse: llmResponse,
      corrections: corrections.map(c => ({
        original: c.original,
        corrected: c.corrected,
        source: cache.has(c.original.toLowerCase()) ? 'cache' as const :
                wordsNeedingLLM.includes(c.original.toLowerCase()) ? 'llm' as const : 'phonetic' as const
      })),
      success: true,
    })
  }

  return corrections
}

/**
 * Check spelling using LLM (fallback for phonetic misses)
 */
async function checkWithLLM(sentence: string): Promise<{ corrections: Correction[], response: string }> {
  const corrections: Correction[] = []

  // Call Ollama for spelling check
  try {
    const response = await fetch(OLLAMA_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: MODEL,
        prompt: `Fix spelling: "${sentence}"`,
        system: SYSTEM_PROMPT,
        stream: false,
        options: {
          temperature: 0.1,
          num_predict: 150,
          num_gpu: 20,
        },
      }),
    })

    if (!response.ok) {
      console.error('Ollama error:', response.status)
      return { corrections: [], response: '' }
    }

    const data = await response.json()
    const result = data.response || ''

    // Parse the response
    const changes = parseResponse(result)

    // Find positions and build corrections
    for (const [original, corrected] of changes) {
      const pos = sentence.toLowerCase().indexOf(original.toLowerCase())
      if (pos !== -1) {
        corrections.push({ original, corrected, position: pos })
        // Cache for next time
        cache.set(original.toLowerCase(), corrected)
      }
    }

    return { corrections, response: result }
  } catch (error) {
    console.error('Spelling check failed:', error)
    return { corrections: [], response: String(error) }
  }
}

/**
 * Parse LLM response into corrections
 */
function parseResponse(response: string): [string, string][] {
  const changes: [string, string][] = []

  // Only look at first line containing CHANGES
  const lines = response.split('\n')
  let changesLine = ''
  for (const line of lines) {
    if (line.toUpperCase().includes('CHANGES:')) {
      changesLine = line
      break
    }
  }

  const match = changesLine.match(/CHANGES:\s*(.+)/i)
  if (!match) return changes

  const changesStr = match[1].trim()
  if (changesStr.toLowerCase() === 'none') return changes

  // Parse "word1->correction1, word2->correction2"
  for (const change of changesStr.split(',')) {
    if (change.includes('->')) {
      const parts = change.trim().split('->')
      if (parts.length === 2) {
        let original = parts[0].trim()
        let corrected = parts[1].trim()

        // Remove any parenthetical notes like "(no change needed)"
        corrected = corrected.replace(/\s*\(.*\).*$/, '').trim()

        // Strip punctuation from edges
        original = original.replace(/^[^\w']+|[^\w']+$/g, '')
        corrected = corrected.replace(/^[^\w']+|[^\w']+$/g, '')

        // Skip invalid corrections
        if (!original || !corrected) continue
        if (original.toLowerCase() === corrected.toLowerCase()) continue
        if (original.includes(' ')) continue // Skip multi-word "corrections"
        if (VALID_WORDS.has(original.toLowerCase())) continue

        changes.push([original, corrected])
      }
    }
  }

  return changes
}

/**
 * Clear the spelling cache
 */
export function clearCache(): void {
  cache.clear()
}
