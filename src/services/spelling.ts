/**
 * Spelling correction service
 *
 * Hybrid approach:
 * 1. Phonetic matching (fast, no LLM) - catches most dyslexic spellings
 * 2. LLM fallback (slower) - for ambiguous cases or unknown words
 */

import { findPhoneticMatches, isLikelyMisspelled } from './phonetic'

const OLLAMA_URL = 'http://localhost:11434/api/generate'
const MODEL = 'dyslexic-speller'

const SYSTEM_PROMPT = `You are a spelling correction assistant.`

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
        prompt: `Fix the spelling mistakes in this sentence. Only output the corrected sentence.\n\n${sentence}`,
        system: SYSTEM_PROMPT,
        stream: false,
        options: {
          temperature: 0.1,
          num_predict: 150,
        },
      }),
    })

    if (!response.ok) {
      console.error('Ollama error:', response.status)
      return { corrections: [], response: '' }
    }

    const data = await response.json()
    const correctedSentence = (data.response || '').trim()

    // Compare input and output to find corrections
    const changes = findDifferences(sentence, correctedSentence)

    // Build corrections with positions
    for (const [original, corrected] of changes) {
      const pos = sentence.toLowerCase().indexOf(original.toLowerCase())
      if (pos !== -1) {
        corrections.push({ original, corrected, position: pos })
        // Cache for next time
        cache.set(original.toLowerCase(), corrected)
      }
    }

    return { corrections, response: correctedSentence }
  } catch (error) {
    console.error('Spelling check failed:', error)
    return { corrections: [], response: String(error) }
  }
}

/**
 * Find word-level differences between original and corrected sentences
 * Uses fuzzy matching to handle slight position shifts
 */
function findDifferences(original: string, corrected: string): [string, string][] {
  const changes: [string, string][] = []

  // Clean up model output - strip common prefixes
  let cleanCorrected = corrected
    .replace(/^(here'?s?\s+(the\s+)?corrected\s+(sentence|text)[:\s]*)/i, '')
    .replace(/^(corrected[:\s]*)/i, '')
    .trim()

  // Split into words
  const originalWords = original.split(/\s+/)
  const correctedWords = cleanCorrected.split(/\s+/)

  // If word counts differ too much, model may have hallucinated
  if (Math.abs(originalWords.length - correctedWords.length) > 3) {
    console.warn('[LLM] Word count mismatch, skipping corrections')
    return changes
  }

  // For each original word, find its best match in corrected sentence
  for (let i = 0; i < originalWords.length; i++) {
    const origWord = originalWords[i]
    const origClean = origWord.replace(/^[^\w']+|[^\w']+$/g, '').toLowerCase()

    if (!origClean || origClean.length < 2) continue
    if (VALID_WORDS.has(origClean)) continue

    // Look for the corresponding word in corrected (allow position drift of ±2)
    const searchStart = Math.max(0, i - 2)
    const searchEnd = Math.min(correctedWords.length, i + 3)

    let bestMatch: string | null = null
    let bestScore = 0

    for (let j = searchStart; j < searchEnd; j++) {
      const corrWord = correctedWords[j]
      const corrClean = corrWord.replace(/^[^\w']+|[^\w']+$/g, '').toLowerCase()

      if (!corrClean) continue

      // Calculate similarity score
      const similarity = stringSimilarity(origClean, corrClean)

      // Prefer words at same position, but accept similar words nearby
      const positionBonus = (i === j) ? 0.1 : 0
      const score = similarity + positionBonus

      if (score > bestScore && similarity > 0.3) {
        bestScore = score
        bestMatch = corrClean
      }
    }

    // If we found a match that's different from original, it's a correction
    if (bestMatch && bestMatch !== origClean) {
      // Extract word including common punctuation mistakes (comma for apostrophe)
      const origText = origWord.match(/[a-zA-Z',]+/)
      if (origText) {
        // Preserve original case pattern in correction
        const corrText = matchCase(origText[0], bestMatch)
        changes.push([origText[0], corrText])
      }
    }
  }

  return changes
}

/**
 * Calculate string similarity (0-1) using Levenshtein-based metric
 */
function stringSimilarity(a: string, b: string): number {
  if (a === b) return 1
  if (!a || !b) return 0

  const maxLen = Math.max(a.length, b.length)
  if (maxLen === 0) return 1

  // Simple Levenshtein distance
  const matrix: number[][] = []
  for (let i = 0; i <= a.length; i++) {
    matrix[i] = [i]
  }
  for (let j = 0; j <= b.length; j++) {
    matrix[0][j] = j
  }
  for (let i = 1; i <= a.length; i++) {
    for (let j = 1; j <= b.length; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1
      matrix[i][j] = Math.min(
        matrix[i - 1][j] + 1,
        matrix[i][j - 1] + 1,
        matrix[i - 1][j - 1] + cost
      )
    }
  }

  const distance = matrix[a.length][b.length]
  return 1 - distance / maxLen
}

/**
 * Match the case pattern of the original word
 */
function matchCase(original: string, corrected: string): string {
  if (original === original.toUpperCase()) {
    return corrected.toUpperCase()
  }
  if (original[0] === original[0].toUpperCase()) {
    return corrected.charAt(0).toUpperCase() + corrected.slice(1)
  }
  return corrected
}

/**
 * Clear the spelling cache
 */
export function clearCache(): void {
  cache.clear()
}
