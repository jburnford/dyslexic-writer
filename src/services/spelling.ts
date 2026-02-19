/**
 * Spelling correction service
 *
 * Hybrid approach:
 * 1. Phonetic matching (fast, no LLM) - catches most dyslexic spellings
 * 2. LLM fallback (slower) - for ambiguous cases or unknown words
 */

import { findPhoneticMatches, isLikelyMisspelled } from './phonetic'

const FLASK_API_URL = 'http://127.0.0.1:5000/correct'

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
 * Uses Flask API which calls the Ollama model
 */
export async function checkSpelling(sentence: string): Promise<Correction[]> {
  const corrections: Correction[] = []

  // Call Flask API directly for all corrections
  console.log(`[Spelling] Checking sentence: "${sentence}"`)
  const { corrections: apiCorrections, response } = await checkWithLLM(sentence)

  corrections.push(...apiCorrections)

  // Log corrections
  if (corrections.length > 0) {
    addLogEntry({
      timestamp: new Date().toISOString(),
      input: sentence,
      llmResponse: response,
      corrections: corrections.map(c => ({
        original: c.original,
        corrected: c.corrected,
        source: 'llm' as const
      })),
      success: true,
    })
  }

  return corrections
}

/**
 * Check spelling using LLM (via Flask API)
 */
async function checkWithLLM(sentence: string): Promise<{ corrections: Correction[], response: string }> {
  const corrections: Correction[] = []

  // Call Flask API for spelling check
  try {
    const response = await fetch(FLASK_API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: sentence }),
    })

    if (!response.ok) {
      console.error('Flask API error:', response.status)
      return { corrections: [], response: '' }
    }

    const data = await response.json()
    const correctedSentence = data.corrected || ''

    // The Flask API already provides the changes array
    if (data.changes && Array.isArray(data.changes)) {
      for (const [original, corrected] of data.changes) {
        const pos = sentence.toLowerCase().indexOf(original.toLowerCase())
        if (pos !== -1) {
          corrections.push({ original, corrected, position: pos })
          // Cache for next time
          cache.set(original.toLowerCase(), corrected)
        }
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
