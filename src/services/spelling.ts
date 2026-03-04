/**
 * Spelling correction service
 *
 * Hybrid approach:
 * 1. Phonetic matching (fast, no LLM) - catches most dyslexic spellings
 * 2. LLM fallback (slower) - for ambiguous cases or unknown words
 */

// Backend URL: use local Flask when running on localhost, Modal when deployed
const MODAL_URL = 'https://dyslexic-writer--dyslexic-writer-web.modal.run'
const LOCAL_URL = 'http://127.0.0.1:5000'

function getBackendUrl(): string {
  const hostname = window.location.hostname
  if (hostname === 'localhost' || hostname === '127.0.0.1') {
    return LOCAL_URL
  }
  return MODAL_URL
}


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
 * Split text into chunks of ~2 sentences each for reliable model processing.
 * Keeps sentence boundaries intact.
 */
function splitIntoChunks(text: string, sentencesPerChunk: number): { text: string; offset: number }[] {
  const chunks: { text: string; offset: number }[] = []

  // Split on sentence endings (. ! ?) while keeping delimiters
  const sentenceRegex = /[^.!?]*[.!?]+\s*/g
  const sentences: { text: string; offset: number }[] = []
  let match
  let lastEnd = 0

  while ((match = sentenceRegex.exec(text)) !== null) {
    sentences.push({ text: match[0], offset: match.index })
    lastEnd = match.index + match[0].length
  }

  // Grab any remaining text without a sentence ending
  if (lastEnd < text.length) {
    const remaining = text.slice(lastEnd)
    if (remaining.trim()) {
      sentences.push({ text: remaining, offset: lastEnd })
    }
  }

  // If no sentences found, treat the whole thing as one chunk
  if (sentences.length === 0) {
    return [{ text, offset: 0 }]
  }

  // Group into pairs
  for (let i = 0; i < sentences.length; i += sentencesPerChunk) {
    const group = sentences.slice(i, i + sentencesPerChunk)
    const chunkText = group.map(s => s.text).join('')
    chunks.push({ text: chunkText, offset: group[0].offset })
  }

  return chunks
}

/**
 * Check text for spelling errors.
 * Splits into ~2 sentence chunks and calls onResults as each chunk completes.
 */
export async function checkSpelling(
  text: string,
  onResults?: (corrections: Correction[]) => void,
): Promise<Correction[]> {
  const allCorrections: Correction[] = []
  const seen = new Set<string>() // deduplicate by original word

  const chunks = splitIntoChunks(text, 2)
  console.log(`[Spelling] Split into ${chunks.length} chunks`)

  for (const chunk of chunks) {
    console.log(`[Spelling] Checking chunk (offset ${chunk.offset}): "${chunk.text.substring(0, 60)}..."`)

    const { corrections, response } = await checkWithLLM(chunk.text)

    const newCorrections: Correction[] = []
    for (const correction of corrections) {
      const key = correction.original.toLowerCase()
      if (!seen.has(key)) {
        seen.add(key)
        const adjusted = {
          ...correction,
          position: correction.position + chunk.offset,
        }
        allCorrections.push(adjusted)
        newCorrections.push(adjusted)
      }
    }

    // Notify caller with this chunk's results immediately
    if (newCorrections.length > 0 && onResults) {
      onResults(newCorrections)
    }

    // Log each chunk
    if (corrections.length > 0) {
      addLogEntry({
        timestamp: new Date().toISOString(),
        input: chunk.text,
        llmResponse: response,
        corrections: corrections.map(c => ({
          original: c.original,
          corrected: c.corrected,
          source: 'llm' as const
        })),
        success: true,
      })
    }
  }

  return allCorrections
}

/**
 * Check spelling using LLM (via Flask API)
 */
async function checkWithLLM(sentence: string): Promise<{ corrections: Correction[], response: string }> {
  const corrections: Correction[] = []

  // Call Flask API for spelling check
  try {
    const response = await fetch(`${getBackendUrl()}/correct`, {
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
 * Clear the spelling cache
 */
export function clearCache(): void {
  cache.clear()
}
