/**
 * Dictionary API client
 * Uses Free Dictionary API (https://api.dictionaryapi.dev/) - no API key required
 */

export interface DictionaryDefinition {
  definition: string
  example?: string
}

export interface DictionaryMeaning {
  partOfSpeech: string
  definitions: DictionaryDefinition[]
}

export interface DictionaryWord {
  word: string
  phonetic?: string
  audioUrl?: string
  meanings: DictionaryMeaning[]
}

// Cache results to avoid repeated API calls
const cache = new Map<string, DictionaryWord[]>()

/**
 * Look up a word in the Free Dictionary API
 */
export async function lookupWord(word: string): Promise<DictionaryWord[]> {
  const key = word.toLowerCase().trim()
  if (!key) return []

  // Check cache first
  if (cache.has(key)) {
    return cache.get(key)!
  }

  try {
    const response = await fetch(
      `https://api.dictionaryapi.dev/api/v2/entries/en/${encodeURIComponent(key)}`
    )

    if (response.status === 404) {
      // Word not found - cache empty result
      cache.set(key, [])
      return []
    }

    if (!response.ok) {
      console.error('[Dictionary] API error:', response.status)
      return []
    }

    const data = await response.json()
    const results: DictionaryWord[] = data.map((entry: any) => {
      // Find the first available audio URL
      let audioUrl: string | undefined
      let phonetic: string | undefined

      if (entry.phonetic) {
        phonetic = entry.phonetic
      }

      if (entry.phonetics && Array.isArray(entry.phonetics)) {
        for (const p of entry.phonetics) {
          if (p.audio) {
            audioUrl = p.audio
          }
          if (p.text && !phonetic) {
            phonetic = p.text
          }
        }
      }

      return {
        word: entry.word,
        phonetic,
        audioUrl,
        meanings: (entry.meanings || []).map((m: any) => ({
          partOfSpeech: m.partOfSpeech,
          definitions: (m.definitions || []).slice(0, 2).map((d: any) => ({
            definition: d.definition,
            example: d.example,
          })),
        })),
      }
    })

    cache.set(key, results)
    return results
  } catch (error) {
    console.error('[Dictionary] Lookup failed:', error)
    return []
  }
}

/**
 * Clear the dictionary cache
 */
export function clearDictionaryCache(): void {
  cache.clear()
}
