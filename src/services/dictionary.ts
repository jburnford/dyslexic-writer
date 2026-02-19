/**
 * Dictionary service
 *
 * Three sources:
 * 1. Free Dictionary API - has phonetics + audio, smaller coverage
 * 2. Open Dictionary (Wiktionary) - 260K+ words, definitions only, no audio
 * 3. Wikidata - proper nouns, places, people, etc.
 *
 * Strategy: Free Dictionary → Open Dictionary → Wikidata.
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

// Cache for Open Dictionary JSON files (keyed by two-letter prefix)
const openDictCache = new Map<string, Record<string, any>>()

// ===== Free Dictionary API =====

async function fetchFromFreeDictionary(word: string): Promise<DictionaryWord[]> {
  try {
    const response = await fetch(
      `https://api.dictionaryapi.dev/api/v2/entries/en/${encodeURIComponent(word)}`
    )

    if (response.status === 404 || !response.ok) {
      return []
    }

    const data = await response.json()
    return data.map((entry: any) => {
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
  } catch {
    return []
  }
}

// ===== Open Dictionary (Wiktionary via GitHub) =====

function stripHtml(text: string): string {
  return text.replace(/<[^>]*>/g, '')
}

async function fetchOpenDictFile(prefix: string): Promise<Record<string, any>> {
  if (openDictCache.has(prefix)) {
    return openDictCache.get(prefix)!
  }

  try {
    const dir = prefix[0]
    const response = await fetch(
      `https://raw.githubusercontent.com/mhollingshead/open-dictionary/main/api/${dir}/${prefix}.json`
    )

    if (!response.ok) {
      openDictCache.set(prefix, {})
      return {}
    }

    const data = await response.json()
    openDictCache.set(prefix, data)
    return data
  } catch {
    openDictCache.set(prefix, {})
    return {}
  }
}

async function fetchFromOpenDictionary(word: string): Promise<DictionaryWord[]> {
  if (word.length < 2) return []

  const prefix = word.slice(0, 2).toLowerCase()
  const data = await fetchOpenDictFile(prefix)
  const entry = data[word.toLowerCase()]

  if (!entry) return []

  const meanings: DictionaryMeaning[] = []

  for (const etymology of entry.etymologies || []) {
    for (const pos of etymology.partsOfSpeech || []) {
      meanings.push({
        partOfSpeech: pos.partOfSpeech || 'unknown',
        definitions: (pos.senses || []).slice(0, 2).map((sense: any) => ({
          definition: stripHtml(sense.sense || ''),
          example: sense.examples?.[0] ? stripHtml(sense.examples[0]) : undefined,
        })),
      })
    }
  }

  if (meanings.length === 0) return []

  return [{
    word: entry.word || word,
    meanings,
  }]
}

// ===== Wikidata (proper nouns, places, people) =====

async function fetchFromWikidata(word: string): Promise<DictionaryWord[]> {
  try {
    const response = await fetch(
      `https://www.wikidata.org/w/api.php?action=wbsearchentities&search=${encodeURIComponent(word)}&language=en&format=json&limit=3&origin=*`
    )

    if (!response.ok) return []

    const data = await response.json()
    const results: DictionaryWord[] = []

    for (const entity of data.search || []) {
      if (!entity.description) continue

      results.push({
        word: entity.label || word,
        meanings: [{
          partOfSpeech: 'proper noun',
          definitions: [{
            definition: entity.description,
          }],
        }],
      })
    }

    return results
  } catch {
    return []
  }
}

// ===== Alternate forms =====

function getAlternateForms(word: string): string[] {
  const forms: string[] = []

  if (word.endsWith('s')) {
    forms.push(word.slice(0, -1))
  } else {
    forms.push(word + 's')
  }

  if (word.endsWith('es')) {
    forms.push(word.slice(0, -2))
  } else {
    forms.push(word + 'es')
  }

  if (word.endsWith('ed')) {
    forms.push(word.slice(0, -2))
    forms.push(word.slice(0, -1))
  }

  if (word.endsWith('ing')) {
    forms.push(word.slice(0, -3))
    forms.push(word.slice(0, -3) + 'e')
  }

  return [...new Set(forms)].filter(f => f.length > 1)
}

// ===== Main lookup =====

/**
 * Look up a word. Tries Free Dictionary API first (phonetics + audio),
 * then Open Dictionary (260K+ words), then alternate forms of both.
 */
export async function lookupWord(word: string): Promise<DictionaryWord[]> {
  const key = word.toLowerCase().trim().replace(/[.,!?;:]+$/, '')
  if (!key) return []

  if (cache.has(key)) {
    return cache.get(key)!
  }

  // 1. Try Free Dictionary API (has phonetics + audio)
  const freeResults = await fetchFromFreeDictionary(key)
  if (freeResults.length > 0) {
    cache.set(key, freeResults)
    return freeResults
  }

  // 2. Try Open Dictionary (much larger coverage)
  const openResults = await fetchFromOpenDictionary(key)
  if (openResults.length > 0) {
    console.log(`[Dictionary] Found "${key}" in Open Dictionary`)
    cache.set(key, openResults)
    return openResults
  }

  // 3. Try Wikidata (proper nouns, places, people)
  const wikiResults = await fetchFromWikidata(key)
  if (wikiResults.length > 0) {
    console.log(`[Dictionary] Found "${key}" in Wikidata`)
    cache.set(key, wikiResults)
    return wikiResults
  }

  // 4. Try alternate forms in both sources
  const alternates = getAlternateForms(key)
  for (const alt of alternates) {
    const altFree = await fetchFromFreeDictionary(alt)
    if (altFree.length > 0) {
      console.log(`[Dictionary] Found "${alt}" as alternate for "${key}"`)
      cache.set(key, altFree)
      return altFree
    }

    const altOpen = await fetchFromOpenDictionary(alt)
    if (altOpen.length > 0) {
      console.log(`[Dictionary] Found "${alt}" in Open Dictionary as alternate for "${key}"`)
      cache.set(key, altOpen)
      return altOpen
    }
  }

  cache.set(key, [])
  return []
}

/**
 * Clear the dictionary cache
 */
export function clearDictionaryCache(): void {
  cache.clear()
  openDictCache.clear()
}
