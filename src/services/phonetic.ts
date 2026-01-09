/**
 * Phonetic matching for dyslexic spelling
 *
 * Uses Double Metaphone algorithm to match misspellings to correct words
 * based on how they SOUND, not how they're spelled.
 *
 * Example: "becuase" sounds like "because" even though spelling is very different
 */

import { doubleMetaphone } from 'double-metaphone'

// Common English words - we'll match misspellings against these
// In production, this would be a much larger dictionary
const COMMON_WORDS = [
  // Common verbs
  'want', 'have', 'make', 'take', 'give', 'think', 'know', 'feel',
  'become', 'leave', 'begin', 'seem', 'help', 'show', 'hear', 'play',
  'run', 'move', 'live', 'believe', 'bring', 'happen', 'write', 'read',
  'learn', 'change', 'watch', 'follow', 'stop', 'speak', 'turn', 'start',
  'might', 'found', 'going', 'getting', 'making', 'coming', 'looking',

  // School/writing related
  'because', 'really', 'friend', 'people', 'school', 'thought', 'through',
  'enough', 'should', 'would', 'could', 'something', 'anything', 'everything',
  'different', 'important', 'beautiful', 'favorite', 'probably', 'actually',
  'definitely', 'especially', 'interesting', 'tomorrow', 'together',

  // Common nouns
  'people', 'world', 'thing', 'child', 'children', 'woman', 'women',
  'place', 'water', 'money', 'story', 'point', 'company', 'problem',
  'game', 'gaming', 'setup', 'computer', 'phone', 'video', 'picture',

  // Tech/gaming words kids use
  'awesome', 'amazing', 'cool', 'minecraft', 'fortnite', 'youtube',
  'subscribe', 'channel', 'stream', 'download', 'update', 'level',
  'character', 'player', 'controller', 'keyboard', 'mouse', 'monitor',
  'headset', 'microphone', 'camera', 'screen', 'desktop', 'laptop',

  // Common adjectives
  'good', 'great', 'little', 'small', 'large', 'young', 'important',
  'different', 'right', 'wrong', 'happy', 'excited', 'dyslexic',

  // Common adverbs
  'very', 'really', 'always', 'never', 'sometimes', 'usually',

  // Contractions (without apostrophe - how kids often type them)
  "i'm", "don't", "can't", "won't", "it's", "that's", "there's",
  "what's", "let's", "didn't", "doesn't", "isn't", "aren't", "wasn't",
]

// Build phonetic index: maps phonetic code -> list of words
interface PhoneticIndex {
  [code: string]: string[]
}

const phoneticIndex: PhoneticIndex = {}

// Build the index on module load
for (const word of COMMON_WORDS) {
  const [primary, secondary] = doubleMetaphone(word)

  if (primary) {
    if (!phoneticIndex[primary]) phoneticIndex[primary] = []
    if (!phoneticIndex[primary].includes(word)) {
      phoneticIndex[primary].push(word)
    }
  }

  if (secondary && secondary !== primary) {
    if (!phoneticIndex[secondary]) phoneticIndex[secondary] = []
    if (!phoneticIndex[secondary].includes(word)) {
      phoneticIndex[secondary].push(word)
    }
  }
}

export interface PhoneticMatch {
  original: string
  candidates: string[]
  bestMatch: string | null
  confidence: number // 0-1, how confident we are in the match
}

/**
 * Find phonetic matches for a potentially misspelled word
 */
export function findPhoneticMatches(word: string): PhoneticMatch {
  const cleanWord = word.toLowerCase().replace(/[^a-z]/g, '')

  if (!cleanWord) {
    return { original: word, candidates: [], bestMatch: null, confidence: 0 }
  }

  // Get phonetic codes for the input word
  const [primary, secondary] = doubleMetaphone(cleanWord)

  // Collect all candidate matches
  const candidateSet = new Set<string>()

  if (primary && phoneticIndex[primary]) {
    for (const match of phoneticIndex[primary]) {
      candidateSet.add(match)
    }
  }

  if (secondary && phoneticIndex[secondary]) {
    for (const match of phoneticIndex[secondary]) {
      candidateSet.add(match)
    }
  }

  // Remove the original word if it's in the candidates (it's spelled correctly)
  candidateSet.delete(cleanWord)

  const candidates = Array.from(candidateSet)

  // If no candidates, no match
  if (candidates.length === 0) {
    return { original: word, candidates: [], bestMatch: null, confidence: 0 }
  }

  // Score candidates by similarity to original
  const scored = candidates.map(candidate => ({
    word: candidate,
    score: similarityScore(cleanWord, candidate)
  }))

  // Sort by score descending
  scored.sort((a, b) => b.score - a.score)

  const bestMatch = scored[0].word
  const confidence = scored[0].score

  // Only return high-confidence matches
  // If multiple candidates have similar scores, confidence is lower
  let adjustedConfidence = confidence
  if (scored.length > 1 && scored[1].score > confidence * 0.8) {
    adjustedConfidence *= 0.7 // Reduce confidence if close alternatives exist
  }

  return {
    original: word,
    candidates: scored.map(s => s.word),
    bestMatch: adjustedConfidence > 0.3 ? bestMatch : null,
    confidence: adjustedConfidence
  }
}

/**
 * Calculate similarity between two words (0-1)
 * Combines multiple factors
 */
function similarityScore(misspelled: string, correct: string): number {
  // Length similarity
  const lenDiff = Math.abs(misspelled.length - correct.length)
  const lenScore = 1 - (lenDiff / Math.max(misspelled.length, correct.length))

  // First letter match (important for dyslexic spelling)
  const firstLetterScore = misspelled[0] === correct[0] ? 1 : 0.5

  // Common subsequence
  const lcsScore = longestCommonSubsequence(misspelled, correct) /
                   Math.max(misspelled.length, correct.length)

  // Edit distance (normalized)
  const editDist = levenshteinDistance(misspelled, correct)
  const editScore = 1 - (editDist / Math.max(misspelled.length, correct.length))

  // Weighted combination
  return (lenScore * 0.1) + (firstLetterScore * 0.2) + (lcsScore * 0.3) + (editScore * 0.4)
}

/**
 * Longest common subsequence length
 */
function longestCommonSubsequence(a: string, b: string): number {
  const m = a.length
  const n = b.length
  const dp: number[][] = Array(m + 1).fill(null).map(() => Array(n + 1).fill(0))

  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      if (a[i - 1] === b[j - 1]) {
        dp[i][j] = dp[i - 1][j - 1] + 1
      } else {
        dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1])
      }
    }
  }

  return dp[m][n]
}

/**
 * Levenshtein edit distance
 */
function levenshteinDistance(a: string, b: string): number {
  const m = a.length
  const n = b.length
  const dp: number[][] = Array(m + 1).fill(null).map(() => Array(n + 1).fill(0))

  for (let i = 0; i <= m; i++) dp[i][0] = i
  for (let j = 0; j <= n; j++) dp[0][j] = j

  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      if (a[i - 1] === b[j - 1]) {
        dp[i][j] = dp[i - 1][j - 1]
      } else {
        dp[i][j] = 1 + Math.min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
      }
    }
  }

  return dp[m][n]
}

/**
 * Check if a word is likely misspelled (not in our dictionary)
 */
export function isLikelyMisspelled(word: string): boolean {
  const clean = word.toLowerCase().replace(/[^a-z']/g, '')
  return clean.length > 1 && !COMMON_WORDS.includes(clean)
}

/**
 * Add a word to the dictionary (for learning new words)
 */
export function addToDictionary(word: string): void {
  const clean = word.toLowerCase()
  if (!COMMON_WORDS.includes(clean)) {
    COMMON_WORDS.push(clean)

    // Update phonetic index
    const [primary, secondary] = doubleMetaphone(clean)
    if (primary) {
      if (!phoneticIndex[primary]) phoneticIndex[primary] = []
      phoneticIndex[primary].push(clean)
    }
    if (secondary && secondary !== primary) {
      if (!phoneticIndex[secondary]) phoneticIndex[secondary] = []
      phoneticIndex[secondary].push(clean)
    }
  }
}
