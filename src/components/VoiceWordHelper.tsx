import { useState, useEffect, useRef, useCallback } from 'react'
import { lookupWord, DictionaryWord } from '../services/dictionary'
import { findPhoneticMatches } from '../services/phonetic'
import {
  isSpeechRecognitionSupported,
  startListening,
  VoiceCommand,
} from '../services/voiceRecognition'

interface VoiceWordHelperProps {
  onInsertWord: (word: string) => void
}

interface SavedWord {
  word: string
  definition: string
}

const SAVED_WORDS_KEY = 'dyslexic-writer-hard-words'

function loadSavedWords(): SavedWord[] {
  try {
    const stored = localStorage.getItem(SAVED_WORDS_KEY)
    return stored ? JSON.parse(stored) : []
  } catch {
    return []
  }
}

function saveSavedWords(words: SavedWord[]): void {
  localStorage.setItem(SAVED_WORDS_KEY, JSON.stringify(words))
}

export default function VoiceWordHelper({ onInsertWord }: VoiceWordHelperProps) {
  // Voice state
  const [isListening, setIsListening] = useState(false)
  const [targetWord, setTargetWord] = useState('')
  const [contextSentence, setContextSentence] = useState('')
  const stopListeningRef = useRef<(() => void) | null>(null)

  // Results state
  const [matches, setMatches] = useState<DictionaryWord[]>([])
  const [phoneticSuggestions, setPhoneticSuggestions] = useState<string[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState('')

  // Saved words state
  const [savedWords, setSavedWords] = useState<SavedWord[]>(loadSavedWords)

  // Manual input ref
  const inputRef = useRef<HTMLInputElement>(null)

  // Browser support check
  const voiceSupported = isSpeechRecognitionSupported()

  // Persist saved words
  useEffect(() => {
    saveSavedWords(savedWords)
  }, [savedWords])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (stopListeningRef.current) {
        stopListeningRef.current()
      }
    }
  }, [])

  const fetchWordMatches = useCallback(async (word: string) => {
    setIsLoading(true)
    setError('')
    setMatches([])
    setPhoneticSuggestions([])

    try {
      // Fetch dictionary results and phonetic matches in parallel
      const [dictResults, phoneticResult] = await Promise.all([
        lookupWord(word),
        Promise.resolve(findPhoneticMatches(word)),
      ])

      setMatches(dictResults)

      // If dictionary found nothing, show phonetic suggestions
      if (dictResults.length === 0 && phoneticResult.candidates.length > 0) {
        setPhoneticSuggestions(phoneticResult.candidates.slice(0, 5))
      }

      if (dictResults.length === 0 && phoneticResult.candidates.length === 0) {
        setError(`No matches found for "${word}". Try a different spelling.`)
      }
    } catch (err) {
      setError('Failed to look up word. Check your internet connection.')
      console.error('[VoiceWordHelper] Lookup error:', err)
    } finally {
      setIsLoading(false)
    }
  }, [])

  const handleVoiceInput = useCallback(() => {
    if (isListening) {
      // Stop listening
      if (stopListeningRef.current) {
        stopListeningRef.current()
        stopListeningRef.current = null
      }
      setIsListening(false)
      return
    }

    // Start listening
    setError('')
    setIsListening(true)

    const stop = startListening({
      onResult: (command: VoiceCommand) => {
        setTargetWord(command.targetWord)
        setContextSentence(command.context || '')
        setIsListening(false)
        stopListeningRef.current = null
        fetchWordMatches(command.targetWord)
      },
      onError: (errMsg: string) => {
        setError(errMsg)
        setIsListening(false)
        stopListeningRef.current = null
      },
      onEnd: () => {
        setIsListening(false)
        stopListeningRef.current = null
      },
    })

    stopListeningRef.current = stop
  }, [isListening, fetchWordMatches])

  const handleManualSearch = useCallback(
    (e: React.FormEvent) => {
      e.preventDefault()
      const word = inputRef.current?.value.trim()
      if (word) {
        setTargetWord(word)
        setContextSentence('')
        fetchWordMatches(word)
      }
    },
    [fetchWordMatches]
  )

  const handleSpeak = useCallback((text: string) => {
    window.speechSynthesis.cancel()
    const utterance = new SpeechSynthesisUtterance(text)
    utterance.rate = 0.8
    window.speechSynthesis.speak(utterance)
  }, [])

  const handleInsert = useCallback(
    (word: string) => {
      onInsertWord(word)
    },
    [onInsertWord]
  )

  const handleSaveWord = useCallback(
    (word: string, definition: string) => {
      setSavedWords((prev) => {
        // Don't add duplicates
        if (prev.some((w) => w.word.toLowerCase() === word.toLowerCase())) {
          return prev
        }
        return [...prev, { word, definition }]
      })
    },
    []
  )

  const handleRemoveSavedWord = useCallback((word: string) => {
    setSavedWords((prev) =>
      prev.filter((w) => w.word.toLowerCase() !== word.toLowerCase())
    )
  }, [])

  return (
    <aside className="voice-sidebar">
      <h2 className="sidebar-title">Word Helper</h2>

      {/* Voice Input */}
      <div className="voice-input-section">
        <button
          className={`mic-button ${isListening ? 'mic-listening' : ''}`}
          onClick={handleVoiceInput}
          disabled={!voiceSupported && !isListening}
          title={
            voiceSupported
              ? 'Click to speak'
              : 'Voice not supported in this browser'
          }
        >
          <span className="mic-icon">{isListening ? '...' : '\uD83C\uDF99\uFE0F'}</span>
          <span className="mic-label">
            {isListening ? 'Listening...' : 'Say a word'}
          </span>
        </button>

        {!voiceSupported && (
          <p className="voice-unsupported">
            Voice not supported. Use Chrome or Edge, or type below.
          </p>
        )}

        {/* Manual text input fallback */}
        <form className="manual-input" onSubmit={handleManualSearch}>
          <input
            ref={inputRef}
            type="text"
            placeholder="Type a word..."
            className="word-input"
          />
          <button type="submit" className="search-button">
            Look up
          </button>
        </form>

        {isListening && (
          <p className="voice-hint">
            Say: "Help me spell [word]"
          </p>
        )}
      </div>

      {/* Current search info */}
      {targetWord && (
        <div className="search-info">
          Looking up: <strong>{targetWord}</strong>
          {contextSentence && (
            <span className="context-text"> in "{contextSentence}"</span>
          )}
        </div>
      )}

      {/* Loading */}
      {isLoading && <div className="sidebar-loading">Looking up word...</div>}

      {/* Error */}
      {error && <div className="sidebar-error">{error}</div>}

      {/* Dictionary matches */}
      {matches.length > 0 && (
        <div className="matches-section">
          <h3 className="section-label">Matches</h3>
          {matches.map((match, i) => (
            <div key={`${match.word}-${i}`} className="word-card">
              <div className="word-card-header">
                <span className="word-card-word">{match.word}</span>
                {match.phonetic && (
                  <span className="word-card-phonetic">{match.phonetic}</span>
                )}
                <div className="word-card-actions">
                  <button
                    className="icon-btn"
                    onClick={() => {
                      if (match.audioUrl) {
                        new Audio(match.audioUrl).play()
                      } else {
                        handleSpeak(match.word)
                      }
                    }}
                    title="Hear pronunciation"
                  >
                    \uD83D\uDD0A
                  </button>
                  <button
                    className="icon-btn"
                    onClick={() => handleInsert(match.word)}
                    title="Insert into editor"
                  >
                    \u2B05\uFE0F
                  </button>
                  <button
                    className="icon-btn"
                    onClick={() =>
                      handleSaveWord(
                        match.word,
                        match.meanings[0]?.definitions[0]?.definition || ''
                      )
                    }
                    title="Save to My Hard Words"
                  >
                    \u2B50
                  </button>
                </div>
              </div>
              {match.meanings.slice(0, 2).map((meaning, mi) => (
                <div key={mi} className="word-card-meaning">
                  <span className="part-of-speech">{meaning.partOfSpeech}</span>
                  {meaning.definitions.slice(0, 1).map((def, di) => (
                    <p key={di} className="definition-text">
                      {def.definition}
                    </p>
                  ))}
                </div>
              ))}
            </div>
          ))}
        </div>
      )}

      {/* Phonetic suggestions (when dictionary found nothing) */}
      {phoneticSuggestions.length > 0 && (
        <div className="matches-section">
          <h3 className="section-label">Did you mean...</h3>
          {phoneticSuggestions.map((suggestion) => (
            <div key={suggestion} className="phonetic-suggestion">
              <span className="suggestion-text">{suggestion}</span>
              <div className="word-card-actions">
                <button
                  className="icon-btn"
                  onClick={() => handleSpeak(suggestion)}
                  title="Hear pronunciation"
                >
                  \uD83D\uDD0A
                </button>
                <button
                  className="icon-btn"
                  onClick={() => {
                    setTargetWord(suggestion)
                    fetchWordMatches(suggestion)
                  }}
                  title="Look up this word"
                >
                  \uD83D\uDD0D
                </button>
                <button
                  className="icon-btn"
                  onClick={() => handleInsert(suggestion)}
                  title="Insert into editor"
                >
                  \u2B05\uFE0F
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Saved Words ("My Hard Words") */}
      <div className="saved-words-section">
        <h3 className="section-label">
          My Hard Words
          {savedWords.length > 0 && (
            <span className="word-count">({savedWords.length})</span>
          )}
        </h3>
        {savedWords.length === 0 ? (
          <p className="empty-saved">
            Words you save will appear here.
          </p>
        ) : (
          <div className="saved-words-list">
            {savedWords.map((saved) => (
              <div key={saved.word} className="saved-word-item">
                <div className="saved-word-main">
                  <button
                    className="saved-word-text"
                    onClick={() => handleInsert(saved.word)}
                    title="Click to insert"
                  >
                    {saved.word}
                  </button>
                  <button
                    className="icon-btn small"
                    onClick={() => handleSpeak(saved.word)}
                    title="Hear pronunciation"
                  >
                    \uD83D\uDD0A
                  </button>
                  <button
                    className="icon-btn small remove-btn"
                    onClick={() => handleRemoveSavedWord(saved.word)}
                    title="Remove"
                  >
                    \u2715
                  </button>
                </div>
                {saved.definition && (
                  <p className="saved-word-def">{saved.definition}</p>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </aside>
  )
}
