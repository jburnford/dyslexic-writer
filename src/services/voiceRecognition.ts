/**
 * Web Speech API wrapper for voice-to-text word helper
 *
 * Listens for commands like: "Help me spell [word]; [sentence]"
 * Extracts the target word for dictionary lookup.
 */

// Web Speech API types (not fully covered by default TS libs)
interface SpeechRecognitionEvent extends Event {
  results: SpeechRecognitionResultList
  resultIndex: number
}

interface SpeechRecognitionErrorEvent extends Event {
  error: string
  message: string
}

export interface VoiceCommand {
  targetWord: string
  context?: string // optional sentence context
  rawTranscript: string
}

/**
 * Check if the browser supports the Web Speech API
 */
export function isSpeechRecognitionSupported(): boolean {
  return !!(
    (window as any).SpeechRecognition ||
    (window as any).webkitSpeechRecognition
  )
}

/**
 * Parse a voice transcript to extract the target word
 * Patterns:
 *   "help me spell mountain"
 *   "help me spell mountain I go skiing at the mountain"
 *   "spell mountain"
 *   just "mountain" (single word fallback)
 */
export function parseVoiceCommand(transcript: string): VoiceCommand | null {
  const raw = transcript.trim()
  if (!raw) return null

  // Pattern: "help me spell [word] [context...]"
  const helpPattern = /help\s+me\s+spell\s+(\w+)(?:\s+(.+))?/i
  const helpMatch = raw.match(helpPattern)
  if (helpMatch) {
    return {
      targetWord: helpMatch[1],
      context: helpMatch[2] || undefined,
      rawTranscript: raw,
    }
  }

  // Pattern: "spell [word] [context...]"
  const spellPattern = /spell\s+(\w+)(?:\s+(.+))?/i
  const spellMatch = raw.match(spellPattern)
  if (spellMatch) {
    return {
      targetWord: spellMatch[1],
      context: spellMatch[2] || undefined,
      rawTranscript: raw,
    }
  }

  // Fallback: treat first word as target
  const words = raw.split(/\s+/)
  if (words.length >= 1 && words[0].length > 1) {
    return {
      targetWord: words[0],
      context: words.length > 1 ? words.slice(1).join(' ') : undefined,
      rawTranscript: raw,
    }
  }

  return null
}

export interface VoiceRecognitionCallbacks {
  onResult: (command: VoiceCommand) => void
  onError: (error: string) => void
  onEnd: () => void
}

/**
 * Start voice recognition. Returns a stop function.
 */
export function startListening(
  callbacks: VoiceRecognitionCallbacks
): (() => void) | null {
  if (!isSpeechRecognitionSupported()) {
    callbacks.onError(
      'Voice recognition is not supported in this browser. Try Chrome or Edge.'
    )
    return null
  }

  const SpeechRecognition =
    (window as any).SpeechRecognition ||
    (window as any).webkitSpeechRecognition

  const recognition = new SpeechRecognition()
  recognition.continuous = false
  recognition.interimResults = false
  recognition.lang = 'en-US'

  recognition.onresult = (event: SpeechRecognitionEvent) => {
    const transcript = event.results[0][0].transcript
    console.log('[Voice] Transcript:', transcript)

    const command = parseVoiceCommand(transcript)
    if (command) {
      callbacks.onResult(command)
    } else {
      callbacks.onError('Could not understand. Try: "Help me spell [word]"')
    }
  }

  recognition.onerror = (event: SpeechRecognitionErrorEvent) => {
    console.error('[Voice] Error:', event.error)
    if (event.error === 'not-allowed') {
      callbacks.onError(
        'Microphone access denied. Please allow microphone access in your browser settings.'
      )
    } else if (event.error === 'no-speech') {
      callbacks.onError('No speech detected. Try again.')
    } else {
      callbacks.onError(`Voice error: ${event.error}`)
    }
  }

  recognition.onend = () => {
    callbacks.onEnd()
  }

  recognition.start()

  return () => {
    recognition.abort()
  }
}
