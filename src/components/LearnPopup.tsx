import { useState, useRef, useEffect } from 'react'

interface LearnPopupProps {
  original: string
  correct: string
  position: { x: number; y: number }
  onSuccess: () => void
  onSkip: () => void
  onSpeak: (text: string) => void
}

export default function LearnPopup({ original, correct, position, onSuccess, onSkip, onSpeak }: LearnPopupProps) {
  const [typed, setTyped] = useState('')
  const [showSuccess, setShowSuccess] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    // Focus the input when popup opens
    setTimeout(() => inputRef.current?.focus(), 50)
  }, [])

  const correctLower = correct.toLowerCase()
  const typedLower = typed.toLowerCase()

  // Check if every typed character matches so far
  const isCorrectSoFar = correctLower.startsWith(typedLower)
  const isComplete = typedLower === correctLower && typed.length > 0

  useEffect(() => {
    if (isComplete && !showSuccess) {
      setShowSuccess(true)
      // Brief celebration, then apply the fix
      setTimeout(() => onSuccess(), 600)
    }
  }, [isComplete, showSuccess, onSuccess])

  // Render the target word with letter-by-letter coloring
  const renderTarget = () => {
    return correct.split('').map((letter, i) => {
      let className = 'learn-letter'
      if (i < typed.length) {
        className += typedLower[i] === correctLower[i] ? ' learn-letter-correct' : ' learn-letter-wrong'
      } else if (i === typed.length) {
        className += ' learn-letter-next'
      }
      return (
        <span key={i} className={className}>
          {letter}
        </span>
      )
    })
  }

  // Clamp position to keep popup on screen
  const style: React.CSSProperties = {
    left: Math.min(position.x, window.innerWidth - 320),
    top: Math.min(position.y, window.innerHeight - 300),
  }

  return (
    <div className="learn-popup" style={style}>
      <div className="learn-popup-header">
        Practice spelling this word
      </div>

      <div className="learn-original">
        You wrote: <span className="learn-original-word">{original}</span>
      </div>

      <div className="learn-target-row">
        <button
          className="speak-button"
          onClick={() => onSpeak(correct)}
          aria-label="Hear the word"
        >
          🔊
        </button>
        <div className="learn-target-word">
          {renderTarget()}
        </div>
      </div>

      <input
        ref={inputRef}
        className={`learn-input ${showSuccess ? 'learn-input-success' : !isCorrectSoFar ? 'learn-input-error' : ''}`}
        type="text"
        value={typed}
        placeholder="Type the correct spelling..."
        autoComplete="off"
        autoCorrect="off"
        autoCapitalize="off"
        spellCheck={false}
        onChange={(e) => {
          if (!showSuccess) setTyped(e.target.value)
        }}
      />

      {showSuccess && (
        <div className="learn-success">
          Great job!
        </div>
      )}

      {!isCorrectSoFar && typed.length > 0 && (
        <div className="learn-try-again">
          Not quite — look at the letters above and try again
        </div>
      )}

      <button className="close-button" onClick={onSkip}>
        Skip for now
      </button>
    </div>
  )
}
