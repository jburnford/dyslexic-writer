import { useEditor, EditorContent } from '@tiptap/react'
import StarterKit from '@tiptap/starter-kit'
import { Mark, mergeAttributes } from '@tiptap/core'
import { useState, useRef } from 'react'
import { checkSpelling, Correction } from '../services/spelling'

// Preserve the case pattern of the original word in the correction
function preserveCase(original: string, correction: string): string {
  if (original === original.toUpperCase()) {
    return correction.toUpperCase()
  }
  if (original === original.toLowerCase()) {
    return correction.toLowerCase()
  }
  // Title case or mixed - just use correction as-is
  return correction
}

// Custom Mark for misspelled words
const Misspelled = Mark.create({
  name: 'misspelled',
  addAttributes() {
    return {
      correction: { default: null },
    }
  },
  parseHTML() {
    return [{ tag: 'span.misspelled' }]
  },
  renderHTML({ HTMLAttributes }) {
    return ['span', mergeAttributes({ class: 'misspelled' }, HTMLAttributes), 0]
  },
})

interface EditorProps {
  learningMode: boolean
}

interface Suggestion {
  original: string
  corrected: string
  position: { x: number; y: number }
  range: { from: number; to: number }
}

export default function Editor({ learningMode }: EditorProps) {
  const [activeSuggestion, setActiveSuggestion] = useState<Suggestion | null>(null)
  const [isChecking, setIsChecking] = useState(false)
  const [corrections, setCorrections] = useState<Correction[]>([])
  const [lastCheckedText, setLastCheckedText] = useState('')
  const editorRef = useRef<HTMLDivElement>(null)

  const editor = useEditor({
    extensions: [StarterKit, Misspelled],
    content: '<p></p>',
    editorProps: {
      attributes: {
        class: 'editor-content',
        spellcheck: 'false',
        autocorrect: 'off',
        autocapitalize: 'off',
      },
    },
  })

  // Check spelling - called manually or on button click
  const runSpellCheck = async () => {
    if (!editor || isChecking) return

    const text = editor.getText()
    if (!text.trim() || text === lastCheckedText) return

    setIsChecking(true)
    setLastCheckedText(text)
    console.log('[SpellCheck] Checking:', text)

    try {
      const results = await checkSpelling(text)
      console.log('[SpellCheck] Results:', results)
      setCorrections(results)

      if (results.length === 0) {
        console.log('[SpellCheck] No corrections needed')
        setIsChecking(false)
        return
      }

      // Clear ALL old marks first
      editor
        .chain()
        .selectAll()
        .unsetMark('misspelled')
        .setTextSelection(editor.state.doc.content.size)
        .run()

      // Apply marks to misspelled words
      for (const correction of results) {
        const searchText = correction.original.toLowerCase()
        const docText = text.toLowerCase()
        let searchPos = 0

        while (searchPos < docText.length) {
          const index = docText.indexOf(searchText, searchPos)
          if (index === -1) break

          // Check word boundaries
          const before = index > 0 ? docText[index - 1] : ' '
          const after = index + searchText.length < docText.length
            ? docText[index + searchText.length]
            : ' '

          if (!/\w/.test(before) && !/\w/.test(after)) {
            const markFrom = index + 1 // +1 for ProseMirror offset
            const markTo = markFrom + correction.original.length

            // Preserve original case when storing correction
            const originalWord = text.slice(index, index + correction.original.length)
            const correctedWord = preserveCase(originalWord, correction.corrected)

            console.log(`[SpellCheck] Marking "${originalWord}" -> "${correctedWord}" at ${markFrom}-${markTo}`)

            editor
              .chain()
              .setTextSelection({ from: markFrom, to: markTo })
              .setMark('misspelled', { correction: correctedWord })
              .run()
          }

          searchPos = index + searchText.length
        }
      }

      // Move cursor to end
      editor.commands.setTextSelection(editor.state.doc.content.size)
    } catch (error) {
      console.error('[SpellCheck] Error:', error)
    } finally {
      setIsChecking(false)
    }
  }

  // Handle key events to trigger spell check on period
  const handleKeyUp = (e: React.KeyboardEvent) => {
    if (e.key === '.' || e.key === '!' || e.key === '?') {
      console.log('[SpellCheck] Period detected, running check...')
      runSpellCheck()
    }
  }

  // Handle clicking on misspelled words
  const handleClick = (e: React.MouseEvent) => {
    const target = e.target as HTMLElement

    if (target.classList.contains('misspelled')) {
      const word = target.textContent || ''
      const correction = corrections.find(
        c => c.original.toLowerCase() === word.toLowerCase()
      )?.corrected

      if (correction) {
        const rect = target.getBoundingClientRect()
        const text = editor?.getText() || ''
        const index = text.toLowerCase().indexOf(word.toLowerCase())

        setActiveSuggestion({
          original: word,
          corrected: correction,
          position: { x: rect.left, y: rect.bottom + 8 },
          range: { from: index + 1, to: index + 1 + word.length },
        })
      }
    } else {
      setActiveSuggestion(null)
    }
  }

  const handleAcceptCorrection = () => {
    if (!activeSuggestion || !editor) return

    if (learningMode) {
      setActiveSuggestion(null)
    } else {
      // Get current position info
      const { from, to } = activeSuggestion.range

      // Remove the mark, delete the word, insert correction
      editor
        .chain()
        .focus()
        .setTextSelection({ from, to })
        .unsetMark('misspelled')
        .deleteSelection()
        .insertContent(activeSuggestion.corrected)
        .run()

      // Remove this correction from the list
      setCorrections(prev =>
        prev.filter(c => c.original.toLowerCase() !== activeSuggestion.original.toLowerCase())
      )
      setActiveSuggestion(null)
      setLastCheckedText('') // Allow re-check
    }
  }

  const handleSpeak = (text: string) => {
    window.speechSynthesis.cancel()
    const utterance = new SpeechSynthesisUtterance(text)
    utterance.rate = 0.8
    window.speechSynthesis.speak(utterance)
  }

  return (
    <div className="editor-wrapper" ref={editorRef}>
      <div
        className="editor-container"
        onClick={handleClick}
        onKeyUp={handleKeyUp}
      >
        <EditorContent editor={editor} />
      </div>

      {isChecking && (
        <div className="checking-indicator">
          Checking spelling...
        </div>
      )}

      {activeSuggestion && (
        <div
          className="suggestion-popup"
          style={{
            left: activeSuggestion.position.x,
            top: activeSuggestion.position.y,
          }}
        >
          <div className="suggestion-header">Did you mean...</div>
          <div className="suggestion-option" onClick={handleAcceptCorrection}>
            <button
              className="speak-button"
              onClick={(e) => {
                e.stopPropagation()
                handleSpeak(activeSuggestion.corrected)
              }}
            >
              🔊
            </button>
            <span className="suggestion-word">{activeSuggestion.corrected}</span>
          </div>
          {learningMode && (
            <div className="learning-hint">
              Delete "{activeSuggestion.original}" and type "{activeSuggestion.corrected}"
            </div>
          )}
          <button className="close-button" onClick={() => setActiveSuggestion(null)}>
            Keep as is
          </button>
        </div>
      )}

      <div className="toolbar">
        <button
          className="toolbar-button"
          onClick={() => handleSpeak(editor?.getText() || '')}
        >
          🔊 Read My Writing
        </button>
        <button
          className="toolbar-button"
          onClick={runSpellCheck}
        >
          ✓ Check Spelling
        </button>
        <button
          className="toolbar-button secondary"
          onClick={() => {
            if (editor) {
              editor.chain().selectAll().unsetMark('misspelled').setTextSelection(editor.state.doc.content.size).run()
              setCorrections([])
              setLastCheckedText('')
            }
          }}
        >
          Clear Highlights
        </button>
      </div>
    </div>
  )
}
