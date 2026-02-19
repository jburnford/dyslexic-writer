import { useEditor, EditorContent } from '@tiptap/react'
import StarterKit from '@tiptap/starter-kit'
import Underline from '@tiptap/extension-underline'
import { Mark, mergeAttributes } from '@tiptap/core'
import { useState, useRef, forwardRef, useImperativeHandle, useCallback } from 'react'
import { checkSpelling, Correction } from '../services/spelling'
import FormatBar from './FormatBar'

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
  onWordCountChange?: (count: number) => void
}

export interface EditorRef {
  insertWord: (word: string) => void
  runSpellCheck: () => void
  clearHighlights: () => void
  readMyWriting: () => void
  getHTML: () => string
  getText: () => string
}

interface Suggestion {
  original: string
  corrected: string
  position: { x: number; y: number }
  range: { from: number; to: number }
}

const Editor = forwardRef<EditorRef, EditorProps>(function Editor({ learningMode, onWordCountChange }, ref) {
  const [activeSuggestion, setActiveSuggestion] = useState<Suggestion | null>(null)
  const [isChecking, setIsChecking] = useState(false)
  const [corrections, setCorrections] = useState<Correction[]>([])
  const editorRef = useRef<HTMLDivElement>(null)
  const isCheckingRef = useRef(false)
  const lastCheckedTextRef = useRef('')

  const editor = useEditor({
    extensions: [StarterKit, Underline, Misspelled],
    content: '<p></p>',
    editorProps: {
      attributes: {
        class: 'editor-content',
        spellcheck: 'false',
        autocorrect: 'off',
        autocapitalize: 'off',
      },
    },
    onUpdate: ({ editor }) => {
      if (onWordCountChange) {
        const text = editor.getText().trim()
        const count = text ? text.split(/\s+/).length : 0
        onWordCountChange(count)
      }
    },
  })

  const handleSpeak = useCallback((text: string) => {
    window.speechSynthesis.cancel()
    const utterance = new SpeechSynthesisUtterance(text)
    utterance.rate = 0.8
    window.speechSynthesis.speak(utterance)
  }, [])

  // Check spelling - called manually or on button click
  const runSpellCheck = useCallback(async () => {
    if (!editor || isCheckingRef.current) return

    const text = editor.getText()
    if (!text.trim() || text === lastCheckedTextRef.current) return

    isCheckingRef.current = true
    setIsChecking(true)
    lastCheckedTextRef.current = text
    console.log('[SpellCheck] Checking:', text)

    try {
      const results = await checkSpelling(text)
      console.log('[SpellCheck] Results:', results)
      setCorrections(results)

      if (results.length === 0) {
        console.log('[SpellCheck] No corrections needed')
        isCheckingRef.current = false
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
      isCheckingRef.current = false
      setIsChecking(false)
    }
  }, [editor])

  // Expose methods to parent via ref
  useImperativeHandle(ref, () => ({
    insertWord(word: string) {
      if (editor) {
        editor.chain().focus().insertContent(word + ' ').run()
      }
    },
    runSpellCheck() {
      runSpellCheck()
    },
    clearHighlights() {
      if (editor) {
        editor.chain().selectAll().unsetMark('misspelled').setTextSelection(editor.state.doc.content.size).run()
        setCorrections([])
        lastCheckedTextRef.current = ''
      }
    },
    readMyWriting() {
      handleSpeak(editor?.getText() || '')
    },
    getHTML() {
      return editor?.getHTML() || ''
    },
    getText() {
      return editor?.getText() || ''
    },
  }), [editor, runSpellCheck, handleSpeak])

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
      lastCheckedTextRef.current = '' // Allow re-check
    }
  }

  return (
    <div className="editor-wrapper" ref={editorRef}>
      <FormatBar editor={editor} />
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
    </div>
  )
})

export default Editor
