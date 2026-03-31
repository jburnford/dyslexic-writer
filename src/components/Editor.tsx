import { useEditor, EditorContent } from '@tiptap/react'
import StarterKit from '@tiptap/starter-kit'
import Underline from '@tiptap/extension-underline'
import { Mark, mergeAttributes } from '@tiptap/core'
import { useState, useRef, useEffect, forwardRef, useImperativeHandle, useCallback } from 'react'
import { checkSpelling } from '../services/spelling'
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
  onCheckWord?: (word: string) => void
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

const AUTOSAVE_KEY = 'dyslexic-writer-draft'

const Editor = forwardRef<EditorRef, EditorProps>(function Editor({ learningMode, onWordCountChange, onCheckWord }, ref) {
  const [activeSuggestion, setActiveSuggestion] = useState<Suggestion | null>(null)
  const [isChecking, setIsChecking] = useState(false)
  const [checkProgress, setCheckProgress] = useState('')
  const editorRef = useRef<HTMLDivElement>(null)
  const isCheckingRef = useRef(false)
  const lastCheckedTextRef = useRef('')
  const saveTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const editor = useEditor({
    extensions: [
      StarterKit.configure({
        codeBlock: false,
        code: false,
      }),
      Underline,
      Misspelled,
    ],
    content: '<p></p>',
    editorProps: {
      attributes: {
        class: 'editor-content',
        spellcheck: 'false',
        autocorrect: 'off',
        autocapitalize: 'off',
      },
    },
    onCreate: ({ editor }) => {
      const saved = localStorage.getItem(AUTOSAVE_KEY)
      if (saved && saved !== '<p></p>') {
        editor.commands.setContent(saved)
        const text = editor.getText().trim()
        if (onWordCountChange) {
          onWordCountChange(text ? text.split(/\s+/).length : 0)
        }
      }
    },
    onUpdate: ({ editor }) => {
      if (onWordCountChange) {
        const text = editor.getText().trim()
        const count = text ? text.split(/\s+/).length : 0
        onWordCountChange(count)
      }
      // Debounced auto-save
      if (saveTimerRef.current) clearTimeout(saveTimerRef.current)
      saveTimerRef.current = setTimeout(() => {
        localStorage.setItem(AUTOSAVE_KEY, editor.getHTML())
      }, 1000)
    },
  })

  // Back-button / accidental navigation protection
  useEffect(() => {
    const handleBeforeUnload = (e: BeforeUnloadEvent) => {
      if (!editor) return
      const text = editor.getText().trim()
      if (text.length > 0) {
        // Save immediately before leaving
        localStorage.setItem(AUTOSAVE_KEY, editor.getHTML())
        e.preventDefault()
      }
    }
    window.addEventListener('beforeunload', handleBeforeUnload)
    return () => window.removeEventListener('beforeunload', handleBeforeUnload)
  }, [editor])

  const handleSpeak = useCallback((text: string) => {
    window.speechSynthesis.cancel()
    const utterance = new SpeechSynthesisUtterance(text)
    utterance.rate = 0.8
    window.speechSynthesis.speak(utterance)
  }, [])

  // Apply misspelled marks for a batch of corrections.
  // Uses ProseMirror doc traversal for correct positions across paragraphs,
  // then applies marks via TipTap's chain API for proper React rendering.
  const applyMarks = useCallback((editor: ReturnType<typeof useEditor>, corrections: { original: string; corrected: string }[]) => {
    if (!editor) return

    // First, collect all positions that need marks
    const marks: { from: number; to: number; correction: string }[] = []

    for (const correction of corrections) {
      const searchText = correction.original.toLowerCase()

      // Walk all text nodes to find matches with correct ProseMirror positions
      editor.state.doc.descendants((node, pos) => {
        if (!node.isText || !node.text) return
        const nodeText = node.text
        const nodeTextLower = nodeText.toLowerCase()
        let idx = 0

        while ((idx = nodeTextLower.indexOf(searchText, idx)) !== -1) {
          // Check word boundaries
          const before = idx > 0 ? nodeTextLower[idx - 1] : ' '
          const after = idx + searchText.length < nodeTextLower.length
            ? nodeTextLower[idx + searchText.length]
            : ' '

          if (!/\w/.test(before) && !/\w/.test(after)) {
            const from = pos + idx
            const to = from + correction.original.length
            const originalWord = nodeText.slice(idx, idx + correction.original.length)
            const correctedWord = preserveCase(originalWord, correction.corrected)

            marks.push({ from, to, correction: correctedWord })
          }

          idx += searchText.length
        }
      })
    }

    // Apply ALL marks in a single transaction through TipTap's pipeline
    if (marks.length > 0) {
      editor.commands.command(({ tr, dispatch }) => {
        const markType = editor.schema.marks.misspelled
        for (const mark of marks) {
          tr.addMark(mark.from, mark.to, markType.create({ correction: mark.correction }))
        }
        if (dispatch) dispatch(tr)
        return true
      })
    }
  }, [])

  // Check spelling - called manually or on button click
  const runSpellCheck = useCallback(async () => {
    if (!editor || isCheckingRef.current) return

    const text = editor.getText()
    if (!text.trim() || text === lastCheckedTextRef.current) return

    isCheckingRef.current = true
    setIsChecking(true)
    setCheckProgress('')
    lastCheckedTextRef.current = text
    console.log('[SpellCheck] Checking:', text)

    // Clear ALL old marks first
    editor
      .chain()
      .selectAll()
      .unsetMark('misspelled')
      .setTextSelection(editor.state.doc.content.size)
      .run()

    const timeoutId = setTimeout(() => {
      if (isCheckingRef.current) {
        isCheckingRef.current = false
        setIsChecking(false)
        setCheckProgress('')
        console.warn('[SpellCheck] Timed out after 20 seconds')
      }
    }, 20000)

    try {
      await checkSpelling(
        text,
        (chunkCorrections) => {
          console.log('[SpellCheck] Chunk results:', chunkCorrections)
          applyMarks(editor, chunkCorrections)
        },
        (current, total) => {
          if (total > 1) setCheckProgress(`Checking ${current}/${total}...`)
        },
      )
    } catch (error) {
      console.error('[SpellCheck] Error:', error)
    } finally {
      clearTimeout(timeoutId)
      isCheckingRef.current = false
      setIsChecking(false)
      setCheckProgress('')
    }
  }, [editor, applyMarks])

  // Expose methods to parent via ref
  useImperativeHandle(ref, () => ({
    insertWord(word: string) {
      if (!editor) return
      // Clear any misspelled mark at cursor before inserting
      const { from } = editor.state.selection
      editor.commands.command(({ tr, dispatch }) => {
        const $pos = tr.doc.resolve(from)
        if ($pos.marks().some(m => m.type.name === 'misspelled')) {
          const nodeStart = from - $pos.textOffset
          const node = $pos.parent.childAfter($pos.parentOffset - $pos.textOffset >= 0 ? 0 : 0)
          if (node.node) {
            tr.removeMark(nodeStart, nodeStart + $pos.parent.content.size, editor.schema.marks.misspelled)
          }
        }
        if (dispatch) dispatch(tr)
        return true
      })
      editor.chain().focus().insertContent(word + ' ').run()
    },
    runSpellCheck() {
      runSpellCheck()
    },
    clearHighlights() {
      if (editor) {
        editor.chain().selectAll().unsetMark('misspelled').setTextSelection(editor.state.doc.content.size).run()
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
    // Walk up DOM to find the misspelled span (handles clicks on nested bold/italic)
    const misspelledEl = target.closest('.misspelled') as HTMLElement | null

    if (misspelledEl && editor) {
      const word = misspelledEl.textContent || ''
      // Read correction directly from the mark's DOM attribute (always in sync)
      const correction = misspelledEl.getAttribute('data-correction')

      if (correction) {
        const rect = misspelledEl.getBoundingClientRect()
        const pos = editor.view.posAtDOM(misspelledEl, 0)

        setActiveSuggestion({
          original: word,
          corrected: correction,
          position: { x: rect.left, y: rect.bottom + 8 },
          range: { from: pos, to: pos + word.length },
        })
      }
    } else {
      setActiveSuggestion(null)
    }
  }

  // Handle "Check this word" — send selected or clicked word to Word Helper
  const handleCheckWord = useCallback((word: string) => {
    if (onCheckWord && word.trim()) {
      onCheckWord(word.trim())
    }
  }, [onCheckWord])

  const handleAcceptCorrection = () => {
    if (!activeSuggestion || !editor) return

    if (learningMode) {
      setActiveSuggestion(null)
      return
    }

    const searchText = activeSuggestion.original.toLowerCase()

    // Find ALL instances of this misspelled word in the document
    const instances: { from: number; to: number; corrected: string }[] = []

    editor.state.doc.descendants((node, pos) => {
      if (!node.isText || !node.text) return
      const nodeText = node.text
      const nodeTextLower = nodeText.toLowerCase()
      let idx = 0

      while ((idx = nodeTextLower.indexOf(searchText, idx)) !== -1) {
        const before = idx > 0 ? nodeTextLower[idx - 1] : ' '
        const after = idx + searchText.length < nodeTextLower.length
          ? nodeTextLower[idx + searchText.length]
          : ' '

        if (!/\w/.test(before) && !/\w/.test(after)) {
          const from = pos + idx
          const to = from + activeSuggestion.original.length
          const originalWord = nodeText.slice(idx, idx + activeSuggestion.original.length)
          instances.push({
            from,
            to,
            corrected: preserveCase(originalWord, activeSuggestion.corrected),
          })
        }

        idx += searchText.length
      }
    })

    // Replace ALL instances in a single transaction (reverse order to preserve positions)
    if (instances.length > 0) {
      editor.commands.command(({ tr, dispatch }) => {
        const markType = editor.schema.marks.misspelled
        for (const inst of [...instances].reverse()) {
          tr.removeMark(inst.from, inst.to, markType)
          tr.insertText(inst.corrected, inst.from, inst.to)
        }
        if (dispatch) dispatch(tr)
        return true
      })
    }

    setActiveSuggestion(null)
    lastCheckedTextRef.current = ''
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
          {checkProgress || 'Checking spelling...'}
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
          <button
            className="check-word-button"
            onClick={() => {
              handleCheckWord(activeSuggestion.original)
              setActiveSuggestion(null)
            }}
          >
            Look up alternatives
          </button>
          <button className="close-button" onClick={() => setActiveSuggestion(null)}>
            Keep as is
          </button>
        </div>
      )}
    </div>
  )
})

export default Editor
