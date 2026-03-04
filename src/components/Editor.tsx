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

  // Apply misspelled marks for a batch of corrections.
  // Uses ProseMirror doc traversal for correct positions across paragraphs,
  // then applies marks via TipTap's chain API for proper React rendering.
  const applyMarks = useCallback((editor: ReturnType<typeof useEditor>, corrections: Correction[]) => {
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
    lastCheckedTextRef.current = text
    console.log('[SpellCheck] Checking:', text)

    // Clear ALL old marks first
    editor
      .chain()
      .selectAll()
      .unsetMark('misspelled')
      .setTextSelection(editor.state.doc.content.size)
      .run()
    setCorrections([])

    try {
      // checkSpelling calls onResults as each chunk completes
      await checkSpelling(text, (chunkCorrections) => {
        console.log('[SpellCheck] Chunk results:', chunkCorrections)
        setCorrections(prev => [...prev, ...chunkCorrections])
        applyMarks(editor, chunkCorrections)
      })
    } catch (error) {
      console.error('[SpellCheck] Error:', error)
    } finally {
      isCheckingRef.current = false
      setIsChecking(false)
    }
  }, [editor, applyMarks])

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

      if (correction && editor) {
        const rect = target.getBoundingClientRect()
        // Use ProseMirror's DOM position for accurate range (avoids indexOf finding wrong occurrence)
        const pos = editor.view.posAtDOM(target, 0)

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

    setCorrections(prev =>
      prev.filter(c => c.original.toLowerCase() !== activeSuggestion.original.toLowerCase())
    )
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
