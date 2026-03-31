import { useState, useEffect, useRef, useCallback } from 'react'
import Editor from './components/Editor'
import { EditorRef } from './components/Editor'
import VoiceWordHelper from './components/VoiceWordHelper'
import Toolbar from './components/Toolbar'
import { exportLog } from './services/spelling'
import { saveAsMarkdown, saveAsDocx, saveAsPdf } from './services/export'

function App() {
  const [learningMode, setLearningMode] = useState(false)
  const [lightMode, setLightMode] = useState(false)
  const [standardFont, setStandardFont] = useState(false)
  const [zenMode, setZenMode] = useState(false)
  const [wordCount, setWordCount] = useState(0)
  const [zenHint, setZenHint] = useState(false)
  const [checkWordText, setCheckWordText] = useState('')
  const editorRef = useRef<EditorRef>(null)

  // Apply theme classes to body
  useEffect(() => {
    document.body.classList.toggle('light-mode', lightMode)
    document.body.classList.toggle('standard-font', standardFont)
  }, [lightMode, standardFont])

  // Escape key exits zen mode
  useEffect(() => {
    if (!zenMode) return
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setZenMode(false)
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [zenMode])

  // Show hint when entering zen mode
  useEffect(() => {
    if (zenMode) {
      setZenHint(true)
      const timer = setTimeout(() => setZenHint(false), 3000)
      return () => clearTimeout(timer)
    }
    setZenHint(false)
  }, [zenMode])

  const handleInsertWord = useCallback((word: string) => {
    editorRef.current?.insertWord(word)
  }, [])

  const handleCheckWord = useCallback((word: string) => {
    // Use a unique key each time so the effect fires even for the same word
    setCheckWordText(word + '\0' + Date.now())
  }, [])

  const handleSave = useCallback((format: 'md' | 'docx' | 'pdf') => {
    const html = editorRef.current?.getHTML() || ''
    if (!html || html === '<p></p>') return
    switch (format) {
      case 'md':
        saveAsMarkdown(html)
        break
      case 'docx':
        saveAsDocx(html)
        break
      case 'pdf':
        saveAsPdf(html)
        break
    }
  }, [])

  const handleExportLog = useCallback(() => {
    const log = exportLog()
    const blob = new Blob([log], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `spelling-log-${new Date().toISOString().split('T')[0]}.json`
    a.click()
    URL.revokeObjectURL(url)
  }, [])

  return (
    <div className={`app${zenMode ? ' zen' : ''}`}>
      <header className="header-minimal">
        <span className="app-title">Dyslexic Writer</span>
      </header>

      <div className="workspace">
        <Toolbar
          onCheckSpelling={() => editorRef.current?.runSpellCheck()}
          onReadMyWriting={() => editorRef.current?.readMyWriting()}
          onClearHighlights={() => editorRef.current?.clearHighlights()}
          learningMode={learningMode}
          onToggleLearningMode={() => setLearningMode(m => !m)}
          lightMode={lightMode}
          onToggleLightMode={() => setLightMode(m => !m)}
          standardFont={standardFont}
          onToggleFont={() => setStandardFont(f => !f)}
          zenMode={zenMode}
          onToggleZenMode={() => setZenMode(z => !z)}
          onSave={handleSave}
          onExportLog={handleExportLog}
        />

        <main className="editor-panel">
          <Editor
            ref={editorRef}
            learningMode={learningMode}
            onWordCountChange={setWordCount}
            onCheckWord={handleCheckWord}
          />
          <div className="word-count">
            {wordCount} {wordCount === 1 ? 'word' : 'words'}
          </div>
        </main>

        <VoiceWordHelper onInsertWord={handleInsertWord} selectedText={checkWordText} />
      </div>

      {zenMode && (
        <button
          className="zen-exit-button"
          onClick={() => setZenMode(false)}
          aria-label="Exit Zen Mode"
        >
          Exit Zen
        </button>
      )}
      {zenMode && zenHint && (
        <div className="zen-exit-hint">
          Press Escape to exit Zen Mode
        </div>
      )}
    </div>
  )
}

export default App
