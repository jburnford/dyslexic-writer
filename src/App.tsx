import { useState, useEffect, useRef, useCallback } from 'react'
import Editor from './components/Editor'
import { EditorRef } from './components/Editor'
import VoiceWordHelper from './components/VoiceWordHelper'
import { exportLog, clearLog, getBackendUrl, setBackendUrl } from './services/spelling'

function App() {
  const [learningMode, setLearningMode] = useState(false)
  const [lightMode, setLightMode] = useState(false)
  const [standardFont, setStandardFont] = useState(false)
  const [backendUrl, setBackendUrlState] = useState(getBackendUrl)
  const [backendStatus, setBackendStatus] = useState<'unknown' | 'ok' | 'error'>('unknown')
  const editorRef = useRef<EditorRef>(null)

  // Apply theme classes to body
  useEffect(() => {
    document.body.classList.toggle('light-mode', lightMode)
    document.body.classList.toggle('standard-font', standardFont)
  }, [lightMode, standardFont])

  // Check backend health on URL change
  useEffect(() => {
    setBackendStatus('unknown')
    const controller = new AbortController()
    fetch(`${backendUrl}/health`, { signal: controller.signal })
      .then(r => r.ok ? setBackendStatus('ok') : setBackendStatus('error'))
      .catch(() => setBackendStatus('error'))
    return () => controller.abort()
  }, [backendUrl])

  const handleBackendUrlChange = useCallback((url: string) => {
    setBackendUrlState(url)
    setBackendUrl(url)
  }, [])

  const handleInsertWord = useCallback((word: string) => {
    editorRef.current?.insertWord(word)
  }, [])

  return (
    <div className="app">
      <header className="header">
        <h1>Dyslexic Writer</h1>
        <div className="settings">
          <label className="toggle">
            <input
              type="checkbox"
              checked={!learningMode}
              onChange={(e) => setLearningMode(!e.target.checked)}
            />
            <span>Click to replace</span>
          </label>
          <label className="toggle">
            <input
              type="checkbox"
              checked={lightMode}
              onChange={(e) => setLightMode(e.target.checked)}
            />
            <span>Light mode</span>
          </label>
          <label className="toggle">
            <input
              type="checkbox"
              checked={standardFont}
              onChange={(e) => setStandardFont(e.target.checked)}
            />
            <span>Standard font</span>
          </label>
          <div className="backend-setting">
            <input
              type="text"
              className="backend-url-input"
              value={backendUrl}
              onChange={(e) => handleBackendUrlChange(e.target.value)}
              placeholder="Backend URL"
            />
            <span
              className={`backend-status ${backendStatus}`}
              title={backendStatus === 'ok' ? 'Connected' : backendStatus === 'error' ? 'Not connected' : 'Checking...'}
            />
          </div>
        </div>
      </header>

      <main className="main main-with-sidebar">
        <VoiceWordHelper onInsertWord={handleInsertWord} />
        <div className="main-content">
          <Editor ref={editorRef} learningMode={learningMode} />
        </div>
      </main>

      <footer className="footer">
        <p>Type a sentence and end with a period. Misspelled words will be highlighted.</p>
        <div className="footer-actions">
          <button
            className="footer-button"
            onClick={() => {
              const log = exportLog()
              const blob = new Blob([log], { type: 'application/json' })
              const url = URL.createObjectURL(blob)
              const a = document.createElement('a')
              a.href = url
              a.download = `spelling-log-${new Date().toISOString().split('T')[0]}.json`
              a.click()
              URL.revokeObjectURL(url)
            }}
          >
            Export Log
          </button>
          <button
            className="footer-button"
            onClick={() => {
              if (confirm('Clear all logged data?')) {
                clearLog()
                alert('Log cleared')
              }
            }}
          >
            Clear Log
          </button>
        </div>
      </footer>
    </div>
  )
}

export default App
