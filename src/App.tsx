import { useState, useEffect } from 'react'
import Editor from './components/Editor'
import { exportLog, clearLog } from './services/spelling'

function App() {
  const [learningMode, setLearningMode] = useState(false)
  const [lightMode, setLightMode] = useState(false)
  const [standardFont, setStandardFont] = useState(false)

  // Apply theme classes to body
  useEffect(() => {
    document.body.classList.toggle('light-mode', lightMode)
    document.body.classList.toggle('standard-font', standardFont)
  }, [lightMode, standardFont])

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
        </div>
      </header>

      <main className="main">
        <Editor learningMode={learningMode} />
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
