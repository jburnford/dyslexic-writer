import { useState, useRef, useEffect } from 'react'
import { createPortal } from 'react-dom'

interface ToolbarProps {
  // Editor actions
  onCheckSpelling: () => void
  onReadMyWriting: () => void
  onClearHighlights: () => void

  // Toggles
  learningMode: boolean
  onToggleLearningMode: () => void
  lightMode: boolean
  onToggleLightMode: () => void
  standardFont: boolean
  onToggleFont: () => void

  // Zen mode
  zenMode: boolean
  onToggleZenMode: () => void

  // Save
  onSave: (format: 'md' | 'docx' | 'pdf') => void

  // Log actions
  onExportLog: () => void
}

export default function Toolbar({
  onCheckSpelling,
  onReadMyWriting,
  onClearHighlights,
  learningMode,
  onToggleLearningMode,
  lightMode,
  onToggleLightMode,
  standardFont,
  onToggleFont,
  zenMode,
  onToggleZenMode,
  onSave,
  onExportLog,
}: ToolbarProps) {
  const [saveOpen, setSaveOpen] = useState(false)
  const saveRef = useRef<HTMLDivElement>(null)
  const dropdownRef = useRef<HTMLDivElement>(null)

  // Close save dropdown when clicking outside
  useEffect(() => {
    if (!saveOpen) return
    const handleClickOutside = (e: MouseEvent) => {
      const target = e.target as Node
      if (
        saveRef.current && !saveRef.current.contains(target) &&
        dropdownRef.current && !dropdownRef.current.contains(target)
      ) {
        setSaveOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [saveOpen])

  return (
    <nav className="toolbar-sidebar" role="toolbar" aria-label="Writing tools">
      {/* Actions group */}
      <button
        className="toolbar-text-btn"
        aria-label="Check Spelling"
        onClick={onCheckSpelling}
      >
        Spell
      </button>
      <button
        className="toolbar-text-btn"
        aria-label="Read My Writing"
        onClick={onReadMyWriting}
      >
        Read
      </button>
      <button
        className="toolbar-text-btn"
        aria-label="Clear Highlights"
        onClick={onClearHighlights}
      >
        Clear
      </button>

      <div className="toolbar-divider" />

      {/* Toggles group */}
      <button
        className={`toolbar-text-btn${learningMode ? ' toggle-active' : ''}`}
        aria-label="Learning Mode"
        aria-pressed={learningMode}
        onClick={onToggleLearningMode}
      >
        Learn
      </button>
      <button
        className={`toolbar-text-btn${lightMode ? ' toggle-active' : ''}`}
        aria-label="Toggle Light/Dark Mode"
        aria-pressed={lightMode}
        onClick={onToggleLightMode}
      >
        {lightMode ? 'Dark' : 'Light'}
      </button>
      <button
        className={`toolbar-text-btn${standardFont ? ' toggle-active' : ''}`}
        aria-label="Toggle Font"
        aria-pressed={standardFont}
        onClick={onToggleFont}
      >
        Font
      </button>
      <button
        className={`toolbar-text-btn${zenMode ? ' toggle-active' : ''}`}
        aria-label="Zen Mode"
        aria-pressed={zenMode}
        onClick={onToggleZenMode}
      >
        Zen
      </button>

      <div className="toolbar-divider" />

      {/* Save dropdown */}
      <div className="toolbar-save-wrapper" ref={saveRef}>
        <button
          className={`toolbar-text-btn${saveOpen ? ' toggle-active' : ''}`}
          aria-label="Save As"
          onClick={() => setSaveOpen(!saveOpen)}
        >
          Save
        </button>
        {saveOpen && createPortal(
          <div
            ref={dropdownRef}
            className="save-dropdown"
            style={{
              position: 'fixed',
              left: saveRef.current ? saveRef.current.getBoundingClientRect().right + 6 : 0,
              top: saveRef.current ? saveRef.current.getBoundingClientRect().top : 0,
            }}
          >
            <button className="save-option" onClick={() => { onSave('md'); setSaveOpen(false) }}>
              Markdown (.md)
            </button>
            <button className="save-option" onClick={() => { onSave('docx'); setSaveOpen(false) }}>
              Word (.docx)
            </button>
            <button className="save-option" onClick={() => { onSave('pdf'); setSaveOpen(false) }}>
              PDF (.pdf)
            </button>
          </div>,
          document.body
        )}
      </div>

      <button
        className="toolbar-text-btn"
        aria-label="Export Spelling Log"
        onClick={onExportLog}
      >
        Log
      </button>

      <div className="toolbar-spacer" />
    </nav>
  )
}
