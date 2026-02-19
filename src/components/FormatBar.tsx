import { Editor } from '@tiptap/react'

interface FormatBarProps {
  editor: Editor | null
}

export default function FormatBar({ editor }: FormatBarProps) {
  if (!editor) return null

  return (
    <div className="format-bar" role="toolbar" aria-label="Text formatting">
      <button
        className={`fmt-btn${editor.isActive('bold') ? ' fmt-active' : ''}`}
        onClick={() => editor.chain().focus().toggleBold().run()}
        aria-label="Bold"
        aria-pressed={editor.isActive('bold')}
        title="Bold (Ctrl+B)"
      >
        <strong>B</strong>
      </button>
      <button
        className={`fmt-btn${editor.isActive('italic') ? ' fmt-active' : ''}`}
        onClick={() => editor.chain().focus().toggleItalic().run()}
        aria-label="Italic"
        aria-pressed={editor.isActive('italic')}
        title="Italic (Ctrl+I)"
      >
        <em>I</em>
      </button>
      <button
        className={`fmt-btn${editor.isActive('underline') ? ' fmt-active' : ''}`}
        onClick={() => editor.chain().focus().toggleUnderline().run()}
        aria-label="Underline"
        aria-pressed={editor.isActive('underline')}
        title="Underline (Ctrl+U)"
      >
        <u>U</u>
      </button>
      <button
        className={`fmt-btn${editor.isActive('strike') ? ' fmt-active' : ''}`}
        onClick={() => editor.chain().focus().toggleStrike().run()}
        aria-label="Strikethrough"
        aria-pressed={editor.isActive('strike')}
        title="Strikethrough"
      >
        <s>S</s>
      </button>

      <div className="fmt-sep" />

      <button
        className={`fmt-btn${editor.isActive('heading', { level: 1 }) ? ' fmt-active' : ''}`}
        onClick={() => editor.chain().focus().toggleHeading({ level: 1 }).run()}
        aria-label="Heading 1"
        aria-pressed={editor.isActive('heading', { level: 1 })}
        title="Heading 1"
      >
        H1
      </button>
      <button
        className={`fmt-btn${editor.isActive('heading', { level: 2 }) ? ' fmt-active' : ''}`}
        onClick={() => editor.chain().focus().toggleHeading({ level: 2 }).run()}
        aria-label="Heading 2"
        aria-pressed={editor.isActive('heading', { level: 2 })}
        title="Heading 2"
      >
        H2
      </button>

      <div className="fmt-sep" />

      <button
        className={`fmt-btn${editor.isActive('bulletList') ? ' fmt-active' : ''}`}
        onClick={() => editor.chain().focus().toggleBulletList().run()}
        aria-label="Bullet List"
        aria-pressed={editor.isActive('bulletList')}
        title="Bullet List"
      >
        &bull; List
      </button>
      <button
        className={`fmt-btn${editor.isActive('orderedList') ? ' fmt-active' : ''}`}
        onClick={() => editor.chain().focus().toggleOrderedList().run()}
        aria-label="Numbered List"
        aria-pressed={editor.isActive('orderedList')}
        title="Numbered List"
      >
        1. List
      </button>
      <button
        className={`fmt-btn${editor.isActive('blockquote') ? ' fmt-active' : ''}`}
        onClick={() => editor.chain().focus().toggleBlockquote().run()}
        aria-label="Quote"
        aria-pressed={editor.isActive('blockquote')}
        title="Block Quote"
      >
        &ldquo; Quote
      </button>

      <div className="fmt-sep" />

      <button
        className="fmt-btn"
        onClick={() => editor.chain().focus().undo().run()}
        disabled={!editor.can().undo()}
        aria-label="Undo"
        title="Undo (Ctrl+Z)"
      >
        Undo
      </button>
      <button
        className="fmt-btn"
        onClick={() => editor.chain().focus().redo().run()}
        disabled={!editor.can().redo()}
        aria-label="Redo"
        title="Redo (Ctrl+Y)"
      >
        Redo
      </button>
    </div>
  )
}
