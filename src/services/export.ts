import { Document, Packer, Paragraph, TextRun } from 'docx'
import { saveAs } from 'file-saver'

function getTimestamp(): string {
  return new Date().toISOString().split('T')[0]
}

/**
 * Convert TipTap HTML to plain text with paragraph breaks.
 */
function htmlToPlainText(html: string): string {
  const div = document.createElement('div')
  div.innerHTML = html
  const paragraphs = div.querySelectorAll('p')
  if (paragraphs.length === 0) {
    return div.textContent || ''
  }
  return Array.from(paragraphs)
    .map(p => p.textContent || '')
    .join('\n\n')
}

export function saveAsMarkdown(html: string): void {
  const text = htmlToPlainText(html)
  const blob = new Blob([text], { type: 'text/markdown;charset=utf-8' })
  saveAs(blob, `writing-${getTimestamp()}.md`)
}

export async function saveAsDocx(html: string): Promise<void> {
  const text = htmlToPlainText(html)
  const paragraphs = text.split('\n\n').map(
    line =>
      new Paragraph({
        children: [
          new TextRun({
            text: line,
            size: 28, // 14pt
            font: 'OpenDyslexic',
          }),
        ],
        spacing: { after: 200 },
      })
  )

  const doc = new Document({
    sections: [{ children: paragraphs }],
  })

  const blob = await Packer.toBlob(doc)
  saveAs(blob, `writing-${getTimestamp()}.docx`)
}

export async function saveAsPdf(html: string): Promise<void> {
  // Dynamic import to avoid SSR issues
  const html2pdf = (await import('html2pdf.js')).default

  // Create a styled container for PDF rendering
  const container = document.createElement('div')
  container.innerHTML = html
  container.style.fontFamily = 'OpenDyslexic, sans-serif'
  container.style.fontSize = '14pt'
  container.style.lineHeight = '1.8'
  container.style.padding = '40px'
  container.style.color = '#333'

  await html2pdf()
    .set({
      margin: 1,
      filename: `writing-${getTimestamp()}.pdf`,
      html2canvas: { scale: 2 },
      jsPDF: { unit: 'in', format: 'letter', orientation: 'portrait' },
    })
    .from(container)
    .save()
}
