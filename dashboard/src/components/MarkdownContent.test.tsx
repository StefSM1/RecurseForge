import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import MarkdownContent from './MarkdownContent'

describe('MarkdownContent', () => {
  it('renders headings and fenced code as semantic elements', () => {
    const html = renderToStaticMarkup(
      <MarkdownContent content={'# Title\n\n## Subtitle\n\n```python\nprint("ok")\n```'} />,
    )
    expect(html).toContain('<h1>Title</h1>')
    expect(html).toContain('<h2>Subtitle</h2>')
    expect(html).toContain('<pre>')
    expect(html).toContain('language-python')
    expect(html).toContain('print(&quot;ok&quot;)')
  })

  it('does not render raw HTML from model output', () => {
    const html = renderToStaticMarkup(
      <MarkdownContent content={'<script>alert("no")</script>'} />,
    )
    expect(html).not.toContain('<script>')
    expect(html).toContain('&lt;script&gt;')
  })
})
