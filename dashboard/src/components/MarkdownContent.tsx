import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

interface MarkdownContentProps {
  content: string
  className?: string
}

export default function MarkdownContent({
  content, className = '',
}: MarkdownContentProps) {
  return (
    <div className={`rf-markdown ${className}`.trim()}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          h1: ({ children }) => <h1>{children}</h1>,
          h2: ({ children }) => <h2>{children}</h2>,
          h3: ({ children }) => <h3>{children}</h3>,
          pre: ({ children }) => <pre>{children}</pre>,
          code: ({ children, className: codeClassName }) => (
            <code className={codeClassName}>{children}</code>
          ),
          a: ({ children, href }) => (
            <a href={href} target="_blank" rel="noreferrer">{children}</a>
          ),
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  )
}
