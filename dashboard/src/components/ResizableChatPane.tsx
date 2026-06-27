import { useEffect, useState } from 'react'
import type { KeyboardEvent, PointerEvent, ReactNode } from 'react'
import {
  CHAT_WIDTH_STORAGE_KEY,
  DEFAULT_CHAT_WIDTH,
  MIN_CHAT_WIDTH,
  chatWidthFromPointer,
  clampChatWidth,
  maximumChatWidth,
} from './chatResizeModel'

function initialWidth(): number {
  if (typeof window === 'undefined') return DEFAULT_CHAT_WIDTH
  const stored = Number(window.localStorage.getItem(CHAT_WIDTH_STORAGE_KEY))
  return clampChatWidth(
    Number.isFinite(stored) && stored > 0 ? stored : DEFAULT_CHAT_WIDTH,
    window.innerWidth,
  )
}

export default function ResizableChatPane({ children }: { children: ReactNode }) {
  const [width, setWidth] = useState(initialWidth)
  const [dragging, setDragging] = useState(false)

  useEffect(() => {
    window.localStorage.setItem(CHAT_WIDTH_STORAGE_KEY, String(width))
  }, [width])

  useEffect(() => {
    const handleResize = () => {
      setWidth(current => clampChatWidth(current, window.innerWidth))
    }
    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [])

  useEffect(() => {
    if (!dragging) return
    const previousCursor = document.body.style.cursor
    const previousSelection = document.body.style.userSelect
    document.body.style.cursor = 'col-resize'
    document.body.style.userSelect = 'none'
    return () => {
      document.body.style.cursor = previousCursor
      document.body.style.userSelect = previousSelection
    }
  }, [dragging])

  const resizeFromPointer = (event: PointerEvent<HTMLDivElement>) => {
    if (!dragging) return
    setWidth(chatWidthFromPointer(event.clientX, window.innerWidth))
  }

  const stopDragging = (event: PointerEvent<HTMLDivElement>) => {
    if (!dragging) return
    setDragging(false)
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId)
    }
  }

  const resizeFromKeyboard = (event: KeyboardEvent<HTMLDivElement>) => {
    const step = event.shiftKey ? 48 : 16
    let nextWidth = width
    if (event.key === 'ArrowLeft') nextWidth += step
    else if (event.key === 'ArrowRight') nextWidth -= step
    else if (event.key === 'Home') nextWidth = MIN_CHAT_WIDTH
    else if (event.key === 'End') nextWidth = maximumChatWidth(window.innerWidth)
    else return
    event.preventDefault()
    setWidth(clampChatWidth(nextWidth, window.innerWidth))
  }

  return (
    <section
      className="rf-chat-pane"
      style={{ width }}
      aria-label="Resizable chat panel"
    >
      <div
        className={`rf-chat-resizer ${dragging ? 'rf-chat-resizer--active' : ''}`}
        role="separator"
        aria-label="Resize Chat Interface"
        aria-orientation="vertical"
        aria-valuemin={MIN_CHAT_WIDTH}
        aria-valuemax={maximumChatWidth(
          typeof window === 'undefined' ? DEFAULT_CHAT_WIDTH : window.innerWidth,
        )}
        aria-valuenow={width}
        tabIndex={0}
        onPointerDown={event => {
          event.currentTarget.setPointerCapture(event.pointerId)
          setDragging(true)
        }}
        onPointerMove={resizeFromPointer}
        onPointerUp={stopDragging}
        onPointerCancel={stopDragging}
        onKeyDown={resizeFromKeyboard}
      />
      <div className="min-w-0 flex-1 h-full">{children}</div>
    </section>
  )
}
