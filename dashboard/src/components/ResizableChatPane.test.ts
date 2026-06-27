import { describe, expect, it } from 'vitest'
import {
  MIN_CHAT_WIDTH,
  chatWidthFromPointer,
  clampChatWidth,
  maximumChatWidth,
} from './chatResizeModel'

describe('resizable chat pane sizing', () => {
  it('enforces the 280px minimum', () => {
    expect(clampChatWidth(100, 1200)).toBe(MIN_CHAT_WIDTH)
    expect(chatWidthFromPointer(1190, 1200)).toBe(MIN_CHAT_WIDTH)
  })

  it('caps width at 70 percent of the viewport', () => {
    expect(maximumChatWidth(1000)).toBe(700)
    expect(clampChatWidth(900, 1000)).toBe(700)
  })

  it('derives chat width from the right viewport edge', () => {
    expect(chatWidthFromPointer(800, 1200)).toBe(400)
  })
})
