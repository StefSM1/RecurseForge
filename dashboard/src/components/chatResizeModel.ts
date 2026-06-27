export const DEFAULT_CHAT_WIDTH = 320
export const MIN_CHAT_WIDTH = 280
export const CHAT_WIDTH_STORAGE_KEY = 'recurseforge.chatWidth'

export function maximumChatWidth(viewportWidth: number): number {
  return Math.max(MIN_CHAT_WIDTH, Math.floor(viewportWidth * 0.7))
}

export function clampChatWidth(width: number, viewportWidth: number): number {
  return Math.min(
    maximumChatWidth(viewportWidth),
    Math.max(MIN_CHAT_WIDTH, Math.round(width)),
  )
}

export function chatWidthFromPointer(
  clientX: number,
  viewportWidth: number,
): number {
  return clampChatWidth(viewportWidth - clientX, viewportWidth)
}
