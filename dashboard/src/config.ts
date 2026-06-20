// Backend connection configuration
// The dashboard connects DIRECTLY to the FastAPI backend (bypasses Vite proxy)
// to avoid WebSocket buffering issues.
export const BACKEND_HOST = 'localhost'
export const BACKEND_PORT = 8100
export const BACKEND_BASE_URL = `http://${BACKEND_HOST}:${BACKEND_PORT}`
export const BACKEND_WS_URL = `ws://${BACKEND_HOST}:${BACKEND_PORT}`
