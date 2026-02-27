# src/api.py
import asyncio
import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.bot.agent import ask, ask_stream, reset_conversation

ROOT = Path(__file__).parent.parent.resolve()

app = FastAPI(title="RappIntelligence API", version="2.0.0")

# Archivos estáticos del frontend
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")


class ChatRequest(BaseModel):
    message: str
    session_id: str = ""   # ID único por pestaña / sesión del navegador


class ChatResponse(BaseModel):
    response: str


@app.get("/")
async def index():
    """Sirve la interfaz de chat."""
    return FileResponse(str(ROOT / "static" / "index.html"))


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Endpoint sincrónico (sin streaming). Útil para testing."""
    if not request.message.strip():
        return ChatResponse(response="Por favor escribe una pregunta.")
    response = await asyncio.to_thread(ask, request.message, request.session_id)
    return ChatResponse(response=response)


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Endpoint de streaming SSE (newline-delimited JSON).

    Eventos emitidos:
      {"type": "status", "text": "..."}   → estado del nodo activo
      {"type": "token",  "text": "..."}   → chunk de token del LLM
      {"type": "chart",  "html": "..."}   → HTML completo de la gráfica generada
      {"type": "done",   "text": ""}      → señal de fin de stream
      {"type": "error",  "text": "..."}   → error irrecuperable
    """
    if not request.message.strip():
        async def _empty():
            yield json.dumps({"type": "done", "text": "Por favor escribe una pregunta."}) + "\n"
        return StreamingResponse(_empty(), media_type="text/event-stream")

    return StreamingResponse(
        ask_stream(request.message, request.session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no", 
        },
    )


@app.post("/reset")
async def reset():
    """Genera un nuevo thread ID. El cliente debe actualizar su session_id local."""
    thread_id = reset_conversation()
    return {"status": "ok", "thread_id": thread_id}


@app.get("/health")
async def health():
    return {"status": "ok"}
