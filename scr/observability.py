# scr/observability.py
"""
Módulo de observabilidad con Langfuse.
Proporciona un CallbackHandler que se inyecta en las invocaciones de LangGraph
para trazar automáticamente cada nodo, LLM call y tool execution.

Credenciales leídas automáticamente de variables de entorno:
  LANGFUSE_SECRET_KEY, LANGFUSE_PUBLIC_KEY, LANGFUSE_BASE_URL
"""
from langfuse.langchain import CallbackHandler


def get_langfuse_handler(session_id: str = "", user_id: str = "anonymous") -> CallbackHandler:
    """Crea un CallbackHandler de Langfuse vinculado a una sesión."""
    handler = CallbackHandler()
    handler.session_id = session_id or None
    handler.user_id = user_id
    handler.trace_name = "rappi-chat"
    return handler



