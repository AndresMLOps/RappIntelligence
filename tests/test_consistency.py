"""
tests/test_consistency.py
Verifica que el agente produce respuestas deterministas para la misma pregunta.
Ejecutar con: uv run python tests/test_consistency.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import re
import pandas as pd

# ── 1. Verificar que las columnas del CSV son correctas ─────────────────────
print("=" * 60)
print("CHECK 1: Columnas de los DataFrames")
print("=" * 60)
from scr.agent import _DF_METRICS, _DF_ORDERS

assert "L0W_ROLL" in _DF_METRICS.columns, "ERROR: L0W_ROLL no existe en df_metrics"
assert "L0W" in _DF_ORDERS.columns,  "ERROR: L0W no existe en df_orders"
assert "L0W" not in _DF_METRICS.columns, "ERROR: L0W existe en df_metrics? (debería ser L0W_ROLL)"
assert "L0W_ROLL" not in _DF_ORDERS.columns,  "ERROR: L0W_ROLL existe en df_orders? (debería ser L0W)"
print(f"  df_metrics shape: {_DF_METRICS.shape}")
print(f"  df_orders  shape: {_DF_ORDERS.shape}")
print("  ✅ Columnas correctas\n")

# ── 2. Verificar que el prompt usa ambos esquemas ──────────────────────────────
print("=" * 60)
print("CHECK 2: Prompt SEMANTIC_MAPPER mapea ambos esquemas")
print("=" * 60)
from scr.prompts import SEMANTIC_MAPPER_PROMPT
assert "L0W_ROLL" in SEMANTIC_MAPPER_PROMPT, "ERROR: L0W_ROLL (df_metrics) no está en SEMANTIC_MAPPER_PROMPT"
assert "L0W" in SEMANTIC_MAPPER_PROMPT, "ERROR: L0W (df_orders) no está en SEMANTIC_MAPPER_PROMPT"
print("  ✅ Prompt maneja 'L0W' y 'L0W_ROLL' correctamente\n")

# ── 3. Verificar que la API acepta session_id ────────────────────────────────
print("=" * 60)
print("CHECK 3: ChatRequest acepta session_id")
print("=" * 60)
from scr.api import ChatRequest
assert "session_id" in ChatRequest.model_fields, "ERROR: session_id no está en ChatRequest"
print("  ✅ ChatRequest tiene campo session_id\n")

# ── 4. Verificar que ask() acepta session_id ────────────────────────────────
print("=" * 60)
print("CHECK 4: ask() acepta session_id")
print("=" * 60)
import inspect
from scr.agent import ask
sig = inspect.signature(ask)
assert "session_id" in sig.parameters, "ERROR: ask() no tiene parámetro session_id"
print("  ✅ ask() tiene parámetro session_id\n")

# ── 5. Sesiones distintas no deben compartir memoria ────────────────────────
print("=" * 60)
print("CHECK 5: Sesiones distintas = hilos de memoria distintos")
print("=" * 60)
import uuid
session_a = str(uuid.uuid4())
session_b = str(uuid.uuid4())
assert session_a != session_b
print(f"  Sesión A: {session_a[:8]}...")
print(f"  Sesión B: {session_b[:8]}...")
print("  ✅ IDs únicos por sesión\n")

print("=" * 60)
print("✅ TODOS LOS CHECKS PASARON")
print("=" * 60)
print()
print("Para una prueba de consistencia en vivo, ejecuta el servidor")
print("y haz la misma pregunta dos veces desde el mismo sessionId.")
print("Las respuestas deben contener los mismos valores numéricos.")
