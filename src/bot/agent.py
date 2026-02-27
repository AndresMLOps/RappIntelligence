# src/bot/agent.py
import json
import uuid
from pathlib import Path
from typing import TypedDict, Annotated, List, AsyncGenerator

import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages

from src.bot.prompts import (
    SYSTEM_PROMPT,
    ROUTER_PROMPT,
    SEMANTIC_MAPPER_PROMPT,
    RESPONSE_FORMATTER_PROMPT,
    SUMMARIZER_PROMPT,
    METRICS_CONTEXT,
)
from src.bot.tools import get_data_schema
from src.core.observability import get_langfuse_handler

load_dotenv()

ROOT = Path(__file__).parent.parent.parent.resolve()
DATA_DIR = ROOT / "data"

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)
llm_analyst = ChatOpenAI(model="gpt-4o", temperature=0)
checkpointer = InMemorySaver()

_DF_METRICS = pd.read_csv(DATA_DIR / "df_metrics.csv")
_DF_ORDERS  = pd.read_csv(DATA_DIR / "df_orders.csv")


class AgentState(TypedDict, total=False):
    messages:       Annotated[List[BaseMessage], add_messages]
    summary:        str   
    route:          str   
    enhanced_query: str   
    data_analysis:  str 

def router_node(state: AgentState) -> dict:
    """
    Clasifica la intención del usuario con el LLM.
    Rutas: "data" (análisis de datos) | "general" (consulta sin datos).
    """
    human_msgs = [m for m in state.get("messages", []) if isinstance(m, HumanMessage)]
    last_msg = human_msgs[-1].content if human_msgs else ""

    prompt = ROUTER_PROMPT.format(user_query=last_msg)
    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        text = resp.content.replace("```json", "").replace("```", "").strip()
        route = json.loads(text).get("route", "data")
        if route not in ("data", "general"):
            route = "data"
    except Exception:
        route = "data"

    return {"route": route}


def semantic_mapper_node(state: AgentState) -> dict:
    """Traduce la consulta del usuario a un plan de análisis estructurado (JSON)."""
    human_msgs = [m for m in state.get("messages", []) if isinstance(m, HumanMessage)]
    last_msg = human_msgs[-1].content if human_msgs else ""

    schema = get_data_schema.func()
    prompt = SEMANTIC_MAPPER_PROMPT.format(schema=schema, user_query=last_msg)
    resp = llm.invoke([SystemMessage(content=prompt)])
    
    raw = resp.content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    
    try:
        plan = json.loads(raw)
        dataset_name = "df1" if plan.get("dataset") == "df1" else "df2"
        instruction_parts = []
        instruction_parts.append(f"PLAN DE ANÁLISIS:")
        instruction_parts.append(f"Dataset: {dataset_name}")
        instruction_parts.append(f"Métrica: {plan.get('metric', 'N/A')}")
        
        filters = plan.get("filters", {})
        if filters:
            filter_strs = [f"{k} == '{v}'" for k, v in filters.items()]
            instruction_parts.append(f"Filtros obligatorios: {', '.join(filter_strs)}")
        else:
            instruction_parts.append("Filtros: ninguno")
        
        group_by = plan.get("group_by", [])
        if group_by:
            instruction_parts.append(f"Agrupar por: {', '.join(group_by)}")
        
        time_cols = plan.get("time_columns", [])
        if time_cols:
            instruction_parts.append(f"Columnas de tiempo: {', '.join(time_cols)}")
        
        instruction_parts.append(f"Operación: {plan.get('operation', 'N/A')}")
        instruction_parts.append(f"Descripción: {plan.get('description', 'N/A')}")
        
        enhanced = "\n".join(instruction_parts)
    except (json.JSONDecodeError, KeyError):
        enhanced = raw
    
    return {"enhanced_query": enhanced}


def pandas_analyst_node(state: AgentState) -> dict:
    """Ejecuta el análisis de datos con el agente ReAct de Pandas (gpt-4o)."""
    analysis_dfs = {
        "df_metrics": _DF_METRICS,
        "df_orders": _DF_ORDERS
    }
    
    agent = create_pandas_dataframe_agent(
        llm_analyst,
        list(analysis_dfs.values()),
        verbose=True,
        agent_type="openai-tools",
        allow_dangerous_code=True,
        max_iterations=10,
    )

    enhanced_query = state.get("enhanced_query", "")
    instruction = f"""
Eres un analista de datos experto. Tienes acceso a dos DataFrames:

- df1 = df_metrics (12573 filas): Métricas como Perfect Orders, Lead Penetration %, Gross Profit UE.
  Columnas: COUNTRY, CITY, ZONE, ZONE_TYPE, ZONE_PRIORITIZATION, METRIC, L8W_ROLL...L0W_ROLL
  L0W_ROLL = semana actual, L8W_ROLL = hace 8 semanas.

- df2 = df_orders (1242 filas): Conteo de órdenes.
  Columnas: COUNTRY, CITY, ZONE, METRIC, L8W...L0W
  L0W = semana actual, L8W = hace 8 semanas.

{enhanced_query}

PROTOCOLO DE EJECUCIÓN:
1. Aplica TODOS los filtros indicados en el plan. Si dice COUNTRY == 'CO', filtra por ese país PRIMERO.
2. Si hay group_by, agrupa y calcula promedios (mean) por grupo.
3. Ordena los resultados de forma relevante.
4. ANTES DE DEVOLVER: verifica que:
   a) Todos los filtros del plan se aplicaron
   b) No hay filas duplicadas
   c) La granularidad es correcta (si piden comparar grupos, muestra grupos, no filas individuales)
   d) El resultado responde la pregunta original

Devuelve una tabla Markdown con los resultados.
"""
    try:
        result = agent.invoke({"input": instruction})
        return {"data_analysis": result["output"]}
    except Exception as e:
        return {"data_analysis": f"Error en el análisis: {str(e)}"}


def responder_node(state: AgentState) -> dict:
    """Formatea la respuesta final en lenguaje natural."""
    route = state.get("route", "data")

    if route == "general":
        human_msgs = [m for m in state.get("messages", []) if isinstance(m, HumanMessage)]
        last_msg = human_msgs[-1].content if human_msgs else ""
        resp = llm.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=last_msg)])
        return {"messages": [AIMessage(content=resp.content)]}

    analysis = state.get("data_analysis", "Sin resultados.")
    system_content = RESPONSE_FORMATTER_PROMPT.format(analysis_result=analysis)
    llm_messages = [
        SystemMessage(content=system_content + "\n\n" + METRICS_CONTEXT),
    ]
    resp = llm.invoke(llm_messages)
    return {"messages": [AIMessage(content=resp.content)]}


def summarizer_node(state: AgentState) -> dict:
    """Comprime la conversación cuando supera los 10 turnos."""
    messages = state.get("messages", [])
    prompt = SUMMARIZER_PROMPT.format(chat_history=messages)
    resp = llm.invoke([HumanMessage(content=prompt)])
    return {"summary": resp.content}


def _route_decision(state: AgentState) -> str:
    return "general" if state.get("route") == "general" else "data"


def _should_summarize(state: AgentState) -> str:
    return "summarize" if len(state.get("messages", [])) >= 10 else "end"


workflow = StateGraph(AgentState)

workflow.add_node("router",          router_node)
workflow.add_node("semantic_mapper", semantic_mapper_node)
workflow.add_node("analyst",         pandas_analyst_node)
workflow.add_node("responder",       responder_node)
workflow.add_node("summarizer",      summarizer_node)

workflow.add_edge(START, "router")

workflow.add_conditional_edges(
    "router",
    _route_decision,
    {
        "data":    "semantic_mapper",
        "general": "responder",
    },
)

workflow.add_edge("semantic_mapper", "analyst")
workflow.add_edge("analyst",         "responder")

workflow.add_conditional_edges(
    "responder",
    _should_summarize,
    {"summarize": "summarizer", "end": END},
)
workflow.add_edge("summarizer", END)

app = workflow.compile(checkpointer=checkpointer)


_NODE_STATUS = {
    "router":          "🔀 Clasificando consulta...",
    "semantic_mapper": "🗺️  Mapeando métricas al dataset...",
    "analyst":         "🔍 Analizando datos...",
    "responder":       "✍️  Redactando respuesta...",
    "summarizer":      "📝 Comprimiendo conversación...",
}


def ask(question: str, session_id: str = "") -> str:
    """Invocación sincrónica (para testing). Acepta un session_id opcional."""
    thread_id = session_id if session_id else str(uuid.uuid4())
    langfuse_handler = get_langfuse_handler(session_id=thread_id)
    config = {
        "configurable": {"thread_id": thread_id},
        "callbacks": [langfuse_handler],
    }
    try:
        result = app.invoke({"messages": [HumanMessage(content=question)]}, config)
        return result["messages"][-1].content
    except Exception as e:
        return f"❌ Error: {str(e)}"


async def ask_stream(question: str, session_id: str = "") -> AsyncGenerator[str, None]:
    """
    Generador SSE (newline-delimited JSON). Emite:
      {"type": "status", "text": "..."}   → estado del nodo activo
      {"type": "token",  "text": "..."}   → chunk de token del LLM (nodo responder)
      {"type": "done",   "text": ""}      → fin del stream
      {"type": "error",  "text": "..."}   → error irrecuperable
    """
    thread_id = session_id if session_id else str(uuid.uuid4())
    langfuse_handler = get_langfuse_handler(session_id=thread_id)
    config = {
        "configurable": {"thread_id": thread_id},
        "callbacks": [langfuse_handler],
    }

    try:
        async for event in app.astream_events(
            {"messages": [HumanMessage(content=question)]},
            config,
            version="v2",
        ):
            kind      = event.get("event", "")
            node_name = event.get("metadata", {}).get("langgraph_node", "")

            if kind == "on_chain_start" and node_name in _NODE_STATUS:
                yield json.dumps({"type": "status", "text": _NODE_STATUS[node_name]}) + "\n"

            elif kind == "on_chat_model_stream" and node_name == "responder":
                chunk = event.get("data", {}).get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    yield json.dumps({"type": "token", "text": chunk.content}) + "\n"

        yield json.dumps({"type": "done", "text": ""}) + "\n"

    except Exception as e:
        yield json.dumps({"type": "error", "text": f"❌ {str(e)}"}) + "\n"


def reset_conversation() -> str:
    """Genera un nuevo thread ID (equivalente a iniciar una sesión nueva)."""
    return str(uuid.uuid4())