import os
import datetime
import logging
import markdown
import pdfkit
from typing import Annotated
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent

from src.insights.tools import (
    get_priority_matrix,
    get_momentum_analysis,
    get_ecosystem_health,
    get_country_summary,
    get_city_level_analysis,
    get_critical_anomalies,
    get_worrisome_trends,
    get_multivariate_risk_zones,
    get_correlations_insights,
    get_benchmarking_insights,
    get_opportunities_insights,
    get_investment_efficiency_insights,
    get_bottleneck_diagnostics,
    get_monetization_gaps,
    get_top_bottom_zones,
)

from src.insights.prompts import GENERATION_PROMPT, REFLECTION_PROMPT, REPORT_DATE

from dotenv import load_dotenv
load_dotenv()

from langfuse.langchain import CallbackHandler

logging.getLogger("opentelemetry").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logging.getLogger("requests").setLevel(logging.CRITICAL)


def _get_insights_langfuse_handler():
    """Crea un CallbackHandler apuntando al proyecto Insights de Langfuse."""
    try:
        os.environ["LANGFUSE_SECRET_KEY"] = os.environ.get("LANGFUSE_INSIGHTS_SECRET_KEY", "")
        os.environ["LANGFUSE_PUBLIC_KEY"] = os.environ.get("LANGFUSE_INSIGHTS_PUBLIC_KEY", "")
        os.environ["LANGFUSE_HOST"] = os.environ.get("LANGFUSE_INSIGHTS_BASE_URL", "https://cloud.langfuse.com")
        handler = CallbackHandler()
        handler.trace_name = "rappi-insights"
        return handler
    except Exception:
        return None


PATH_WKHTMLTOPDF = r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe'

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


llm = ChatOpenAI(model="gpt-4o", temperature=0.3, max_tokens=16384)

TOOLS = [
    get_priority_matrix,
    get_momentum_analysis,
    get_ecosystem_health,
    get_country_summary,
    get_city_level_analysis,
    get_critical_anomalies,
    get_worrisome_trends,
    get_multivariate_risk_zones,
    get_correlations_insights,
    get_benchmarking_insights,
    get_opportunities_insights,
    get_investment_efficiency_insights,
    get_bottleneck_diagnostics,
    get_monetization_gaps,
    get_top_bottom_zones,
]
agent = create_react_agent(llm, TOOLS)

agent = create_react_agent(llm, TOOLS)


def generation_node(state: AgentState) -> dict:
    print("\n[GENERADOR] Ejecutando herramientas de analisis y construyendo reporte...")
    feedback = ""
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage) and "corregir:" in m.content.lower():
            feedback = f"\n\nCORRECCION SOLICITADA: {m.content}"
            break

    callbacks = []
    handler = _get_insights_langfuse_handler()
    if handler:
        callbacks.append(handler)

    result = agent.invoke(
        {"messages": [HumanMessage(content=GENERATION_PROMPT + feedback)]},
        config={"callbacks": callbacks},
    )
    return {"messages": [result["messages"][-1]]}


def reflection_node(state: AgentState) -> dict:
    print("[REVISOR] Auditando calidad del reporte...")
    reporte = state["messages"][-1].content

    critique = REFLECTION_PROMPT.format(reporte=reporte)

    callbacks = []
    handler = _get_insights_langfuse_handler()
    if handler:
        callbacks.append(handler)

    res = llm.invoke(critique, config={"callbacks": callbacks})
    verdict = res.content[:200]
    print(f"   -> {verdict}...")
    return {"messages": [HumanMessage(content=f"corregir: {res.content}")]}


def should_continue(state: AgentState):
    last = state["messages"][-1].content.upper()
    if "APROBADO" in last:
        print("[CONTROL] Reporte aprobado. ✓")
        return END
    # max 3 generation cycles: init + gen1 + refl1 + gen2 + refl2 + gen3 = 6
    if len(state["messages"]) > 5:
        print("[LIMITE] Max 3 iteraciones alcanzado. Cerrando con version actual.")
        return END
    return "reflect"


workflow = StateGraph(AgentState)
workflow.add_node("generate", generation_node)
workflow.add_node("reflect",  reflection_node)
workflow.add_edge(START, "generate")
workflow.add_conditional_edges("generate", should_continue, {"reflect": "reflect", END: END})
workflow.add_edge("reflect", "generate")
app = workflow.compile()

CSS = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Segoe UI',Roboto,Helvetica,Arial,sans-serif;font-size:13.5px;color:#1a202c;
     line-height:1.75;background:#f0f2f5;padding:32px}}
.wrap{{background:#fff;border-radius:16px;padding:48px 56px;max-width:1060px;margin:0 auto;
       box-shadow:0 8px 32px rgba(0,0,0,.09)}}
h1{{color:#FF441F;border-bottom:3px solid #FF441F;padding-bottom:10px;font-size:1.9em;
    margin-top:0;margin-bottom:24px;letter-spacing:-.3px}}
h2{{color:#1a202c;font-size:1.22em;font-weight:700;margin-top:44px;margin-bottom:16px;
    padding:10px 18px;border-left:5px solid #FF441F;background:#fff5f3;
    border-radius:0 8px 8px 0}}
h3{{color:#c0341a;font-size:1.06em;margin-top:28px;margin-bottom:12px;font-weight:600;
    padding-left:4px;border-bottom:1px solid #ffe0d9;padding-bottom:4px}}
p{{margin-bottom:16px;color:#374151;max-width:820px}}
ul,ol{{margin:10px 0 16px 22px}}
li{{margin-bottom:8px;padding:9px 14px;background:#fafafa;
    border:1px solid #e5e7eb;border-radius:6px;border-left:3px solid #d1d5db;list-style:none}}
li strong{{color:#1a202c}}
table{{width:100%;border-collapse:collapse;margin:18px 0 30px;font-size:12px;
       border-radius:10px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,.07)}}
thead tr{{background:#FF441F;color:#fff;text-align:left}}
thead th{{padding:10px 14px;font-weight:600;font-size:11.5px;letter-spacing:.4px;border:none}}
tbody tr:nth-child(even){{background:#fff5f3}}
tbody tr:nth-child(odd){{background:#fff}}
tbody tr:hover{{background:#fce8e3}}
tbody td{{padding:9px 14px;border-bottom:1px solid #f0f0f0;vertical-align:top;color:#374151}}
tbody td:first-child{{font-weight:500;color:#1a202c}}
hr{{border:none;border-top:2px solid #f0f0f0;margin:36px 0}}
strong{{color:#1a202c;font-weight:700}}
em{{color:#6b7280;font-style:italic}}
blockquote{{border-left:4px solid #FF441F;margin:18px 0;padding:12px 20px;
            background:#fff5f3;color:#374151;font-style:italic;border-radius:0 8px 8px 0}}
.insight-card{{background:#fffaf9;border:1px solid #ffd4c8;border-left:4px solid #FF441F;
               border-radius:0 8px 8px 0;padding:14px 20px;margin:16px 0 24px;
               color:#2d3748;font-size:13px;line-height:1.7}}
.insight-card strong{{color:#c0341a}}
code,pre{{background:#f8f9fa;border-radius:4px;
          font-family:'Consolas',monospace;font-size:11.4px}}
code{{padding:2px 6px}}
pre{{padding:14px 18px;overflow-x:auto;border-left:3px solid #FF441F;margin-bottom:20px}}
.footer{{text-align:center;font-size:10.5px;color:#9ca3af;margin-top:48px;
         padding-top:18px;border-top:1px solid #e5e7eb;letter-spacing:.3px}}
</style>
</head>
<body>
<div class="wrap">
{content}
<div class="footer">
  RappIntelligence Analytics Engine &nbsp;&middot;&nbsp; {date} &nbsp;&middot;&nbsp;
  Confidencial &mdash; Solo Alta Direccion
</div>
</div>
</body>
</html>"""


if __name__ == "__main__":
    print("RappIntelligence - Pipeline Estrategico v4.0")
    print(f"   Fecha: {REPORT_DATE}")
    print(f"   Modelo: gpt-4o | Max tokens: 16,384 | Max iteraciones: 3")
    print(f"   Herramientas: {len(TOOLS)} | Narrativa integrada\n")

    callbacks = []
    handler = _get_insights_langfuse_handler()
    if handler:
        callbacks.append(handler)

    final = app.invoke(
        {"messages": [HumanMessage(content="Genera el reporte estrategico.")]},
        config={"callbacks": callbacks},
    )

    md_content = next(
        (m.content for m in reversed(final["messages"]) if isinstance(m, AIMessage)),
        "Error: no se genero contenido."
    )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    md_path  = os.path.join(script_dir, "Reporte_Estrategico_Rappi.md")
    pdf_path = os.path.join(script_dir, "Reporte_Estrategico_Rappi.pdf")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"\nMarkdown: {md_path}")

    html_body = markdown.markdown(
        md_content,
        extensions=["tables", "fenced_code", "nl2br", "sane_lists"]
    )
    html_full = CSS.format(content=html_body, date=REPORT_DATE)

    try:
        cfg = pdfkit.configuration(wkhtmltopdf=PATH_WKHTMLTOPDF)
        opts = {
            "encoding": "UTF-8",
            "page-size": "A4",
            "margin-top": "10mm", "margin-bottom": "10mm",
            "margin-left": "8mm",  "margin-right": "8mm",
            "enable-local-file-access": "",
        }
        pdfkit.from_string(html_full, pdf_path, configuration=cfg, options=opts)
        print(f"PDF: {pdf_path}")
    except Exception as e:
        print(f"PDF fallo: {e} - Markdown disponible en {md_path}")