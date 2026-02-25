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

from tools_rappi import (
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
REPORT_DATE = datetime.datetime.now().strftime("%d de %B de %Y")


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

_DATE = REPORT_DATE

GENERATION_PROMPT = (
    "Eres el Director de Estrategia de Rappi. Produce un reporte ejecutivo que un VP o CEO "
    "pueda usar para TOMAR DECISIONES en 15 minutos.\n\n"

    "===================================================\n"
    "REGLAS FUNDAMENTALES\n"
    "===================================================\n"
    "1. PROHIBIDO incluir notas meta ('Nota de correccion', 'Aqui el reporte', 'He aplicado...', etc.). "
    "   El documento empieza DIRECTAMENTE con el titulo del reporte.\n"
    "2. PROHIBIDO usar titulos de seccion predeterminados o genericos. CADA titulo de seccion debe ser "
    "   un titulo PROFESIONAL, ESPECIFICO y DESCRIPTIVO que refleje el hallazgo principal de esa seccion. "
    "   Ejemplos buenos: 'Erosion de Margen por Pedido en Mercados Clave', "
    "   'Presion sobre Conversion en Funnel de Restaurantes', 'Oportunidades de Crecimiento en Zonas Prioritarias'.\n"
    "   Ejemplos PROHIBIDOS: 'De lo General a lo Particular', 'Lo Que Esta Funcionando Bien', "
    "   'Alertas Criticas', 'Por Pais', 'Por Ciudad'.\n"
    "3. PROHIBIDO separar hallazgos positivos y negativos en secciones distintas. Cada seccion tematica "
    "   debe mostrar AMBOS extremos: las zonas/metricas que mas sufren Y las que mejor funcionan en ese "
    "   mismo tema. Esto construye narrativa y permite comparar.\n"
    "4. PROHIBIDO citar porcentajes sin explicar que significa para el negocio.\n"
    "5. PROHIBIDO usar nombres de categorias ('Monetizacion', 'Expansion') en lugar de metricas reales.\n"
    "6. PROHIBIDO recomendaciones vagas ('investigar causas', 'hacer seguimiento', 'monitorear').\n"
    "7. OBLIGATORIO: cada mencion de metrica incluye su impacto de negocio.\n"
    "8. OBLIGATORIO: recomendaciones = zona + metrica + accion tactica + horizonte temporal.\n"
    "9. MAXIMA CALIDAD: 5 insights perfectos > 20 estadisticas sin contexto.\n"
    "10. NARRATIVA CONSISTENTE: Si reportas que gran parte del ecosistema se deteriora pero la rentabilidad (Gross Profit) mejora, DEBES conectar ambos hechos logica y ejecutivamente ('A pesar de la contraccion en volumen, la contencion de gastos mejoro el Gross Profit'). Nunca dejes datos contradictorios sueltos.\n"
    "11. CAUSALIDAD (ROOT CAUSE): No te limites a describir 'que' cayo. Formula hipotesis basadas en negocio (ej. si cae SS > ATC CVR, sugiere 'falta de fotos optimas o precios desalineados en menues' en lugar de un generico 'cae conversion').\n"
    "12. ACCIONABILIDAD TACTICA: Prohibido usar 'estrategias de retencion', 'marketing focalizado' o 'revisar'. Usa verbos tacticos ('Auditar top 20 restaurantes', 'Pausar campañas de descuento en X').\n"
    "13. NARRATIVA COMPARATIVA: NUNCA menciones un numero aislado (ej. 'Engativa tiene 2.114 de Gross Profit'). SIEMPRE comparalo con la media de la ciudad, pais o segmento que el dato provea (ej. 'Engativa esta en 2.114, una brecha severa contra la media de 3.5 en Bogota').\n\n"

    "DICCIONARIO DE METRICAS (usa SIEMPRE para explicar cada metrica):\n"
    "- % PRO Users Who Breakeven: Usuarios con suscripcion Pro cuyo valor generado ha cubierto el costo total de su membresia / Total de usuarios suscripcion Pro\n"
    "- % Restaurants Sessions With Optimal Assortment: Sesiones con un minimo de 40 restaurantes / Total de sesiones\n"
    "- Gross Profit UE: Margen bruto de ganancia / Total de ordenes\n"
    "- Lead Penetration: Tiendas habilitadas en Rappi / (Tiendas previamente identificadas como prospectos + Tiendas habilitadas + tiendas que salieron)\n"
    "- MLTV Top Verticals Adoption: Usuarios con ordenes en diferentes verticales (restaurantes, super, pharmacy, liquors) / Total usuarios\n"
    "- Non-Pro PTC > OP: Conversion de usuarios No Pro en 'Proceed to Checkout' a 'Order Placed'\n"
    "- Perfect Orders: Orders sin cancelaciones o defectos o demora / Total de ordenes\n"
    "- Pro Adoption: Usuarios suscripcion Pro / Total usuarios de Rappi\n"
    "- Restaurants Markdowns / GMV: Descuentos totales en ordenes de restaurantes / Total Gross Merchandise Value Restaurantes\n"
    "- Restaurants SS > ATC CVR: Conversion en restaurantes de 'Select Store' a 'Add to Cart'\n"
    "- Restaurants SST > SS CVR: Porcentaje de usuarios que, despues de seleccionar Restaurantes o Supermercados, seleccionan una tienda particular de la lista\n"
    "- Retail SST > SS CVR: Porcentaje de usuarios que, despues de seleccionar Supermercados, seleccionan una tienda particular de la lista\n"
    "- Turbo Adoption: Total de usuarios que compran en Turbo / total de usuarios de Rappi con tiendas de turbo disponible\n\n"

    "===================================================\n"
    "HERRAMIENTAS — EJECUTAR TODAS EN ORDEN\n"
    "===================================================\n"
    "1. get_ecosystem_health        2. get_priority_matrix\n"
    "3. get_country_summary         4. get_city_level_analysis\n"
    "5. get_critical_anomalies      6. get_worrisome_trends\n"
    "7. get_momentum_analysis       8. get_multivariate_risk_zones\n"
    "9. get_correlations_insights   10. get_benchmarking_insights\n"
    "11. get_opportunities_insights 12. get_investment_efficiency_insights\n"
    "13. get_bottleneck_diagnostics 14. get_monetization_gaps\n"
    "15. get_top_bottom_zones\n\n"

    "===================================================\n"
    "ESTRUCTURA DEL REPORTE (Markdown)\n"
    "===================================================\n"
    "El reporte tiene ~6-8 secciones tematicas + resumen ejecutivo + conclusiones.\n"
    "TU ELIGES los titulos de cada seccion basandote en los hallazgos.\n"
    "La estructura general es:\n\n"

    f"# Reporte Estrategico Semanal — Rappi\n"
    f"**Fecha:** {_DATE} | **Clasificacion:** Confidencial — Alta Direccion\n\n"
    "---\n\n"

    "## [Titulo profesional: Resumen de hallazgos clave]\n"
    "(Escribelo AL FINAL despues de analizar todo, pero POSICIONALO AL INICIO del documento.)\n"
    "(3-4 parrafos: estado general, hallazgo critico, hallazgo positivo, accion prioritaria.)\n"
    "(Tabla resumen: # | Hallazgo | Zona/Pais | Metrica | Por que importa — maximo 4 filas.)\n\n"
    "---\n\n"

    "## [Titulo profesional sobre el estado general del ecosistema]\n"
    "(Usa get_ecosystem_health + get_priority_matrix.)\n"
    "(Parrafo de contexto: paises, ciudades, zonas, ordenes.)\n"
    "(Integra la matriz de prioridades: las zonas High Priority estan mejor o peor? Contrasta.)\n"
    "(Tabla del ecosistema + tabla de prioridades, separadas por parrafo interpretativo.)\n\n"
    "---\n\n"

    "## [Titulo profesional sobre desempeno geografico]\n"
    "(Usa get_country_summary + get_city_level_analysis.)\n"
    "(1 parrafo pais: los 2 peores + el mejor. 1 parrafo ciudad: las 2 peores + las 2 mejores.)\n"
    "(Muestra AMBOS extremos en cada tabla y narrativa — NO separar secciones buenas/malas.)\n"
    "(Incluye tablas con parrafos interpretativos entre ellas.)\n\n"
    "---\n\n"

    "## [Titulo profesional sobre la metrica o tema mas critico esta semana]\n"
    "(Usa get_critical_anomalies + get_worrisome_trends + get_momentum_analysis.)\n"
    "(Integra en UNA narrativa: cambios bruscos WoW, deterioros estructurales 3+W, aceleracion.)\n"
    "(Para cada patron, nombra la zona peor Y la zona que esta mejorando en la misma metrica.)\n"
    "(Las mejoras significativas van junto a los deterioros, no en seccion aparte.)\n\n"
    "---\n\n"

    "## [Titulo profesional sobre riesgos compuestos]\n"
    "(Usa get_multivariate_risk_zones + get_correlations_insights.)\n"
    "(Explica impacto COMBINADO de metricas que fallan juntas — no solo listarlas.)\n"
    "(Usa correlaciones como early warning: si cae A, vigila B.)\n\n"
    "---\n\n"

    "## [Titulo profesional sobre brechas y oportunidades]\n"
    "(Usa get_benchmarking_insights + get_opportunities_insights.)\n"
    "(Muestra brechas entre zonas similares + zonas prioritarias bajo su potencial.)\n"
    "(Para cada oportunidad: accion concreta en 2-4 semanas.)\n\n"
    "---\n\n"
    
    "## [Titulo profesional sobre insights estrategicos de negocio]\n"
    "(Usa get_investment_efficiency_insights + get_bottleneck_diagnostics + get_monetization_gaps.)\n"
    "(Analiza la eficiencia de inversion vs crecimiento, diagnostica cuellos de botella en el funnel, y revela brechas de monetizacion (Pro, Turbo, MLTV).)\n\n"
    "---\n\n"

    "## [Titulo profesional sobre ranking general]\n"
    "(Usa get_top_bottom_zones.)\n"
    "(Top 5 vs Bottom 5: que patron las diferencia? Que metrica es la mas debil del bottom?)\n\n"
    "---\n\n"

    "## [Titulo profesional: Conclusiones y Hoja de Ruta]\n"
    "(4-5 conclusiones numeradas. Cada una conecta 2+ secciones y cita un numero.)\n"
    "(Tabla Plan de Accion: # | Iniciativa | Zona(s) | Metrica | Accion | Horizonte — 4 filas max.)\n\n"
    "---\n"
    f"*RappIntelligence Analytics Engine — {_DATE} — Confidencial*\n\n"

    "===================================================\n"
    "REGLAS DE FORMATO\n"
    "===================================================\n"
    "- PROHIBIDO dos tablas seguidas sin parrafo entre ellas.\n"
    "- Antes de cada tabla: 1 parrafo explicando que muestra y por que importa.\n"
    "- Despues de cada tabla (si hay otra): 1 oracion de transicion.\n"
    "- Tablas compactas: maximo 10 filas por tabla. Selecciona las mas relevantes.\n"
    "- El documento completo no debe exceder 5000 palabras.\n"
    "- Crea parrafos ricos en informacion, parrafos con 2,3,4 lineas no dicen mucho (tampoco inventes pero crea una buena narrativa).\n"
    "- En el texto que generas en cada parrafo incluye datos numericos para enriquecer la narrativa.\n"
)


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

    critique = (
        "Eres el VP de Estrategia de Rappi. Audita este reporte en 7 criterios:\n\n"
        + reporte
        + "\n\n---- CRITERIOS ----\n"
        "1. ¿El documento empieza directamente con el titulo y resumen ejecutivo (SIN notas meta del LLM)?\n"
        "2. ¿TODOS los titulos de seccion son profesionales, especificos y descriptivos "
        "   (NO genericos como 'Alertas Criticas', 'Lo Que Esta Funcionando Bien', 'Por Pais')?\n"
        "3. ¿Cada seccion tematica integra AMBOS extremos (positivos y negativos) en la misma narrativa, "
        "   en lugar de separarlos en secciones distintas?\n"
        "4. ¿Las Conclusiones conectan datos de al menos 2 secciones distintas en cada punto?\n"
        "5. ¿El Plan de Accion nombra zonas y metricas reales (no frases vagas)?\n"
        "6. ¿Las tablas son compactas (max ~10 filas) y cada una tiene parrafo introductorio?\n"
        "7. ¿No hay dos tablas consecutivas sin parrafo de texto entre ellas?\n\n"
        "Si los 7 son SI, responde EXACTAMENTE: 'APROBADO'\n"
        "Si alguno es NO, di cual falla y que debe corregirse. Se breve y especifico."
    )

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