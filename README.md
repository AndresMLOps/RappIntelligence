<div align="center">
  <img src="data/RappiBot.png" alt="RappIntelligence Bot" width="600" />
</div>

# 🤖 RappIntelligence

RappIntelligence es un sistema impulsado por IA diseñado para democratizar el acceso a datos. Permite a los equipos consultar métricas operacionales utilizando lenguaje natural a través de un Bot interactivo, además de generar automatizadamente reportes estratégicos ejecutivos semanales mediante su motor de Insights.

## 🚀 Cómo Ejecutar el Proyecto

### 1. Requisitos Previos

1. Asegúrate de tener instalado Python 3.10+ y [uv](https://docs.astral.sh/uv/) (el gestor de dependencias utilizado en el proyecto).
2. Clona el repositorio e instala las dependencias:
   ```bash
   uv sync
   ```
3. Crea un archivo `.env` en la raíz del proyecto para alojar tus credenciales. Necesitarás como mínimo:
   ```env
   OPENAI_API_KEY=tu_clave_aqui
   ```
4. Agrega los datos de entrada en la ruta `data/`. Debes tener:
   - `df_metrics.csv`
   - `df_orders.csv`

---

### 2. Ejecutar el Agente Conversacional (Bot)

El bot sirve como una interfaz interactiva donde cualquier usuario puede explorar los datos haciendo preguntas sin necesidad de conocimientos técnicos o SQL.

```bash
uv run python main.py
```
Abre tu navegador web en: **http://localhost:8000** 

- **¿Qué puedes analizar?**
  - **Filtros rápidos:** *¿Cuáles son las 5 zonas con mayor Lead Penetration esta semana?*
  - **Comparativas:** *Compara Perfect Order entre zonas Wealthy y Non Wealthy en México.*
  - **Identificación de tendencias:** *Evolución de Gross Profit UE en Chapinero.*
  - **Explicación de insights:** *¿Qué zonas crecen más en órdenes y qué lo explica?*

---

### 3. Ejecutar el Generador Executivo de Reportes (Insights)

El módulo de Insights utiliza un pipeline avanzado (LangGraph ReAct) para analizar cruces de métricas a nivel país, ciudad y zona, y emitir recomendaciones tácticas fundamentadas.

```bash
uv run python Insights/main.py
```

- **¿Qué genera?**
  - El script evaluará todo el ecosistema y escribirá una pieza de narrativa profesional.
  - Podrás encontrar el resultado en formato `Reporte_Estrategico_Rappi.md` (Markdown) y `Reporte_Estrategico_Rappi.pdf` (PDF) dentro de la carpeta `Insights/`.
  - *Nota: Para la generación y el renderizado correcto del archivo PDF, debes tener `wkhtmltopdf` instalado en tu sistema y referenciado en el proyecto.*

---

## 🏗️ Arquitectura del Proyecto

- **Bot Principal (`scr/`)**: FastAPI (`api.py`) como orquestador del backend, Langgraph agents (`agent.py`) manejando enrutamiento e investigación técnica, y un Frontend en HTML/JS (`static/`).
- **Insights Pipeline (`Insights/`)**: Lógicas robustas de análisis pandas de Momentum, Z-score Benchmarking y Riesgo Multivariable cruzando métricas (`tools_rappi.py`), que alimentan la generación de contenido narrativo en `main.py`.
