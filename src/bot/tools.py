# scr/tools.py
import json
import os
from pathlib import Path

import pandas as pd
from langchain.tools import tool

ROOT = Path(os.environ.get("RAPP_ROOT", Path(__file__).parent.parent.parent)).resolve()
DATA_DIR = ROOT / "data"


@tool
def get_data_schema() -> str:
    """Retorna el esquema completo de los datasets: columnas, valores únicos de filtros, y diccionario temporal."""
    schema = {
        "temporal_dictionary": {
            "description": "Los datos tienen 9 semanas de historia. L0W = semana actual (más reciente), L8W = hace 8 semanas.",
            "df_metrics_columns": "L8W_ROLL, L7W_ROLL, L6W_ROLL, L5W_ROLL, L4W_ROLL, L3W_ROLL, L2W_ROLL, L1W_ROLL, L0W_ROLL",
            "df_orders_columns": "L8W, L7W, L6W, L5W, L4W, L3W, L2W, L1W, L0W",
            "examples": {
                "esta semana": "L0W_ROLL (metrics) / L0W (orders)",
                "últimas 4 semanas": "comparar L4W_ROLL vs L0W_ROLL (metrics) / L4W vs L0W (orders)",
                "tendencia 8 semanas": "usar L8W_ROLL hasta L0W_ROLL en secuencia",
            },
        },
    }

    for fname in ["df_metrics.csv", "df_orders.csv"]:
        fpath = DATA_DIR / fname
        if not fpath.exists():
            continue
        df = pd.read_csv(fpath)
        
        categorical_values = {}
        for col in df.select_dtypes(include=["object"]).columns:
            unique_vals = sorted(df[col].dropna().unique().tolist())
            if len(unique_vals) <= 50:
                categorical_values[col] = unique_vals
            else:
                categorical_values[col] = f"{len(unique_vals)} valores únicos (ej: {unique_vals[:10]})"

        schema[fname] = {
            "columns": df.columns.tolist(),
            "shape": list(df.shape),
            "filter_values": categorical_values,
            "sample": df.head(3).to_dict(orient="records"),
        }

    return json.dumps(schema, ensure_ascii=False, indent=2)