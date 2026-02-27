import os
import warnings
import pandas as pd
import numpy as np
from langchain_core.tools import tool

warnings.filterwarnings("ignore")

try:
    df_metrics = pd.read_csv("../data/df_metrics.csv")
    df_orders  = pd.read_csv("../data/df_orders.csv")

    if "L0W_ROLL" in df_metrics.columns:
        L0, L1, L2, L3 = "L0W_ROLL", "L1W_ROLL", "L2W_ROLL", "L3W_ROLL"
        L4 = "L4W_ROLL" if "L4W_ROLL" in df_metrics.columns else None
    else:
        L0 = "L0W" if "L0W" in df_metrics.columns else None
        L1 = "L1W" if "L1W" in df_metrics.columns else None
        L2 = "L2W" if "L2W" in df_metrics.columns else None
        L3 = "L3W" if "L3W" in df_metrics.columns else None
        L4 = "L4W" if "L4W" in df_metrics.columns else None

    _dfo = df_orders[df_orders["METRIC"] == "Orders"].copy()
    _oc  = next((c for c in ["L0W", "L0W_ROLL"] if c in _dfo.columns), None)
    _oc1 = next((c for c in ["L1W", "L1W_ROLL"] if c in _dfo.columns), None)
    if _oc:
        _dfo = _dfo.rename(columns={_oc: "ORDERS_L0W"})
    if _oc1:
        _dfo = _dfo.rename(columns={_oc1: "ORDERS_L1W"})
        
    merge_cols = ["COUNTRY","CITY","ZONE"]
    if "ORDERS_L0W" in _dfo.columns: merge_cols.append("ORDERS_L0W")
    if "ORDERS_L1W" in _dfo.columns: merge_cols.append("ORDERS_L1W")
        
    df_merged = pd.merge(df_metrics, _dfo[merge_cols],
                         on=["COUNTRY","CITY","ZONE"], how="left")
    if "ORDERS_L0W" in df_merged.columns:
        df_merged["ORDERS_L0W"] = pd.to_numeric(df_merged["ORDERS_L0W"], errors="coerce").fillna(0)
    if "ORDERS_L1W" in df_merged.columns:
        df_merged["ORDERS_L1W"] = pd.to_numeric(df_merged["ORDERS_L1W"], errors="coerce").fillna(0)
except Exception as e:
    print(f"⚠️ {e}")
    df_merged = df_metrics = pd.DataFrame()
    L0 = L1 = L2 = L3 = L4 = None

MP = {  # polarity: 1 = higher is better, -1 = lower is better
    "% PRO Users Who Breakeven": 1,
    "% Restaurants Sessions With Optimal Assortment": 1,
    "Gross Profit UE": 1,
    "Lead Penetration": 1,
    "MLTV Top Verticals Adoption": 1,
    "Non-Pro PTC > OP": 1,
    "Perfect Orders": 1,
    "Pro Adoption": 1,
    "Restaurants Markdowns / GMV": -1,
    "Restaurants SS > ATC CVR": 1,
    "Restaurants SST > SS CVR": 1,
    "Retail SST > SS CVR": 1,
    "Turbo Adoption": 1,
}

DIMS = {
    "🛒 Experiencia del Usuario": ["Perfect Orders", "Non-Pro PTC > OP", "% Restaurants Sessions With Optimal Assortment"],
    "💰 Monetización y Rentabilidad": ["Gross Profit UE", "Pro Adoption", "% PRO Users Who Breakeven", "Restaurants Markdowns / GMV"],
    "🌍 Expansión de Mercado": ["Lead Penetration", "MLTV Top Verticals Adoption", "Turbo Adoption"],
    "🔄 Conversión en Verticales": ["Restaurants SS > ATC CVR", "Restaurants SST > SS CVR", "Retail SST > SS CVR"],
}

RISK_COMBOS = [
    ("Riesgo Experiencia + Conversión",
     ["Perfect Orders", "Non-Pro PTC > OP", "Restaurants SS > ATC CVR"],
     "Mala calidad de entrega combinada con bajo funnel de conversión → impacto directo en retención."),
    ("Riesgo Monetización",
     ["Gross Profit UE", "Pro Adoption", "% PRO Users Who Breakeven"],
     "Márgenes comprimidos con baja adopción Pro → modelo de unit economics comprometido."),
    ("Riesgo Expansión de Mercado",
     ["Lead Penetration", "MLTV Top Verticals Adoption", "Turbo Adoption"],
     "Pocas tiendas habilitadas + baja adopción de verticales → zona con potencial no explotado."),
    ("Riesgo de Assortment",
     ["% Restaurants Sessions With Optimal Assortment", "Restaurants SST > SS CVR", "Retail SST > SS CVR"],
     "Catálogo insuficiente que erosiona la conversión en toda la vertical."),
]

def _n(v, d=3):
    try:
        return f"{float(v):.{d}f}"
    except Exception:
        return str(v)

def _prep(source=None):
    """Return a copy of df_merged with polarity, WoW change and derived flags.

    Derived columns added:
    - WOW          : week-over-week change L0 vs L1
    - WOW_PREV     : week-over-week change L1 vs L2 (previous period)
    - ACCEL        : WOW - WOW_PREV (negative = deterioration accelerating)
    - STREAK       : consecutive weeks of deterioration (0-8)
    - VOLATILITY   : std of available weekly values (L8W..L0W normalised by L0)
    - DETERI       : boolean — is this row deteriorating (vs polarity)?
    - ABS_WOW      : abs value of WOW
    """
    df = (source if source is not None else df_merged).copy()

    # Week columns available in this dataset
    week_cols = [c for c in ["L8W","L7W","L6W","L5W","L4W","L3W","L2W","L1W","L0W",
                              "L8W_ROLL","L7W_ROLL","L6W_ROLL","L5W_ROLL","L4W_ROLL",
                              "L3W_ROLL","L2W_ROLL","L1W_ROLL","L0W_ROLL"]
                 if c in df.columns]
    for c in week_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["POL"] = df["METRIC"].map(MP).fillna(1)

    if L0 and L1 and L0 in df.columns and L1 in df.columns:
        df = df.dropna(subset=[L0, L1])
        df = df[df[L1].abs() > 5e-4]
        df["WOW"] = (df[L0] - df[L1]) / df[L1].abs()
        df["DETERI"] = (df["WOW"] * df["POL"]) < 0
        df["ABS_WOW"] = df["WOW"].abs()

        # Momentum: acceleration = WoW this week minus WoW last week
        if L2 and L2 in df.columns:
            safe_l1 = df[L1].abs().replace(0, np.nan)
            safe_l2 = df[L2].abs().replace(0, np.nan)
            df["WOW_PREV"] = ((df[L1] - df[L2]) / safe_l2).where(safe_l2.notna(), np.nan)
            df["ACCEL"] = df["WOW"] - df["WOW_PREV"]  # negative & pol=+1 → deterioration accelerating
        else:
            df["WOW_PREV"] = np.nan
            df["ACCEL"] = np.nan

        # Streak: consecutive weeks of bad direction (max 8)
        ordered = []
        for col in ["L8W","L7W","L6W","L5W","L4W","L3W","L2W","L1W","L0W",
                    "L8W_ROLL","L7W_ROLL","L6W_ROLL","L5W_ROLL","L4W_ROLL",
                    "L3W_ROLL","L2W_ROLL","L1W_ROLL","L0W_ROLL"]:
            if col in df.columns:
                ordered.append(col)
        # deduplicate preserving order (ROLL and plain may coexist)
        seen = set(); ordered = [c for c in ordered if not (c.replace("_ROLL","") in seen or seen.add(c.replace("_ROLL","")))]
        # oldest → newest
        if len(ordered) >= 2:
            def _streak_row(row):
                vals = [row[c] for c in ordered if pd.notna(row.get(c))]
                if len(vals) < 2:
                    return 0
                pol = row["POL"]
                streak = 0
                # walk from newest (last) backwards
                for i in range(len(vals)-1, 0, -1):
                    diff = (vals[i] - vals[i-1]) * pol
                    if diff < 0:
                        streak += 1
                    else:
                        break
                return streak
            df["STREAK"] = df.apply(_streak_row, axis=1)
        else:
            df["STREAK"] = 0

        # Volatility: coefficient of variation over available weeks
        avail_w = [c for c in ordered if c in df.columns]
        if len(avail_w) >= 3:
            df["VOLATILITY"] = df[avail_w].std(axis=1) / (df[avail_w].mean(axis=1).abs() + 1e-9)
        else:
            df["VOLATILITY"] = np.nan

    return df


@tool
def get_priority_matrix() -> str:
    """
    Tabla cruzada Priorización × Tipo de Zona con % de zonas saludables en cada segmento.
    Responde: ¿las zonas prioritarias están peor que las no prioritarias?
    ¿El problema es más agudo en zonas Wealthy o Non Wealthy?
    """
    if df_merged.empty or L0 is None:
        return "Error: datos no disponibles."
    df = _prep()
    if "WOW" not in df.columns:
        return "Error: columnas WoW no disponibles."

    prio_order = ["High Priority", "Prioritized", "Not Prioritized"]
    type_order = ["Wealthy", "Non Wealthy"]

    rows_out = []
    for prio in prio_order:
        row = {"Priorización": prio}
        for ztype in type_order:
            seg = df[(df["ZONE_PRIORITIZATION"].str.contains(prio, case=False, na=False)) &
                     (df["ZONE_TYPE"].str.contains(ztype, case=False, na=False))]
            nz = seg["ZONE"].nunique()
            if nz == 0:
                row[ztype] = "—"
                row[f"{ztype}_n"] = 0
                continue
            anom = seg[seg["DETERI"] & (seg["ABS_WOW"] > 0.10)]["ZONE"].nunique()
            health = round((1 - anom / nz) * 100, 1)
            avg_wow = (seg["WOW"] * seg["POL"]).mean() * 100
            ico = "🔴" if health < 70 else ("🟡" if health < 85 else "🟢")
            row[ztype] = f"{ico} {health}% estables ({nz} zonas, Δ{avg_wow:+.1f}% WoW)"
            row[f"{ztype}_n"] = nz
        rows_out.append(row)

    table = "| Priorización | Wealthy | Non Wealthy |\n"
    table += "|:-------------|:--------|:------------|\n"
    for r in rows_out:
        table += f"| **{r['Priorización']}** | {r.get('Wealthy','—')} | {r.get('Non Wealthy','—')} |\n"

    # Key facts
    hp_rows = [r for r in rows_out if "High Priority" in r["Priorización"]]
    hp_worst = None
    hp_worst_val = 999
    for r in rows_out:
        for zt in type_order:
            cell = r.get(zt, "—")
            if "%" in str(cell):
                try:
                    val = float(str(cell).split("%")[0].split()[-1])
                    if val < hp_worst_val and "High Priority" in r["Priorización"]:
                        hp_worst_val = val
                        hp_worst = f"{r['Priorización']} / {zt}"
                except Exception:
                    pass

    total_hp = df[df["ZONE_PRIORITIZATION"].str.contains("High Priority", case=False, na=False)]["ZONE"].nunique()
    total_hp_anom = df[
        df["ZONE_PRIORITIZATION"].str.contains("High Priority", case=False, na=False) &
        df["DETERI"] & (df["ABS_WOW"] > 0.10)
    ]["ZONE"].nunique()

    facts = (
        f"- **Zonas High Priority en la red**: {total_hp} ({total_hp_anom} con anomalías críticas esta semana)\n"
        f"- **Segmento más crítico** (menor % estables): {hp_worst or 'N/A'} → solo {hp_worst_val:.0f}% de las zonas están estables\n"
        "- **Cómo leer:** 🔴 <70% de zonas estables · 🟡 70–85% · 🟢 >85%. Δ = cambio WoW promedio.\n"
    )

    return (
        "## 🗺️ Mapa Estratégico: Priorización × Tipo de Zona\n\n"
        "**HECHOS CLAVE**:\n"
        f"{facts}\n"
        f"{table}\n"
        "_Zonas Estables = % de zonas del segmento sin anomalías críticas (WoW >10% deterioro)._\n"
    )


@tool
def get_momentum_analysis() -> str:
    """
    Detecta zonas donde el RITMO de deterioro se está acelerando esta semana vs la anterior.
    Una zona con WoW de -5% que venía de -1% es más urgente que una con WoW de -8% que viene de -10%.
    Prioriza zonas High Priority / Prioritized.
    """
    if df_merged.empty or L0 is None or L2 is None:
        return "Error: datos insuficientes para momentum (se necesita L0, L1, L2)."
    df = _prep()
    if "ACCEL" not in df.columns or df["ACCEL"].isna().all():
        return "## ⚡ Momentum\n\nDatos insuficientes para calcular aceleración.\n"

    acc = df[df["DETERI"] & df["ACCEL"].notna()].copy()
    acc["ACCEL_SIGNED"] = acc["ACCEL"] * acc["POL"]
    acc = acc[acc["ACCEL_SIGNED"] < -0.01]

    if acc.empty:
        return "## ⚡ Momentum\n\nSin zonas con deterioro acelerando esta semana.\n"

    # Priority weight
    prio_weight = {"high priority": 3, "prioritized": 2, "not prioritized": 1}
    acc["PRIO_W"] = acc["ZONE_PRIORITIZATION"].str.lower().map(
        lambda x: next((v for k, v in prio_weight.items() if k in str(x)), 1)
    )
    acc = acc.sort_values(["PRIO_W", "ACCEL_SIGNED"], ascending=[False, True]).head(12)

    n_acc = len(acc); nz_acc = acc["ZONE"].nunique()
    hp_acc = acc[acc["PRIO_W"] >= 2]["ZONE"].nunique()
    worst = acc.iloc[0]

    facts = (
        f"- **Deterioros acelerando**: {n_acc} combinaciones zona/métrica en {nz_acc} zonas únicas\n"
        f"- **De esas, en zonas prioritarias**: {hp_acc} zonas (High Priority + Prioritized)\n"
        f"- **Caso más urgente**: {worst['ZONE']} ({worst['COUNTRY']}) — '{worst['METRIC']}' "
        f"pasó de WoW {worst['WOW_PREV']*100:+.1f}% a {worst['WOW']*100:+.1f}% → aceleración de "
        f"{worst['ACCEL_SIGNED']*100:.1f} pp\n"
    )

    table = "| País | Ciudad | Zona | Tipo | Priorización | Métrica | WoW L1→L0 | WoW L2→L1 | Aceleración |\n"
    table += "|------|--------|------|------|:------------:|---------|:---------:|:---------:|:-----------:|\n"
    for _, r in acc.iterrows():
        accel_str = f"⚠️ {r['ACCEL_SIGNED']*100:.1f} pp" if r["ACCEL_SIGNED"] < -0.05 else f"{r['ACCEL_SIGNED']*100:.1f} pp"
        table += (f"| {r['COUNTRY']} | {r['CITY']} | {r['ZONE']} "
                  f"| {r.get('ZONE_TYPE','—')} | {r.get('ZONE_PRIORITIZATION','—')} "
                  f"| {r['METRIC']} | {r['WOW']*100:+.1f}% | {r['WOW_PREV']*100:+.1f}% "
                  f"| {accel_str} |\n")

    return (
        "## ⚡ Momentum — Deterioros que se Están Acelerando\n\n"
        "**HECHOS CLAVE**:\n"
        f"{facts}\n"
        f"{table}\n"
        "_Aceleración = diferencia entre WoW actual y WoW de la semana anterior. "
        "Cuanto más negativo, más rápido empeora._\n"
    )



METRIC_BUSINESS_MEANING = {
    "% PRO Users Who Breakeven": "Usuarios con suscripción Pro cuyo valor generado para la empresa (a través de compras, comisiones, etc.) ha cubierto el costo total de su membresía / Total de usuarios suscripción Pro",
    "% Restaurants Sessions With Optimal Assortment": "Sesiones con un mínimo de 40 restaurantes/ Total de sesiones",
    "Gross Profit UE": "Margen bruto de ganancia / Total de órdenes",
    "Lead Penetration": "Tiendas habilitadas en Rappi / (Tiendas, previamente identificadas como prospectos (leads) + Tiendas habilitadas + tiendas salieron de Rappi )",
    "MLTV Top Verticals Adoption": "Usuarios con órdenes en diferentes verticales (restaurantes, super, pharmacy, liquors) / Total usuarios.",
    "Non-Pro PTC > OP": "Conversión de usuarios No Pro en \"Proceed to Checkout\" a \"Order Placed\"",
    "Perfect Orders": "Orders sin cancelaciones o defectos o demora / Total de órdenes",
    "Pro Adoption": "Usuarios suscripción Pro / Total usuarios de Rappi",
    "Restaurants Markdowns / GMV": "Descuentos totales en órdenes de restaurantes  / Total Gross Merchandise Value Restaurantes",
    "Restaurants SS > ATC CVR": "Conversión en restaurantes de \"Select Store\" a \"Add to Cart\"",
    "Restaurants SST > SS CVR": "Porcentaje de usuarios que, después de seleccionar un Restaurantes o \"Supermercados\"), proceden a seleccionar una tienda en particular de la lista que se les presenta.",
    "Retail SST > SS CVR": "Porcentaje de usuarios que, después de seleccionar un Supermercados, proceden a seleccionar una tienda en particular de la lista que se les presenta.",
    "Turbo Adoption": "Total de usuarios que compran en Turbo (Servicio fast de Rappi) / total de usuarios de Rappi con tiendas de turbo disponible",
}

@tool
def get_ecosystem_health() -> str:
    """
    Punto de partida del reporte. Muestra el estado general del negocio identificando cuáles métricas
    específicas están liderando el deterioro y cuáles muestran recuperación, con su definición
    de negocio. Sin agregados abstractos por categoría.
    """
    if df_merged.empty or L0 is None:
        return "Error: datos no disponibles."
    df = _prep()
    N = len(df)
    if N == 0:
        return "Error: sin registros."

    nc  = df["COUNTRY"].nunique(); nci = df["CITY"].nunique()
    nz  = df["ZONE"].nunique();    nm  = df["METRIC"].nunique()
    ord_total = df.drop_duplicates(["COUNTRY","CITY","ZONE"])["ORDERS_L0W"].sum()
    n_det = int(df["DETERI"].sum()); n_imp = int((~df["DETERI"] & (df["WOW"] > 0.01)).sum())

    metric_stats = (df.groupby("METRIC")
                    .apply(lambda g: pd.Series({
                        "zones_det": int(g["DETERI"].sum()),
                        "zones_imp": int((~g["DETERI"] & (g["WOW"] > 0.01)).sum()),
                        "avg_signed_wow": float((g["WOW"] * g["POL"]).mean()) * 100,
                        "polarity": int(g["POL"].iloc[0]),
                    }))
                    .reset_index())

    top_det = (metric_stats[metric_stats["zones_det"] > 0]
               .sort_values(["zones_det", "avg_signed_wow"])
               .head(6))
    top_imp = (metric_stats[metric_stats["zones_imp"] > 0]
               .sort_values("zones_imp", ascending=False)
               .head(6))

    coverage = (
        f"- **Cobertura**: {nc} países · {nci} ciudades · {nz} zonas\n"
        f"- **Métricas monitoreadas**: {nm} indicadores · {N:,} combinaciones zona/métrica analizadas\n"
        f"- **Órdenes totales L0W**: {int(ord_total):,}\n"
        f"- **Balance general**: {n_det:,} registros deteriorando ({n_det/N*100:.1f}%) vs "
        f"{n_imp:,} mejorando ({n_imp/N*100:.1f}%)\n"
    )

    det_rows = "| Métrica | Zonas Afectadas | Δ WoW Promedio | Definición |\n"
    det_rows += "|---------|:--------------:|:--------------:|------------|\n"
    for _, r in top_det.iterrows():
        meaning = METRIC_BUSINESS_MEANING.get(r["METRIC"], "Métrica del ecosistema")
        det_rows += (f"| {r['METRIC']} | {int(r['zones_det'])} | {r['avg_signed_wow']:+.2f}% "
                     f"| {meaning} |\n")

    imp_rows = "| Métrica | Zonas Mejorando | Δ WoW Promedio | Definición |\n"
    imp_rows += "|---------|:--------------:|:--------------:|------------|\n"
    for _, r in top_imp.iterrows():
        meaning = METRIC_BUSINESS_MEANING.get(r["METRIC"], "Métrica del ecosistema")
        imp_rows += (f"| {r['METRIC']} | {int(r['zones_imp'])} | {r['avg_signed_wow']:+.2f}% "
                     f"| {meaning} |\n")

    return (
        "## 🌡️ Estado del Ecosistema Rappi — Semana Actual\n\n"
        "**CONTEXTO GEOGRÁFICO Y DE VOLUMEN** *(cita estos datos en la intro)*:\n"
        f"{coverage}\n"
        "**⚠️ Métricas Liderando el Deterioro** *(usa los NOMBRES de estas métricas en la narrativa, "
        "NO nombres de categorías. Usa la columna 'Qué significa' para explicar el impacto)*:\n"
        f"{det_rows}\n"
        "**✅ Métricas con Mayor Recuperación** *(igualmente, usa nombres de métricas y el impacto)*:\n"
        f"{imp_rows}\n"
    )


@tool
def get_country_summary() -> str:
    """
    Agrega el desempeño al nivel de país: zonas con anomalías, zonas en tendencia negativa,
    métrica crítica por país, volumen de órdenes y score de estabilidad.
    Incluye hechos clave para que la narrativa cite números reales.
    """
    if df_merged.empty or L0 is None or L1 is None:
        return "Error: datos no disponibles."
    df = _prep()
    if "WOW" not in df.columns:
        return "Error: columnas WoW no disponibles."

    rows = []
    for country, g in df.groupby("COUNTRY"):
        tz = g["ZONE"].nunique()
        az = g[g["DETERI"] & (g["ABS_WOW"] > 0.10)]["ZONE"].nunique()
        if L2 and L3 and L2 in df.columns and L3 in df.columns:
            cond = (
                ((g["POL"]==1) & (g[L0]<g[L1]) & (g[L1]<g[L2]) & (g[L2]<g[L3])) |
                ((g["POL"]==-1) & (g[L0]>g[L1]) & (g[L1]>g[L2]) & (g[L2]>g[L3]))
            )
            tz3 = g[cond]["ZONE"].nunique()
        else:
            tz3 = 0
        orders = g.drop_duplicates(["ZONE"])["ORDERS_L0W"].sum()
        wm_ser = g[g["DETERI"]].groupby("METRIC").size()
        wm = wm_ser.idxmax() if not wm_ser.empty else "—"
        estabil = round((1 - az / max(tz, 1)) * 100, 1)
        ico = "🔴" if estabil < 70 else ("🟡" if estabil < 85 else "🟢")
        rows.append((ico, country, tz, az, tz3, wm, f"{int(orders):,}", f"{estabil}%"))

    rows.sort(key=lambda x: float(x[7].rstrip('%')))

    worst_c = rows[0][1]; worst_h = rows[0][7]
    best_c  = rows[-1][1]; best_h  = rows[-1][7]
    total_anom = sum(r[3] for r in rows)

    facts = (
        f"- **País más crítico**: {worst_c} ({worst_h} de zonas estables)\n"
        f"- **País más estable**: {best_c} ({best_h})\n"
        f"- **Total de zonas con anomalías críticas** en toda la red: {total_anom}\n"
    )

    table = "| | País | Zonas | Con Anomalías | Tendencia ↓ 3W | Métrica Crítica | Órdenes L0W | Estables % |\n"
    table += "|-|------|:-----:|:-------------:|:--------------:|-----------------|:-----------:|:----------:|\n"
    for r in rows:
        table += f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} | {r[6]} | {r[7]} |\n"

    return (
        "## 🌎 Desempeño por País\n\n"
        "**HECHOS CLAVE**:\n"
        f"{facts}\n"
        f"{table}\n"
        "_Ordenado de menor a mayor % de zonas estables._\n"
    )


@tool
def get_city_level_analysis() -> str:
    """
    Agrega el desempeño al nivel de ciudad. Identifica las 5 ciudades con más problemas
    y las 5 con mejor desempeño, usando métricas ponderadas por volumen de órdenes.
    """
    if df_merged.empty or L0 is None or L1 is None:
        return "Error: datos no disponibles."
    df = _prep()
    if "WOW" not in df.columns:
        return "Error: columnas WoW no disponibles."

    city_rows = []
    for (country, city), g in df.groupby(["COUNTRY","CITY"]):
        nz   = g["ZONE"].nunique()
        anom = g[g["DETERI"] & (g["ABS_WOW"] > 0.10)]["ZONE"].nunique()
        avg_signed = (g["WOW"] * g["POL"]).mean() * 100
        orders = g.drop_duplicates("ZONE")["ORDERS_L0W"].sum()
        estabil = round((1 - anom/max(nz,1))*100, 1)
        wm_ser = g[g["DETERI"]].groupby("METRIC").size()
        wm = wm_ser.idxmax() if not wm_ser.empty else "—"
        city_rows.append({
            "País": country, "Ciudad": city, "Zonas": nz,
            "Con Anomalías": anom, "Δ WoW Avg": avg_signed,
            "Órdenes": orders, "Estables%": estabil, "Métrica Débil": wm
        })

    df_c = pd.DataFrame(city_rows)
    if df_c.empty:
        return "Sin datos de ciudades."

    bottom_df = df_c.nsmallest(8, "Estables%")
    top_df    = df_c.nlargest(6,  "Estables%")

    def _tbl(sub, title, ico):
        t  = f"\n### {ico} {title}\n"
        t += "| País | Ciudad | Zonas | Anomalías | Órdenes L0W | Δ WoW Promedio | Estables% | Métrica Débil |\n"
        t += "|------|--------|:-----:|:---------:|:-----------:|:--------------:|:---------:|---------------|\n"
        for _, r in sub.iterrows():
            t += (f"| {r['País']} | {r['Ciudad']} | {r['Zonas']} | {r['Con Anomalías']} "
                  f"| {int(r['Órdenes']):,} | {r['Δ WoW Avg']:+.2f}% | {r['Estables%']}% | {r['Métrica Débil']} |\n")
        return t

    worst_city = bottom_df.iloc[0]['Ciudad']; worst_h = bottom_df.iloc[0]['Estables%']
    best_city  = top_df.iloc[0]['Ciudad'];   best_h  = top_df.iloc[0]['Estables%']
    facts = (
        f"- **Ciudad más crítica**: {worst_city} (solo {worst_h}% de zonas estables)\n"
        f"- **Ciudad más estable**: {best_city} ({best_h}% de zonas estables)\n"
        f"- **Total ciudades analizadas**: {len(df_c)}\n"
    )

    return (
        "## 🏙️ Desempeño por Ciudad\n\n"
        "**HECHOS CLAVE**:\n"
        f"{facts}\n"
        + _tbl(bottom_df, "Ciudades con Mayor Riesgo", "🔴")
        + _tbl(top_df,    "Ciudades con Mejor Desempeño", "🟢")
        + "\n_Estables% = % de zonas sin anomalías críticas._\n"
    )



@tool
def get_critical_anomalies(threshold: float = 0.10, top_n: int = 15) -> str:
    """
    Detecta zonas con cambios drásticos WoW > umbral.
    EDGE CASE: Aplica Volume Threshold (P25 de la ciudad) para evitar falsos positivos
    por ley de números pequeños (ej. zona con 2→4 órdenes = +100% pero irrelevante).
    """
    if df_merged.empty or L0 is None or L1 is None:
        return "Error: datos no disponibles."
    df = _prep()
    if "WOW" not in df.columns:
        return "Error."

    city_p25 = df.groupby("CITY")["ORDERS_L1W"].transform(lambda x: x.quantile(0.25)) if "ORDERS_L1W" in df.columns else df.groupby("CITY")["ORDERS_L0W"].transform(lambda x: x.quantile(0.25))
    df = df[df["ORDERS_L1W"] >= city_p25] if "ORDERS_L1W" in df.columns else df[df["ORDERS_L0W"] >= city_p25]

    anom = df[df["ABS_WOW"] > threshold].copy()
    det  = anom[anom["DETERI"]].sort_values(["ORDERS_L0W","ABS_WOW"], ascending=[False,False]).head(top_n)
    mej  = anom[~anom["DETERI"]].sort_values(["ORDERS_L0W","ABS_WOW"], ascending=[False,False]).head(10)

    n_anom = len(anom); n_zones = anom["ZONE"].nunique()
    worst_z = det.iloc[0]["ZONE"] if not det.empty else "N/A"
    worst_m = det.iloc[0]["METRIC"] if not det.empty else "N/A"
    worst_c = det.iloc[0]["ABS_WOW"]*100 if not det.empty else 0
    worst_country = det.iloc[0]["COUNTRY"] if not det.empty else "N/A"

    top_met = det.groupby("METRIC").size().idxmax() if not det.empty else "N/A"
    top_met_n = det.groupby("METRIC").size().max() if not det.empty else 0

    facts = (
        f"- **Total anomalías detectadas** (WoW > {threshold*100:.0f}%, filtradas por volumen P25): {n_anom:,} en {n_zones} zonas\n"
        f"- **Deterioros críticos**: {len(det)} · **Mejoras destacadas**: {len(mej)}\n"
        f"- **Peor deterioro individual**: {worst_z} ({worst_country}) — '{worst_m}' cayó {worst_c:.1f}% en 1 semana\n"
        f"- **Métrica más afectada**: '{top_met}' aparece en {top_met_n} zonas con deterioro crítico\n"
    )

    def _tbl(src, ico):
        if src.empty:
            return "Sin registros.\n"
        t = "| País | Ciudad | Zona | Priorización | Métrica | L1W | L0W | Δ% |\n"
        t += "|------|--------|------|:------------:|---------|:---:|:---:|:--:|\n"
        for _, r in src.iterrows():
            sgn = "🔴" if r["DETERI"] else "🟢"
            t += (f"| {r['COUNTRY']} | {r['CITY']} | {r['ZONE']} "
                  f"| {r.get('ZONE_PRIORITIZATION','—')} | {r['METRIC']} "
                  f"| {_n(r[L1])} | {_n(r[L0])} | {sgn} {r['ABS_WOW']*100:.1f}% |\n")
        return t

    return (
        f"## 🚨 Anomalías Críticas (Cambio WoW > {threshold*100:.0f}%)\n\n"
        "**HECHOS CLAVE**:\n"
        f"{facts}\n"
        "### 🔴 Deterioros Críticos\n"
        f"{_tbl(det, '🔴')}\n"
        "### 🟢 Mejoras Destacadas\n"
        f"{_tbl(mej, '🟢')}\n"
    )


@tool
def get_worrisome_trends(top_n: int = 15) -> str:
    """
    Identifica zonas con deterioro consecutivo en 3+ semanas.
    EDGE CASE: Solo reporta si delta acumulado (L0W vs L3W) > 1 desviación estándar
    histórica de esa métrica. Evita reportar fluctuaciones mínimas como deterioro.
    """
    if df_merged.empty or not all([L0, L1, L2, L3]):
        return "Error: datos insuficientes para tendencias."
    df = _prep()
    for c in [L2, L3]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[L0, L1, L2, L3])

    cond_pos = (df["POL"]==1) & (df[L0]<df[L1]) & (df[L1]<df[L2]) & (df[L2]<df[L3])
    cond_neg = (df["POL"]==-1) & (df[L0]>df[L1]) & (df[L1]>df[L2]) & (df[L2]>df[L3])
    tr = df[cond_pos | cond_neg].copy()
    tr["TOTAL_DETERI"] = ((tr[L0] - tr[L3]) / tr[L3].abs()).abs()

    week_cols_here = [c for c in ["L8W","L7W","L6W","L5W","L4W","L3W","L2W","L1W","L0W"] if c in tr.columns]
    tr["METRIC_STD"] = tr[week_cols_here].std(axis=1) if week_cols_here else 0
    tr["ABS_DELTA"] = (tr[L0] - tr[L3]).abs()
    tr = tr[tr["ABS_DELTA"] > tr["METRIC_STD"].fillna(0)]

    # 4-week check
    if L4 and L4 in df.columns:
        df[L4] = pd.to_numeric(df[L4], errors="coerce")
        c4p = cond_pos & df[L4].notna() & (df[L3] < df[L4])
        c4n = cond_neg & df[L4].notna() & (df[L3] > df[L4])
        n4w = int((c4p | c4n).sum())
    else:
        n4w = 0

    n_hp_trend = tr[tr["ZONE_PRIORITIZATION"].str.contains("High Priority", case=False, na=False)]["ZONE"].nunique()

    tr = tr.sort_values(["ORDERS_L0W", "TOTAL_DETERI"], ascending=[False, False]).head(top_n)

    n_tr = len(tr); nz_tr = tr["ZONE"].nunique() if not tr.empty else 0
    worst_tr = tr.iloc[0] if not tr.empty else None
    worst_tz = worst_tr["ZONE"] if worst_tr is not None else "N/A"
    worst_tm = worst_tr["METRIC"] if worst_tr is not None else "N/A"
    worst_td = worst_tr["TOTAL_DETERI"]*100 if worst_tr is not None else 0
    worst_streak = int(worst_tr.get("STREAK", 0)) if worst_tr is not None else 0

    facts = (
        f"- **Tendencias significativas** (3+W consecutivas, delta > 1σ): {n_tr} en {nz_tr} zonas\n"
        f"- **Zonas High Priority con tendencia estructural**: {n_hp_trend}\n"
        f"- **Deterioro 4+ semanas consecutivas**: {n4w} registros\n"
        f"- **Caso mas grave**: {worst_tz} — '{worst_tm}' acumula {worst_td:.1f}% desde L3W "
        f"(streak: {worst_streak} semanas consecutivas)\n"
    )

    table = "| Pais | Ciudad | Zona | Tipo | Priorizacion | Metrica | L3W | L2W | L1W | L0W | Deterioro | Streak |\n"
    table += "|------|--------|------|------|:------------:|---------|:---:|:---:|:---:|:---:|:---------:|:------:|\n"
    for _, r in tr.iterrows():
        streak = int(r.get("STREAK", 0))
        streak_str = f"🔥 {streak}W" if streak >= 5 else f"{streak}W"
        table += (f"| {r['COUNTRY']} | {r['CITY']} | {r['ZONE']} "
                  f"| {r.get('ZONE_TYPE', '—')} | {r.get('ZONE_PRIORITIZATION', '—')} "
                  f"| {r['METRIC']} "
                  f"| {_n(r[L3])} | {_n(r[L2])} | {_n(r[L1])} | {_n(r[L0])} "
                  f"| 🔴 {r['TOTAL_DETERI']*100:.1f}% | {streak_str} |\n")

    return (
        "## 📉 Tendencias Preocupantes (3+ Semanas, Estadísticamente Significativas)\n\n"
        "**HECHOS CLAVE**:\n"
        f"{facts}\n"
        f"{table}\n"
        "_Filtro: solo tendencias cuyo delta acumulado supera 1σ de la métrica. "
        "Streak = semanas consecutivas. 🔥 = 5+W._\n"
    )

@tool
def get_multivariate_risk_zones() -> str:
    """
    Análisis multivariado: identifica zonas con MÚLTIPLES métricas simultáneamente
    en el cuartil inferior de su país. Agrupa por clúster de riesgo de negocio
    (Experiencia, Monetización, Expansión, Conversión). Estas son las zonas de mayor riesgo sistémico.
    """
    if df_merged.empty or L0 is None:
        return "Error: datos no disponibles."
    df = _prep()

    df["Q25"] = df.groupby(["COUNTRY","METRIC"])[L0].transform(lambda x: x.quantile(0.25))
    df["Q75"] = df.groupby(["COUNTRY","METRIC"])[L0].transform(lambda x: x.quantile(0.75))
    df["IN_DANGER"] = (
        ((df["POL"]==1)  & (df[L0] <= df["Q25"])) |   
        ((df["POL"]==-1) & (df[L0] >= df["Q75"]))     
    )

    all_risk_rows = []
    for cluster_name, metrics_list, cluster_desc in RISK_COMBOS:
        sub = df[df["METRIC"].isin(metrics_list)]
        zone_danger_count = (sub[sub["IN_DANGER"]]
                             .groupby(["COUNTRY","CITY","ZONE"])["METRIC"]
                             .apply(lambda x: list(x.unique())).reset_index())
        zone_danger_count.columns = ["COUNTRY","CITY","ZONE","METRICAS_EN_RIESGO"]
        zone_danger_count["N_RIESGO"] = zone_danger_count["METRICAS_EN_RIESGO"].apply(len)
        zone_danger_count = zone_danger_count[zone_danger_count["N_RIESGO"] >= 2]

        if zone_danger_count.empty:
            continue

        ord_df = df.drop_duplicates(["COUNTRY","CITY","ZONE"])[["COUNTRY","CITY","ZONE","ORDERS_L0W"]]
        zone_danger_count = zone_danger_count.merge(ord_df, on=["COUNTRY","CITY","ZONE"], how="left")
        zone_danger_count = zone_danger_count.sort_values(["N_RIESGO","ORDERS_L0W"], ascending=[False,False]).head(5)

        for _, r in zone_danger_count.iterrows():
            all_risk_rows.append({
                "Clúster": cluster_name,
                "País": r["COUNTRY"], "Ciudad": r["CITY"], "Zona": r["ZONE"],
                "Métricas en Riesgo": ", ".join(r["METRICAS_EN_RIESGO"]),
                "# Dimensiones": r["N_RIESGO"],
                "Órdenes L0W": int(r.get("ORDERS_L0W",0)),
                "_desc": cluster_desc,
            })

    if not all_risk_rows:
        return "## 🔺 Análisis Multivariado\n\nNo se detectaron zonas con riesgo compuesto significativo.\n"

    df_risk = pd.DataFrame(all_risk_rows)
    n_zones_risk = df_risk["Zona"].nunique()
    worst_z = df_risk.loc[df_risk["# Dimensiones"].idxmax()]

    facts = (
        f"- **Zonas con riesgo compuesto** (2+ métricas en cuartil peligroso): {n_zones_risk}\n"
        f"- **Zona de mayor riesgo sistémico**: {worst_z['Zona']} ({worst_z['País']}) "
        f"— {worst_z['# Dimensiones']} métricas comprometidas simultáneamente\n"
        f"- **Clústers de riesgo detectados**: {df_risk['Clúster'].nunique()}\n"
    )

    table = "| Clúster de Riesgo | País | Ciudad | Zona | Métricas Comprometidas | # |\n"
    table += "|-------------------|------|--------|------|------------------------|:-:|\n"
    prev_cluster = None
    for _, r in df_risk.iterrows():
        cluster_label = r["Clúster"] if r["Clúster"] != prev_cluster else "↳"
        prev_cluster  = r["Clúster"]
        table += (f"| {cluster_label} | {r['País']} | {r['Ciudad']} | {r['Zona']} "
                  f"| {r['Métricas en Riesgo']} | {r['# Dimensiones']} |\n")

    cluster_defs = "\n".join(f"- **{n}**: {d}" for n,ml,d in RISK_COMBOS)

    return (
        "## 🔺 Análisis Multivariado — Zonas de Riesgo Sistémico\n\n"
        "**HECHOS CLAVE**:\n"
        f"{facts}\n"
        "**Definición de clústers de riesgo:**\n"
        f"{cluster_defs}\n\n"
        f"{table}\n"
    )


@tool
def get_correlations_insights() -> str:
    """
    Correlaciones contemporáneas Y con rezago temporal (1-2 semanas).
    Time-lagged: un bajo assortment en L3W puede impactar Orders en L0W.
    Calcula Pearson por zona a lo largo de las semanas disponibles.
    """
    if df_metrics.empty or L0 is None:
        return "Error: datos no disponibles."

    vc = L0 if L0 in df_metrics.columns else ("L0W_ROLL" if "L0W_ROLL" in df_metrics.columns else "L0W")

    week_cols = [c for c in ["L0W","L1W","L2W","L3W","L4W","L5W","L6W","L7W",
                              "L0W_ROLL","L1W_ROLL","L2W_ROLL","L3W_ROLL","L4W_ROLL",
                              "L5W_ROLL","L6W_ROLL","L7W_ROLL"]
                 if c in df_metrics.columns]
    
    seen_w = set()
    clean_weeks = []
    for c in week_cols:
        base = c.split("_")[0]
        if base not in seen_w:
            seen_w.add(base)
            clean_weeks.append(c)
            
    melted = df_metrics.melt(id_vars=["COUNTRY","CITY","ZONE","METRIC"], 
                             value_vars=clean_weeks,
                             var_name="WEEK", value_name="VALUE")
    melted["VALUE"] = pd.to_numeric(melted["VALUE"], errors="coerce")
    melted["T"] = melted["WEEK"].str.extract(r'L(\d+)W').astype(int)
    
    pivot_tz = melted.pivot_table(index=["COUNTRY","CITY","ZONE","T"], 
                                  columns="METRIC", values="VALUE").reset_index()

    corr = pivot_tz.drop(columns=["COUNTRY","CITY","ZONE","T"]).corr()
    corr.index.name   = "MetA"
    corr.columns.name = "MetB"

    pairs = corr.unstack().reset_index()
    pairs.columns = ["Métrica A","Métrica B","Correlación"]
    pairs = pairs[pairs["Métrica A"] < pairs["Métrica B"]].dropna()
    pairs["Abs"] = pairs["Correlación"].abs()
    strong = pairs[pairs["Abs"] > 0.45].sort_values("Abs", ascending=False).head(8)

    INTERP = {
        frozenset(["Lead Penetration","Non-Pro PTC > OP"]):
            "Más tiendas habilitadas → mayor conversión de checkout para usuarios No Pro.",
        frozenset(["Pro Adoption","MLTV Top Verticals Adoption"]):
            "Usuarios Pro adoptan más verticales, elevando el LTV y reduciendo el churn.",
        frozenset(["Perfect Orders","Non-Pro PTC > OP"]):
            "Mejor calidad operativa correlaciona con mayor intención de compra.",
        frozenset(["Gross Profit UE","Perfect Orders"]):
            "Operaciones sin defectos generan mayor eficiencia y margen por orden.",
        frozenset(["Turbo Adoption","Restaurants SS > ATC CVR"]):
            "Zonas con adopción Turbo alta muestran mayor conversión en restaurantes.",
        frozenset(["Restaurants Markdowns / GMV","Gross Profit UE"]):
            "Alto descuento relativo erosiona el margen: sacrificar GMV no recupera rentabilidad.",
        frozenset(["Retail SST > SS CVR","Restaurants SST > SS CVR"]):
            "UX competitiva en una vertical se traslada a la otra.",
        frozenset(["% PRO Users Who Breakeven","Pro Adoption"]):
            "Más usuarios Pro = más que cubren el costo → modelo sostenible.",
        frozenset(["% Restaurants Sessions With Optimal Assortment","Restaurants SS > ATC CVR"]):
            "Mejor surtido → más conversión en tienda.",
    }

    LAG_PAIRS = [
        ("% Restaurants Sessions With Optimal Assortment", "Restaurants SS > ATC CVR", "Assortment bajo en L2W impacta conversión en L0W"),
        ("% Restaurants Sessions With Optimal Assortment", "Non-Pro PTC > OP", "Bajo surtido rezaga la conversión final del funnel"),
        ("Perfect Orders", "Pro Adoption", "Mala calidad operativa rezaga la adopción Pro"),
        ("Lead Penetration", "Gross Profit UE", "Menos tiendas habilitadas rezaga el margen por orden"),
        ("Restaurants Markdowns / GMV", "Gross Profit UE", "Descuentos excesivos rezagan la erosión de margen"),
        ("% Restaurants Sessions With Optimal Assortment", "Orders", "Bajo assortment rezaga caída en volumen de órdenes"),
    ]

    lag_results = []
    pivot_tz_sorted = pivot_tz.sort_values(["COUNTRY","CITY","ZONE","T"])

    for met_lead, met_lag, description in LAG_PAIRS:
        if met_lead not in pivot_tz_sorted.columns or met_lag not in pivot_tz_sorted.columns: 
            continue
            
        for lag in [1, 2]:
            lead_shifted = pivot_tz_sorted.groupby(["COUNTRY","CITY","ZONE"])[met_lead].shift(-lag)
            r_val = lead_shifted.corr(pivot_tz_sorted[met_lag])
            if abs(r_val) > 0.35:
                lag_results.append({
                    "Leading": met_lead, "Lagging": met_lag,
                    "Lag": f"{lag}W", "r": r_val,
                    "Interpretación": description
                })
                break

    n_str = len(strong)
    top_pair = (f"{strong.iloc[0]['Métrica A']} ↔ {strong.iloc[0]['Métrica B']}"
                f" (r={strong.iloc[0]['Correlación']:.2f})") if not strong.empty else "N/A"

    facts = (
        f"- **Correlaciones contemporáneas fuertes** (|r|>0.45): {n_str} pares\n"
        f"- **Correlación más intensa**: {top_pair}\n"
        f"- **Correlaciones con rezago temporal**: {len(lag_results)} pares detectados (early warnings)\n"
    )

    table = "| Métrica A | Métrica B | Correlación | Dirección | Implicación de Negocio |\n"
    table += "|-----------|-----------|:-----------:|:---------:|------------------------|\n"
    for _, r in strong.iterrows():
        tipo = "📈 Directa" if r["Correlación"] > 0 else "📉 Inversa"
        interp = INTERP.get(frozenset([r["Métrica A"],r["Métrica B"]]),
                            "Relación estadística identificada en los datos.")
        table += f"| {r['Métrica A']} | {r['Métrica B']} | {r['Correlación']:.2f} | {tipo} | {interp} |\n"

    lag_table = ""
    if lag_results:
        lag_table = (
            "\n### ⏱️ Early Warnings — Correlaciones con Rezago Temporal\n"
            "| Indicador Líder | Indicador Rezagado | Lag | r | Interpretación |\n"
            "|-----------------|--------------------:|:---:|:---:|----------------|\n"
        )
        for lr in lag_results:
            lag_table += f"| {lr['Leading']} | {lr['Lagging']} | {lr['Lag']} | {lr['r']:.2f} | {lr['Interpretación']} |\n"

    return (
        "## 🔗 Correlaciones y Patrones Sistémicos\n\n"
        "**HECHOS CLAVE**:\n"
        f"{facts}\n"
        f"{table}\n"
        f"{lag_table}\n"
        "_Contemporáneas: zona × métrica usando L0W. "
        "Rezagadas: leading indicator LnW vs lagging indicator L0W._\n"
    )


@tool
def get_benchmarking_insights() -> str:
    """
    Benchmarking con Z-score > 1.5σ. Agrupa zonas por COUNTRY + CITY + ZONE_TYPE
    + ZONE_PRIORITIZATION. Solo marca zonas a > 1.5 desviaciones estándar de su grupo.
    EDGE CASE resuelto: nunca compara Wealthy con Non Wealthy ni zonas con priorización distinta.
    """
    if df_merged.empty or L0 is None:
        return "Error: datos no disponibles."
    df = _prep()

    GROUP_COLS = ["COUNTRY", "CITY", "ZONE_TYPE", "ZONE_PRIORITIZATION"]
    for c in GROUP_COLS:
        if c not in df.columns:
            return f"Error: columna '{c}' no encontrada."

    grp = df.groupby(GROUP_COLS + ["METRIC"])[L0]
    df["GRP_MEAN"] = grp.transform("mean")
    df["GRP_STD"]  = grp.transform("std")
    df["GRP_N"]    = grp.transform("count")
    df["Z_SCORE"]  = ((df[L0] - df["GRP_MEAN"]) / df["GRP_STD"].replace(0, np.nan))
    df["Z_SIGNED"] = df["Z_SCORE"] * df["POL"]

    outliers = df[(df["Z_SIGNED"] < -1.5) & (df["GRP_N"] >= 3)].copy()
    outliers["Z_ABS"] = outliers["Z_SIGNED"].abs()
    outliers = outliers.sort_values(["ORDERS_L0W", "Z_ABS"], ascending=[False, False]).head(20)

    if outliers.empty:
        return "## ⚖️ Benchmarking Divergente\n\nSin zonas significativamente divergentes de su grupo.\n"

    n_out = len(outliers); nz_out = outliers["ZONE"].nunique()
    worst = outliers.iloc[0]

    facts = (
        f"- **Zonas divergentes** (Z-score < -1.5σ vs su grupo): {n_out} en {nz_out} zonas\n"
        f"- **Agrupación**: COUNTRY + CITY + ZONE_TYPE + ZONE_PRIORITIZATION (comparación justa)\n"
        f"- **Contexto Narrativo Obligatorio**: NUNCA menciones un dato aislado. Explica la brecha citando el valor de la zona vs la columna 'Media Grupo'. Ejemplo: 'cayó a {worst['Z_SIGNED']:.1f}σ, reportando {_n(worst[L0])} frente a una media esperada de {_n(worst['GRP_MEAN'])}'.\n"
    )

    table = "| País | Ciudad | Zona | Tipo | Prio | Métrica | Valor | Media Grupo | Z-score |\n"
    table += "|------|--------|------|------|:----:|---------|:-----:|:-----------:|:-------:|\n"
    for _, r in outliers.iterrows():
        z_str = f"🔴 {r['Z_SIGNED']:.1f}σ" if r["Z_SIGNED"] < -2.0 else f"{r['Z_SIGNED']:.1f}σ"
        table += (f"| {r['COUNTRY']} | {r['CITY']} | {r['ZONE']} "
                  f"| {r.get('ZONE_TYPE', '—')} | {r.get('ZONE_PRIORITIZATION', '—')} "
                  f"| {r['METRIC']} "
                  f"| {_n(r[L0])} | {_n(r['GRP_MEAN'])} | {z_str} |\n")

    return (
        "## ⚖️ Benchmarking Divergente — Zonas Fuera de Rango vs Pares\n\n"
        "**HECHOS CLAVE**:\n"
        f"{facts}\n"
        f"{table}\n"
        "_Cluster: mismo País × Ciudad × Tipo × Priorización. "
        "Solo zonas con ≥3 pares. Z < -1.5σ = significativamente peor que su grupo._\n"
    )


@tool
def get_opportunities_insights() -> str:
    """
    Zonas con priorizacion 'High Priority' o 'Prioritized' que estan por debajo
    del percentil 40 en su segmento (mismo pais + mismo ZONE_TYPE).
    La brecha se calcula vs la mediana de zonas comparables (no nacional), lo que
    hace el benchmark mas justo y la oportunidad mas accionable.
    """
    if df_merged.empty or L0 is None:
        return "Error: datos no disponibles."
    df = _prep()

    df["PTILE"] = df.groupby(["COUNTRY", "METRIC"])[L0].transform(lambda x: x.rank(pct=True) * 100)

    seg_medians = (
        df.groupby(["COUNTRY", "ZONE_TYPE", "METRIC"])[L0]
        .median()
        .rename("SEG_MEDIAN")
        .reset_index()
    )
    df = df.merge(seg_medians, on=["COUNTRY", "ZONE_TYPE", "METRIC"], how="left")

    nat_medians = df.groupby(["COUNTRY", "METRIC"])[L0].median().rename("NAT_MEDIAN").reset_index()
    df = df.merge(nat_medians, on=["COUNTRY", "METRIC"], how="left")

    pri = df["ZONE_PRIORITIZATION"].str.contains("High Priority|Prioritized", case=False, na=False)
    sub = df[pri].copy()
    if sub.empty:
        sub = df.copy()

    under = ((sub["POL"] == 1) & (sub["PTILE"] < 40))
    over  = ((sub["POL"] == -1) & (sub["PTILE"] > 60))
    opp = sub[under | over].copy()

    opp["SEG_GAP"] = (opp["SEG_MEDIAN"] - opp[L0]) * opp["POL"]
    opp["NAT_GAP"] = (opp["NAT_MEDIAN"] - opp[L0]) * opp["POL"]
    opp = opp.sort_values(["ORDERS_L0W", "SEG_GAP"], ascending=[False, False]).head(20)

    if opp.empty:
        return "## 🚀 Oportunidades\n\nSin oportunidades criticas detectadas.\n"

    n_opp = len(opp); nz_opp = opp["ZONE"].nunique()
    top_opp = opp.iloc[0]
    hp_opp = opp[opp["ZONE_PRIORITIZATION"].str.contains("High Priority", case=False, na=False)]["ZONE"].nunique()

    facts = (
        f"- **Oportunidades detectadas** (zonas prioritarias bajo P40 en su pais): {n_opp} en {nz_opp} zonas\n"
        f"- **De esas, High Priority**: {hp_opp} zonas con mayor urgencia estrategica\n"
        f"- **Contexto Narrativo Obligatorio**: Explica la oportunidad usando la 'Brecha Segmento' para que el negocio sepa cuánto debe mejorar (ej. para llegar a {_n(top_opp['SEG_MEDIAN'])} en zonas similares).\n"
    )

    table = "| Pais | Ciudad | Zona | Tipo | Priorizacion | Metrica | Valor Actual | Mediana Segmento | Brecha Segmento | Percentil |\n"
    table += "|------|--------|------|------|:------------:|---------|:------------:|:----------------:|:---------------:|:---------:|\n"
    for _, r in opp.iterrows():
        table += (f"| {r['COUNTRY']} | {r['CITY']} | {r['ZONE']} "
                  f"| {r.get('ZONE_TYPE', '—')} | {r.get('ZONE_PRIORITIZATION', '—')} "
                  f"| {r['METRIC']} "
                  f"| {_n(r[L0])} | {_n(r['SEG_MEDIAN'])} | {_n(r['SEG_GAP'])} | P{r['PTILE']:.0f} |\n")

    return (
        "## 🚀 Oportunidades de Alto Impacto Estrategico\n\n"
        "**HECHOS CLAVE**:\n"
        f"{facts}\n"
        f"{table}\n"
        "_Mediana Segmento = mediana de zonas del MISMO pais y tipo (Wealthy/Non Wealthy). "
        "Benchmark mas justo que la mediana nacional._\n"
    )


@tool
def get_investment_efficiency_insights() -> str:
    """
    A. Eficiencia de Inversión (Burn ROI vs. Growth)
    Cruza inversión promocional con crecimiento real y rentabilidad.
    Detecta zonas donde Markdowns suben >15%, Orders se estancan/caen, y Gross Profit UE cae a negativo.
    """
    if 'df_orders' not in globals() or df_merged.empty or not all([L0, L3]):
        return "Error: datos de órdenes o semanas históricas no disponibles."
        
    df = _prep()
    m_df = df[df["METRIC"].isin(["Restaurants Markdowns / GMV", "Gross Profit UE"])].copy()
    for col in [L0, L3]: m_df[col] = pd.to_numeric(m_df[col], errors="coerce")
    
    pivot = m_df.pivot_table(index=["COUNTRY","CITY","ZONE","ZONE_PRIORITIZATION"], columns="METRIC", values=[L0, L3]).reset_index()
    pivot.columns = [f"{c[1]}_{c[0]}" if c[1] else c[0] for c in pivot.columns]
    
    req = [f"Restaurants Markdowns / GMV_{L0}", f"Restaurants Markdowns / GMV_{L3}", f"Gross Profit UE_{L0}"]
    if not all(r in pivot.columns for r in req):
        return "## 💸 Eficiencia de Inversión\n\nFaltan métricas de rentabilidad."
        
    ord_L0 = L0.replace("_ROLL", "")
    ord_L3 = L3.replace("_ROLL", "")
    ord_df = df_orders[df_orders["METRIC"] == "Orders"][["COUNTRY","CITY","ZONE", ord_L0, ord_L3]]
    ord_df = ord_df.rename(columns={ord_L0: f"Orders_{L0}", ord_L3: f"Orders_{L3}"})
    for col in [f"Orders_{L0}", f"Orders_{L3}"]: ord_df[col] = pd.to_numeric(ord_df[col], errors="coerce")
    
    merged = pivot.merge(ord_df, on=["COUNTRY","CITY","ZONE"], how="inner")
    
    merged["Mkdn_Delta"] = merged[f"Restaurants Markdowns / GMV_{L0}"] - merged[f"Restaurants Markdowns / GMV_{L3}"]
    merged["Mkdn_Growth"] = merged["Mkdn_Delta"] / merged[f"Restaurants Markdowns / GMV_{L3}"].replace(0, np.nan).abs()
    merged["Ord_Growth"] = (merged[f"Orders_{L0}"] - merged[f"Orders_{L3}"]) / merged[f"Orders_{L3}"].replace(0, np.nan)
    merged["GP_Negative"] = merged[f"Gross Profit UE_{L0}"] < 0
    
    alarms = merged[(merged["Mkdn_Growth"] > 0.15) & (merged["Ord_Growth"] <= 0.02) & merged["GP_Negative"]].copy()
    alarms = alarms.sort_values(f"Orders_{L0}", ascending=False).head(15)
    
    if alarms.empty:
        return "## 💸 Eficiencia de Inversión (Burn ROI vs. Growth)\n\nNo se detectaron zonas quemando dinero sin crecimiento.\n"
        
    facts = (
        f"- **Zonas con alerta de Burn ROI (Dinero Quemado)**: {len(alarms)} zonas.\n"
        f"- **Contexto Narrativo**: Destaca siempre la magnitud del aumento de Markdowns frente a la caída o estancamiento de Orders y Profit negativo.\n"
    )
    
    table = "| País | Ciudad | Zona | Markdowns L3W → L0W | Crecimiento Órdenes | Profit UE L0W |\n"
    table += "|------|--------|------|---------------------|---------------------|---------------|\n"
    for _, r in alarms.iterrows():
        mkdn0 = r[f'Restaurants Markdowns / GMV_{L0}'] * 100
        mkdn3 = r[f'Restaurants Markdowns / GMV_{L3}'] * 100
        table += f"| {r['COUNTRY']} | {r['CITY']} | {r['ZONE']} | {mkdn3:.1f}% → {mkdn0:.1f}% | 🔴 {r['Ord_Growth']*100:+.1f}% | 🔴 {r[f'Gross Profit UE_{L0}']:.2f} |\n"
        
    return (
        "## 💸 Eficiencia de Inversión (Burn ROI vs. Growth)\n\n"
        "**Alerta de Dinero Quemado**: Zonas donde la inversión promocional subió (>15%), "
        "pero las órdenes se estancaron y el margen cayó a negativo. Sugerencia: pausar campañas de descuento y revisar surtido.\n\n"
        f"{facts}\n"
        f"{table}\n"
    )

@tool
def get_bottleneck_diagnostics() -> str:
    """
    B. Diagnóstico de Cuellos de Botella (Funnel Drop-offs).
    Compara las métricas de conversión como una cascada frente a la media de la ciudad,
    identificando la fuga de conversión principal en cada zona prioritaria.
    """
    if df_merged.empty or L0 is None:
        return "Error: datos no disponibles."
    df = _prep()
    
    funnel = ["Restaurants SST > SS CVR", "Restaurants SS > ATC CVR", "Non-Pro PTC > OP"]
    fdf = df[df["METRIC"].isin(funnel)].copy()
    
    city_avg = fdf.groupby(["CITY", "METRIC"])[L0].mean().rename("CITY_AVG").reset_index()
    fdf = fdf.merge(city_avg, on=["CITY", "METRIC"], how="left")
    
    fdf["GAP_TO_CITY"] = (fdf[L0] - fdf["CITY_AVG"]) / fdf["CITY_AVG"].replace(0, np.nan)
    
    worst_bottleneck = fdf.loc[fdf.groupby(["COUNTRY","CITY","ZONE"])["GAP_TO_CITY"].idxmin()]
    worst_bottleneck = worst_bottleneck[worst_bottleneck["GAP_TO_CITY"] < -0.15]
    
    pri = worst_bottleneck["ZONE_PRIORITIZATION"].str.contains("High Priority|Prioritized", case=False, na=False)
    worst_bottleneck = worst_bottleneck[pri].sort_values(["ORDERS_L0W", "GAP_TO_CITY"], ascending=[False, True]).head(15)
    
    if worst_bottleneck.empty:
        return "## 🔍 Diagnóstico de Cuellos de Botella (Funnel Drop-offs)\n\nNo se detectaron fugas severas de conversión vs el promedio de la ciudad en zonas prioritarias.\n"
        
    facts = (
        f"- **Zonas prioritarias con fuga severa de conversión**: {len(worst_bottleneck)} analizadas.\n"
        f"- **Contexto Narrativo Obligatorio**: Compara siempre el 'CVR L0W' de la zona contra la 'Media Ciudad' para ilustrar la gravedad del cuello de botella.\n"
    )
    
    table = "| País | Ciudad | Zona | Etapa del Funnel Rota | CVR L0W | Media Ciudad | Brecha |\n"
    table += "|------|--------|------|-----------------------|---------|--------------|--------|\n"
    for _, r in worst_bottleneck.iterrows():
        table += f"| {r['COUNTRY']} | {r['CITY']} | {r['ZONE']} | **{r['METRIC']}** | {r[L0]*100:.1f}% | {r['CITY_AVG']*100:.1f}% | 🔴 {r['GAP_TO_CITY']*100:.1f}% |\n"
        
    return (
        "## 🔍 Diagnóstico de Cuellos de Botella (Funnel Drop-offs)\n\n"
        "**Fugas de Conversión**: El análisis lee el funnel en cascada (SST>SS → SS>ATC → PTC>OP) "
        "y detecta exactamente en qué paso la zona se rompe severamente vs la ciudad.\n\n"
        f"{facts}\n"
        f"{table}\n"
    )

@tool
def get_monetization_gaps() -> str:
    """
    C. Brechas de Monetización del Ecosistema (Cross-Sell & Loyalty Gaps).
    Zonas con alta madurez (Top 25% Orders) pero baja penetración (Bottom 25%) 
    en adopción de productos de alto margen/retención (Pro, Turbo, MLTV).
    """
    if df_merged.empty or L0 is None:
        return "Error: datos no disponibles."
    df = _prep()
    
    valid = df["ZONE_PRIORITIZATION"].str.contains("High Priority|Prioritized", case=False, na=False) & \
            df["ZONE_TYPE"].str.contains("Wealthy", case=False, na=False)
    mdf = df[valid].copy()
    if mdf.empty: return "Sin zonas prioritarias Wealthy."
    
    city_ord_q75 = mdf.groupby("CITY")["ORDERS_L0W"].transform(lambda x: x.quantile(0.75))
    mdf = mdf[mdf["ORDERS_L0W"] >= city_ord_q75]
    
    target_metrics = ["Pro Adoption", "Turbo Adoption", "MLTV Top Verticals Adoption"]
    mdf = mdf[mdf["METRIC"].isin(target_metrics)].copy()
    
    city_met_q25 = mdf.groupby(["CITY", "METRIC"])[L0].transform(lambda x: x.quantile(0.25))
    city_met_mean = mdf.groupby(["CITY", "METRIC"])[L0].transform("mean")
    mdf["CITY_MEAN"] = city_met_mean
    
    gaps = mdf[mdf[L0] <= city_met_q25].copy()
    gaps["GAP_PP"] = gaps["CITY_MEAN"] - gaps[L0]
    gaps = gaps.sort_values(["ORDERS_L0W", "GAP_PP"], ascending=[False, False]).head(15)
    
    if gaps.empty:
        return "## 💎 Brechas de Monetización del Ecosistema\n\nNo se detectaron gaps de adopción en las zonas top volume.\n"
        
    facts = (
        f"- **Oportunidades de Alto Margen detectadas**: {len(gaps)} en zonas maduras (Wealthy / Top Volume).\n"
        f"- **Contexto Narrativo Obligatorio**: Al recomendar acciones, resalta el gap contra la 'Media Ciudad' (ej. cerrando el gap de {gaps.iloc[0]['GAP_PP']*100:.1f} pp).\n"
    )
    
    table = "| Ciudad | Zona | Vol. Órdenes | Producto Subpenitrado | Adopción L0W | Media Ciudad | Oportunidad |\n"
    table += "|--------|------|--------------|-----------------------|--------------|--------------|-------------|\n"
    for _, r in gaps.iterrows():
        table += f"| {r['CITY']} | {r['ZONE']} | {int(r['ORDERS_L0W'])} | **{r['METRIC']}** | {r[L0]*100:.1f}% | {r['CITY_MEAN']*100:.1f}% | 🟢 +{r['GAP_PP']*100:.1f} pp |\n"
        
    return (
        "## 💎 Brechas de Monetización del Ecosistema (Cross-Sell & Loyalty Gaps)\n\n"
        "**Oportunidad de Oro ('Low hanging fruit')**: Zonas que generan alto volumen de órdenes "
        "pero sub-indexan fuertemente en retención o cross-sell (Pro, Turbo, MLTV). Ideales para marketing focalizado.\n\n"
        f"{facts}\n"
        f"{table}\n"
    )

@tool
def get_top_bottom_zones() -> str:
    """
    Ranking de zonas por score compuesto (0-100): promedio de métricas normalizadas
    por país, ajustadas por polaridad. Top 5 mejores y peores zonas de toda la red.
    """
    if df_merged.empty or L0 is None:
        return "Error: datos no disponibles."
    df = _prep()

    def norm(series, polarity):
        mn, mx = series.min(), series.max()
        if mx == mn:
            return pd.Series(0.5, index=series.index)
        n = (series - mn) / (mx - mn)
        return n if polarity == 1 else (1 - n)

    df["NORM"] = 0.0
    for (m, p), g in df.groupby(["METRIC","POL"]):
        df.loc[g.index, "NORM"] = norm(g[L0], p).values

    scores = (df.groupby(["COUNTRY","CITY","ZONE"])
              .agg(SCORE=("NORM","mean"), N=("METRIC","count"))
              .reset_index())
    scores = scores[scores["N"] >= 3]
    orders = df.drop_duplicates(["COUNTRY","CITY","ZONE"])[["COUNTRY","CITY","ZONE","ORDERS_L0W"]]
    scores = scores.merge(orders, on=["COUNTRY","CITY","ZONE"], how="left")
    scores["SCORE"] = scores["SCORE"] * 100

    # Weakest metric per zone
    weak = (df.sort_values("NORM")
            .groupby(["COUNTRY","CITY","ZONE"])
            .first()
            .reset_index()[["COUNTRY","CITY","ZONE","METRIC"]]
            .rename(columns={"METRIC":"MÉTRICA_DÉBIL"}))

    top5 = scores.nlargest(5,"SCORE").merge(weak, on=["COUNTRY","CITY","ZONE"], how="left")
    bot5 = scores.nsmallest(5,"SCORE").merge(weak, on=["COUNTRY","CITY","ZONE"], how="left")

    avg_score = scores["SCORE"].mean()
    gap = top5.iloc[0]["SCORE"] - bot5.iloc[0]["SCORE"]
    facts = (
        f"- **Zonas rankeadas**: {len(scores)} (con ≥3 métricas disponibles)\n"
        f"- **Score medio de la red**: {avg_score:.1f}/100\n"
        f"- **Brecha top/bottom**: {gap:.1f} puntos de score compuesto\n"
        f"- **Zona #1**: {top5.iloc[0]['ZONE']} ({top5.iloc[0]['COUNTRY']}) — Score {top5.iloc[0]['SCORE']:.1f}\n"
        f"- **Zona de mayor riesgo**: {bot5.iloc[0]['ZONE']} ({bot5.iloc[0]['COUNTRY']}) — Score {bot5.iloc[0]['SCORE']:.1f}, métrica débil: {bot5.iloc[0].get('MÉTRICA_DÉBIL','—')}\n"
    )

    def _tbl(src, ico, title):
        t  = f"\n### {ico} {title}\n"
        t += "| # | País | Ciudad | Zona | Score | Órdenes L0W | Métrica más Débil |\n"
        t += "|:-:|------|--------|------|:-----:|:-----------:|-------------------|\n"
        for i, (_, r) in enumerate(src.iterrows(), 1):
            t += (f"| {i} | {r['COUNTRY']} | {r['CITY']} | {r['ZONE']} "
                  f"| {r['SCORE']:.1f} | {int(r['ORDERS_L0W']):,} | {r.get('MÉTRICA_DÉBIL','—')} |\n")
        return t

    return (
        "## 🏆 Ranking de Zonas — Performance Compuesta\n\n"
        "**HECHOS CLAVE**:\n"
        f"{facts}\n"
        + _tbl(top5, "🥇","Top 5 — Zonas de Mayor Excelencia Operativa")
        + _tbl(bot5, "⚠️","Bottom 5 — Zonas de Mayor Riesgo Sistémico")
        + "\n_Score 0-100 = promedio de métricas normalizadas por país, ajustadas por polaridad._\n"
    )