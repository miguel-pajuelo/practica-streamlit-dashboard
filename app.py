import os
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# -----------------------------
# Configuración de la aplicación
# -----------------------------
"""
Configura la página principal de Streamlit con título, ícono y layout amplio.
"""
st.set_page_config(
    page_title="Dashboard Ventas - Empresa Alimentación",
    page_icon="📈",
    layout="wide",
)

# CSS personalizado mejorado para una interfaz más humana: colores suaves, tipografía legible, sombras suaves, transiciones y un diseño más cálido y acogedor
st.markdown(
    """
    <style>
      /* Fondo general oscuro */
      .stApp {
        background-color: #0b0f19;
        color: #e5e7eb;
      }

      /* Contenedor principal sin “tarjetas” claras */
      .block-container { 
        padding-top: 1.2rem; 
        padding-bottom: 2.5rem; 
        background-color: transparent;
        border-radius: 0px;
        box-shadow: none;
      }

      /* Métricas: sin contenedor extra y con buen contraste */
      [data-testid="stMetric"]{
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
      }
      [data-testid="stMetricLabel"] { 
        font-size: 1rem; 
        color: #cbd5e1;
      }
      [data-testid="stMetricValue"] { 
        font-size: 2.1rem; 
        font-weight: 800; 
        color: #60a5fa;
      }
      [data-testid="stMetricDelta"]{
        color: #a3a3a3;
      }

      /* Pestañas */
      .stTabs [data-baseweb="tab"] { 
        font-size: 1.05rem; 
        padding: 0.7rem 1.1rem; 
        border-radius: 10px; 
        transition: background-color 0.25s ease;
        color: #e5e7eb;
      }
      .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(96,165,250,0.12);
      }

      /* Texto muted */
      .muted { 
        opacity: 0.9; 
        font-size: 0.95rem; 
        color: #94a3b8; 
      }

      /* Títulos */
      h1, h2, h3 {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #e5e7eb;
      }

      /* Inputs */
      .stSelectbox > div, .stSelectbox div[data-baseweb="select"] {
        background: rgba(17,24,39,0.85) !important;
        border-radius: 10px !important;
      }
      .stSelectbox *{
        color: #e5e7eb !important;
      }
      .stSelectbox, .stButton button {
        border-radius: 10px;
        border: 1px solid rgba(148,163,184,0.25);
        transition: border-color 0.25s ease, background-color 0.25s ease;
      }
      .stSelectbox:hover, .stButton button:hover {
        border-color: rgba(96,165,250,0.55);
      }
      .stButton button{
        background: rgba(17,24,39,0.85);
        color: #e5e7eb;
      }

      /* Gráficos */
      .plotly-chart {
        border-radius: 12px;
        overflow: hidden;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# Constantes para el orden de los días de la semana (en inglés y español)
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
DAY_ES = {
    "Monday": "Lunes",
    "Tuesday": "Martes",
    "Wednesday": "Miércoles",
    "Thursday": "Jueves",
    "Friday": "Viernes",
    "Saturday": "Sábado",
    "Sunday": "Domingo",
}

# -----------------------------
# Estilo visual para gráficos (Plotly)
# -----------------------------
CHART_BG = "#0b0f19"
GRID_COLOR = "rgba(255,255,255,0.10)"
AXIS_COLOR = "rgba(255,255,255,0.28)"
FONT_COLOR = "#e5e7eb"

# Paletas para fondo oscuro (evitan tonos demasiado oscuros)
SCALE_TEAL = ["#ccfbf1", "#99f6e4", "#5eead4", "#2dd4bf", "#14b8a6"]
SCALE_BLUE = ["#dbeafe", "#bfdbfe", "#93c5fd", "#60a5fa", "#3b82f6"]
SCALE_PURPLE = ["#ede9fe", "#ddd6fe", "#c4b5fd", "#a78bfa", "#8b5cf6"]
SCALE_CYAN = ["#ecfeff", "#cffafe", "#a5f3fc", "#67e8f9", "#22d3ee"]

def style_base(fig):
    """Aplica un tema oscuro coherente a cualquier figura Plotly."""
    fig.update_layout(
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
        font=dict(color=FONT_COLOR),
        title=dict(font=dict(color=FONT_COLOR)),
        legend=dict(font=dict(color=FONT_COLOR)),
        hoverlabel=dict(bgcolor="rgba(10,15,25,0.96)", font=dict(color=FONT_COLOR)),
    )
    fig.update_xaxes(
        showline=True,
        linecolor=AXIS_COLOR,
        tickcolor=AXIS_COLOR,
        showgrid=True,
        gridcolor=GRID_COLOR,
        zeroline=False,
    )
    fig.update_yaxes(
        showline=True,
        linecolor=AXIS_COLOR,
        tickcolor=AXIS_COLOR,
        showgrid=True,
        gridcolor=GRID_COLOR,
        zeroline=False,
    )
    try:
        fig.update_coloraxes(colorbar=dict(tickfont=dict(color=FONT_COLOR), titlefont=dict(color=FONT_COLOR)))
    except Exception:
        pass

def style_bar(fig, orientation: str = "v"):
    """Tema oscuro + borde sutil para barras. orientation: 'v' o 'h'."""
    style_base(fig)
    fig.update_traces(marker_line_width=0.6, marker_line_color="rgba(255,255,255,0.18)")
    if orientation == "h":
        fig.update_yaxes(showgrid=False)
        fig.update_xaxes(showgrid=True)
    else:
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True)

def style_time_line(fig, dtick=None):
    """Tema oscuro + rejilla + puntos + spikes para ver mejor cada instante temporal."""
    style_base(fig)
    fig.update_layout(hovermode="x unified")
    fig.update_traces(mode="lines+markers", marker=dict(size=4, opacity=0.9), line=dict(width=2))
    fig.update_xaxes(
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikedash="dot",
        spikecolor="rgba(255,255,255,0.38)",
        spikethickness=1,
    )
    if dtick is not None:
        fig.update_xaxes(dtick=dtick)

def style_scatter(fig):
    """Tema oscuro para scatter (coherencia visual)."""
    style_base(fig)
    fig.update_traces(marker=dict(opacity=0.82))

# -----------------------------
# Carga de datos
# -----------------------------
@st.cache_data(show_spinner=True)
def load_data() -> pd.DataFrame:
    """
    Carga el dataset desde dos CSV alojados en Dropbox (parte_1 y parte_2)
    y los concatena en un único DataFrame.
    Compatible 100% con Streamlit Cloud.
    """
    URL1 = "https://dl.dropboxusercontent.com/scl/fi/oni9yw8jyv2zh3e09ebgc/parte_1.csv?rlkey=4trd7syzk8yuoerz4angussmo"
    URL2 = "https://dl.dropboxusercontent.com/scl/fi/88mexprkeldc8d6g4wlob/parte_2.csv?rlkey=hjnbwcx63mavy3it7v5r7cjem"
    # Columnas necesarias
    usecols = [
        "date", "store_nbr", "family", "sales", "onpromotion",
        "holiday_type", "locale", "locale_name", "transferred",
        "dcoilwtico", "city", "state", "store_type", "cluster",
        "transactions", "year", "month", "week", "quarter", "day_of_week",
    ]
    # Tipos optimizados
    dtypes = {
        "store_nbr": "int16",
        "sales": "float32",
        "onpromotion": "int16",
        "dcoilwtico": "float32",
        "cluster": "int16",
        "transactions": "float32",
        "year": "int16",
        "month": "int8",
        "week": "int16",
        "quarter": "int8",
    }
    # 🔥 FORZAR SIEMPRE DROPBOX
    df1 = pd.read_csv(URL1, usecols=usecols, dtype=dtypes, parse_dates=["date"], low_memory=False)
    df2 = pd.read_csv(URL2, usecols=usecols, dtype=dtypes, parse_dates=["date"], low_memory=False)
    # Concatenar
    df = pd.concat([df1, df2], ignore_index=True)
    # Categoricals → menos memoria
    cat_cols = [
        "family", "holiday_type", "locale", "locale_name", "transferred",
        "city", "state", "store_type", "day_of_week",
    ]
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype("category")
    return df

@st.cache_data(show_spinner=False)
def build_derived_tables(df: pd.DataFrame) -> dict:
    """
    Precalcula tablas agregadas derivadas del DataFrame principal para mejorar el rendimiento de la app.
   
    - Evita duplicados en agregaciones (ej. 'transactions' se deduplica por (date, store_nbr)).
    - Crea tablas para análisis diarios, por tienda y promociones.
   
    Args:
        df (pd.DataFrame): DataFrame principal con datos de ventas.
   
    Returns:
        dict: Diccionario con tablas derivadas precalculadas.
    """
    # Tabla única por día para estacionalidad y efectos globales
    date_dim = df[["date", "year", "month", "week", "quarter", "day_of_week", "dcoilwtico", "holiday_type", "transferred"]].drop_duplicates("date")
    daily_sales = (
        df.groupby("date", as_index=False)["sales"].sum()
        .merge(date_dim, on="date", how="left")
    )
    # Tabla única por tienda-día para transacciones sin duplicados
    store_day = df[["date", "store_nbr", "transactions", "city", "state", "store_type", "cluster", "year"]].drop_duplicates(["date", "store_nbr"])
    # Ventas agregadas por tienda-día
    store_day_sales = df.groupby(["date", "store_nbr"], as_index=False)["sales"].sum()
    # Ventas de productos en promoción por tienda-día
    promo = df[df["onpromotion"] > 0]
    promo_store_day_sales = promo.groupby(["date", "store_nbr"], as_index=False)["sales"].sum()
    return {
        "date_dim": date_dim,
        "daily_sales": daily_sales,
        "store_day": store_day,
        "store_day_sales": store_day_sales,
        "promo_store_day_sales": promo_store_day_sales,
    }

def format_int(x: float | int) -> str:
    """
    Formatea un número entero con separadores de miles (usando punto como separador).
   
    Args:
        x (float | int): Número a formatear.
   
    Returns:
        str: Número formateado como string.
    """
    try:
        return f"{int(x):,}".replace(",", ".")
    except Exception:
        return str(x)

# -----------------------------
# Interfaz principal de la app
# -----------------------------
st.title("Dashboard de Ventas")
st.markdown("Practica Final Visualización de Datos - Miguel Pajuelo Gómez")  # Añadido texto acogedor para humanizar

# Carga de datos y tablas derivadas
df = load_data()
derived = build_derived_tables(df)
daily_sales = derived["daily_sales"]
store_day = derived["store_day"]
store_day_sales = derived["store_day_sales"]
promo_store_day_sales = derived["promo_store_day_sales"]

# Cálculo de KPIs globales
n_stores = int(df["store_nbr"].nunique())
n_products = int(df["family"].nunique())
n_states = int(df["state"].nunique())
n_months = int(df["date"].dt.to_period("M").nunique())
date_min = df["date"].min().date()
date_max = df["date"].max().date()

# Creación de pestañas para organizar el contenido
tabs = st.tabs(["1) Global", "2) Por tienda", "3) Por estado", "4) Insights extra"])  # Añadidos emojis para humanizar

# -----------------------------
# Pestaña 1: Visión Global
# -----------------------------
with tabs[0]:
    st.subheader("Visión global")
    # KPIs básicos globales en tarjetas más amigables
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Tiendas", format_int(n_stores))
    with c2:
        st.metric("Productos (families)", format_int(n_products))
    with c3:
        st.metric("Estados", format_int(n_states))
    with c4:
        st.metric("Meses con datos", format_int(n_months))

    st.markdown(f"<div class='muted'>Rango de fechas: <b>{date_min}</b> → <b>{date_max}</b></div>", unsafe_allow_html=True)
    st.divider()
    st.subheader("Análisis en términos medios")

    # i) Top 10 productos más vendidos (media diaria)
    prod_mean = (
        df.groupby("family", observed=True)["sales"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
        .rename(columns={"sales": "mean_sales"})
    )
    fig_prod = px.bar(
        prod_mean, x="mean_sales", y="family", orientation="h",
        title="Top 10 productos por venta media (unidades) por registro",
        labels={"mean_sales": "Venta media", "family": "Producto"},
        color="mean_sales", color_continuous_scale=SCALE_BLUE
    )
    fig_prod.update_layout(height=420, margin=dict(l=20, r=20, t=60, b=20))
    style_bar(fig_prod, orientation="h")
    st.plotly_chart(fig_prod, use_container_width=True)

    # ii) Venta media diaria por tienda (todas las tiendas)
    store_mean = (
        store_day_sales.groupby("store_nbr", observed=True)["sales"]
        .mean()
        .reset_index()
        .rename(columns={"sales": "mean_daily_sales"})
    )
    store_mean = store_mean.sort_values("mean_daily_sales", ascending=False)
    store_mean["store_nbr"] = store_mean["store_nbr"].astype(str)

    fig_store_dist = px.bar(
        store_mean,
        x="store_nbr",
        y="mean_daily_sales",
        title="Venta media diaria total por tienda",
        labels={"store_nbr": "Tienda", "mean_daily_sales": "Venta media diaria (total)"},
        text_auto=False,
        color="mean_daily_sales", color_continuous_scale=SCALE_TEAL
    )
    fig_store_dist.update_layout(
        height=420,
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis=dict(
            type="category",
            categoryorder="array",
            categoryarray=store_mean["store_nbr"].tolist(),
            title="Tienda",
        ),
        yaxis=dict(title="Venta media diaria (total)"),
        showlegend=False,
    )
    style_bar(fig_store_dist, orientation="v")
    st.plotly_chart(fig_store_dist, use_container_width=True)

    # iii) Top 10 tiendas por venta media diaria de productos en promoción
    promo_store_mean = (
        promo_store_day_sales.groupby("store_nbr", observed=True)["sales"]
        .mean()
        .reset_index()
        .rename(columns={"sales": "mean_daily_promo_sales"})
    )
    promo_store_mean = promo_store_mean.sort_values("mean_daily_promo_sales", ascending=False).head(10)
    promo_store_mean["store_nbr"] = promo_store_mean["store_nbr"].astype(str)

    fig_promo_store = px.bar(
        promo_store_mean,
        x="store_nbr",
        y="mean_daily_promo_sales",
        title="Top 10 tiendas por venta media diaria de productos en promoción",
        labels={"store_nbr": "Tienda", "mean_daily_promo_sales": "Venta media diaria (promo)"},
        text_auto=False,
        color="mean_daily_promo_sales", color_continuous_scale=SCALE_CYAN
    )
    fig_promo_store.update_layout(
        height=380,
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis=dict(
            type="category",
            categoryorder="array",
            categoryarray=promo_store_mean["store_nbr"].tolist(),
            title="Tienda",
        ),
        yaxis=dict(title="Venta media diaria (promo)"),
        showlegend=False,
    )
    style_bar(fig_promo_store, orientation="v")
    st.plotly_chart(fig_promo_store, use_container_width=True)

    # iv) Día de la semana con más ventas (media diaria total)
    dow = (
        daily_sales.groupby("day_of_week", observed=True)["sales"]
        .mean()
        .reindex(DAY_ORDER)
        .reset_index()
    )
    dow["day_es"] = dow["day_of_week"].map(DAY_ES).fillna(dow["day_of_week"].astype(str))
    best_dow = dow.loc[dow["sales"].idxmax(), "day_es"]
    best_val = dow["sales"].max()

    m1, m2 = st.columns([1, 3])
    with m1:
        st.metric("Mejor día (media)", best_dow, f"{best_val:,.0f}".replace(",", "."))
    with m2:
        fig_dow = px.bar(
            dow,
            x="day_es",
            y="sales",
            title="Venta media diaria total por día de la semana",
            labels={"day_es": "Día", "sales": "Venta media diaria (total)"},
            color="sales", color_continuous_scale=SCALE_PURPLE
        )
        fig_dow.update_layout(height=360, margin=dict(l=20, r=20, t=60, b=20))
        style_bar(fig_dow, orientation="v")
        st.plotly_chart(fig_dow, use_container_width=True)

    # v) Volumen medio por semana del año (promedio entre años)
    weekly_total = daily_sales.groupby(["year", "week"], as_index=False)["sales"].sum()
    weekly_avg = weekly_total.groupby("week", as_index=False)["sales"].mean()

    fig_week = px.line(
        weekly_avg,
        x="week",
        y="sales",
        title="Venta semanal media por semana del año (promedio entre años)",
        labels={"week": "Semana del año", "sales": "Venta semanal media"},
        line_shape="spline",
        markers=True
    )
    fig_week.update_layout(height=360, margin=dict(l=20, r=20, t=60, b=20))
    style_time_line(fig_week, dtick=1)
    st.plotly_chart(fig_week, use_container_width=True)

    # vi) Volumen medio por mes (promedio entre años)
    monthly_total = daily_sales.groupby(["year", "month"], as_index=False)["sales"].sum()
    monthly_avg = monthly_total.groupby("month", as_index=False)["sales"].mean()

    fig_month = px.line(
        monthly_avg,
        x="month",
        y="sales",
        title="Venta mensual media por mes (promedio entre años)",
        labels={"month": "Mes", "sales": "Venta mensual media"},
        line_shape="spline",
        markers=True
    )
    fig_month.update_layout(height=360, margin=dict(l=20, r=20, t=60, b=20))
    style_time_line(fig_month, dtick=1)
    st.plotly_chart(fig_month, use_container_width=True)

# -----------------------------
# Pestaña 2: Análisis por Tienda
# -----------------------------
with tabs[1]:
    st.subheader("Análisis por tienda")
    # Selector de tienda con etiqueta amigable
    stores = sorted(df["store_nbr"].unique().tolist())
    store_sel = st.selectbox("Elige una tienda para explorar sus datos 📊", stores, index=0)
    # Filtrado de datos por tienda seleccionada
    df_s = df[df["store_nbr"] == store_sel]
    store_sales_year = df_s.groupby("year", as_index=False)["sales"].sum().sort_values("year")
    # KPIs específicos de la tienda en tarjetas
    total_units = float(df_s["sales"].sum())
    unique_products_sold = int(df_s.loc[df_s["sales"] > 0, "family"].nunique())
    unique_promo_products = int(df_s.loc[(df_s["sales"] > 0) & (df_s["onpromotion"] > 0), "family"].nunique())

    k1, k2, k3 = st.columns(3)
    with k1:
        st.metric("Ventas totales (unidades)", f"{total_units:,.0f}".replace(",", "."))
    with k2:
        st.metric("Productos vendidos (distinct)", format_int(unique_products_sold))
    with k3:
        st.metric("Productos vendidos en promo (distinct)", format_int(unique_promo_products))

    # Gráfico de ventas por año para la tienda
    fig_store_year = px.bar(
        store_sales_year,
        x="year",
        y="sales",
        title=f"Ventas totales por año — tienda {store_sel}",
        labels={"year": "Año", "sales": "Ventas (unidades)"},
        color="sales", color_continuous_scale=SCALE_BLUE
    )
    fig_store_year.update_layout(height=420, margin=dict(l=20, r=20, t=60, b=20))
    style_bar(fig_store_year, orientation="v")
    st.plotly_chart(fig_store_year, use_container_width=True)

# -----------------------------
# Pestaña 3: Análisis por Estado
# -----------------------------
with tabs[2]:
    st.subheader("Análisis por estado")
    # Selector de estado con etiqueta amigable
    states = sorted(df["state"].dropna().unique().tolist())
    state_sel = st.selectbox("Selecciona un estado para ver sus insights 🏙️", states, index=0)

    # a) Transacciones por año en el estado (deduplicadas)
    sd_state = store_day[store_day["state"] == state_sel].copy()
    trans_year = sd_state.groupby("year", as_index=False)["transactions"].sum().sort_values("year")

    fig_trans_year = px.line(
        trans_year,
        x="year",
        y="transactions",
        markers=True,
        title=f"Transacciones totales por año — {state_sel}",
        labels={"year": "Año", "transactions": "Transacciones"},
        line_shape="spline"
    )
    fig_trans_year.update_layout(height=380, margin=dict(l=20, r=20, t=60, b=20))
    style_time_line(fig_trans_year, dtick=1)
    st.plotly_chart(fig_trans_year, use_container_width=True)

    # b) Ranking de tiendas con más ventas en el estado
    df_state = df[df["state"] == state_sel]
    store_rank = (
        df_state.groupby("store_nbr", observed=True)["sales"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    fig_store_rank = px.bar(
        store_rank,
        x="store_nbr",
        y="sales",
        title=f"Top 10 tiendas por ventas (unidades) — {state_sel}",
        labels={"store_nbr": "Tienda", "sales": "Ventas (unidades)"},
        color="sales", color_continuous_scale=SCALE_TEAL
    )
    fig_store_rank.update_layout(height=380, margin=dict(l=20, r=20, t=60, b=20))
    style_bar(fig_store_rank, orientation="v")
    st.plotly_chart(fig_store_rank, use_container_width=True)

    # c) Producto más vendido en el estado (top 10)
    prod_rank = (
        df_state.groupby("family", observed=True)["sales"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    top_prod = prod_rank.iloc[0]["family"] if len(prod_rank) else None
    top_units = prod_rank.iloc[0]["sales"] if len(prod_rank) else 0

    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Producto #1 (estado)", str(top_prod), f"{top_units:,.0f}".replace(",", "."))
    with c2:
        fig_prod_rank = px.bar(
            prod_rank,
            x="sales",
            y="family",
            orientation="h",
            title=f"Top 10 productos por ventas — {state_sel}",
            labels={"family": "Producto", "sales": "Ventas (unidades)"},
            color="sales", color_continuous_scale=SCALE_PURPLE
        )
        fig_prod_rank.update_layout(height=360, margin=dict(l=20, r=20, t=60, b=20))
        style_bar(fig_prod_rank, orientation="h")
        st.plotly_chart(fig_prod_rank, use_container_width=True)

# -----------------------------
# Pestaña 4: Insights Extra
# -----------------------------
with tabs[3]:
    st.subheader("Insights extra para acelerar conclusiones")

    # 1) Efecto de promociones: cuota diaria de ventas en promo
    promo_daily = df[df["onpromotion"] > 0].groupby("date", as_index=False)["sales"].sum().rename(columns={"sales": "promo_sales"})
    daily = daily_sales[["date", "sales"]].rename(columns={"sales": "total_sales"})
    promo_share = daily.merge(promo_daily, on="date", how="left")
    promo_share["promo_sales"] = promo_share["promo_sales"].fillna(0.0)
    promo_share["promo_share"] = np.where(promo_share["total_sales"] > 0, promo_share["promo_sales"] / promo_share["total_sales"], 0.0)

    c1, c2 = st.columns(2)
    with c1:
        fig_share = px.line(
            promo_share,
            x="date",
            y="promo_share",
            title="Cuota diaria de ventas con promoción (promo/total)",
            labels={"date": "Fecha", "promo_share": "Cuota promo"},
            line_shape="spline"
        )
        fig_share.update_layout(height=360, margin=dict(l=20, r=20, t=60, b=20))
        style_time_line(fig_share)
        fig_share.update_traces(marker=dict(size=3, opacity=0.75))
        st.plotly_chart(fig_share, use_container_width=True)

    with c2:
        fig_promo_vs = px.scatter(
            promo_share.sample(min(4000, len(promo_share)), random_state=7),
            x="promo_sales",
            y="total_sales",
            title="Relación: ventas promo vs ventas totales (muestra)",
            labels={"promo_sales": "Ventas promo", "total_sales": "Ventas totales"},
            color="promo_share", color_continuous_scale=SCALE_CYAN
        )
        fig_promo_vs.update_layout(height=360, margin=dict(l=20, r=20, t=60, b=20))
        style_scatter(fig_promo_vs)
        st.plotly_chart(fig_promo_vs, use_container_width=True)

    st.divider()

    # 2) Efecto de holidays: comparación de ventas medias con vs sin holiday_type
    ds = daily_sales.copy()
    ds["is_holiday"] = ds["holiday_type"].notna()
    holiday_mean = ds.groupby("is_holiday", as_index=False)["sales"].mean()
    holiday_mean["label"] = holiday_mean["is_holiday"].map({True: "Día con holiday_type", False: "Día sin holiday_type"})

    c1, c2 = st.columns([1, 2])
    with c1:
        diff = holiday_mean.loc[holiday_mean["is_holiday"] == True, "sales"].values
        base = holiday_mean.loc[holiday_mean["is_holiday"] == False, "sales"].values
        if len(diff) and len(base) and base[0] > 0:
            lift = (diff[0] / base[0] - 1) * 100
            st.metric("Cambio medio (holiday vs no)", f"{lift:,.1f}%".replace(",", "."))
        else:
            st.metric("Cambio medio (holiday vs no)", "N/A")

    with c2:
        fig_holiday = px.bar(
            holiday_mean,
            x="label",
            y="sales",
            title="Venta media diaria: con vs sin holiday_type",
            labels={"label": "", "sales": "Venta media diaria (total)"},
            color="is_holiday", color_discrete_sequence=["#94a3b8", "#60a5fa"]
        )
        fig_holiday.update_layout(height=320, margin=dict(l=20, r=20, t=60, b=20))
        style_bar(fig_holiday, orientation="v")
        st.plotly_chart(fig_holiday, use_container_width=True)

    # 3) Top estados por ventas acumuladas
    state_sales = (
        df.groupby("state", observed=True)["sales"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    fig_state = px.bar(
        state_sales,
        x="sales",
        y="state",
        orientation="h",
        title="Top 10 estados por ventas acumuladas",
        labels={"state": "Estado", "sales": "Ventas (unidades)"},
        color="sales", color_continuous_scale=SCALE_BLUE
    )
    fig_state.update_layout(height=380, margin=dict(l=20, r=20, t=60, b=20))
    style_bar(fig_state, orientation="h")
    st.plotly_chart(fig_state, use_container_width=True)

    st.caption("Nota: 'transactions' se agrega sin duplicar por (fecha, tienda), porque el dataset repite ese valor por cada family.")
