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

# -----------------------------
# Estética "humana" (sistema visual consistente)
# -----------------------------
HUMAN_UI_CSS = """
<style>
  :root{
    --bg: #f7f4ee;
    --panel: rgba(255,255,255,0.72);
    --text:#1f2937;
    --muted:#6b7280;

    --brand:#3b82f6;

    --border: rgba(31,41,55,0.10);
    --shadow-soft: 0 6px 18px rgba(15,23,42,0.06);

    --radius: 16px;
  }

  /* Fondo general */
  .stApp{
    background:
      radial-gradient(1200px 600px at 10% 0%, rgba(59,130,246,0.10), transparent 60%),
      radial-gradient(900px 500px at 90% 10%, rgba(96,165,250,0.10), transparent 55%),
      var(--bg);
    color: var(--text);
  }

  /* Contenedor principal */
  .main .block-container{
    padding-top: 1.2rem;
    padding-bottom: 2.5rem;
    max-width: 1400px;
  }

  /* Cards */
  .card{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    box-shadow: var(--shadow-soft);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
  }
  .card.pad{ padding: 1rem 1.2rem; }

  /* Header */
  .hero{
    padding: 1.1rem 1.2rem;
    display:flex;
    align-items:flex-start;
    justify-content:space-between;
    gap: 1rem;
  }
  .hero h1{
    margin: 0;
    font-size: 1.55rem;
    letter-spacing: -0.02em;
  }
  .hero p{
    margin: .25rem 0 0 0;
    color: var(--muted);
    font-size: 0.98rem;
  }
  .pill{
    display:inline-flex;
    align-items:center;
    gap:.5rem;
    padding:.45rem .75rem;
    border-radius: 999px;
    border: 1px solid var(--border);
    background: rgba(255,255,255,0.55);
    color: var(--muted);
    font-size: .9rem;
    white-space: nowrap;
  }

  /* Métricas */
  [data-testid="stMetricValue"]{
    font-size: 1.9rem !important;
    font-weight: 750 !important;
    color: var(--text) !important;
    letter-spacing: -0.02em;
  }
  [data-testid="stMetricLabel"]{
    color: var(--muted) !important;
    font-size: .95rem !important;
  }
  [data-testid="stMetricDelta"]{
    color: var(--muted) !important;
  }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"]{ gap: .35rem; }
  .stTabs [data-baseweb="tab"]{
    border-radius: 999px !important;
    padding: .55rem .95rem !important;
    border: 1px solid var(--border) !important;
    background: rgba(255,255,255,0.50) !important;
    color: var(--muted) !important;
    transition: all .18s ease;
  }
  .stTabs [aria-selected="true"]{
    background: rgba(255,255,255,0.85) !important;
    color: var(--text) !important;
    border-color: rgba(59,130,246,0.25) !important;
    box-shadow: 0 8px 22px rgba(59,130,246,0.10) !important;
  }

  /* Sidebar */
  section[data-testid="stSidebar"]{
    background: rgba(255,255,255,0.35);
    border-right: 1px solid var(--border);
    backdrop-filter: blur(10px);
  }

  /* Inputs */
  div[data-baseweb="select"] > div{
    border-radius: 999px !important;
    border: 1px solid var(--border) !important;
    background: rgba(255,255,255,0.70) !important;
    transition: all .15s ease;
  }
  div[data-baseweb="select"] > div:hover{
    border-color: rgba(59,130,246,0.30) !important;
  }
  div[data-baseweb="select"] > div:focus-within{
    border-color: rgba(59,130,246,0.55) !important;
    box-shadow: 0 0 0 4px rgba(59,130,246,0.12) !important;
  }

  /* Botones */
  .stButton button{
    border-radius: 999px !important;
    border: 1px solid rgba(59,130,246,0.25) !important;
    background: rgba(59,130,246,0.10) !important;
    color: var(--text) !important;
    transition: all .15s ease;
  }
  .stButton button:hover{
    background: rgba(59,130,246,0.16) !important;
    transform: translateY(-1px);
  }

  /* Plotly */
  .js-plotly-plot, .plotly, .plot-container{
    border-radius: var(--radius);
    overflow: hidden;
  }

  /* Tipografía */
  h1,h2,h3,h4{
    font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Arial;
    color: var(--text);
    letter-spacing: -0.01em;
  }
  .muted{ color: var(--muted); }

  @media (max-width: 900px){
    .hero{ flex-direction: column; align-items: stretch; }
    .pill{ width: fit-content; }
  }
</style>
"""
st.markdown(HUMAN_UI_CSS, unsafe_allow_html=True)

# -----------------------------
# Constantes / helpers
# -----------------------------
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

PLOT_FONT = "ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial"
PLOT_TEXT = "#1f2937"
PLOT_GRID = "rgba(31,41,55,0.08)"

def humanize_plotly(fig, height=None):
    """Unifica estilo de Plotly para que encaje con la UI (fondos transparentes, tipografía, grid suave)."""
    fig.update_layout(
        template="plotly_white",
        height=height if height is not None else fig.layout.height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=PLOT_FONT, size=13, color=PLOT_TEXT),
        title=dict(x=0, xanchor="left", font=dict(size=16, color=PLOT_TEXT)),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor=PLOT_GRID, zeroline=False)
    return fig

def format_int(x: float | int) -> str:
    """
    Formatea un número entero con separadores de miles (usando punto como separador).
    """
    try:
        return f"{int(x):,}".replace(",", ".")
    except Exception:
        return str(x)

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

    usecols = [
        "date", "store_nbr", "family", "sales", "onpromotion",
        "holiday_type", "locale", "locale_name", "transferred",
        "dcoilwtico", "city", "state", "store_type", "cluster",
        "transactions", "year", "month", "week", "quarter", "day_of_week",
    ]
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

    df1 = pd.read_csv(URL1, usecols=usecols, dtype=dtypes, parse_dates=["date"], low_memory=False)
    df2 = pd.read_csv(URL2, usecols=usecols, dtype=dtypes, parse_dates=["date"], low_memory=False)
    df = pd.concat([df1, df2], ignore_index=True)

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
    """
    date_dim = df[
        ["date", "year", "month", "week", "quarter", "day_of_week", "dcoilwtico", "holiday_type", "transferred"]
    ].drop_duplicates("date")

    daily_sales = (
        df.groupby("date", as_index=False)["sales"].sum()
        .merge(date_dim, on="date", how="left")
    )

    store_day = df[
        ["date", "store_nbr", "transactions", "city", "state", "store_type", "cluster", "year"]
    ].drop_duplicates(["date", "store_nbr"])

    store_day_sales = df.groupby(["date", "store_nbr"], as_index=False)["sales"].sum()

    promo = df[df["onpromotion"] > 0]
    promo_store_day_sales = promo.groupby(["date", "store_nbr"], as_index=False)["sales"].sum()

    return {
        "date_dim": date_dim,
        "daily_sales": daily_sales,
        "store_day": store_day,
        "store_day_sales": store_day_sales,
        "promo_store_day_sales": promo_store_day_sales,
    }

# -----------------------------
# Interfaz principal de la app
# -----------------------------
df = load_data()
derived = build_derived_tables(df)
daily_sales = derived["daily_sales"]
store_day = derived["store_day"]
store_day_sales = derived["store_day_sales"]
promo_store_day_sales = derived["promo_store_day_sales"]

# KPIs globales
n_stores = int(df["store_nbr"].nunique())
n_products = int(df["family"].nunique())
n_states = int(df["state"].nunique())
n_months = int(df["date"].dt.to_period("M").nunique())
date_min = df["date"].min().date()
date_max = df["date"].max().date()

# Header “humano”
st.markdown(
    f"""
    <div class="card hero">
      <div>
        <h1>📈 Dashboard de Ventas</h1>
        <p>Ventas, promociones y patrones por tienda/estado, con foco en lo accionable.</p>
      </div>
      <div class="pill">📅 Rango: <b>{date_min}</b> → <b>{date_max}</b></div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")

# Pestañas
tabs = st.tabs(["1) Global 🌍", "2) Por tienda 🏪", "3) Por estado 🗺️", "4) Insights extra 💡"])

# -----------------------------
# Pestaña 1: Visión Global
# -----------------------------
with tabs[0]:
    st.subheader("Visión global")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("<div class='card pad'>", unsafe_allow_html=True)
        st.metric("Tiendas", format_int(n_stores))
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='card pad'>", unsafe_allow_html=True)
        st.metric("Productos (families)", format_int(n_products))
        st.markdown("</div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='card pad'>", unsafe_allow_html=True)
        st.metric("Estados", format_int(n_states))
        st.markdown("</div>", unsafe_allow_html=True)
    with c4:
        st.markdown("<div class='card pad'>", unsafe_allow_html=True)
        st.metric("Meses con datos", format_int(n_months))
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(f"<span class='muted'>Rango de fechas: <b>{date_min}</b> → <b>{date_max}</b></span>", unsafe_allow_html=True)
    st.divider()

    st.subheader("Análisis en términos medios")

    # i) Top 10 productos más vendidos (media)
    prod_mean = (
        df.groupby("family", observed=True)["sales"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
        .rename(columns={"sales": "mean_sales"})
    )
    fig_prod = px.bar(
        prod_mean,
        x="mean_sales",
        y="family",
        orientation="h",
        title="Top 10 productos por venta media (unidades) por registro",
        labels={"mean_sales": "Venta media", "family": "Producto"},
        color="mean_sales",
        color_continuous_scale="Blues",
    )
    fig_prod = humanize_plotly(fig_prod, height=420)
    st.plotly_chart(fig_prod, use_container_width=True)

    # ii) Venta media diaria por tienda (todas las tiendas)
    store_mean = (
        store_day_sales.groupby("store_nbr", observed=True)["sales"]
        .mean()
        .reset_index()
        .rename(columns={"sales": "mean_daily_sales"})
        .sort_values("mean_daily_sales", ascending=False)
    )
    store_mean["store_nbr"] = store_mean["store_nbr"].astype(str)

    fig_store_dist = px.bar(
        store_mean,
        x="store_nbr",
        y="mean_daily_sales",
        title="Venta media diaria total por tienda",
        labels={"store_nbr": "Tienda", "mean_daily_sales": "Venta media diaria (total)"},
        color="mean_daily_sales",
        color_continuous_scale="Sunset",
    )
    fig_store_dist.update_layout(
        xaxis=dict(
            type="category",
            categoryorder="array",
            categoryarray=store_mean["store_nbr"].tolist(),
            title="Tienda",
        ),
        yaxis=dict(title="Venta media diaria (total)"),
        showlegend=False,
    )
    fig_store_dist = humanize_plotly(fig_store_dist, height=420)
    st.plotly_chart(fig_store_dist, use_container_width=True)

    # iii) Top 10 tiendas por venta media diaria en promoción
    promo_store_mean = (
        promo_store_day_sales.groupby("store_nbr", observed=True)["sales"]
        .mean()
        .reset_index()
        .rename(columns={"sales": "mean_daily_promo_sales"})
        .sort_values("mean_daily_promo_sales", ascending=False)
        .head(10)
    )
    promo_store_mean["store_nbr"] = promo_store_mean["store_nbr"].astype(str)

    fig_promo_store = px.bar(
        promo_store_mean,
        x="store_nbr",
        y="mean_daily_promo_sales",
        title="Top 10 tiendas por venta media diaria de productos en promoción",
        labels={"store_nbr": "Tienda", "mean_daily_promo_sales": "Venta media diaria (promo)"},
        color="mean_daily_promo_sales",
        color_continuous_scale="Viridis",
    )
    fig_promo_store.update_layout(
        xaxis=dict(
            type="category",
            categoryorder="array",
            categoryarray=promo_store_mean["store_nbr"].tolist(),
            title="Tienda",
        ),
        yaxis=dict(title="Venta media diaria (promo)"),
        showlegend=False,
    )
    fig_promo_store = humanize_plotly(fig_promo_store, height=380)
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
        st.markdown("<div class='card pad'>", unsafe_allow_html=True)
        st.metric("Mejor día (media)", best_dow, f"{best_val:,.0f}".replace(",", "."))
        st.markdown("</div>", unsafe_allow_html=True)
    with m2:
        fig_dow = px.bar(
            dow,
            x="day_es",
            y="sales",
            title="Venta media diaria total por día de la semana",
            labels={"day_es": "Día", "sales": "Venta media diaria (total)"},
            color="sales",
            color_continuous_scale="Teal",
        )
        fig_dow = humanize_plotly(fig_dow, height=360)
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
        markers=True,
    )
    fig_week = humanize_plotly(fig_week, height=360)
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
        markers=True,
    )
    fig_month = humanize_plotly(fig_month, height=360)
    st.plotly_chart(fig_month, use_container_width=True)

# -----------------------------
# Pestaña 2: Análisis por Tienda
# -----------------------------
with tabs[1]:
    st.subheader("Análisis por tienda")

    stores = sorted(df["store_nbr"].unique().tolist())
    store_sel = st.selectbox("Elige una tienda para explorar sus datos 📊", stores, index=0)

    df_s = df[df["store_nbr"] == store_sel]
    store_sales_year = df_s.groupby("year", as_index=False)["sales"].sum().sort_values("year")

    total_units = float(df_s["sales"].sum())
    unique_products_sold = int(df_s.loc[df_s["sales"] > 0, "family"].nunique())
    unique_promo_products = int(df_s.loc[(df_s["sales"] > 0) & (df_s["onpromotion"] > 0), "family"].nunique())

    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown("<div class='card pad'>", unsafe_allow_html=True)
        st.metric("Ventas totales (unidades)", f"{total_units:,.0f}".replace(",", "."))
        st.markdown("</div>", unsafe_allow_html=True)
    with k2:
        st.markdown("<div class='card pad'>", unsafe_allow_html=True)
        st.metric("Productos vendidos (distinct)", format_int(unique_products_sold))
        st.markdown("</div>", unsafe_allow_html=True)
    with k3:
        st.markdown("<div class='card pad'>", unsafe_allow_html=True)
        st.metric("Productos vendidos en promo (distinct)", format_int(unique_promo_products))
        st.markdown("</div>", unsafe_allow_html=True)

    fig_store_year = px.bar(
        store_sales_year,
        x="year",
        y="sales",
        title=f"Ventas totales por año — tienda {store_sel}",
        labels={"year": "Año", "sales": "Ventas (unidades)"},
        color="sales",
        color_continuous_scale="Blues",
    )
    fig_store_year = humanize_plotly(fig_store_year, height=420)
    st.plotly_chart(fig_store_year, use_container_width=True)

# -----------------------------
# Pestaña 3: Análisis por Estado
# -----------------------------
with tabs[2]:
    st.subheader("Análisis por estado")

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
        line_shape="spline",
    )
    fig_trans_year = humanize_plotly(fig_trans_year, height=380)
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
        color="sales",
        color_continuous_scale="Sunset",
    )
    fig_store_rank = humanize_plotly(fig_store_rank, height=380)
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
        st.markdown("<div class='card pad'>", unsafe_allow_html=True)
        st.metric("Producto #1 (estado)", str(top_prod), f"{top_units:,.0f}".replace(",", "."))
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        fig_prod_rank = px.bar(
            prod_rank,
            x="sales",
            y="family",
            orientation="h",
            title=f"Top 10 productos por ventas — {state_sel}",
            labels={"family": "Producto", "sales": "Ventas (unidades)"},
            color="sales",
            color_continuous_scale="Blues",
        )
        fig_prod_rank = humanize_plotly(fig_prod_rank, height=360)
        st.plotly_chart(fig_prod_rank, use_container_width=True)

# -----------------------------
# Pestaña 4: Insights Extra
# -----------------------------
with tabs[3]:
    st.subheader("Insights extra para acelerar conclusiones")

    # 1) Efecto de promociones: cuota diaria de ventas en promo
    promo_daily = (
        df[df["onpromotion"] > 0]
        .groupby("date", as_index=False)["sales"]
        .sum()
        .rename(columns={"sales": "promo_sales"})
    )
    daily = daily_sales[["date", "sales"]].rename(columns={"sales": "total_sales"})
    promo_share = daily.merge(promo_daily, on="date", how="left")
    promo_share["promo_sales"] = promo_share["promo_sales"].fillna(0.0)
    promo_share["promo_share"] = np.where(
        promo_share["total_sales"] > 0,
        promo_share["promo_sales"] / promo_share["total_sales"],
        0.0,
    )

    c1, c2 = st.columns(2)
    with c1:
        fig_share = px.line(
            promo_share,
            x="date",
            y="promo_share",
            title="Cuota diaria de ventas con promoción (promo/total)",
            labels={"date": "Fecha", "promo_share": "Cuota promo"},
            line_shape="spline",
        )
        fig_share = humanize_plotly(fig_share, height=360)
        st.plotly_chart(fig_share, use_container_width=True)

    with c2:
        if len(promo_share) > 0:
            sample_df = promo_share.sample(min(4000, len(promo_share)), random_state=7)
            fig_promo_vs = px.scatter(
                sample_df,
                x="promo_sales",
                y="total_sales",
                title="Relación: ventas promo vs ventas totales (muestra)",
                labels={"promo_sales": "Ventas promo", "total_sales": "Ventas totales"},
                color="promo_share",
                color_continuous_scale="Plasma",
            )
            fig_promo_vs = humanize_plotly(fig_promo_vs, height=360)
            st.plotly_chart(fig_promo_vs, use_container_width=True)
        else:
            st.info("No hay datos suficientes para mostrar la relación promo vs total en el rango actual.")

    st.divider()

    # 2) Efecto de holidays: comparación de ventas medias con vs sin holiday_type
    ds = daily_sales.copy()
    ds["is_holiday"] = ds["holiday_type"].notna()
    holiday_mean = ds.groupby("is_holiday", as_index=False)["sales"].mean()
    holiday_mean["label"] = holiday_mean["is_holiday"].map({True: "Día con holiday_type", False: "Día sin holiday_type"})

    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown("<div class='card pad'>", unsafe_allow_html=True)
        diff = holiday_mean.loc[holiday_mean["is_holiday"] == True, "sales"].values
        base = holiday_mean.loc[holiday_mean["is_holiday"] == False, "sales"].values
        if len(diff) and len(base) and base[0] > 0:
            lift = (diff[0] / base[0] - 1) * 100
            st.metric("Cambio medio (holiday vs no)", f"{lift:,.1f}%".replace(",", "."))
        else:
            st.metric("Cambio medio (holiday vs no)", "N/A")
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        fig_holiday = px.bar(
            holiday_mean,
            x="label",
            y="sales",
            title="Venta media diaria: con vs sin holiday_type",
            labels={"label": "", "sales": "Venta media diaria (total)"},
            color="is_holiday",
            color_discrete_sequence=["#cbd5e1", "#60a5fa"],
        )
        fig_holiday = humanize_plotly(fig_holiday, height=320)
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
        color="sales",
        color_continuous_scale="Sunsetdark",
    )
    fig_state = humanize_plotly(fig_state, height=380)
    st.plotly_chart(fig_state, use_container_width=True)

    st.caption("Nota: 'transactions' se agrega sin duplicar por (fecha, tienda), porque el dataset repite ese valor por cada family.")
