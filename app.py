import os
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# -----------------------------
# Configuración de la aplicación
# -----------------------------
st.set_page_config(
    page_title="Dashboard Ventas - Empresa Alimentación",
    page_icon="🛒",
    layout="wide",
)

# CSS personalizado con diseño más humano y amigable
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
      
      /* Estilos generales */
      .block-container { 
        padding-top: 2rem; 
        padding-bottom: 3rem;
        max-width: 1400px;
      }
      
      /* Tipografía más amigable */
      h1, h2, h3, h4, h5, h6, p, div, span {
        font-family: 'Inter', sans-serif !important;
      }
      
      /* Título principal con estilo conversacional */
      .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
      }
      
      .subtitle {
        font-size: 1.1rem;
        color: #64748b;
        font-weight: 400;
        margin-bottom: 2rem;
      }
      
      /* Tarjetas KPI renovadas */
      .kpi-card {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        border-radius: 20px;
        padding: 1.5rem;
        border: 2px solid #e0e7ff;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
        margin: 0.5rem 0;
      }
      
      .kpi-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 15px rgba(102, 126, 234, 0.2);
      }
      
      /* Métricas con mejor diseño */
      [data-testid="stMetricValue"] { 
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
      }
      
      [data-testid="stMetricLabel"] { 
        font-size: 0.9rem;
        font-weight: 500;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 0.5px;
      }
      
      /* Pestañas estilizadas */
      .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f8fafc;
        padding: 0.5rem;
        border-radius: 12px;
      }
      
      .stTabs [data-baseweb="tab"] {
        font-size: 1rem;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        color: #64748b;
        transition: all 0.3s ease;
      }
      
      .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
      }
      
      /* Dividers más suaves */
      hr {
        margin: 2rem 0;
        border: none;
        height: 1px;
        background: linear-gradient(to right, transparent, #e2e8f0, transparent);
      }
      
      /* Textos informativos */
      .muted {
        color: #64748b;
        font-size: 0.95rem;
        font-weight: 400;
        padding: 0.75rem;
        background: #f8fafc;
        border-radius: 8px;
        border-left: 4px solid #667eea;
      }
      
      /* Secciones con encabezados amigables */
      .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1e293b;
        margin: 2rem 0 1rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
      }
      
      .section-subheader {
        font-size: 1rem;
        color: #64748b;
        font-weight: 400;
        margin-bottom: 1.5rem;
      }
      
      /* Selectores más amigables */
      .stSelectbox label {
        font-weight: 600;
        color: #475569;
        font-size: 1rem;
      }
      
      /* Botones y elementos interactivos */
      .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
      }
      
      .stButton button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
      }
      
      /* Tarjeta informativa */
      .info-card {
        background: #f0f9ff;
        border-left: 4px solid #3b82f6;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
      }
      
      .info-card-title {
        font-weight: 600;
        color: #1e40af;
        margin-bottom: 0.5rem;
      }
      
      /* Animación sutil */
      @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
      }
      
      .animated {
        animation: fadeIn 0.5s ease-out;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# Constantes
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
DAY_ES = {
    "Monday": "Lunes", "Tuesday": "Martes", "Wednesday": "Miércoles",
    "Thursday": "Jueves", "Friday": "Viernes", "Saturday": "Sábado", "Sunday": "Domingo",
}

# Paleta de colores humanizada
COLOR_PALETTE = ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#43e97b', '#fa709a']

# -----------------------------
# Funciones de carga y procesamiento
# -----------------------------
@st.cache_data(show_spinner=True)
def load_data() -> pd.DataFrame:
    URL1 = "https://dl.dropboxusercontent.com/scl/fi/oni9yw8jyv2zh3e09ebgc/parte_1.csv?rlkey=4trd7syzk8yuoerz4angussmo"
    URL2 = "https://dl.dropboxusercontent.com/scl/fi/88mexprkeldc8d6g4wlob/parte_2.csv?rlkey=hjnbwcx63mavy3it7v5r7cjem"
    
    usecols = [
        "date", "store_nbr", "family", "sales", "onpromotion",
        "holiday_type", "locale", "locale_name", "transferred",
        "dcoilwtico", "city", "state", "store_type", "cluster",
        "transactions", "year", "month", "week", "quarter", "day_of_week",
    ]
    
    dtypes = {
        "store_nbr": "int16", "sales": "float32", "onpromotion": "int16",
        "dcoilwtico": "float32", "cluster": "int16", "transactions": "float32",
        "year": "int16", "month": "int8", "week": "int16", "quarter": "int8",
    }
    
    df1 = pd.read_csv(URL1, usecols=usecols, dtype=dtypes, parse_dates=["date"], low_memory=False)
    df2 = pd.read_csv(URL2, usecols=usecols, dtype=dtypes, parse_dates=["date"], low_memory=False)
    df = pd.concat([df1, df2], ignore_index=True)
    
    cat_cols = ["family", "holiday_type", "locale", "locale_name", "transferred",
                "city", "state", "store_type", "day_of_week"]
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype("category")
    
    return df

@st.cache_data(show_spinner=False)
def build_derived_tables(df: pd.DataFrame) -> dict:
    date_dim = df[["date", "year", "month", "week", "quarter", "day_of_week", 
                   "dcoilwtico", "holiday_type", "transferred"]].drop_duplicates("date")
    daily_sales = df.groupby("date", as_index=False)["sales"].sum().merge(date_dim, on="date", how="left")
    store_day = df[["date", "store_nbr", "transactions", "city", "state", 
                    "store_type", "cluster", "year"]].drop_duplicates(["date", "store_nbr"])
    store_day_sales = df.groupby(["date", "store_nbr"], as_index=False)["sales"].sum()
    promo = df[df["onpromotion"] > 0]
    promo_store_day_sales = promo.groupby(["date", "store_nbr"], as_index=False)["sales"].sum()
    
    return {
        "date_dim": date_dim, "daily_sales": daily_sales, "store_day": store_day,
        "store_day_sales": store_day_sales, "promo_store_day_sales": promo_store_day_sales,
    }

def format_int(x: float | int) -> str:
    try:
        return f"{int(x):,}".replace(",", ".")
    except Exception:
        return str(x)

def create_plotly_chart(fig, height=400):
    """Aplica estilo humanizado a los gráficos de Plotly"""
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=60, b=40),
        font=dict(family="Inter, sans-serif", size=12, color="#475569"),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        title_font=dict(size=16, color="#1e293b", family="Inter, sans-serif", weight=600),
        hovermode='x unified',
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Inter, sans-serif"),
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#e2e8f0', showline=True, linewidth=1, linecolor='#e2e8f0')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#e2e8f0', showline=True, linewidth=1, linecolor='#e2e8f0')
    return fig

# -----------------------------
# Interfaz principal
# -----------------------------
st.markdown("<h1 class='main-title'>🛒 Panel de Ventas Inteligente</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Explora tus datos de ventas de forma clara y conversacional</p>", unsafe_allow_html=True)

# Carga de datos
with st.spinner("⏳ Cargando tus datos..."):
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

# Pestañas
tabs = st.tabs(["🌍 Visión Global", "🏪 Por Tienda", "📍 Por Estado", "💡 Insights Extras"])

# -----------------------------
# Pestaña 1: Visión Global
# -----------------------------
with tabs[0]:
    st.markdown("<div class='section-header'>📊 Tu negocio en números</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-subheader'>Un vistazo rápido a las métricas clave de tu operación</div>", unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
        st.metric("🏪 Tiendas activas", format_int(n_stores))
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
        st.metric("📦 Líneas de producto", format_int(n_products))
        st.markdown("</div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
        st.metric("🗺️ Estados cubiertos", format_int(n_states))
        st.markdown("</div>", unsafe_allow_html=True)
    with c4:
        st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
        st.metric("📅 Meses de datos", format_int(n_months))
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown(f"<div class='muted'>📅 Analizando desde <b>{date_min}</b> hasta <b>{date_max}</b></div>", unsafe_allow_html=True)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    st.markdown("<div class='section-header'>📈 ¿Qué productos venden más?</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-subheader'>Descubre las estrellas de tu catálogo</div>", unsafe_allow_html=True)
    
    prod_mean = df.groupby("family", observed=True)["sales"].mean().sort_values(ascending=False).head(10).reset_index()
    prod_mean.rename(columns={"sales": "mean_sales"}, inplace=True)
    
    fig_prod = px.bar(prod_mean, x="mean_sales", y="family", orientation="h",
                      labels={"mean_sales": "Venta promedio", "family": "Producto"},
                      color_discrete_sequence=[COLOR_PALETTE[0]])
    fig_prod = create_plotly_chart(fig_prod, 420)
    st.plotly_chart(fig_prod, use_container_width=True)
    
    st.markdown("<div class='section-header'>🏬 Rendimiento por tienda</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-subheader'>¿Cuáles son tus tiendas más productivas?</div>", unsafe_allow_html=True)
    
    store_mean = store_day_sales.groupby("store_nbr", observed=True)["sales"].mean().reset_index()
    store_mean.rename(columns={"sales": "mean_daily_sales"}, inplace=True)
    store_mean = store_mean.sort_values("mean_daily_sales", ascending=False)
    store_mean["store_nbr"] = store_mean["store_nbr"].astype(str)
    
    fig_store_dist = px.bar(store_mean, x="store_nbr", y="mean_daily_sales",
                            labels={"store_nbr": "Tienda", "mean_daily_sales": "Venta diaria promedio"},
                            color_discrete_sequence=[COLOR_PALETTE[1]])
    fig_store_dist = create_plotly_chart(fig_store_dist, 420)
    st.plotly_chart(fig_store_dist, use_container_width=True)
    
    st.markdown("<div class='section-header'>🎯 Impacto de las promociones</div>", unsafe_allow_html=True)
    
    promo_store_mean = promo_store_day_sales.groupby("store_nbr", observed=True)["sales"].mean().reset_index()
    promo_store_mean.rename(columns={"sales": "mean_daily_promo_sales"}, inplace=True)
    promo_store_mean = promo_store_mean.sort_values("mean_daily_promo_sales", ascending=False).head(10)
    promo_store_mean["store_nbr"] = promo_store_mean["store_nbr"].astype(str)
    
    fig_promo_store = px.bar(promo_store_mean, x="store_nbr", y="mean_daily_promo_sales",
                             labels={"store_nbr": "Tienda", "mean_daily_promo_sales": "Ventas diarias en promo"},
                             color_discrete_sequence=[COLOR_PALETTE[2]])
    fig_promo_store = create_plotly_chart(fig_promo_store, 380)
    st.plotly_chart(fig_promo_store, use_container_width=True)
    
    st.markdown("<div class='section-header'>📅 Patrones semanales</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-subheader'>¿Qué días venden más tus tiendas?</div>", unsafe_allow_html=True)
    
    dow = daily_sales.groupby("day_of_week", observed=True)["sales"].mean().reindex(DAY_ORDER).reset_index()
    dow["day_es"] = dow["day_of_week"].map(DAY_ES).fillna(dow["day_of_week"].astype(str))
    best_dow = dow.loc[dow["sales"].idxmax(), "day_es"]
    best_val = dow["sales"].max()
    
    m1, m2 = st.columns([1, 3])
    with m1:
        st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
        st.metric("🏆 Mejor día", best_dow, f"{best_val:,.0f}".replace(",", ".") + " unidades")
        st.markdown("</div>", unsafe_allow_html=True)
    with m2:
        fig_dow = px.bar(dow, x="day_es", y="sales",
                        labels={"day_es": "Día", "sales": "Venta promedio diaria"},
                        color_discrete_sequence=[COLOR_PALETTE[3]])
        fig_dow = create_plotly_chart(fig_dow, 360)
        st.plotly_chart(fig_dow, use_container_width=True)
    
    st.markdown("<div class='section-header'>📊 Tendencias temporales</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        weekly_total = daily_sales.groupby(["year", "week"], as_index=False)["sales"].sum()
        weekly_avg = weekly_total.groupby("week", as_index=False)["sales"].mean()
        fig_week = px.line(weekly_avg, x="week", y="sales",
                          labels={"week": "Semana del año", "sales": "Venta semanal promedio"},
                          color_discrete_sequence=[COLOR_PALETTE[4]])
        fig_week = create_plotly_chart(fig_week, 360)
        st.plotly_chart(fig_week, use_container_width=True)
    
    with col2:
        monthly_total = daily_sales.groupby(["year", "month"], as_index=False)["sales"].sum()
        monthly_avg = monthly_total.groupby("month", as_index=False)["sales"].mean()
        fig_month = px.line(monthly_avg, x="month", y="sales",
                           labels={"month": "Mes", "sales": "Venta mensual promedio"},
                           color_discrete_sequence=[COLOR_PALETTE[5]])
        fig_month = create_plotly_chart(fig_month, 360)
        st.plotly_chart(fig_month, use_container_width=True)

# -----------------------------
# Pestaña 2: Por Tienda
# -----------------------------
with tabs[1]:
    st.markdown("<div class='section-header'>🏪 Análisis detallado por tienda</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-subheader'>Selecciona una tienda para ver su desempeño individual</div>", unsafe_allow_html=True)
    
    stores = sorted(df["store_nbr"].unique().tolist())
    store_sel = st.selectbox("Elige una tienda", stores, index=0)
    
    df_s = df[df["store_nbr"] == store_sel]
    store_sales_year = df_s.groupby("year", as_index=False)["sales"].sum().sort_values("year")
    
    total_units = float(df_s["sales"].sum())
    unique_products_sold = int(df_s.loc[df_s["sales"] > 0, "family"].nunique())
    unique_promo_products = int(df_s.loc[(df_s["sales"] > 0) & (df_s["onpromotion"] > 0), "family"].nunique())
    
    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
        st.metric("💰 Ventas totales", f"{total_units:,.0f}".replace(",", ".") + " unidades")
        st.markdown("</div>", unsafe_allow_html=True)
    with k2:
        st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
        st.metric("📦 Productos únicos vendidos", format_int(unique_products_sold))
        st.markdown("</div>", unsafe_allow_html=True)
    with k3:
        st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
        st.metric("🎯 Con promociones", format_int(unique_promo_products))
        st.markdown("</div>", unsafe_allow_html=True)
    
    fig_store_year = px.bar(store_sales_year, x="year", y="sales",
                            labels={"year": "Año", "sales": "Ventas totales"},
                            color_discrete_sequence=[COLOR_PALETTE[0]])
    fig_store_year = create_plotly_chart(fig_store_year, 420)
    st.plotly_chart(fig_store_year, use_container_width=True)

# -----------------------------
# Pestaña 3: Por Estado
# -----------------------------
with tabs[2]:
    st.markdown("<div class='section-header'>📍 Visión por estado</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-subheader'>Compara el rendimiento geográfico de tu negocio</div>", unsafe_allow_html=True)
    
    states = sorted(df["state"].dropna().unique().tolist())
    state_sel = st.selectbox("Selecciona un estado", states, index=0)
    
    sd_state = store_day[store_day["state"] == state_sel].copy()
    trans_year = sd_state.groupby("year", as_index=False)["transactions"].sum().sort_values("year")
    
    fig_trans_year = px.line(trans_year, x="year", y="transactions", markers=True,
                             labels={"year": "Año", "transactions": "Transacciones"},
                             color_discrete_sequence=[COLOR_PALETTE[1]])
    fig_trans_year = create_plotly_chart(fig_trans_year, 380)
    st.plotly_chart(fig_trans_year, use_container_width=True)
    
    df_state = df[df["state"] == state_sel]
    store_rank = df_state.groupby("store_nbr", observed=True)["sales"].sum().sort_values(ascending=False).head(10).reset_index()
    
    fig_store_rank = px.bar(store_rank, x="store_nbr", y="sales",
                           labels={"store_nbr": "Tienda", "sales": "Ventas totales"},
                           color_discrete_sequence=[COLOR_PALETTE[2]])
    fig_store_rank = create_plotly_chart(fig_store_rank, 380)
    st.plotly_chart(fig_store_rank, use_container_width=True)
    
    prod_rank = df_state.groupby("family", observed=True)["sales"].sum().sort_values(ascending=False).head(10).reset_index()
    top_prod = prod_rank.iloc[0]["family"] if len(prod_rank) else None
    top_units = prod_rank.iloc[0]["sales"] if len(prod_rank) else 0
    
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
        st.metric("🥇 Producto líder", str(top_prod), f"{top_units:,.0f}".replace(",", ".") + " unidades")
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        fig_prod_rank = px.bar(prod_rank, x="sales", y="family", orientation="h",
                              labels={"family": "Producto", "sales": "Ventas"},
                              color_discrete_sequence=[COLOR_PALETTE[3]])
        fig_prod_rank = create_plotly_chart(fig_prod_rank, 360)
        st.plotly_chart(fig_prod_rank, use_container_width=True)

# -----------------------------
# Pestaña 4: Insights Extra
# -----------------------------
with tabs[3]:
    st.markdown("<div class='section-header'>💡 Descubrimientos interesantes</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-subheader'>Hallazgos que te ayudarán a tomar mejores decisiones</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='info-card'><div class='info-card-title'>🎯 Efecto de las promociones</div>¿Cuánto impactan realmente las promociones en tus ventas diarias?</div>", unsafe_allow_html=True)
    
    promo_daily = df[df["onpromotion"] > 0].groupby("date", as_index=False)["sales"].sum().rename(columns={"sales": "promo_sales"})
    daily = daily_sales[["date", "sales"]].rename(columns={"sales": "total_sales"})
    promo_share = daily.merge(promo_daily, on="date", how="left")
    promo_share["promo_sales"] = promo_share["promo_sales"].fillna(0.0)
    promo_share["promo_share"] = np.where(promo_share["total_sales"] > 0, 
                                          promo_share["promo_sales"] / promo_share["total_sales"], 0.0)
    
    c1, c2 = st.columns(2)
    with c1:
        fig_share = px.line(promo_share, x="date", y="promo_share",
                           labels={"date": "Fecha", "promo_share": "% de ventas con promo"},
                           color_discrete_sequence=[COLOR_PALETTE[4]])
        fig_share = create_plotly_chart(fig_share, 360)
        st.plotly_chart(fig_share, use_container_width=True)
    with c2:
        fig_promo_vs = px.scatter(promo_share.sample(min(4000, len(promo_share)), random_state=7),
                                 x="promo_sales", y="total_sales",
                                 labels={"promo_sales": "Ventas en promo", "total_sales": "Ventas totales"},
                                 color_discrete_sequence=[COLOR_PALETTE[5]])
        fig_promo_vs = create_plotly_chart(fig_promo_vs, 360)
        st.plotly_chart(fig_promo_vs, use_container_width=True)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    st.markdown("<div class='info-card'><div class='info-card-title'>🎊 Días festivos vs días regulares</div>¿Aumentan las ventas en días festivos?</div>", unsafe_allow_html=True)
    
    ds = daily_sales.copy()
    ds["is_holiday"] = ds["holiday_type"].notna()
    holiday_mean = ds.groupby("is_holiday", as_index=False)["sales"].mean()
    holiday_mean["label"] = holiday_mean["is_holiday"].map({True: "Con festividad", False: "Día regular"})
    
    c1, c2 = st.columns([1, 2])
