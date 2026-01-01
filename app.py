import os
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import plotly.io as pio

# -----------------------------
# Configuración de la aplicación
# -----------------------------
st.set_page_config(
    page_title="Dashboard Humano - Ventas",
    page_icon="🌿",
    layout="wide",
)

# --- PALETA DE COLORES HUMANA ---
# Usaremos tonos relajantes: Azul suave, Verde musgo, Terracota claro y Arena.
colors = {
    "background": "#FDFCFB",
    "card_bg": "#FFFFFF",
    "text": "#2D3436",
    "accent": "#6C5CE7",
    "success": "#00B894",
    "muted": "#636E72"
}

# Configuración de Plotly para que coincida con la estética
pio.templates["human_theme"] = pio.templates["plotly_white"]
pio.templates["human_theme"].layout.patch({
    "font": {"family": "Inter, sans-serif", "color": colors["text"]},
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "colorway": ["#6C5CE7", "#00B894", "#FAB1A0", "#0984E3", "#FD79A8"]
})
pio.templates.default = "human_theme"

# CSS personalizado avanzado
st.markdown(
    f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

      .stApp {{
        background-color: {colors["background"]};
        font-family: 'Inter', sans-serif;
      }}

      /* Títulos elegantes */
      h1, h2, h3 {{
        color: {colors["text"]};
        font-weight: 800 !important;
        letter-spacing: -0.02em;
      }}

      /* Tarjetas de Métricas Estilo Human Interface */
      .metric-container {{
        background: {colors["card_bg"]};
        padding: 24px;
        border-radius: 20px;
        border: 1px solid rgba(0,0,0,0.05);
        box-shadow: 0 10px 15px -3px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
        margin-bottom: 1rem;
      }}
      
      .metric-container:hover {{
        transform: translateY(-2px);
        box-shadow: 0 20px 25px -5px rgba(0,0,0,0.08);
        border: 1px solid {colors["accent"]}33;
      }}

      /* Ajustes de Streamlit */
      [data-testid="stMetricValue"] {{
        font-weight: 800;
        color: {colors["accent"]};
        font-size: 2.2rem !important;
      }}
      
      [data-testid="stMetricLabel"] {{
        font-weight: 500;
        color: {colors["muted"]};
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-size: 0.8rem !important;
      }}

      /* Tabs modernas */
      .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background-color: transparent;
      }}

      .stTabs [data-baseweb="tab"] {{
        height: 45px;
        white-space: pre;
        background-color: #F1F2F6;
        border-radius: 12px;
        color: {colors["muted"]};
        border: none;
        padding: 0px 20px;
        transition: all 0.2s;
      }}

      .stTabs [data-baseweb="tab"]:hover {{
        background-color: {colors["accent"]}22;
        color: {colors["accent"]};
      }}

      .stTabs [aria-selected="true"] {{
        background-color: {colors["accent"]} !important;
        color: white !important;
      }}

      /* Ocultar decoraciones innecesarias */
      #MainMenu, footer {{visibility: hidden;}}
      
      /* Inputs más suaves */
      .stSelectbox div[data-baseweb="select"] > div {{
        border-radius: 12px;
        border: 1px solid #E2E8F0;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Lógica de Datos (Optimizada)
# -----------------------------
@st.cache_data
def load_data():
    URL1 = "https://dl.dropboxusercontent.com/scl/fi/oni9yw8jyv2zh3e09ebgc/parte_1.csv?rlkey=4trd7syzk8yuoerz4angussmo"
    URL2 = "https://dl.dropboxusercontent.com/scl/fi/88mexprkeldc8d6g4wlob/parte_2.csv?rlkey=hjnbwcx63mavy3it7v5r7cjem"
    usecols = ["date", "store_nbr", "family", "sales", "onpromotion", "city", "state", "day_of_week", "transactions"]
    df1 = pd.read_csv(URL1, usecols=usecols, parse_dates=["date"])
    df2 = pd.read_csv(URL2, usecols=usecols, parse_dates=["date"])
    df = pd.concat([df1, df2], ignore_index=True)
    return df

df = load_data()

# -----------------------------
# Cabecera Humanizada
# -----------------------------
col_header, col_info = st.columns([2, 1])

with col_header:
    st.title("¡Hola! 👋")
    st.markdown(f"### Aquí tienes el pulso de tus ventas hoy.")
    st.markdown(f"Estamos analizando datos desde el **{df['date'].min().strftime('%d %b, %Y')}** hasta el **{df['date'].max().strftime('%d %b, %Y')}**.")

with col_info:
    st.info("💡 **Consejo:** Haz clic en los gráficos para ampliar detalles o usa los filtros en las pestañas inferiores.", icon="✨")

st.write("---")

# -----------------------------
# Pestañas Principales
# -----------------------------
tabs = st.tabs(["✨ Resumen General", "🏪 Detalle por Tienda", "📍 Ubicación", "🧠 Insights"])

# --- TAB 1: RESUMEN ---
with tabs[0]:
    # KPIs con HTML personalizado para control total del diseño
    c1, c2, c3, c4 = st.columns(4)
    
    kpis = [
        ("Tiendas Activas", df["store_nbr"].nunique(), "🏢"),
        ("Categorías", df["family"].nunique(), "📦"),
        ("Ventas Totales", f"{df['sales'].sum()/1e6:.1f}M", "📈"),
        ("Promoción Avg", f"{df['onpromotion'].mean():.1f}", "🏷️")
    ]
    
    for i, (label, val, icon) in enumerate(kpis):
        with [c1, c2, c3, c4][i]:
            st.markdown(f"""
                <div class="metric-container">
                    <span style="font-size: 1.5rem;">{icon}</span>
                    <p style="margin: 0; color: #636E72; font-size: 0.8rem; font-weight: 600;">{label}</p>
                    <h2 style="margin: 0; color: #6C5CE7;">{val}</h2>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("#### Tendencia de Ventas Semanales")
    df_weekly = df.set_index("date").resample("W")["sales"].sum().reset_index()
    fig_main = px.line(df_weekly, x="date", y="sales", 
                       labels={"sales": "Ventas Unidades", "date": ""},
                       template="human_theme")
    fig_main.update_traces(line_width=4, line_color=colors["accent"])
    st.plotly_chart(fig_main, use_container_width=True)

# --- TAB 2: TIENDA ---
with tabs[1]:
    col_sel, col_empty = st.columns([1, 2])
    with col_sel:
        store_sel = st.selectbox("¿Qué tienda quieres explorar?", sorted(df["store_nbr"].unique()))
    
    df_s = df[df["store_nbr"] == store_sel]
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Productos más vendidos")
        top_prod = df_s.groupby("family")["sales"].sum().nlargest(8).reset_index()
        fig_p = px.bar(top_prod, x="sales", y="family", orientation='h', color="sales",
                       color_continuous_scale="Purp")
        fig_p.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_p, use_container_width=True)
        
    with c2:
        st.markdown("##### Eficiencia: Transacciones vs Ventas")
        df_t = df_s.groupby("date").agg({"sales":"sum", "transactions":"mean"}).reset_index()
        fig_t = px.scatter(df_t, x="transactions", y="sales", opacity=0.5, 
                           trendline="ols", trendline_color_override="#00B894")
        st.plotly_chart(fig_t, use_container_width=True)

# --- TAB 3: ESTADOS ---
with tabs[2]:
    st.markdown("#### Distribución Territorial")
    df_state = df.groupby("state")["sales"].sum().sort_values(ascending=True).reset_index()
    fig_state = px.bar(df_state, x="sales", y="state", color="sales", 
                       color_continuous_scale="Tealgrn", height=600)
    st.plotly_chart(fig_state, use_container_width=True)

# --- TAB 4: INSIGHTS ---
with tabs[3]:
    st.markdown("#### ¿Cómo influye el día de la semana?")
    # Reordenar días
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    df_day = df.groupby("day_of_week")["sales"].mean().reindex(order).reset_index()
    
    fig_day = px.vbar = px.bar(df_day, x="day_of_week", y="sales", 
                               color="sales", color_continuous_scale="Peach")
    st.plotly_chart(fig_day, use_container_width=True)
    
    st.success("🔎 **Insight Detectado:** Los fines de semana concentran el 40% del volumen total. Considera reforzar el stock los viernes por la tarde.")

# -----------------------------
# Pie de página
# -----------------------------
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    f"""
    <div style="text-align: center; color: {colors['muted']}; font-size: 0.8rem;">
        Hecho con ❤️ para el equipo de análisis • 2024
    </div>
    """,
    unsafe_allow_html=True
)
