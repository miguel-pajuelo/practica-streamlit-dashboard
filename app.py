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

# CSS personalizado para mejorar la estética y usabilidad del dashboard
st.markdown(
    """
    <style>
      .block-container { padding-top: 1.2rem; padding-bottom: 2.5rem; }
      [data-testid="stMetricValue"] { font-size: 1.7rem; }
      [data-testid="stMetricLabel"] { font-size: 0.95rem; }
      .stTabs [data-baseweb="tab"] { font-size: 1rem; }
      .kpi-card { border-radius: 14px; padding: 0.75rem 0.9rem; background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); }
      .muted { opacity: 0.8; font-size: 0.95rem; }
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
# Carga de datos
# -----------------------------
@st.cache_data(show_spinner=True)
def load_data() -> pd.DataFrame:
    """
    Carga el dataset desde dos archivos CSV (parte_1.csv y parte_2.csv) y los concatena en un solo DataFrame.
    
    - Optimiza los tipos de datos para reducir el uso de memoria y mejorar el rendimiento en agregaciones.
    - Convierte columnas de strings a categorías para ahorrar memoria en datasets grandes.
    - Permite sobrescribir las rutas de los archivos mediante variables de entorno (útil para despliegues).
    
    Returns:
        pd.DataFrame: DataFrame concatenado y optimizado con los datos de ventas.
    """
    base_dir = Path(__file__).parent

    p1 = base_dir / "parte_1.csv"
    p2 = base_dir / "parte_2.csv"

    # Sobrescribe rutas si se definen en variables de entorno
    p1_env = os.getenv("DATA_PATH_1")
    p2_env = os.getenv("DATA_PATH_2")
    if p1_env:
        p1 = Path(p1_env)
    if p2_env:
        p2 = Path(p2_env)

    # Columnas específicas a cargar para evitar datos innecesarios
    usecols = [
        "date", "store_nbr", "family", "sales", "onpromotion",
        "holiday_type", "locale", "locale_name", "transferred",
        "dcoilwtico", "city", "state", "store_type", "cluster",
        "transactions", "year", "month", "week", "quarter", "day_of_week",
    ]

    # Tipos de datos optimizados para reducir memoria
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

    # Carga de los CSV con optimizaciones
    df1 = pd.read_csv(p1, usecols=usecols, dtype=dtypes, parse_dates=["date"], low_memory=False)
    df2 = pd.read_csv(p2, usecols=usecols, dtype=dtypes, parse_dates=["date"], low_memory=False)

    # Concatenación de los DataFrames
    df = pd.concat([df1, df2], ignore_index=True)

    # Conversión de columnas categóricas para optimizar memoria
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
st.title("📈 Dashboard de Ventas")

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
tabs = st.tabs(["1) Global", "2) Por tienda", "3) Por estado", "4) Insights extra"])

# -----------------------------
# Pestaña 1: Visión Global
# -----------------------------
with tabs[0]:
    st.subheader("Visión global")

    # KPIs básicos globales
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tiendas", format_int(n_stores))
    c2.metric("Productos (families)", format_int(n_products))
    c3.metric("Estados", format_int(n_states))
    c4.metric("Meses con datos", format_int(n_months))
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
    fig_prod = px.bar(prod_mean, x="mean_sales", y="family", orientation="h",
                      title="Top 10 productos por venta media (unidades) por registro",
                      labels={"mean_sales": "Venta media", "family": "Producto"})
    fig_prod.update_layout(height=420, margin=dict(l=20, r=20, t=60, b=20))
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
        st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
        st.metric("Mejor día (media)", best_dow, f"{best_val:,.0f}".replace(",", "."))
        st.markdown("</div>", unsafe_allow_html=True)

    with m2:
        fig_dow = px.bar(
            dow,
            x="day_es",
            y="sales",
            title="Venta media diaria total por día de la semana",
            labels={"day_es": "Día", "sales": "Venta media diaria (total)"}
        )
        fig_dow.update_layout(height=360, margin=dict(l=20, r=20, t=60, b=20))
        st.plotly_chart(fig_dow, use_container_width=True)

    # v) Volumen medio por semana del año (promedio entre años)
    weekly_total = daily_sales.groupby(["year", "week"], as_index=False)["sales"].sum()
    weekly_avg = weekly_total.groupby("week", as_index=False)["sales"].mean()

    fig_week = px.line(
        weekly_avg,
        x="week",
        y="sales",
        title="Venta semanal media por semana del año (promedio entre años)",
        labels={"week": "Semana del año", "sales": "Venta semanal media"}
    )
    fig_week.update_layout(height=360, margin=dict(l=20, r=20, t=60, b=20))
    st.plotly_chart(fig_week, use_container_width=True)

    # vi) Volumen medio por mes (promedio entre años)
    monthly_total = daily_sales.groupby(["year", "month"], as_index=False)["sales"].sum()
    monthly_avg = monthly_total.groupby("month", as_index=False)["sales"].mean()
    fig_month = px.line(
        monthly_avg,
        x="month",
        y="sales",
        title="Venta mensual media por mes (promedio entre años)",
        labels={"month": "Mes", "sales": "Venta mensual media"}
    )
    fig_month.update_layout(height=360, margin=dict(l=20, r=20, t=60, b=20))
    st.plotly_chart(fig_month, use_container_width=True)


# -----------------------------
# Pestaña 2: Análisis por Tienda
# -----------------------------
with tabs[1]:
    st.subheader("Análisis por tienda")

    # Selector de tienda
    stores = sorted(df["store_nbr"].unique().tolist())
    store_sel = st.selectbox("Selecciona una tienda (store_nbr)", stores, index=0)

    # Filtrado de datos por tienda seleccionada
    df_s = df[df["store_nbr"] == store_sel]
    store_sales_year = df_s.groupby("year", as_index=False)["sales"].sum().sort_values("year")

    # KPIs específicos de la tienda
    total_units = float(df_s["sales"].sum())
    unique_products_sold = int(df_s.loc[df_s["sales"] > 0, "family"].nunique())
    unique_promo_products = int(df_s.loc[(df_s["sales"] > 0) & (df_s["onpromotion"] > 0), "family"].nunique())

    k1, k2, k3 = st.columns(3)
    k1.metric("Ventas totales (unidades)", f"{total_units:,.0f}".replace(",", "."))
    k2.metric("Productos vendidos (distinct)", format_int(unique_products_sold))
    k3.metric("Productos vendidos en promo (distinct)", format_int(unique_promo_products))

    # Gráfico de ventas por año para la tienda
    fig_store_year = px.bar(
        store_sales_year,
        x="year",
        y="sales",
        title=f"Ventas totales por año — tienda {store_sel}",
        labels={"year": "Año", "sales": "Ventas (unidades)"}
    )
    fig_store_year.update_layout(height=420, margin=dict(l=20, r=20, t=60, b=20))
    st.plotly_chart(fig_store_year, use_container_width=True)


# -----------------------------
# Pestaña 3: Análisis por Estado
# -----------------------------
with tabs[2]:
    st.subheader("Análisis por estado")

    # Selector de estado
    states = sorted(df["state"].dropna().unique().tolist())
    state_sel = st.selectbox("Selecciona un estado (state)", states, index=0)

    # a) Transacciones por año en el estado (deduplicadas)
    sd_state = store_day[store_day["state"] == state_sel].copy()
    trans_year = sd_state.groupby("year", as_index=False)["transactions"].sum().sort_values("year")

    fig_trans_year = px.line(
        trans_year,
        x="year",
        y="transactions",
        markers=True,
        title=f"Transacciones totales por año — {state_sel}",
        labels={"year": "Año", "transactions": "Transacciones"}
    )
    fig_trans_year.update_layout(height=380, margin=dict(l=20, r=20, t=60, b=20))
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
        labels={"store_nbr": "Tienda", "sales": "Ventas (unidades)"}
    )
    fig_store_rank.update_layout(height=380, margin=dict(l=20, r=20, t=60, b=20))
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
        st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
        st.metric("Producto #1 (estado)", str(top_prod), f"{top_units:,.0f}".replace(",", "."))
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        fig_prod_rank = px.bar(
            prod_rank,
            x="sales",
            y="family",
            orientation="h",
            title=f"Top 10 productos por ventas — {state_sel}",
            labels={"family": "Producto", "sales": "Ventas (unidades)"}
        )
        fig_prod_rank.update_layout(height=360, margin=dict(l=20, r=20, t=60, b=20))
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
            labels={"date": "Fecha", "promo_share": "Cuota promo"}
        )
        fig_share.update_layout(height=360, margin=dict(l=20, r=20, t=60, b=20))
        st.plotly_chart(fig_share, use_container_width=True)

    with c2:
        fig_promo_vs = px.scatter(
            promo_share.sample(min(4000, len(promo_share)), random_state=7),
            x="promo_sales",
            y="total_sales",
            title="Relación: ventas promo vs ventas totales (muestra)",
            labels={"promo_sales": "Ventas promo", "total_sales": "Ventas totales"}
        )
        fig_promo_vs.update_layout(height=360, margin=dict(l=20, r=20, t=60, b=20))
        st.plotly_chart(fig_promo_vs, use_container_width=True)

    st.divider()

    # 2) Efecto de holidays: comparación de ventas medias con vs sin holiday_type
    ds = daily_sales.copy()
    ds["is_holiday"] = ds["holiday_type"].notna()
    holiday_mean = ds.groupby("is_holiday", as_index=False)["sales"].mean()
    holiday_mean["label"] = holiday_mean["is_holiday"].map({True: "Día con holiday_type", False: "Día sin holiday_type"})

    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
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
            labels={"label": "", "sales": "Venta media diaria (total)"}
        )
        fig_holiday.update_layout(height=320, margin=dict(l=20, r=20, t=60, b=20))
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
        labels={"state": "Estado", "sales": "Ventas (unidades)"}
    )
    fig_state.update_layout(height=380, margin=dict(l=20, r=20, t=60, b=20))
    st.plotly_chart(fig_state, use_container_width=True)

    st.caption("Nota: 'transactions' se agrega sin duplicar por (fecha, tienda), porque el dataset repite ese valor por cada family.")