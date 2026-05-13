# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, classification_report,
    ConfusionMatrixDisplay, roc_curve, roc_auc_score
)

st.set_page_config(
    page_title="Captación de Clientes Bancarios",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Minería de Datos — Captación de Clientes Bancarios")
st.markdown(
    '<p style="font-size:1.15rem; margin-top:-0.6rem; margin-bottom:0.5rem;">'
    '<strong>TP Integrador · CAECE 2026</strong>'
    ' &nbsp;·&nbsp; Lic. Lorena López &amp; Lic. Gisela Poliak'
    ' &nbsp;|&nbsp; <a href="https://github.com/gpoliak25/caece-mineria-datos" target="_blank">📂 Ver código en GitHub</a>'
    '</p>',
    unsafe_allow_html=True
)

st.markdown('<meta name="google" content="notranslate">', unsafe_allow_html=True)

st.markdown("""
<style>
/* ── degradé pastel de fondo ── */
.stApp {
    background: linear-gradient(135deg,
        #fce4ec 0%,
        #e8eaf6 30%,
        #e0f7fa 65%,
        #f1f8e9 100%);
    background-attachment: fixed;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,
        #f3e5f5 0%,
        #e3f2fd 55%,
        #e8f5e9 100%) !important;
}
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.60);
    border-radius: 12px;
    padding: 0.6rem 1rem;
    border: 1px solid rgba(255,255,255,0.85);
}
/* ── columnas apiladas en mobile ── */
@media (max-width: 640px) {
    [data-testid="column"] {
        width: 100% !important;
        flex: 1 1 100% !important;
        min-width: 100% !important;
    }
    /* tabs: texto más chico */
    .stTabs [data-baseweb="tab"] {
        font-size: 0.72rem;
        padding: 6px 8px;
    }
    /* métricas más compactas */
    [data-testid="stMetricValue"] { font-size: 1.1rem; }
    [data-testid="stMetricLabel"] { font-size: 0.75rem; }
    /* botones full-width */
    .stButton > button,
    .stLinkButton > a { width: 100%; }
}
</style>
""", unsafe_allow_html=True)

# ── IDs de Google Drive ───────────────────────────────────────────────────────
DRIVE_FILES = {
    "banca_train.csv": "1LBBXWqeepoIDrP-lbCLoKIJGiN7XuQgK",
    "banca_test.csv":  "1_5moPuB3MZsHAHe4Iea47kfRTUu_t-6H",
}
DRIVE_FOLDER = "https://drive.google.com/drive/folders/1thsu2nqNoYj1s41ElEmf8klnW5TpcyBX?usp=sharing"

# ── carga de datos (cached) ──────────────────────────────────────────────────
@st.cache_data(show_spinner="Cargando datos...")
def cargar_datos():
    import os
    local_train = os.path.join(os.path.dirname(__file__), "banca_train.csv")
    local_test  = os.path.join(os.path.dirname(__file__), "banca_test.csv")
    if os.path.exists(local_train) and os.path.exists(local_test):
        train = pd.read_csv(local_train, sep=";")
        test  = pd.read_csv(local_test,  sep=";")
    else:
        url_train = f"https://drive.google.com/uc?id={DRIVE_FILES['banca_train.csv']}"
        url_test  = f"https://drive.google.com/uc?id={DRIVE_FILES['banca_test.csv']}"
        train = pd.read_csv(url_train, sep=";")
        test  = pd.read_csv(url_test,  sep=";")
    return train, test

@st.cache_data
def entrenar_modelo(max_depth, min_samples_split, min_samples_leaf,
                    criterion, usar_duration, class_weight="balanced"):
    train, test = cargar_datos()
    le = LabelEncoder()
    X_train = train.drop("y", axis=1)
    y_train = train["y"]
    X_test  = test.drop("y",  axis=1)
    y_test  = test["y"]

    if not usar_duration:
        X_train = X_train.drop("duration", axis=1)
        X_test  = X_test.drop("duration",  axis=1)

    X_total = pd.get_dummies(pd.concat([X_train, X_test], axis=0), drop_first=True)
    X_train_enc = X_total.iloc[:len(X_train), :]
    X_test_enc  = X_total.iloc[len(X_train):,  :]

    y_train_enc = le.fit_transform(y_train)
    y_test_enc  = le.transform(y_test)

    model = DecisionTreeClassifier(
        max_depth=max_depth if max_depth > 0 else None,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        criterion=criterion,
        class_weight=class_weight if class_weight != "none" else None,
        random_state=42
    )
    model.fit(X_train_enc, y_train_enc)
    y_pred  = model.predict(X_test_enc)
    y_prob  = model.predict_proba(X_test_enc)[:, 1]

    return model, X_train_enc, y_test_enc, y_pred, y_prob, le

# ── dialogs de previsualización ───────────────────────────────────────────────
@st.dialog("Vista previa — banca_train.csv", width="large")
def preview_train():
    train, _ = cargar_datos()
    st.caption(f"{len(train):,} filas · {train.shape[1]} columnas")
    st.dataframe(train.head(50), use_container_width=True)

@st.dialog("Vista previa — banca_test.csv", width="large")
def preview_test():
    _, test = cargar_datos()
    st.caption(f"{len(test):,} filas · {test.shape[1]} columnas")
    st.dataframe(test.head(50), use_container_width=True)

# ── tabs ─────────────────────────────────────────────────────────────────────
tab1, tab_eda, tab2, tab3, tab4, tab5 = st.tabs([
    "📂 Fuentes de Datos",
    "🔬 Análisis del Dataset",
    "📊 Resultados del Modelo",
    "🎛️ Dashboard Interactivo",
    "🔍 Explorar Datos",
    "📓 Notebook Colab"
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — FUENTES DE DATOS
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Fuentes de Datos")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Dataset: Bank Marketing (UCI)")
        st.markdown("""
        El dataset proviene de campañas de marketing directo (llamadas telefónicas)
        realizadas por una institución bancaria. El objetivo es predecir si el cliente
        suscribirá un depósito a plazo fijo.

        ☁️ [Carpeta Google Drive — TP Final Integrador](https://drive.google.com/drive/folders/1thsu2nqNoYj1s41ElEmf8klnW5TpcyBX?usp=sharing)
        """)

        st.subheader("Archivos de datos")
        with st.container(border=True):
            st.markdown("📄 **banca_train.csv** — entrenamiento (45.211 registros)")
            b1, b2 = st.columns(2)
            b1.link_button("⬇️ Descargar", "https://drive.google.com/uc?id=1LBBXWqeepoIDrP-lbCLoKIJGiN7XuQgK", use_container_width=True)
            if b2.button("👁️ Previsualizar", key="prev_train", use_container_width=True):
                preview_train()
        with st.container(border=True):
            st.markdown("📄 **banca_test.csv** — prueba (4.521 registros)")
            b3, b4 = st.columns(2)
            b3.link_button("⬇️ Descargar", "https://drive.google.com/uc?id=1_5moPuB3MZsHAHe4Iea47kfRTUu_t-6H", use_container_width=True)
            if b4.button("👁️ Previsualizar", key="prev_test", use_container_width=True):
                preview_test()

        st.subheader("Variables del Dataset")
        variables = pd.DataFrame({
            "Variable": ["age","job","marital","education","default","balance",
                         "housing","loan","contact","day","month","duration",
                         "campaign","pdays","previous","poutcome","y"],
            "Tipo": ["Numérica","Categórica","Categórica","Categórica","Binaria",
                     "Numérica","Binaria","Binaria","Categórica","Numérica",
                     "Categórica","Numérica*","Numérica","Numérica","Numérica",
                     "Categórica","Target"],
            "Descripción": [
                "Edad del cliente",
                "Tipo de trabajo",
                "Estado civil",
                "Nivel educativo",
                "¿Tiene crédito en default?",
                "Saldo promedio anual (€)",
                "¿Tiene préstamo hipotecario?",
                "¿Tiene préstamo personal?",
                "Tipo de contacto (celular/teléfono)",
                "Día del último contacto",
                "Mes del último contacto",
                "Duración del último contacto (seg) ⚠️ leakage",
                "N° de contactos en esta campaña",
                "Días desde el último contacto anterior",
                "N° de contactos anteriores",
                "Resultado de la campaña anterior",
                "¿Suscribió depósito? (yes/no)"
            ]
        })
        st.dataframe(variables, use_container_width=True, hide_index=True)

    with col2:
        train_t1, test_t1 = cargar_datos()
        conv_tr = train_t1["y"].value_counts()
        conv_te = test_t1["y"].value_counts()
        pct_tr  = round(conv_tr.get("yes", 0) / len(train_t1) * 100, 1)
        pct_te  = round(conv_te.get("yes", 0) / len(test_t1)  * 100, 1)

        st.markdown("""
        <style>
        .stat-card {
            background: rgba(255,255,255,0.55);
            border-radius: 14px;
            padding: 14px 18px;
            margin-bottom: 10px;
            border: 1px solid rgba(255,255,255,0.8);
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        }
        .stat-card .label {
            font-size: 0.72rem;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 2px;
        }
        .stat-row { display: flex; gap: 12px; margin-bottom: 10px; }
        .stat-half {
            flex: 1;
            background: rgba(255,255,255,0.55);
            border-radius: 14px;
            padding: 12px 14px;
            border: 1px solid rgba(255,255,255,0.8);
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        }
        .stat-half .badge {
            display: inline-block;
            font-size: 0.68rem;
            font-weight: 700;
            border-radius: 20px;
            padding: 2px 10px;
            margin-bottom: 8px;
            color: #fff;
        }
        .badge-train { background: #5c9bd6; }
        .badge-test  { background: #66bb6a; }
        .stat-half .val { font-size: 1.45rem; font-weight: 700; color: #333; line-height: 1.1; }
        .stat-half .sub { font-size: 0.70rem; color: #888; margin-top: 2px; }
        .progress-wrap { margin: 4px 0 8px; }
        .progress-bar {
            height: 7px; border-radius: 4px;
            background: linear-gradient(90deg, #f48fb1, #ce93d8);
        }
        </style>
        """, unsafe_allow_html=True)

        st.subheader("Estadísticas Generales")

        # Tarjetas de registros
        st.markdown(f"""
        <div class="stat-row">
          <div class="stat-half">
            <span class="badge badge-train">Train</span>
            <div class="val">📋 {len(train_t1):,}</div>
            <div class="sub">registros</div>
          </div>
          <div class="stat-half">
            <span class="badge badge-test">Test</span>
            <div class="val">📋 {len(test_t1):,}</div>
            <div class="sub">registros</div>
          </div>
        </div>
        <div class="stat-row">
          <div class="stat-half">
            <span class="badge badge-train">Train</span>
            <div class="val">🎯 {pct_tr}%</div>
            <div class="sub">tasa de conversión</div>
            <div class="progress-wrap">
              <div class="progress-bar" style="width:{pct_tr*3}%"></div>
            </div>
          </div>
          <div class="stat-half">
            <span class="badge badge-test">Test</span>
            <div class="val">🎯 {pct_te}%</div>
            <div class="sub">tasa de conversión</div>
            <div class="progress-wrap">
              <div class="progress-bar" style="width:{pct_te*3}%"></div>
            </div>
          </div>
        </div>
        <div class="stat-card">
          <div class="label">Variables en común</div>
          <div style="font-size:1.6rem;font-weight:700;color:#333">
            🔢 {train_t1.shape[1]}
          </div>
          <div style="font-size:0.72rem;color:#888">16 predictoras + 1 target (y)</div>
        </div>
        """, unsafe_allow_html=True)

        # Gráfico combinado: barras horizontales apiladas por dataset
        st.subheader("Distribución variable objetivo")
        no_tr  = conv_tr.get("no",  0)
        yes_tr = conv_tr.get("yes", 0)
        no_te  = conv_te.get("no",  0)
        yes_te = conv_te.get("yes", 0)

        fig_t1, ax_t1 = plt.subplots(figsize=(4.2, 2.4))
        fig_t1.patch.set_alpha(0)
        ax_t1.set_facecolor("none")
        datasets = ["Train", "Test"]
        totals   = [len(train_t1), len(test_t1)]
        nos      = [no_tr,  no_te]
        yess     = [yes_tr, yes_te]
        y_pos    = [1, 0]
        ax_t1.barh(y_pos, nos,  height=0.45, color="#90caf9", label="no",  left=0)
        ax_t1.barh(y_pos, yess, height=0.45, color="#a5d6a7", label="yes", left=nos)
        total_max = max(totals)
        min_w = total_max * 0.08  # umbral: segmento menor al 8% del total_max → etiqueta afuera
        for i, (tot, n, y) in enumerate(zip(totals, nos, yess)):
            # etiqueta "no"
            if n >= min_w:
                ax_t1.text(n / 2, y_pos[i], f"{n/tot*100:.1f}%",
                           ha="center", va="center", fontsize=8, color="#333")
            else:
                ax_t1.text(n / 2, y_pos[i] + 0.28, f"{n/tot*100:.1f}%",
                           ha="center", va="bottom", fontsize=7.5, color="#333")
            # etiqueta "yes"
            if y >= min_w:
                ax_t1.text(n + y / 2, y_pos[i], f"{y/tot*100:.1f}%",
                           ha="center", va="center", fontsize=8, color="#333")
            else:
                ax_t1.text(n + y, y_pos[i] + 0.225, f"{y/tot*100:.1f}%",
                           ha="right", va="bottom", fontsize=7.5, color="#333")
        ax_t1.set_yticks(y_pos)
        ax_t1.set_yticklabels(datasets, fontsize=10)
        ax_t1.set_xlabel("Registros", fontsize=9)
        ax_t1.set_title("Proporción no / yes por dataset", fontsize=10)
        ax_t1.legend(loc="lower right", fontsize=8)
        ax_t1.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig_t1)
        plt.close()

        st.caption("⚠️ Ambos datasets desbalanceados: ~88% no suscribe")

# ════════════════════════════════════════════════════════════════════════════
# TAB EDA — ANÁLISIS DEL DATASET (PASO 2)
# ════════════════════════════════════════════════════════════════════════════
with tab_eda:
    from scipy import stats as _stats

    st.header("🔬 Análisis del Dataset — Paso 2")
    st.caption("Basado en banca_test.csv (4.521 registros · 17 variables)")
    train_eda, test_eda = cargar_datos()
    df_eda = test_eda.copy()

    sub_dim, sub_cal, sub_tgt, sub_cor, sub_dec = st.tabs([
        "📐 Dimensiones y Variables",
        "❓ Calidad de Datos",
        "🎯 Variable Objetivo",
        "🔗 Correlaciones",
        "✅ Decisiones",
    ])

    # ── SUBTAB 1: Dimensiones y Variables ─────────────────────────────────────
    with sub_dim:
        st.subheader("Tamaño del Dataset")
        n_rows, n_cols = df_eda.shape
        mem_mb = df_eda.memory_usage(deep=True).sum() / 1024 ** 2
        dims_df = pd.DataFrame({
            "Dimensión":   ["Filas", "Columnas", "Memoria aproximada", "Celdas totales"],
            "Valor":       [f"{n_rows:,}", str(n_cols), f"≈ {mem_mb:.1f} MB", f"≈ {n_rows * n_cols:,}"],
            "Implicancia": [
                "Dataset pequeño-mediano; cabe en memoria sin problemas.",
                "Cantidad razonable; no requiere reducción de dimensionalidad.",
                "Permite cross-validation y grid search en segundos.",
                "Apto para algoritmos clásicos; no justifica deep learning.",
            ],
        })
        st.dataframe(dims_df, use_container_width=True, hide_index=True)

        st.subheader("Tipos de Variables")
        var_rows = []
        for col in df_eda.columns:
            if col == "y":
                vtype, nuniq, detalle = "🎯 Target", 2, "yes / no"
            elif pd.api.types.is_numeric_dtype(df_eda[col]):
                vtype  = "🔢 Numérica"
                nuniq  = df_eda[col].nunique()
                detalle = f"{df_eda[col].min():.0f} – {df_eda[col].max():.0f}  ({nuniq} únicos)"
            else:
                vtype  = "🔤 Categórica"
                nuniq  = df_eda[col].nunique()
                sample = ", ".join(df_eda[col].value_counts().head(3).index.tolist())
                detalle = f"{sample}{'…' if nuniq > 3 else ''}  ({nuniq} niveles)"
            nota = "⚠️ DATA LEAKAGE — excluida" if col == "duration" else ""
            var_rows.append({"Variable": col, "Tipo": vtype, "Rango / Niveles": detalle, "Nota": nota})
        st.dataframe(pd.DataFrame(var_rows), use_container_width=True, hide_index=True)

        st.subheader("⚠️ Data Leakage — Variable `duration`")
        st.error(
            "**`duration`** registra cuántos segundos duró la llamada al cliente. "
            "Cuando el modelo debe decidir **a quién llamar**, esa duración todavía no existe: "
            "se conoce recién después de realizar la llamada.\n\n"
            "- Correlación con target: **r = +0,40** (la más alta del dataset).\n"
            "- Con `duration` incluida: AUC > 0,90 — métricas infladas, modelo inútil en producción.\n"
            "- Con `duration` excluida: AUC ≈ 0,72 — performance real y desplegable.\n\n"
            "**Decisión:** `duration` se excluye de todos los modelos de este TP."
        )

    # ── SUBTAB 2: Calidad de Datos ─────────────────────────────────────────────
    with sub_cal:
        col_q1, col_q2 = st.columns(2)

        with col_q1:
            st.subheader("Faltantes encubiertos — categoría 'unknown'")
            tratamientos = {
                "poutcome": "MANTENER — cliente nuevo (informativo)",
                "contact":  "MANTENER — patrón con campañas antiguas",
                "education":"Imputar con moda o tratar como categoría",
                "job":      "Imputar con moda",
            }
            unk_rows = []
            for col in df_eda.select_dtypes(include="object").columns:
                n = (df_eda[col] == "unknown").sum()
                if n > 0:
                    unk_rows.append({
                        "Variable":    col,
                        "Cantidad":    n,
                        "%":           f"{n / len(df_eda) * 100:.1f}%",
                        "Tratamiento": tratamientos.get(col, "Evaluar"),
                    })
            st.dataframe(pd.DataFrame(unk_rows), use_container_width=True, hide_index=True)

            st.subheader("Faltante numérico — `pdays = −1`")
            n_pdays = (df_eda["pdays"] == -1).sum()
            st.info(
                f"**{n_pdays:,} registros ({n_pdays / len(df_eda) * 100:.0f}%)** tienen `pdays = −1` "
                f"(*cliente nunca contactado anteriormente*). Coincide con `poutcome = unknown`.\n\n"
                "**No es un faltante a imputar:** es información valiosa. "
                "Estrategia: crear flag `nunca_contactado = (pdays == −1)`."
            )

        with col_q2:
            st.subheader("Outliers — Regla del IQR")
            out_rows = []
            for col in df_eda.select_dtypes(include="number").columns:
                q1  = df_eda[col].quantile(0.25)
                q3  = df_eda[col].quantile(0.75)
                iqr = q3 - q1
                lim_sup = q3 + 1.5 * iqr
                lim_inf = q1 - 1.5 * iqr
                n_out = ((df_eda[col] > lim_sup) | (df_eda[col] < lim_inf)).sum()
                if n_out > 0:
                    out_rows.append({
                        "Variable":  col,
                        "Mediana":   f"{df_eda[col].median():.0f}",
                        "Lím. sup.": f"{lim_sup:.0f}",
                        "Outliers":  n_out,
                        "%":         f"{n_out / len(df_eda) * 100:.1f}%",
                    })
            st.dataframe(pd.DataFrame(out_rows), use_container_width=True, hide_index=True)
            st.caption("Para modelos basados en árboles los outliers no requieren tratamiento.")

            st.subheader("Boxplots")
            num_viz = [c for c in df_eda.select_dtypes(include="number").columns if c != "duration"]
            fig_box, axes_box = plt.subplots(1, len(num_viz), figsize=(14, 4))
            for i, col in enumerate(num_viz):
                axes_box[i].boxplot(
                    df_eda[col].dropna(), patch_artist=True,
                    boxprops=dict(facecolor="#b3e5fc"),
                    medianprops=dict(color="#0277bd", linewidth=2),
                )
                axes_box[i].set_title(col, fontsize=8)
                axes_box[i].tick_params(axis="y", labelsize=7)
            plt.suptitle("Variables numéricas (sin duration)", fontsize=9)
            plt.tight_layout()
            st.pyplot(fig_box)
            plt.close()

    # ── SUBTAB 3: Variable Objetivo ────────────────────────────────────────────
    with sub_tgt:
        col_t1, col_t2 = st.columns(2)
        counts_y = df_eda["y"].value_counts()
        ratio    = counts_y.max() / counts_y.min()

        with col_t1:
            st.subheader("Distribución del Target (y)")
            fig_pie, ax_pie = plt.subplots(figsize=(5, 5))
            ax_pie.pie(
                counts_y.values,
                labels=counts_y.index,
                autopct="%1.1f%%",
                colors=["#f48fb1", "#a5d6a7"],
                startangle=90,
                textprops={"fontsize": 13},
            )
            ax_pie.set_title("Distribución de la variable objetivo")
            plt.tight_layout()
            st.pyplot(fig_pie)
            plt.close()
            st.metric("Ratio de desbalance", f"{ratio:.1f} : 1",
                      help="Clase mayoritaria (no) vs minoritaria (yes)")

        with col_t2:
            st.subheader("Recuento por clase")
            st.dataframe(
                pd.DataFrame({
                    "Clase":      counts_y.index,
                    "Cantidad":   counts_y.values,
                    "Proporción": [f"{v / counts_y.sum() * 100:.2f}%" for v in counts_y.values],
                }),
                use_container_width=True, hide_index=True,
            )
            st.warning(
                "**Desbalance fuerte (ratio ≈ 7,7 : 1)**\n\n"
                "- La **accuracy** es una métrica **engañosa**: un modelo trivial que prediga "
                "siempre 'no' alcanza el 88,5% sin aportar ningún valor.\n"
                "- Se priorizarán: **AUC-ROC · F1 · Recall** sobre la clase positiva · **AUC-PR**.\n"
                "- La partición train/test debe ser **estratificada**.\n"
                "- Estrategias: `class_weight='balanced'`, SMOTE, ajuste del umbral de decisión."
            )

            st.subheader("Tasa de suscripción por mes")
            conv_month = (
                df_eda.groupby("month")["y"]
                .apply(lambda s: (s == "yes").mean() * 100)
                .sort_values(ascending=False)
            )
            fig_m, ax_m = plt.subplots(figsize=(6, 3))
            conv_month.plot(kind="bar", ax=ax_m, color="#a5d6a7", edgecolor="white")
            ax_m.set_ylabel("Tasa de conversión (%)")
            ax_m.set_title("% suscripción por mes")
            ax_m.set_xticklabels(ax_m.get_xticklabels(), rotation=30, ha="right")
            plt.tight_layout()
            st.pyplot(fig_m)
            plt.close()

    # ── SUBTAB 4: Correlaciones ────────────────────────────────────────────────
    with sub_cor:
        st.subheader("Correlación lineal entre variables numéricas (Pearson)")
        num_all = df_eda.select_dtypes(include="number").columns.tolist()
        corr_m  = df_eda[num_all].corr()
        fig_cor, ax_cor = plt.subplots(figsize=(9, 7))
        im_cor = ax_cor.imshow(corr_m.values, cmap="PiYG", vmin=-1, vmax=1)
        plt.colorbar(im_cor, ax=ax_cor)
        ax_cor.set_xticks(range(len(num_all)))
        ax_cor.set_yticks(range(len(num_all)))
        ax_cor.set_xticklabels(num_all, rotation=45, ha="right", fontsize=9)
        ax_cor.set_yticklabels(num_all, fontsize=9)
        for i in range(len(num_all)):
            for j in range(len(num_all)):
                ax_cor.text(j, i, f"{corr_m.iloc[i, j]:.2f}",
                            ha="center", va="center", fontsize=7)
        ax_cor.set_title("Correlación de Pearson — Variables numéricas")
        plt.tight_layout()
        st.pyplot(fig_cor)
        plt.close()

        col_c1, col_c2 = st.columns(2)

        with col_c1:
            st.subheader("Numérica vs. Target (Punto-Biserial)")
            y_bin = (df_eda["y"] == "yes").astype(int)
            pb_rows = []
            for col in [c for c in num_all]:
                r_pb, p_pb = _stats.pointbiserialr(df_eda[col].fillna(df_eda[col].median()), y_bin)
                pb_rows.append({"Variable": col, "r": round(r_pb, 3), "p-value": f"{p_pb:.2e}"})
            df_pb = pd.DataFrame(pb_rows).sort_values("r", key=abs, ascending=False)
            st.dataframe(df_pb, use_container_width=True, hide_index=True)
            st.caption("`duration` tiene r = +0,40 — la más predictiva, pero produce data leakage.")

        with col_c2:
            st.subheader("Categórica vs. Target (V de Cramér)")
            cat_cols_eda = [c for c in df_eda.select_dtypes(include="object").columns if c != "y"]
            cr_rows = []
            for col in cat_cols_eda:
                ct_eda = pd.crosstab(df_eda[col], df_eda["y"])
                chi2, p_chi, _, _ = _stats.chi2_contingency(ct_eda)
                n_chi = ct_eda.sum().sum()
                v = (chi2 / (n_chi * (min(ct_eda.shape) - 1))) ** 0.5
                cr_rows.append({"Variable": col, "V de Cramér": round(v, 3), "p-value": f"{p_chi:.2e}"})
            df_cr = pd.DataFrame(cr_rows).sort_values("V de Cramér", ascending=False)
            st.dataframe(df_cr, use_container_width=True, hide_index=True)
            st.caption("`poutcome` es la más informativa: éxito previo → 64% de conversión.")

    # ── SUBTAB 5: Decisiones ───────────────────────────────────────────────────
    with sub_dec:
        st.subheader("✅ Decisiones de Preprocesamiento — Paso 2")
        dec_rows = [
            {
                "N°": "1",
                "Decisión": "Excluir `duration` por data leakage",
                "Justificación": "Solo se conoce DESPUÉS de hacer la llamada; usarla en producción sería tramposo.",
            },
            {
                "N°": "2",
                "Decisión": "Codificar categóricas con One-Hot Encoding",
                "Justificación": "Se usa `handle_unknown='ignore'` para robustez ante categorías nuevas en el test.",
            },
            {
                "N°": "3",
                "Decisión": "Estandarizar numéricas solo para modelos lineales",
                "Justificación": "Los árboles no requieren escalado; aplica únicamente a regresión logística.",
            },
            {
                "N°": "4",
                "Decisión": "Tratar 'unknown' como categoría legítima",
                "Justificación": "No se imputa: aporta información (especialmente en poutcome).",
            },
            {
                "N°": "5",
                "Decisión": "Partición estratificada 70/30",
                "Justificación": "Train: 3.164 registros; Test: 1.357. Tasa de positivos preservada (≈ 11,5%).",
            },
            {
                "N°": "6",
                "Decisión": "NO eliminar outliers en este TP",
                "Justificación": "Variables con outliers (balance, campaign) se modelan con árboles/boosting robustos.",
            },
        ]
        st.dataframe(pd.DataFrame(dec_rows), use_container_width=True, hide_index=True)
        st.info("Estas decisiones están implementadas directamente en `entrenar_modelo()` — el resto de la app las aplica en todas las configuraciones.")

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — RESULTADOS DEL MODELO
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Resultados del Modelo — Árbol de Decisión")
    st.info(
        "Algoritmo: **CART (Gini)** · `max_depth=5` · `min_samples_leaf=20` · "
        "`class_weight='balanced'` · `duration` excluida (data leakage)."
    )

    model, X_train_enc, y_test_enc, y_pred, y_prob, le = entrenar_modelo(
        max_depth=5, min_samples_split=2, min_samples_leaf=20,
        criterion="gini", usar_duration=False, class_weight="balanced"
    )

    acc = accuracy_score(y_test_enc, y_pred)
    auc = roc_auc_score(y_test_enc, y_prob)
    cm2 = confusion_matrix(y_test_enc, y_pred)
    tn2, fp2, fn2, tp2 = cm2.ravel()
    f1_2 = (2 * tp2 / (2 * tp2 + fp2 + fn2)) if (2 * tp2 + fp2 + fn2) > 0 else 0

    c1, c2 = st.columns(2)
    c1.metric("Accuracy", f"{acc:.4f}")
    c2.metric("AUC-ROC", f"{auc:.4f}")
    c3, c4 = st.columns(2)
    c3.metric("F1-Score (Sí)", f"{f1_2:.4f}")
    c4.metric("Profundidad real", str(model.get_depth()))

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Matriz de Confusión")
        fig, ax = plt.subplots(figsize=(5, 4))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm2, display_labels=["No", "Sí"])
        disp.plot(ax=ax, colorbar=False, cmap="PuBu")
        ax.set_title("Matriz de Confusión (modelo controlado)")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_b:
        st.subheader("Curva ROC")
        fpr, tpr, _ = roc_curve(y_test_enc, y_prob)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, color="#90caf9", lw=2, label=f"AUC = {auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Curva ROC")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.subheader("Reporte de Clasificación")
    report = classification_report(y_test_enc, y_pred, target_names=["No", "Sí"], output_dict=True)
    st.dataframe(pd.DataFrame(report).T.round(4), use_container_width=True)

    st.subheader("Importancia de Variables (Top 15)")
    feat_imp = pd.Series(model.feature_importances_, index=X_train_enc.columns)
    feat_imp = feat_imp.sort_values(ascending=True).tail(15)
    fig, ax = plt.subplots(figsize=(7, 5))
    feat_imp.plot(kind="barh", ax=ax, color="#90caf9")
    ax.set_title("Importancia de variables")
    ax.set_xlabel("Importancia")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — DASHBOARD INTERACTIVO
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("🎛️ Dashboard Interactivo — Ajustá los Parámetros")

    with st.sidebar:
        st.header("⚙️ Parámetros del Modelo")
        usar_duration = st.toggle("Incluir variable 'duration'", value=False,
                                   help="Duration tiene data leakage — no se conoce antes de llamar al cliente")
        CRITERIOS = {
            "CART — Gini": "gini",
            "C4.5 — Entropy": "entropy",
        }
        criterio_label = st.selectbox(
            "Algoritmo / Criterio de división",
            list(CRITERIOS.keys()),
            help="CART usa Gini (índice de impureza) · C4.5 usa Entropy (ganancia de información)"
        )
        criterio = CRITERIOS[criterio_label]
        max_depth = st.slider("Profundidad máxima del árbol (0 = sin límite)", 0, 20, 5)
        min_split = st.slider("Mínimo de muestras para dividir un nodo", 2, 100, 2)
        min_leaf  = st.slider("Mínimo de muestras en una hoja", 1, 50, 20)
        class_weight_opt = st.selectbox(
            "Peso de clases",
            ["balanced", "none"],
            help="'balanced' compensa el desbalance 88% No / 12% Sí"
        )

    model_i, X_tr_i, y_te_i, y_pred_i, y_prob_i, le_i = entrenar_modelo(
        max_depth, min_split, min_leaf, criterio, usar_duration, class_weight_opt
    )

    acc_i = accuracy_score(y_te_i, y_pred_i)
    auc_i = roc_auc_score(y_te_i, y_prob_i)
    cm_i  = confusion_matrix(y_te_i, y_pred_i)
    tn, fp, fn, tp = cm_i.ravel()
    precision_i = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_i    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_i        = 2 * precision_i * recall_i / (precision_i + recall_i) if (precision_i + recall_i) > 0 else 0
    depth_real  = model_i.get_depth()

    # métricas — 2 filas de 3
    m1, m2, m3 = st.columns(3)
    m1.metric("Accuracy",   f"{acc_i:.4f}")
    m2.metric("AUC-ROC",    f"{auc_i:.4f}")
    m3.metric("Precision",  f"{precision_i:.4f}")
    m4, m5, m6 = st.columns(3)
    m4.metric("Recall",          f"{recall_i:.4f}")
    m5.metric("F1-Score",        f"{f1_i:.4f}")
    m6.metric("Profundidad real", str(depth_real))

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Matriz de Confusión")
        fig, ax = plt.subplots(figsize=(5, 4))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_i, display_labels=["No", "Sí"])
        disp.plot(ax=ax, colorbar=False, cmap="PuBu")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Curva ROC")
        fpr_i, tpr_i, _ = roc_curve(y_te_i, y_prob_i)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr_i, tpr_i, color="#ce93d8", lw=2, label=f"AUC = {auc_i:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Curva ROC")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Importancia de Variables (Top 15)")
        feat_imp_i = pd.Series(model_i.feature_importances_, index=X_tr_i.columns)
        feat_imp_i = feat_imp_i.sort_values(ascending=True).tail(15)
        fig, ax = plt.subplots(figsize=(6, 5))
        feat_imp_i.plot(kind="barh", ax=ax, color="#ce93d8")
        ax.set_xlabel("Importancia")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col4:
        st.subheader("Distribución de Predicciones")
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.hist(y_prob_i[y_te_i == 0], bins=40, alpha=0.6, label="Real: No", color="#f48fb1")
        ax.hist(y_prob_i[y_te_i == 1], bins=40, alpha=0.6, label="Real: Sí", color="#a5d6a7")
        ax.set_xlabel("Probabilidad predicha de suscripción")
        ax.set_ylabel("Frecuencia")
        ax.set_title("Separación de clases")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.subheader("Comparación de configuraciones")
    st.caption("Referencia: modelo controlado (max_depth=5, balanced, sin duration)")
    base_res  = entrenar_modelo(5, 2, 20, "gini", False, "balanced")
    y_te_b, y_pred_b = base_res[2], base_res[3]
    base_acc  = accuracy_score(y_te_b, y_pred_b)
    cm_b      = confusion_matrix(y_te_b, y_pred_b)
    tn_b, fp_b, fn_b, tp_b = cm_b.ravel()
    base_prec = tp_b / (tp_b + fp_b) if (tp_b + fp_b) > 0 else 0
    base_rec  = tp_b / (tp_b + fn_b) if (tp_b + fn_b) > 0 else 0
    base_f1   = 2 * base_prec * base_rec / (base_prec + base_rec) if (base_prec + base_rec) > 0 else 0

    cmp1, cmp2, cmp3, cmp4 = st.columns(4)
    cmp1.metric("Accuracy",  f"{acc_i:.4f}",       delta=f"{acc_i - base_acc:+.4f}")
    cmp2.metric("Precision", f"{precision_i:.4f}", delta=f"{precision_i - base_prec:+.4f}")
    cmp3.metric("Recall",    f"{recall_i:.4f}",    delta=f"{recall_i - base_rec:+.4f}")
    cmp4.metric("F-Measure", f"{f1_i:.4f}",        delta=f"{f1_i - base_f1:+.4f}")

# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — EXPLORAR DATOS
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("🔍 Explorar el Dataset")
    train, test = cargar_datos()

    subtab1, subtab2, subtab3 = st.tabs(["Vista previa", "Distribuciones", "Correlaciones"])

    with subtab1:
        dataset_sel = st.radio("Dataset", ["Train", "Test"], horizontal=True)
        df_sel = train if dataset_sel == "Train" else test
        st.dataframe(df_sel.head(100), use_container_width=True)
        st.caption(f"{len(df_sel):,} registros · {df_sel.shape[1]} variables")

    with subtab2:
        col_num = [c for c in train.columns if pd.api.types.is_numeric_dtype(train[c]) and c != "y"]
        col_cat = [c for c in train.columns if not pd.api.types.is_numeric_dtype(train[c]) and c != "y"]

        st.subheader("Variables numéricas")
        var_num = st.selectbox("Variable numérica", col_num)
        fig, axes = plt.subplots(2, 1, figsize=(7, 6))
        train[var_num].hist(bins=30, ax=axes[0], color="#90caf9", edgecolor="white")
        axes[0].set_title(f"Distribución de {var_num}")
        train.boxplot(column=var_num, by="y", ax=axes[1])
        axes[1].set_title(f"{var_num} por resultado")
        plt.suptitle("")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.subheader("Variables categóricas")
        var_cat = st.selectbox("Variable categórica", col_cat)
        ct = pd.crosstab(train[var_cat], train["y"], normalize="index") * 100
        fig, ax = plt.subplots(figsize=(10, 3))
        ct.plot(kind="bar", ax=ax, color=["#f48fb1","#a5d6a7"], edgecolor="white")
        ax.set_ylabel("% dentro de cada categoría")
        ax.set_title(f"Tasa de suscripción por {var_cat}")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
        ax.legend(["No", "Sí"])
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with subtab3:
        st.subheader("Correlación entre variables numéricas")
        num_cols = [c for c in train.columns if pd.api.types.is_numeric_dtype(train[c])]
        corr = train[num_cols].corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(corr, cmap="PiYG", vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(len(num_cols)))
        ax.set_yticks(range(len(num_cols)))
        ax.set_xticklabels(num_cols, rotation=45, ha="right")
        ax.set_yticklabels(num_cols)
        for i in range(len(num_cols)):
            for j in range(len(num_cols)):
                ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center", fontsize=7)
        ax.set_title("Mapa de correlaciones")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — NOTEBOOK COLAB
# ════════════════════════════════════════════════════════════════════════════
with tab5:
    import requests
    import nbformat
    import streamlit.components.v1 as components
    from nbconvert import HTMLExporter

    COLAB_URL   = "https://colab.research.google.com/drive/1iAVeSlS6_UWwYTl9sEHtu9stV-YnTsO9#scrollTo=bAKe4qOQ1ae6"
    NOTEBOOK_ID = "1iAVeSlS6_UWwYTl9sEHtu9stV-YnTsO9"

    @st.cache_data(ttl=3600, show_spinner="Descargando notebook desde Google Drive...")
    def obtener_html_notebook():
        url  = f"https://drive.google.com/uc?id={NOTEBOOK_ID}"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        nb   = nbformat.reads(resp.text, as_version=4)
        exp  = HTMLExporter(template_name="lab")
        html, _ = exp.from_notebook_node(nb)
        return html

    st.header("📓 Notebook — clientes.ipynb")
    st.markdown("Contenido descargado dinámicamente desde Google Drive · se refresca cada hora.")
    st.link_button("🚀 Abrir en Google Colab", COLAB_URL, use_container_width=True)

    try:
        html_nb = obtener_html_notebook()
        components.html(html_nb, height=1400, scrolling=True)
    except Exception as e:
        st.error(f"No se pudo cargar el notebook: {e}")
        st.link_button("🚀 Abrir en Google Colab", COLAB_URL)
