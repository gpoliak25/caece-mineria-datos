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
@st.cache_data(show_spinner="Cargando datos desde Google Drive...")
def cargar_datos():
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
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📂 Fuentes de Datos",
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
        de una institución bancaria portuguesa. El objetivo es predecir si el cliente
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
        train, _ = cargar_datos()
        st.subheader("Estadísticas Generales")
        st.metric("Registros Train", f"{len(train):,}")
        st.metric("Variables", str(train.shape[1]))
        conv = train["y"].value_counts()
        pct  = round(conv["yes"] / len(train) * 100, 1)
        st.metric("Tasa de conversión", f"{pct}%")

        st.subheader("Distribución variable objetivo")
        fig, ax = plt.subplots(figsize=(4, 3))
        conv.plot(kind="bar", ax=ax, edgecolor="white")
        ax.set_title("Distribución de clientes convertidos")
        ax.set_xlabel("Conversión")
        ax.set_xticklabels(["no", "yes"], rotation=0)
        ax.set_ylabel("Cantidad")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.caption("⚠️ Dataset desbalanceado: ~88% no suscribe")

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
