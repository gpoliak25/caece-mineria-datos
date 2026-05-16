# =============================================================
#  ¿A QUIÉN LLAMAR? — Captación de Clientes Bancarios
#  Aplicación de storytelling · TP de Minería de Datos
#  VERSIÓN COMPLETA (Etapa 1 + Etapa 2)
# =============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score,
                             classification_report, roc_curve, roc_auc_score)
from sklearn.model_selection import train_test_split

# -------------------------------------------------------------
# CONFIGURACIÓN DE LA PÁGINA
# -------------------------------------------------------------
st.set_page_config(
    page_title="¿A quién llamar? · Banco",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paleta de colores del proyecto
AZUL = "#1e3a8a"
VERDE = "#16a34a"
AMBAR = "#d97706"
ROJO = "#dc2626"

# -------------------------------------------------------------
# CARGA DE DATOS (cacheada: se ejecuta una sola vez)
# -------------------------------------------------------------
@st.cache_data
def cargar_datos():
    df = pd.read_csv("bancaV1_actualizado.csv", sep=";")
    return df

# -------------------------------------------------------------
# ENTRENAMIENTO DE MODELOS (cacheado: se ejecuta una sola vez)
# -------------------------------------------------------------
@st.cache_resource
def entrenar_modelos(df):
    X = df.drop("y", axis=1)
    y = df["y"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)

    # Eliminar 'duration' (data leakage)
    X_train = X_train.drop("duration", axis=1)
    X_test = X_test.drop("duration", axis=1)

    # One-Hot Encoding
    X_total = pd.concat([X_train, X_test], axis=0)
    X_total = pd.get_dummies(X_total, drop_first=True)
    X_train_enc = X_total.iloc[:len(X_train), :]
    X_test_enc = X_total.iloc[len(X_train):, :]

    # Codificar variable objetivo
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    # Modelo CART controlado
    modelo = DecisionTreeClassifier(
        max_depth=5, min_samples_leaf=20,
        class_weight="balanced", random_state=42)
    modelo.fit(X_train_enc, y_train_enc)

    # Modelo Random Forest
    modelo_rf = RandomForestClassifier(
        n_estimators=100, max_depth=5, min_samples_leaf=20,
        class_weight="balanced", random_state=42)
    modelo_rf.fit(X_train_enc, y_train_enc)

    # Predicciones sobre el conjunto de prueba
    y_pred = modelo.predict(X_test_enc)
    y_prob = modelo.predict_proba(X_test_enc)[:, 1]
    y_pred_rf = modelo_rf.predict(X_test_enc)
    y_prob_rf = modelo_rf.predict_proba(X_test_enc)[:, 1]

    return {
        "X_train_enc": X_train_enc, "X_test_enc": X_test_enc,
        "y_train_enc": y_train_enc, "y_test_enc": y_test_enc,
        "modelo": modelo, "modelo_rf": modelo_rf,
        "y_pred": y_pred, "y_prob": y_prob,
        "y_pred_rf": y_pred_rf, "y_prob_rf": y_prob_rf,
    }

# -------------------------------------------------------------
# CARGAR TODO
# -------------------------------------------------------------
df = cargar_datos()
R = entrenar_modelos(df)

# Columnas predictoras (todas menos 'y' y 'duration')
FEATURES = [c for c in df.columns if c not in ("y", "duration")]
NUMERICAS = list(df[FEATURES].select_dtypes(include="number").columns)
CATEGORICAS = [c for c in FEATURES if c not in NUMERICAS]

# -------------------------------------------------------------
# MENÚ LATERAL
# -------------------------------------------------------------
with st.sidebar:
    st.title("🏦 TP Minería de Datos")
    st.markdown("**Tema:** Captación de clientes bancarios")
    st.markdown("**Modelos:** CART y Random Forest")
    st.markdown("**Integrantes:** Lic. Lorena Lopez · Lic. Gisela Poliak")
    st.divider()
    st.markdown("### La pregunta del proyecto")
    st.info("¿A quién debería llamar el banco para no desperdiciar "
            "sus campañas?")
    st.divider()
    st.caption("Aplicación desarrollada con Streamlit.")

# -------------------------------------------------------------
# TÍTULO PRINCIPAL
# -------------------------------------------------------------
st.title("🏦 ¿A quién llamar?")
st.markdown("##### Un modelo predictivo para la captación de clientes bancarios")
st.divider()

# -------------------------------------------------------------
# CAPÍTULOS (SOLAPAS)
# -------------------------------------------------------------
tabs = st.tabs([
    "🏦 El desafío",
    "👥 Los clientes",
    "🔍 Lo oculto",
    "⚠️ La trampa",
    "🤖 El modelo",
    "📈 ¿Qué tan bueno es?",
    "🎯 Probá el modelo",
    "💡 Simulador",
    "✅ Conclusión",
])

# =============================================================
# CAPÍTULO 0 — EL DESAFÍO
# =============================================================
with tabs[0]:
    st.header("El banco tiene un problema de plata")
    st.markdown("""
    Cada vez que el banco lanza una campaña, llama a miles de clientes para
    ofrecerles un producto. Pero **solo 12 de cada 100 personas aceptan**.
    Las otras 88 llamadas son tiempo y dinero perdidos.
    """)

    c1, c2, c3 = st.columns(3)
    c1.metric("Clientes en el dataset", f"{df.shape[0]:,}")
    c2.metric("Tasa de conversión actual", "~12 %")
    c3.metric("Llamadas desperdiciadas", "~88 %")

    st.divider()
    st.subheader("🎯 La pregunta de este trabajo")
    st.markdown("""
    ### ¿Podemos predecir a quién llamar, y dejar de desperdiciar
    ### 88 llamadas de cada 100?
    """)
    st.success("En las próximas pestañas vamos a construir, paso a paso, "
               "un modelo que lo logra. Recorrelas en orden, de izquierda "
               "a derecha.")

# =============================================================
# CAPÍTULO 1 — LOS CLIENTES
# =============================================================
with tabs[1]:
    st.caption("*Antes de predecir nada, conozcamos a quién le estamos hablando.*")
    st.header("👥 Conociendo a los clientes")

    st.subheader("Archivos de origen")
    archivos = pd.DataFrame({
        "Archivo": ["banca_train.csv", "banca_test.csv",
                    "bancaV1_actualizado.csv"],
        "Descripción": ["Conjunto de entrenamiento original",
                        "Conjunto de prueba original",
                        "Dataset unificado y actualizado (usado en el análisis)"],
    })
    st.table(archivos)

    st.subheader("Primeras filas del dataset")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Estructura del dataset")
    info = pd.DataFrame({
        "Columna": df.columns,
        "Tipo de dato": df.dtypes.astype(str),
        "Valores no nulos": df.notnull().sum().values,
    })
    st.dataframe(info, use_container_width=True)

    st.subheader("Estadísticas descriptivas (variables numéricas)")
    st.dataframe(df.describe(), use_container_width=True)

    st.info("💡 **¿Qué nos dice esto?** El banco trabaja con perfiles muy "
            "diversos: clientes jóvenes y mayores, con ahorros y en descubierto "
            "(saldo negativo). Un mismo guion de llamada no le sirve a todos.")

# =============================================================
# CAPÍTULO 2 — LO QUE LOS DATOS ESCONDÍAN
# =============================================================
with tabs[2]:
    st.caption("*Los datos rara vez vienen limpios. Esto es lo que "
               "encontramos al mirarlos de cerca.*")
    st.header("🔍 Lo que los datos escondían")

    st.subheader("Faltantes codificados como categoría 'unknown'")
    st.markdown("Antes de modelar se analizaron los valores faltantes que el "
                "dataset traía codificados como la categoría `unknown`.")
    faltantes = pd.DataFrame({
        "Variable": ["poutcome", "contact", "education", "job"],
        "Cantidad": ["3.705", "1.324", "187", "38"],
        "Porcentaje": ["82,0 %", "29,3 %", "4,1 %", "0,8 %"],
        "Tratamiento": [
            "Mantener como categoría (informativo: cliente nuevo).",
            "Mantener como categoría; ver patrón con campañas antiguas.",
            "Imputar con la moda o tratar como categoría.",
            "Imputar con la moda."],
    })
    st.table(faltantes)

    st.subheader("Verificación de valores nulos")
    nulos = df.isnull().sum().reset_index()
    nulos.columns = ["Columna", "Valores nulos"]
    st.dataframe(nulos, use_container_width=True)

    st.subheader("La variable objetivo: ¿cuántos clientes convierten?")
    conteo = df["y"].value_counts()
    porcentaje = df["y"].value_counts(normalize=True) * 100

    c1, c2 = st.columns(2)
    c1.metric("Clase 'No' (no convierte)",
              f"{conteo.get('no', 0):,}", f"{porcentaje.get('no', 0):.1f} %")
    c2.metric("Clase 'Sí' (convierte)",
              f"{conteo.get('yes', 0):,}", f"{porcentaje.get('yes', 0):.1f} %")

    fig = px.bar(x=conteo.index, y=conteo.values,
                 labels={"x": "Conversión", "y": "Cantidad de clientes"},
                 color=conteo.index,
                 color_discrete_sequence=[AZUL, VERDE],
                 title="Distribución de clientes convertidos")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.warning("⚠️ **Atención:** solo ~12 % de los clientes convierte. Este "
               "desbalance condiciona todo lo que viene: un modelo que dijera "
               "'nadie convierte' acertaría el 88 % de las veces... y sería "
               "inútil. Vamos a necesitar métricas más honestas que el simple "
               "porcentaje de aciertos.")

# =============================================================
# CAPÍTULO 3 — LA TRAMPA EN LOS DATOS
# =============================================================
with tabs[3]:
    st.caption("*Los datos rara vez vienen sin sorpresas...*")
    st.header("⚠️ Una trampa en los datos")

    st.markdown("""
    Entre las variables había una llamada **`duration`** (cuánto duró la
    llamada). Era un predictor casi perfecto... **demasiado perfecto.**

    El problema: la duración de la llamada solo se conoce **después** de
    hacerla. Usarla para decidir a quién llamar sería como predecir quién gana
    un partido... mirando el resultado final. Esto se llama **data leakage**
    (fuga de información).
    """)

    st.error("🚫 **Decisión:** eliminamos `duration` del modelo. Preferimos un "
             "modelo más modesto pero **honesto y usable en la realidad**.")

    st.divider()
    st.subheader("Preparación de los datos para el modelo")

    c1, c2 = st.columns(2)
    c1.metric("Filas de entrenamiento (70 %)",
              f"{R['X_train_enc'].shape[0]:,}")
    c2.metric("Filas de prueba (30 %)",
              f"{R['X_test_enc'].shape[0]:,}")

    st.markdown("""
    - **División 70/30** con `stratify`, para mantener la proporción de clases.
    - **One-Hot Encoding** de las variables categóricas (job, marital,
      education, etc.), en lugar de `LabelEncoder`, para no introducir un orden
      falso entre categorías.
    """)
    st.metric("Variables tras la codificación", R["X_train_enc"].shape[1])

    st.info("💡 **¿Qué nos dice esto?** Detectar el data leakage es lo que "
            "separa un modelo de laboratorio de uno que el banco podría usar "
            "de verdad.")

# =============================================================
# CAPÍTULO 4 — EL MODELO
# =============================================================
with tabs[4]:
    st.caption("*El primer intento no siempre es el bueno.*")
    st.header("🤖 Construyendo el modelo")

    st.subheader("El primer modelo nos engañó")
    st.markdown("""
    Entrenamos un primer árbol de decisión **sin restricciones**. El resultado
    fue espectacular: **100 % de aciertos.**

    Pero era una ilusión. El árbol no había *aprendido*, había **memorizado**
    cada cliente del entrenamiento — como un alumno que se sabe las respuestas
    de memoria y reprueba con preguntas nuevas. Esto se llama **overfitting**.
    """)
    st.warning("⚠️ Un 100 % de aciertos casi nunca es buena noticia: suele ser "
               "la señal de un modelo que no sabe generalizar.")

    st.divider()
    st.subheader("La solución: dos modelos controlados")

    st.markdown("**🌳 Modelo 1 — Árbol de decisión controlado (CART)**")
    st.markdown("""
    - `max_depth=5` — le ponemos un límite de profundidad para que no memorice.
    - `min_samples_leaf=20` — cada decisión se basa en al menos 20 clientes.
    - `class_weight='balanced'` — más peso a la clase minoritaria (los que sí
      convierten).
    """)

    st.markdown("**🌲 Modelo 2 — Random Forest**")
    st.markdown("""
    En vez de confiar en un solo árbol, consultamos a **100 árboles** y
    promediamos sus votos. Como pedir una segunda (y centésima) opinión. Usa
    los mismos hiperparámetros que el CART para una comparación justa.
    """)
    c1, c2, c3 = st.columns(3)
    c1.metric("N° de árboles", 100)
    c2.metric("Profundidad máx.", 5)
    c3.metric("Mín. clientes por hoja", 20)

    st.info("💡 **¿Qué nos dice esto?** Pasamos de un modelo que se lucía y "
            "fallaba, a dos modelos honestos. Ahora toca medir cuál es mejor.")

# =============================================================
# CAPÍTULO 5 — EVALUACIÓN
# =============================================================
with tabs[5]:
    st.caption("*Un modelo no vale por lo que promete, sino por lo que "
               "acierta con datos que nunca vio.*")
    st.header("📈 ¿Qué tan bueno es?")

    y_test = R["y_test_enc"]

    def matriz_confusion(cm, titulo, color):
        etiquetas = ["No", "Sí"]
        fig = px.imshow(cm, text_auto=True,
                        x=[f"Predijo {e}" for e in etiquetas],
                        y=[f"Real {e}" for e in etiquetas],
                        color_continuous_scale=color, title=titulo)
        fig.update_layout(coloraxis_showscale=False)
        return fig

    sub = st.tabs(["🌳 CART", "🌲 Random Forest"])

    # --- CART ---
    with sub[0]:
        cm = confusion_matrix(y_test, R["y_pred"])
        st.plotly_chart(matriz_confusion(cm, "Matriz de confusión — CART",
                                         "Blues"), use_container_width=True)
        c1, c2 = st.columns(2)
        c1.metric("Accuracy", f"{accuracy_score(y_test, R['y_pred']):.4f}")
        c2.metric("AUC", f"{roc_auc_score(y_test, R['y_prob']):.4f}")

        fpr, tpr, _ = roc_curve(y_test, R["y_prob"])
        figroc = go.Figure()
        figroc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                         name="CART", line=dict(color=AZUL, width=3)))
        figroc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                         name="Azar", line=dict(color="gray", dash="dash")))
        figroc.update_layout(title="Curva ROC — CART",
                             xaxis_title="Falsos positivos",
                             yaxis_title="Verdaderos positivos")
        st.plotly_chart(figroc, use_container_width=True)

        feat = pd.Series(R["modelo"].feature_importances_,
                         index=R["X_test_enc"].columns).sort_values().tail(15)
        figf = px.bar(x=feat.values, y=feat.index, orientation="h",
                      labels={"x": "Importancia", "y": ""},
                      title="Top 15 variables más importantes — CART",
                      color_discrete_sequence=[AZUL])
        st.plotly_chart(figf, use_container_width=True)

    # --- Random Forest ---
    with sub[1]:
        cm_rf = confusion_matrix(y_test, R["y_pred_rf"])
        st.plotly_chart(matriz_confusion(cm_rf,
                        "Matriz de confusión — Random Forest", "Greens"),
                        use_container_width=True)
        c1, c2 = st.columns(2)
        c1.metric("Accuracy", f"{accuracy_score(y_test, R['y_pred_rf']):.4f}")
        c2.metric("AUC", f"{roc_auc_score(y_test, R['y_prob_rf']):.4f}")

        feat_rf = pd.Series(R["modelo_rf"].feature_importances_,
                            index=R["X_test_enc"].columns).sort_values().tail(15)
        figf = px.bar(x=feat_rf.values, y=feat_rf.index, orientation="h",
                      labels={"x": "Importancia", "y": ""},
                      title="Top 15 variables más importantes — Random Forest",
                      color_discrete_sequence=[VERDE])
        st.plotly_chart(figf, use_container_width=True)

    st.divider()
    st.markdown("**Cómo leer estas métricas:**")
    st.markdown("""
    - **Accuracy:** de cada 100 clientes, en cuántos acertó. Engaña con datos
      desbalanceados.
    - **AUC:** probabilidad de que el modelo le dé más chances de convertir a
      un cliente que sí convierte que a uno que no. 0.5 = azar, 1.0 = perfecto.
    """)
    st.info("💡 **¿Qué nos dice esto?** El Random Forest gana en AUC. No es una "
            "diferencia enorme, pero es consistente. Y ahora viene lo "
            "importante: **¿para qué sirve esto?**")

# =============================================================
# CAPÍTULO 6 — PREDICTOR EN VIVO  (ESTRELLA)
# =============================================================
with tabs[6]:
    st.caption("*Hasta acá vimos números generales. Ahora probemos el modelo "
               "con un cliente concreto.*")
    st.header("🎯 Probá el modelo")
    st.markdown("Cargá los datos de un cliente y mirá qué predice el modelo "
                "Random Forest sobre su probabilidad de conversión.")

    # --- Valores por defecto e inicialización ---
    def valor_inicial(col):
        if col in NUMERICAS:
            return int(df[col].median())
        return df[col].mode()[0]

    for col in FEATURES:
        if f"pred_{col}" not in st.session_state:
            st.session_state[f"pred_{col}"] = valor_inicial(col)

    # --- Botón: cargar un cliente real de ejemplo ---
    def cargar_ejemplo():
        fila = df.sample(1).iloc[0]
        for c in FEATURES:
            if c in NUMERICAS:
                st.session_state[f"pred_{c}"] = int(fila[c])
            else:
                st.session_state[f"pred_{c}"] = str(fila[c])

    st.button("🎲 Cargar un cliente de ejemplo", on_click=cargar_ejemplo)
    st.divider()

    col_izq, col_der = st.columns([1.3, 1])

    # --- Columna izquierda: controles ---
    with col_izq:
        st.subheader("Datos del cliente")
        cc = st.columns(2)
        for i, col in enumerate(FEATURES):
            destino = cc[i % 2]
            if col in NUMERICAS:
                destino.number_input(
                    col, value=int(st.session_state[f"pred_{col}"]),
                    step=1, key=f"pred_{col}")
            else:
                opciones = sorted(df[col].dropna().unique().tolist())
                destino.selectbox(col, opciones, key=f"pred_{col}")

    # --- Armar la fila del cliente y predecir ---
    fila = {c: st.session_state[f"pred_{c}"] for c in FEATURES}
    fila_df = pd.DataFrame([fila])
    fila_enc = pd.get_dummies(fila_df)
    fila_enc = fila_enc.reindex(columns=R["X_train_enc"].columns, fill_value=0)
    proba = R["modelo_rf"].predict_proba(fila_enc)[0, 1]
    proba_pct = proba * 100

    # --- Columna derecha: resultado ---
    with col_der:
        st.subheader("Resultado")
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba_pct,
            number={"suffix": " %"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": AZUL},
                "steps": [
                    {"range": [0, 25], "color": "#fee2e2"},
                    {"range": [25, 50], "color": "#fef3c7"},
                    {"range": [50, 100], "color": "#dcfce7"},
                ],
            },
            title={"text": "Probabilidad de conversión"},
        ))
        gauge.update_layout(height=300, margin=dict(t=60, b=10))
        st.plotly_chart(gauge, use_container_width=True)

        if proba_pct >= 50:
            st.success("🟢 **Cliente prometedor** — vale la pena llamarlo.")
        elif proba_pct >= 25:
            st.warning("🟡 **Cliente dudoso** — depende del presupuesto.")
        else:
            st.error("🔴 **Baja probabilidad** — mejor priorizar a otros.")

    st.info("💡 **¿Qué nos dice esto?** El modelo no da un sí/no rígido, sino "
            "una probabilidad. Eso permite ordenar a los clientes y priorizar "
            "— justo lo que vamos a hacer en el simulador.")

# =============================================================
# CAPÍTULO 7 — SIMULADOR DE CAMPAÑA  (ESTRELLA)
# =============================================================
with tabs[7]:
    st.caption("*Esta es la pregunta del millón: con un presupuesto limitado "
               "de llamadas, ¿cómo las usamos mejor?*")
    st.header("💡 Simulador de campaña")
    st.markdown("""
    El banco tiene presupuesto para una cantidad limitada de llamadas.
    Comparamos dos estrategias sobre clientes reales del conjunto de prueba:
    **llamar al azar** vs. **llamar guiados por el modelo** (a los clientes con
    mayor probabilidad de conversión).
    """)

    y_test = R["y_test_enc"]
    y_prob_rf = R["y_prob_rf"]
    total = len(y_test)
    tasa_base = y_test.mean()
    conversiones_totales = int(y_test.sum())

    # --- Controles ---
    st.subheader("Configurá la campaña")
    n_llamadas = st.slider(
        "¿Cuántas llamadas puede hacer el call center?",
        min_value=100, max_value=total,
        value=min(2000, total), step=100)

    c1, c2 = st.columns(2)
    costo_llamada = c1.number_input(
        "Costo por llamada ($)", value=50, step=10, min_value=0)
    ingreso_conv = c2.number_input(
        "Ingreso por cliente convertido ($)", value=2000, step=100, min_value=0)

    # --- Cálculos ---
    orden = np.argsort(y_prob_rf)[::-1]      # de mayor a menor probabilidad
    y_ordenado = y_test[orden]
    conv_modelo = int(y_ordenado[:n_llamadas].sum())
    conv_azar = n_llamadas * tasa_base
    mejora_pct = (conv_modelo - conv_azar) / conv_azar * 100 if conv_azar else 0

    costo_total = n_llamadas * costo_llamada
    ingreso_modelo = conv_modelo * ingreso_conv
    ingreso_azar = conv_azar * ingreso_conv
    roi_modelo = (ingreso_modelo - costo_total) / costo_total * 100 if costo_total else 0
    roi_azar = (ingreso_azar - costo_total) / costo_total * 100 if costo_total else 0

    st.divider()
    st.subheader("Resultados de la campaña")

    c1, c2, c3 = st.columns(3)
    c1.metric("Llamadas realizadas", f"{n_llamadas:,}")
    c2.metric("Conversiones llamando al azar", f"{conv_azar:,.0f}")
    c3.metric("Conversiones llamando con el modelo",
              f"{conv_modelo:,}", f"{mejora_pct:+.0f} %")

    c1, c2, c3 = st.columns(3)
    c1.metric("Tasa de éxito (azar)", f"{tasa_base*100:.1f} %")
    tasa_modelo = conv_modelo / n_llamadas * 100
    c2.metric("Tasa de éxito (con modelo)", f"{tasa_modelo:.1f} %")
    c3.metric("ROI con el modelo", f"{roi_modelo:+.0f} %",
              f"{roi_modelo - roi_azar:+.0f} % vs azar")

    # --- Curva de ganancia acumulada ---
    st.subheader("Curva de ganancia acumulada")
    ganancia = np.cumsum(y_ordenado) / conversiones_totales * 100
    pct_contactados = np.arange(1, total + 1) / total * 100

    figg = go.Figure()
    figg.add_trace(go.Scatter(
        x=pct_contactados, y=ganancia, mode="lines",
        name="Estrategia con modelo", line=dict(color=VERDE, width=3)))
    figg.add_trace(go.Scatter(
        x=[0, 100], y=[0, 100], mode="lines",
        name="Estrategia al azar", line=dict(color="gray", dash="dash")))
    figg.add_vline(x=n_llamadas / total * 100, line_color=AZUL,
                   line_dash="dot",
                   annotation_text="Tu presupuesto de llamadas")
    figg.update_layout(
        title="¿Qué porcentaje de clientes que convierten capturamos?",
        xaxis_title="% de clientes contactados",
        yaxis_title="% de conversiones capturadas")
    st.plotly_chart(figg, use_container_width=True)

    st.success(f"💡 **¿Qué nos dice esto?** Con {n_llamadas:,} llamadas, llamar "
               f"al azar captaría unos {conv_azar:,.0f} clientes. Guiando las "
               f"llamadas con el modelo, se captan {conv_modelo:,} "
               f"(**{mejora_pct:+.0f} %**) con el mismo esfuerzo. "
               "Esto responde la pregunta con la que empezamos.")

    st.caption("Nota: este simulador es la frontera entre el análisis "
               "predictivo (lo que hace el modelo) y el prescriptivo "
               "(recomendar una acción concreta).")

# =============================================================
# CAPÍTULO 8 — CONCLUSIÓN
# =============================================================
with tabs[8]:
    st.caption("*Toda historia merece un cierre.*")
    st.header("✅ Volviendo a la pregunta inicial")

    st.markdown("""
    Empezamos con una pregunta: *¿podemos dejar de desperdiciar 88 llamadas de
    cada 100?* La respuesta es **sí, en buena medida.** Construimos un modelo
    que, en lugar de llamar al azar, prioriza a los clientes con más
    probabilidad de convertir.
    """)

    st.subheader("Comparación final de modelos")
    y_test = R["y_test_enc"]
    comparacion = pd.DataFrame({
        "CART (1 árbol)": [
            accuracy_score(y_test, R["y_pred"]),
            roc_auc_score(y_test, R["y_prob"])],
        "Random Forest (100 árboles)": [
            accuracy_score(y_test, R["y_pred_rf"]),
            roc_auc_score(y_test, R["y_prob_rf"])],
    }, index=["Accuracy", "AUC"]).round(4)
    st.table(comparacion)

    fpr, tpr, _ = roc_curve(y_test, R["y_prob"])
    fpr_rf, tpr_rf, _ = roc_curve(y_test, R["y_prob_rf"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="CART",
                  line=dict(color=AZUL, width=3)))
    fig.add_trace(go.Scatter(x=fpr_rf, y=tpr_rf, mode="lines",
                  name="Random Forest", line=dict(color=VERDE, width=3)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Azar",
                  line=dict(color="gray", dash="dash")))
    fig.update_layout(title="Comparación de curvas ROC",
                      xaxis_title="Falsos positivos",
                      yaxis_title="Verdaderos positivos")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Lo que aprendimos")
    st.markdown("""
    El Random Forest obtuvo un mejor desempeño que el árbol único. La mejora
    se explica porque el ensemble promedia las predicciones de múltiples
    árboles, reduciendo el overfitting. Dado el desbalance de clases, el AUC
    resultó una métrica más representativa que el accuracy.
    """)

    st.subheader("Limitaciones del trabajo")
    st.warning("""
    Es justo reconocer los límites: el modelo se entrenó con datos históricos y
    asume que el comportamiento futuro será similar; la mejora del Random
    Forest fue moderada; y el análisis es **predictivo** — la decisión final de
    a quién llamar combina esta predicción con criterios de negocio.
    """)

    st.subheader("Trabajo futuro")
    st.markdown("""
    - Ajuste de hiperparámetros mediante validación cruzada.
    - Probar otros modelos (por ejemplo, XGBoost).
    - Incorporar un análisis de retorno de inversión de las campañas.
    """)
