# -*- coding: utf-8 -*-
"""
Created on Sun May 10 12:12:28 2026

@author: gpoliak
"""
import streamlit as st

st.set_page_config(page_title="Demo Captación de Clientes", layout="wide")

st.title("Demo de Minería de Datos")
st.subheader("Predicción de captación de clientes bancarios")

st.write("""
Esta aplicación muestra una demo del modelo predictivo desarrollado para estimar
si un cliente se convertirá o no en cliente del banco.
""")
import pandas as pd

# Cargar dataset
train = pd.read_csv("banca_train.csv", sep=';')

st.subheader("Vista previa del Dataset")

st.dataframe(train.head())
