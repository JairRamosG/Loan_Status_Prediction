import streamlit as st
import pandas as pd
import joblib
import os

from pathlib import Path
from datetime import date, datetime

st.set_page_config(
    page_title="Loan Status Prediction",
    page_icon="",
    layout="wide")

def cargar_modelo():
    '''
    Cargar el modelo ya hecho en el train.py
    '''
    try:
        MODELO_FILE = Path(__file__).parent.parent / "models" / "EXP_02.pkl"

        if not MODELO_FILE.exists():
            st.error(f'Modelo no encontrado en {MODELO_FILE}')
            return None
        
        with st.spinner('Cargando modelo...'):
            modelo = joblib.load(MODELO_FILE)
            st.success('Modelo cargado  ')
            return modelo
    except Exception as e:
        st.error(f'Error al cargar el modelo: {str(e)}')
        return None
modelo = cargar_modelo()

def alimentar_pipeline(datos_usuario):
    '''
    Convertir la información del formulario a una entrada que si acepte el pipeline
    Args:
        datos_usuario (dict): Todos los valores que venian en el formulario
    Outputs:
        df (pd.DataFrame): DataFrame con los datos del usuario
    '''
    df = pd.DataFrame([datos_usuario], index=None)
    return df