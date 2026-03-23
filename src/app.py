import streamlit as st
import pandas as pd
import joblib
import os

from pathlib import Path
from datetime import date, datetime

# rutas
BASE_DIR = Path(__file__).resolve().parent.parent
learning_curve_path = BASE_DIR / "metadata" / "EXP_02_learning_curve.png"
cm_path = BASE_DIR / "metadata" / "EXP_02_matriz.png"
data_path = BASE_DIR / "data" / "raw" / "loan_dataset_20000.csv"
bagging_classifier_path = BASE_DIR / "img" / "BaggingClassifier.png"
column_transformer_path = BASE_DIR / "img" / "ColumnTransformer.png"

st.set_page_config(
    page_title="Loan Status Prediction",
    page_icon="",
    layout="wide")

###########################################################################################################
# Cabecera
###########################################################################################################
# Inicializar estado
if "pagina" not in st.session_state:
    st.session_state.pagina = "Inicio"

# Crear columnas para los botones
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Inicio", use_container_width=True):
        st.session_state.pagina = "Inicio"

with col2:
    if st.button("Análisis", use_container_width=True):
        st.session_state.pagina = "Análisis"

with col3:
    if st.button("App", use_container_width=True):
        st.session_state.pagina = "App"

###########################################################################################################
# INICIO
###########################################################################################################
if st.session_state.pagina == "Inicio":

    def show_home():
        # --- Estilos personalizados para tarjetas y sombras ---
        st.markdown("""
            <style>
            .card {
                background-color: #f8f9fa;
                border-radius: 15px;
                padding: 20px;
                margin: 10px 0;
                box-shadow: 0 4px 8px rgba(0,0,0,0.05);
                transition: transform 0.2s;
            }
            .card:hover {
                transform: translateY(-3px);
                box-shadow: 0 6px 12px rgba(0,0,0,0.1);
            }
            .big-number {
                font-size: 2.5rem;
                font-weight: bold;
                color: #2c3e50;
            }
            .metric-label {
                font-size: 1rem;
                color: #6c757d;
            }
            hr {
                margin: 1.5rem 0;
            }
            </style>
        """, unsafe_allow_html=True)

    # --- Encabezado principal ---
    st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h1 style="color: #1f77b4;"><strong>Loan Status Prediction</strong></h1>
            <h4 style="color: #6c757d;">Predicción de incumplimiento de préstamos con Machine Learning</h4>
            <hr>
        </div>
    """, unsafe_allow_html=True)

    # --- Fila de tarjetas resumen ---
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Registros analizados",
                    f"20k+",
                    delta = 'Registrados')

    with col2:
        st.metric("Características",
                    f"40+",
                )

    with col3:
        st.metric("Precision",
                    f"0.8961",
                    delta = 'Mejor modelo')

    # --- Descripción del problema y solución en dos columnas ---
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("""
            <div class="card">
                <h3>¿Cuál es el problema?</h3>
                <p>Las entidades financieras necesitan evaluar el riesgo de impago de los solicitantes de crédito. 
                Un modelo predictivo preciso ayuda a reducir pérdidas, optimizar el portafolio y tomar decisiones más justas.</p>
            </div>
        """, unsafe_allow_html=True)
    with col_right:
        st.markdown("""
            <div class="card">
                <h3>Solución propuesta</h3>
                <p>Se desarrollado un <strong>pipeline de Machine Learning</strong> que:
                <ul>
                    <li>Realiza ingeniería de características (grupos de edad, ratios financieros, etc.).</li>
                    <li>Preprocesa automáticamente variables numéricas y categóricas.</li>
                    <li>Entrena un <strong>Bagging de Random Forests</strong> optimizado con <strong>RandomizedSearchCV</strong>.</li>
                    <li>Equilibra las clases usando <strong>SMOTE</strong>.</li>
                </ul>
                </p>
            </div>
        """, unsafe_allow_html=True)
    st.markdown("---")
############################################################################################################################3

    st.markdown("""
            <div style="text-align: center; margin-bottom: 2rem;">
                <h2 style="color: #1f77b4;">Datos</h2>
                <p style="color: #6c757d;">Archivo original</p>
            </div>
        """, unsafe_allow_html=True)

    data = pd.read_csv(data_path)
    st.dataframe(data.head(10))
    st.info("Dataset original")
############################################################################################################################3
    # --- Visualización del Preprocesamiento ---
    st.markdown("""
            <div style="text-align: center; margin-bottom: 2rem;">
                <h2 style="color: #1f77b4;">Preprocesamiento</h2>
                <p style="color: #6c757d;">Transformaciones aplicadas y formato final de las variables</p>
            </div>
        """, unsafe_allow_html=True)
    with st.expander("ColumnTransformer"):
        st.markdown("""
            <div style="background-color: #f0f2f6; border-radius: 10px; padding: 15px; color: black">
                <strong>Flujo completo:</strong><br>
                
            </div>
        """, unsafe_allow_html=True)
        st.image(column_transformer_path, caption="Diagrama del Preprocesamiento", use_container_width=True)
#####################################################################################################################################3    


    def show_data_dictionary():
        # --- Variables originales ---
        with st.expander("Variables originales", expanded=False):
            datos_originales = [
                ("age", "NUM", "Discretizar (genera age_group)"),
                ("gender", "CAT_NOM", "Encoding + OneHotEncoder (drop first)"),
                ("marital_status", "CAT_NOM", "Encoding + OneHotEncoder (drop first)"),
                ("education_level", "CAT_ORD", "OrdinalEncoder"),
                ("annual_income", "NUM", "log + Normalizar"),
                ("monthly_income", "NUM", "log + Normalizar"),
                ("employment_status", "CAT_NOM", "Encoding + OneHotEncoder (drop first)"),
                ("debt_to_income_ratio", "NUM", "log + Normalizar"),
                ("credit_score", "NUM", "Normalizar"),
                ("loan_amount", "NUM", "Normalizar"),
                ("loan_purpose", "CAT_NOM", "Encoding + OneHotEncoder (drop first)"),
                ("interest_rate", "NUM", "Normalizar"),
                ("loan_term", "CAT_NOM", "OneHotEncoder (drop first)"),
                ("installment", "NUM", "Normalizar"),
                ("grade_subgrade", "CAT_ORD", "OrdinalEncoder"),
                ("num_of_open_accounts", "NUM", "Normalizar"),
                ("total_credit_limit", "NUM", "log + Normalizar"),
                ("current_balance", "NUM", "log + Normalizar"),
                ("delinquency_history", "NUM", "log1p + Normalizar"),
                ("public_records", "NUM", "Normalizar"),
                ("num_of_delinquencies", "NUM", "Normalizar"),
                ("loan_paid_back", "NUM(BIN)", "Objetivo")
            ]
            df_original = pd.DataFrame(datos_originales, columns=["Variable", "Tipo", "Tratamiento"])
            st.dataframe(df_original, use_container_width=True, hide_index=True)

        # --- Variables finales ---
        with st.expander("Variables finales (después del pipeline)", expanded=False):
            datos_finales = [
                ("age_group", "CAT_ORD", "Grupos: joven, adulto_joven, adulto, adulto_mayor, 3_Edad"),
                ("marital_status_*", "NUM(BIN)", "4 dummies: Divorced, Widowed, Married, Single"),
                ("gender_*", "NUM(BIN)", "3 dummies: Male, Female, Other"),
                ("education_level", "CAT_ORD", "Ordinal (High School → Doctorate)"),
                ("annual_income", "NUM NORM", "Escalado (log transformado)"),
                ("monthly_income", "NUM NORM", "Escalado (log transformado)"),
                ("employment_status_*", "NUM(BIN)", "5 dummies: Unemployed, Retired, Student, Employed, Self-employed"),
                ("debt_to_income_ratio", "NUM NORM", "Escalado (log transformado)"),
                ("credit_score", "NUM NORM", "Escalado"),
                ("loan_amount", "NUM NORM", "Escalado"),
                ("loan_purpose_*", "NUM(BIN)", "8 dummies: Education, Medical, Home, Car, Other, Business, Vacation, Debt consolidation"),
                ("interest_rate", "NUM NORM", "Escalado"),
                ("loan_term_60", "NUM(BIN)", "Dummy (1 si loan_term=60)"),
                ("installment", "NUM NORM", "Escalado"),
                ("grade_subgrade", "CAT_ORD", "Ordinal (A1 → G1)"),
                ("num_of_open_accounts", "NUM NORM", "Escalado"),
                ("total_credit_limit", "NUM NORM", "Escalado (log transformado)"),
                ("current_balance", "NUM NORM", "Escalado (log transformado)"),
                ("delinquency_history", "NUM NORM", "Escalado (log1p transformado)"),
                ("public_records", "NUM NORM", "Escalado"),
                ("num_of_delinquencies", "NUM NORM", "Escalado"),
                ("loan_to_income", "NUM NORM", "Feature: loan_amount - annual_income"),
                ("has_delinquency_history", "NUM(BIN)", "Feature: (delinquency_history > 0)"),
                ("severity_score", "NUM NORM", "Feature: num_of_delinquencies + public_records"),
                ("payment_income", "NUM NORM", "Feature: installment / monthly_income")
            ]
            df_final = pd.DataFrame(datos_finales, columns=["Variable", "Tipo", "Descripción"])
            st.dataframe(df_final, use_container_width=True, hide_index=True)

        # --- Leyenda visual con columnas (más funcional) ---
        st.markdown("**Leyenda de tipos**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("🔵 **NUM** - Numérica continua")
        with col2:
            st.markdown("🟢 **NUM NORM** - Numérica estandarizada")
        with col3:
            st.markdown("🟠 **CAT_ORD** - Categórica ordinal")
        with col4:
            st.markdown("🟣 **NUM(BIN)** - Binaria (0/1)")
    show_data_dictionary()
    st.markdown("---")


####################################################################################################################################
    
    # --- Visualización del modelo ---
    st.markdown("""
            <div style="text-align: center; margin-bottom: 2rem;">
                <h2 style="color: #1f77b4;">Modelo final</h2>
                <p style="color: #6c757d;">EXP_02</p>
            </div>
        """, unsafe_allow_html=True)
    with st.expander("BagginClassifier"):
        st.markdown("""
            <div style="background-color: #f0f2f6; border-radius: 10px; padding: 15px; color: black">
                <strong>Flujo completo:</strong><br>
            </div>
        """, unsafe_allow_html=True)
        st.image(bagging_classifier_path, caption="Diagrama del Preprocesamiento", use_container_width=True)

    # --- Imágenes de diagnóstico del modelo (aprovechando tus archivos) ---
    # Crear dos columnas para las imágenes
    img_col1, img_col2 = st.columns(2)

    with img_col1:
        st.markdown("##### Curva de aprendizaje")
        try:
            st.image(learning_curve_path, caption="Curva de aprendizaje (mejor modelo)", use_container_width=True)
        except FileNotFoundError:
            st.warning("Imagen de curva de aprendizaje no encontrada. Asegúrate de que el archivo exista.")

    with img_col2:
        st.markdown("##### Matriz de confusión")
        try:
            st.image(cm_path, caption="Matriz de confusión en test", use_container_width=True)
        except FileNotFoundError:
            st.warning("Imagen de matriz de confusión no encontrada. Asegúrate de que el archivo exista.")

    # --- Métricas en formato tabla (o con columnas) ---
    st.markdown("---")
    st.markdown("""
            <div style="text-align: center; margin-bottom: 2rem;">
                <h2 style="color: #1f77b4;">Resultados en el conjunto de prueba</h2>
            </div>
        """, unsafe_allow_html=True)
    col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
    # Puedes cargar estos valores desde el JSON de metadatos si lo prefieres
    with col_metric1:
        st.metric("Precisión (Precision)", "0.8961", "medida de interés")
    with col_metric2:
        st.metric("Recall", "0.9784", "")
    with col_metric3:
        st.metric("F1-score", "0.9355", "")
    with col_metric4:
        st.metric("Accuracy", "0.8920", "")

    # --- Créditos finales ---
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #9e9e9e; padding: 1rem;">
            <strong>Jair Ramos</strong> · 
            Repositorio en <a href="https://github.com/JairRamosG/Loan_Status_Prediction" target="_blank">GitHub</a> · 
        </div>
    """, unsafe_allow_html=True)

###########################################################################################################
# ANÁLISIS
###########################################################################################################
elif st.session_state.pagina == "Analisis":
    st.header("Análisis de datos")
    st.text('Aquí pongo el EDA', help=None, width="content", text_alignment="left")

###########################################################################################################
# APP
###########################################################################################################
elif st.session_state.pagina == "App":
    
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
    
    st.header("App")
    st.text('Aqui van los botones', help=None, width="content", text_alignment="left")

