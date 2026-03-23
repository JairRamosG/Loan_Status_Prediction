import streamlit as st
import pandas as pd
import joblib
import os

from pathlib import Path
from datetime import date, datetime
import math
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns

# INICIO
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
        st.dataframe(df_original, width='stretch', hide_index=True)

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
        st.dataframe(df_final, width='stretch', hide_index=True)

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

# APP
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

# ANÁLISIS
def plot_hist_variable_binaria(df, variable, title, xlabel, label1, label2, subtitle):
    # Obtener conteos
    counts = df[variable].value_counts().reset_index()
    counts.columns = [variable, 'count']
    labels_map = {1: label1, 0: label2}
    counts['etiqueta'] = counts[variable].map(labels_map)
    fig = px.bar(
        counts,
        x='etiqueta',
        y='count', 
        title=title, 
        labels={'etiqueta': xlabel, 'count': 'Frecuencia'},
        text='count')
    fig.update_traces(marker_color=['#4CAF50', '#F44336'], textposition='outside')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', title_x = 0.5)
    return fig

def plot_distribucion_box(df, numeric_cols, bins=30):
    for col in numeric_cols:
        mean_val = df[col].mean()
        median_val = df[col].median()
        skew_val = df[col].skew()

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(f'<b>Distribución de {col}</b>', f'<b>Boxplot de {col}</b>'),
            horizontal_spacing=0.1,
            column_widths=[0.6, 0.4]
        )

        # Histograma
        fig.add_trace(
            go.Histogram(
                x=df[col],
                nbinsx=bins,
                name='Frecuencia',
                marker=dict(color='#4C72B0', line=dict(color='black', width=1)),
                opacity=0.75,
                showlegend=False
            ),
            row=1, col=1
        )

        # Líneas en histograma
        fig.add_vline(
            x=mean_val,
            line=dict(color='red', dash='dash', width=2),
            annotation_text=f'Media = {mean_val:.2f}',
            annotation_position='top right',
            row=1, col=1
        )
        fig.add_vline(
            x=median_val,
            line=dict(color='green', dash='dot', width=2),
            annotation_text=f'Mediana = {median_val:.2f}',
            annotation_position='bottom right',
            row=1, col=1
        )

        # ================= BOXPLOT MEJORADO =================
        fig.add_trace(
            go.Box(
                x=df[col],
                name='',
                marker_color='#55A868',
                line=dict(color='#2E6B3E', width=2),
                fillcolor='#B3E0B3',
                boxmean=False,
                boxpoints='outliers',      # muestra outliers
                jitter=0,                  # sin dispersión horizontal
                pointpos=0,                # puntos alineados con la caja
                showlegend=False
            ),
            row=1, col=2
        )
        # ================================================

        # Anotación de sesgo
        fig.add_annotation(
            x=0.95, y=0.85,
            xref='paper', yref='paper',
            text=f"Sesgo = {skew_val:.2f}",
            showarrow=False,
            font=dict(size=10),
            bgcolor='white', bordercolor='gray', borderwidth=1, borderpad=4,
            row=1, col=1
        )

        # Layout
        fig.update_layout(
            title_text=f'<b>Análisis de {col}</b>',
            title_x=0.5,
            height=500,
            showlegend=False,
            template='plotly_white',
            margin=dict(t=70, b=50, l=50, r=50)
        )
        fig.update_xaxes(title_text=col, row=1, col=1)
        fig.update_yaxes(title_text="Frecuencia", row=1, col=1)
        fig.update_xaxes(title_text=col, row=1, col=2)
        fig.update_yaxes(title_text="", row=1, col=2)

        yield col, fig

def plot_boxplots_numericas_vs_target(df, numeric_cols, target, cols=2, height_per_row=300):
    """
    Genera un grid de boxplots horizontales interactivos (cada variable numérica vs la variable objetivo).
    Devuelve una figura de Plotly lista para st.plotly_chart().
    """
    n = len(numeric_cols)
    rows = math.ceil(n / cols)
    
    # Crear subplots con una columna para cada variable
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[col for col in numeric_cols],
        horizontal_spacing=0.1,
        vertical_spacing=0.12
    )
    
    # Detectar clases automáticamente
    clases = sorted(df[target].unique())
    # Asignar colores (rojo para la primera clase, verde para la última)
    color_map = {clases[0]: '#e74c3c', clases[-1]: '#55A868'}
    
    # Si hay más de 2 clases, asignar colores con una paleta
    if len(clases) > 2:
        from plotly.express.colors import qualitative
        palette = qualitative.Plotly
        color_map = {cls: palette[i % len(palette)] for i, cls in enumerate(clases)}
    
    for i, col in enumerate(numeric_cols):
        row = i // cols + 1
        col_idx = i % cols + 1
        
        # Para cada clase, crear un trace de boxplot (orientación horizontal)
        for cls in clases:
            subset = df[df[target] == cls][col]
            if not subset.empty:
                fig.add_trace(
                    go.Box(
                        x=subset,
                        name=str(cls),            # nombre de la clase
                        orientation='h',
                        boxmean=False,            # sin línea de media (evita cuadrado extra)
                        boxpoints='outliers',     # mostrar outliers
                        jitter=0.2,
                        pointpos=0,
                        marker=dict(color=color_map[cls]),
                        legendgroup=str(cls),     # agrupar leyenda
                        showlegend=(i == 0)       # mostrar leyenda solo una vez
                    ),
                    row=row, col=col_idx
                )
    
    # Ajustar layout general
    fig.update_layout(
        #title_text=f"Distribución de variables numéricas según {target}",
        #title_x=0.5,
        height=rows * height_per_row,
        showlegend=True,
        legend_title=target,
        template='plotly_white'
    )
    
    # Mejorar apariencia de cada subplot
    fig.update_xaxes(title_text="Valor", row=row, col=col_idx)  # solo el último? se aplicará a todos
    for i in range(1, rows+1):
        for j in range(1, cols+1):
            fig.update_xaxes(title_text="Valor", row=i, col=j)
            fig.update_yaxes(title_text=target, row=i, col=j)
    
    return fig

def plot_frecuencias_categoricas(
    df,
    categoric_cols,
    show_percentage=False,
    height_per_row=400,
    theme='plotly_white'
):
    """
    Genera un grid de gráficos de barras horizontales interactivos.
    Cada categoría dentro de un mismo gráfico tiene un color diferente automático (usando la paleta del tema).
    
    Parámetros:
        df: DataFrame
        categoric_cols: lista de columnas categóricas
        show_percentage: bool, si True muestra porcentaje en lugar de frecuencia absoluta
        height_per_row: altura en píxeles por fila
        theme: str, tema de Plotly ('plotly', 'plotly_white', 'ggplot2', 'seaborn', 'plotly_dark', etc.)
    """
    n = len(categoric_cols)
    cols = 2 if n > 1 else 1
    rows = math.ceil(n / cols)
    
    # Crear subplots
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[col for col in categoric_cols],   # títulos de cada subplot
        horizontal_spacing=0.1,
        vertical_spacing=0.15
    )
    
    total = len(df)
    
    for i, col in enumerate(categoric_cols):
        # Calcular frecuencias y ordenar de mayor a menor (para barras horizontales ascendente)
        counts = df[col].value_counts().reset_index()
        counts.columns = [col, 'count']
        counts = counts.sort_values('count', ascending=True)  # ascendente para que la barra más grande quede arriba
        
        categories = counts[col].tolist()
        frequencies = counts['count'].tolist()
        
        if show_percentage:
            values = [(freq / total) * 100 for freq in frequencies]
            hover_text = [f"{cat}: {freq} ({val:.1f}%)" for cat, freq, val in zip(categories, frequencies, values)]
            text = [f"{val:.1f}%" for val in values]
        else:
            values = frequencies
            hover_text = [f"{cat}: {freq}" for cat, freq in zip(categories, frequencies)]
            text = [str(val) for val in values]
        
        row = i // cols + 1
        col_idx = i % cols + 1
        
        # IMPORTANTE: NO fijamos un color único. Plotly usará la paleta del tema.
        fig.add_trace(
            go.Bar(
                y=categories,
                x=values,
                orientation='h',
                text=text,
                textposition='outside',
                marker=dict(line=dict(color='black', width=1)),   # sin color fijo
                hovertemplate='%{hovertext}<extra></extra>',
                hovertext=hover_text,
                showlegend=False
            ),
            row=row, col=col_idx
        )
        
        # Etiquetas de ejes
        fig.update_xaxes(title_text="Frecuencia" if not show_percentage else "Porcentaje (%)", row=row, col=col_idx)
        fig.update_yaxes(title_text=col, row=row, col=col_idx)
    
    # Eliminar títulos de subplots que no existen (si hay menos variables que subplots)
    # (En nuestro caso, no hay subplots vacíos porque la cuadrícula se ajusta exactamente al número de columnas,
    # pero por si acaso, lo dejamos para evitar títulos huérfanos)
    for r in range(1, rows+1):
        for c in range(1, cols+1):
            # Verificar si este subplot tiene algún trace
            has_trace = False
            for trace in fig.data:
                if trace.xaxis == f"x{r}" and trace.yaxis == f"y{r}":
                    has_trace = True
                    break
            if not has_trace:
                # Buscar la anotación correspondiente y ocultarla
                idx = (r-1)*cols + (c-1)
                if idx < len(fig.layout.annotations):
                    fig.layout.annotations[idx].text = ""
    
    total_height = rows * height_per_row
    fig.update_layout(
        title_text="Distribución de Variables Categóricas" if n > 1 else None,
        title_x=0.5,
        height=total_height,
        template=theme,                # El tema controla los colores de las barras y el estilo general
        margin=dict(t=80 if n > 1 else 40, b=40, l=40, r=40)
    )
    
    return fig

def plot_crosstab_categoricas_interactive(
    df,
    categoric_cols,
    target,
    ylabel="Proporción",
    label_pos="Pagó",
    label_neg="No pagó",
    rotation=30,
    colors=None,
    theme='plotly_white',
    height_per_row=400
):
    """
    Genera un grid de gráficos de barras apiladas interactivos para variables categóricas,
    mostrando la proporción de la variable objetivo en cada categoría.

    Parámetros:
        df: DataFrame
        categoric_cols: lista de columnas categóricas a analizar
        target: nombre de la variable objetivo (binaria)
        ylabel: etiqueta del eje Y
        label_pos: nombre de la clase positiva (usado en leyenda)
        label_neg: nombre de la clase negativa
        rotation: ángulo de rotación de las etiquetas del eje X (grados)
        colors: lista de colores para las dos clases [color_neg, color_pos] (ej. ['#e74c3c', '#2ecc71'])
        theme: tema de Plotly ('plotly', 'plotly_white', 'ggplot2', etc.)
        height_per_row: altura en píxeles por fila
    """
    n = len(categoric_cols)
    cols = 2
    rows = math.ceil(n / cols)

    # Colores por defecto si no se especifican
    if colors is None:
        colors = ['#e74c3c', '#2ecc71']  # rojo para negativos, verde para positivos

    # Crear subplots
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[col for col in categoric_cols],
        horizontal_spacing=0.1,
        vertical_spacing=0.15
    )

    # Determinar el orden de clases (suponemos binario)
    classes = sorted(df[target].unique())
    # Asignar colores: primero clase negativa, luego positiva
    if len(classes) == 2:
        color_map = {classes[0]: colors[0], classes[1]: colors[1]}
    else:
        # Si hay más de 2, usar paleta extendida (pero esperamos binario)
        from plotly.express.colors import qualitative
        palette = qualitative.Plotly
        color_map = {cls: palette[i % len(palette)] for i, cls in enumerate(classes)}

    for i, col in enumerate(categoric_cols):
        # Calcular proporciones por categoría
        crosstab = pd.crosstab(df[col], df[target], normalize='index') * 100  # porcentajes
        crosstab = crosstab.reset_index()
        categories = crosstab[col].tolist()
        # Para cada clase, añadir una barra (apilada)
        for cls in classes:
            values = crosstab[cls].tolist() if cls in crosstab.columns else [0]*len(categories)
            # Evitar texto cuando valor es 0 (para que no aparezca etiqueta)
            text = [f"{v:.1f}%" if v > 0 else "" for v in values]

            row = i // cols + 1
            col_idx = i % cols + 1

            fig.add_trace(
                go.Bar(
                    x=categories,
                    y=values,
                    name=str(cls) if i == 0 else None,  # mostrar leyenda solo en el primer subplot
                    text=text,
                    textposition='inside',
                    textfont=dict(size=10),
                    marker=dict(color=color_map[cls]),
                    showlegend=(i == 0),   # solo una leyenda global
                    legendgroup=str(cls)   # agrupar por clase
                ),
                row=row, col=col_idx
            )

        # Personalizar ejes y rotación de etiquetas X
        fig.update_xaxes(title_text=col, tickangle=rotation, row=row, col=col_idx)
        fig.update_yaxes(title_text=ylabel, row=row, col=col_idx)

    # Ajustar layout general
    total_height = rows * height_per_row
    fig.update_layout(
        #title_text="Proporción de Pago del Préstamo por Variables Categóricas",
        #title_x=0.5,
        height=total_height,
        template=theme,
        barmode='stack',               # apilado
        legend_title="Estado del préstamo",
        margin=dict(t=80, b=40, l=40, r=40)
    )

    # Personalizar nombres de leyenda si se proporcionan
    if label_neg and label_pos and len(classes) == 2:
        # Renombrar leyenda (por defecto usa los valores originales de la columna target)
        # Plotly asigna los nombres automáticamente; podemos cambiarlos en la leyenda
        # Pero como añadimos los traces con name=str(cls), podemos mapear los nombres.
        # Es más fácil pasar los nombres deseados a través de los parámetros.
        # Vamos a reasignar los nombres de los traces manualmente:
        for trace in fig.data:
            if trace.name == str(classes[0]):
                trace.name = label_neg
            elif trace.name == str(classes[1]):
                trace.name = label_pos

    return fig


# rutas
BASE_DIR = Path(__file__).resolve().parent.parent
learning_curve_path = BASE_DIR / "metadata" / "EXP_02_learning_curve.png"
cm_path = BASE_DIR / "metadata" / "EXP_02_matriz.png"
data_path = BASE_DIR / "data" / "raw" / "loan_dataset_20000.csv"
bagging_classifier_path = BASE_DIR / "img" / "BaggingClassifier.png"
column_transformer_path = BASE_DIR / "img" / "ColumnTransformer.png"

data = pd.read_csv(data_path)

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
    if st.button("Inicio", width='stretch'):
        st.session_state.pagina = "Inicio"

with col2:
    if st.button("Análisis", width='stretch'):
        st.session_state.pagina = "Análisis"

with col3:
    if st.button("App", width='stretch'):
        st.session_state.pagina = "App"

###########################################################################################################
# INICIO
###########################################################################################################
if st.session_state.pagina == "Inicio":

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

    #data = pd.read_csv(data_path)
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
        st.image(column_transformer_path, caption="Diagrama del Preprocesamiento", width='stretch')
#####################################################################################################################################3    


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
        st.image(bagging_classifier_path, caption="Diagrama del Preprocesamiento", width='stretch')

    # --- Imágenes de diagnóstico del modelo (aprovechando tus archivos) ---
    # Crear dos columnas para las imágenes
    img_col1, img_col2 = st.columns(2)

    with img_col1:
        st.markdown("##### Curva de aprendizaje")
        try:
            st.image(learning_curve_path, caption="Curva de aprendizaje (mejor modelo)", width='stretch')
        except FileNotFoundError:
            st.warning("Imagen de curva de aprendizaje no encontrada. Asegúrate de que el archivo exista.")

    with img_col2:
        st.markdown("##### Matriz de confusión")
        try:
            st.image(cm_path, caption="Matriz de confusión en test", width='stretch')
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
elif st.session_state.pagina == "Análisis":
    st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h2 style="color: #1f77b4;">Análisis</h2>
        </div>
    """, unsafe_allow_html=True)

    st.subheader("Estadísticas descriptivas")
    st.dataframe(data.describe(include='all').T, use_container_width=True)

    st.subheader("Distribución de la variable objetivo")
    fig = plot_hist_variable_binaria(
        data,
        'loan_paid_back', 
        'Estado del préstamo',
        'Tipo de pago',
        'Si pagó', 'No pagó',
        'Distribución de prestamos pagados')
    st.plotly_chart(fig, width='content')

    st.subheader("Algunas variables numéricas")
    numeric_cols = ['age', 'annual_income', 'credit_score', 'total_credit_limit']
    for col, fig in plot_distribucion_box(data, numeric_cols, bins=30):
        st.subheader(f" > {col}")
        st.plotly_chart(fig, width='stretch')

    st.subheader("Variables según el target")
    for col in numeric_cols:
        fig = plot_boxplots_numericas_vs_target(
        df=data,
        numeric_cols=numeric_cols,
        target='loan_paid_back',
        cols=2,
        height_per_row=300)
    st.plotly_chart(fig, width='stretch')


    
    st.subheader("Distribución de variables categóricas")
    categoric_cols = ['gender', 'marital_status', 'education_level', 'loan_purpose']

    for col in categoric_cols:
        fig = plot_frecuencias_categoricas(
            df=data,
            categoric_cols=[col],
            show_percentage=True,
            theme='plotly_white')
        st.plotly_chart(fig, key=f"categ_{col}", width='stretch')

    st.subheader("Proporción de pago por variables categóricas")
    fig = plot_crosstab_categoricas_interactive(
        df=data,
        categoric_cols=categoric_cols,
        target='loan_paid_back',
        label_pos="Pagó",
        label_neg="No pagó",
        rotation=30,
        colors=['#e74c3c', '#2ecc71'],   # rojo (no pagó), verde (pagó)
        theme='plotly_white'
    )
    st.plotly_chart(fig, key="crosstab_categoricas", use_container_width=True)

###########################################################################################################
# APP
###########################################################################################################

elif st.session_state.pagina == "App":
    modelo = cargar_modelo()

    st.header("App")
    st.text('Aqui van los botones')