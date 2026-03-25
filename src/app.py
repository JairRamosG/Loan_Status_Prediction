import streamlit as st
import pandas as pd
import joblib
import os

from pathlib import Path
from datetime import date, datetime
import math
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns

paleta = 'Magma' ### 'Plasma', 'Inferno', 'Magma', 'YlGnBu', 'viridis', 'Turbo', 'plotly3', 'Pastel1'

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


# ANÁLISIS
def plot_pie_binaria(
    df,
    variable,
    label1,
    label2,
    subtitle=None,
    colors=None,
    outline_colors=None,
    height=500,
    width=600,
    hole=0.3
):
    """
    Genera un gráfico de pastel interactivo para una variable binaria,
    con colores suaves y contorno en tonos más oscuros.

    Parámetros:
        df: DataFrame
        variable: nombre de la columna binaria (valores 0/1)
        label1: etiqueta para la clase positiva (1)
        label2: etiqueta para la clase negativa (0)
        subtitle: subtítulo (opcional)
        colors: lista de colores personalizados (si es None, usa rojo/verde pastel)
        outline_colors: lista de colores para el contorno (si es None, usa tonos más oscuros)
        height: altura de la figura
        width: ancho de la figura
        hole: tamaño del agujero central (0 = pastel, 0.3 = donut)
    Returns:
        fig: plotly.graph_objects.Figure
    """
    # Calcular conteos y porcentajes
    counts = df[variable].value_counts().reset_index()
    counts.columns = [variable, 'count']
    total = counts['count'].sum()
    counts['percentage'] = (counts['count'] / total * 100).round(1)
    
    labels_map = {1: label1, 0: label2}
    counts['etiqueta'] = counts[variable].map(labels_map)
    
    default_colors = ['#e74c3c', '#2ecc71']  
    if colors is None:
        colors = default_colors
    
    default_outline_colors = ['#c0392b', '#27ae60']  
    if outline_colors is None:
        outline_colors = default_outline_colors
    
    fig = px.pie(
        counts,
        values='count',
        names='etiqueta',
        color='etiqueta',
        color_discrete_sequence=colors,
        hole=hole,
        height=height,
        width=width
    )
    
    # Personalizar el texto dentro del gráfico y el contorno
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        textfont_size=14,
        textfont_color='white',          # texto blanco para mejor contraste
        marker=dict(
            line=dict(
                color=outline_colors,    # contorno en tonos oscuros
                width=3                  # grosor del contorno
            )
        ),
        hovertemplate='<b>%{label}</b><br>Frecuencia: %{value}<br>Porcentaje: %{percent:.1f}%<extra></extra>'
    )
    
    # Ajustar layout
    fig.update_layout(
        title_x=0.5,
        title_font_size=16,
        showlegend=False,                # ya tenemos etiquetas dentro
        margin=dict(l=40, r=40, t=80, b=40),
        annotations=[]                   # limpiar anotaciones automáticas
    )
    
    # Añadir subtítulo si se proporciona
    if subtitle:
        fig.update_layout(
            annotations=[
                dict(
                    text=subtitle,
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=1.08,
                    showarrow=False,
                    font=dict(size=12, color="gray")
                )
            ]
        )
    
    # Añadir total en el centro (opcional, solo para donut)
    if hole > 0:
        # Crear texto central con el total
        center_text = f"Total<br>{total:,}"
        
        # Configurar anotaciones del centro
        center_annotation = dict(
            text=center_text,
            x=0.5,
            y=0.5,
            font_size=16,
            showarrow=False,
            font=dict(size=14, weight='bold', color="#92beea")
        )
        
        # Combinar anotaciones
        if subtitle:
            fig.update_layout(
                annotations=[
                    dict(
                        text=subtitle,
                        xref="paper",
                        yref="paper",
                        x=0.5,
                        y=1.08,
                        showarrow=False,
                        font=dict(size=12, color="gray")
                    ),
                    center_annotation
                ]
            )
        else:
            fig.update_layout(annotations=[center_annotation])
    
    return fig

def plot_numeric_distribution(
    df,
    col,
    bins=30,
    show_kde=True,
    colorscale=paleta,   
    height=500,
    width=800
):
    """
    Genera un histograma interactivo (sin boxplot) con línea de densidad KDE.
    Las barras se colorean con la paleta 'Viridis' (o la que elijas).
    """
    data = df[col].dropna()
    hist, bin_edges = np.histogram(data, bins=bins, density=False)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=bin_centers,
        y=hist,
        name='Frecuencia',
        marker=dict(
            colorscale=colorscale,
            color=bin_centers,           # cada barra se colorea según su valor
            showscale=False,             # ocultar barra de colores
            line=dict(color='black', width=1)
        ),
        hovertemplate='Valor: %{x:.2f}<br>Frecuencia: %{y}<extra></extra>'
    ))
    
    if show_kde and len(data) > 1:
        kde = gaussian_kde(data)
        x_grid = np.linspace(data.min(), data.max(), 200)
        y_kde = kde(x_grid) * len(data) * (bin_edges[1] - bin_edges[0])
        fig.add_trace(go.Scatter(
            x=x_grid,
            y=y_kde,
            mode='lines',
            name='Densidad (KDE)',
            line=dict(color='#FF7F0E', width=2),
            hovertemplate='%{x:.2f}<br>Densidad: %{y:.2f}<extra></extra>'
        ))
    
    fig.update_layout(
        title=None,
        xaxis_title=col,
        yaxis_title='Frecuencia',
        template='plotly_white',
        height=height,
        width=width,
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    return fig

def plot_boxplot_single(
    df,
    col,
    target,
    palette=paleta,
    height=400,
    width=800,
    label_neg="No pagó",
    label_pos="Pagó"
):
    """
    Genera un boxplot horizontal interactivo para una variable numérica vs la variable objetivo.
    Cada clase (target) se colorea con un color diferente usando la paleta especificada.

    Parámetros:
        df: DataFrame
        col: nombre de la columna numérica
        target: nombre de la variable objetivo (binaria)
        palette: paleta de Plotly (ej. 'viridis', 'plasma', 'set1', etc.)
        height: altura de la figura
        width: ancho de la figura
        label_neg: etiqueta para la clase negativa (por defecto "No pagó")
        label_pos: etiqueta para la clase positiva (por defecto "Pagó")
    Returns:
        fig: plotly.graph_objects.Figure
    """
    # Detectar clases automáticamente
    clases = sorted(df[target].unique())
    
    # Crear figura
    fig = go.Figure()
    
    # Si la paleta es cualitativa (string de la lista de plotly.express.colors.qualitative)
    # usamos la paleta cualitativa; si es secuencial, usamos colores escalados.
    # Para simplificar, asignaremos colores según el índice de la clase, usando la paleta.
    # Obtener colores según el número de clases
    if len(clases) <= 2:
        # Para binario, asignar colores específicos (por defecto rojo/verde)
        # pero puedes cambiarlos según la paleta
        colors = ['#e74c3c', '#2ecc71']  # rojo para clase negativa, verde para positiva
    else:
        # Para múltiples clases, usar una paleta cualitativa
        from plotly.express.colors import qualitative
        # Intentar obtener la paleta por nombre
        if hasattr(qualitative, palette):
            palette_colors = getattr(qualitative, palette)
        else:
            palette_colors = qualitative.Plotly  # fallback
        colors = [palette_colors[i % len(palette_colors)] for i in range(len(clases))]
    
    # Mapear colores a clases
    color_map = {clases[i]: colors[i] for i in range(len(clases))}
    
    # Añadir un boxplot por cada clase (orientación horizontal)
    for cls in clases:
        subset = df[df[target] == cls][col]
        if not subset.empty:
            # Determinar etiqueta de leyenda según la clase
            if len(clases) == 2:
                if cls == clases[0]:
                    label = label_neg
                else:
                    label = label_pos
            else:
                label = str(cls)
            
            fig.add_trace(go.Box(
                x=subset,
                name=label,
                orientation='h',
                boxmean=False,
                boxpoints='outliers',
                jitter=0.2,
                pointpos=0,
                marker=dict(color=color_map[cls]),
                legendgroup=label,
                showlegend=True
            ))
    
    # Ajustar layout
    fig.update_layout(
        title=None,
        xaxis_title=col,
        yaxis_title=target,
        height=height,
        width=width,
        template='plotly_white',
        showlegend=True,
        legend_title="Estado del préstamo",
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    return fig

def plot_categorical(df, col, show_percentage=False, palette='Viridis', height=500):
    """
    Genera un gráfico de barras horizontales interactivo para una variable categórica.
    Cada barra tiene un color diferente (usando la paleta especificada).

    Parámetros:
        df: DataFrame
        col: nombre de la columna categórica
        show_percentage: bool, si True muestra porcentaje sobre el total en lugar de frecuencia absoluta
        palette: paleta de Plotly (ej. 'Viridis', 'Plasma', 'Set1', 'Blues', etc.)
        height: altura de la figura
    Returns:
        fig: plotly.graph_objects.Figure
    """
    # Obtener frecuencias y ordenar de mayor a menor
    counts = df[col].value_counts().reset_index()
    counts.columns = [col, 'count']
    # Orden descendente para que la barra más grande quede arriba en horizontal
    counts = counts.sort_values('count', ascending=True)
    
    total = len(df)
    
    if show_percentage:
        counts['value'] = (counts['count'] / total) * 100
        counts['label'] = counts.apply(lambda row: f"{row['count']} ({row['value']:.1f}%)", axis=1)
        xlabel = "Porcentaje (%)"
    else:
        counts['value'] = counts['count']
        counts['label'] = counts['count'].astype(str)
        xlabel = "Frecuencia"
    
    # Crear gráfico de barras horizontales
    fig = px.bar(
        counts,
        y=col,
        x='value',
        orientation='h',
        text='label',
        color=col,                     # Asigna un color diferente por categoría
        color_discrete_sequence=px.colors.sequential.__dict__.get(palette, px.colors.qualitative.Plotly),
        labels={col: '', 'value': xlabel},
        height=height
    )
    
    # Personalizar la apariencia
    fig.update_traces(
        textposition='outside',
        marker_line_color='black',
        marker_line_width=1,
        showlegend = False
    )
    fig.update_layout(
        title=None,                    # Sin título general
        xaxis_title=xlabel,
        yaxis_title=col,
        showlegend=False,
        template='plotly_white'        # Tema claro (puedes cambiarlo)
    )
    # Ajustar el ancho de la figura (opcional)
    fig.update_layout(width=800)
    return fig

def plot_crosstab_single(
    df,
    col,
    target,
    ylabel="Proporción (%)",
    label_pos="Pagó",
    label_neg="No pagó",
    rotation=30,
    colors=None,
    outline_colors=None,
    theme='plotly_white',
    height=500,
    width=800
):
    """
    Genera un gráfico de barras apiladas interactivo para una variable categórica,
    mostrando la proporción de la variable objetivo en cada categoría.

    Parámetros:
        df: DataFrame
        col: nombre de la columna categórica
        target: nombre de la variable objetivo (binaria)
        ylabel: etiqueta del eje Y
        label_pos: nombre de la clase positiva
        label_neg: nombre de la clase negativa
        rotation: ángulo de rotación de las etiquetas del eje X (grados)
        colors: lista de colores [color_neg, color_pos] (default: ['#e74c3c', '#2ecc71'])
        outline_colors: lista de colores para el contorno (default: ['#c0392b', '#27ae60'])
        theme: tema de Plotly ('plotly', 'plotly_white', 'ggplot2', etc.)
        height: altura de la figura
        width: ancho de la figura
    Returns:
        fig: plotly.graph_objects.Figure
    """
    # Colores por defecto
    if colors is None:
        colors = ['#e74c3c', '#2ecc71']  # rojo para negativos, verde para positivos
    
    # Colores de contorno por defecto (tonos más oscuros)
    if outline_colors is None:
        outline_colors = ['#c0392b', '#27ae60']  # rojo oscuro, verde oscuro
    
    # Detectar clases
    classes = sorted(df[target].unique())
    
    # Asignar colores: clase negativa (primer índice) y positiva (segundo)
    if len(classes) >= 2:
        color_map = {classes[0]: colors[0], classes[1]: colors[1]}
        outline_map = {classes[0]: outline_colors[0], classes[1]: outline_colors[1]}
    else:
        # Si solo hay una clase (caso raro), usar el primer color
        color_map = {classes[0]: colors[0]}
        outline_map = {classes[0]: outline_colors[0]}
    
    # Calcular proporciones por categoría (porcentajes)
    crosstab = pd.crosstab(df[col], df[target], normalize='index') * 100
    crosstab = crosstab.reset_index()
    categories = crosstab[col].tolist()
    
    # Crear figura
    fig = go.Figure()
    
    # Añadir barra para cada clase
    for cls in classes:
        if cls in crosstab.columns:
            values = crosstab[cls].tolist()
        else:
            values = [0] * len(categories)
        
        # Texto solo para valores > 0
        text = [f"{v:.1f}%" if v > 0 else "" for v in values]
        
        # Determinar nombre de la clase para la leyenda
        if cls == classes[0] and len(classes) >= 2:
            name = label_neg
        elif cls == classes[1] and len(classes) >= 2:
            name = label_pos
        else:
            name = str(cls)
        
        fig.add_trace(go.Bar(
            x=categories,
            y=values,
            name=name,
            text=text,
            textposition='inside',
            textfont=dict(size=10, color='white'),
            marker=dict(
                color=color_map[cls],
                line=dict(color=outline_map[cls], width=2)
            ),
            hovertemplate='%{x}<br>%{fullData.name}: %{y:.1f}%<extra></extra>'
        ))
    
    # Ajustar layout
    fig.update_layout(
        title=None,
        xaxis_title=col,
        yaxis_title=ylabel,
        barmode='stack',
        template=theme,
        legend_title="Estado del préstamo",
        height=height,
        width=width,
        margin=dict(l=40, r=40, t=40, b=60)
    )
    
    # Rotar etiquetas del eje X
    fig.update_xaxes(tickangle=rotation)
    
    return fig


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

# rutas
BASE_DIR = Path(__file__).resolve().parent.parent
learning_curve_path = BASE_DIR / "metadata" / "EXP_02_learning_curve.png"
cm_path = BASE_DIR / "metadata" / "EXP_02_matriz.png"
data_path = BASE_DIR / "data" / "raw" / "loan_dataset_20000.csv"
bagging_classifier_path = BASE_DIR / "img" / "BaggingClassifier.png"
column_transformer_path = BASE_DIR / "img" / "ColumnTransformer.png"
umbral_path = BASE_DIR / 'img' / 'umbral.png'

data = pd.read_csv(data_path)

st.set_page_config(
    page_title="Loan Status Prediction",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded")

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
    if st.button("Modelo", width='stretch'):
        st.session_state.pagina = "Modelo"

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
    st.dataframe(data.describe(include='all').T, width='stretch')

    st.subheader("Distribución del estado de los préstamos")
    fig_custom = plot_pie_binaria(
    df=data,
    variable='loan_paid_back',
    label1='Pagó',
    label2='No pagó',
    subtitle=None,
    colors=[ '#1e6e43', '#7a2e29'],          
    outline_colors=['#2ecc71', '#e74c3c'],  
    hole=0.4)
    st.plotly_chart(fig_custom, key="pie_custom", width='stretch')

    st.subheader("Algunas variables numéricas")
    numeric_cols = ['age', 'annual_income', 'credit_score', 'total_credit_limit']
    for col in numeric_cols:
        fig = plot_numeric_distribution(
            df=data,
            col=col,
            bins=30,
            show_kde=True,
            colorscale=paleta   
        )
        st.plotly_chart(fig, key=f"num_{col}", width='stretch')

    st.subheader("Variables según el target")
    for col in numeric_cols:
        fig = plot_boxplot_single(
            df=data,
            col=col,
            target='loan_paid_back',
            palette=paleta,          
            label_neg="No pagó",
            label_pos="Pagó"
        )
        st.plotly_chart(fig, key=f"boxplot_{col}", width='stretch')

    
    st.subheader("Distribución de variables categóricas")
    categoric_cols = ['employment_status', 'marital_status', 'education_level', 'loan_purpose']

    for col in categoric_cols:
        fig = plot_categorical(
            df=data,
            col=col,
            show_percentage=True,          
            palette=paleta)
        st.plotly_chart(fig, key=f"categ_{col}", width='stretch')

    st.subheader("Proporción de pago por variables")
    for col in categoric_cols:
        fig = plot_crosstab_single(
            df=data,
            col=col,
            target='loan_paid_back',
            label_pos="Si Pagó",
            label_neg="No Pagó",
            rotation=30,
            colors=['#7a2e29', '#1e6e43'],          
            outline_colors=['#e74c3c', '#2ecc71'],  
            theme='plotly_white'
        )
        st.plotly_chart(fig, key=f"crosstab_{col}", width='stretch')

###########################################################################################################
# MODELO
###########################################################################################################

elif st.session_state.pagina == "Modelo":
    modelo = cargar_modelo()

    st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h2 style="color: #1f77b4;">Ingrese los datos para implementar el modelo</h2>
        </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Información personal")

        # age
        age = st.number_input(
            '***Edad***',
            min_value = 18,
            max_value = 100,
            value = 30,
            step = 1)

        # gender
        gender = st.radio(
            '***Género***',
            ['Male',
            'Female',
            'Other'],
            index = None)

        # marital_status
        marital_status = st.radio(
            '***Estado Civil***',
            ['Single',
            'Married',
            'Divorced', 
            'Widowed'],
            index = None)

        # education_level
        education_level = st.radio(
            '***Nivel de educación***',
            ["Bachelor's",
            'High School',
            "Master's",
            'PhD',
            'Other'],
            index = None)

        # annual_income
        annual_income = st.slider(
            '***Ingreso anual***',
            min_value = 5000,
            max_value = 500000,
            value = 30000,
            step = 1)

        # monthly_income
        monthly_income = annual_income / 12

        # employment_status
        employment_status = st.selectbox(
            '***Ocupación***',
            options = ['Employed',
                    'Self-employed',
                    'Unemployed',
                    'Retired',
                    'Student'])

    with c2:
        st.subheader("Información financiera")

        # debt_to_income_ratio
        debt_to_income_ratio = st.number_input(
            '***Ratio deuda / ingreso***',
            min_value = 0.01,
            max_value = 0.7,
            value = 0.1,
            step = 0.01)
        
        # credit_score
        credit_score = st.slider(
            '***Score crediticio***',
            min_value = 300,
            max_value = 850,
            value = 550,
            step = 1)

        # loan_amount
        loan_amount = st.number_input(
            '***Cantidad del prestamo***',
            min_value = 500,
            max_value = 50000,
            value = 1000,
            step = 1)
        
        # loan_purpose
        loan_purpose = st.selectbox(
            '***Propósito del prestamo***',
            options = ['Debt consolidation',
                    'Car',
                    'Home',
                    'Education',
                    'Business',
                    'Medical',
                    'Vacation',
                    'Other'])
        
        # interest_rate
        interest_rate = st.number_input(
            '***Tasa de interés***',
            min_value = 3.14,
            max_value = 21.0,
            value = 12.0,
            step = 0.01)
        
        # loan_term
        loan_term= st.radio(
            '***Plazo a meses***',
            options = [36, 60])

        # installment
        installment = st.number_input(
            '***Pagos a plazos***',
            min_value = 9.0,
            max_value = 1690.0,
            value = 270.0,
            step = 0.1)
        
        # grade_subgrade
        grade_subgrade = st.selectbox(
            '***Calificación Crediticia***',
            options = ['A1',
                    'A2',
                    'A3',
                    'A4',
                    'A5',
                    'B1',
                    'B2',
                    'B3',
                    'B4',
                    'B5',
                    'C1',
                    'C2',
                    'C3',
                    'C4',
                    'C5',
                    'D1',
                    'D2',
                    'D3',
                    'D4',
                    'D5',
                    'E1',
                    'E2',
                    'E3',
                    'E4',
                    'E5',
                    'F1',
                    'F2',
                    'F3',
                    'F4',
                    'F5',])
        
        # num_of_open_accounts
        num_of_open_accounts = st.number_input(
            '***Cuentas abiertas***',
            min_value = 0,
            max_value = 15,
            value = 1,
            step = 1)
        
        # total_credit_limit
        total_credit_limit = st.number_input(
            '***Límite de Crédito***',
            min_value = 6100.0,
            max_value = 450000.0,
            value = 50000.0,
            step = 1.0)
        
        # current_balance
        current_balance = st.number_input(
            '***Saldo Actual***',
            min_value = 450,
            max_value = 355000,
            value = 25000,
            step = 1)
        
    with c3:
        st.subheader("Pagos atrasados y morosidad")
        # delinquency_history
        delinquency_history = st.number_input(
            '***Historial de morosidad***',
            min_value = 0,
            max_value = 11,
            value = 0,
            step = 1)
        
        # public_records
        public_records = st.selectbox(
            '***Registros publicos***',
            options = [0, 1, 2])
        
        # num_of_delinquencies
        num_of_delinquencies = st.number_input(
            '***Número de veces con morosidad***', 
            min_value = 0,
            max_value = 11,
            value = 0,
            step = 1)
        
        st.image(umbral_path, caption = "Umbral de desición", width='stretch')
        
        umbral = st.slider(
            '***Umbral para tomar la desición***',
            min_value = 10, 
            max_value = 90, 
            value = 50, 
            step = 1,
            help = "Bajarlo haría la predicción mas sensible y aumentarían los FP," \
            " subirlo haría la predicción más confiable pero aumentarían los FN")
        
        
        
    if modelo is not None:
        try:
            campos_requeridos = {
                'age': age,
                'gender' : gender,
                'marital_status' : marital_status,
                'education_level' : education_level,
                'annual_income' : annual_income,
                'monthly_income' : monthly_income,
                'employment_status' : employment_status,
                'debt_to_income_ratio': debt_to_income_ratio,
                'credit_score' : credit_score,
                'loan_amount' : loan_amount,
                'loan_purpose' : loan_purpose,
                'interest_rate' : interest_rate,
                'loan_term' : loan_term,
                'installment' : installment,
                'grade_subgrade' : grade_subgrade,
                'num_of_open_accounts' : num_of_open_accounts,
                'total_credit_limit' : total_credit_limit,
                'current_balance' : current_balance,
                'delinquency_history' : delinquency_history,
                'public_records' : public_records,
                'num_of_delinquencies' : num_of_delinquencies}
            
            # Verificar campos vacíos
            campos_vacios = [k for k, v in campos_requeridos.items() if v is None or v == '']
            if campos_vacios:
                st.error(f"Por favor completa los siguientes campos: {', '.join(campos_vacios)}")
                st.stop()
        
            # Construir diccionario sin loan_paid_back
            datos_usuario = campos_requeridos
            df = alimentar_pipeline(datos_usuario)       

            st.markdown("---")
            st.header("Predicción")

            # PREDICCIÓN
            if hasattr(modelo, 'predict_proba'):
                probabilidad = np.round(modelo.predict_proba(df)[0][1], 4)
                pred = 1 if probabilidad >= (umbral/100) else 0
                tabla_resultados = pd.DataFrame({
                    'Probabilidad de no pago' : [f"{1-probabilidad:.1%}"],
                    'Probabilidad de pago': [f"{probabilidad}"],
                    'Predicción' : ['Si Paga' if pred == 1 else 'No Paga']
                    })
                st.dataframe(tabla_resultados, hide_index = True)
            else:
                pred = modelo.predict(df)[0]
                probabilidad = None
                if pred == 1:
                    st.error('Potencial riesgo de que el usuario no pague')
                else:
                    st.success('Existe menor riesgo de que el usuario no pague')

        except Exception as e:
            st.error(f'Error leyedo los datos: {str(e)}')


# --- Créditos finales ---
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #9e9e9e; padding: 1rem;">
        <strong>Jair Ramos</strong> · 
        Repositorio en <a href="https://github.com/JairRamosG/Loan_Status_Prediction" target="_blank">GitHub</a> · 
    </div>
""", unsafe_allow_html=True)