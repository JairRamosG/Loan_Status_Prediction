# Loan Status Prediction

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-1.54.0-red?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-1.6.1-orange?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/PySpark-3.5.0-yellow?style=for-the-badge&logo=apachespark&logoColor=white"/>
  <img src="https://img.shields.io/badge/Status-Production%20Ready-brightgreen?style=for-the-badge"/>
</div>

<br>

## Descripción del Proyecto

Este proyecto desarrolla un **pipeline modular de Machine Learning** para predecir el incumplimiento de préstamos en una entidad financiera. Utilizando un enfoque de **Bagging de Random Forests** combinado con **SMOTE** para manejar el desbalance de clases, se construye un modelo robusto que alcanza una **precisión del 89.6%** y un **F1-Score del 93.5%**.

La solución incluye una **arquitectura completamente configurable** mediante archivos YAML, permitiendo experimentar con diferentes hiperparámetros, transformaciones y selección de variables de forma reproducible. Además, cuenta con una **aplicación interactiva en Streamlit** que permite visualizar métricas clave y realizar predicciones en tiempo real.

---

## Objetivos

- **Predecir** si un solicitante de préstamo pagará o no su deuda.
- **Identificar** patrones de riesgo crediticio a través de variables financieras y personales.
- **Construir** un pipeline modular y reproducible para experimentación y despliegue.
- **Visualizar** resultados mediante dashboards interactivos.

---

## Características Principales

| Característica | Descripción |
|----------------|-------------|
| **Feature Engineering** | Creación de nuevas variables: grupos de edad, ratio préstamo-ingreso, indicador de mora, puntaje de gravedad, etc. |
| **Preprocesamiento Dinámico** | Escalado, log-transform, One-Hot Encoding y Ordinal Encoding configurados desde YAML. |
| **Balanceo de Clases** | SMOTE para manejar clases desbalanceadas (80% pagó / 20% no pagó). |
| **Modelo** | BaggingClassifier con base RandomForestClassifier, optimizado con RandomizedSearchCV. |
| **Evaluación** | Matriz de confusión, curvas de aprendizaje, métricas de clasificación (precisión, recall, F1-score). |
| **Reproducibilidad** | Configuración completa en YAML, semillas fijas y logs detallados. |
| **Despliegue** | Aplicación interactiva en Streamlit para visualización y predicciones en tiempo real. |

---

## Resultados del Mejor Modelo

| Métrica | Valor |
|---------|-------|
| **Precisión (Precision)** | 89.6% |
| **Recall** | 97.8% |
| **F1-Score** | 93.5% |
| **Accuracy** | 89.2% |

> *El modelo optimizado demuestra una alta capacidad para detectar correctamente los casos de impago (recall del 97.8%), lo que lo hace especialmente útil para la gestión del riesgo crediticio.*

---

## Tecnologías Utilizadas

| Área | Tecnologías |
|------|-------------|
| **Lenguaje** | Python 3.11 |
| **Machine Learning** | Scikit-learn, XGBoost, Random Forest, SMOTE |
| **Big Data** | PySpark (análisis exploratorio) |
| **Visualización** | Streamlit, Matplotlib, Seaborn, Plotly |
| **Procesamiento de Datos** | Pandas, NumPy |
| **Versionamiento** | Git, Git LFS |
| **Configuración** | YAML |

---
## Metodología

### Pipeline de Preprocesamiento

1. **Feature Engineering**: Creación de nuevas variables
   - Grupos de edad (age_group)
   - Ratio préstamo-ingreso (loan_to_income)
   - Indicador de morosidad (has_delinquency_history)
   - Puntaje de gravedad (severity_score)
   - Proporción de pago (payment_income)

2. **ColumnTransformer**: Transformaciones específicas por tipo de dato
   - **Numéricas**: escalado (StandardScaler) y log-transform para variables con sesgo positivo
   - **Categóricas nominales**: OneHotEncoder con `drop='first'`
   - **Categóricas ordinales**: OrdinalEncoder con categorías predefinidas

3. **SMOTE**: Balanceo de clases para manejar el desbalance (80% pagó / 20% no pagó)

4. **Bagging de Random Forests**: Modelo final
   - 500 estimadores base (RandomForests)
   - Muestreo bootstrap con 50% de las muestras
   - Paralelización con `n_jobs=-1`

### Optimización de Hiperparámetros

- **RandomizedSearchCV** con 3 folds
- **Métrica**: Matthews Correlation Coefficient (MCC)
- **Búsqueda sobre**:
  - Número de estimadores en Bagging: [100, 300, 500]
  - Tamaño de muestra por estimador: [0.3, 0.5, 0.7]
  - Parámetros del Random Forest base:
    - `n_estimators`: [30, 50]
    - `max_leaf_nodes`: [20, 30]
    - `min_samples_split`: [2, 5, 10]
    - `max_features`: ['sqrt', 0.5]

### Evaluación

- **Métricas**: Precisión, Recall, F1-Score, AUC-ROC
- **Validación**: Conjunto de prueba independiente (20% de los datos)
- **Curvas de aprendizaje**: Diagnóstico de sesgo-varianza
- **Matriz de confusión**: Análisis de falsos positivos y falsos negativos

---

## Trabajo a Futuro

Con el objetivo de mejorar este trabajo, se plantean las siguientes líneas de cambio:

### Mejoras en el Modelado

- **Implementación de Deep Learning**: Explorar arquitecturas de redes neuronales (MLP, TabNet) para capturar relaciones no lineales más complejas.
- **Ensembles avanzados**: Probar combinaciones de modelos como XGBoost + LightGBM + CatBoost con stacking o voting classifiers.
- **Optimización de hiperparámetros**: Ampliar la búsqueda con técnicas más eficientes como Optuna, aumentando el número de iteraciones.

### Ingeniería de Características

- **Nuevas transformaciones**: Aplicar PCA o selección automática de características con técnicas como Boruta o SHAP para reducir dimensionalidad.
- **Feature Store**: Implementar un repositorio centralizado de características para facilitar la reproducibilidad y reutilización.

