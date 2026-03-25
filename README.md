# 🏦 Loan Status Prediction

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-1.54.0-red?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-1.6.1-orange?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/PySpark-3.5.0-yellow?style=for-the-badge&logo=apachespark&logoColor=white"/>
  <img src="https://img.shields.io/badge/Status-Production%20Ready-brightgreen?style=for-the-badge"/>
</div>

<br>

## 📌 Descripción del Proyecto

Este proyecto desarrolla un **pipeline modular de Machine Learning** para predecir el incumplimiento de préstamos en una entidad financiera. Utilizando un enfoque de **Bagging de Random Forests** combinado con **SMOTE** para manejar el desbalance de clases, se construye un modelo robusto que alcanza una **precisión del 89.6%** y un **F1-Score del 93.5%**.

La solución incluye una **arquitectura completamente configurable** mediante archivos YAML, permitiendo experimentar con diferentes hiperparámetros, transformaciones y selección de variables de forma reproducible. Además, cuenta con una **aplicación interactiva en Streamlit** que permite visualizar métricas clave y realizar predicciones en tiempo real.

---

## 🎯 Objetivos

- **Predecir** si un solicitante de préstamo pagará o no su deuda.
- **Identificar** patrones de riesgo crediticio a través de variables financieras y personales.
- **Construir** un pipeline modular y reproducible para experimentación y despliegue.
- **Visualizar** resultados mediante dashboards interactivos.

---

## 🚀 Características Principales

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

## 📊 Resultados del Mejor Modelo

| Métrica | Valor |
|---------|-------|
| **Precisión (Precision)** | 89.6% |
| **Recall** | 97.8% |
| **F1-Score** | 93.5% |
| **Accuracy** | 89.2% |

> *El modelo optimizado demuestra una alta capacidad para detectar correctamente los casos de impago (recall del 97.8%), lo que lo hace especialmente útil para la gestión del riesgo crediticio.*

---

## 📁 Estructura del Proyecto
