from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import (accuracy_score, recall_score, balanced_accuracy_score, precision_score, f1_score, matthews_corrcoef)
from sklearn.model_selection import learning_curve
from pathlib import Path

class Feature_Engineering(BaseEstimator, TransformerMixin):
    ''' 
    Creación de las nuevas características a partir de los datos originales
    '''
    def __init__(
            self,
            create_age_group = True,
            age_bins = None,
            age_labels = None,
            create_loan_to_income = True,
            create_has_delinquency_history = True,
            create_severity_score = True,
            create_payment_income = True):
        
        self.create_age_group = create_age_group
        self.age_bins = age_bins
        self.age_labels = age_labels
        self.create_loan_to_income = create_loan_to_income
        self.create_has_delinquency_history = create_has_delinquency_history
        self.create_severity_score = create_severity_score
        self.create_payment_income = create_payment_income
        pass
    
    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        X_new = X.copy()

        # age_group
        if self.create_age_group and self.age_bins is not None and self.age_labels is not None:
            X_new['age_group'] = pd.cut(X_new['age'], bins = self.age_bins, labels = self.age_labels)

        # loan_to_incomel
        if self.create_loan_to_income:
            X_new['loan_to_income'] = X_new['loan_amount'] - X_new['annual_income']

        # has_delinquency_history
        if self.create_has_delinquency_history:
            X_new['has_delinquency_history'] = (X_new['delinquency_history']>0).astype(int)
        
        # severity_score
        if self.create_severity_score:
            X_new['severity_score'] = X_new['num_of_delinquencies'] + X_new['public_records']
        
        # payment_income
        if self.create_payment_income:
            X_new['payment_income'] = X_new['installment'] / X_new['monthly_income'].replace(0, 1)

        return X_new


def save_confusion_matrix(y_test, y_pred, ruta_img):
    """
    Genera y muestra una matriz de confusión personalizada
    con la disposición [[TP, FN], [FP, TN]].
    """

    cm = confusion_matrix(y_test, y_pred)

    TN, FP = cm[0, 0], cm[0, 1]
    FN, TP = cm[1, 0], cm[1, 1]

    cm_custom = np.array([
        [TP, FN],
        [FP, TN]
    ])

    cm_df = pd.DataFrame(
        cm_custom,
        index=['Pred Positivo', 'Pred Negativo'],
        columns=['Real Positivo', 'Real Negativo']
    )

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm_custom,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Real +', 'Real -'],
        yticklabels=['Pred +', 'Pred -']
    )

    plt.xlabel('Clase real')
    plt.ylabel('Clase predicha')
    plt.title('Matriz de Confusión (TP FN / FP TN)')
    plt.tight_layout()
    plt.savefig(ruta_img)

def save_medidas_biclase(y_test, y_pred, ruta_medidas):
    '''
    Calculas las medidas de desempeño solicitadas en las instrucciones
    regresa una tabla con las medidas
    '''

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0 

    metrics = {

        'Accuracy': np.round(accuracy_score(y_test, y_pred), 4),
        'Error Rate': np.round(1 - accuracy_score(y_test, y_pred), 4),
        'Recall (Sensitivity)': np.round(recall_score(y_test, y_pred), 4),
        'Specificity': np.round(specificity, 4),
        'Balanced Accuracy': np.round(balanced_accuracy_score(y_test, y_pred), 4),
        'Precision': np.round(precision_score(y_test, y_pred), 4),
        'F1 Score': np.round(f1_score(y_test, y_pred), 4),
        'MCC': np.round(matthews_corrcoef(y_test, y_pred), 4)
    }

    #resultados =  pd.DataFrame(list(metrics.items()), columns = ['Medida', 'Valor'])
    resultados = pd.DataFrame(metrics.items(), columns = ['Medida', 'Valor'])
    resultados.to_csv(ruta_medidas)
    return resultados

def save_learning_curve(
    estimator,
    X,
    y,
    ruta_img,
    scoring='accuracy',
    cv=5,
    train_sizes=np.linspace(0.1, 1.0, 5),
    n_jobs=-1,
    verbose=0,
    title='Curva de Aprendizaje',
    ylim=None,
    figsize=(8, 6),
    random_state=42
):
    """
    Genera y guarda la curva de aprendizaje para el estimador dado.

    Parámetros
    ----------
    estimator : objeto con API de scikit-learn (pipeline)
        Modelo ya construido (no entrenado).
    X : array-like, shape (n_samples, n_features)
        Datos de entrenamiento.
    y : array-like, shape (n_samples,)
        Variable objetivo.
    scoring : str o callable, default='accuracy'
        Métrica de evaluación.
    cv : int o cross-validator, default=5
        Número de folds o generador de validación.
    train_sizes : array-like, default=np.linspace(0.1, 1.0, 5)
        Porcentajes del conjunto de entrenamiento a usar.
    n_jobs : int, default=-1
        Número de procesos paralelos.
    verbose : int, default=0
        Nivel de detalle.
    title : str, default='Curva de Aprendizaje'
        Título del gráfico.
    ylim : tuple, optional
        Límites del eje Y.
    figsize : tuple, default=(8, 6)
        Tamaño de la figura.
    random_state : int, default=42
        Semilla para reproducibilidad.
    save_path : str o Path, optional
        Ruta completa donde guardar la imagen. Si no se proporciona, no se guarda.
    dpi : int, default=100
        Resolución de la imagen guardada.
    show : bool, default=True
        Si se muestra la gráfica en pantalla (False si solo se quiere guardar).

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figura generada.
    """
    # Crear scorer a partir de string
    if isinstance(scoring, str):
        from sklearn.metrics import get_scorer
        scorer = get_scorer(scoring)
    else:
        scorer = scoring

    # Calcular curva de aprendizaje
    train_sizes_abs, train_scores, test_scores = learning_curve(
        estimator=estimator,
        X=X,
        y=y,
        train_sizes=train_sizes,
        cv=cv,
        scoring=scorer,
        n_jobs=n_jobs,
        verbose=verbose,
        random_state=random_state
    )

    # Estadísticas
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Graficar
    fig, ax = plt.subplots(figsize=figsize)
    ax.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std,
                    alpha=0.1, color="blue")
    ax.fill_between(train_sizes_abs, test_mean - test_std, test_mean + test_std,
                    alpha=0.1, color="orange")
    ax.plot(train_sizes_abs, train_mean, 'o-', color="blue", label="Entrenamiento")
    ax.plot(train_sizes_abs, test_mean, 'o-', color="orange", label="Validación")
    ax.set_xlabel("Tamaño del conjunto de entrenamiento")
    ax.set_ylabel("Puntuación (scoring)")
    ax.set_title(title)
    if ylim:
        ax.set_ylim(ylim)
    ax.legend(loc="best")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(ruta_img)
    plt.close(fig)