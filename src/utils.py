from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

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