from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

class Feature_Engineering(BaseEstimator, TransformerMixin):
    ''' 
    Creación de las nuevas características a partir de los datos originales
    '''
    def __init__(self, age_bins = None, age_labels = None, **kwargs):
        self.age_bins = age_bins
        self.age_labels = age_labels
        pass
    
    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        X_new = X.copy()

        # Age_group
        if self.age_bins is not None and self.age_bins is not None:
            if 'age' in X.columns():
                X_new['age_group'] = pd.cut(X_new['age'], bins = self.age_bins, labels = self.age_labels)

        # Aspecto financiero
        X_new['loan_to_income'] = X_new['loan_amount'] - X_new['annual_income']

        # Incumplimientos
        X_new['has_delinquency_history'] = (X_new['delinquency_history'] > 0).astype(int)
        X_new['sevetity_score'] = X_new['num_of_delinquencies'] + X_new['public_records']

        ## Pagos
        X_new['payment_income'] = X_new['installment'] / X_new['monthly_income']

        return X_new