from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

class Feature_Engineering(BaseEstimator, TransformerMixin):
    ''' 
    Creación de las nuevas características a partir de los datos originales
    '''
    def __init__(self):
        pass
    
    def fit(self, X, y = None):
        return self
    def transform(self, X):
        X = X.copy()

        # Aspecto financiero
        X['loan_to_income'] = X['loan_amount'] - X['annual_income']

        # Incumplimientos
        X['has_delinquency_history'] = (X['delinquency_history'] > 0).astype(int)
        X['sevetity_score'] = X['num_of_delinquencies'] + X['public_records']

        ## Pagos
        X['payment_income'] = X['installment'] / X['monthly_income']

        return X