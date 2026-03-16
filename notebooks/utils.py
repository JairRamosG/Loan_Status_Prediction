from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

class Feature_Engineering(BaseEstimator, TransformerMixin):
    ''' 
    Creación de las nuevas características a partir de los datos originales
    '''
    def __init__(self,
                 create_age_group = True,
                 age_bins = None,
                 age_labels = None,
                 **kwargs):
        
        self.create_age_group = create_age_group
        self.age_bins = age_bins
        self.age_labels = age_labels
        pass
    
    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        X_new = X.copy()

        # Age_group
        if self.create_age_group and self.age_bins is not None and self.age_labels is not None:
            X_new['age_group'] = pd.cut(X_new['age'], bins = self.age_bins, labels = self.age_labels)


        # lo mismo apra todas las variables originales que necesiten modificación
        # Lo mismo para todas las variabels que se van a inventar

        return X_new