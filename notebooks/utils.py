from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class Feature_Engineering(BaseEstimator, TransformerMixin):
    ''' 
    Creación de las nuevas características a partir de los datos originales
    '''
    def __init__(self,
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


df_test = pd.DataFrame({
    'age': [25, 45, 60],
    'loan_amount': [10000, 20000, 15000],
    'annual_income': [30000, 50000, 40000],
    'delinquency_history': [0, 2, 1],
    'num_of_delinquencies': [0, 2, 1],
    'public_records': [0, 1, 0],
    'installment': [300, 500, 400],
    'monthly_income': [2500, 4000, 3500]
})

fe = Feature_Engineering(age_bins=[0,30,50,100], age_labels=['joven','adulto','mayor'])
df_result = fe.transform(df_test)
print(df_result)