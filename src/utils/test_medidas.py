import numpy as np

def cohen_d(x, y):
    """
    Calcula la d de Cohen para dos grupos independientes.
    x: grupo 1 (ej. loan_paid_back=1)
    y: grupo 0 (ej. loan_paid_back=0)
    """
    n1, n2 = len(x), len(y)
    var1, var2 = np.var(x, ddof=1), np.var(y, ddof=1)
    
    # Desviación estándar combinada (pooled)
    pooled_se = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    d = (np.mean(x) - np.mean(y)) / pooled_se
    return d