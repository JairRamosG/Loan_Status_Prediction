import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

def cohen_ranking(df, target, numeric_cols=None, plot=True):
    
    # seleccionar variables numéricas automáticamente
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        numeric_cols.remove(target)
    
    # detectar clases automáticamente
    clases = df[target].dropna().unique()
    
    if len(clases) != 2:
        raise ValueError("Cohen's d solo funciona con variables objetivo binarias.")
    
    g1 = df[df[target] == clases[0]]
    g2 = df[df[target] == clases[1]]
    
    def cohen_d(x, y):
        n1, n2 = len(x), len(y)
        var1, var2 = np.var(x, ddof=1), np.var(y, ddof=1)
        
        pooled_sd = np.sqrt(((n1 - 1)*var1 + (n2 - 1)*var2) / (n1 + n2 - 2))
        d = (np.mean(x) - np.mean(y)) / pooled_sd
        
        return d
    
    resultados = {}
    
    for col in numeric_cols:
        d = cohen_d(g1[col], g2[col])
        resultados[col] = abs(d)
    
    ranking = sorted(resultados.items(), key=lambda x: x[1], reverse=True)
    
    ranking_df = pd.DataFrame(ranking, columns=["Variable", "Cohen_d"])
    
    print(tabulate(ranking_df, headers="keys", tablefmt="fancy_grid", showindex=False))
    
    if plot:
        vars_ordenadas = ranking_df["Variable"]
        d_values = ranking_df["Cohen_d"]
        
        plt.figure(figsize=(8,6))
        plt.barh(vars_ordenadas, d_values)
        plt.axvline(0.2, color='red', linestyle='--', label='d = 0.2 (bajo)')
        plt.axvline(0.5, color='orange', linestyle='--', label='d = 0.5 (medio)')
        plt.axvline(0.8, color='green', linestyle='--', label='d = 0.8 (alto)')
        
        plt.xlabel("Cohen's d (valor absoluto)")
        plt.title(f"Separación de variables según {target}")
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    return ranking_df