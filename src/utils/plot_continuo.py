import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_distribucion_box(df, numeric_cols, bins=30):
    
    sns.set_theme(style="whitegrid", palette="deep")
    
    for col in numeric_cols:

        mean_val = df[col].mean()
        median_val = df[col].median()
        skew_val = df[col].skew()

        fig, axes = plt.subplots(1, 2, figsize=(14,4))

        # Histograma
        sns.histplot(
            df[col],
            kde=True,
            bins=bins,
            color="#4C72B0",
            edgecolor="black",
            alpha=0.75,
            ax=axes[0]
        )

        # Línea de la media
        axes[0].axvline(
            mean_val,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Media = {mean_val:.2f}"
        )

        # Línea de la mediana
        axes[0].axvline(
            median_val,
            color="green",
            linestyle="-.",
            linewidth=2,
            label=f"Mediana = {median_val:.2f}"
        )

        axes[0].set_title(f'Distribución de {col}', fontsize=13, fontweight='bold')
        axes[0].legend()

        # Sesgo
        axes[0].text(
            0.95, 0.85,
            f"Sesgo = {skew_val:.2f}",
            transform=axes[0].transAxes,
            ha='right',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray')
        )

        # Boxplot
        sns.boxplot(
            x=df[col],
            ax=axes[1],
            color="#55A868",
            width=0.5,
            fliersize=4
        )

        axes[1].axvline(mean_val, color="red", linestyle="--", linewidth=2)
        axes[1].set_title(f'Boxplot de {col}', fontsize=13, fontweight='bold')

        plt.suptitle(f'Análisis de {col}', fontsize=15, fontweight='bold')
        plt.tight_layout()
        plt.show()

def plot_distribuciones_kde(df, numeric_cols):

    n_vars = len(numeric_cols)
    n_cols = 4
    n_rows = (n_vars + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4))

    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, col in enumerate(numeric_cols):
        row_idx = idx // n_cols
        col_idx = idx % n_cols
        
        ax = axes[row_idx, col_idx]
        
        # Histograma con KDE
        sns.histplot(df[col], kde=True, ax=ax, bins=30, 
                    color='steelblue', edgecolor='black', alpha=0.7)
        
        # Calcular estadísticas
        stats = {
            'mean': df[col].mean(),
            'median': df[col].median(),
            'std': df[col].std(),
            'skew': df[col].skew()
        }
        
        # Añadir líneas
        ax.axvline(stats['mean'], color='red', linestyle='--', linewidth=2, 
                alpha=0.8, label=f"μ={stats['mean']:.2f}")
        ax.axvline(stats['median'], color='green', linestyle=':', linewidth=2,
                alpha=0.8, label=f"med={stats['median']:.2f}")
        
        # Colorear título según skewness
        skew_color = 'red' if abs(stats['skew']) > 1 else 'orange' if abs(stats['skew']) > 0.5 else 'green'
        
        ax.set_title(f'{col}\nskew={stats["skew"]:.2f}', fontsize=11, 
                    fontweight='bold', color=skew_color)
        ax.set_xlabel('')
        ax.set_ylabel('Densidad', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Ocultar ejes vacíos
    for idx in range(n_vars, n_rows * n_cols):
        row_idx = idx // n_cols
        col_idx = idx % n_cols
        axes[row_idx, col_idx].set_visible(False)

    plt.suptitle('DISTRIBUCIÓN CON KDE DE TODAS LAS VARIABLES NUMÉRICAS', 
                fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()
        

def plot_box_plots(df, numeric_cols):
    n_vars = len(numeric_cols)
    n_cols = 4
    n_rows = (n_vars + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4))

    axes = axes.flatten()  # simplifica el manejo de índices

    for idx, col in enumerate(numeric_cols):

        ax = axes[idx]

        # Boxplot
        sns.boxplot(x=df[col], ax=ax, color='steelblue')

        # Estadísticas
        mean_val = df[col].mean()
        median_val = df[col].median()
        std_val = df[col].std()
        skew_val = df[col].skew()

        # Líneas de referencia
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                label=f"μ={mean_val:.2f}")
        
        ax.axvline(median_val, color='green', linestyle=':', linewidth=2,
                label=f"med={median_val:.2f}")

        # Color del título según skewness
        if abs(skew_val) > 1:
            skew_color = 'red'
        elif abs(skew_val) > 0.5:
            skew_color = 'orange'
        else:
            skew_color = 'green'

        ax.set_title(f'{col}\nskew={skew_val:.2f}',
                    fontsize=11,
                    fontweight='bold',
                    color=skew_color)

        ax.set_xlabel(col)
        ax.set_ylabel('')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # Ocultar ejes sobrantes
    for idx in range(n_vars, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('BOXPLOT DE TODAS LAS VARIABLES NUMÉRICAS',
                fontsize=18,
                fontweight='bold')

    plt.tight_layout()
    plt.show()

def plot_violin(df, numeric_cols):

    n_vars = len(numeric_cols)
    n_cols = 4
    n_rows = (n_vars + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4))

    axes = axes.flatten()

    for idx, col in enumerate(numeric_cols):

        ax = axes[idx]

        # Violin plot
        sns.violinplot(x=df[col], ax=ax, color='steelblue', inner='quartile')

        # Estadísticas
        mean_val = df[col].mean()
        median_val = df[col].median()
        std_val = df[col].std()
        skew_val = df[col].skew()

        # Líneas de referencia
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                   label=f"μ={mean_val:.2f}")

        ax.axvline(median_val, color='green', linestyle=':', linewidth=2,
                   label=f"med={median_val:.2f}")

        # Color del título según skewness
        if abs(skew_val) > 1:
            skew_color = 'red'
        elif abs(skew_val) > 0.5:
            skew_color = 'orange'
        else:
            skew_color = 'green'

        ax.set_title(f'{col}\nskew={skew_val:.2f}',
                     fontsize=11,
                     fontweight='bold',
                     color=skew_color)

        ax.set_xlabel(col)
        ax.set_ylabel('')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # Ocultar ejes sobrantes
    for idx in range(n_vars, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('VIOLIN PLOTS DE TODAS LAS VARIABLES NUMÉRICAS',
                 fontsize=18,
                 fontweight='bold')

    plt.tight_layout()
    plt.show()

