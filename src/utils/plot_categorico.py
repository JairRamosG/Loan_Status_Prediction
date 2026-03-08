import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import seaborn as sns
import matplotlib.pyplot as plt
import math

def plot_frecuencias_categoricas(df, categoric_cols, show_percentage=False):

    sns.set_theme(style="whitegrid")

    n = len(categoric_cols)
    cols = 2
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 4))
    axes = axes.flatten()

    total = len(df)

    for i, col in enumerate(categoric_cols):

        orden = df[col].value_counts().index

        ax = sns.countplot(
            y = col,
            data = df,
            order = orden,
            hue = col,
            palette = "viridis",
            edgecolor = "black",
            legend = False,
            ax = axes[i]
        )

        for p in ax.patches:

            width = p.get_width()
            y = p.get_y() + p.get_height()/2

            if show_percentage:
                porcentaje = 100 * width / total
                texto = f'{int(width)} ({porcentaje:.1f}%)'
            else:
                texto = f'{int(width)}'

            ax.text(width + 0.5, y, texto, va='center')

        axes[i].set_title(f'Frecuencia de {col}', fontsize=13, fontweight="bold")
        axes[i].set_xlabel("Frecuencia")
        axes[i].set_ylabel(col)

    # eliminar ejes vacíos si hay
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("Distribución de Variables Categóricas", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_crosstab_categorica(
    df,
    variable,
    target,
    titulo=None,
    xlabel=None,
    ylabel="Proporción",
    label_pos="Pagó",
    label_neg="No pagó",
    rotation=30
):
    
    # Crosstab normalizado
    tabla = pd.crosstab(
        df[variable],
        df[target],
        normalize='index'
    )
    
    fig, ax = plt.subplots(figsize=(8,5))

    tabla.plot(
        kind='bar',
        stacked=True,
        color=['#e74c3c', '#2ecc71'],
        edgecolor='black',
        ax=ax
    )

    # Título
    if titulo is None:
        titulo = f'Proporción de pago por {variable}'
    ax.set_title(titulo, fontsize=15, fontweight='bold')

    # Labels
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel if xlabel else variable)

    # Leyenda
    ax.legend([label_neg, label_pos], title='Estado del préstamo')

    # Porcentajes dentro de barras
    for container in ax.containers:
        labels = [
            f"{v.get_height()*100:.1f}%" if v.get_height() > 0 else ""
            for v in container
        ]
        ax.bar_label(container, labels=labels, label_type='center', fontsize=9)

    plt.xticks(rotation=rotation)
    sns.despine()
    plt.tight_layout()
    plt.show()

def plot_kde_por_clase(df, numeric_cols, target):
    sns.set_theme(style="whitegrid")

    n = len(numeric_cols)
    cols = 2
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(15, rows*4))
    axes = axes.flatten()

    colores_medias = {
        0: "#d62728",
        1: "#1f77b4"
    }

    for i, col in enumerate(numeric_cols):

        ax = sns.kdeplot(
            data=df,
            x=col,
            hue=target,
            fill=True,
            common_norm=False,
            alpha=0.35,
            linewidth=2,
            palette="Set2",
            ax=axes[i]
        )

        for clase, color in colores_medias.items():

            mean_val = df[df[target] == clase][col].mean()

            axes[i].axvline(
                mean_val,
                linestyle="--",
                linewidth=3,
                color=color,
                label=f"Media clase {clase}: {mean_val:.2f}"
            )

        axes[i].set_title(f'Distribución de {col} por clase', fontsize=12, fontweight="bold")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Densidad")

    # eliminar ejes vacíos
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("Distribución de Variables Numéricas por Clase", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_hist_variable_binaria(df, column, x_label, y_label, positive_tick, negative_tick, titulo):
    plt.figure(figsize=(8,7))
    ax = sns.countplot(
        x=column,
        data=df,
        palette=['#e74c3c', '#2ecc71']
    )

    # Título
    plt.title(titulo, fontsize=16, fontweight='bold')

    # Etiquetas
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)

    # Cambiar etiquetas del eje x
    ax.set_xticklabels([negative_tick, positive_tick])

    # Agregar valores encima de las barras
    total = len(df)
    for p in ax.patches:
        count = int(p.get_height())
        percentage = 100 * count / total
        ax.annotate(f'{count}\n({percentage:.1f}%)',
                    (p.get_x() + p.get_width()/2, p.get_height()),
                    ha='center', va='bottom',
                    fontsize=11)

    sns.despine()
    plt.tight_layout()
    plt.show()
            