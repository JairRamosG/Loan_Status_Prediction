import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import seaborn as sns
import matplotlib.pyplot as plt

def plot_categorical_frequencies(df, categoric_cols, show_percentage=False):

    sns.set_theme(style="whitegrid")

    for col in categoric_cols:

        plt.figure(figsize=(10,4))

        orden = df[col].value_counts().index

        ax = sns.countplot(
            y=col,
            data=df,
            order=orden,
            palette="viridis",
            edgecolor="black"
        )

        total = len(df)

        # agregar valores en las barras
        for p in ax.patches:

            width = p.get_width()
            y = p.get_y() + p.get_height()/2

            if show_percentage:
                porcentaje = 100 * width / total
                texto = f'{int(width)} ({porcentaje:.1f}%)'
            else:
                texto = f'{int(width)}'

            ax.text(
                width + 0.5,
                y,
                texto,
                va='center'
            )

        plt.title(f'Frecuencia de {col}', fontsize=14, fontweight="bold")
        plt.xlabel("Frecuencia")
        plt.ylabel(col)

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
            