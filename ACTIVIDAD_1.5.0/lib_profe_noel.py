import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def prueba_normalidad(serie, alpha=0.05):
    datos = serie.dropna()

    print(f"Análisis de normalidad para la variable: {serie.name}")
    print(f"Número de valores analizados: {len(datos)}")

    # Shapiro-Wilk Test (mejor para n < 5000)
    shapiro_stat, shapiro_p = stats.shapiro(datos.sample(min(len(datos), 5000), random_state=42))
    print(f"\nShapiro-Wilk:")
    print(f"  Estadístico = {shapiro_stat:.4f}, p = {shapiro_p:.4f}")
    if shapiro_p < alpha:
        print("No sigue una distribución normal (rechaza H0)")
    else:
        print("Podría ser normal (no se rechaza H0)")

    # D’Agostino and Pearson’s Test
    dagostino_stat, dagostino_p = stats.normaltest(datos)
    print(f"\nD’Agostino-Pearson:")
    print(f"  Estadístico = {dagostino_stat:.4f}, p = {dagostino_p:.4f}")
    if dagostino_p < alpha:
        print("No sigue una distribución normal (rechaza H0)")
    else:
        print("Podría ser normal (no se rechaza H0)")

def analizar_variable(serie):
      
    # Crear figura
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    fig.suptitle(f"Análisis de la variable: {serie.name}", fontsize=16)

    # Pregunta 1: ¿Es numérica o categórica?
    if pd.api.types.is_numeric_dtype(serie):
        tipo = "Numérica"
    else:
        tipo = "Categórica"

    axes[0].axis('off')  # No es un gráfico, solo texto
    axes[0].text(0.5, 0.5, f"Tipo: {tipo}", ha='center', va='center', fontsize=14, weight='bold')

    # Pregunta 2: Distribución. Histograma para numericas, conteo para categóricas. 
    if tipo == "Numérica":
        sns.histplot(serie.dropna(), kde=True, ax=axes[1])
        axes[1].set_title("Distribución")
    else:
        serie.value_counts().plot(kind='bar', ax=axes[1])
        axes[1].set_title("Frecuencia por categoría")

    # Pregunta 3: Outliers (solo para numéricas)
    if tipo == "Numérica":
        sns.boxplot(x=serie, ax=axes[2])
        axes[2].set_title("Boxplot (outliers)")
    else:
        axes[2].axis('off')
        axes[2].text(0.5, 0.5, "No aplica (categórica)", ha='center', va='center', fontsize=12)

    plt.tight_layout()
    plt.show()