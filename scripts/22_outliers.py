import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("Generando Analisis de Outliers...")
    df = pd.read_csv(os.path.join("..", "data", "dataset_limpio_100k.csv"))
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df[['densidad_g_cm3', 'volumen_A3', 'nsites']])
    plt.yscale('log')
    plt.title("Boxplot of Material Properties (Outlier Detection)")
    plt.savefig(os.path.join("..", "final_figures", "fig22_outliers.png"), dpi=300)
    print("✅ Guardado.")

if __name__ == "__main__":
    main()