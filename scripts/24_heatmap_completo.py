import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("Generando Heatmap Completo...")
    df = pd.read_csv(os.path.join("..", "data", "dataset_limpio_100k.csv"))
    cols = ['densidad_g_cm3', 'volumen_A3', 'nsites', 'band_gap_eV', 'modulo_volumetrico_GPa', 'densidad_atomica']
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[cols].corr(), annot=True, cmap="RdYlBu", fmt=".2f")
    plt.title("Global Correlation Heatmap")
    plt.savefig(os.path.join("..", "final_figures", "fig24_heatmap_full.png"), dpi=300)
    print("✅ Guardado.")

if __name__ == "__main__":
    main()