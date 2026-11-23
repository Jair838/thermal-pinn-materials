import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("Generando Violin Plot...")
    df = pd.read_csv(os.path.join("..", "data", "dataset_limpio_100k.csv"))
    plt.figure(figsize=(10, 6))
    sns.violinplot(x=df['densidad_g_cm3'], color="cyan")
    plt.xlabel("Density (g/cm³)")
    plt.title("Density Distribution (Violin Plot)")
    plt.savefig(os.path.join("..", "final_figures", "fig30_violin.png"), dpi=300)
    print("✅ Guardado.")

if __name__ == "__main__":
    main()