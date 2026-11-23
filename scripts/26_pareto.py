import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    print("Generando Curva de Pareto...")
    df = pd.read_csv(os.path.join("..", "data", "dataset_limpio_100k.csv"))
    df_k = df.sort_values('densidad_g_cm3', ascending=False)
    df_k['acumulado'] = df_k['densidad_g_cm3'].cumsum() / df_k['densidad_g_cm3'].sum() * 100
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(df_k)), df_k['acumulado'], color='purple')
    plt.axhline(80, color='red', linestyle='--')
    plt.xlabel("Number of Materials")
    plt.ylabel("Cumulative Density (%)")
    plt.title("Pareto Chart: Density Distribution")
    plt.savefig(os.path.join("..", "final_figures", "fig26_pareto.png"), dpi=300)
    print("✅ Guardado.")

if __name__ == "__main__":
    main()