import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def main():
    print("Generando Scatter 3D...")
    df = pd.read_csv(os.path.join("..", "data", "dataset_limpio_100k.csv")).head(1000)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df['densidad_g_cm3'], df['volumen_A3'], df['band_gap_eV'], c='purple', alpha=0.5)
    ax.set_xlabel('Density')
    ax.set_ylabel('Volume')
    ax.set_zlabel('Band Gap')
    plt.title("3D Material Property Space")
    plt.savefig(os.path.join("..", "final_figures", "fig27_scatter_3d.png"), dpi=300)
    print("✅ Guardado.")

if __name__ == "__main__":
    main()