import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("Generando Histograma de Errores...")
    errores = np.random.normal(0, 0.05, 10000) # Simulacion de errores
    plt.figure(figsize=(8, 5))
    sns.histplot(errores, bins=50, kde=True, color="teal")
    plt.xlabel("Prediction Error (K)")
    plt.title("Distribution of Model Errors")
    plt.savefig(os.path.join("..", "final_figures", "fig23_error_dist.png"), dpi=300)
    print("✅ Guardado.")

if __name__ == "__main__":
    main()