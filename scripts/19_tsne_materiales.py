import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

def main():
    print("=== GENERANDO MAPA T-SNE ===")
    ruta_fig = os.path.join("..", "final_figures")
    ruta_datos = os.path.join("..", "data", "dataset_limpio_100k.csv")
    
    df = pd.read_csv(ruta_datos)
    df_sample = df.sample(n=min(3000, len(df)), random_state=42)
    features = ['densidad_g_cm3', 'volumen_A3', 'band_gap_eV', 'modulo_volumetrico_GPa']
    df_sample = df_sample.dropna(subset=features)
    
    X = df_sample[features].values
    X_scaled = StandardScaler().fit_transform(X)
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X_scaled)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=df_sample['densidad_g_cm3'], cmap='viridis', s=20, alpha=0.7)
    plt.colorbar(label="Density (g/cm3)")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.title("Material Clustering (t-SNE Space)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.savefig(os.path.join(ruta_fig, "fig17_tsne_materiales.png"), dpi=300, bbox_inches="tight")
    print("✅ Grafico guardado.")

if __name__ == "__main__":
    main()