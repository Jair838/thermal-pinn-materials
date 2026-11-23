import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def main():
    print("=== GENERANDO GRAFICAS EXTRA DE ALTO IMPACTO ===")
    
    # Rutas
    ruta_datos = os.path.join("..", "data", "dataset_gigante.csv")
    ruta_figuras = os.path.join("..", "final_figures")
    
    # 1. Cargar una muestra (10 millones de filas) usando chunks
    print("Cargando datos masivos...")
    bloques = []
    with pd.read_csv(ruta_datos, chunksize=5000000) as reader:
        for i, chunk in enumerate(reader):
            bloques.append(chunk.sample(frac=0.1)) # 10% de cada bloque
            if i >= 19: break # 20 bloques = 10 millones de filas
            
    df = pd.concat(bloques, ignore_index=True)
    print(f"Datos cargados: {len(df):,} filas de {df['formula'].nunique():,} materiales.")

    # 2. Gráfica 1: Top 15 Materiales con Mayor Capacidad Térmica (k)
    print("Generando: Top Materiales por Conductividad Termica...")
    # Agrupamos por material y sacamos el promedio
    top_k = df.groupby('formula')['thermal_conductivity_W_mK'].mean().sort_values(ascending=False).head(15)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_k.values, y=top_k.index, palette="viridis")
    plt.xlabel("Average Thermal Conductivity (W/m·K)")
    plt.ylabel("Material Formula")
    plt.title("Top 15 Materials by Thermal Conductivity")
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(ruta_figuras, "fig7_top_thermal_conductors.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Gráfica 2: Scatter de Densidad vs Temperatura Máxima
    print("Generando: Relacion Densidad vs Temperatura...")
    # Tomamos solo 5,000 puntos al azar para que el scatter no sea una mancha negra
    df_scatter = df.sample(n=5000, random_state=42)
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_scatter, x="density_kg_m3", y="temperature_K", 
                    hue="thermal_conductivity_W_mK", palette="coolwarm", alpha=0.7, edgecolor=None)
    plt.xlabel("Density (kg/m³)")
    plt.ylabel("Temperature (K)")
    plt.title("Density vs Temperature colored by Conductivity")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(ruta_figuras, "fig8_density_vs_temp.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 4. Gráfica 3: Curva de Aprendizaje del Modelo
    print("Generando: Curva de Aprendizaje (Learning Curve)...")
    # Entrenamos con diferentes cantidades para ver como mejora
    X = df[["time_s", "thermal_conductivity_W_mK", "heat_capacity_J_kgK", "density_kg_m3"]].values
    y = df["temperature_K"].values
    
    tamaños = [10000, 50000, 100000, 500000]
    errores = []
    
    for n in tamaños:
        X_train, X_test, y_train, y_test = train_test_split(X[:n], y[:n], test_size=0.2, random_state=42)
        rf = RandomForestRegressor(n_estimators=20, random_state=42, n_jobs=-1) # Pocos arboles para rapidez
        rf.fit(X_train, y_train)
        # Usamos un subconjunto del test global para evaluar
        score = rf.score(X[500000:510000], y[500000:510000]) # R2 sobre datos no vistos
        errores.append(score)
        print(f"  Entrenado con {n} datos -> R2: {score:.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(tamaños, errores, marker='o', linestyle='-', color='purple', linewidth=2, markersize=8)
    plt.xscale('log')
    plt.xlabel("Number of Training Samples (log scale)")
    plt.ylabel("R² Score")
    plt.title("Learning Curve: Model Performance vs Data Size")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(ruta_figuras, "fig9_learning_curve.png"), dpi=300, bbox_inches="tight")
    plt.close()

    print("\n✅ ¡GRAFICAS EXTRA GENERADAS! Revisa la carpeta 'final_figures'.")

if __name__ == "__main__":
    main()