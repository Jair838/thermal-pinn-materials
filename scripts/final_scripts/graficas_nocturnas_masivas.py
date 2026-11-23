import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import shap

def main():
    print("=== EXTRACCIÓN MASIVA NOCTURNA (Miles de Materiales) ===")
    print("Iniciando proceso. Esto tomará varias horas. NO CIERRES LA VENTANA.")
    
    ruta_datos = os.path.join("..", "data", "dataset_gigante.csv")
    ruta_figuras = os.path.join("..", "final_figures")
    if not os.path.exists(ruta_figuras):
        os.makedirs(ruta_figuras)
        
    print("\n1. Leyendo el archivo de 103 GB por bloques...")
    
    # Vamos a guardar pedazos en una lista
    bloques_muestra = []
    filas_totales = 0
    
    # Leemos en bloques de 5 millones de filas
    with pd.read_csv(ruta_datos, chunksize=5000000) as reader:
        for i, chunk in enumerate(reader):
            # Nos quedamos con el 10% de cada bloque al azar
            muestra_chunk = chunk.sample(frac=0.10, random_state=42)
            bloques_muestra.append(muestra_chunk)
            filas_totales += len(muestra_chunk)
            
            # Imprimimos progreso
            print(f"   Bloque {i+1} leído. Filas acumuladas para análisis: {filas_totales:,}")
            
            # Si llegamos a 20 millones de filas, paramos de leer (para no hacerlo eterno)
            if filas_totales >= 20000000:
                print("   Límite de 20 millones de filas alcanzado. Deteniendo lectura.")
                break
                
    # Unimos todos los bloques en un solo DataFrame
    print("\n2. Uniendo bloques...")
    df = pd.concat(bloques_muestra, ignore_index=True)
    
    num_materiales = df['formula'].nunique()
    print(f"   ¡DATOS LISTOS! Total filas: {len(df):,} | Materiales distintos: {num_materiales:,}")

    # Tomamos una muestra para que las gráficas se generen rápido (500,000 filas)
    df_sample = df.sample(n=min(500000, len(df)), random_state=42)

    # 3. Entrenar Random Forest
    print("\n3. Entrenando Random Forest con la muestra masiva...")
    X = df_sample[["time_s", "thermal_conductivity_W_mK", "heat_capacity_J_kgK", "density_kg_m3"]].values
    y = df_sample["temperature_K"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"   RF: MAE = {mae:.4f} K | R2 = {r2:.4f}")

    # 4. Generar las Gráficas
    print("\n4. Generando gráficas finales...")

    # Figura 1: Data Distribution
    plt.figure(figsize=(8, 5))
    plt.hist(y, bins=50, color="purple", alpha=0.7)
    plt.xlabel("Temperature (K)")
    plt.ylabel("Frequency")
    plt.title(f"Thermal Data Distribution ({num_materiales:,} Materials)")
    plt.grid(True)
    plt.savefig(os.path.join(ruta_figuras, "fig1_data_distribution_night.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Figura 2: Feature Importance
    plt.figure(figsize=(8, 5))
    nombres = ["Time (s)", "Thermal Conductivity (k)", "Heat Capacity (Cp)", "Density (rho)"]
    plt.barh(nombres, rf.feature_importances_, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
    plt.xlabel("Relative Importance")
    plt.title(f"Global Feature Importance ({num_materiales:,} Materials)")
    plt.grid(axis="x")
    plt.savefig(os.path.join(ruta_figuras, "fig2_feature_importance_night.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Figura 3: Correlation Heatmap
    plt.figure(figsize=(8, 6))
    columnas = ["time_s", "temperature_K", "thermal_conductivity_W_mK", "heat_capacity_J_kgK", "density_kg_m3"]
    sns.heatmap(df_sample[columnas].corr(), annot=True, cmap="coolwarm", fmt=".2f",
                xticklabels=["Time", "Temp", "k", "Cp", "rho"], yticklabels=["Time", "Temp", "k", "Cp", "rho"])
    plt.title("Thermal Correlation Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(ruta_figuras, "fig3_correlation_night.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Figura 4: SHAP Summary
    print("   Generando SHAP (esto tarda un poco)...")
    df_shap = df_sample.head(1000)
    X_shap = df_shap[["time_s", "thermal_conductivity_W_mK", "heat_capacity_J_kgK", "density_kg_m3"]]
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_shap)
    
    plt.figure(figsize=(8, 5))
    shap.summary_plot(shap_values, X_shap, show=False)
    plt.title("SHAP Feature Impact")
    plt.savefig(os.path.join(ruta_figuras, "fig4_shap_analysis_night.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Figura 5: Predicción vs Realidad
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.3, s=10, color="blue")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel("Actual Temperature (K)")
    plt.ylabel("Predicted Temperature (K)")
    plt.title(f"Random Forest Prediction (R² = {r2:.4f})")
    plt.grid(True)
    plt.savefig(os.path.join(ruta_figuras, "fig5_prediction_scatter_night.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Guardar reporte en texto
    with open(os.path.join("..", "results", "resultados_nocturnos.txt"), "w") as f:
        f.write(f"=== ANALISIS MASIVO NOCTURNO ===\n")
        f.write(f"Materiales analizados: {num_materiales:,}\n")
        f.write(f"Filas totales extraidas: {len(df):,}\n")
        f.write(f"Random Forest MAE: {mae:.4f} K\n")
        f.write(f"Random Forest R2: {r2:.4f}\n")

    print("\n✅ ¡TODO LISTO! Puedes apagar el monitor y dormir tranquilo.")
    print(f"Se analizaron {num_materiales:,} materiales distintos.")

if __name__ == "__main__":
    main()