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
    print("=== GENERANDO FIGURAS FINALES PARA EL ARTICULO Q1 ===")
    
    # Rutas
    ruta_datos = os.path.join("..", "data", "muestra_expres.csv")
    ruta_figuras = os.path.join("..", "final_figures")
    
    # Crear carpeta si no existe
    if not os.path.exists(ruta_figuras):
        os.makedirs(ruta_figuras)

    print("Cargando muestra...")
    df = pd.read_csv(ruta_datos)
    
    # Tomamos una muestra de 100,000 filas para que las gráficas se generen rápido
    df_sample = df.sample(n=min(100000, len(df)), random_state=42)
    
    # 1. Entrenar Random Forest
    print("Entrenando modelo para extraer métricas...")
    X = df_sample[["time_s", "thermal_conductivity_W_mK", "heat_capacity_J_kgK", "density_kg_m3"]].values
    y = df_sample["temperature_K"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # FIGURA 1: Data Distribution (Histograma)
    print("Generando Figura 1...")
    plt.figure(figsize=(8, 5))
    plt.hist(y, bins=50, color="purple", alpha=0.7)
    plt.xlabel("Temperature (K)")
    plt.ylabel("Frequency")
    plt.title(f"Thermal Data Distribution (Sample of {len(df_sample):,} points)")
    plt.grid(True)
    plt.savefig(os.path.join(ruta_figuras, "fig1_data_distribution.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # FIGURA 2: Feature Importance (Barras)
    print("Generando Figura 2...")
    plt.figure(figsize=(8, 5))
    nombres = ["Time (s)", "Thermal Conductivity (k)", "Heat Capacity (Cp)", "Density (rho)"]
    plt.barh(nombres, rf.feature_importances_, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
    plt.xlabel("Relative Importance")
    plt.title("Global Feature Importance")
    plt.grid(axis="x")
    plt.savefig(os.path.join(ruta_figuras, "fig2_feature_importance.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # FIGURA 3: Correlation Heatmap
    print("Generando Figura 3...")
    plt.figure(figsize=(8, 6))
    columnas = ["time_s", "temperature_K", "thermal_conductivity_W_mK", "heat_capacity_J_kgK", "density_kg_m3"]
    sns.heatmap(df_sample[columnas].corr(), annot=True, cmap="coolwarm", fmt=".2f",
                xticklabels=["Time", "Temp", "k", "Cp", "rho"], yticklabels=["Time", "Temp", "k", "Cp", "rho"])
    plt.title("Thermal Correlation Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(ruta_figuras, "fig3_correlation.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # FIGURA 4: SHAP Summary
    print("Generando Figura 4 (SHAP)...")
    df_shap = df_sample.head(1000)
    X_shap = df_shap[["time_s", "thermal_conductivity_W_mK", "heat_capacity_J_kgK", "density_kg_m3"]]
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_shap)
    
    plt.figure(figsize=(8, 5))
    shap.summary_plot(shap_values, X_shap, show=False)
    plt.title("SHAP Feature Impact")
    plt.savefig(os.path.join(ruta_figuras, "fig4_shap_analysis.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # FIGURA 5: Predicción vs Realidad
    print("Generando Figura 5...")
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.3, s=10, color="blue")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel("Actual Temperature (K)")
    plt.ylabel("Predicted Temperature (K)")
    plt.title(f"Random Forest Prediction (R² = {r2:.4f})")
    plt.grid(True)
    plt.savefig(os.path.join(ruta_figuras, "fig5_prediction_scatter.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # FIGURA 6: Boxplot de Densidad
    print("Generando Figura 6...")
    plt.figure(figsize=(8, 5))
    sns.boxplot(x="formula", y="density_kg_m3", data=df_sample.head(1000), palette="Set3")
    plt.xticks(rotation=90)
    plt.xlabel("Material")
    plt.ylabel("Density (kg/m³)")
    plt.title("Density Distribution Across Materials")
    plt.tight_layout()
    plt.savefig(os.path.join(ruta_figuras, "fig6_density_boxplot.png"), dpi=300, bbox_inches="tight")
    plt.close()

    print("\n✅ ¡TODAS LAS FIGURAS FINALES GENERADAS!")
    print(f"Revisa la carpeta 'final_figures'.")

if __name__ == "__main__":
    main()