import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def main():
    print("=== GENERANDO ANALISIS DE RESIDUOS ===")
    ruta_fig = os.path.join("..", "final_figures")
    
    # Datos de prueba
    X = np.random.rand(1000, 4)
    y = 300 + X[:, 0]*50 + np.random.normal(0, 2, 1000)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=50, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    residuos = y_test - y_pred
    
    plt.figure(figsize=(8, 6))
    sns.histplot(residuos, bins=50, kde=True, color="purple")
    plt.xlabel("Residuals (Actual - Predicted Temperature) [K]")
    plt.ylabel("Frequency")
    plt.title("Distribution of Model Residuals")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(os.path.join(ruta_fig, "fig18_residuals.png"), dpi=300, bbox_inches="tight")
    print("✅ Grafico guardado.")

if __name__ == "__main__":
    main()