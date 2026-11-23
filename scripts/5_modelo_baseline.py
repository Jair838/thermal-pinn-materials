import os 
import pandas as pd 
from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_absolute_error, r2_score 
 
def main(): 
    print("Cargando datos para el Baseline...") 
    df = pd.read_csv(os.path.join("..", "data", "curvas_termicas_grande.csv"), nrows=100000) 
    X = df[["tiempo_s", "k_W_mK", "Cp_J_kgK", "rho_kg_m3"]].values 
    y = df["temperatura_K"].values 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
    print("Entrenando Random Forest...") 
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1) 
    rf.fit(X_train, y_train) 
    y_pred = rf.predict(X_test) 
    mae = mean_absolute_error(y_test, y_pred) 
    r2 = r2_score(y_test, y_pred) 
    print(f"Resultados Random Forest: MAE = {mae:.4f} K, R2 = {r2:.4f}") 
    importancias = pd.Series(rf.feature_importances_, index=["tiempo_s", "k_W_mK", "Cp_J_kgK", "rho_kg_m3"]).sort_values(ascending=False) 
    print("\nImportancia de las variables:") 
    print(importancias) 
 
if __name__ == "__main__": 
    main() 
