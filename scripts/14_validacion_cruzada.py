import os 
import pandas as pd 
from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection import cross_val_score 
 
def main(): 
    print("Iniciando Validacion Cruzada (5-fold)...") 
    df = pd.read_csv(os.path.join("..", "data", "curvas_termicas.csv")) 
    X = df[["tiempo_s", "k_W_mK", "Cp_J_kgK", "rho_kg_m3"]].values 
    y = df["temperatura_K"].values 
    rf = RandomForestRegressor(n_estimators=100, random_state=42) 
    scores = cross_val_score(rf, X, y, cv=5, scoring="r2") 
    print(f"R2 por pliegue: {scores}") 
    print(f"R2 promedio: {scores.mean():.4f} +/- {scores.std():.4f}") 
    ruta = os.path.join("..", "results", "cross_validation.txt") 
    with open(ruta, "w") as f: 
        f.write(f"R2 individual: {scores}\nR2 Promedio: {scores.mean():.4f}\nDesviacion: {scores.std():.4f}") 
    print("Resultados guardados en results/cross_validation.txt") 
if __name__ == "__main__": 
    main() 
