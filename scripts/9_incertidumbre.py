import os 
import pandas as pd 
import torch 
import torch.nn as nn 
import numpy as np 
import matplotlib.pyplot as plt 
 
class PINN(nn.Module): 
    def __init__(self): 
        super().__init__() 
        self.net = nn.Sequential(nn.Linear(1, 32), nn.Tanh(), nn.Dropout(0.1), nn.Linear(32, 32), nn.Tanh(), nn.Dropout(0.1), nn.Linear(32, 1)) 
    def forward(self, t): 
        return self.net(t) 
 
def main(): 
    print("1. Preparando datos...") 
    df = pd.read_csv(os.path.join("..", "data", "curvas_termicas.csv")) 
    sub = df[df["formula"] == df["formula"].iloc[0]].copy() 
    t_data = torch.tensor(sub["tiempo_s"].values, dtype=torch.float32).view(-1, 1) 
    T_data = sub["temperatura_K"].values 
    T_tensor = torch.tensor(T_data, dtype=torch.float32).view(-1, 1) 
    modelo = PINN() 
    optimizador = torch.optim.Adam(modelo.parameters(), lr=0.01) 
    print("2. Entrenando modelo base (500 epocas)...") 
    modelo.train() 
    for epoca in range(500): 
        optimizador.zero_grad() 
        pred = modelo(t_data) 
        loss = torch.mean((pred - T_tensor) ** 2) 
        loss.backward() 
        optimizador.step() 
    print("3. Calculando incertidumbre (Monte Carlo Dropout)...") 
    predicciones = [] 
    for _ in range(50): 
        with torch.no_grad(): 
            predicciones.append(modelo(t_data).numpy()) 
    predicciones = np.array(predicciones).squeeze() 
    media = predicciones.mean(axis=0) 
    std = predicciones.std(axis=0) 
    plt.figure(figsize=(8, 5)) 
    plt.plot(t_data.numpy(), T_data, "o", label="Datos Reales", markersize=3) 
    plt.plot(t_data.numpy(), media, "-", label="Prediccion Media", linewidth=2) 
    plt.fill_between(t_data.numpy().flatten(), media - 1.96*std, media + 1.96*std, alpha=0.3, label="Intervalo 95%") 
    plt.xlabel("Tiempo (s)") 
    plt.ylabel("Temperatura (K)") 
    plt.title("Incertidumbre del PINN (Monte Carlo Dropout)") 
    plt.legend() 
    plt.grid(True) 
    plt.savefig(os.path.join("..", "figures", "incertidumbre_pinn.png")) 
    print("Grafico guardado en figures/incertidumbre_pinn.png") 
if __name__ == "__main__": 
    main() 
