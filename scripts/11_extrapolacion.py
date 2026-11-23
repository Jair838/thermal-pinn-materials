import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
    def forward(self, t):
        return self.net(t)

def main():
    print("Evaluando capacidad de extrapolacion...")
    
    # Cargar datos
    df = pd.read_csv(os.path.join("..", "data", "curvas_termicas.csv"))
    sub = df[df["formula"] == df["formula"].iloc[0]].copy()
    
    t_all = torch.tensor(sub["tiempo_s"].values, dtype=torch.float32).view(-1, 1)
    T_all = torch.tensor(sub["temperatura_K"].values, dtype=torch.float32).view(-1, 1)
    
    # Definir la máscara (entrenar solo hasta t=70)
    mask_train = t_all <= 70.0
    t_train = t_all[mask_train].view(-1, 1)
    T_train = T_all[mask_train].view(-1, 1)
    
    # Entrenar
    modelo = PINN()
    optim = torch.optim.Adam(modelo.parameters(), lr=0.01)
    
    for _ in range(1000):
        optim.zero_grad()
        pred = modelo(t_train)
        loss = torch.mean((pred - T_train) ** 2)
        loss.backward()
        optim.step()
        
    # Predecir en TODO el rango (incluyendo extrapolación)
    with torch.no_grad():
        T_pred_all = modelo(t_all).numpy()
        
    plt.figure(figsize=(8, 5))
    plt.plot(t_all.numpy(), T_all.numpy(), "o", label="Datos Reales", markersize=3)
    plt.plot(t_all.numpy(), T_pred_all, "-", label="Prediccion PINN", linewidth=2)
    plt.axvline(x=70, color="red", linestyle="--", label="Limite de Entrenamiento")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Temperatura (K)")
    plt.title("Capacidad de Extrapolacion del PINN")
    plt.legend()
    plt.grid(True)
    
    plt.savefig(os.path.join("..", "figures", "extrapolacion_pinn.png"))
    print("Grafico guardado en figures/extrapolacion_pinn.png")

if __name__ == "__main__":
    main()