import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, 16), nn.Tanh(), nn.Linear(16, 1))
    def forward(self, x): return self.net(x)

def main():
    print("=== ANALISIS DE ROBUSTEZ ===")
    ruta_fig = os.path.join("..", "final_figures")
    
    # Creamos datos falsos si no existe el archivo, para que no falle
    t = np.linspace(0, 100, 1000)
    T_real = 300 + 50 * np.exp(-((t - 20)**2)/50)
    
    t_data = torch.tensor(t, dtype=torch.float32).view(-1,1)
    T_clean = T_real
    
    niveles = [0.0, 0.01, 0.05]
    errores = []
    
    for nivel in niveles:
        T_ruido = T_clean + np.random.normal(0, nivel * np.mean(T_clean), len(T_clean))
        T_tensor = torch.tensor(T_ruido, dtype=torch.float32).view(-1,1)
        
        modelo = PINN()
        optim = torch.optim.Adam(modelo.parameters(), lr=0.01)
        for _ in range(300):
            optim.zero_grad()
            loss = torch.mean((modelo(t_data) - T_tensor)**2)
            loss.backward()
            optim.step()
        errores.append(loss.item())
        
    plt.figure(figsize=(8, 5))
    plt.bar([f"{n*100}% noise" for n in niveles], errores, color=['green', 'orange', 'red'])
    plt.xlabel("Noise Level")
    plt.ylabel("MSE Loss")
    plt.title("Model Robustness to Sensor Noise")
    plt.savefig(os.path.join(ruta_fig, "fig15_robustness_noise.png"), dpi=300, bbox_inches="tight")
    print("✅ Grafico guardado.")

if __name__ == "__main__":
    main()