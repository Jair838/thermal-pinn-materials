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
    def forward(self, t): 
        return self.net(t) 
 
def main(): 
    print("Analizando estabilidad (Multi-Seed)...") 
    df = pd.read_csv(os.path.join("..", "data", "curvas_termicas.csv")) 
    sub = df[df["formula"] == df["formula"].iloc[0]].copy() 
    t_data = torch.tensor(sub["tiempo_s"].values, dtype=torch.float32).view(-1, 1) 
    T_data = torch.tensor(sub["temperatura_K"].values, dtype=torch.float32).view(-1, 1) 
    errores_finales = [] 
    for semilla in [42, 123, 777]: 
        torch.manual_seed(semilla) 
        modelo = PINN() 
        optim = torch.optim.Adam(modelo.parameters(), lr=0.01) 
        for _ in range(500): 
            optim.zero_grad() 
            pred = modelo(t_data) 
            loss = torch.mean((pred - T_data) ** 2) 
            loss.backward() 
            optim.step() 
        errores_finales.append(loss.item()) 
        print(f"Semilla {semilla} - Error final: {loss.item():.6f}") 
    plt.figure(figsize=(6, 4)) 
    plt.bar(["Seed 42", "Seed 123", "Seed 777"], errores_finales, color=["blue", "green", "red"], alpha=0.6) 
    plt.ylabel("Error Final (MSE)") 
    plt.title("Estabilidad del PINN ante distintas inicializaciones") 
    plt.grid(axis="y") 
    plt.savefig(os.path.join("..", "figures", "estabilidad_pinn.png")) 
    print("Grafico guardado en figures/estabilidad_pinn.png") 
if __name__ == "__main__": 
    main() 
