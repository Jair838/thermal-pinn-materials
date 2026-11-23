import os 
import pandas as pd 
import numpy as np 
import torch 
import torch.nn as nn 
import matplotlib.pyplot as plt 
 
class PINN(nn.Module): 
    def __init__(self): 
        super().__init__() 
        self.net = nn.Sequential(nn.Linear(1, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 1)) 
    def forward(self, t): 
        return self.net(t) 
 
def main(): 
    print("Evaluando robustez ante ruido...") 
    df = pd.read_csv(os.path.join("..", "data", "curvas_termicas.csv")) 
    sub = df[df["formula"] == df["formula"].iloc[0]].copy() 
    t_data = torch.tensor(sub["tiempo_s"].values, dtype=torch.float32).view(-1, 1) 
    T_clean = sub["temperatura_K"].values 
    niveles_ruido = [0.0, 0.01, 0.02, 0.05] 
    errores = [] 
    for nivel in niveles_ruido: 
        ruido = np.random.normal(0, nivel * np.mean(T_clean), len(T_clean)) 
        T_ruidoso = T_clean + ruido 
        T_tensor = torch.tensor(T_ruidoso, dtype=torch.float32).view(-1, 1) 
        modelo = PINN() 
        optim = torch.optim.Adam(modelo.parameters(), lr=0.01) 
        for _ in range(500): 
            optim.zero_grad() 
            pred = modelo(t_data) 
            loss = torch.mean((pred - T_tensor) ** 2) 
            loss.backward() 
            optim.step() 
        errores.append(loss.item()) 
        print(f"Ruido {nivel*100}% -> Error: {loss.item():.4f}") 
    plt.figure(figsize=(6, 4)) 
    plt.plot([f"{n*100}%" for n in niveles_ruido], errores, "o-", color="red") 
    plt.xlabel("Nivel de Ruido") 
    plt.ylabel("Error Final (MSE)") 
    plt.title("Robustez del PINN ante ruido en los datos") 
    plt.grid(True) 
    plt.savefig(os.path.join("..", "figures", "robustez_ruido.png")) 
    print("Grafico guardado en figures/robustez_ruido.png") 
if __name__ == "__main__": 
    main() 
