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
    print("Validando conservacion de energia (Residuo Fisico)...") 
    df = pd.read_csv(os.path.join("..", "data", "curvas_termicas.csv")) 
    sub = df[df["formula"] == df["formula"].iloc[0]].copy() 
    t_data = torch.tensor(sub["tiempo_s"].values, dtype=torch.float32).view(-1, 1) 
    T_data = torch.tensor(sub["temperatura_K"].values, dtype=torch.float32).view(-1, 1) 
    k_val = sub["k_W_mK"].iloc[0] 
    Cp_val = sub["Cp_J_kgK"].iloc[0] 
    rho_val = sub["rho_kg_m3"].iloc[0] 
    A, L, T_inf = 0.01, 0.01, 300.0 
    C_val = rho_val * Cp_val * A * L 
    modelo = PINN() 
    optim = torch.optim.Adam(modelo.parameters(), lr=0.01) 
    for _ in range(1000): 
        optim.zero_grad() 
        t_data.requires_grad = True 
        T_pred = modelo(t_data) 
        dTdt = torch.autograd.grad(T_pred, t_data, grad_outputs=torch.ones_like(T_pred), create_graph=True)[0] 
        Q_t = 500.0 * torch.exp(-((t_data - 20.0) ** 2) / (2 * 5.0 ** 2)) 
        residuo = C_val * dTdt + (k_val * A / L) * (T_pred - T_inf) - Q_t 
        loss = torch.mean((T_pred - T_data) ** 2) + 1e-4 * torch.mean(residuo ** 2) 
        loss.backward() 
        optim.step() 
    with torch.no_grad(): 
        T_pred = modelo(t_data) 
    t_data.requires_grad = True 
    T_pred = modelo(t_data) 
    dTdt = torch.autograd.grad(T_pred, t_data, grad_outputs=torch.ones_like(T_pred), create_graph=True)[0] 
    Q_t = 500.0 * torch.exp(-((t_data - 20.0) ** 2) / (2 * 5.0 ** 2)) 
    residuo = C_val * dTdt + (k_val * A / L) * (T_pred - T_inf) - Q_t 
    residuo_np = residuo.detach().numpy().flatten() 
    print(f"Residuo Fisico Medio: {np.mean(np.abs(residuo_np)):.4f}") 
    plt.figure(figsize=(8, 4)) 
    plt.plot(t_data.detach().numpy(), residuo_np, color="purple") 
    plt.axhline(0, color="black", linestyle="--") 
    plt.xlabel("Tiempo (s)") 
    plt.ylabel("Residuo Fisico (Ecuacion de Calor)") 
    plt.title("Validacion de Conservacion de Energia") 
    plt.grid(True) 
    plt.savefig(os.path.join("..", "figures", "residuo_fisico.png")) 
    print("Grafico guardado en figures/residuo_fisico.png") 
if __name__ == "__main__": 
    main() 
