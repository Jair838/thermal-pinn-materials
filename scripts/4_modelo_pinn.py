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
    print("Cargando datos...") 
    df = pd.read_csv(os.path.join("..", "data", "curvas_termicas_grande.csv"), nrows=100000) 
    formula = df["formula"].iloc[0] 
    sub = df[df["formula"] == formula].copy() 
    t_data = torch.tensor(sub["tiempo_s"].values, dtype=torch.float32).view(-1, 1) 
    T_data = torch.tensor(sub["temperatura_K"].values, dtype=torch.float32).view(-1, 1) 
    k_val = sub["k_W_mK"].iloc[0] 
    Cp_val = sub["Cp_J_kgK"].iloc[0] 
    rho_val = sub["rho_kg_m3"].iloc[0] 
    A, L, T_inf = 0.01, 0.01, 300.0 
    C_val = rho_val * Cp_val * A * L 
    modelo = PINN() 
    optimizador = torch.optim.Adam(modelo.parameters(), lr=0.01) 
    epocas = 2000 
    print("Entrenando PINN...") 
    for epoca in range(epocas): 
        optimizador.zero_grad() 
        t_data.requires_grad = True 
        T_pred = modelo(t_data) 
        dTdt = torch.autograd.grad(T_pred, t_data, grad_outputs=torch.ones_like(T_pred), create_graph=True)[0] 
        Q_t = 500.0 * torch.exp(-((t_data - 20.0) ** 2) / (2 * 5.0 ** 2)) 
        residuo = C_val * dTdt + (k_val * A / L) * (T_pred - T_inf) - Q_t 
        loss_data = torch.mean((T_pred - T_data) ** 2) 
        loss_fisica = torch.mean(residuo ** 2) 
        loss_total = loss_data + 1e-4 * loss_fisica 
        loss_total.backward() 
        optimizador.step() 
        if (epoca + 1) % 500 == 0: 
            print(f"Epoca {epoca+1}: Loss Total = {loss_total.item():.4f}") 
    torch.save(modelo.state_dict(), os.path.join("..", "results", "modelo_pinn.pth")) 
    print("Modelo guardado en results/modelo_pinn.pth") 
    modelo.eval() 
    with torch.no_grad(): 
        T_final = modelo(t_data).numpy() 
    plt.figure(figsize=(8, 5)) 
    plt.plot(t_data.detach().numpy(), T_data.numpy(), "o", label="Datos Reales",           markersize=3)
    plt.plot(t_data.detach().numpy(), T_final, "-", label="Prediccion PINN", linewidth=2)
    plt.xlabel("Tiempo (s)") 
    plt.ylabel("Temperatura (K)") 
    plt.title(f"PINN vs Datos Reales - {formula}") 
    plt.legend() 
    plt.grid(True) 
    plt.savefig(os.path.join("..", "figures", "resultado_pinn.png")) 
    print("Grafico guardado en figures/resultado_pinn.png") 
 
if __name__ == "__main__": 
    main() 
