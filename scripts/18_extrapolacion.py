import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 1))
    def forward(self, x): return self.net(x)

def main():
    print("=== TEST DE EXTRAPOLACION ===")
    ruta_fig = os.path.join("..", "final_figures")
    
    t = np.linspace(0, 100, 1000)
    T_real = 300 + 50 * np.exp(-((t - 20)**2)/50)
    
    t_all = torch.tensor(t, dtype=torch.float32).view(-1,1)
    T_all = torch.tensor(T_real, dtype=torch.float32).view(-1,1)
    
    mask = t_all <= 50.0
    t_train = t_all[mask].view(-1,1)
    T_train = T_all[mask].view(-1,1)
    
    modelo = PINN()
    optim = torch.optim.Adam(modelo.parameters(), lr=0.01)
    for _ in range(1000):
        optim.zero_grad()
        loss = torch.mean((modelo(t_train) - T_train)**2)
        loss.backward()
        optim.step()
        
    with torch.no_grad():
        T_pred = modelo(t_all).numpy()
        
    plt.figure(figsize=(10, 6))
    plt.plot(t_all.numpy(), T_all.numpy(), 'o', label="Real Data", alpha=0.3)
    plt.plot(t_all.numpy(), T_pred, '-', label="PINN Prediction", color="red")
    plt.axvline(x=50, color="black", linestyle="--", label="Training Limit")
    plt.xlabel("Time (s)")
    plt.ylabel("Temperature (K)")
    plt.title("Temporal Extrapolation Capability")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(ruta_fig, "fig16_extrapolation.png"), dpi=300, bbox_inches="tight")
    print("✅ Grafico guardado.")

if __name__ == "__main__":
    main()