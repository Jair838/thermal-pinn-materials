import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    print("=== ANALISIS DE SENSIBILIDAD ===")
    ruta_fig = os.path.join("..", "final_figures")
    k0, Cp0, rho0 = 50.0, 500.0, 5000.0
    A, L, T_inf, T0 = 0.01, 0.01, 300.0, 300.0
    t = np.linspace(0, 100, 500)
    
    def simular(factor_k, factor_Cp):
        k = k0 * factor_k
        Cp = Cp0 * factor_Cp
        C = rho0 * Cp * A * L
        T = np.zeros_like(t)
        T[0] = T0
        for i in range(1, len(t)):
            Q = 500.0 * np.exp(-((t[i-1] - 20.0) ** 2) / 50.0)
            dTdt = (-(k * A / L) * (T[i-1] - T_inf) + Q) / C
            T[i] = T[i-1] + dTdt * 0.2
        return T

    plt.figure(figsize=(10, 6))
    for f in [0.5, 1.0, 1.5]:
        plt.plot(t, simular(f, 1.0), label=f"k x{f}")
    for f in [0.5, 1.5]:
        plt.plot(t, simular(1.0, f), "--", label=f"Cp x{f}")
    plt.xlabel("Time (s)")
    plt.ylabel("Temperature (K)")
    plt.title("Parameter Sensitivity Analysis (k and Cp)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(ruta_fig, "fig14_sensitivity_analysis.png"), dpi=300, bbox_inches="tight")
    print("✅ Grafico guardado.")

if __name__ == "__main__":
    main()