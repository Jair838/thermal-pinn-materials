import os
import numpy as np
import matplotlib.pyplot as plt

def main():
    print("Generando Serie Temporal...")
    t = np.linspace(0, 100, 500)
    T = 300 + 50 * np.exp(-((t-20)**2)/50)
    plt.figure(figsize=(10, 6))
    plt.plot(t, T, color='red', linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Temperature (K)")
    plt.title("Thermal Response Simulation")
    plt.savefig(os.path.join("..", "final_figures", "fig28_timeseries.png"), dpi=300)
    print("✅ Guardado.")

if __name__ == "__main__":
    main()