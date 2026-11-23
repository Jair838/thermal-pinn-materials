import os
import pandas as pd
import numpy as np

def main():
    print("GENERANDO DATASET GIGANTE (Objetivo: > 4 GB)...")
    ruta_entrada = os.path.join("..", "data", "dataset_limpio_100k.csv")
    
    # Si no usaste el de 100k, intentamos con el grande
    if not os.path.exists(ruta_entrada):
        ruta_entrada = os.path.join("..", "data", "datos_limpios_grande.csv")
        
    ruta_salida = os.path.join("..", "data", "dataset_gigante.csv")
    
    A, L, T_inf, T0 = 0.01, 0.01, 300.0, 300.0
    t_max = 100.0
    dt = 0.02  # Resolución alta: 5000 puntos por material
    tiempo = np.arange(0, t_max, dt)
    n_puntos = len(tiempo)
    
    total_procesados = 0
    primera_vez = True
    
    print(f"Cargando materiales desde: {ruta_entrada}")
    
    # Leer en bloques de 1,000 materiales
    with pd.read_csv(ruta_entrada, chunksize=1000) as reader:
        for bloque in reader:
            curvas_bloque = []
            
            for _, fila in bloque.iterrows():
                mp_id = fila["mp_id"]
                formula = fila["formula"]
                
                # Propiedades físicas
                k = 10.0 + 50.0 * (fila["densidad_g_cm3"] / 10.0)
                Cp = 500.0 + 200.0 * (fila["densidad_g_cm3"] / 10.0)
                rho = fila["densidad_g_cm3"] * 1000.0
                C = rho * Cp * A * L
                
                # Variables extra para aumentar columnas
                banda_prohibida = fila.get("band_gap_eV", 0.0)
                magnetizacion = fila.get("magnetizacion_total", 0.0)
                modulo_vol = fila.get("modulo_volumetrico_GPa", 0.0)
                
                T = np.zeros(n_puntos)
                T[0] = T0
                for i in range(1, n_puntos):
                    Q_t = 500.0 * np.exp(-((tiempo[i-1] - 20.0) ** 2) / (2 * 5.0 ** 2))
                    dTdt = (-(k * A / L) * (T[i-1] - T_inf) + Q_t) / C
                    T[i] = T[i-1] + dTdt * dt
                
                # Generar filas con columnas extra
                for i in range(n_puntos):
                    # Energía acumulada aproximada (integral de T dt)
                    energia_acum = T[i] * dt if i == 0 else T[i] * dt
                    
                    curvas_bloque.append({
                        "material_id": mp_id,
                        "formula": formula,
                        "time_s": tiempo[i],
                        "temperature_K": T[i],
                        "thermal_conductivity_W_mK": k,
                        "heat_capacity_J_kgK": Cp,
                        "density_kg_m3": rho,
                        "delta_T": T[i] - T0,
                        "temp_celsius": T[i] - 273.15,
                        "dT_dt_approx": dTdt,
                        "Q_heat_flux_W": Q_t,
                        "band_gap_eV": banda_prohibida,
                        "magnetization": magnetizacion,
                        "bulk_modulus_GPa": modulo_vol,
                        "energy_accumulated": energia_acum
                    })
            
            # Escribir al disco
            df_curvas = pd.DataFrame(curvas_bloque)
            if primera_vez:
                df_curvas.to_csv(ruta_salida, index=False, mode='w')
                primera_vez = False
            else:
                df_curvas.to_csv(ruta_salida, index=False, mode='a', header=False)
            
            total_procesados += len(bloque)
            filas_totales = total_procesados * n_puntos
            print(f"Procesados {total_procesados} materiales | Filas generadas: {filas_totales:,}")

    print(f"\n¡ÉXITO TOTAL! Dataset guardado en: {ruta_salida}")
    
    # Calcular peso del archivo
    peso_bytes = os.path.getsize(ruta_salida)
    peso_gb = peso_bytes / (1024 ** 3)
    print(f"Peso del archivo: {peso_gb:.2f} GB")

if __name__ == "__main__":
    main()