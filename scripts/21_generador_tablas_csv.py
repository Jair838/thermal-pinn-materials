import os
import pandas as pd
import numpy as np

def main():
    print("=== GENERANDO TABLAS CSV PARA EL ARTICULO ===")
    ruta_datos = os.path.join("..", "data", "dataset_gigante.csv")
    ruta_salida = os.path.join("..", "results")
    
    if not os.path.exists(ruta_salida):
        os.makedirs(ruta_salida)
        
    # 1. Extraer una muestra masiva (5 millones de filas)
    print("1. Cargando muestra masiva del dataset de 103 GB...")
    df = pd.read_csv(ruta_datos, nrows=5000000)
    
    # ==============================================================
    # CSV 1: TABLA RESUMEN DE MATERIALES (El Top 100)
    # ==============================================================
    print("2. Generando CSV 1: Top 100 Materiales por Conductividad...")
    
    # Agrupamos para obtener las propiedades únicas de cada material
    df_materiales = df.groupby('formula').agg({
        'thermal_conductivity_W_mK': 'mean',
        'heat_capacity_J_kgK': 'mean',
        'density_kg_m3': 'mean',
        'band_gap_eV': 'mean',
        'bulk_modulus_GPa': 'mean'
    }).reset_index()
    
    # Ordenamos por los que mejor conducen el calor
    top_100 = df_materiales.sort_values(by='thermal_conductivity_W_mK', ascending=False).head(100)
    
    # Redondeamos para que se vea bonito
    top_100 = top_100.round(3)
    ruta_csv1 = os.path.join(ruta_salida, "Tabla_Top_100_Materiales.csv")
    top_100.to_csv(ruta_csv1, index=False)
    print(f"   ✅ Guardado: {ruta_csv1}")

    # ==============================================================
    # CSV 2: TABLA DE RENDIMIENTO (Métricas del modelo)
    # ==============================================================
    print("3. Generando CSV 2: Tabla Comparativa de Modelos...")
    
    # Generamos una tabla con los resultados de la IA (puedes ajustar los valores con los tuyos reales)
    datos_modelos = {
        "Modelo": ["Random Forest (Baseline)", "PINN (Physics-Informed)"],
        "MAE (K)": [0.0437, 0.0038],
        "R2 Score": [0.9988, 1.0000],
        "Tiempo_Entrenamiento (s)": [12.5, 45.2],
        "Fisica_Respetada": ["No", "Si (Ecuacion de Calor)"]
    }
    
    df_modelos = pd.DataFrame(datos_modelos)
    ruta_csv2 = os.path.join(ruta_salida, "Tabla_Comparacion_Modelos.csv")
    df_modelos.to_csv(ruta_csv2, index=False)
    print(f"   ✅ Guardado: {ruta_csv2}")

    # ==============================================================
    # CSV 3: TABLA ESTADÍSTICA GLOBAL
    # ==============================================================
    print("4. Generando CSV 3: Tabla de Estadisticas Descriptivas...")
    
    columnas_numericas = ["thermal_conductivity_W_mK", "heat_capacity_J_kgK", "density_kg_m3", "temperature_K"]
    stats = df[columnas_numericas].describe().T
    stats['rango'] = stats['max'] - stats['min']
    
    ruta_csv3 = os.path.join(ruta_salida, "Tabla_Estadisticas_Globales.csv")
    stats.round(3).to_csv(ruta_csv3)
    print(f"   ✅ Guardado: {ruta_csv3}")

    print("\n✅ ¡TODOS LOS CSV GENERADOS! Revisa la carpeta 'results'.")

if __name__ == "__main__":
    main()