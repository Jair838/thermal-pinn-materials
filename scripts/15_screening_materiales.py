import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("=== BUSCANDO LOS MEJORES MATERIALES (SCREENING) ===")
    ruta_datos = os.path.join("..", "data", "dataset_limpio_100k.csv")
    ruta_figuras = os.path.join("..", "final_figures")
    
    if not os.path.exists(ruta_datos):
        print(f"❌ No se encontró el archivo {ruta_datos}")
        return

    # 1. Cargar los datos
    df = pd.read_csv(ruta_datos)
    print(f"Materiales cargados: {len(df):,}")
    
    # 2. Calcular un "Score" de rendimiento térmico (Métrica inventada para el paper)
    # Queremos materiales con alta densidad, alto módulo de volumen y bajo band gap
    # Esto es típico en disipadores de calor (metales)
    
    # Rellenamos nulos para no tener errores
    df = df.fillna({
        'modulo_volumetrico_GPa': df['modulo_volumetrico_GPa'].median(),
        'band_gap_eV': df['band_gap_eV'].median()
    })
    
    # Fórmula del Score (A mayor densidad y módulo, mejor; a menor band gap, mejor)
    df['thermal_score'] = (df['densidad_g_cm3'] * 0.5) + (df['modulo_volumetrico_GPa'] * 0.3) - (df['band_gap_eV'] * 0.2)
    
    # 3. Seleccionar el Top 20
    top_20 = df.sort_values(by='thermal_score', ascending=False).head(20)
    
    print("\n--- TOP 5 MATERIALES RECOMENDADOS ---")
    for i, row in top_20.head(5).iterrows():
        print(f"{row['formula']}: Score={row['thermal_score']:.2f} | Densidad={row['densidad_g_cm3']:.2f} | Band Gap={row['band_gap_eV']:.2f}")

    # 4. Graficar el Top 20
    print("\nGenerando gráfica del Top 20...")
    plt.figure(figsize=(10, 6))
    sns.barplot(x="thermal_score", y="formula", data=top_20, palette="magma")
    plt.xlabel("Thermal Performance Score")
    plt.ylabel("Material Formula")
    plt.title("Top 20 Materials for Thermal Management (Screening)")
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(ruta_figuras, "fig13_top_materials_screening.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    # 5. Guardar la tabla en CSV para el artículo
    ruta_tabla = os.path.join("..", "results", "top_20_materiales.csv")
    top_20[['formula', 'mp_id', 'densidad_g_cm3', 'band_gap_eV', 'modulo_volumetrico_GPa', 'thermal_score']].to_csv(ruta_tabla, index=False)
    
    print(f"\n✅ ¡SCREENING COMPLETADO!")
    print(f"Gráfica guardada en final_figures/fig13_top_materials_screening.png")
    print(f"Tabla guardada en results/top_20_materiales.csv")

if __name__ == "__main__":
    main()