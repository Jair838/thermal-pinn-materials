import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("=== GENERANDO GRAFICAS DE DESCUBRIMIENTO CIENTIFICO ===")
    
    ruta_datos = os.path.join("..", "data", "dataset_gigante.csv")
    ruta_figuras = os.path.join("..", "final_figures")
    
    # 1. Cargar datos (muestra de 5 millones para que sea rápido)
    print("Cargando muestra de datos masivos...")
    bloques = []
    with pd.read_csv(ruta_datos, chunksize=5000000) as reader:
        for i, chunk in enumerate(reader):
            bloques.append(chunk.sample(frac=0.1)) # 10% de cada bloque
            if i >= 9: break # 10 bloques = 5 millones de filas
            
    df = pd.concat(bloques, ignore_index=True)
    print(f"Datos cargados: {len(df):,} filas de {df['formula'].nunique():,} materiales.")

    # ==============================================================
    # GRÁFICA 1: El Efecto de los Elementos (Complejidad vs Conductividad)
    # ==============================================================
    print("Generando Figura 10: Complejidad Quimica vs Conductividad...")
    # Agrupamos por material para sacar su promedio
    df_materiales = df.groupby('formula').agg({
        'thermal_conductivity_W_mK': 'mean',
        'density_kg_m3': 'mean',
        'band_gap_eV': 'mean'
    }).reset_index()
    
    # Creamos una categoría: Materiales simples (1-2 elementos) vs Complejos (3+)
    def tipo_material(formula):
        num_elem = sum(1 for c in formula if c.isupper())
        return "Simple (1-2 elements)" if num_elem <= 2 else "Complex (3+ elements)"
    
    df_materiales['complejidad'] = df_materiales['formula'].apply(tipo_material)
    
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df_materiales, x='complejidad', y='thermal_conductivity_W_mK', 
                   palette="Set2", inner="quart")
    plt.yscale('log') # Escala logarítmica porque hay valores muy grandes
    plt.xlabel("Chemical Complexity")
    plt.ylabel("Thermal Conductivity (W/m·K) [Log Scale]")
    plt.title("Impact of Chemical Complexity on Thermal Conductivity")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(ruta_figuras, "fig10_complexity_vs_conductivity.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # ==============================================================
    # GRÁFICA 2: El "Mapa Estelar" de Materiales (Densidad vs Band Gap)
    # ==============================================================
    print("Generando Figura 11: Mapa Estelar de Materiales...")
    # Usamos una muestra de 10,000 materiales para el scatter
    df_scatter = df_materiales.sample(n=min(10000, len(df_materiales)), random_state=42)
    
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(df_scatter['density_kg_m3'], df_scatter['band_gap_eV'], 
                          c=df_scatter['thermal_conductivity_W_mK'], cmap='plasma', 
                          s=20, alpha=0.7, norm=plt.matplotlib.colors.LogNorm())
    
    plt.colorbar(scatter, label="Thermal Conductivity (W/m·K) [Log Scale]")
    plt.xlabel("Density (kg/m³)")
    plt.ylabel("Band Gap (eV)")
    plt.title("Material Property Landscape: Density vs Band Gap")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(ruta_figuras, "fig11_property_landscape.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # ==============================================================
    # GRÁFICA 3: La Ley de Escala Térmica (Temperatura vs Conductividad)
    # ==============================================================
    print("Generando Figura 12: Efecto de la Conductividad en la Temperatura...")
    # Tomamos 3 materiales con conductividades baja, media y alta
    df_k = df_materiales.sort_values('thermal_conductivity_W_mK')
    material_bajo = df_k.iloc[10]['formula']
    material_medio = df_k.iloc[len(df_k)//2]['formula']
    material_alto = df_k.iloc[-10]['formula']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for mat, color in zip([material_bajo, material_medio, material_alto], ['blue', 'green', 'red']):
        sub = df[df['formula'] == mat]
        # Hacemos un promedio de temperatura en el tiempo para ese material
        curva = sub.groupby('time_s')['temperature_K'].mean()
        k_val = sub['thermal_conductivity_W_mK'].mean()
        ax.plot(curva.index, curva.values, label=f"{mat} (k = {k_val:.1f})", color=color, linewidth=2)
        
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Average Temperature (K)")
    ax.set_title("Thermal Response: Low vs Medium vs High Conductivity")
    ax.legend(title="Material (Conductivity)")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(ruta_figuras, "fig12_thermal_response_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()

    print("\n✅ ¡GRAFICAS DE DESCUBRIMIENTO GENERADAS!")
    print("Revisa la carpeta 'final_figures'.")

if __name__ == "__main__":
    main()