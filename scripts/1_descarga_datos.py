import os
import pandas as pd
from mp_api.client import MPRester

def main():
    print("Iniciando descarga MASIVA de 100,000 materiales...")
    api_key = "TsrB5UPc6VcX73esawlgeaM8xgOfD8Ak" 
    
    registros = []
    
    with MPRester(api_key) as mpr:
        # Pedimos 100,000 materiales en 1000 lotes de 100
        docs = mpr.materials.summary.search(
            fields=[
                "material_id", "formula_pretty", "density", "volume", "symmetry", "nsites", "elements",
                "band_gap", "energy_above_hull", "formation_energy_per_atom", "efermi",
                "is_magnetic", "total_magnetization", "bulk_modulus", "shear_modulus",
                "universal_anisotropy", "homogeneous_poisson", "weighted_work_function",
                "nelements", "chemsys", "is_metal", "is_gap_direct", "total_magnetization_normalized_vol"
            ],
            num_chunks=1000,
            chunk_size=100
        )
        
        for i, d in enumerate(docs):
            b_mod = d.bulk_modulus.get("VRH") if d.bulk_modulus else None
            s_mod = d.shear_modulus.get("VRH") if d.shear_modulus else None
            
            registros.append({
                "mp_id": str(d.material_id),
                "formula": d.formula_pretty,
                "densidad_g_cm3": d.density,
                "volumen_A3": d.volume,
                "grupo_espacial": str(d.symmetry.symbol) if d.symmetry else "Unknown",
                "nsites": d.nsites,
                "elementos": ",".join([str(e) for e in d.elements]),
                "n_elementos": d.nelements,
                "sistema_quimico": d.chemsys,
                "es_metal": d.is_metal,
                "band_gap_eV": d.band_gap,
                "es_gap_directo": d.is_gap_direct,
                "energia_formacion_eV": d.formation_energy_per_atom,
                "energia_sobre_hull_eV": d.energy_above_hull,
                "efermi_eV": d.efermi,
                "es_magnetico": d.is_magnetic,
                "magnetizacion_total": d.total_magnetization,
                "magnetizacion_por_volumen": d.total_magnetization_normalized_vol,
                "modulo_volumetrico_GPa": b_mod,
                "modulo_cizalladura_GPa": s_mod,
                "anisotropia_universal": d.universal_anisotropy,
                "poisson": d.homogeneous_poisson,
                "funcion_trabajo_eV": d.weighted_work_function,
            })
            
            if (i+1) % 10000 == 0:
                print(f"Descargados {i+1} materiales...")

    df = pd.DataFrame(registros)
    
    # Descriptores calculados
    df["densidad_atomica"] = df["nsites"] / df["volumen_A3"]
    df["volumen_por_atomo"] = df["volumen_A3"] / df["nsites"]
    
    ruta = os.path.join("..", "data", "dataset_maestro_100k.csv")
    df.to_csv(ruta, index=False)
    print(f"\n¡ÉXITO! Se descargaron {len(df)} materiales con {len(df.columns)} columnas.")

if __name__ == "__main__":
    main()